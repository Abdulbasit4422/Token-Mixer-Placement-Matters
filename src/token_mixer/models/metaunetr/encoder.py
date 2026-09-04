"""Shared five-level MetaUNETR encoder.

Convolutional boundaries use ``[B, C, D, H, W]`` and token mixers use
``[B, D, H, W, C]``. The stem and four stride-two transitions follow the
official MetaUNETR implementation (https://github.com/lyupengju/MetaUNETR),
with the project-specific Mamba placement selected by the variant builder.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .mamba import TriCruciMamba3D


def _drop_path(x: Tensor, drop_prob: float, training: bool) -> Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = x.new_empty(shape).bernoulli_(keep_prob)
    return x * mask.div(keep_prob)


def _load_unetr_basic_block() -> type[nn.Module]:
    try:
        from monai.networks.blocks import UnetrBasicBlock
    except ImportError as exc:
        raise ImportError(
            "MetaUNETR encoder residual adapters require optional MONAI imaging dependencies"
        ) from exc
    return UnetrBasicBlock


class MLP(nn.Module):
    """Channels-last feed-forward sub-block."""

    def __init__(self, dim: int, ratio: float = 4.0) -> None:
        super().__init__()
        hidden_dim = max(1, int(dim * ratio))
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class CNNTokenMixer(nn.Module):
    """Depthwise large-kernel CNN token mixer for channels-last volumes."""

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        kernel_size: int = 7,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0 or kernel_size < 1:
            raise ValueError("kernel_size must be a positive odd integer")
        padding = kernel_size // 2
        self.norm1 = nn.LayerNorm(dim)
        self.dwconv = nn.Conv3d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=dim,
            bias=False,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio)
        self.drop_path = float(drop_path)

    def forward(self, x: Tensor) -> Tensor:
        mixed = self.dwconv(
            self.norm1(x).permute(0, 4, 1, 2, 3).contiguous()
        ).permute(0, 2, 3, 4, 1).contiguous()
        x = x + _drop_path(mixed, self.drop_path, self.training)
        return x + _drop_path(self.mlp(self.norm2(x)), self.drop_path, self.training)


class Downsample3D(nn.Module):
    """Channels-last layer that halves each spatial axis and doubles width."""

    def __init__(self, dim: int, out_dim: int | None = None) -> None:
        super().__init__()
        self.dim = int(dim)
        self.out_dim = 2 * self.dim if out_dim is None else int(out_dim)
        self.norm = nn.LayerNorm(self.dim)
        self.reduction = nn.Conv3d(
            self.dim,
            self.out_dim,
            kernel_size=2,
            stride=2,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x).permute(0, 4, 1, 2, 3).contiguous()
        return self.reduction(x).permute(0, 2, 3, 4, 1).contiguous()


def _cfg_value(cfg: Mapping[str, object] | object, name: str, default: object) -> object:
    if isinstance(cfg, Mapping):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


class Encoder3D(nn.Module):
    """Four token-mixer stages plus a 16x-width bottleneck.

    ``forward`` returns ``(bottleneck, skips)`` where bottleneck and every
    skip are channels-first. ``skips`` is ordered from full resolution to the
    coarsest stage: widths ``[C, C, 2C, 4C, 8C]``.
    """

    def __init__(
        self,
        in_channels: int | Mapping[str, object] = 4,
        base_channels: int = 48,
        depths: Sequence[int] = (2, 2, 2, 2),
        stage_mixer: str = "cnn",
        bottleneck_mixer: str = "cnn",
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        window_size: int | Sequence[int] = 7,
        num_heads: int | Sequence[int] = (3, 6, 12, 24),
        d_state: int = 16,
        d_conv: int = 4,
        mamba_expand: int = 2,
        axis_fusion: str = "sum",
        execution_device: torch.device | str | None = None,
        norm_name: str | tuple[str, dict[str, object]] = (
            "group",
            {"num_groups": 1},
        ),
    ) -> None:
        if isinstance(in_channels, Mapping):
            cfg = in_channels
            in_channels = int(_cfg_value(cfg, "in_channels", 4))
            base_channels = int(_cfg_value(cfg, "base_channels", 48))
            depths = tuple(_cfg_value(cfg, "depths", (2, 2, 2, 2)))  # type: ignore[arg-type]
            mlp_ratio = float(_cfg_value(cfg, "mlp_ratio", 4.0))
            drop_path = float(_cfg_value(cfg, "drop_path", 0.0))
            window_size = _cfg_value(cfg, "window_size", 7)  # type: ignore[assignment]
            num_heads = _cfg_value(cfg, "num_heads", (3, 6, 12, 24))  # type: ignore[assignment]
            d_state = int(_cfg_value(cfg, "d_state", 16))
            d_conv = int(_cfg_value(cfg, "d_conv", 4))
            mamba_expand = int(_cfg_value(cfg, "mamba_expand", 2))
            axis_fusion = str(_cfg_value(cfg, "axis_fusion", "sum"))
            execution_device = _cfg_value(
                cfg,
                "execution_device",
                None,
            )  # type: ignore[assignment]
            norm_name = _cfg_value(
                cfg,
                "norm_name",
                norm_name,
            )  # type: ignore[assignment]

        super().__init__()
        if len(depths) != 4 or any(int(depth) < 1 for depth in depths):
            raise ValueError("depths must contain four positive stage depths")
        if stage_mixer not in {"cnn", "mamba"}:
            raise ValueError("stage_mixer must be 'cnn' or 'mamba'")
        if bottleneck_mixer not in {"cnn", "mamba"}:
            raise ValueError("bottleneck_mixer must be 'cnn' or 'mamba'")

        self.in_channels = int(in_channels)
        self.base_channels = int(base_channels)
        self.depths = tuple(int(depth) for depth in depths)
        self.widths = tuple(self.base_channels * 2**index for index in range(4))
        self.bottleneck_channels = self.base_channels * 16
        self.window_size = window_size
        self.num_heads = num_heads
        self.stage_mixer = stage_mixer
        self.bottleneck_mixer = bottleneck_mixer

        unetr_basic_block = _load_unetr_basic_block()
        self.encoder1 = unetr_basic_block(
            spatial_dims=3,
            in_channels=self.in_channels,
            out_channels=self.base_channels,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.encoder2 = unetr_basic_block(
            spatial_dims=3,
            in_channels=self.base_channels,
            out_channels=self.base_channels,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.encoder3 = unetr_basic_block(
            spatial_dims=3,
            in_channels=2 * self.base_channels,
            out_channels=2 * self.base_channels,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.encoder4 = unetr_basic_block(
            spatial_dims=3,
            in_channels=4 * self.base_channels,
            out_channels=4 * self.base_channels,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.encoder5 = unetr_basic_block(
            spatial_dims=3,
            in_channels=8 * self.base_channels,
            out_channels=8 * self.base_channels,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.encoder10 = unetr_basic_block(
            spatial_dims=3,
            in_channels=self.bottleneck_channels,
            out_channels=self.bottleneck_channels,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.stem = nn.Conv3d(
            self.in_channels,
            self.base_channels,
            kernel_size=2,
            stride=2,
        )

        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for index, (dim, depth) in enumerate(zip(self.widths, self.depths)):
            stage_drop = float(drop_path) * (index + 1) / 4.0
            blocks: list[nn.Module] = []
            for _ in range(depth):
                if stage_mixer == "mamba":
                    blocks.append(
                        TriCruciMamba3D(
                            dim=dim,
                            mlp_ratio=mlp_ratio,
                            drop_path=stage_drop,
                            d_state=d_state,
                            d_conv=d_conv,
                            expand=mamba_expand,
                            axis_fusion=axis_fusion,
                            execution_device=execution_device,
                        )
                    )
                else:
                    blocks.append(
                        CNNTokenMixer(
                            dim=dim,
                            mlp_ratio=mlp_ratio,
                            drop_path=stage_drop,
                        )
                    )
            self.stages.append(nn.Sequential(*blocks))
            self.downsamples.append(Downsample3D(dim))

        self.bottleneck = (
            TriCruciMamba3D(
                dim=self.bottleneck_channels,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path,
                d_state=d_state,
                d_conv=d_conv,
                expand=mamba_expand,
                axis_fusion=axis_fusion,
                execution_device=execution_device,
            )
            if bottleneck_mixer == "mamba"
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        if x.ndim != 5:
            raise ValueError(f"Encoder3D expects [B, C, D, H, W], got {tuple(x.shape)}")
        if any(size % 32 != 0 for size in x.shape[-3:]):
            raise ValueError("input spatial dimensions must be divisible by 32")

        skips = [self.encoder1(x)]
        x = self.stem(x).permute(0, 2, 3, 4, 1).contiguous()
        stem_hidden = F.layer_norm(x, [x.shape[-1]])
        skips.append(
            self.encoder2(
                stem_hidden.permute(0, 4, 1, 2, 3).contiguous()
            )
        )
        adapters = (self.encoder3, self.encoder4, self.encoder5)
        for index, (stage, downsample) in enumerate(zip(self.stages, self.downsamples)):
            x = stage(x)
            x = downsample(x)
            hidden = F.layer_norm(x, [x.shape[-1]]).permute(0, 4, 1, 2, 3).contiguous()
            if index < 3:
                skips.append(adapters[index](hidden))
            else:
                bottleneck_input = self.encoder10(hidden)

        x = bottleneck_input.permute(0, 2, 3, 4, 1).contiguous()
        x = self.bottleneck(x)
        bottleneck = x.permute(0, 4, 1, 2, 3).contiguous()
        return bottleneck, skips


MetaUNETREncoder = Encoder3D


__all__ = [
    "CNNTokenMixer",
    "Downsample3D",
    "Encoder3D",
    "MLP",
    "MetaUNETREncoder",
]
