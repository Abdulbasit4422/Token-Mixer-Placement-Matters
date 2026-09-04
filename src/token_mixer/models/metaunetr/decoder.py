"""Shared MetaUNETR decoders.

Decoder tensors stay channels-first. CNN paths use MONAI ``UnetrUpBlock``
semantics loaded at construction time; Mod B replaces only its four coarse
refinement blocks with TriCruci scans and keeps ``final`` CNN-only. The
placement study is project-specific and is not a claim of reported paper
weights or metrics.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .mamba import TriCruciMamba3D


NormSpec = str | tuple[str, dict[str, object]]


def _load_monai_blocks() -> tuple[type[nn.Module], type[nn.Module]]:
    try:
        from monai.networks.blocks import UnetrBasicBlock, UnetrUpBlock
    except ImportError as exc:
        raise ImportError(
            "MetaUNETR CNN decoders require optional MONAI imaging dependencies"
        ) from exc
    return UnetrBasicBlock, UnetrUpBlock


def _make_up_block(
    in_channels: int,
    out_channels: int,
    norm_name: NormSpec,
) -> nn.Module:
    _, up_block = _load_monai_blocks()
    return up_block(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        upsample_kernel_size=2,
        norm_name=norm_name,
        res_block=True,
    )


def _align_to_skip(x: Tensor, skip: Tensor) -> Tensor:
    if x.shape[-3:] == skip.shape[-3:]:
        return x
    return F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)


class CnnDecoder(nn.Module):
    """Five-stage CNN decoder using MONAI UNETR up-block behavior."""

    def __init__(
        self,
        base_channels: int = 48,
        norm_name: NormSpec = ("group", {"num_groups": 1}),
    ) -> None:
        super().__init__()
        c = int(base_channels)
        self.up5 = _make_up_block(16 * c, 8 * c, norm_name)
        self.up4 = _make_up_block(8 * c, 4 * c, norm_name)
        self.up3 = _make_up_block(4 * c, 2 * c, norm_name)
        self.up2 = _make_up_block(2 * c, c, norm_name)
        self.final = _make_up_block(c, c, norm_name)

    def forward(self, bottleneck: Tensor, skips: Sequence[Tensor]) -> Tensor:
        if len(skips) != 5:
            raise ValueError("decoder expects five skips ordered full-resolution first")
        x = self.up5(bottleneck, skips[4])
        x = self.up4(x, skips[3])
        x = self.up3(x, skips[2])
        x = self.up2(x, skips[1])
        return self.final(x, skips[0])


class MambaDecoder3DBlock(nn.Module):
    """One coarse decoder upsample, skip fusion, and TriCruci refinement."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        d_state: int = 16,
        d_conv: int = 4,
        mamba_expand: int = 2,
        axis_fusion: str = "sum",
        execution_device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )
        self.merge_norm = nn.LayerNorm(2 * out_channels)
        self.merge = nn.Linear(2 * out_channels, out_channels)
        self.mixer = TriCruciMamba3D(
            dim=out_channels,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            d_state=d_state,
            d_conv=d_conv,
            expand=mamba_expand,
            axis_fusion=axis_fusion,
            execution_device=execution_device,
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = _align_to_skip(self.up(x), skip)
        if x.shape[1] != skip.shape[1]:
            raise ValueError(
                f"skip channel width {skip.shape[1]} does not match {x.shape[1]}"
            )
        x = torch.cat((x, skip), dim=1)
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = self.merge(self.merge_norm(x))
        x = self.mixer(x)
        return x.permute(0, 4, 1, 2, 3).contiguous()


class MambaDecoder3D(nn.Module):
    """Mod B decoder with Mamba in four coarse stages and CNN ``final``."""

    def __init__(
        self,
        base_channels: int = 48,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        d_state: int = 16,
        d_conv: int = 4,
        mamba_expand: int = 2,
        axis_fusion: str = "sum",
        execution_device: torch.device | str | None = None,
        norm_name: NormSpec = ("group", {"num_groups": 1}),
    ) -> None:
        super().__init__()
        c = int(base_channels)
        self.coarse = nn.ModuleList(
            [
                MambaDecoder3DBlock(
                    in_channels=16 * c,
                    out_channels=8 * c,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path,
                    d_state=d_state,
                    d_conv=d_conv,
                    mamba_expand=mamba_expand,
                    axis_fusion=axis_fusion,
                    execution_device=execution_device,
                ),
                MambaDecoder3DBlock(
                    in_channels=8 * c,
                    out_channels=4 * c,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path,
                    d_state=d_state,
                    d_conv=d_conv,
                    mamba_expand=mamba_expand,
                    axis_fusion=axis_fusion,
                    execution_device=execution_device,
                ),
                MambaDecoder3DBlock(
                    in_channels=4 * c,
                    out_channels=2 * c,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path,
                    d_state=d_state,
                    d_conv=d_conv,
                    mamba_expand=mamba_expand,
                    axis_fusion=axis_fusion,
                    execution_device=execution_device,
                ),
                MambaDecoder3DBlock(
                    in_channels=2 * c,
                    out_channels=c,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path,
                    d_state=d_state,
                    d_conv=d_conv,
                    mamba_expand=mamba_expand,
                    axis_fusion=axis_fusion,
                    execution_device=execution_device,
                ),
            ]
        )
        self.final = _make_up_block(c, c, norm_name)

    def forward(self, bottleneck: Tensor, skips: Sequence[Tensor]) -> Tensor:
        if len(skips) != 5:
            raise ValueError("decoder expects five skips ordered full-resolution first")
        x = self.coarse[0](bottleneck, skips[4])
        x = self.coarse[1](x, skips[3])
        x = self.coarse[2](x, skips[2])
        x = self.coarse[3](x, skips[1])
        return self.final(x, skips[0])


CnnMambaDecoder = MambaDecoder3D
MetaUNETRDecoder = CnnDecoder


__all__ = [
    "CnnDecoder",
    "CnnMambaDecoder",
    "MambaDecoder3DBlock",
    "MambaDecoder3D",
    "MetaUNETRDecoder",
]
