"""Configurable 3-D residual U-Net segmentation baseline.

The model consumes ``[B, 4, D, H, W]`` MRI tensors and returns raw logits with
shape ``[B, 3, D, H, W]`` in canonical ``[ET, TC, WT]`` order. The residual
blocks and stages are extracted from ``finetune_nnunet_brats.py:312-428``.
That script describes the network as an nnU-Net ResEncUNet-M equivalent; the
original nnU-Net provenance is Isensee et al., *nnU-Net: Self-adapting
Framework for U-Net-Based Medical Image Segmentation*, arXiv:1809.10486,
https://arxiv.org/abs/1809.10486, with reference implementation at
https://github.com/MIC-DKFZ/nnUNet. Weight loading remains outside this module;
the encoder is exposed as ``model.encoder`` for pure state-dictionary transfer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any

import torch
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.nn import functional as F

from token_mixer.data.labels import REGION_NAMES


NormSpec = str | Mapping[str, Any] | Sequence[Any]
_MISSING = object()
_CANONICAL_INPUT_CHANNELS = 4
_CANONICAL_OUTPUT_CHANNELS = len(REGION_NAMES)
_NUM_ENCODER_STAGES = 5
_NUM_DOWNSAMPLES = 4


def _config_member(config: Any, key: str) -> Any:
    if isinstance(config, Mapping):
        return config[key] if key in config else _MISSING
    return getattr(config, key, _MISSING)


def _config_value(
    config: Any,
    names: Sequence[str],
    default: Any,
    sections: Sequence[str] = ("model",),
) -> Any:
    containers: list[Any] = []
    for section in sections:
        nested = _config_member(config, section)
        if nested is not _MISSING and nested is not None:
            containers.append(nested)
    containers.append(config)

    for container in containers:
        for name in names:
            value = _config_member(container, name)
            if value is not _MISSING and value is not None:
                return value
    return default


def _as_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _as_widths(value: Any) -> tuple[int, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise ValueError("widths must contain five positive integers")
    try:
        values = tuple(value)
    except TypeError as exc:
        raise ValueError("widths must contain five positive integers") from exc
    if len(values) != _NUM_ENCODER_STAGES:
        raise ValueError("widths must contain five positive integers")
    return tuple(
        _as_positive_int(item, "widths")
        for item in values
    )


def _as_depths(value: Any) -> tuple[int, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise ValueError("depths must contain five positive integers")
    try:
        values = tuple(value)
    except TypeError as exc:
        raise ValueError("depths must contain five positive integers") from exc
    if len(values) != _NUM_ENCODER_STAGES:
        raise ValueError("depths must contain five positive integers")
    return tuple(
        _as_positive_int(item, "depths")
        for item in values
    )


def _as_spatial_size(value: Any) -> tuple[int, int, int]:
    if isinstance(value, bool):
        raise ValueError("spatial_size must contain three positive integers")
    if isinstance(value, Integral):
        values = (int(value),) * 3
    else:
        if isinstance(value, (str, bytes, bytearray)):
            raise ValueError("spatial_size must contain three positive integers")
        try:
            values = tuple(value)
        except TypeError as exc:
            raise ValueError(
                "spatial_size must contain three positive integers"
            ) from exc
    if len(values) != 3:
        raise ValueError("spatial_size must contain three positive integers")
    return tuple(
        _as_positive_int(item, "spatial_size")
        for item in values
    )  # type: ignore[return-value]


def _normalization_spec(
    value: NormSpec,
    configured_groups: Any,
) -> tuple[str, int | None]:
    groups = configured_groups
    if isinstance(value, Mapping):
        name = value.get("name", value.get("type", _MISSING))
        groups = value.get("num_groups", value.get("groups", groups))
    elif not isinstance(value, str):
        try:
            parts = tuple(value)
        except TypeError as exc:
            raise ValueError(
                "normalization must be instance, batch, group, or identity"
            ) from exc
        if len(parts) != 2 or not isinstance(parts[0], str):
            raise ValueError(
                "normalization must be instance, batch, group, or identity"
            )
        name = parts[0]
        options = parts[1]
        if isinstance(options, Mapping):
            groups = options.get("num_groups", options.get("groups", groups))
    else:
        name = value

    if not isinstance(name, str):
        raise ValueError(
            "normalization must be instance, batch, group, or identity"
        )
    normalized = name.strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "instance": "instance",
        "instancenorm": "instance",
        "instancenorm3d": "instance",
        "batch": "batch",
        "batchnorm": "batch",
        "batchnorm3d": "batch",
        "group": "group",
        "groupnorm": "group",
        "groupnorm3d": "group",
        "identity": "identity",
        "none": "identity",
    }
    try:
        normalized = aliases[normalized]
    except KeyError as exc:
        raise ValueError(
            "normalization must be instance, batch, group, or identity"
        ) from exc

    if groups is None:
        return normalized, None
    return normalized, _as_positive_int(groups, "normalization groups")


def _normalization_groups(channels: int, configured: int | None) -> int:
    if configured is not None:
        if channels % configured:
            raise ValueError(
                f"normalization groups {configured} do not divide {channels} channels"
            )
        return configured
    for groups in range(min(8, channels), 0, -1):
        if channels % groups == 0:
            return groups
    raise ValueError(f"cannot construct normalization for {channels} channels")


def _make_normalization(
    channels: int,
    normalization: NormSpec,
    norm_num_groups: int | None,
    norm_affine: bool,
) -> nn.Module:
    name, groups = _normalization_spec(normalization, norm_num_groups)
    if name == "instance":
        return nn.InstanceNorm3d(channels, affine=norm_affine)
    if name == "batch":
        return nn.BatchNorm3d(channels, affine=norm_affine)
    if name == "group":
        return nn.GroupNorm(
            _normalization_groups(channels, groups),
            channels,
            affine=norm_affine,
        )
    return nn.Identity()


class ResBlock3D(nn.Module):
    """Two ``3x3x3`` convolutions with a residual projection when required."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        *,
        normalization: NormSpec = "instance",
        norm_num_groups: int | None = None,
        norm_affine: bool = True,
    ) -> None:
        super().__init__()
        in_ch = _as_positive_int(in_ch, "in_ch")
        out_ch = _as_positive_int(out_ch, "out_ch")
        stride = _as_positive_int(stride, "stride")

        self.conv1 = nn.Conv3d(
            in_ch,
            out_ch,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm1 = _make_normalization(
            out_ch,
            normalization,
            norm_num_groups,
            norm_affine,
        )
        self.act = nn.LeakyReLU(0.01, inplace=True)
        self.conv2 = nn.Conv3d(
            out_ch,
            out_ch,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm2 = _make_normalization(
            out_ch,
            normalization,
            norm_num_groups,
            norm_affine,
        )
        self.skip: nn.Module = (
            nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                _make_normalization(
                    out_ch,
                    normalization,
                    norm_num_groups,
                    norm_affine,
                ),
            )
            if in_ch != out_ch or stride != 1
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        hidden = self.act(self.norm1(self.conv1(x)))
        hidden = self.norm2(self.conv2(hidden))
        return self.act(hidden + self.skip(x))


class EncStage(nn.Module):
    """Optional strided residual block followed by plain residual blocks."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        n_blocks: int = 2,
        downsample: bool = True,
        *,
        normalization: NormSpec = "instance",
        norm_num_groups: int | None = None,
        norm_affine: bool = True,
    ) -> None:
        super().__init__()
        n_blocks = _as_positive_int(n_blocks, "n_blocks")
        stride = 2 if downsample else 1
        blocks = [
            ResBlock3D(
                in_ch,
                out_ch,
                stride=stride,
                normalization=normalization,
                norm_num_groups=norm_num_groups,
                norm_affine=norm_affine,
            )
        ]
        blocks.extend(
            ResBlock3D(
                out_ch,
                out_ch,
                normalization=normalization,
                norm_num_groups=norm_num_groups,
                norm_affine=norm_affine,
            )
            for _ in range(n_blocks - 1)
        )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)


class DecStage(nn.Module):
    """Transpose-convolution upsample, skip concatenation, and residual block."""

    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        *,
        normalization: NormSpec = "instance",
        norm_num_groups: int | None = None,
        norm_affine: bool = True,
    ) -> None:
        super().__init__()
        in_ch = _as_positive_int(in_ch, "in_ch")
        skip_ch = _as_positive_int(skip_ch, "skip_ch")
        out_ch = _as_positive_int(out_ch, "out_ch")
        self.up = nn.ConvTranspose3d(
            in_ch,
            out_ch,
            kernel_size=2,
            stride=2,
            bias=False,
        )
        self.block = ResBlock3D(
            out_ch + skip_ch,
            out_ch,
            normalization=normalization,
            norm_num_groups=norm_num_groups,
            norm_affine=norm_affine,
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x,
                size=skip.shape[2:],
                mode="trilinear",
                align_corners=False,
            )
        return self.block(torch.cat((x, skip), dim=1))


class ResUNetEncoder(nn.Module):
    """Five-stage encoder returning bottleneck and four decoder skip tensors."""

    def __init__(
        self,
        in_channels: int,
        widths: Sequence[int],
        depths: Sequence[int],
        *,
        normalization: NormSpec,
        norm_num_groups: int | None,
        norm_affine: bool,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.widths = tuple(int(width) for width in widths)
        self.depths = tuple(int(depth) for depth in depths)
        self.stages = nn.ModuleList(
            [
                EncStage(
                    self.in_channels if index == 0 else self.widths[index - 1],
                    width,
                    n_blocks=self.depths[index],
                    downsample=index != 0,
                    normalization=normalization,
                    norm_num_groups=norm_num_groups,
                    norm_affine=norm_affine,
                )
                for index, width in enumerate(self.widths)
            ]
        )

    @property
    def enc0(self) -> EncStage:
        return self.stages[0]

    @property
    def enc1(self) -> EncStage:
        return self.stages[1]

    @property
    def enc2(self) -> EncStage:
        return self.stages[2]

    @property
    def enc3(self) -> EncStage:
        return self.stages[3]

    @property
    def enc4(self) -> EncStage:
        return self.stages[4]

    def forward(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        skips: list[Tensor] = []
        for stage in self.stages:
            x = stage(x)
            skips.append(x)
        return x, skips[:-1]


class ResUNetDecoder(nn.Module):
    """Four-stage decoder matching the encoder's four spatial reductions."""

    def __init__(
        self,
        widths: Sequence[int],
        *,
        normalization: NormSpec,
        norm_num_groups: int | None,
        norm_affine: bool,
    ) -> None:
        super().__init__()
        widths = tuple(int(width) for width in widths)
        self.dec3 = DecStage(
            widths[4],
            widths[3],
            widths[3],
            normalization=normalization,
            norm_num_groups=norm_num_groups,
            norm_affine=norm_affine,
        )
        self.dec2 = DecStage(
            widths[3],
            widths[2],
            widths[2],
            normalization=normalization,
            norm_num_groups=norm_num_groups,
            norm_affine=norm_affine,
        )
        self.dec1 = DecStage(
            widths[2],
            widths[1],
            widths[1],
            normalization=normalization,
            norm_num_groups=norm_num_groups,
            norm_affine=norm_affine,
        )
        self.dec0 = DecStage(
            widths[1],
            widths[0],
            widths[0],
            normalization=normalization,
            norm_num_groups=norm_num_groups,
            norm_affine=norm_affine,
        )

    def forward(self, bottleneck: Tensor, skips: Sequence[Tensor]) -> Tensor:
        if len(skips) != _NUM_DOWNSAMPLES:
            raise ValueError("decoder expects four skips ordered full-resolution first")
        x = self.dec3(bottleneck, skips[3])
        x = self.dec2(x, skips[2])
        x = self.dec1(x, skips[1])
        return self.dec0(x, skips[0])


def _validate_widths_and_depths(
    base_features: Any,
    widths: Any,
    depths: Any,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    base_features = _as_positive_int(base_features, "base_features")
    validated_widths = (
        tuple(base_features * 2**index for index in range(_NUM_ENCODER_STAGES))
        if widths is None
        else _as_widths(widths)
    )
    validated_depths = _as_depths(depths)
    return validated_widths, validated_depths


def _validated_model_config(
    *,
    in_channels: Any,
    out_channels: Any,
    base_features: Any,
    widths: Any,
    depths: Any,
    normalization: NormSpec,
    norm_num_groups: Any,
    norm_affine: Any,
    spatial_size: Any,
) -> dict[str, Any]:
    in_channels = _as_positive_int(in_channels, "in_channels")
    out_channels = _as_positive_int(out_channels, "out_channels")
    if in_channels != _CANONICAL_INPUT_CHANNELS:
        raise ValueError(
            f"in_channels must be {_CANONICAL_INPUT_CHANNELS} for canonical MRI input"
        )
    if out_channels != _CANONICAL_OUTPUT_CHANNELS:
        raise ValueError(
            f"out_channels must be {_CANONICAL_OUTPUT_CHANNELS} for ET/TC/WT logits"
        )

    widths, depths = _validate_widths_and_depths(base_features, widths, depths)
    normalized_name, normalized_groups = _normalization_spec(
        normalization,
        norm_num_groups,
    )
    if normalized_name == "group" and normalized_groups is not None:
        for width in widths:
            _normalization_groups(width, normalized_groups)

    validated_spatial = None if spatial_size is None else _as_spatial_size(spatial_size)
    return {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "base_features": int(widths[0]),
        "widths": widths,
        "depths": depths,
        "normalization": normalized_name,
        "norm_num_groups": normalized_groups,
        "norm_affine": bool(norm_affine),
        "spatial_size": validated_spatial,
    }


class ResUNet3D(nn.Module):
    """Five-level residual U-Net with canonical three-region raw logits."""

    output_regions = REGION_NAMES
    spatial_divisor = 2**_NUM_DOWNSAMPLES

    def __init__(
        self,
        in_channels: int = _CANONICAL_INPUT_CHANNELS,
        out_channels: int = _CANONICAL_OUTPUT_CHANNELS,
        base_features: int = 32,
        depths: Sequence[int] = (2, 2, 2, 2, 2),
        *,
        widths: Sequence[int] | None = None,
        normalization: NormSpec = "instance",
        norm_num_groups: int | None = None,
        norm_affine: bool = True,
        spatial_size: Sequence[int] | int | None = None,
    ) -> None:
        config = _validated_model_config(
            in_channels=in_channels,
            out_channels=out_channels,
            base_features=base_features,
            widths=widths,
            depths=depths,
            normalization=normalization,
            norm_num_groups=norm_num_groups,
            norm_affine=norm_affine,
            spatial_size=spatial_size,
        )
        super().__init__()

        self.in_channels = config["in_channels"]
        self.out_channels = config["out_channels"]
        self.num_classes = self.out_channels
        self.base_features = config["base_features"]
        self.widths = config["widths"]
        self.depths = config["depths"]
        self.normalization = config["normalization"]
        self.norm_num_groups = config["norm_num_groups"]
        self.spatial_size = config["spatial_size"]

        self.encoder = ResUNetEncoder(
            self.in_channels,
            self.widths,
            self.depths,
            normalization=self.normalization,
            norm_num_groups=self.norm_num_groups,
            norm_affine=config["norm_affine"],
        )
        self.decoder = ResUNetDecoder(
            self.widths,
            normalization=self.normalization,
            norm_num_groups=self.norm_num_groups,
            norm_affine=config["norm_affine"],
        )
        self.head = nn.Conv3d(self.widths[0], self.out_channels, kernel_size=1)
        self._init_decoder()

    def _init_decoder(self) -> None:
        for module in (*self.decoder.modules(), self.head):
            if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="leaky_relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @property
    def enc0(self) -> EncStage:
        return self.encoder.enc0

    @property
    def enc1(self) -> EncStage:
        return self.encoder.enc1

    @property
    def enc2(self) -> EncStage:
        return self.encoder.enc2

    @property
    def enc3(self) -> EncStage:
        return self.encoder.enc3

    @property
    def enc4(self) -> EncStage:
        return self.encoder.enc4

    @property
    def dec3(self) -> DecStage:
        return self.decoder.dec3

    @property
    def dec2(self) -> DecStage:
        return self.decoder.dec2

    @property
    def dec1(self) -> DecStage:
        return self.decoder.dec1

    @property
    def dec0(self) -> DecStage:
        return self.decoder.dec0

    def encoder_params(self) -> list[nn.Parameter]:
        return list(self.encoder.parameters())

    def decoder_params(self) -> list[nn.Parameter]:
        return [*self.decoder.parameters(), *self.head.parameters()]

    def freeze_encoder(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)

    def unfreeze_encoder(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(True)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 5:
            raise ValueError(f"ResUNet3D expects [B, C, D, H, W], got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"ResUNet3D expects {self.in_channels} input channels, got {x.shape[1]}"
            )
        spatial = tuple(int(size) for size in x.shape[-3:])
        if any(size <= 0 for size in spatial):
            raise ValueError(f"input spatial dimensions must be positive, got {spatial}")

        bottleneck, skips = self.encoder(x)
        decoded = self.decoder(bottleneck, skips)
        logits = self.head(decoded)
        expected_shape = (x.shape[0], self.out_channels, *spatial)
        if tuple(logits.shape) != expected_shape:
            raise RuntimeError(
                f"ResUNet3D output shape {tuple(logits.shape)} does not match "
                f"expected {expected_shape}"
            )
        return logits


def _model_config(cfg: DictConfig) -> dict[str, Any]:
    base_features = _config_value(
        cfg,
        ("base_features", "base_channels", "feature_size"),
        32,
    )
    return {
        "in_channels": _config_value(cfg, ("in_channels",), 4),
        "out_channels": _config_value(cfg, ("out_channels", "num_classes"), 3),
        "base_features": base_features,
        "widths": _config_value(cfg, ("widths", "channels"), None),
        "depths": _config_value(cfg, ("depths",), (2, 2, 2, 2, 2)),
        "normalization": _config_value(
            cfg,
            ("normalization", "norm_name", "norm"),
            "instance",
        ),
        "norm_num_groups": _config_value(
            cfg,
            ("norm_num_groups", "num_groups"),
            None,
        ),
        "norm_affine": _config_value(cfg, ("norm_affine",), True),
        "spatial_size": _config_value(
            cfg,
            ("spatial_size", "patch_size", "roi_size", "volume_size"),
            None,
            sections=("model", "data", "dataset", "run"),
        ),
    }


def build_resunet3d(cfg: DictConfig) -> nn.Module:
    """Build the residual U-Net from nested or flat Hydra configuration."""

    return ResUNet3D(**_model_config(cfg))


__all__ = [
    "DecStage",
    "EncStage",
    "ResBlock3D",
    "ResUNet3D",
    "ResUNetDecoder",
    "ResUNetEncoder",
    "build_resunet3d",
]
