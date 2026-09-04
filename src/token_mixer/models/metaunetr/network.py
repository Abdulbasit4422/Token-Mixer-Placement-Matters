"""Paper-facing MetaUNETR network and placement-specific assembly.

All variants use the same stem, widths, depths, five skips, decoder scales,
and raw-logit head. The placement differences are the baseline bottleneck
Mamba, Mod A encoder Mambas, and Mod B coarse decoder Mambas. Architecture
references: Oyetunji et al., *Token Mixer Placement Matters: A Systematic
Encoder-Decoder Ablation Study of Mamba for Brain Tumour Segmentation on
BraTS-Africa* (supplied project manuscript); Lyu et al. (MICCAI 2024), the
official MetaUNETR repository, and Gu and Dao's Mamba implementation.
"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral

import torch
from torch import Tensor, nn

from token_mixer.data.labels import REGION_NAMES

from .decoder import CnnDecoder, MambaDecoder3D
from .encoder import Encoder3D


VALID_VARIANTS = ("metaunetr_mamba", "mod_a", "mod_b")
_CANONICAL_INPUT_CHANNELS = 4
_CANONICAL_OUTPUT_CLASSES = 3


class MetaUNETR(nn.Module):
    """UNETR-style 3D segmenter consuming four channels and emitting logits."""

    output_regions = REGION_NAMES

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 3,
        base_channels: int = 48,
        depths: Sequence[int] = (2, 2, 2, 2),
        window_size: int | Sequence[int] = 7,
        num_heads: int | Sequence[int] = (3, 6, 12, 24),
        d_state: int = 16,
        d_conv: int = 4,
        mamba_expand: int = 2,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        axis_fusion: str = "sum",
        execution_device: torch.device | str | None = None,
        variant: str = "metaunetr_mamba",
        norm_name: str | tuple[str, dict[str, object]] = (
            "group",
            {"num_groups": 1},
        ),
    ) -> None:
        super().__init__()
        if variant not in VALID_VARIANTS:
            raise ValueError(
                f"invalid MetaUNETR variant {variant!r}; "
                f"choose one of {VALID_VARIANTS}"
            )
        if (
            isinstance(in_channels, bool)
            or not isinstance(in_channels, Integral)
            or int(in_channels) != _CANONICAL_INPUT_CHANNELS
        ):
            raise ValueError(
                "MetaUNETR requires exactly 4 input channels; "
                f"got {in_channels}"
            )
        if (
            isinstance(num_classes, bool)
            or not isinstance(num_classes, Integral)
            or int(num_classes) != _CANONICAL_OUTPUT_CLASSES
        ):
            raise ValueError(
                "MetaUNETR requires exactly 3 output classes; "
                f"got {num_classes}"
            )
        if in_channels < 1 or num_classes < 1 or base_channels < 1:
            raise ValueError("in_channels, num_classes, and base_channels must be positive")

        self.variant = variant
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.base_channels = int(base_channels)
        self.depths = tuple(int(depth) for depth in depths)
        self.axis_fusion = axis_fusion
        self.spatial_divisor = 32

        stage_mixer = "mamba" if variant == "mod_a" else "cnn"
        bottleneck_mixer = "mamba" if variant == "metaunetr_mamba" else "cnn"
        self.encoder = Encoder3D(
            in_channels=self.in_channels,
            base_channels=self.base_channels,
            depths=self.depths,
            stage_mixer=stage_mixer,
            bottleneck_mixer=bottleneck_mixer,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            window_size=window_size,
            num_heads=num_heads,
            d_state=d_state,
            d_conv=d_conv,
            mamba_expand=mamba_expand,
            axis_fusion=axis_fusion,
            execution_device=execution_device,
            norm_name=norm_name,
        )
        if variant == "mod_b":
            self.decoder = MambaDecoder3D(
                base_channels=self.base_channels,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path,
                d_state=d_state,
                d_conv=d_conv,
                mamba_expand=mamba_expand,
                axis_fusion=axis_fusion,
                execution_device=execution_device,
                norm_name=norm_name,
            )
        else:
            self.decoder = CnnDecoder(
                base_channels=self.base_channels,
                norm_name=norm_name,
            )
        self.head = nn.Conv3d(self.base_channels, self.num_classes, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 5:
            raise ValueError(f"MetaUNETR expects [B, C, D, H, W], got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"MetaUNETR expects {self.in_channels} input channels, got {x.shape[1]}"
            )
        spatial = tuple(int(size) for size in x.shape[-3:])
        if any(size % self.spatial_divisor != 0 for size in spatial):
            raise ValueError(
                f"input spatial dimensions {spatial} must be divisible by "
                f"{self.spatial_divisor} for five downsampling operations"
            )

        bottleneck, skips = self.encoder(x)
        decoded = self.decoder(bottleneck, skips)
        return self.head(decoded)


__all__ = ["MetaUNETR", "VALID_VARIANTS"]
