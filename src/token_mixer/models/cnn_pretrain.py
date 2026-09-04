"""2-D CNN denoising autoencoder for ImageNet encoder pretraining.

The residual CNN blocks preserve the ResNet-style skip structure described by
He et al., *Deep Residual Learning for Image Recognition*,
https://arxiv.org/abs/1512.03385, while retaining the four-stage hierarchy from
the repository's original ``pretrain_cnn.py``. This module owns tensor
operations only; datasets, paths, checkpoints, and tracking stay in the
pretraining pipeline.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any

import torch
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import Dataset


_MISSING = object()
_MODEL_DEFAULTS: dict[str, Any] = {
    "in_channels": 3,
    "feature_size": 32,
    "depths": (1, 1, 1, 1),
    "image_size": 96,
    "mlp_ratio": 4.0,
    "norm_num_groups": None,
}


def _path_value(config: Any, path: Sequence[str], default: Any = _MISSING) -> Any:
    current = config
    for key in path:
        if isinstance(current, Mapping):
            if key not in current:
                return default
            current = current[key]
        else:
            try:
                current = getattr(current, key)
            except AttributeError:
                return default
    return current


def _first_value(config: Any, paths: Sequence[Sequence[str]], default: Any = None) -> Any:
    for path in paths:
        value = _path_value(config, path)
        if value is not _MISSING and value is not None:
            return value
    return default


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_plain(item) for item in value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain(item) for item in value]
    return value


def _as_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _as_depths(value: Any) -> tuple[int, int, int, int]:
    if isinstance(value, (str, bytes, bytearray)):
        raise ValueError("depths must contain four non-negative integers")
    try:
        values = tuple(value)
    except TypeError as exc:
        raise ValueError("depths must contain four non-negative integers") from exc
    if len(values) != 4 or any(isinstance(item, bool) or not isinstance(item, Integral) for item in values):
        raise ValueError("depths must contain four non-negative integers")
    result = tuple(int(item) for item in values)
    if any(item < 0 for item in result):
        raise ValueError("depths must contain four non-negative integers")
    return result  # type: ignore[return-value]


def _as_image_size(value: Any) -> tuple[int, int]:
    if isinstance(value, bool):
        raise ValueError("image_size must contain two positive integers divisible by 32")
    if isinstance(value, Integral):
        result = (int(value), int(value))
    else:
        if isinstance(value, (str, bytes, bytearray)):
            raise ValueError("image_size must contain two positive integers divisible by 32")
        try:
            values = tuple(value)
        except TypeError as exc:
            raise ValueError(
                "image_size must contain two positive integers divisible by 32"
            ) from exc
        if len(values) != 2 or any(
            isinstance(item, bool) or not isinstance(item, Integral) for item in values
        ):
            raise ValueError("image_size must contain two positive integers divisible by 32")
        result = (int(values[0]), int(values[1]))
    if any(item <= 0 or item % 32 for item in result):
        raise ValueError("image_size must contain two positive integers divisible by 32")
    return result


def _model_config(cfg: DictConfig | Mapping[str, Any]) -> dict[str, Any]:
    configured = _first_value(cfg, (("model",),), default={})
    if configured is None:
        configured = {}
    if not isinstance(configured, Mapping):
        raise TypeError("model configuration must be a mapping")

    result = dict(_plain(configured))
    paths: dict[str, tuple[tuple[str, ...], ...]] = {
        "in_channels": (("model", "in_channels"), ("in_channels",)),
        "feature_size": (("model", "feature_size"), ("feature_size",)),
        "depths": (("model", "depths"), ("depths",)),
        "image_size": (
            ("model", "image_size"),
            ("image_size",),
            ("data", "image_size"),
            ("data", "img_size"),
            ("img_size",),
        ),
        "mlp_ratio": (("model", "mlp_ratio"), ("mlp_ratio",)),
        "norm_num_groups": (
            ("model", "norm_num_groups"),
            ("norm_num_groups",),
        ),
    }
    for key, candidates in paths.items():
        if key not in result:
            value = _first_value(cfg, candidates, default=_MISSING)
            if value is not _MISSING:
                result[key] = _plain(value)
        result.setdefault(key, _MODEL_DEFAULTS[key])

    result["in_channels"] = _as_positive_int(result["in_channels"], "in_channels")
    result["feature_size"] = _as_positive_int(result["feature_size"], "feature_size")
    result["depths"] = _as_depths(result["depths"])
    result["image_size"] = _as_image_size(result["image_size"])

    try:
        result["mlp_ratio"] = float(result["mlp_ratio"])
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("mlp_ratio must be a positive finite number") from exc
    if not math.isfinite(result["mlp_ratio"]) or result["mlp_ratio"] <= 0:
        raise ValueError("mlp_ratio must be a positive finite number")

    groups = result["norm_num_groups"]
    if groups is not None:
        result["norm_num_groups"] = _as_positive_int(groups, "normalization groups")
        if result["feature_size"] % result["norm_num_groups"]:
            raise ValueError(
                "normalization groups must divide feature_size and every decoder channel"
            )
    return result


def _normalization_groups(channels: int, configured: int | None = None) -> int:
    """Choose valid GroupNorm groups, including small debug channel widths."""
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


class DenoisingDataset(Dataset[tuple[Tensor, Tensor]]):
    """Wrap an image dataset and return ``(noisy_image, clean_image)`` pairs."""

    def __init__(
        self,
        dataset: Dataset[Any],
        noise_std: float = 0.15,
        channels: int | None = None,
    ) -> None:
        if not math.isfinite(float(noise_std)) or float(noise_std) < 0:
            raise ValueError("noise_std must be a non-negative finite number")
        if channels is not None:
            channels = _as_positive_int(channels, "channels")
        self.ds = dataset
        self.noise_std = float(noise_std)
        self.channels = channels

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        if len(self) == 0:
            raise IndexError("cannot draw from an empty denoising dataset")

        clean: Tensor | None = None
        for attempt in range(10):
            try:
                sample = self.ds[(index + attempt) % len(self)]
                image = sample[0] if isinstance(sample, (tuple, list)) else sample
                candidate = torch.as_tensor(image)
                clean = candidate.float()
                break
            except Exception:
                if attempt == 9:
                    raise
        if clean is None:
            raise RuntimeError("denoising dataset did not return an image")
        if clean.ndim != 3:
            raise ValueError(f"denoising images must have shape [C, H, W], got {tuple(clean.shape)}")
        if self.channels is not None and clean.shape[0] != self.channels:
            raise ValueError(
                f"denoising image has {clean.shape[0]} channels; expected {self.channels}"
            )
        noisy = (clean + self.noise_std * torch.randn_like(clean)).clamp(0.0, 1.0)
        return noisy, clean


class Downsample2D(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv2d(dim, 2 * dim, kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(self.norm(x).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


class CNNBlock2D(nn.Module):
    """Residual convolution plus channel-last MLP refinement."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        dim = _as_positive_int(dim, "block channels")
        hidden = max(1, int(dim * float(mlp_ratio)))
        self.norm1 = nn.LayerNorm(dim)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.norm1(x)
        x = self.conv1(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = residual + x
        return x + self.mlp(self.norm2(x))


class PretrainCNNEncoder(nn.Module):
    """Four-level 2-D CNN encoder returning bottleneck and skip features."""

    def __init__(self, cfg: DictConfig | Mapping[str, Any]) -> None:
        super().__init__()
        model_cfg = _model_config(cfg)
        channels = model_cfg["in_channels"]
        feature_size = model_cfg["feature_size"]
        depths = model_cfg["depths"]
        mlp_ratio = model_cfg["mlp_ratio"]
        self.in_channels = channels
        self.feature_size = feature_size

        self.stem = nn.Sequential(
            nn.Conv2d(channels, max(1, feature_size // 2), kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(max(1, feature_size // 2), feature_size, kernel_size=3, stride=2, padding=1),
        )
        self.stem_norm = nn.LayerNorm(feature_size)

        self.stage1 = nn.ModuleList(
            [CNNBlock2D(feature_size, mlp_ratio) for _ in range(depths[0])]
        )
        self.down1 = Downsample2D(feature_size)
        self.stage2 = nn.ModuleList(
            [CNNBlock2D(2 * feature_size, mlp_ratio) for _ in range(depths[1])]
        )
        self.down2 = Downsample2D(2 * feature_size)
        self.stage3 = nn.ModuleList(
            [CNNBlock2D(4 * feature_size, mlp_ratio) for _ in range(depths[2])]
        )
        self.down3 = Downsample2D(4 * feature_size)
        self.stage4 = nn.ModuleList(
            [CNNBlock2D(8 * feature_size, mlp_ratio) for _ in range(depths[3])]
        )

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        if x.ndim != 4:
            raise ValueError(f"encoder expects [B, C, H, W], got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"encoder expects {self.in_channels} input channels, got {x.shape[1]}"
            )

        x = self.stem(x)
        x = self.stem_norm(x.permute(0, 2, 3, 1))
        s1 = x.permute(0, 3, 1, 2).contiguous()

        for block in self.stage1:
            x = block(x)
        s2 = x.permute(0, 3, 1, 2).contiguous()
        x = self.down1(x)

        for block in self.stage2:
            x = block(x)
        s3 = x.permute(0, 3, 1, 2).contiguous()
        x = self.down2(x)

        for block in self.stage3:
            x = block(x)
        s4 = x.permute(0, 3, 1, 2).contiguous()
        x = self.down3(x)

        for block in self.stage4:
            x = block(x)
        bottleneck = x.permute(0, 3, 1, 2).contiguous()
        return bottleneck, [s1, s2, s3, s4]


class DecoderStage2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        norm_num_groups: int | None = None,
    ) -> None:
        super().__init__()
        groups = _normalization_groups(out_channels, norm_num_groups)
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.refine(torch.cat([x, skip], dim=1))


class DenoisingAutoencoder(nn.Module):
    """CNN encoder-decoder mapping noisy 2-D images back to clean images."""

    def __init__(self, cfg: DictConfig | Mapping[str, Any]) -> None:
        super().__init__()
        model_cfg = _model_config(cfg)
        feature_size = model_cfg["feature_size"]
        in_channels = model_cfg["in_channels"]
        norm_num_groups = model_cfg["norm_num_groups"]
        self.config = model_cfg
        self.in_channels = in_channels
        self.image_size = model_cfg["image_size"]
        self.encoder = PretrainCNNEncoder(model_cfg)

        self.dec4 = DecoderStage2D(
            8 * feature_size, 4 * feature_size, 4 * feature_size, norm_num_groups
        )
        self.dec3 = DecoderStage2D(
            4 * feature_size, 2 * feature_size, 2 * feature_size, norm_num_groups
        )
        self.dec2 = DecoderStage2D(
            2 * feature_size, feature_size, feature_size, norm_num_groups
        )
        decoder_groups = _normalization_groups(feature_size, norm_num_groups)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(feature_size, feature_size, kernel_size=2, stride=2),
            nn.GroupNorm(decoder_groups, feature_size),
            nn.GELU(),
            nn.ConvTranspose2d(feature_size, feature_size, kernel_size=2, stride=2),
            nn.GroupNorm(decoder_groups, feature_size),
            nn.GELU(),
            nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(decoder_groups, feature_size),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(feature_size, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        if not isinstance(x, Tensor) or x.ndim != 4:
            shape = getattr(x, "shape", None)
            raise ValueError(f"autoencoder expects [B, C, H, W], got {shape}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"autoencoder expects {self.in_channels} input channels, got {x.shape[1]}"
            )
        input_size = x.shape[-2:]
        bottleneck, skips = self.encoder(x)
        decoded = self.dec4(bottleneck, skips[3])
        decoded = self.dec3(decoded, skips[2])
        decoded = self.dec2(decoded, skips[1])
        decoded = self.dec1(decoded)
        output = self.head(decoded)
        if output.shape[-2:] != input_size:
            output = F.interpolate(output, size=input_size, mode="bilinear", align_corners=False)
        return output


def build_denoising_model(cfg: DictConfig) -> DenoisingAutoencoder:
    """Build the configured 2-D denoising autoencoder without touching disk."""
    return DenoisingAutoencoder(cfg)


def mse_loss(prediction: Tensor, target: Tensor) -> Tensor:
    """Mean squared reconstruction loss for images in the unit range."""
    return F.mse_loss(prediction, target)


def compute_psnr(mse: float | Tensor) -> float:
    """Convert unit-range MSE to the same PSNR convention as the source script."""
    if torch.is_tensor(mse):
        if mse.numel() != 1:
            raise ValueError("PSNR requires a scalar MSE")
        mse_value = float(mse.detach().cpu().item())
    else:
        mse_value = float(mse)
    if mse_value <= 0:
        return 100.0
    return 10.0 * math.log10(1.0 / mse_value)


def _batch_images(batch: Any) -> tuple[Any, Any]:
    if isinstance(batch, Mapping):
        noisy = batch.get("noisy", batch.get("input", batch.get("image")))
        clean = batch.get("clean", batch.get("target", batch.get("label")))
        if noisy is None or clean is None:
            raise KeyError("denoising batch must contain noisy/input/image and clean/target")
        return noisy, clean
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise TypeError("denoising loader must yield (noisy, clean) pairs")


def _module_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


@torch.no_grad()
def evaluate_denoising(model: nn.Module, loader: Any) -> dict[str, float]:
    """Return batch-size-weighted validation MSE and its unit-range PSNR."""
    device = _module_device(model)
    was_training = model.training
    model.eval()
    total_mse = 0.0
    total_items = 0
    try:
        for batch in loader:
            noisy, clean = _batch_images(batch)
            noisy = noisy if isinstance(noisy, Tensor) else torch.as_tensor(noisy)
            clean = clean if isinstance(clean, Tensor) else torch.as_tensor(clean)
            noisy = noisy.to(device)
            clean = clean.to(device)
            batch_mse = mse_loss(model(noisy), clean)
            batch_size = int(noisy.shape[0]) if noisy.ndim else 1
            total_mse += float(batch_mse.item()) * batch_size
            total_items += batch_size
    finally:
        if was_training:
            model.train()
    if total_items == 0:
        raise ValueError("denoising validation loader yielded no batches")
    mean_mse = total_mse / total_items
    return {"mse": mean_mse, "psnr": compute_psnr(mean_mse)}


__all__ = [
    "CNNBlock2D",
    "DecoderStage2D",
    "DenoisingAutoencoder",
    "DenoisingDataset",
    "Downsample2D",
    "PretrainCNNEncoder",
    "build_denoising_model",
    "compute_psnr",
    "evaluate_denoising",
    "mse_loss",
]
