"""Lazy MONAI SwinUNETR adapter for the canonical BraTS segmentation contract.

The builder accepts four-channel ``[B, 4, D, H, W]`` inputs and returns raw
three-channel logits ``[B, 3, D, H, W]`` in canonical ``[ET, TC, WT]`` order;
it does not apply activation, thresholding, or channel reordering. The
constructor is based on the official MONAI implementation
(https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/swin_unetr.py)
and the Swin UNETR paper (https://arxiv.org/abs/2201.01266), cross-checked
against the legacy constructor in ``train_swinunetr_new.py:56-61``.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Mapping
from numbers import Integral
from typing import Any

from omegaconf import DictConfig
from torch import Tensor, nn

from token_mixer.data.labels import REGION_NAMES


_MISSING = object()
_CANONICAL_INPUT_CHANNELS = 4
_CANONICAL_OUTPUT_CHANNELS = len(REGION_NAMES)
_OPTIONAL_CONSTRUCTOR_DEFAULTS = {
    "patch_size": 2,
    "depths": (2, 2, 2, 2),
    "num_heads": (3, 6, 12, 24),
    "window_size": 7,
}
_CONSTRUCTOR_ALIASES = {
    "use_checkpoint": ("use_checkpoint", "checkpoint"),
    "drop_rate": ("drop_rate", "drop"),
}


def _config_entry(cfg: DictConfig, name: str) -> tuple[bool, Any]:
    if not isinstance(cfg, Mapping):
        raise TypeError("SwinUNETR configuration must be a mapping")

    model_cfg = cfg.get("model", _MISSING)
    if model_cfg is not _MISSING and model_cfg is not None:
        if not isinstance(model_cfg, Mapping):
            raise TypeError("model configuration must be a mapping")
        if name in model_cfg:
            return True, model_cfg[name]
    if name in cfg:
        return True, cfg[name]
    return False, _MISSING


def _config_value(cfg: DictConfig, name: str, default: Any = _MISSING) -> Any:
    configured, value = _config_entry(cfg, name)
    return value if configured else default


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _rate(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number between 0 and 1")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number between 0 and 1") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be a finite number between 0 and 1")
    return result


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _positive_sequence(
    value: Any,
    name: str,
    expected_length: int,
) -> tuple[int, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{name} must contain {expected_length} positive integers")
    try:
        values = tuple(value)
    except TypeError as exc:
        raise ValueError(
            f"{name} must contain {expected_length} positive integers"
        ) from exc
    if len(values) != expected_length:
        raise ValueError(f"{name} must contain {expected_length} positive integers")
    return tuple(_positive_int(item, name) for item in values)


def _window_size(value: Any) -> int | tuple[int, ...]:
    if isinstance(value, Integral) and not isinstance(value, bool):
        return _positive_int(value, "window_size")
    return _positive_sequence(value, "window_size", 3)


def _aliased_config_value(
    cfg: DictConfig,
    names: tuple[str, ...],
    default: Any,
    validator: Any,
    label: str,
) -> tuple[Any, str | None]:
    configured = [(name, _config_entry(cfg, name)[1]) for name in names if _config_entry(cfg, name)[0]]
    if len(configured) > 1:
        aliases = ", ".join(names)
        raise ValueError(f"configure only one of {aliases} for {label}")
    if not configured:
        return default, None
    source, value = configured[0]
    return validator(value, label), source


def _spatial_size(cfg: DictConfig, spatial_dims: int) -> tuple[int, ...]:
    value = _config_value(cfg, "img_size", _MISSING)
    if value is _MISSING:
        value = _config_value(cfg, "roi_size", _MISSING)
    if value is _MISSING:
        value = _config_value(cfg, "spatial_size", _MISSING)
    if value is _MISSING:
        return (96,) * spatial_dims

    if isinstance(value, bool):
        raise ValueError("img_size must contain positive integers")
    if isinstance(value, Integral):
        sizes = (int(value),) * spatial_dims
    else:
        if isinstance(value, (str, bytes, bytearray)):
            raise ValueError("img_size must contain positive integers")
        try:
            sizes = tuple(value)
        except TypeError as exc:
            raise ValueError("img_size must contain positive integers") from exc
        if len(sizes) != spatial_dims:
            raise ValueError(
                f"img_size must contain {spatial_dims} positive integers"
            )

    if any(
        isinstance(size, bool) or not isinstance(size, Integral) or int(size) <= 0
        for size in sizes
    ):
        raise ValueError("img_size must contain positive integers")
    return tuple(int(size) for size in sizes)


def _load_swinunetr() -> type[nn.Module]:
    try:
        from monai.networks.nets import SwinUNETR
    except ImportError as exc:
        raise ImportError(
            "SwinUNETR requires optional MONAI imaging dependencies; "
            "install the imaging extra with `uv sync --extra imaging`"
        ) from exc
    return SwinUNETR


def _supported_kwargs(
    constructor: type[nn.Module],
    kwargs: Mapping[str, Any],
    legacy_img_size: tuple[int, ...],
    explicit_overrides: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    explicit_overrides = explicit_overrides or {}
    try:
        parameters = inspect.signature(constructor).parameters
    except (TypeError, ValueError) as exc:
        if explicit_overrides:
            options = ", ".join(sorted(explicit_overrides.values()))
            raise TypeError(
                "cannot verify explicit SwinUNETR options without a readable "
                f"constructor signature: {options}"
            ) from exc
        return {
            name: kwargs[name]
            for name in ("in_channels", "out_channels", "feature_size")
            if name in kwargs
        }

    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )

    def accepts(name: str) -> bool:
        parameter = parameters.get(name)
        return parameter is not None and parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )

    filtered: dict[str, Any] = {}
    for name, value in kwargs.items():
        candidates = _CONSTRUCTOR_ALIASES.get(name, (name,))
        if accepts_kwargs:
            target = name
        else:
            target = next((candidate for candidate in candidates if accepts(candidate)), None)
        if target is None:
            source = explicit_overrides.get(name)
            if source is not None:
                raise TypeError(
                    f"explicitly configured option '{source}' is not supported "
                    "by MONAI SwinUNETR"
                )
            continue
        filtered[target] = value

    for name in ("in_channels", "out_channels"):
        if name not in filtered:
            raise TypeError(
                "MONAI SwinUNETR signature cannot enforce canonical "
                f"{name}"
            )

    img_size_parameter = parameters.get("img_size")
    if img_size_parameter is not None and accepts("img_size"):
        filtered["img_size"] = legacy_img_size

    for name, default in _OPTIONAL_CONSTRUCTOR_DEFAULTS.items():
        if name not in filtered and accepts(name) and parameters[name].default is inspect.Parameter.empty:
            filtered[name] = default

    return filtered


class SwinUNETRAdapter(nn.Module):
    """Wrap one MONAI network while exposing its transformer encoder boundary."""

    output_regions = REGION_NAMES

    def __init__(self, network: nn.Module) -> None:
        super().__init__()
        if not isinstance(network, nn.Module):
            raise TypeError("MONAI SwinUNETR constructor must return an nn.Module")
        if not isinstance(getattr(network, "swinViT", None), nn.Module):
            raise TypeError(
                "MONAI SwinUNETR must expose its transformer encoder as 'swinViT'"
            )
        self.network = network
        self.in_channels = _CANONICAL_INPUT_CHANNELS
        self.out_channels = _CANONICAL_OUTPUT_CHANNELS
        self.num_classes = self.out_channels
        self.spatial_dims = 3

    @property
    def encoder(self) -> nn.Module:
        """Return encoder without registering a second module path for it."""
        return self.network.swinViT

    def forward(self, inputs: Tensor) -> Tensor:
        if not isinstance(inputs, Tensor):
            raise TypeError("SwinUNETR expects a torch.Tensor input")
        if inputs.ndim != self.spatial_dims + 2:
            raise ValueError(
                "SwinUNETR expects 5 dimensions [B, C, D, H, W], "
                f"got {tuple(inputs.shape)}"
            )
        if inputs.shape[1] != self.in_channels:
            raise ValueError(
                f"SwinUNETR expects {self.in_channels} input channels, "
                f"got {inputs.shape[1]}"
            )
        spatial = tuple(int(size) for size in inputs.shape[-3:])
        if any(size <= 0 for size in spatial):
            raise ValueError(f"input spatial dimensions must be positive, got {spatial}")

        logits = self.network(inputs)
        if not isinstance(logits, Tensor):
            raise RuntimeError("MONAI SwinUNETR forward must return a tensor of logits")
        expected_shape = (inputs.shape[0], self.out_channels, *spatial)
        if tuple(logits.shape) != expected_shape:
            raise RuntimeError(
                f"SwinUNETR output shape {tuple(logits.shape)} does not match "
                f"expected {expected_shape}"
            )
        return logits


def build_swinunetr(cfg: DictConfig) -> nn.Module:
    """Build configured MONAI SwinUNETR without importing or mutating state eagerly."""

    feature_size_configured, feature_size_value = _config_entry(cfg, "feature_size")
    feature_size = _positive_int(
        feature_size_value if feature_size_configured else 48,
        "feature_size",
    )

    spatial_dims_configured, spatial_dims_value = _config_entry(cfg, "spatial_dims")
    spatial_dims = _positive_int(
        spatial_dims_value if spatial_dims_configured else 3,
        "spatial_dims",
    )
    if spatial_dims != 3:
        raise ValueError("spatial_dims must be exactly 3 for SwinUNETR")

    in_channels_configured, in_channels_value = _config_entry(cfg, "in_channels")
    in_channels = _positive_int(
        in_channels_value if in_channels_configured else _CANONICAL_INPUT_CHANNELS,
        "in_channels",
    )
    if in_channels != _CANONICAL_INPUT_CHANNELS:
        raise ValueError("in_channels must be exactly 4 for canonical BraTS inputs")

    out_channels_configured, out_channels_value = _config_entry(cfg, "out_channels")
    out_channels = _positive_int(
        out_channels_value if out_channels_configured else _CANONICAL_OUTPUT_CHANNELS,
        "out_channels",
    )
    if out_channels != _CANONICAL_OUTPUT_CHANNELS:
        raise ValueError("out_channels must be exactly 3 for canonical BraTS regions")

    use_checkpoint, checkpoint_source = _aliased_config_value(
        cfg,
        ("use_checkpoint", "checkpoint"),
        True,
        _boolean,
        "checkpoint",
    )
    use_v2_configured, use_v2_value = _config_entry(cfg, "use_v2")
    use_v2 = _boolean(use_v2_value if use_v2_configured else False, "use_v2")

    drop_rate, drop_source = _aliased_config_value(
        cfg,
        ("drop_rate", "drop"),
        0.0,
        _rate,
        "drop_rate",
    )
    attn_drop_configured, attn_drop_value = _config_entry(cfg, "attn_drop_rate")
    attn_drop_rate = _rate(
        attn_drop_value if attn_drop_configured else 0.0,
        "attn_drop_rate",
    )
    dropout_path_configured, dropout_path_value = _config_entry(cfg, "dropout_path_rate")
    dropout_path_rate = _rate(
        dropout_path_value if dropout_path_configured else 0.0,
        "dropout_path_rate",
    )

    constructor_kwargs: dict[str, Any] = {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "feature_size": feature_size,
        "use_checkpoint": use_checkpoint,
        "use_v2": use_v2,
        "drop_rate": drop_rate,
        "attn_drop_rate": attn_drop_rate,
        "dropout_path_rate": dropout_path_rate,
        "spatial_dims": spatial_dims,
    }
    explicit_overrides: dict[str, str] = {}

    if feature_size_configured:
        explicit_overrides["feature_size"] = "feature_size"
    if spatial_dims_configured:
        explicit_overrides["spatial_dims"] = "spatial_dims"
    if in_channels_configured:
        explicit_overrides["in_channels"] = "in_channels"
    if out_channels_configured:
        explicit_overrides["out_channels"] = "out_channels"
    if checkpoint_source is not None:
        explicit_overrides["use_checkpoint"] = checkpoint_source
    if use_v2_configured:
        explicit_overrides["use_v2"] = "use_v2"
    if drop_source is not None:
        explicit_overrides["drop_rate"] = drop_source
    if attn_drop_configured:
        explicit_overrides["attn_drop_rate"] = "attn_drop_rate"
    if dropout_path_configured:
        explicit_overrides["dropout_path_rate"] = "dropout_path_rate"

    option_validators = {
        "patch_size": lambda value: _positive_int(value, "patch_size"),
        "depths": lambda value: _positive_sequence(value, "depths", 4),
        "num_heads": lambda value: _positive_sequence(value, "num_heads", 4),
        "window_size": _window_size,
    }
    for name, validator in option_validators.items():
        configured, value = _config_entry(cfg, name)
        if configured:
            constructor_kwargs[name] = validator(value)
            explicit_overrides[name] = name

    swinunetr = _load_swinunetr()
    network = swinunetr(
        **_supported_kwargs(
            swinunetr,
            constructor_kwargs,
            _spatial_size(cfg, spatial_dims),
            explicit_overrides,
        )
    )
    return SwinUNETRAdapter(network)


__all__ = ["SwinUNETRAdapter", "build_swinunetr"]
