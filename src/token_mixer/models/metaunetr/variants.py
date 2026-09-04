"""Configuration-backed constructors for the three manuscript variants."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any

from .network import MetaUNETR, VALID_VARIANTS


_MISSING = object()
_CANONICAL_INPUT_CHANNELS = 4
_CANONICAL_OUTPUT_CLASSES = 3


def _value(cfg: Mapping[str, Any] | object, name: str, default: Any) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _canonical_contract(cfg: Mapping[str, Any] | object) -> tuple[int, int]:
    raw_in_channels = _value(cfg, "in_channels", _CANONICAL_INPUT_CHANNELS)
    if isinstance(raw_in_channels, bool) or not isinstance(raw_in_channels, Integral):
        raise ValueError(
            "MetaUNETR requires exactly 4 input channels; "
            f"got {raw_in_channels!r}"
        )
    in_channels = int(raw_in_channels)
    if in_channels != _CANONICAL_INPUT_CHANNELS:
        raise ValueError(
            "MetaUNETR requires exactly 4 input channels; "
            f"got {in_channels}"
        )

    raw_num_classes = _value(cfg, "num_classes", _MISSING)
    raw_out_channels = _value(cfg, "out_channels", _MISSING)
    if raw_num_classes is _MISSING:
        raw_num_classes = (
            _CANONICAL_OUTPUT_CLASSES
            if raw_out_channels is _MISSING
            else raw_out_channels
        )
    if isinstance(raw_num_classes, bool) or not isinstance(raw_num_classes, Integral):
        raise ValueError(
            "MetaUNETR requires exactly 3 output classes; "
            f"got {raw_num_classes!r}"
        )
    num_classes = int(raw_num_classes)
    if num_classes != _CANONICAL_OUTPUT_CLASSES:
        raise ValueError(
            "MetaUNETR requires exactly 3 output classes; "
            f"got {num_classes}"
        )
    if raw_out_channels is not _MISSING:
        if isinstance(raw_out_channels, bool) or not isinstance(raw_out_channels, Integral):
            raise ValueError(
                "MetaUNETR requires exactly 3 output classes; "
                f"got {raw_out_channels!r}"
            )
        out_channels = int(raw_out_channels)
        if out_channels != _CANONICAL_OUTPUT_CLASSES:
            raise ValueError(
                "MetaUNETR requires exactly 3 output classes; "
                f"got {out_channels}"
            )
    return in_channels, num_classes


def _norm_name(cfg: Mapping[str, Any] | object) -> str | tuple[str, dict[str, object]]:
    configured = _value(cfg, "norm_name", "group")
    if isinstance(configured, Sequence) and not isinstance(configured, (str, bytes)):
        values = tuple(configured)
        if len(values) != 2 or not isinstance(values[0], str) or not isinstance(values[1], Mapping):
            raise ValueError("norm_name must be a string or (name, argument mapping) pair")
        return str(values[0]), dict(values[1])
    if not isinstance(configured, str):
        raise ValueError("norm_name must be a string or (name, argument mapping) pair")
    if configured.lower() == "group":
        groups = _value(cfg, "norm_num_groups", 1)
        if groups is not None:
            groups = int(groups)
            if groups < 1:
                raise ValueError("norm_num_groups must be positive")
            return "group", {"num_groups": groups}
    return configured


def build_metaunetr(
    cfg: Mapping[str, Any] | object,
    variant: str,
) -> MetaUNETR:
    """Build one shared MetaUNETR core with variant-specific mixer placement."""

    if variant not in VALID_VARIANTS:
        raise ValueError(
            f"invalid MetaUNETR variant {variant!r}; "
            f"choose one of {VALID_VARIANTS}"
        )
    axis_fusion = str(_value(cfg, "axis_fusion", "sum"))
    if axis_fusion not in {"sum", "cat"}:
        raise ValueError("axis_fusion must be 'sum' or 'cat'")
    in_channels, num_classes = _canonical_contract(cfg)

    depths = tuple(_value(cfg, "depths", (2, 2, 2, 2)))
    num_heads = _value(cfg, "num_heads", (3, 6, 12, 24))
    if isinstance(num_heads, Sequence) and not isinstance(num_heads, (str, bytes)):
        num_heads = tuple(num_heads)

    return MetaUNETR(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=int(_value(cfg, "base_channels", 48)),
        depths=depths,
        window_size=_value(cfg, "window_size", 7),
        num_heads=num_heads,
        d_state=int(_value(cfg, "d_state", 16)),
        d_conv=int(_value(cfg, "d_conv", 4)),
        mamba_expand=int(_value(cfg, "mamba_expand", 2)),
        mlp_ratio=float(_value(cfg, "mlp_ratio", 4.0)),
        drop_path=float(_value(cfg, "drop_path", 0.0)),
        axis_fusion=axis_fusion,
        execution_device=_value(cfg, "execution_device", None),
        variant=variant,
        norm_name=_norm_name(cfg),
    )


__all__ = ["build_metaunetr"]
