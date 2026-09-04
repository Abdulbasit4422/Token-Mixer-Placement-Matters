"""Training entrypoint for the custom 3-D residual U-Net baseline."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from omegaconf import DictConfig
from torch import Tensor, nn

from token_mixer.models.resunet3d import build_resunet3d
from token_mixer.models.weight_transfer import (
    TIMM_RESNET18_NAME,
    load_imagenet_resnet18_weights,
)
from token_mixer.pipelines._baseline_common import (
    LoaderBundle,
    build_checkpoints,
    build_loss,
    build_phases,
    build_volume_evaluator,
    build_volume_loaders,
    run_3d_baseline,
)
from token_mixer.reproducibility import seed_everything
from token_mixer.training.engine import FitResult, fit
from token_mixer.training.tracking import create_tracker


_MISSING = object()
_TRANSFER_CONFIG_PATHS = (
    ("imagenet_transfer",),
    ("imagenet_pretrained",),
    ("use_imagenet_pretrained",),
    ("pretrained",),
    ("weight_transfer",),
    ("transfer",),
    ("model", "imagenet_transfer"),
    ("model", "imagenet_pretrained"),
    ("model", "pretrained"),
    ("model", "weight_transfer"),
    ("model", "transfer"),
)


def _path_value(config: Any, path: tuple[str, ...], default: Any = _MISSING) -> Any:
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


def _first_configured(config: Any, paths: tuple[tuple[str, ...], ...]) -> Any:
    for path in paths:
        value = _path_value(config, path)
        if value is not _MISSING:
            return value
    return _MISSING


def _bool_option(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "on"}:
            return True
        if normalized in {"false", "no", "0", "off"}:
            return False
    raise ValueError(f"ResUNet3D ImageNet transfer option '{name}' must be boolean")


def _transfer_settings(
    cfg: DictConfig | Mapping[str, Any],
    source_model: nn.Module | Mapping[str, Tensor] | None,
) -> dict[str, Any]:
    configured = _first_configured(cfg, _TRANSFER_CONFIG_PATHS)
    explicit_enabled = _MISSING
    options: Mapping[str, Any]
    if configured is _MISSING or configured is None:
        options = {}
    elif isinstance(configured, bool) or isinstance(configured, str):
        options = {"enabled": configured}
    elif isinstance(configured, Mapping):
        options = configured
    else:
        raise TypeError(
            "ResUNet3D ImageNet transfer configuration must be a boolean or mapping"
        )

    configured_source_model = options.get("source_model", _MISSING)
    effective_source_model = source_model
    if effective_source_model is None and isinstance(
        configured_source_model, (nn.Module, Mapping)
    ):
        effective_source_model = configured_source_model

    for key in ("enabled", "requested", "use", "load"):
        if key in options:
            explicit_enabled = options[key]
            break
    if explicit_enabled is _MISSING:
        requested = effective_source_model is not None
    else:
        requested = _bool_option(explicit_enabled, "enabled")

    raw_download = options.get("download", options.get("allow_download", False))
    if "download" not in options and "allow_download" not in options:
        configured_download = _first_configured(
            cfg,
            (
                ("imagenet_download",),
                ("pretrained_download",),
                ("model", "imagenet_download"),
            ),
        )
        if configured_download is not _MISSING:
            raw_download = configured_download
    download = _bool_option(raw_download, "download")
    if explicit_enabled is _MISSING and download:
        requested = True

    raw_cache_dir = options.get("cache_dir", options.get("cache", None))
    if raw_cache_dir is None:
        raw_cache_dir = _first_configured(
            cfg,
            (
                ("paths", "imagenet_cache"),
                ("paths", "cache_dir"),
                ("imagenet_cache_dir",),
                ("pretrained_cache_dir",),
                ("imagenet_cache",),
            ),
        )
        if raw_cache_dir is _MISSING:
            raw_cache_dir = None
    cache_dir = None if raw_cache_dir is None else Path(raw_cache_dir)

    configured_source = options.get("source", options.get("source_name", _MISSING))
    if configured_source is _MISSING:
        configured_source = _first_configured(
            cfg,
            (("imagenet_source",), ("pretrained_source",)),
        )
    if configured_source is _MISSING or configured_source is None:
        source = (
            "injected"
            if effective_source_model is not None
            else f"timm:{TIMM_RESNET18_NAME}"
            if requested
            else None
        )
    elif isinstance(configured_source, (str, Path)):
        source = str(configured_source)
    else:
        source = "configured"

    return {
        "requested": requested,
        "download": download,
        "cache_dir": cache_dir,
        "source": source,
        "source_model": effective_source_model,
    }


def _transfer_report(settings: Mapping[str, Any]) -> dict[str, Any]:
    requested = bool(settings["requested"])
    return {
        "requested": requested,
        "status": "requested" if requested else "not_requested",
        "count": 0,
        "counts": {},
        "source": settings["source"],
        "download": bool(settings["download"]),
        "cache_dir": (
            None
            if settings["cache_dir"] is None
            else str(settings["cache_dir"])
        ),
    }


def _copy_transfer_counts(counts: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in counts.items():
        result[str(key)] = value.item() if isinstance(value, Tensor) else value
    return result


def _transferred_count(counts: Mapping[str, Any]) -> int:
    copied = counts.get("copied")
    if copied is None:
        copied = sum(
            int(counts.get(key, 0))
            for key in ("direct", "inflated", "adapted", "loaded")
        )
    try:
        result = int(copied)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("ImageNet transfer returned a non-numeric copied count") from exc
    if result < 1:
        raise RuntimeError("ImageNet transfer returned no copied encoder tensors")
    return result


def _build_model(
    cfg: DictConfig | Mapping[str, Any],
    settings: Mapping[str, Any],
    report: dict[str, Any],
) -> nn.Module:
    model = build_resunet3d(cfg)
    if not settings["requested"]:
        return model

    try:
        transfer_kwargs: dict[str, Any] = {
            "cache_dir": settings["cache_dir"],
            "download": settings["download"],
        }
        if settings["source_model"] is not None:
            transfer_kwargs["source_model"] = settings["source_model"]
        counts = load_imagenet_resnet18_weights(model, **transfer_kwargs)
        if not isinstance(counts, Mapping):
            raise TypeError("ImageNet transfer must return a count mapping")
        copied_counts = _copy_transfer_counts(counts)
        report["counts"] = copied_counts
        report["count"] = _transferred_count(copied_counts)
        report["status"] = "loaded"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        message = (
            "ResUNet3D ImageNet transfer requested but unavailable "
            f"(source={report['source']!r}, download={report['download']}): {exc}"
        )
        error = RuntimeError(message)
        setattr(error, "transfer_report", dict(report))
        raise error from exc
    return model


def _sync_transfer_metadata(
    metadata: dict[str, Any], report: Mapping[str, Any]
) -> None:
    metadata.update(
        {
            "transfer_requested": report["requested"],
            "transfer_status": report["status"],
            "transfer_count": report["count"],
            "transfer_counts": report["counts"],
            "transfer_source": report["source"],
        }
    )


def build_loaders(cfg: DictConfig | Mapping[str, Any], generator: Any) -> LoaderBundle:
    return build_volume_loaders(cfg, generator)


def _build_evaluator(cfg: DictConfig | Mapping[str, Any], device: Any):
    return build_volume_evaluator(cfg, device)


def _build_checkpoints(cfg: DictConfig | Mapping[str, Any]):
    return build_checkpoints(cfg)


def _build_loss(cfg: DictConfig | Mapping[str, Any]):
    return build_loss(cfg)


def _build_phases(cfg: DictConfig | Mapping[str, Any]):
    return build_phases(cfg)


def run_resunet3d(
    cfg: DictConfig | Mapping[str, Any],
    *,
    source_model: nn.Module | Mapping[str, Tensor] | None = None,
) -> FitResult:
    """Run manifest-backed 3-D training and evaluate the held-out test split."""
    settings = _transfer_settings(cfg, source_model)
    report = _transfer_report(settings)
    metadata_extra = {
        "weight_transfer": report,
        "imagenet_transfer": report,
        "transfer_requested": report["requested"],
        "transfer_status": report["status"],
        "transfer_count": report["count"],
        "transfer_counts": report["counts"],
        "transfer_source": report["source"],
    }

    def model_builder(model_cfg: Mapping[str, Any]) -> nn.Module:
        model = _build_model(model_cfg, settings, report)
        _sync_transfer_metadata(metadata_extra, report)
        return model

    return run_3d_baseline(
        cfg,
        architecture="ResUNet3D",
        model_builder=model_builder,
        loader_builder=build_loaders,
        evaluator_builder=_build_evaluator,
        metadata_extra=metadata_extra,
        seed_fn=seed_everything,
        loss_builder=_build_loss,
        phases_builder=_build_phases,
        checkpoint_builder=_build_checkpoints,
        tracker_builder=create_tracker,
        fit_fn=fit,
    )


__all__ = ["build_loaders", "run_resunet3d"]
