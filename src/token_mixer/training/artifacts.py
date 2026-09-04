"""Small, dependency-free completion artifacts for one training run."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from token_mixer import __version__

if TYPE_CHECKING:
    from token_mixer.training.engine import FitResult


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]

    item = getattr(value, "item", None)
    if callable(item):
        try:
            scalar = item()
        except (TypeError, ValueError):
            scalar = None
        if scalar is not None and scalar is not value:
            return _json_safe(scalar)
    return str(value)


def _config_value(config: Mapping[str, Any], *path: str) -> Any:
    value: Any = config
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _first_value(
    metadata: Mapping[str, Any], config: Mapping[str, Any], *paths: tuple[str, ...]
) -> Any:
    for path in paths:
        value = metadata
        found = True
        for key in path:
            if not isinstance(value, Mapping) or key not in value:
                found = False
                break
            value = value[key]
        if found and value is not None:
            return value
        value = config
        found = True
        for key in path:
            if not isinstance(value, Mapping) or key not in value:
                found = False
                break
            value = value[key]
        if found and value is not None:
            return value
    return None


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if descriptor != -1:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if temporary_path.exists():
            temporary_path.unlink()


def write_run_artifacts(
    output_dir: Path,
    config: Mapping[str, Any],
    result: FitResult,
) -> dict[str, Path]:
    """Write JSON metrics and provenance after successful pipeline completion."""
    if not isinstance(config, Mapping):
        raise TypeError("run configuration must be a mapping")
    from token_mixer.training.engine import FitResult

    if not isinstance(result, FitResult):
        raise TypeError("run completion must return FitResult")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = dict(result.metadata or {})
    tracking = _config_value(config, "tracking")
    if tracking is None:
        tracking = {}

    experiment = _config_value(config, "experiment", "name")
    architecture = _first_value(
        metadata,
        config,
        ("architecture",),
        ("model", "architecture"),
        ("model", "name"),
    )
    variant = _first_value(
        metadata,
        config,
        ("variant",),
        ("model", "variant"),
        ("experiment", "variant"),
    )
    model_config = _first_value(
        metadata,
        config,
        ("model_config",),
        ("source_model_config",),
        ("model",),
    )
    seed = _first_value(
        metadata,
        config,
        ("seed",),
        ("reproducibility", "seed"),
        ("run", "seed"),
    )
    device = _first_value(
        metadata,
        config,
        ("execution_device",),
        ("device",),
    )
    manifest_hash = _first_value(
        metadata,
        config,
        ("manifest_hash",),
    )
    code_version = _first_value(metadata, config, ("code_version",)) or __version__
    monitor = _first_value(metadata, config, ("monitor",)) or "mean_dice"
    direction = _first_value(metadata, config, ("direction",))
    if direction is None:
        maximize = _first_value(metadata, config, ("maximize",))
        direction = "maximize" if maximize is not False else "minimize"
    source_checkpoint = _first_value(
        metadata,
        config,
        ("source_checkpoint",),
        ("resume",),
        ("resume_path",),
        ("warm_start",),
        ("warm_start_path",),
        ("paths", "resume"),
        ("paths", "warm_start"),
    )

    metrics_path = output_dir / "metrics.json"
    provenance_path = output_dir / "provenance.json"
    _write_json(
        metrics_path,
        _json_safe(
            {
                "best_epoch": result.best_epoch,
                "best_metric": result.best_metric,
                "history": result.history,
                "test_metrics": result.test_metrics,
            }
        ),
    )
    _write_json(
        provenance_path,
        _json_safe(
            {
                "schema_version": 1,
                "code_version": code_version,
                "experiment": experiment,
                "architecture": architecture,
                "variant": variant,
                "model_config": model_config,
                "seed": seed,
                "device": device,
                "manifest_hash": manifest_hash,
                "monitor": monitor,
                "direction": direction,
                "source_checkpoint": source_checkpoint,
                "runtime": _config_value(config, "runtime"),
                "metadata": metadata,
                "tracking": tracking,
            }
        ),
    )
    return {"metrics": metrics_path, "provenance": provenance_path}


def write_failed_run_artifact(
    output_dir: Path,
    config: Mapping[str, Any],
    error: BaseException | str,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Write provenance for a failed run without fabricating metric artifacts."""
    if not isinstance(config, Mapping):
        raise TypeError("run configuration must be a mapping")
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, Mapping):
        raise TypeError("failed-run metadata must be a mapping")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = dict(metadata)
    tracking = _config_value(config, "tracking")
    if tracking is None:
        tracking = {}
    architecture = _first_value(
        metadata,
        config,
        ("architecture",),
        ("model", "architecture"),
        ("model", "name"),
    )
    model_config = _first_value(
        metadata,
        config,
        ("model_config",),
        ("model",),
    )
    seed = _first_value(
        metadata,
        config,
        ("seed",),
        ("reproducibility", "seed"),
        ("run", "seed"),
    )
    device = _first_value(
        metadata,
        config,
        ("execution_device",),
        ("device",),
    )
    manifest_hash = _first_value(metadata, config, ("manifest_hash",))
    provenance_path = output_dir / "provenance.json"
    _write_json(
        provenance_path,
        _json_safe(
            {
                "schema_version": 1,
                "status": "failed",
                "code_version": __version__,
                "experiment": _config_value(config, "experiment", "name"),
                "architecture": architecture,
                "model_config": model_config,
                "seed": seed,
                "device": device,
                "manifest_hash": manifest_hash,
                "error": str(error),
                "metadata": metadata,
                "tracking": tracking,
            }
        ),
    )
    return provenance_path


__all__ = ["write_failed_run_artifact", "write_run_artifacts"]
