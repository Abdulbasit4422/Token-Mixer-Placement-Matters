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
from token_mixer.privacy import (
    _redact_local_artifact,
    redact_case_identifiers,
    safe_error_message,
)

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

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            listed = tolist()
        except (TypeError, ValueError, RuntimeError):
            listed = None
        if listed is not None and listed is not value:
            return _json_safe(listed)

    item = getattr(value, "item", None)
    if callable(item):
        try:
            scalar = item()
        except (TypeError, ValueError, RuntimeError):
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


_TIMING_KEYS = frozenset(
    {
        "timing_scope",
        "segment_elapsed_seconds",
        "cumulative_train_seconds",
        "cumulative_validation_seconds",
        "cumulative_elapsed_seconds",
        "measured_train_seconds",
        "measured_validation_seconds",
        "measured_elapsed_seconds",
        "time_to_best_seconds",
        "time_to_best_scope",
    }
)


def _timing_fields(metadata: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        str(key): value
        for key, value in metadata.items()
        if str(key) in _TIMING_KEYS
        or str(key)
        in {
            "train/epoch_seconds",
            "val/epoch_seconds",
            "run/elapsed_seconds",
            "train/time_to_best_seconds",
            "train/time_to_best_scope",
        }
    }
    nested = metadata.get("timing")
    if isinstance(nested, Mapping):
        fields.update({str(key): value for key, value in nested.items()})
    if "timing_scope" in fields:
        fields.setdefault("scope", fields["timing_scope"])
    return fields


def _efficiency_fields(metadata: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        str(key): value
        for key, value in metadata.items()
        if str(key).startswith(("train/", "val/", "run/", "power/"))
    }
    nested = metadata.get("efficiency")
    if isinstance(nested, Mapping):
        fields.update({str(key): value for key, value in nested.items()})
    return fields


def _early_stopping_fields(metadata: Mapping[str, Any]) -> dict[str, Any]:
    configured = metadata.get("early_stopping")
    if isinstance(configured, Mapping):
        result = dict(configured)
    else:
        result = {}
    result.setdefault("stopped", metadata.get("early_stopping/stopped"))
    result.setdefault("stop_epoch", metadata.get("early_stopping/stop_epoch"))
    result.setdefault("best_epoch", metadata.get("early_stopping/best_epoch"))
    return result


def _provenance_fields(
    metadata: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "code_version": _first_value(metadata, config, ("code_version",)) or __version__,
        "architecture": _first_value(
            metadata,
            config,
            ("architecture",),
            ("model", "architecture"),
            ("model", "name"),
        ),
        "variant": _first_value(
            metadata,
            config,
            ("variant",),
            ("model", "variant"),
            ("experiment", "variant"),
        ),
        "protocol": _first_value(
            metadata,
            config,
            ("protocol",),
        ),
        "evaluation_split": _first_value(
            metadata,
            config,
            ("evaluation_split",),
        ),
        "seed": _first_value(
            metadata,
            config,
            ("seed",),
            ("reproducibility", "seed"),
            ("run", "seed"),
        ),
        "device": _first_value(
            metadata,
            config,
            ("execution_device",),
            ("device",),
        ),
        "manifest_hash": _first_value(metadata, config, ("manifest_hash",)),
    }


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


def _read_json_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


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
    timing = _timing_fields(metadata)
    efficiency = _efficiency_fields(metadata)
    early_stopping = _early_stopping_fields(metadata)
    _write_json(
        metrics_path,
        _json_safe(
            _redact_local_artifact(
                # Keep these legacy top-level keys stable for downstream
                # consumers while adding the versioned evidence sections.
                {
                    "schema_version": 2,
                    "best_epoch": result.best_epoch,
                    "best_metric": result.best_metric,
                    "history": result.history,
                    "test_metrics": result.test_metrics,
                    "provenance": _provenance_fields(metadata, config),
                    "timing": timing,
                    "efficiency": efficiency,
                    "early_stopping": early_stopping,
                    "metadata": metadata,
                }
            )
        ),
    )
    _write_json(
        provenance_path,
        _json_safe(
            _redact_local_artifact(
                {
                    "schema_version": 2,
                    "status": "completed",
                    "code_version": code_version,
                    "experiment": experiment,
                    "architecture": architecture,
                    "variant": variant,
                    "protocol": metadata.get("protocol"),
                    "evaluation_split": metadata.get("evaluation_split"),
                    "model_config": model_config,
                    "seed": seed,
                    "device": device,
                    "manifest_hash": manifest_hash,
                    "monitor": monitor,
                    "direction": direction,
                    "source_checkpoint": source_checkpoint,
                    "runtime": _config_value(config, "runtime"),
                    "metadata": metadata,
                    "timing": timing,
                    "efficiency": efficiency,
                    "early_stopping": early_stopping,
                    "tracking": tracking,
                }
            )
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
    metrics_path = output_dir / "metrics.json"
    provenance_path = output_dir / "provenance.json"
    existing = _read_json_mapping(provenance_path)
    prior_evidence = metrics_path.is_file() or provenance_path.is_file()

    def retain(key: str, value: Any) -> Any:
        return value if value is not None else existing.get(key)

    existing_metadata = existing.get("metadata")
    merged_metadata = (
        dict(existing_metadata) if isinstance(existing_metadata, Mapping) else {}
    )
    merged_metadata.update(metadata)
    timing = dict(existing.get("timing", {})) if isinstance(existing.get("timing"), Mapping) else {}
    timing.update(_timing_fields(metadata))
    efficiency = (
        dict(existing.get("efficiency", {}))
        if isinstance(existing.get("efficiency"), Mapping)
        else {}
    )
    efficiency.update(_efficiency_fields(metadata))
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
    code_version = (
        _first_value(metadata, config, ("code_version",))
        or existing.get("code_version")
        or __version__
    )
    _write_json(
        provenance_path,
        _json_safe(
            redact_case_identifiers(
                {
                    "schema_version": 2,
                    "status": "failed",
                    "partial": prior_evidence,
                    "failure_status": "failed",
                    "code_version": code_version,
                    "experiment": retain(
                        "experiment", _config_value(config, "experiment", "name")
                    ),
                    "architecture": retain("architecture", architecture),
                    "variant": retain(
                        "variant",
                        _first_value(
                            metadata,
                            config,
                            ("variant",),
                            ("model", "variant"),
                            ("experiment", "variant"),
                        ),
                    ),
                    "protocol": retain(
                        "protocol", _first_value(metadata, config, ("protocol",))
                    ),
                    "evaluation_split": retain(
                        "evaluation_split",
                        _first_value(metadata, config, ("evaluation_split",)),
                    ),
                    "model_config": retain("model_config", model_config),
                    "seed": retain("seed", seed),
                    "device": retain("device", device),
                    "manifest_hash": retain("manifest_hash", manifest_hash),
                    "error": safe_error_message(
                        error,
                        known_case_ids={
                            "metadata": merged_metadata,
                            "config": config,
                        },
                    ),
                    "metadata": merged_metadata,
                    "timing": timing,
                    "efficiency": efficiency,
                    "tracking": tracking if tracking else existing.get("tracking", {}),
                }
            )
        ),
    )
    return provenance_path


__all__ = ["write_failed_run_artifact", "write_run_artifacts"]
