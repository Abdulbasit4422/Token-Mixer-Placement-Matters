"""Explicit benchmark pipeline for restored model checkpoints.

The pipeline deliberately stays separate from training.  It reuses the model
and loader builders owned by the training entrypoints, restores one selected
``best.pt`` through :class:`CheckpointManager` or ``Tracker.restore_artifact``,
then delegates model-only measurements to
``token_mixer.evaluation.benchmark``.  No W&B SDK import is made here; the
existing tracker abstraction is the only tracking boundary.
"""

from __future__ import annotations

import inspect
import math
import re
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from numbers import Integral
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from token_mixer.data.labels import REGION_NAMES
from token_mixer.evaluation import inference as inference_module
from token_mixer.evaluation.benchmark import (
    BenchmarkResult,
    _json_safe,
    _redact_case_ids,
    hash_case_id,
    run_model_protocol,
    serialize_benchmark,
)
from token_mixer.evaluation.metrics import (
    dice_by_region,
    hd95_by_region,
    hd95_excluded_by_region,
    logits_to_regions,
)
from token_mixer.models.cnn_pretrain import build_denoising_model
from token_mixer.models.metaunetr.variants import build_metaunetr
from token_mixer.models.resunet3d import build_resunet3d
from token_mixer.models.swinunetr import build_swinunetr
from token_mixer.models.transunet import build_transunet, validate_transunet_config
from token_mixer.pipelines._baseline_common import (
    _as_slice_region_target,
    _loader_parts,
    build_slice_loaders,
    build_volume_evaluator,
    build_volume_loaders,
    effective_device,
)
from token_mixer.pipelines.train_metaunetr import (
    _model_config as _metaunetr_model_config,
    build_loaders as build_metaunetr_loaders,
)
from token_mixer.pipelines.pretrain_cnn import (
    _model_config as _cnn_model_config,
    build_dataloaders as build_cnn_dataloaders,
)
from token_mixer.reproducibility import seed_everything
from token_mixer.training.checkpoints import CheckpointManager
from token_mixer.training.engine import _resolve_device
from token_mixer.training.tracking import create_tracker


_MISSING = object()
_PROTOCOLS = {
    "native_3d_full_volume",
    "transunet_2d_slice",
    "cnn_denoising_validation",
}
_DEFAULT_WARMUP_ITERATIONS = 20
_DEFAULT_REPETITIONS = 100
_DEFAULT_BATCH_SIZES = (1, 2, 4, 8)
_DEFAULT_NATIVE_INPUT_SHAPE = (1, 4, 32, 32, 32)
_DEFAULT_SLICE_INPUT_SHAPE = (1, 4, 32, 32)
_DEFAULT_CNN_INPUT_SHAPE = (1, 3, 32, 32)
_ARTIFACT_VERSION_PATTERN = re.compile(r"^v[0-9]+$")
_ARTIFACT_DIGEST_PATTERNS = (
    re.compile(r"^sha256:[0-9a-fA-F]{64}$"),
    re.compile(r"^sha256-[0-9a-fA-F]{64}$"),
    re.compile(r"^md5:[0-9a-fA-F]{32}$"),
    re.compile(r"^md5-[0-9a-fA-F]{32}$"),
    re.compile(r"^xxh128:[0-9a-fA-F]{32}$"),
    re.compile(r"^xxh128-[0-9a-fA-F]{32}$"),
)
_ARTIFACT_AT_DIGEST_PATTERN = re.compile(r"^(?:[0-9a-fA-F]{32}|[0-9a-fA-F]{64})$")
_CASE_HASH_PATTERN = re.compile(r"^[0-9a-fA-F]{16}$")
_CASE_SINGULAR_ALIASES = frozenset({"case_id", "caseId", "id"})
_CASE_PLURAL_ALIASES = frozenset({"case_ids", "caseIds", "ids"})
_CASE_POWER_REASON = (
    "case-level power measurement unsupported; NVML sampler scope is "
    "model-level protocol only"
)
_CASE_POWER_DEFAULTS = {
    "power/average_watts": None,
    "power/max_watts": None,
    "power/energy_joules": None,
    "power/sample_count": 0,
    "power/sample_interval_ms": None,
    "power/status": "unsupported",
    "power/reason": _CASE_POWER_REASON,
    "power/measurement_scope": "model_protocol_only",
    "power/device_index": None,
    "power/nvml_version": None,
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


def _first_value(
    config: Any,
    paths: Sequence[Sequence[str]],
    default: Any = None,
    *,
    keep_none: bool = False,
) -> Any:
    for path in paths:
        value = _path_value(config, path, _MISSING)
        if value is not _MISSING and (keep_none or value is not None):
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


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None or value is _MISSING:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} configuration must be a mapping")
    return value


def _normalized_name(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _experiment_candidates(cfg: Mapping[str, Any]) -> list[str]:
    values: list[Any] = []
    experiment = cfg.get("experiment")
    if isinstance(experiment, Mapping):
        values.extend(
            experiment.get(key)
            for key in ("family", "name", "architecture", "model", "variant")
            if experiment.get(key) is not None
        )
    elif experiment is not None:
        values.append(experiment)
    model = cfg.get("model")
    if isinstance(model, Mapping):
        values.extend(
            model.get(key)
            for key in ("family", "name", "architecture", "variant")
            if model.get(key) is not None
        )
    elif model is not None:
        values.append(model)
    values.extend(
        cfg.get(key)
        for key in ("family", "architecture", "variant", "model_name")
        if cfg.get(key) is not None
    )
    return [_normalized_name(value) for value in values if value is not None]


def _family_from_candidates(candidates: Sequence[str]) -> str | None:
    for candidate in candidates:
        compact = candidate.replace("_", "")
        if compact in {"metaunetr", "metaunetrmamba", "moda", "modb"}:
            return "metaunetr"
        if compact in {"resunet3d", "resunet", "resencunet"}:
            return "resunet3d"
        if compact in {"swinunetr", "swinunetr3d", "swin"}:
            return "swinunetr"
        if compact in {"transunet", "transunet2d", "transunetslice"}:
            return "transunet"
        if compact in {
            "cnn",
            "cnndenoising",
            "cnndenoisingvalidation",
            "cnndenoisingpretrain",
            "cnnpretrain",
            "pretraincnn",
            "denoising",
            "denoisingautoencoder",
        }:
            return "cnn"
    return None


def _family_for_config(cfg: Mapping[str, Any]) -> str:
    family = _family_from_candidates(_experiment_candidates(cfg))
    if family is None:
        raise ValueError(
            "benchmark configuration must select a supported experiment family: "
            "metaunetr_mamba/mod_a/mod_b, ResUNet3D, SwinUNETR, TransUNet, or CNN"
        )
    return family


def _variant_for_config(cfg: Mapping[str, Any]) -> str:
    candidates = [
        _first_value(
            cfg,
            (
                ("benchmark", "variant"),
                ("variant",),
                ("model", "variant"),
                ("experiment", "variant"),
            ),
            default=None,
        )
    ]
    candidates.extend(_experiment_candidates(cfg))
    for candidate in candidates:
        if candidate is None:
            continue
        normalized = _normalized_name(candidate)
        if normalized in {"metaunetr_mamba", "mod_a", "mod_b"}:
            return normalized
    return "metaunetr_mamba"


def select_protocol(cfg: Mapping[str, Any]) -> str:
    """Select one protocol family without merging incompatible model lanes."""

    if not isinstance(cfg, Mapping):
        raise TypeError("benchmark configuration must be a mapping")
    family = _family_for_config(cfg)
    default = {
        "metaunetr": "native_3d_full_volume",
        "resunet3d": "native_3d_full_volume",
        "swinunetr": "native_3d_full_volume",
        "transunet": "transunet_2d_slice",
        "cnn": "cnn_denoising_validation",
    }[family]
    configured = _first_value(
        cfg,
        (("benchmark", "protocol"), ("protocol",)),
        default=None,
    )
    configured_protocol = None if configured is None else str(configured).strip()
    protocol = (
        default
        if configured_protocol is None or configured_protocol.lower() == "auto"
        else configured_protocol
    )
    if protocol not in _PROTOCOLS:
        raise ValueError(
            f"unsupported benchmark protocol {protocol!r}; choose from {sorted(_PROTOCOLS)}"
        )
    expected_family = {
        "native_3d_full_volume": {"metaunetr", "resunet3d", "swinunetr"},
        "transunet_2d_slice": {"transunet"},
        "cnn_denoising_validation": {"cnn"},
    }
    if family not in expected_family[protocol]:
        raise ValueError(
            f"protocol {protocol!r} is incompatible with experiment family {family!r}"
        )
    return protocol


def _model_name(family: str, cfg: Mapping[str, Any]) -> str:
    if family == "metaunetr":
        return _variant_for_config(cfg)
    return {
        "resunet3d": "ResUNet3D",
        "swinunetr": "SwinUNETR",
        "transunet": "TransUNet",
        "cnn": "CNN",
    }[family]


def _benchmark_section(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(cfg.get("benchmark", {}), "benchmark")


def _build_meta_model_config(cfg: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    configured = _first_value(cfg, (("model",), ("experiment", "model")), default={})
    result = dict(_plain(_mapping(configured, "model")))
    for key in (
        "in_channels",
        "num_classes",
        "out_channels",
        "base_channels",
        "depths",
        "window_size",
        "num_heads",
        "d_state",
        "d_conv",
        "mamba_expand",
        "mlp_ratio",
        "drop_path",
        "axis_fusion",
        "norm_name",
        "norm_num_groups",
    ):
        if key not in result and _path_value(cfg, (key,), _MISSING) is not _MISSING:
            result[key] = _plain(_path_value(cfg, (key,)))
    result["execution_device"] = device
    return result


def _injected_transunet_model(cfg: Mapping[str, Any]) -> nn.Module | None:
    value = _first_value(
        cfg,
        (
            ("external_model",),
            ("external_network",),
            ("model", "external_model"),
            ("model", "external_network"),
        ),
        default=None,
    )
    return value if isinstance(value, nn.Module) else None


def _build_model(cfg: Mapping[str, Any], family: str, device: torch.device) -> nn.Module:
    if family == "metaunetr":
        model = build_metaunetr(_build_meta_model_config(cfg, device), _variant_for_config(cfg))
    elif family == "resunet3d":
        model = build_resunet3d(cfg)
    elif family == "swinunetr":
        model = build_swinunetr(cfg)
    elif family == "transunet":
        injected = _injected_transunet_model(cfg)
        if injected is None:
            validate_transunet_config(cfg)
            model = build_transunet(cfg)
        else:
            model = build_transunet(cfg, external_model=injected)
    elif family == "cnn":
        model = build_denoising_model(cfg)
    else:  # pragma: no cover - guarded by _family_for_config
        raise ValueError(f"unsupported benchmark model family {family!r}")
    if not isinstance(model, nn.Module):
        raise TypeError("benchmark model builder must return a torch.nn.Module")
    return model.to(device)


def _build_loaders(
    cfg: Mapping[str, Any], family: str, generator: torch.Generator
) -> Any:
    if family == "metaunetr":
        return build_metaunetr_loaders(cfg, generator)
    if family in {"resunet3d", "swinunetr"}:
        return build_volume_loaders(cfg, generator)
    if family == "transunet":
        return build_slice_loaders(cfg, generator)
    if family == "cnn":
        return build_cnn_dataloaders(cfg, generator)
    raise ValueError(f"unsupported benchmark model family {family!r}")


def _output_dir(cfg: Mapping[str, Any]) -> Path:
    configured = _first_value(
        cfg,
        (
            ("benchmark", "output_dir"),
            ("benchmark", "directory"),
            ("paths", "experiment_output"),
            ("paths", "output_dir"),
            ("experiment", "output_dir"),
            ("paths", "output_root"),
            ("output_dir",),
            ("output_root",),
        ),
        default=None,
    )
    if configured is None:
        raise ValueError("benchmark requires a configured output directory")
    return Path(configured)


def _device(cfg: Mapping[str, Any]) -> torch.device:
    configured = _first_value(
        cfg,
        (
            ("benchmark", "device"),
            ("device",),
            ("training", "device"),
            ("run", "device"),
        ),
        default="auto",
    )
    # Keep the existing baseline resolver as the default boundary.  A
    # benchmark-local device override is resolved through the same engine
    # validation when present.
    if _path_value(cfg, ("benchmark", "device"), _MISSING) is _MISSING:
        return effective_device(cfg)
    return _resolve_device(configured)


def _validated_nonnegative_int(value: Any, name: str, default: int) -> int:
    configured = default if value is None else value
    if isinstance(configured, bool) or not isinstance(configured, Integral):
        raise ValueError(f"{name} must be a non-negative integer")
    result = int(configured)
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _validated_positive_int(value: Any, name: str, default: int) -> int:
    result = _validated_nonnegative_int(value, name, default)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _validated_batch_sizes(value: Any) -> tuple[int, ...]:
    configured = _DEFAULT_BATCH_SIZES if value is None else value
    if isinstance(configured, (str, bytes, bytearray)):
        raise ValueError("batch_sizes must contain positive integers")
    try:
        values = tuple(configured)
    except (TypeError, ValueError) as exc:
        raise ValueError("batch_sizes must contain positive integers") from exc
    if not values or any(
        isinstance(item, bool) or not isinstance(item, Integral) or int(item) < 1
        for item in values
    ):
        raise ValueError("batch_sizes must contain positive integers")
    return tuple(int(item) for item in values)


def _validated_case_limit(value: Any) -> int | None:
    if value is None:
        return None
    return _validated_nonnegative_int(value, "case_limit", 0)


def _case_limit(cfg: Mapping[str, Any]) -> int | None:
    return _validated_case_limit(_benchmark_section(cfg).get("case_limit"))


def _validated_shape(value: Any, name: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{name} must contain positive integers")
    try:
        values = tuple(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain positive integers") from exc
    if not values or any(
        isinstance(item, bool) or not isinstance(item, Integral) or int(item) < 1
        for item in values
    ):
        raise ValueError(f"{name} must contain positive integers")
    return tuple(int(item) for item in values)


def _first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, Mapping):
        for item in value.values():
            found = _first_tensor(item)
            if found is not None:
                return found
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


def _batch_image(batch: Any, family: str) -> Any:
    if isinstance(batch, Mapping):
        keys = (
            ("noisy", "input", "image", "images", "x")
            if family == "cnn"
            else ("image", "images", "input", "inputs", "x")
        )
        for key in keys:
            if key in batch:
                return batch[key]
        raise KeyError("benchmark batch does not contain an input value")
    if isinstance(batch, (tuple, list)) and batch:
        return batch[0]
    raise TypeError("benchmark loader must yield a mapping or non-empty tuple")


def _case_batch_size(batch: Any, family: str) -> int:
    tensor = _first_tensor(_batch_image(batch, family))
    if tensor is None or tensor.ndim == 0:
        return 1
    if family in {"metaunetr", "resunet3d", "swinunetr"}:
        return int(tensor.shape[0]) if tensor.ndim == 5 else 1
    return int(tensor.shape[0]) if tensor.ndim == 4 else 1


def _require_single_case_batch(batch_size: int) -> None:
    if batch_size != 1:
        raise ValueError(
            "case-level latency measurement requires evaluation batch size 1"
        )


def _slice_case_value(
    value: Any,
    batch_size: int,
    count: int,
    *,
    field: str | None = None,
) -> Any:
    field_name = "" if field is None else field.lower()
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value
        if field_name in {"spacing", "spacings"} and value.ndim == 2:
            if int(value.shape[0]) == batch_size:
                return value[:count]
            if int(value.shape[1]) == batch_size:
                return value[:, :count]
        if int(value.shape[0]) == batch_size:
            return value[:count]
        return value
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value
        if field_name in {"spacing", "spacings"} and value.ndim == 2:
            if int(value.shape[0]) == batch_size:
                return value[:count]
            if int(value.shape[1]) == batch_size:
                return value[:, :count]
        if int(value.shape[0]) == batch_size:
            return value[:count]
        return value
    if isinstance(value, Mapping):
        return {
            key: _slice_case_value(item, batch_size, count, field=str(key))
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(
            _slice_case_value(item, batch_size, count, field=field)
            for item in value
        )
    if isinstance(value, list):
        if field_name in {"case_id", "case_ids", "id", "ids"}:
            return value[:count] if len(value) == batch_size else value
        if field_name in {"spacing", "spacings"}:
            if len(value) == batch_size and value and isinstance(value[0], (list, tuple)):
                return value[:count]
            if (
                len(value) == 3
                and value
                and all(isinstance(item, (list, tuple)) for item in value)
                and all(len(item) == batch_size for item in value)
            ):
                return [item[:count] for item in value]
            return value
        if len(value) == batch_size and value and not all(
            isinstance(item, (str, bytes, bytearray, int, float, bool))
            for item in value
        ):
            return value[:count]
        return [
            _slice_case_value(item, batch_size, count, field=field)
            for item in value
        ]
    return value


def _slice_case_batch(batch: Any, batch_size: int, count: int) -> Any:
    if isinstance(batch, Mapping):
        return {
            key: _slice_case_value(value, batch_size, count, field=str(key))
            for key, value in batch.items()
        }
    if isinstance(batch, tuple):
        values = list(batch)
        if len(values) > 2:
            values[2] = _slice_case_value(
                values[2], batch_size, count, field="case_id"
            )
        if len(values) > 3:
            values[3] = _slice_case_value(
                values[3], batch_size, count, field="spacing"
            )
        for index in range(min(2, len(values))):
            values[index] = _slice_case_value(values[index], batch_size, count)
        return tuple(values)
    if isinstance(batch, list):
        values = list(batch)
        if len(values) > 2:
            values[2] = _slice_case_value(
                values[2], batch_size, count, field="case_id"
            )
        if len(values) > 3:
            values[3] = _slice_case_value(
                values[3], batch_size, count, field="spacing"
            )
        for index in range(min(2, len(values))):
            values[index] = _slice_case_value(values[index], batch_size, count)
        return values
    return _slice_case_value(batch, batch_size, count)


class _LimitedCaseLoader:
    def __init__(self, loader: Any, case_limit: int, family: str):
        self.loader = loader
        self.case_limit = case_limit
        self.family = family

    def __iter__(self):
        remaining = self.case_limit
        for batch in self.loader:
            batch_size = _case_batch_size(batch, self.family)
            if batch_size < 1:
                raise ValueError("benchmark loader yielded an empty case batch")
            if batch_size > remaining:
                yield _slice_case_batch(batch, batch_size, remaining)
                return
            yield batch
            remaining -= batch_size
            if remaining == 0:
                return

    def __len__(self) -> int:
        try:
            return min(len(self.loader), self.case_limit)
        except TypeError as exc:
            raise TypeError("limited benchmark loader has no known length") from exc

    def __getattr__(self, name: str) -> Any:
        return getattr(self.loader, name)


def _as_model_input(value: Any, family: str) -> Any:
    if isinstance(value, Mapping):
        return {key: _as_model_input(item, family) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_as_model_input(item, family) for item in value)
    if isinstance(value, list):
        return [_as_model_input(item, family) for item in value]
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    if value.ndim in (2, 3, 4):
        # A four-dimensional value is already batched for 2-D models, but is
        # channel-first/unbatched for native 3-D data.  Family controls that
        # otherwise ambiguous case.
        if family in {"metaunetr", "resunet3d", "swinunetr"} and value.ndim == 4:
            value = value.unsqueeze(0)
        elif family in {"transunet", "cnn"} and value.ndim in (2, 3):
            value = value.unsqueeze(0) if value.ndim == 3 else value.unsqueeze(0).unsqueeze(0)
    return value.float()


def _input_shape_for_family(family: str) -> tuple[int, ...]:
    return {
        "metaunetr": _DEFAULT_NATIVE_INPUT_SHAPE,
        "resunet3d": _DEFAULT_NATIVE_INPUT_SHAPE,
        "swinunetr": _DEFAULT_NATIVE_INPUT_SHAPE,
        "transunet": _DEFAULT_SLICE_INPUT_SHAPE,
        "cnn": _DEFAULT_CNN_INPUT_SHAPE,
    }[family]


def _synthetic_inputs_enabled(cfg: Mapping[str, Any]) -> bool:
    configured = _first_value(
        cfg,
        (
            ("benchmark", "allow_synthetic_inputs"),
            ("benchmark", "synthetic_inputs"),
            ("benchmark", "debug_synthetic_inputs"),
            ("allow_synthetic_inputs",),
        ),
        default=False,
    )
    if isinstance(configured, str):
        return configured.strip().lower() in {"true", "yes", "1", "on"}
    return bool(configured)


def _prepare_inputs(
    cfg: Mapping[str, Any], family: str, loader: Any, device: torch.device
) -> Any:
    configured = _first_value(
        cfg,
        (("benchmark", "input"), ("benchmark", "inputs")),
        default=_MISSING,
    )
    if configured is not _MISSING and configured is not None:
        value = _as_model_input(configured, family)
        return _move_to_device(value, device)

    shape = _first_value(
        cfg,
        (("benchmark", "input_shape"), ("input_shape",)),
        default=None,
    )
    if shape is not None:
        normalized_shape = _validated_shape(shape, "input_shape")
        return torch.zeros(normalized_shape, dtype=torch.float32, device=device)

    try:
        batch = next(iter(loader))
        value = _as_model_input(_batch_image(batch, family), family)
        return _move_to_device(value, device)
    except Exception as error:
        if _synthetic_inputs_enabled(cfg):
            return torch.zeros(
                _input_shape_for_family(family), dtype=torch.float32, device=device
            )
        raise ValueError(
            "failed to prepare benchmark inputs for "
            f"family {family!r} from evaluation loader: "
            f"{type(error).__name__}: {error}"
        ) from error


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    return value


def _source_value(cfg: Mapping[str, Any], key: str) -> Any:
    return _first_value(
        cfg,
        (("benchmark", key), (key,), ("paths", key)),
        default=None,
    )


def _configured_source(value: Any) -> bool:
    if value is None or value is False:
        return False
    return not isinstance(value, str) or bool(value.strip())


def _validated_artifact_reference(value: Any) -> str:
    message = (
        "benchmark.source_artifact must be an immutable W&B artifact reference "
        "qualified by a version such as ':v0' or by a supported digest; "
        "mutable aliases such as ':latest' and ':production' are not allowed"
    )
    if not isinstance(value, str) or value != value.strip() or not value:
        raise ValueError(message)

    if "@" in value:
        collection, separator, digest = value.rpartition("@")
        if not separator or not collection or not (
            digest.startswith("sha256:")
            or digest.startswith("md5:")
            or digest.startswith("xxh128:")
            or _ARTIFACT_AT_DIGEST_PATTERN.fullmatch(digest)
        ):
            raise ValueError(message)
        if ":" in collection:
            raise ValueError(message)
        if ":" in digest:
            algorithm, digest_value = digest.split(":", 1)
            if algorithm == "sha256" and not re.fullmatch(r"[0-9a-fA-F]{64}", digest_value):
                raise ValueError(message)
            if algorithm in {"md5", "xxh128"} and not re.fullmatch(
                r"[0-9a-fA-F]{32}", digest_value
            ):
                raise ValueError(message)
    else:
        collection, separator, qualifier = value.partition(":")
        if not separator or not collection:
            raise ValueError(message)
        if not _ARTIFACT_VERSION_PATTERN.fullmatch(qualifier) and not any(
            pattern.fullmatch(qualifier) for pattern in _ARTIFACT_DIGEST_PATTERNS
        ):
            raise ValueError(message)

    path_parts = collection.split("/")
    if not 1 <= len(path_parts) <= 3 or any(
        not part or part in {".", ".."} or any(char.isspace() for char in part)
        for part in path_parts
    ):
        raise ValueError(message)
    return value


def _validate_benchmark_sources(
    cfg: Mapping[str, Any],
) -> tuple[Any | None, Any | None]:
    source_checkpoint = _source_value(cfg, "source_checkpoint")
    source_artifact = _source_value(cfg, "source_artifact")
    if _configured_source(source_artifact):
        source_artifact = _validated_artifact_reference(source_artifact)
    configured = [
        _configured_source(source_checkpoint),
        _configured_source(source_artifact),
    ]
    if sum(configured) != 1:
        raise ValueError(
            "benchmark requires exactly one source: set either "
            "benchmark.source_checkpoint or benchmark.source_artifact"
        )
    return source_checkpoint, source_artifact


def _local_checkpoint(cfg: Mapping[str, Any], output_dir: Path) -> Path | None:
    configured = _source_value(cfg, "source_checkpoint")
    if configured is not None and configured is not False:
        path = Path(configured).expanduser()
        if path.is_dir():
            path = path / "best.pt"
        return path

    candidates = [
        _first_value(
            cfg,
            (
                ("paths", "checkpoint_dir"),
                ("paths", "checkpoint_root"),
                ("checkpoints", "root"),
                ("checkpoints", "directory"),
                ("checkpoint_dir",),
            ),
            default=None,
        ),
        output_dir,
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        path = Path(candidate)
        path = path / "best.pt" if path.is_dir() or path.suffix == "" else path
        if path.is_file():
            return path
    return None


def _restored_path(value: Any) -> Path:
    path = Path(value)
    if path.is_dir():
        path = path / "best.pt"
    return path


def _checkpoint_model_config(
    cfg: Mapping[str, Any], family: str
) -> dict[str, Any] | None:
    explicit = _first_value(
        cfg,
        (
            ("benchmark", "model_config"),
            ("model_config",),
        ),
        default=_MISSING,
    )
    if explicit is not _MISSING:
        result = dict(_plain(_mapping(explicit, "model_config")))
    elif family == "metaunetr":
        result = dict(_plain(_metaunetr_model_config(cfg)))
    elif family == "cnn":
        result = dict(_plain(_cnn_model_config(cfg)))
    else:
        configured = _first_value(
            cfg,
            (("model",), ("experiment", "model")),
            default=_MISSING,
        )
        if configured is _MISSING:
            return None
        result = dict(_plain(_mapping(configured, "model_config")))
    result.pop("execution_device", None)
    return result or None


def _expected_checkpoint_metadata(
    cfg: Mapping[str, Any], family: str, loader_metadata: Mapping[str, Any]
) -> dict[str, Any]:
    configured = _first_value(
        cfg,
        (("benchmark", "expected_checkpoint_metadata"), ("checkpoint_metadata",)),
        default={},
    )
    expected = dict(_mapping(configured, "expected_checkpoint_metadata"))
    expected.setdefault(
        "architecture",
        {
            "metaunetr": "MetaUNETR",
            "resunet3d": "ResUNet3D",
            "swinunetr": "SwinUNETR",
            "transunet": "TransUNet",
            "cnn": "denoising_autoencoder",
        }[family],
    )
    if family == "metaunetr":
        expected.setdefault("variant", _variant_for_config(cfg))
    model_config = _checkpoint_model_config(cfg, family)
    if model_config is not None:
        expected.setdefault("model_config", model_config)
    manifest_hash = loader_metadata.get("manifest_hash")
    if manifest_hash is not None:
        expected.setdefault("manifest_hash", manifest_hash)
    return expected


def _restore_checkpoint(
    cfg: Mapping[str, Any],
    tracker: Any,
    model: nn.Module,
    output_dir: Path,
    family: str,
    loader_metadata: Mapping[str, Any],
) -> tuple[Path, str | None]:
    source_artifact = _source_value(cfg, "source_artifact")
    if source_artifact is not None and source_artifact is not False:
        source_artifact = _validated_artifact_reference(source_artifact)
        restore = getattr(tracker, "restore_artifact", None)
        if not callable(restore):
            raise RuntimeError("benchmark tracker does not support artifact restoration")
        destination = output_dir / ".benchmark_restore"
        destination.mkdir(parents=True, exist_ok=True)
        restored = restore(str(source_artifact), destination)
        checkpoint = _restored_path(restored)
        source_reference = str(source_artifact)
    else:
        checkpoint = _local_checkpoint(cfg, output_dir)
        if checkpoint is None:
            raise FileNotFoundError(
                "benchmark requires an existing local best.pt or benchmark.source_artifact"
            )
        source_reference = None

    if not checkpoint.is_file():
        raise FileNotFoundError(f"benchmark checkpoint does not exist: {checkpoint}")
    manager = CheckpointManager(checkpoint.parent)
    manager.load_model(
        checkpoint,
        model,
        expected_metadata=_expected_checkpoint_metadata(cfg, family, loader_metadata),
    )
    return checkpoint, source_reference


def _tracker_config(
    cfg: Mapping[str, Any], protocol: str, source_artifact: str | None
) -> dict[str, Any]:
    configured = _mapping(cfg.get("tracking", {}), "tracking")
    result = dict(_plain(configured))
    result["job_type"] = "benchmark"
    if result.get("group") is None:
        study_id = _first_value(
            cfg,
            (("study_id",), ("tracking", "study_id"), ("benchmark", "study_id")),
            default=None,
        )
        if study_id is not None:
            result["group"] = str(study_id)
    if source_artifact is not None:
        result["source_checkpoint_artifact"] = _validated_artifact_reference(
            source_artifact
        )
    source_train_run_id = _source_value(cfg, "source_train_run_id")
    if source_train_run_id is not None:
        result["source_train_run_id"] = str(source_train_run_id)
    result["protocol_id"] = protocol
    return result


def _run_config(
    cfg: Mapping[str, Any],
    family: str,
    model_name: str,
    protocol: str,
    input_shape: Sequence[int],
    device: torch.device,
    loader_metadata: Mapping[str, Any],
    warmup_iterations: int,
    repetitions: int,
    batch_sizes: Sequence[int],
) -> dict[str, Any]:
    result = dict(_plain(cfg))
    result.update(
        {
            "architecture": model_name,
            "model": model_name,
            "family": family,
            "variant": _variant_for_config(cfg) if family == "metaunetr" else None,
            "protocol": protocol,
            "protocol_id": protocol,
            "input_shape": tuple(int(size) for size in input_shape),
            "device": str(device),
            "warmup_iterations": warmup_iterations,
            "repetitions": repetitions,
            "batch_sizes": tuple(int(size) for size in batch_sizes),
            **dict(loader_metadata),
        }
    )
    source_train_run_id = _source_value(cfg, "source_train_run_id")
    if source_train_run_id is not None:
        result["source_train_run_id"] = str(source_train_run_id)
    source_artifact = _source_value(cfg, "source_artifact")
    if source_artifact is not None:
        result["source_checkpoint_artifact"] = _validated_artifact_reference(
            source_artifact
        )
    return result


def _accepts_keyword(function: Callable[..., Any], name: str) -> bool:
    try:
        parameters = inspect.signature(function).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters) or any(
        parameter.name == name
        and parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        for parameter in parameters
    )


def _invoke_model_protocol(
    model: nn.Module,
    inputs: Any,
    *,
    protocol: str,
    warmup_iterations: int,
    repetitions: int,
    batch_sizes: Sequence[int],
    checkpoint: Path,
    benchmark_cfg: Mapping[str, Any],
    efficiency_cfg: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    efficiency = _mapping(efficiency_cfg, "efficiency")
    nvml = _mapping(efficiency.get("nvml", {}), "efficiency.nvml")
    profiler = _mapping(
        efficiency.get("profiler", {}), "efficiency.profiler"
    )
    efficiency_enabled = bool(efficiency.get("enabled", True))
    kwargs: dict[str, Any] = {
        "protocol": protocol,
        "warmup_iterations": warmup_iterations,
        "repetitions": repetitions,
        "batch_sizes": batch_sizes,
    }
    if _accepts_keyword(run_model_protocol, "checkpoint_path"):
        kwargs["checkpoint_path"] = checkpoint
    if _accepts_keyword(run_model_protocol, "power_device_index"):
        configured_power_index = nvml.get(
            "device_index", benchmark_cfg.get("power_device_index")
        )
        if configured_power_index is not None:
            kwargs["power_device_index"] = _validated_nonnegative_int(
                configured_power_index, "power_device_index", 0
            )
    if _accepts_keyword(run_model_protocol, "power_interval_seconds"):
        interval = nvml.get(
            "sample_interval_seconds",
            benchmark_cfg.get("power_interval_seconds", 0.01),
        )
        try:
            interval = float(interval)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("power_interval_seconds must be positive and finite") from exc
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("power_interval_seconds must be positive and finite")
        kwargs["power_interval_seconds"] = interval
    if _accepts_keyword(run_model_protocol, "power_enabled"):
        kwargs["power_enabled"] = efficiency_enabled and bool(
            nvml.get("enabled", True)
        )
    if _accepts_keyword(run_model_protocol, "profiler_enabled"):
        kwargs["profiler_enabled"] = efficiency_enabled and bool(
            profiler.get("enabled", True)
        )
    if _accepts_keyword(run_model_protocol, "mac_tool"):
        kwargs["mac_tool"] = str(profiler.get("mac_tool", "thop"))
    if _accepts_keyword(run_model_protocol, "flop_tool"):
        kwargs["flop_tool"] = str(profiler.get("flop_tool", "fvcore"))
    return dict(run_model_protocol(model, inputs, **kwargs))


def _model_row(model_name: str, protocol: str, output: Mapping[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "row_type": "model",
        "model": model_name,
        "protocol": protocol,
    }
    for key, value in output.items():
        if not isinstance(key, str) or "/" not in key:
            continue
        row[key] = value
        prefix, suffix = key.split("/", 1)
        if prefix in {"model", "inference", "power"}:
            row.setdefault(suffix, value)
    for suffix in (
        "parameters",
        "trainable_parameters",
        "checkpoint_bytes",
        "macs",
        "flops",
        "mac_tool",
        "mac_tool_version",
        "mac_convention",
        "mac_status",
        "flop_tool",
        "flop_tool_version",
        "flop_convention",
        "flop_status",
        "unsupported_ops",
        "uncalled_modules",
        "precision",
        "timing_boundary",
        "hardware",
        "software",
    ):
        namespaced = f"model/{suffix}"
        value = output.get(namespaced, output.get(suffix))
        row.setdefault(namespaced, value)
        row.setdefault(suffix, value)
    for suffix, default in (
        ("average_watts", None),
        ("max_watts", None),
        ("energy_joules", None),
        ("sample_count", 0),
        ("sample_interval_ms", None),
        ("status", "unavailable"),
        ("reason", None),
        ("device_index", None),
        ("nvml_version", None),
    ):
        namespaced = f"power/{suffix}"
        value = output.get(namespaced, default)
        row.setdefault(namespaced, value)
        row.setdefault(suffix, value)
    row["protocol"] = protocol
    return row


def _case_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(_plain(cfg))
    benchmark = dict(_mapping(result.get("benchmark", {}), "benchmark"))
    benchmark.update({"collect_case_records": True, "measure_latency": True})
    result["benchmark"] = benchmark
    return result


def _case_records_from_evaluator(
    evaluation: Any, model_name: str, protocol: str
) -> list[dict[str, Any]]:
    if not isinstance(evaluation, Mapping):
        raise TypeError("benchmark evaluator must return a mapping")
    configured = evaluation.get("case_records", evaluation.get("rows", ()))
    if configured is None:
        return []
    if isinstance(configured, Mapping) or isinstance(configured, (str, bytes, bytearray)):
        raise TypeError("benchmark case records must be a sequence of mappings")
    rows: list[dict[str, Any]] = []
    for index, raw_record in enumerate(configured):
        if not isinstance(raw_record, Mapping):
            raise TypeError("each benchmark case record must be a mapping")
        record = dict(raw_record)
        raw_case_id: Any = _MISSING
        for key in tuple(record):
            key_text = str(key)
            if key_text in _CASE_SINGULAR_ALIASES or key_text in _CASE_PLURAL_ALIASES:
                if raw_case_id is _MISSING:
                    raw_case_id = record[key]
                record.pop(key)

        if isinstance(raw_case_id, (list, tuple)) and len(raw_case_id) == 1:
            raw_case_id = raw_case_id[0]
        supplied_hash = record.get("case_id_hash")
        if raw_case_id is not _MISSING and raw_case_id is not None:
            case_hash = hash_case_id(str(raw_case_id))
        elif isinstance(supplied_hash, str) and _CASE_HASH_PATTERN.fullmatch(
            supplied_hash
        ):
            case_hash = supplied_hash.lower()
        elif supplied_hash is not None:
            case_hash = hash_case_id(str(supplied_hash))
        else:
            case_hash = hash_case_id(str(index))
        record["case_id_hash"] = case_hash
        had_exclusion_flags = "exclusion_flags" in record
        exclusion_flags = record.get("exclusion_flags")
        if "hd95_excluded_by_region" not in record:
            if had_exclusion_flags and isinstance(exclusion_flags, Sequence) and not isinstance(
                exclusion_flags, (str, bytes, bytearray)
            ):
                excluded_regions = {str(region) for region in exclusion_flags}
                record["hd95_excluded_by_region"] = {
                    region: int(region in excluded_regions) for region in REGION_NAMES
                }
            else:
                record["hd95_excluded_by_region"] = {}
        record["row_type"] = "case"
        record["model"] = model_name
        record["protocol"] = protocol
        record.setdefault("dice_by_region", {})
        record.setdefault("hd95_by_region", {})
        record.setdefault("latency_ms", None)
        record.setdefault("voxel_count", None)
        record.setdefault("spacing", None)
        record.setdefault("sliding_window_count", None)
        record.setdefault("exclusion_flags", [])
        if protocol == "cnn_denoising_validation":
            record.setdefault("hd95/status", "not_applicable")
        record.update(
            {
                key: record.get(key, default)
                for key, default in _CASE_POWER_DEFAULTS.items()
            }
        )
        record = _redact_case_ids(record)
        rows.append(record)
    return rows


def _aggregate_case_rows(rows: Sequence[Mapping[str, Any]], protocol: str) -> dict[str, Any]:
    latencies: list[float] = []
    durations: list[float] = []
    window_counts: list[int] = []
    excluded_by_region = {region: 0 for region in REGION_NAMES}
    excluded_case_ids: set[str] = set()
    for index, row in enumerate(rows):
        latency = row.get("latency_ms")
        try:
            latency_value = float(latency)
        except (TypeError, ValueError, OverflowError):
            latency_value = float("nan")
        if math.isfinite(latency_value) and latency_value >= 0:
            latencies.append(latency_value)
            durations.append(latency_value / 1000.0)
        for key in ("end_to_end_seconds",):
            try:
                duration = float(row.get(key))
            except (TypeError, ValueError, OverflowError):
                duration = float("nan")
            if math.isfinite(duration) and duration >= 0:
                if latency_value != latency_value:
                    durations.append(duration)
                break
        try:
            count = int(row.get("sliding_window_count"))
        except (TypeError, ValueError, OverflowError):
            count = 0
        if count > 0:
            window_counts.append(count)
        row_has_exclusions = False
        excluded = row.get("hd95_excluded_by_region")
        if isinstance(excluded, Mapping):
            for region in REGION_NAMES:
                try:
                    region_count = int(excluded.get(region, 0))
                except (TypeError, ValueError, OverflowError):
                    region_count = 0
                if region_count > 0:
                    excluded_by_region[region] += region_count
                    row_has_exclusions = True
        else:
            for region in REGION_NAMES:
                value = row.get(f"hd95_excluded_{region}")
                try:
                    region_count = int(value)
                except (TypeError, ValueError, OverflowError):
                    region_count = 0
                if region_count > 0:
                    excluded_by_region[region] += region_count
                    row_has_exclusions = True
        flags = row.get("exclusion_flags")
        if isinstance(flags, Sequence) and not isinstance(
            flags, (str, bytes, bytearray)
        ):
            row_has_exclusions = row_has_exclusions or any(
                str(region) in REGION_NAMES for region in flags
            )
        if row_has_exclusions:
            case_hash = row.get("case_id_hash")
            excluded_case_ids.add(
                str(case_hash) if case_hash is not None else f"row-{index}"
            )
    total_seconds = sum(durations)
    summary: dict[str, Any] = {
        "inference/case_latency_median_ms": (
            float(np.median(latencies)) if latencies else None
        ),
        "inference/case_latency_p95_ms": (
            float(np.percentile(latencies, 95)) if latencies else None
        ),
        "inference/cases_per_second": (
            len(rows) / total_seconds if total_seconds > 0 else None
        ),
        "inference/sliding_window_count": (
            int(sum(window_counts)) if window_counts else None
        ),
        "inference/n_cases": len(rows),
        "inference/protocol": protocol,
        "inference/hd95_excluded_by_region": excluded_by_region,
        "inference/hd95_excluded_cases": len(excluded_case_ids),
    }
    for region, count in excluded_by_region.items():
        summary[f"inference/hd95_excluded_{region}"] = count
    return summary


def _native_case_protocol(
    cfg: Mapping[str, Any],
    model: nn.Module,
    loader: Any,
    device: torch.device,
    protocol: str,
    model_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    case_limit = _case_limit(cfg)
    if loader is None or isinstance(loader, (str, bytes, bytearray)):
        return [], _aggregate_case_rows([], protocol)
    if case_limit == 0:
        return [], _aggregate_case_rows([], protocol)
    evaluation_loader = (
        loader
        if case_limit is None
        else _LimitedCaseLoader(loader, case_limit, "resunet3d")
    )
    configured = _first_value(
        cfg,
        (
            ("benchmark", "case_evaluator"),
            ("benchmark", "evaluator"),
            ("test_evaluator",),
            ("evaluation", "test_evaluator"),
        ),
        default=None,
    )
    if callable(configured):
        evaluation = configured(model, evaluation_loader)
    else:
        evaluation = build_volume_evaluator(_case_config(cfg), device)(
            model, evaluation_loader
        )
    rows = _case_records_from_evaluator(evaluation, model_name, protocol)
    if case_limit is not None:
        rows = rows[:case_limit]
    return rows, _aggregate_case_rows(rows, protocol)


def _spacing_for_slice(cfg: Mapping[str, Any]) -> tuple[float, float, float]:
    configured = _first_value(
        cfg,
        (
            ("spacing",),
            ("data", "spacing"),
            ("dataset", "spacing"),
            ("evaluation", "spacing"),
            ("benchmark", "spacing"),
        ),
        default=(1.0, 1.0, 1.0),
    )
    try:
        values = tuple(float(item) for item in configured)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("spacing must contain three positive finite values") from exc
    if len(values) != 3 or any(not math.isfinite(item) or item <= 0 for item in values):
        raise ValueError("spacing must contain three positive finite values")
    return values


def _append_slice_sample(
    groups: OrderedDict[str, dict[str, Any]],
    case_id: str,
    image: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    spacing: tuple[float, float, float],
    timings: Mapping[str, float],
) -> None:
    group = groups.setdefault(
        case_id,
        {
            "dice": {region: [] for region in REGION_NAMES},
            "hd95": {region: [] for region in REGION_NAMES},
            "excluded": set(),
            "excluded_counts": {region: 0 for region in REGION_NAMES},
            "voxel_count": 0,
            "spacing": spacing,
            "sliding_window_count": 0,
            "load_seconds": 0.0,
            "preprocess_seconds": 0.0,
            "model_compute_seconds": 0.0,
            "postprocess_seconds": 0.0,
            "metric_seconds": 0.0,
            "end_to_end_seconds": 0.0,
        },
    )
    dice = dice_by_region(prediction, target)
    hd95 = hd95_by_region(prediction, target, spacing=spacing)
    excluded = hd95_excluded_by_region(prediction, target)
    for region in REGION_NAMES:
        group["dice"][region].append(dice[region])
        group["hd95"][region].append(hd95[region])
        if excluded[region]:
            group["excluded"].add(region)
        group["excluded_counts"][region] += int(excluded[region])
    group["voxel_count"] += int(math.prod(image.shape[-2:]))
    group["sliding_window_count"] += 1
    for key, value in timings.items():
        group[key] += float(value)


def _finish_slice_group(case_id: str, group: Mapping[str, Any], model_name: str, protocol: str) -> dict[str, Any]:
    dice = {
        region: float(np.mean(values)) if values else float("nan")
        for region, values in group["dice"].items()
    }
    hd95: dict[str, float] = {}
    for region, values in group["hd95"].items():
        finite = [float(value) for value in values if math.isfinite(float(value))]
        hd95[region] = float(np.mean(finite)) if finite else float("nan")
    end_to_end = float(group["end_to_end_seconds"])
    row: dict[str, Any] = {
        "row_type": "case",
        "case_id_hash": hash_case_id(case_id),
        "model": model_name,
        "protocol": protocol,
        "dice_by_region": dice,
        "hd95_by_region": hd95,
        "hd95_excluded_by_region": dict(group["excluded_counts"]),
        "latency_ms": end_to_end * 1000.0,
        "voxel_count": int(group["voxel_count"]),
        "spacing": group["spacing"],
        "sliding_window_count": int(group["sliding_window_count"]),
        "exclusion_flags": sorted(group["excluded"]),
        **_CASE_POWER_DEFAULTS,
    }
    for key in (
        "load_seconds",
        "preprocess_seconds",
        "model_compute_seconds",
        "postprocess_seconds",
        "metric_seconds",
        "end_to_end_seconds",
    ):
        row[key] = float(group[key])
    return row


def _selected_case_indices(
    case_ids: Sequence[str], selected_case_ids: set[str], case_limit: int | None
) -> list[int]:
    if case_limit is None:
        return list(range(len(case_ids)))
    selected: list[int] = []
    for index, case_id in enumerate(case_ids):
        if case_id in selected_case_ids or len(selected_case_ids) < case_limit:
            selected_case_ids.add(case_id)
            selected.append(index)
    return selected


def _slice_case_protocol(
    cfg: Mapping[str, Any],
    model: nn.Module,
    loader: Any,
    device: torch.device,
    protocol: str,
    model_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    case_limit = _case_limit(cfg)
    if loader is None or isinstance(loader, (str, bytes, bytearray)):
        return [], _aggregate_case_rows([], protocol)
    if case_limit == 0:
        return [], _aggregate_case_rows([], protocol)
    groups: OrderedDict[str, dict[str, Any]] = OrderedDict()
    selected_case_ids: set[str] = set()
    spacing_default = _spacing_for_slice(cfg)
    states = [module.training for module in model.modules()]
    model.eval()
    sample_offset = 0
    try:
        iterator = iter(loader)
        with torch.inference_mode():
            while True:
                load_started = time.perf_counter()
                try:
                    batch = next(iterator)
                except StopIteration:
                    break
                load_seconds = time.perf_counter() - load_started
                image, target, case_ids, batch_spacing = inference_module._unpack_batch(batch)
                image_tensor = image if isinstance(image, torch.Tensor) else torch.as_tensor(image)
                if image_tensor.ndim == 3:
                    image_tensor = image_tensor.unsqueeze(0)
                if image_tensor.ndim != 4:
                    raise ValueError(
                        "TransUNet case inputs must have shape [B, C, H, W]"
                    )
                batch_size = int(image_tensor.shape[0])
                ids = inference_module._as_case_ids(case_ids, batch_size, sample_offset)
                spacings = inference_module._case_spacings(
                    batch_spacing,
                    batch_size,
                    default_spacing=spacing_default,
                )
                target_regions = _as_slice_region_target(target, batch_size)
                selected_indices = _selected_case_indices(
                    ids, selected_case_ids, case_limit
                )
                if not selected_indices:
                    break
                if len(selected_indices) != batch_size:
                    image_tensor = image_tensor[selected_indices]
                    target_regions = target_regions[selected_indices]
                    ids = [ids[index] for index in selected_indices]
                    spacings = [spacings[index] for index in selected_indices]
                    batch_size = len(selected_indices)
                _require_single_case_batch(batch_size)
                preprocess_started = time.perf_counter()
                image_device = image_tensor.to(device)
                target_regions = target_regions.to(device)
                preprocess_seconds = time.perf_counter() - preprocess_started

                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                model_started = time.perf_counter()
                logits = model(image_device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                model_compute_seconds = time.perf_counter() - model_started
                postprocess_started = time.perf_counter()
                prediction = logits_to_regions(logits)
                prediction = prediction.detach()
                postprocess_seconds = time.perf_counter() - postprocess_started

                for index, case_id in enumerate(ids):
                    metric_started = time.perf_counter()
                    _append_slice_sample(
                        groups,
                        case_id,
                        image_tensor[index],
                        prediction[index : index + 1].unsqueeze(2),
                        target_regions[index : index + 1].unsqueeze(2),
                        spacings[index],
                        {
                            "load_seconds": load_seconds / batch_size,
                            "preprocess_seconds": preprocess_seconds / batch_size,
                            "model_compute_seconds": model_compute_seconds / batch_size,
                            "postprocess_seconds": postprocess_seconds / batch_size,
                            "metric_seconds": 0.0,
                            "end_to_end_seconds": 0.0,
                        },
                    )
                    metric_seconds = (
                        time.perf_counter() - metric_started
                    )
                    groups[case_id]["metric_seconds"] += metric_seconds
                    groups[case_id]["end_to_end_seconds"] += (
                        load_seconds / batch_size
                        + preprocess_seconds / batch_size
                        + model_compute_seconds / batch_size
                        + postprocess_seconds / batch_size
                        + metric_seconds
                    )
                sample_offset += batch_size
    finally:
        for module, was_training in zip(model.modules(), states):
            module.training = was_training
    rows = [
        _finish_slice_group(case_id, group, model_name, protocol)
        for case_id, group in groups.items()
    ]
    return rows, _aggregate_case_rows(rows, protocol)


def _cnn_case_protocol(
    model: nn.Module,
    loader: Any,
    device: torch.device,
    protocol: str,
    model_name: str,
    *,
    case_limit: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    case_limit = _validated_case_limit(case_limit)
    if loader is None or isinstance(loader, (str, bytes, bytearray)):
        return [], _aggregate_case_rows([], protocol)
    if case_limit == 0:
        return [], _aggregate_case_rows([], protocol)
    rows: list[dict[str, Any]] = []
    states = [module.training for module in model.modules()]
    model.eval()
    sample_offset = 0
    processed_cases = 0
    try:
        with torch.inference_mode():
            iterator = iter(loader)
            while True:
                load_started = time.perf_counter()
                try:
                    batch = next(iterator)
                except StopIteration:
                    break
                load_seconds = time.perf_counter() - load_started

                preprocess_started = time.perf_counter()
                if isinstance(batch, Mapping):
                    noisy = batch.get("noisy", batch.get("input", batch.get("image")))
                    clean = batch.get("clean", batch.get("target", batch.get("label")))
                    case_ids = batch.get("case_id", batch.get("case_ids"))
                elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
                    noisy, clean = batch[0], batch[1]
                    case_ids = batch[2] if len(batch) > 2 else None
                else:
                    raise TypeError("CNN benchmark loader must yield input/target pairs")
                noisy_tensor = noisy if isinstance(noisy, torch.Tensor) else torch.as_tensor(noisy)
                clean_tensor = clean if isinstance(clean, torch.Tensor) else torch.as_tensor(clean)
                if noisy_tensor.ndim == 3:
                    noisy_tensor = noisy_tensor.unsqueeze(0)
                if clean_tensor.ndim == 3:
                    clean_tensor = clean_tensor.unsqueeze(0)
                if noisy_tensor.ndim != 4 or clean_tensor.ndim != 4:
                    raise ValueError("CNN benchmark images must have shape [B, C, H, W]")
                batch_size = int(noisy_tensor.shape[0])
                ids = inference_module._as_case_ids(case_ids, batch_size, sample_offset)
                if case_limit is not None:
                    remaining = case_limit - processed_cases
                    if remaining <= 0:
                        break
                    if batch_size > remaining:
                        noisy_tensor = noisy_tensor[:remaining]
                        clean_tensor = clean_tensor[:remaining]
                        ids = ids[:remaining]
                        batch_size = remaining
                _require_single_case_batch(batch_size)
                noisy_device = noisy_tensor.to(device)
                preprocess_seconds = time.perf_counter() - preprocess_started

                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                model_started = time.perf_counter()
                prediction = model(noisy_device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                model_compute_seconds = time.perf_counter() - model_started

                postprocess_started = time.perf_counter()
                if not isinstance(prediction, torch.Tensor):
                    raise TypeError("CNN benchmark model must return a torch.Tensor")
                prediction = prediction.detach().cpu()
                clean_tensor = clean_tensor.detach().cpu().float()
                postprocess_seconds = time.perf_counter() - postprocess_started
                if prediction.shape != clean_tensor.shape:
                    raise ValueError(
                        "CNN benchmark prediction and target shapes must match, got "
                        f"{tuple(prediction.shape)} and {tuple(clean_tensor.shape)}"
                    )
                for index, case_id in enumerate(ids):
                    metric_started = time.perf_counter()
                    mse = float(torch.mean((prediction[index].float() - clean_tensor[index]) ** 2).item())
                    metric_seconds = time.perf_counter() - metric_started
                    end_to_end_seconds = (
                        load_seconds / batch_size
                        + preprocess_seconds / batch_size
                        + model_compute_seconds / batch_size
                        + postprocess_seconds / batch_size
                        + metric_seconds
                    )
                    rows.append(
                        {
                            "row_type": "case",
                            "case_id_hash": hash_case_id(case_id),
                            "model": model_name,
                            "protocol": protocol,
                            "dice_by_region": {},
                            "hd95_by_region": {},
                            "hd95_excluded_by_region": {},
                            "hd95/status": "not_applicable",
                            "latency_ms": end_to_end_seconds * 1000.0,
                            "voxel_count": int(math.prod(noisy_tensor.shape[-2:])),
                            "spacing": None,
                            "sliding_window_count": 1,
                            "exclusion_flags": [],
                            **_CASE_POWER_DEFAULTS,
                            "mse": mse,
                            "load_seconds": load_seconds / batch_size,
                            "preprocess_seconds": preprocess_seconds / batch_size,
                            "model_compute_seconds": model_compute_seconds / batch_size,
                            "postprocess_seconds": postprocess_seconds / batch_size,
                            "metric_seconds": metric_seconds,
                            "end_to_end_seconds": end_to_end_seconds,
                        }
                    )
                processed_cases += batch_size
                sample_offset += batch_size
                if case_limit is not None and processed_cases >= case_limit:
                    break
    finally:
        for module, was_training in zip(model.modules(), states):
            module.training = was_training
    return rows, _aggregate_case_rows(rows, protocol)


def _run_case_protocol(
    cfg: Mapping[str, Any],
    family: str,
    protocol: str,
    model_name: str,
    model: nn.Module,
    loader: Any,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    case_limit = _case_limit(cfg)
    if protocol == "native_3d_full_volume":
        return _native_case_protocol(cfg, model, loader, device, protocol, model_name)
    if protocol == "transunet_2d_slice":
        return _slice_case_protocol(cfg, model, loader, device, protocol, model_name)
    if protocol == "cnn_denoising_validation":
        return _cnn_case_protocol(
            model,
            loader,
            device,
            protocol,
            model_name,
            case_limit=case_limit,
        )
    raise ValueError(f"unsupported benchmark protocol {protocol!r}")


def _summary_from_model_output(
    output: Mapping[str, Any], protocol: str
) -> dict[str, Any]:
    summary = {
        str(key): value for key, value in output.items() if isinstance(key, str) and "/" in key
    }
    summary["inference/protocol"] = protocol
    summary["benchmark/protocol"] = protocol
    return summary


def _table_payload(rows: Sequence[Mapping[str, Any]]) -> tuple[list[str], list[list[Any]]]:
    safe_rows = [_redact_case_ids(row) for row in rows]
    columns: list[str] = []
    for row in safe_rows:
        for key in row:
            key = str(key)
            if key not in columns:
                columns.append(key)
    if not columns:
        columns = ["row_type"]
    return columns, [[_json_safe(row.get(column)) for column in columns] for row in safe_rows]


def _tracker_run_id(tracker: Any) -> str | None:
    value = getattr(tracker, "run_id", None)
    return None if value is None else str(value)


def run_benchmark(cfg: Mapping[str, Any]) -> BenchmarkResult:
    """Restore one checkpoint and execute model- and case-level protocols."""

    if not isinstance(cfg, Mapping):
        raise TypeError("benchmark configuration must be a mapping")
    _source_checkpoint, configured_source_artifact = _validate_benchmark_sources(cfg)
    family = _family_for_config(cfg)
    protocol = select_protocol(cfg)
    model_name = _model_name(family, cfg)
    benchmark_cfg = _benchmark_section(cfg)
    warmup_iterations = _validated_nonnegative_int(
        benchmark_cfg.get("warmup_iterations"),
        "warmup_iterations",
        _DEFAULT_WARMUP_ITERATIONS,
    )
    repetitions = _validated_positive_int(
        benchmark_cfg.get("repetitions"), "repetitions", _DEFAULT_REPETITIONS
    )
    batch_sizes = _validated_batch_sizes(benchmark_cfg.get("batch_sizes"))
    case_limit = _validated_case_limit(benchmark_cfg.get("case_limit"))
    seed = _first_value(
        cfg,
        (("seed",), ("reproducibility", "seed"), ("run", "seed")),
        default=0,
    )
    deterministic = bool(
        _first_value(
            cfg,
            (("deterministic",), ("reproducibility", "deterministic")),
            default=True,
        )
    )
    generator = seed_everything(int(seed), deterministic=deterministic)
    device = _device(cfg)
    output_dir = _output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = _build_model(cfg, family, device)
    loader_result = _build_loaders(cfg, family, generator)
    _train_loader, _val_loader, test_loader, loader_metadata = _loader_parts(loader_result)
    if not isinstance(loader_metadata, Mapping):
        raise TypeError("benchmark loader metadata must be a mapping")
    evaluation_loader = test_loader if test_loader is not None else _val_loader
    inputs = _prepare_inputs(cfg, family, evaluation_loader, device)
    input_tensor = _first_tensor(inputs)
    input_shape = tuple(int(size) for size in input_tensor.shape) if input_tensor is not None else ()
    source_artifact = (
        None
        if configured_source_artifact in (None, False)
        else _validated_artifact_reference(configured_source_artifact)
    )
    tracker_cfg = _tracker_config(
        cfg, protocol, source_artifact
    )
    run_config = _run_config(
        cfg,
        family,
        model_name,
        protocol,
        input_shape,
        device,
        loader_metadata,
        warmup_iterations,
        repetitions,
        batch_sizes,
    )
    tracker = create_tracker(tracker_cfg, run_config)
    try:
        checkpoint, restored_artifact = _restore_checkpoint(
            cfg,
            tracker,
            model,
            output_dir,
            family,
            loader_metadata,
        )
        model_output = _invoke_model_protocol(
            model,
            inputs,
            protocol=protocol,
            warmup_iterations=warmup_iterations,
            repetitions=repetitions,
            batch_sizes=batch_sizes,
            checkpoint=checkpoint,
            benchmark_cfg=benchmark_cfg,
            efficiency_cfg=_mapping(cfg.get("efficiency", {}), "efficiency"),
        )
        model_row = _model_row(model_name, protocol, model_output)
        case_rows, case_summary = _run_case_protocol(
            cfg,
            family,
            protocol,
            model_name,
            model,
            evaluation_loader,
            device,
        )
        rows = [_redact_case_ids(row) for row in [model_row, *case_rows]]
        summary = _summary_from_model_output(model_output, protocol)
        summary.update(case_summary)
        provenance: dict[str, Any] = {
            "schema_version": 1,
            "model": model_name,
            "architecture": model_name,
            "family": family,
            "variant": _variant_for_config(cfg) if family == "metaunetr" else None,
            "protocol": protocol,
            "protocol_id": protocol,
            "input_shape": input_shape,
            "device": str(device),
            "warmup_iterations": warmup_iterations,
            "repetitions": repetitions,
            "batch_sizes": batch_sizes,
            "case_limit": case_limit,
            "source_checkpoint": str(checkpoint),
            "source_artifact": restored_artifact,
            "source_checkpoint_artifact": restored_artifact,
            "source_train_run_id": _source_value(cfg, "source_train_run_id"),
            "manifest_hash": loader_metadata.get("manifest_hash"),
            "split_counts": loader_metadata.get("split_counts"),
            "hd95_excluded_by_region": dict(
                case_summary.get("inference/hd95_excluded_by_region", {})
            ),
            "hd95_excluded_cases": case_summary.get(
                "inference/hd95_excluded_cases", 0
            ),
            **{
                f"hd95_excluded_{region}": case_summary.get(
                    f"inference/hd95_excluded_{region}", 0
                )
                for region in REGION_NAMES
            },
            "checkpoint_restoration_excluded_from_measurements": True,
            "tracker_run_id": _tracker_run_id(tracker),
        }
        result = BenchmarkResult(summary=summary, rows=rows, provenance=provenance)
        benchmark_path = serialize_benchmark(
            output_dir,
            result,
            output_name=benchmark_cfg.get("output_name"),
        )

        summary_method = getattr(tracker, "log_summary", None)
        if callable(summary_method):
            summary_method(_json_safe(_redact_case_ids(summary)))
        table_method = getattr(tracker, "log_table", None)
        if callable(table_method):
            columns, table_rows = _table_payload(rows)
            table_method("benchmark/rows", columns, table_rows)
        artifact_method = getattr(tracker, "log_artifact", None)
        if callable(artifact_method):
            artifact_method(
                "benchmark",
                {benchmark_path.name: benchmark_path},
                artifact_type="benchmark",
                aliases=("latest",),
            )
        return result
    finally:
        finish = getattr(tracker, "finish", None)
        if callable(finish):
            try:
                finish()
            except BaseException:
                # Preserve measurement/restoration/logging failures as the
                # authoritative exception, matching training lifecycle rules.
                pass


__all__ = ["run_benchmark", "select_protocol"]
