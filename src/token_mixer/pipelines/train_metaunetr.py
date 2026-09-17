"""Training boundary for the paper MetaUNETR variant experiments."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from numbers import Integral
from pathlib import Path
from typing import Any, Callable, cast

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from token_mixer.data.cases import discover_cases
from token_mixer.data.datasets import BratsPatchDataset, BratsVolumeDataset
from token_mixer.data.splits import load_split_manifest
from token_mixer.evaluation.inference import evaluate_full_volumes
from token_mixer.models.metaunetr.variants import build_metaunetr
from token_mixer.pipelines._baseline_common import (
    _copy_resume_best,
    _finalize_training_run,
    _flatten_training_config,
    _finish_tracker,
    _loader_parts,
    _limit_cases,
    _max_cases,
    _invoke_fit,
    _result_with_test_metrics,
    _restore_best,
    _write_pipeline_failure,
    build_loss as _build_shared_loss,
    resume_path,
    warm_start_path,
)
from token_mixer.reproducibility import seed_everything, seed_worker
from token_mixer.training.checkpoints import CheckpointManager
from token_mixer.training.engine import FitResult, _checkpoint_metadata, _resolve_device, fit
from token_mixer.training.phases import PhaseSpec
from token_mixer.training.tracking import create_tracker


VALID_VARIANTS = ("metaunetr_mamba", "mod_a", "mod_b")
_MISSING = object()
_MODEL_KEYS = (
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
    "execution_device",
    "norm_name",
    "norm_num_groups",
)
_MODEL_SPATIAL_DIVISOR = 32
_SPATIAL_CROP_KEYS = frozenset(
    {"patch_size", "crop_size", "roi_size", "volume_size"}
)
_TRAINING_SPATIAL_KEYS = frozenset(
    {
        "patch_size",
        "crop_size",
        "volume_size",
        "spatial_size",
        "train_patch_size",
        "train_crop_size",
        "train_spatial_size",
        "training_spatial_size",
    }
)


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


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} configuration must be a mapping")
    return value


def _variant_value(value: Any) -> str:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = list(value)
        if len(values) != 1:
            raise ValueError(
                "MetaUNETR configuration must select exactly one variant"
            )
        value = values[0]
    if not isinstance(value, str):
        raise ValueError(
            "MetaUNETR configuration must select exactly one variant"
        )
    if value not in VALID_VARIANTS:
        raise ValueError(
            f"invalid MetaUNETR variant {value!r}; choose one of {VALID_VARIANTS}"
        )
    return value


def _resolve_variant(cfg: DictConfig | Mapping[str, Any]) -> str:
    candidates: list[str] = []
    explicit_paths = (
        ("variant",),
        ("variants",),
        ("model", "variant"),
        ("experiment", "variant"),
        ("experiment", "name"),
    )
    for path in explicit_paths:
        value = _path_value(cfg, path)
        if value is _MISSING or value is None:
            continue
        if path == ("experiment", "name") and isinstance(value, str):
            if value not in VALID_VARIANTS:
                continue
        candidates.append(_variant_value(value))

    if not candidates:
        raise ValueError(
            "MetaUNETR configuration must select exactly one variant from "
            f"{VALID_VARIANTS}"
        )
    if len(set(candidates)) != 1:
        raise ValueError(
            "MetaUNETR configuration must select exactly one variant; "
            f"got {tuple(candidates)}"
        )
    return candidates[0]


def _model_config(cfg: DictConfig | Mapping[str, Any]) -> dict[str, Any]:
    configured = _first_value(
        cfg,
        (("model",), ("experiment", "model")),
        default={},
    )
    result = dict(_plain(_mapping(configured, "model")))
    result.pop("scan_direction", None)
    for key in _MODEL_KEYS:
        if key in result:
            continue
        value = _path_value(cfg, (key,))
        if value is not _MISSING:
            result[key] = _plain(value)
    return result


def _int_tuple(value: Any, name: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a sequence of integers") from exc
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _metadata(
    cfg: DictConfig | Mapping[str, Any],
    model_cfg: Mapping[str, Any],
    variant: str,
) -> dict[str, Any]:
    base_channels = int(model_cfg.get("base_channels", 48))
    depths = _int_tuple(model_cfg.get("depths", (2, 2, 2, 2)), "depths")
    base_widths = tuple(base_channels * 2**index for index in range(4))
    scan_direction = _scan_direction(cfg, model_cfg)
    axis_fusion = str(
        model_cfg.get(
            "axis_fusion",
            _first_value(cfg, (("model", "axis_fusion"), ("axis_fusion",)), default="sum"),
        )
    )
    return {
        "architecture": "MetaUNETR",
        "dimensionality": "3-D",
        "spatial_dims": 3,
        "native_3d": True,
        "slice_based": False,
        "canonical_regions": ("ET", "TC", "WT"),
        "output_kind": "raw_logits",
        "model_config": {
            str(key): _plain(value)
            for key, value in model_cfg.items()
            if key != "execution_device"
        },
        "variant": variant,
        "base_channels": base_channels,
        "base_widths": base_widths,
        "widths": base_widths,
        "depths": depths,
        "scan_direction": scan_direction,
        "axis_fusion": axis_fusion,
    }


def _scan_direction(
    cfg: DictConfig | Mapping[str, Any], model_cfg: Mapping[str, Any]
) -> str:
    configured_values = [
        _path_value(cfg, path)
        for path in (
            ("model", "scan_direction"),
            ("scan_direction",),
            ("experiment", "scan_direction"),
            ("experiment", "model", "scan_direction"),
        )
    ]
    configured_values = [
        value for value in configured_values if value is not _MISSING and value is not None
    ]
    if not configured_values and "scan_direction" in model_cfg:
        configured_values.append(model_cfg["scan_direction"])
    invalid = next(
        (value for value in configured_values if value != "forward"),
        None,
    )
    if invalid is not None:
        raise ValueError(
            "scan_direction is metadata-only; only the implemented 'forward' "
            f"value is supported, got {invalid!r}"
        )
    return "forward"


def _dataset_config(cfg: DictConfig | Mapping[str, Any]) -> dict[str, Any]:
    return dict(_plain(cfg))


def _without_spatial_crop(config: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in config.items():
        if key in _SPATIAL_CROP_KEYS:
            continue
        result[key] = _without_spatial_crop_value(value)
    return result


def _without_spatial_crop_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _without_spatial_crop(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        cleaned = [_without_spatial_crop_value(item) for item in value]
        return tuple(cleaned) if isinstance(value, tuple) else cleaned
    return value


def _loader_value(
    cfg: DictConfig | Mapping[str, Any], key: str, default: Any
) -> Any:
    return _first_value(
        cfg,
        (
            ("loader", key),
            ("data", "loader", key),
            ("experiment", "training", key),
            ("training", key),
            ("run", key),
            (key,),
        ),
        default=default,
    )


def _manifest_path(cfg: DictConfig | Mapping[str, Any]) -> Path:
    configured = _first_value(
        cfg,
        (
            ("paths", "manifest"),
            ("paths", "split_manifest"),
            ("data", "manifest"),
            ("manifest",),
        ),
    )
    if configured is None:
        raise ValueError("MetaUNETR loaders require paths.data_root and paths.manifest")
    return Path(configured)


def _configured_split_metadata(cfg: DictConfig | Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dataset_id": _first_value(
            cfg,
            (
                ("dataset_id",),
                ("data", "dataset_id"),
                ("dataset", "dataset_id"),
                ("dataset", "id"),
                ("split", "dataset_id"),
            ),
            default=_MISSING,
        ),
        "split_seed": _first_value(
            cfg,
            (
                ("split_seed",),
                ("data", "split_seed"),
                ("dataset", "split_seed"),
                ("reproducibility", "split_seed"),
                ("split", "seed"),
                ("data", "split", "seed"),
                ("dataset", "split", "seed"),
                ("run", "split_seed"),
                ("seed",),
            ),
            default=_MISSING,
        ),
        "val_fraction": _first_value(
            cfg,
            (
                ("val_fraction",),
                ("validation_fraction",),
                ("data", "val_fraction"),
                ("data", "validation_fraction"),
                ("dataset", "val_fraction"),
                ("split", "val_fraction"),
                ("split", "validation_fraction"),
                ("data", "split", "val_fraction"),
                ("dataset", "split", "val_fraction"),
            ),
            default=_MISSING,
        ),
        "test_fraction": _first_value(
            cfg,
            (
                ("test_fraction",),
                ("data", "test_fraction"),
                ("dataset", "test_fraction"),
                ("split", "test_fraction"),
                ("data", "split", "test_fraction"),
                ("dataset", "split", "test_fraction"),
            ),
            default=_MISSING,
        ),
    }


def _configured_fraction(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"configured {name} must be a finite number between 0 and 1")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"configured {name} must be a finite number between 0 and 1"
        ) from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"configured {name} must be a finite number between 0 and 1")
    return result


def _validate_manifest_configuration(
    cfg: DictConfig | Mapping[str, Any], manifest: Any
) -> None:
    configured = _configured_split_metadata(cfg)
    comparisons = {
        "dataset_id": manifest.dataset_id,
        "split_seed": manifest.seed,
        "val_fraction": manifest.val_fraction,
        "test_fraction": manifest.test_fraction,
    }
    for name, actual in comparisons.items():
        expected = configured[name]
        if expected is _MISSING:
            continue
        if name == "dataset_id":
            if not isinstance(expected, str):
                raise ValueError("configured dataset_id must be a string")
        elif name == "split_seed":
            if not isinstance(expected, int) or isinstance(expected, bool):
                raise ValueError("configured split_seed must be an integer")
        else:
            expected = _configured_fraction(expected, name)
        if expected != actual:
            raise ValueError(
                f"configured {name} {expected!r} does not match split manifest "
                f"{actual!r}"
            )


def _validate_manifest_ids(manifest: Any) -> None:
    split_ids = [
        (name, list(getattr(manifest, name))) for name in ("train", "val", "test")
    ]
    all_ids: list[str] = []
    for name, case_ids in split_ids:
        if len(case_ids) != len(set(case_ids)):
            raise ValueError(f"split manifest {name} cases contain duplicate IDs")
        all_ids.extend(case_ids)
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("split manifest contains duplicate IDs across splits")


def _manifest_metadata(path: Path, manifest: Any) -> dict[str, Any]:
    try:
        manifest_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ValueError(f"Cannot hash split manifest '{path}': {exc}") from exc
    return {
        "manifest_path": str(path),
        "manifest_hash": manifest_hash,
        "dataset_id": manifest.dataset_id,
        "split_seed": manifest.seed,
        "manifest_seed": manifest.seed,
        "val_fraction": manifest.val_fraction,
        "test_fraction": manifest.test_fraction,
        "split_counts": {
            "train": len(manifest.train),
            "val": len(manifest.val),
            "test": len(manifest.test),
        },
    }


def _split_cases(cases: Sequence[Any], case_ids: Sequence[str], split: str) -> list[Any]:
    requested_ids = list(case_ids)
    if len(requested_ids) != len(set(requested_ids)):
        raise ValueError(f"split manifest {split} cases contain duplicate IDs")
    case_ids_in_data = [case.case_id for case in cases]
    if len(case_ids_in_data) != len(set(case_ids_in_data)):
        raise ValueError("data contains duplicate case IDs")
    by_id = {case.case_id: case for case in cases}
    missing = [case_id for case_id in requested_ids if case_id not in by_id]
    if missing:
        raise ValueError(f"split manifest {split} cases are missing from data: {missing}")
    return [by_id[case_id] for case_id in requested_ids]


class _LoaderBundle(tuple):
    metadata: dict[str, Any]
    test_loader: DataLoader[Any] | None

    def __new__(
        cls,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        metadata: Mapping[str, Any],
        test_loader: DataLoader[Any] | None = None,
    ) -> "_LoaderBundle":
        result = super().__new__(cls, (train_loader, val_loader))
        result.metadata = dict(metadata)
        result.test_loader = test_loader
        return result


def build_loaders(
    cfg: DictConfig | Mapping[str, Any],
    generator: torch.Generator,
) -> _LoaderBundle:
    """Build canonical patch-training and full-volume validation loaders."""
    data_root = _first_value(
        cfg,
        (("paths", "data_root"), ("data", "data_root"), ("data_root",)),
    )
    if data_root is None:
        raise ValueError("MetaUNETR loaders require paths.data_root and paths.manifest")

    manifest_path = _manifest_path(cfg)
    manifest = load_split_manifest(manifest_path)
    _validate_manifest_configuration(cfg, manifest)
    _validate_manifest_ids(manifest)
    cases = discover_cases(Path(data_root))
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("data contains duplicate case IDs")
    train_cases = _split_cases(cases, manifest.train, "train")
    val_cases = _split_cases(cases, manifest.val, "val")
    test_cases = _split_cases(cases, manifest.test, "test")
    train_cases = _limit_cases(train_cases, cfg)
    val_cases = _limit_cases(val_cases, cfg)
    test_cases = _limit_cases(test_cases, cfg)
    dataset_config = _dataset_config(cfg)

    train_dataset = BratsPatchDataset(train_cases, dataset_config, training=True)
    val_dataset = BratsVolumeDataset(
        val_cases,
        _without_spatial_crop(dataset_config),
    )
    test_dataset = BratsVolumeDataset(
        test_cases,
        _without_spatial_crop(dataset_config),
    )

    batch_size = int(_loader_value(cfg, "batch_size", 1))
    num_workers = int(_loader_value(cfg, "num_workers", 0))
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers cannot be negative")
    pin_memory = bool(_loader_value(cfg, "pin_memory", False))
    persistent_workers = bool(_loader_value(cfg, "persistent_workers", False))
    drop_last = bool(_loader_value(cfg, "drop_last", False))
    common: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "generator": generator,
    }
    if num_workers:
        common["worker_init_fn"] = seed_worker
        common["persistent_workers"] = persistent_workers
        prefetch_factor = _loader_value(cfg, "prefetch_factor", None)
        if prefetch_factor is not None:
            common["prefetch_factor"] = int(prefetch_factor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        **common,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        **common,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        **common,
    )
    metadata = _manifest_metadata(manifest_path, manifest)
    metadata["split_counts"] = {
        "train": len(train_cases),
        "val": len(val_cases),
        "test": len(test_cases),
    }
    metadata["max_cases"] = _max_cases(cfg)
    return _LoaderBundle(train_loader, val_loader, metadata, test_loader)


def _build_phases(cfg: DictConfig | Mapping[str, Any]) -> list[PhaseSpec]:
    raw_phases = _first_value(
        cfg,
        (
            ("experiment", "training", "phases"),
            ("phases",),
            ("training", "phases"),
            ("experiment", "phases"),
        ),
    )
    if raw_phases is None:
        learning_rate = float(
            _first_value(
                cfg,
                (
                    ("experiment", "training", "learning_rate"),
                    ("experiment", "training", "lr"),
                    ("learning_rate",),
                    ("lr",),
                    ("training", "learning_rate"),
                    ("training", "lr"),
                ),
                default=1e-4,
            )
        )
        raw_phases = [
            {
                "name": "train",
                "epochs": _first_value(
                    cfg,
                    (
                        ("experiment", "training", "epochs"),
                        ("epochs",),
                        ("run", "epochs"),
                    ),
                    default=1,
                ),
                "freeze_encoder": False,
                "encoder_lr": learning_rate,
                "decoder_lr": learning_rate,
            }
        ]

    if isinstance(raw_phases, Mapping):
        if "name" in raw_phases or "epochs" in raw_phases:
            raw_phases = [raw_phases]
        else:
            raw_phases = [
                {"name": name, **dict(spec)}
                for name, spec in raw_phases.items()
                if isinstance(spec, Mapping)
            ]
    if not isinstance(raw_phases, Sequence) or isinstance(raw_phases, (str, bytes)):
        raise TypeError("phases configuration must be a sequence of mappings")

    phases: list[PhaseSpec] = []
    for index, raw_phase in enumerate(raw_phases, start=1):
        phase = _mapping(raw_phase, f"phase {index}")
        decoder_lr = float(phase.get("decoder_lr", phase.get("lr", 1e-4)))
        phases.append(
            PhaseSpec(
                name=str(phase.get("name", f"phase{index}")),
                epochs=int(phase.get("epochs", 0)),
                freeze_encoder=bool(phase.get("freeze_encoder", False)),
                encoder_lr=float(phase.get("encoder_lr", decoder_lr)),
                decoder_lr=decoder_lr,
            )
        )
    return phases


def _build_loss(
    cfg: DictConfig | Mapping[str, Any],
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    return _build_shared_loss(cfg)


def _size3(value: Any, name: str) -> tuple[int, int, int]:
    try:
        if isinstance(value, Integral) and not isinstance(value, bool):
            result = (int(value),) * 3
        else:
            if isinstance(value, (str, bytes, bytearray)):
                raise TypeError
            values = tuple(value)
            if len(values) != 3 or any(
                not isinstance(item, Integral) or isinstance(item, bool)
                for item in values
            ):
                raise TypeError
            result = tuple(int(item) for item in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain three positive integers") from exc
    if any(item <= 0 for item in result):
        raise ValueError(f"{name} must contain three positive integers")
    return (result[0], result[1], result[2])


def _model_spatial_size(value: Any, name: str) -> tuple[int, int, int]:
    result = _size3(value, name)
    if any(size % _MODEL_SPATIAL_DIVISOR for size in result):
        raise ValueError(
            f"{name} {result} must be divisible by model divisor "
            f"{_MODEL_SPATIAL_DIVISOR}"
        )
    return result


def _iter_mappings(value: Any):
    if isinstance(value, Mapping):
        yield value
        for nested in value.values():
            yield from _iter_mappings(nested)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for nested in value:
            yield from _iter_mappings(nested)


def _validate_training_spatial_sizes(cfg: DictConfig | Mapping[str, Any]) -> None:
    for mapping in _iter_mappings(_plain(cfg)):
        for key, value in mapping.items():
            if str(key) in _TRAINING_SPATIAL_KEYS and value is not None:
                _model_spatial_size(value, str(key))


def _spacing3(value: Any) -> tuple[float, float, float]:
    try:
        if isinstance(value, (str, bytes, bytearray)):
            raise TypeError
        spacing = tuple(float(part) for part in value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("spacing must contain three positive finite values") from exc
    if len(spacing) != 3 or any(
        isinstance(part, bool) or not math.isfinite(part) or part <= 0
        for part in spacing
    ):
        raise ValueError("spacing must contain three positive finite values")
    return spacing


def _validation_spacing(cfg: DictConfig | Mapping[str, Any]) -> tuple[float, float, float]:
    configured = _first_value(
        cfg,
        (
            ("spacing",),
            ("data", "spacing"),
            ("dataset", "spacing"),
            ("preprocessing", "spacing"),
            ("transform", "spacing"),
            ("evaluation", "spacing"),
            ("inference", "spacing"),
            ("default_spacing",),
            ("evaluation", "default_spacing"),
            ("inference", "default_spacing"),
        ),
        default=_MISSING,
    )
    if configured is _MISSING:
        raise ValueError(
            "MetaUNETR validation requires explicit positive finite spacing; "
            "unit spacing is not inferred"
        )
    return _spacing3(configured)


def _effective_device(cfg: DictConfig | Mapping[str, Any]) -> torch.device:
    requested = _first_value(
        cfg,
        (
            ("device",),
            ("experiment", "training", "device"),
            ("training", "device"),
            ("run", "device"),
        ),
        default="auto",
    )
    return _resolve_device(requested)


def _build_evaluator(
    cfg: DictConfig | Mapping[str, Any],
    device: torch.device | str | None = None,
) -> Callable[[torch.nn.Module, Any], Mapping[str, Any]]:
    spacing = _validation_spacing(cfg)
    roi_size = _model_spatial_size(
        _first_value(
            cfg,
            (
                ("inference", "roi_size"),
                ("evaluation", "roi_size"),
                ("roi_size",),
                ("data", "patch_size"),
            ),
            default=(96, 96, 96),
        ),
        "roi_size",
    )
    try:
        sw_batch_size = int(
            _first_value(
                cfg,
                (
                    ("inference", "sw_batch_size"),
                    ("evaluation", "sw_batch_size"),
                    ("sw_batch_size",),
                ),
                default=1,
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("sw_batch_size must be at least one") from exc
    if sw_batch_size < 1:
        raise ValueError("sw_batch_size must be at least one")

    try:
        overlap = float(
            _first_value(
                cfg,
                (("inference", "overlap"), ("evaluation", "overlap"), ("overlap",)),
                default=0.25,
            )
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("overlap must satisfy 0 <= overlap < 1") from exc
    if not math.isfinite(overlap) or not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must satisfy 0 <= overlap < 1")

    evaluation_device = _effective_device(cfg) if device is None else device
    configured = _first_value(
        cfg,
        (("evaluator",), ("evaluation", "evaluator")),
    )
    if callable(configured):
        return cast(Callable[[torch.nn.Module, Any], Mapping[str, Any]], configured)

    def evaluator(model: torch.nn.Module, loader: Any) -> Mapping[str, Any]:
        return evaluate_full_volumes(
            model,
            loader,
            roi_size,
            sw_batch_size,
            overlap,
            evaluation_device,
            default_spacing=spacing,
        )

    return evaluator


def _build_checkpoints(
    cfg: DictConfig | Mapping[str, Any],
) -> CheckpointManager | None:
    configured = _first_value(cfg, (("checkpoints",),))
    if configured is False:
        return None
    checkpoint_cfg = _mapping(configured, "checkpoints")
    if checkpoint_cfg and not bool(checkpoint_cfg.get("enabled", True)):
        return None

    root = _first_value(
        cfg,
        (
            ("checkpoints", "root"),
            ("checkpoints", "directory"),
            ("paths", "checkpoint_dir"),
            ("paths", "checkpoint_root"),
            ("checkpoint_dir",),
        ),
    )
    if root is None:
        output_root = _first_value(cfg, (("paths", "output_root"), ("output_root",)))
        if output_root is not None:
            root = Path(output_root) / "checkpoints"
    return None if root is None else CheckpointManager(Path(root))


def _engine_config(
    cfg: DictConfig | Mapping[str, Any], metadata: Mapping[str, Any]
) -> dict[str, Any]:
    result = _flatten_training_config(cfg)
    result.update(metadata)
    result.setdefault("monitor", "mean_dice")
    result.setdefault("maximize", True)
    result.setdefault("weight_decay", 0.0)
    result.setdefault("use_amp", False)
    return result


def run_metaunetr(cfg: DictConfig | Mapping[str, Any]) -> FitResult:
    """Build configured paper variant and delegate training to the shared engine."""
    variant = _resolve_variant(cfg)
    model_cfg = _model_config(cfg)
    scan_direction = _scan_direction(cfg, model_cfg)
    effective_device = _effective_device(cfg)
    spacing = _validation_spacing(cfg)
    _validate_training_spatial_sizes(cfg)

    loss_fn = _build_loss(cfg)
    evaluator = _build_evaluator(cfg, effective_device)
    phases = _build_phases(cfg)
    checkpoints = _build_checkpoints(cfg)
    resume = resume_path(cfg, checkpoints)
    warm_start = warm_start_path(cfg, checkpoints)
    if resume is not None and warm_start is not None:
        raise ValueError("resume and warm_start are mutually exclusive")

    seed = int(
        _first_value(
            cfg,
            (("seed",), ("reproducibility", "seed"), ("run", "seed")),
            default=0,
        )
    )
    deterministic = bool(
        _first_value(
            cfg,
            (("deterministic",), ("reproducibility", "deterministic")),
            default=True,
        )
    )
    generator = seed_everything(seed, deterministic=deterministic)

    model_cfg["execution_device"] = effective_device
    model = build_metaunetr(model_cfg, variant)
    loader_result = build_loaders(cfg, generator)
    train_loader, val_loader, test_loader, loader_metadata = _loader_parts(loader_result)
    if not isinstance(loader_metadata, Mapping):
        raise TypeError("loader metadata must be a mapping")
    if not loader_metadata and _first_value(
        cfg,
        (
            ("paths", "manifest"),
            ("paths", "split_manifest"),
            ("data", "manifest"),
            ("manifest",),
        ),
        default=_MISSING,
    ) is not _MISSING:
        raise ValueError(
            "configured split manifest requires loader metadata from build_loaders; "
            "manifest reload fallback is not supported"
        )

    metadata = _metadata(cfg, model_cfg, variant)
    metadata.update(loader_metadata)
    metadata["spacing"] = spacing
    metadata["execution_device"] = str(effective_device)
    metadata["scan_direction"] = scan_direction
    if resume is not None:
        metadata["source_checkpoint"] = str(resume)
        metadata["resume_mode"] = "exact"
    elif warm_start is not None:
        metadata["source_checkpoint"] = str(warm_start)
        metadata["resume_mode"] = "warm_start"
    run_config = _engine_config(cfg, metadata)
    run_config["device"] = str(effective_device)

    if resume is not None:
        if checkpoints is None:
            raise ValueError("resume and warm_start require a checkpoint manager")
        _copy_resume_best(checkpoints, resume)
    tracking_config = _plain(_first_value(cfg, (("tracking",),), default={}))
    tracker = create_tracker(_mapping(tracking_config, "tracking"), run_config)
    try:
        result = _invoke_fit(
            fit,
            (
                model,
                train_loader,
                val_loader,
                loss_fn,
                evaluator,
                phases,
                run_config,
                tracker,
                checkpoints,
            ),
            resume=resume,
            warm_start=warm_start,
            loader_generator=generator,
        )
        if not isinstance(result, FitResult):
            raise TypeError("shared training engine must return FitResult")
        # Preserve legacy direct fixtures that intentionally omit both a test
        # loader and checkpoint/output configuration.  Real runs always have
        # checkpointing enabled and use validation as their final split when
        # no held-out test loader is supplied.
        if test_loader is None and checkpoints is None:
            return result
        if checkpoints is None:
            raise ValueError("MetaUNETR final evaluation requires enabled checkpointing")

        _restore_best(
            model,
            checkpoints,
            expected_metadata=_checkpoint_metadata(run_config, phases),
        )
        if test_loader is None:
            evaluation_loader = val_loader
            test_evaluator = _first_value(
                cfg,
                (("validation_evaluator",), ("evaluation", "validation_evaluator")),
            )
            if not callable(test_evaluator):
                test_evaluator = evaluator
            protocol = "native_3d_validation"
            evaluation_split = "validation"
        else:
            evaluation_loader = test_loader
            test_evaluator = _first_value(
                cfg,
                (("test_evaluator",), ("evaluation", "test_evaluator")),
            )
            if not callable(test_evaluator):
                test_evaluator = evaluator
            protocol = "native_3d_full_volume"
            evaluation_split = "test"
        test_metrics = test_evaluator(model, evaluation_loader)
        if not isinstance(test_metrics, Mapping):
            raise TypeError("test evaluator must return a mapping of scalar metrics")
        final_metadata = dict(metadata)
        final_metadata.update(
            {"protocol": protocol, "evaluation_split": evaluation_split}
        )
        result = _result_with_test_metrics(result, test_metrics, final_metadata)
        _finalize_training_run(
            cfg,
            tracker,
            result,
            checkpoints,
            protocol=protocol,
        )
        return result
    except BaseException as error:
        _write_pipeline_failure(cfg, error, metadata)
        raise
    finally:
        _finish_tracker(tracker)


__all__ = ["VALID_VARIANTS", "build_loaders", "run_metaunetr"]
