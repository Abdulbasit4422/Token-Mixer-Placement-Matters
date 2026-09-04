"""Shared orchestration and canonical adapters for baseline pipelines."""

from __future__ import annotations

import hashlib
import inspect
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from numbers import Integral, Number
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from token_mixer.data.cases import CaseRecord, discover_cases
from token_mixer.data.datasets import (
    BratsPatchDataset,
    BratsSliceDataset,
    BratsVolumeDataset,
)
from token_mixer.data.labels import REGION_NAMES, multiclass_to_regions
from token_mixer.data.splits import load_split_manifest
from token_mixer.evaluation.inference import evaluate_full_volumes
from token_mixer.evaluation.metrics import (
    dice_by_region,
    hd95_by_region,
    hd95_excluded_by_region,
    logits_to_regions,
)
from token_mixer.reproducibility import seed_everything, seed_worker
from token_mixer.training.checkpoints import CheckpointManager
from token_mixer.training.engine import (
    FitResult,
    _checkpoint_metadata,
    _resolve_device,
    fit,
)
from token_mixer.training.phases import PhaseSpec
from token_mixer.training.tracking import Tracker, create_tracker


_MISSING = object()
_SPATIAL_CROP_KEYS = frozenset({"patch_size", "crop_size", "roi_size", "volume_size"})


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
    config: Any, paths: Sequence[Sequence[str]], default: Any = None
) -> Any:
    for path in paths:
        value = _path_value(config, path)
        if value is not _MISSING and value is not None:
            return value
    return default


def _first_configured(
    config: Any, paths: Sequence[Sequence[str]], default: Any = _MISSING
) -> Any:
    for path in paths:
        value = _path_value(config, path)
        if value is not _MISSING:
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


def _flatten_training_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    result = _plain(cfg)
    if not isinstance(result, dict):
        raise TypeError("training configuration must be a mapping")
    for section_name in ("training", "engine"):
        section = result.get(section_name)
        if isinstance(section, Mapping):
            for key, value in section.items():
                result.setdefault(str(key), value)
    experiment = result.get("experiment")
    nested_training = experiment.get("training") if isinstance(experiment, Mapping) else None
    if isinstance(nested_training, Mapping):
        result.update({str(key): value for key, value in nested_training.items()})
    return result


def _max_cases(cfg: Mapping[str, Any]) -> int | None:
    configured = _first_configured(
        cfg,
        (
            ("run", "max_cases"),
            ("experiment", "training", "max_cases"),
            ("max_cases",),
            ("training", "max_cases"),
        ),
    )
    if configured is _MISSING or configured is None:
        return None
    if isinstance(configured, bool) or not isinstance(configured, Integral):
        raise ValueError("max_cases must be a positive integer or null")
    result = int(configured)
    if result < 1:
        raise ValueError("max_cases must be a positive integer or null")
    return result


def _limit_cases(cases: Sequence[Any], cfg: Mapping[str, Any]) -> list[Any]:
    limit = _max_cases(cfg)
    values = list(cases)
    return values if limit is None else values[:limit]


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None or value is _MISSING:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} configuration must be a mapping")
    return value


def _manifest_path(cfg: Mapping[str, Any]) -> Path:
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
        raise ValueError("baseline loaders require paths.data_root and paths.manifest")
    return Path(configured)


def _data_root(cfg: Mapping[str, Any]) -> Path:
    configured = _first_value(
        cfg,
        (("paths", "data_root"), ("data", "data_root"), ("data_root",)),
    )
    if configured is None:
        raise ValueError("baseline loaders require paths.data_root and paths.manifest")
    return Path(configured)


def _configured_split_metadata(cfg: Mapping[str, Any]) -> dict[str, Any]:
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


def _validate_manifest_configuration(cfg: Mapping[str, Any], manifest: Any) -> None:
    configured = _configured_split_metadata(cfg)
    actual = {
        "dataset_id": manifest.dataset_id,
        "split_seed": manifest.seed,
        "val_fraction": manifest.val_fraction,
        "test_fraction": manifest.test_fraction,
    }
    for name, actual_value in actual.items():
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
        if expected != actual_value:
            raise ValueError(
                f"configured {name} {expected!r} does not match split manifest "
                f"{actual_value!r}"
            )


def _validate_manifest_ids(manifest: Any) -> None:
    all_ids: list[str] = []
    for split in ("train", "val", "test"):
        case_ids = list(getattr(manifest, split))
        if len(case_ids) != len(set(case_ids)):
            raise ValueError(f"split manifest {split} cases contain duplicate IDs")
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


def _split_cases(cases: Sequence[CaseRecord], case_ids: Sequence[str], split: str):
    requested_ids = list(case_ids)
    by_id = {case.case_id: case for case in cases}
    missing = [case_id for case_id in requested_ids if case_id not in by_id]
    if missing:
        raise ValueError(f"split manifest {split} cases are missing from data: {missing}")
    return [by_id[case_id] for case_id in requested_ids]


class LoaderBundle(tuple):
    """Tuple-compatible train/validation/test loaders with manifest metadata."""

    metadata: dict[str, Any]

    def __new__(
        cls,
        train_loader: Any,
        val_loader: Any,
        test_loader: Any,
        metadata: Mapping[str, Any],
    ) -> "LoaderBundle":
        result = super().__new__(cls, (train_loader, val_loader, test_loader))
        result.metadata = dict(metadata)
        return result


def _loader_value(cfg: Mapping[str, Any], key: str, default: Any) -> Any:
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


def _loader_kwargs(
    cfg: Mapping[str, Any], generator: torch.Generator
) -> tuple[int, dict[str, Any]]:
    batch_size = int(_loader_value(cfg, "batch_size", 1))
    num_workers = int(_loader_value(cfg, "num_workers", 0))
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers cannot be negative")
    common: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": bool(_loader_value(cfg, "pin_memory", False)),
        "generator": generator,
    }
    if num_workers:
        common["worker_init_fn"] = seed_worker
        common["persistent_workers"] = bool(
            _loader_value(cfg, "persistent_workers", False)
        )
        prefetch_factor = _loader_value(cfg, "prefetch_factor", None)
        if prefetch_factor is not None:
            common["prefetch_factor"] = int(prefetch_factor)
    return batch_size, common


def _without_spatial_crop(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _without_spatial_crop(item)
            for key, item in value.items()
            if str(key) not in _SPATIAL_CROP_KEYS
        }
    if isinstance(value, tuple):
        return tuple(_without_spatial_crop(item) for item in value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_without_spatial_crop(item) for item in value]
    return value


def _canonical_case_loaders(
    cfg: Mapping[str, Any], generator: torch.Generator
) -> tuple[list[CaseRecord], dict[str, Any]]:
    manifest_path = _manifest_path(cfg)
    manifest = load_split_manifest(manifest_path)
    _validate_manifest_configuration(cfg, manifest)
    _validate_manifest_ids(manifest)

    cases = discover_cases(_data_root(cfg))
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("data contains duplicate case IDs")
    train_cases = _split_cases(cases, manifest.train, "train")
    val_cases = _split_cases(cases, manifest.val, "val")
    test_cases = _split_cases(cases, manifest.test, "test")
    max_cases = _max_cases(cfg)
    if max_cases is not None:
        train_cases = train_cases[:max_cases]
        val_cases = val_cases[:max_cases]
        test_cases = test_cases[:max_cases]
    del generator
    metadata = _manifest_metadata(manifest_path, manifest)
    metadata["split_counts"] = {
        "train": len(train_cases),
        "val": len(val_cases),
        "test": len(test_cases),
    }
    metadata["max_cases"] = max_cases
    return train_cases, {
        "val_cases": val_cases,
        "test_cases": test_cases,
        "metadata": metadata,
    }


def build_volume_loaders(
    cfg: Mapping[str, Any], generator: torch.Generator
) -> LoaderBundle:
    """Build canonical patch-training and full-volume val/test loaders."""
    train_cases, split = _canonical_case_loaders(cfg, generator)
    dataset_config = _plain(cfg)
    if not isinstance(dataset_config, Mapping):
        raise TypeError("baseline configuration must be a mapping")
    val_config = _without_spatial_crop(dataset_config)
    batch_size, common = _loader_kwargs(cfg, generator)
    train_dataset = BratsPatchDataset(train_cases, dataset_config, training=True)
    val_dataset = BratsVolumeDataset(split["val_cases"], val_config)
    test_dataset = BratsVolumeDataset(split["test_cases"], val_config)
    drop_last = bool(_loader_value(cfg, "drop_last", False))
    return LoaderBundle(
        DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=drop_last,
            **common,
        ),
        DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            **common,
        ),
        DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            **common,
        ),
        split["metadata"],
    )


def _batch_image_target(batch: Any) -> tuple[Any, Any]:
    if isinstance(batch, Mapping):
        image = next(
            (batch[key] for key in ("image", "images", "input", "inputs", "x") if key in batch),
            None,
        )
        target = next(
            (batch[key] for key in ("label", "labels", "target", "targets", "y") if key in batch),
            None,
        )
        if image is None or target is None:
            raise KeyError("batch must contain image and label/target values")
        return image, target
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise TypeError("loader must yield a mapping or (image, target, ...) tuple")


def _as_slice_region_target(target: Any, batch_size: int) -> torch.Tensor:
    tensor = target if isinstance(target, torch.Tensor) else torch.as_tensor(target)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim == 4 and tensor.shape[1] == len(REGION_NAMES):
        regions = tensor.float()
    elif tensor.ndim == 3 and tensor.shape[0] == len(REGION_NAMES) and batch_size == 1:
        regions = tensor.unsqueeze(0).float()
    elif tensor.ndim == 3 and tensor.shape[0] == batch_size:
        if not torch.isfinite(tensor.float()).all().item():
            raise ValueError("slice targets must contain only finite values")
        if (tensor != tensor.round()).any().item() or tensor.min().item() < 0 or tensor.max().item() > 3:
            raise ValueError("slice multiclass targets must contain canonical labels 0 through 3")
        converted = []
        for item in tensor:
            label = item.detach().cpu().numpy()[None, ...]
            converted.append(multiclass_to_regions(label)[:, 0])
        regions = torch.from_numpy(np.stack(converted, axis=0)).float()
    else:
        raise ValueError(
            "slice targets must have shape [B, H, W] or [B, 3, H, W], "
            f"got {tuple(tensor.shape)}"
        )
    if regions.shape[0] != batch_size:
        raise ValueError("slice target and image batch sizes must match")
    if not torch.isfinite(regions).all().item():
        raise ValueError("slice region targets must contain only finite values")
    return regions


class _CanonicalSliceLoader:
    """Convert canonical multiclass slice labels before engine consumption."""

    def __init__(self, loader: Any):
        self.loader = loader

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self):
        for batch in self.loader:
            image, target = _batch_image_target(batch)
            image_tensor = image if isinstance(image, torch.Tensor) else torch.as_tensor(image)
            if image_tensor.ndim == 3:
                image_tensor = image_tensor.unsqueeze(0)
            canonical_target = _as_slice_region_target(target, int(image_tensor.shape[0]))
            if isinstance(batch, Mapping):
                result = dict(batch)
                for key in ("label", "labels", "target", "targets", "y"):
                    if key in result:
                        result[key] = canonical_target
                        break
                yield result
            else:
                result = list(batch)
                result[1] = canonical_target
                yield tuple(result) if isinstance(batch, tuple) else result

    def __getattr__(self, name: str) -> Any:
        return getattr(self.loader, name)


def build_slice_loaders(
    cfg: Mapping[str, Any], generator: torch.Generator
) -> LoaderBundle:
    """Build canonical manifest-backed slice loaders for TransUNet."""
    train_cases, split = _canonical_case_loaders(cfg, generator)
    dataset_config = _plain(cfg)
    if not isinstance(dataset_config, Mapping):
        raise TypeError("baseline configuration must be a mapping")
    train_dataset = BratsSliceDataset(train_cases, dataset_config, training=True)
    val_dataset = BratsSliceDataset(split["val_cases"], dataset_config, training=False)
    test_dataset = BratsSliceDataset(split["test_cases"], dataset_config, training=False)
    batch_size, common = _loader_kwargs(cfg, generator)
    validation_batch_size = int(_loader_value(cfg, "validation_batch_size", 1))
    if validation_batch_size < 1:
        raise ValueError("validation_batch_size must be positive")
    drop_last = bool(_loader_value(cfg, "drop_last", False))
    return LoaderBundle(
        _CanonicalSliceLoader(
            DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=drop_last,
                **common,
            )
        ),
        _CanonicalSliceLoader(
            DataLoader(
                val_dataset,
                batch_size=validation_batch_size,
                shuffle=False,
                drop_last=False,
                **common,
            )
        ),
        _CanonicalSliceLoader(
            DataLoader(
                test_dataset,
                batch_size=validation_batch_size,
                shuffle=False,
                drop_last=False,
                **common,
            )
        ),
        split["metadata"],
    )


def _size3(value: Any, name: str) -> tuple[int, int, int]:
    if isinstance(value, Integral) and not isinstance(value, bool):
        values = (int(value),) * 3
    else:
        if isinstance(value, (str, bytes, bytearray)):
            raise ValueError(f"{name} must contain three positive integers")
        try:
            values = tuple(int(item) for item in value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain three positive integers") from exc
    if len(values) != 3 or any(item <= 0 for item in values):
        raise ValueError(f"{name} must contain three positive integers")
    return values  # type: ignore[return-value]


def _spacing3(value: Any) -> tuple[float, float, float]:
    try:
        spacing = tuple(float(part) for part in value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("spacing must contain three positive finite values") from exc
    if len(spacing) != 3 or any(not math.isfinite(part) or part <= 0 for part in spacing):
        raise ValueError("spacing must contain three positive finite values")
    return spacing


def effective_device(cfg: Mapping[str, Any]) -> torch.device:
    requested = _first_value(
        cfg,
        (("device",), ("training", "device"), ("run", "device")),
        default="auto",
    )
    return _resolve_device(requested)


def explicit_spacing(cfg: Mapping[str, Any]) -> tuple[float, float, float]:
    configured = _first_value(
        cfg,
        (
            ("spacing",),
            ("data", "spacing"),
            ("dataset", "spacing"),
            ("preprocessing", "spacing"),
            ("evaluation", "spacing"),
            ("inference", "spacing"),
        ),
        default=_MISSING,
    )
    if configured is _MISSING:
        raise ValueError(
            "3-D baseline validation requires explicit positive finite spacing; "
            "unit spacing is not inferred"
        )
    return _spacing3(configured)


def _volume_roi_size(cfg: Mapping[str, Any]) -> tuple[int, int, int]:
    value = _first_value(
        cfg,
        (("inference", "roi_size"), ("evaluation", "roi_size"), ("roi_size",), ("data", "patch_size")),
        default=(96, 96, 96),
    )
    return _size3(value, "roi_size")


def _volume_overlap(cfg: Mapping[str, Any]) -> float:
    value = _first_value(
        cfg,
        (("inference", "overlap"), ("evaluation", "overlap"), ("overlap",)),
        default=0.25,
    )
    try:
        overlap = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("overlap must satisfy 0 <= overlap < 1") from exc
    if not math.isfinite(overlap) or not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must satisfy 0 <= overlap < 1")
    return overlap


def build_volume_evaluator(
    cfg: Mapping[str, Any], device: torch.device | str | None = None
) -> Callable[[nn.Module, Iterable[Any]], Mapping[str, Any]]:
    configured = _first_value(
        cfg,
        (("validation_evaluator",), ("evaluator",), ("evaluation", "evaluator")),
    )
    if callable(configured):
        return configured
    spacing = explicit_spacing(cfg)
    roi_size = _volume_roi_size(cfg)
    sw_batch_size = int(
        _first_value(
            cfg,
            (("inference", "sw_batch_size"), ("evaluation", "sw_batch_size"), ("sw_batch_size",)),
            default=1,
        )
    )
    if sw_batch_size < 1:
        raise ValueError("sw_batch_size must be at least one")
    overlap = _volume_overlap(cfg)
    evaluation_device = effective_device(cfg) if device is None else device

    def evaluator(model: nn.Module, loader: Iterable[Any]) -> Mapping[str, Any]:
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


def _slice_metrics(
    prediction: torch.Tensor, target: torch.Tensor
) -> dict[str, float]:
    prediction = prediction.unsqueeze(2)
    target = target.unsqueeze(2)
    dice = dice_by_region(prediction, target)
    hd95 = hd95_by_region(prediction, target, spacing=(1.0, 1.0, 1.0))
    excluded = hd95_excluded_by_region(prediction, target)
    result: dict[str, float] = {}
    for region in REGION_NAMES:
        result[f"{region}_dice"] = dice[region]
        result[f"dice_{region}"] = dice[region]
        result[f"{region}_hd95"] = hd95[region]
        result[f"hd95_{region}"] = hd95[region]
        result[f"hd95_excluded_{region}"] = float(excluded[region])
    result["mean_dice"] = float(np.mean([dice[region] for region in REGION_NAMES]))
    finite_hd95 = [hd95[region] for region in REGION_NAMES]
    finite_hd95 = [value for value in finite_hd95 if math.isfinite(value)]
    result["mean_hd95"] = float(np.mean(finite_hd95)) if finite_hd95 else float("nan")
    return result


@torch.no_grad()
def evaluate_slices(model: nn.Module, loader: Iterable[Any]) -> dict[str, float]:
    """Evaluate 2-D raw region logits with the canonical region metrics."""
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    states = [module.training for module in model.modules()]
    model.eval()
    values: dict[str, list[float]] = {
        key: []
        for region in REGION_NAMES
        for key in (f"{region}_dice", f"{region}_hd95")
    }
    excluded = {region: 0 for region in REGION_NAMES}
    count = 0
    try:
        for batch in loader:
            image, target = _batch_image_target(batch)
            image_tensor = image if isinstance(image, torch.Tensor) else torch.as_tensor(image)
            if image_tensor.ndim == 3:
                image_tensor = image_tensor.unsqueeze(0)
            image_tensor = image_tensor.to(device)
            target_regions = _as_slice_region_target(target, int(image_tensor.shape[0]))
            logits = model(image_tensor)
            prediction = logits_to_regions(logits)
            if prediction.shape != target_regions.shape:
                raise ValueError(
                    "slice prediction and target shapes must match, got "
                    f"{tuple(prediction.shape)} and {tuple(target_regions.shape)}"
                )
            prediction_cpu = prediction.cpu()
            for sample_index in range(int(image_tensor.shape[0])):
                sample_metrics = _slice_metrics(
                    prediction_cpu[sample_index], target_regions[sample_index]
                )
                for region in REGION_NAMES:
                    values[f"{region}_dice"].append(sample_metrics[f"{region}_dice"])
                    values[f"{region}_hd95"].append(sample_metrics[f"{region}_hd95"])
                    excluded[region] += int(sample_metrics[f"hd95_excluded_{region}"])
            count += int(image_tensor.shape[0])
    finally:
        for module, was_training in zip(model.modules(), states):
            module.training = was_training
    if count == 0:
        raise ValueError("slice evaluation loader yielded no samples")

    result: dict[str, float] = {}
    dice_values: list[float] = []
    hd95_values: list[float] = []
    for region in REGION_NAMES:
        dice = float(np.mean(values[f"{region}_dice"]))
        finite_hd95 = [
            value
            for value in values[f"{region}_hd95"]
            if math.isfinite(value)
        ]
        hd95 = float(np.mean(finite_hd95)) if finite_hd95 else float("nan")
        dice_values.append(dice)
        hd95_values.append(hd95)
        result[f"{region}_dice"] = dice
        result[f"dice_{region}"] = dice
        result[f"{region}_hd95"] = hd95
        result[f"hd95_{region}"] = hd95
        result[f"hd95_excluded_{region}"] = float(excluded[region])
    result["mean_dice"] = float(np.mean(dice_values))
    finite_mean_hd95 = [value for value in hd95_values if math.isfinite(value)]
    result["mean_hd95"] = (
        float(np.mean(finite_mean_hd95)) if finite_mean_hd95 else float("nan")
    )
    return result


def build_slice_evaluator(
    cfg: Mapping[str, Any], _device: torch.device | str | None = None
) -> Callable[[nn.Module, Iterable[Any]], Mapping[str, Any]]:
    configured = _first_value(
        cfg,
        (("validation_evaluator",), ("evaluator",), ("evaluation", "evaluator")),
    )
    return configured if callable(configured) else evaluate_slices


def build_loss(cfg: Mapping[str, Any]) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    configured = _first_value(
        cfg,
        (
            ("experiment", "training", "loss_fn"),
            ("experiment", "training", "loss"),
            ("loss_fn",),
            ("loss",),
            ("training", "loss_fn"),
            ("training", "loss"),
        ),
    )
    if callable(configured):
        return configured

    if configured is None:
        configured_name = "binary_cross_entropy_with_logits"
    elif isinstance(configured, str):
        configured_name = configured.lower().replace("-", "_").replace(" ", "_")
    else:
        raise TypeError("loss configuration must be a callable or string")

    if configured_name in {
        "bce",
        "bce_with_logits",
        "binary_cross_entropy",
        "binary_cross_entropy_with_logits",
    }:

        def loss_fn(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return F.binary_cross_entropy_with_logits(logits, target.float())

        return loss_fn

    if configured_name in {"dice_ce", "dicece", "dice_cross_entropy"}:
        try:
            from monai.losses import DiceCELoss
        except ImportError as exc:
            raise ImportError(
                "dice_ce requires optional MONAI imaging dependencies"
            ) from exc
        # Match legacy SwinUNETR training: sigmoid Dice, squared prediction.
        return DiceCELoss(to_onehot_y=False, sigmoid=True, squared_pred=True)

    raise ValueError(
        f"unsupported loss '{configured}'; choose from binary_cross_entropy_with_logits or dice_ce"
    )


def build_phases(cfg: Mapping[str, Any]) -> list[PhaseSpec]:
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
                        ("training", "epochs"),
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


def build_checkpoints(cfg: Mapping[str, Any]) -> CheckpointManager | None:
    configured = _first_configured(cfg, (("checkpoints",),))
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


def resume_path(
    cfg: Mapping[str, Any], checkpoints: CheckpointManager | None
) -> Path | None:
    mode = _first_configured(
        cfg,
        (("resume_mode",), ("training", "resume_mode"), ("run", "resume_mode")),
    )
    if mode is not _MISSING and mode is not None and str(mode).lower() == "warm_start":
        return None
    configured = _first_configured(
        cfg,
        (
            ("resume",),
            ("resume_path",),
            ("paths", "resume"),
            ("checkpoints", "resume"),
            ("training", "resume"),
            ("experiment", "training", "resume"),
            ("run", "resume"),
        ),
    )
    if configured is _MISSING or configured is None or configured is False:
        return None
    if configured is True:
        if checkpoints is None:
            raise ValueError("resume requires enabled checkpointing")
        return checkpoints.root / "last.pt"
    return Path(configured)


def warm_start_path(
    cfg: Mapping[str, Any], checkpoints: CheckpointManager | None
) -> Path | None:
    mode = _first_configured(
        cfg,
        (("resume_mode",), ("training", "resume_mode"), ("run", "resume_mode")),
    )
    configured = _first_configured(
        cfg,
        (
            ("warm_start",),
            ("warm_start_path",),
            ("paths", "warm_start"),
            ("checkpoints", "warm_start"),
            ("training", "warm_start"),
            ("experiment", "training", "warm_start"),
            ("run", "warm_start"),
        ),
    )
    if (
        (configured is _MISSING or configured is None)
        and mode is not _MISSING
        and str(mode).lower() == "warm_start"
    ):
        configured = _first_configured(
            cfg,
            (
                ("resume",),
                ("resume_path",),
                ("paths", "resume"),
                ("checkpoints", "resume"),
                ("run", "resume"),
            ),
        )
    if configured is _MISSING or configured is None or configured is False:
        return None
    if configured is True:
        if checkpoints is None:
            raise ValueError("warm_start requires enabled checkpointing")
        return checkpoints.root / "best.pt"
    return Path(configured)


def _engine_config(
    cfg: Mapping[str, Any], metadata: Mapping[str, Any], device: torch.device
) -> dict[str, Any]:
    result = _flatten_training_config(cfg)
    result.update(metadata)
    result["device"] = str(device)
    result.setdefault("monitor", "mean_dice")
    result.setdefault("maximize", True)
    result.setdefault("weight_decay", 0.0)
    result.setdefault("use_amp", False)
    return result


def _tracking_config(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(_plain(_first_value(cfg, (("tracking",),), default={})), "tracking")


def architecture_metadata(
    cfg: Mapping[str, Any],
    architecture: str,
    dimensionality: str,
    *,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    model_cfg = _plain(_first_value(cfg, (("model",),), default={}))
    metadata: dict[str, Any] = {
        "architecture": architecture,
        "dimensionality": dimensionality,
        "spatial_dims": 3 if dimensionality == "3-D" else 2,
        "native_3d": dimensionality == "3-D",
        "slice_based": dimensionality == "2-D",
        "canonical_regions": tuple(REGION_NAMES),
        "output_kind": "raw_logits",
        "model_config": model_cfg,
    }
    if extra:
        metadata.update(_plain(extra))
    return metadata


def _numeric_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                continue
            value = value.detach().cpu().item()
        if isinstance(value, Number) and not isinstance(value, bool):
            result[str(key)] = float(value)
    return result


def _restore_best(
    model: nn.Module,
    checkpoints: CheckpointManager,
    *,
    expected_metadata: Mapping[str, Any] | None = None,
) -> None:
    path = checkpoints.root / "best.pt"
    if not path.is_file():
        raise FileNotFoundError(
            f"best checkpoint '{path}' is absent; cannot evaluate test split"
        )
    checkpoints.load_model(path, model, expected_metadata=expected_metadata)


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


def _invoke_fit(
    fit_fn: Callable[..., FitResult],
    args: tuple[Any, ...],
    *,
    resume: Path | None,
    warm_start: Path | None,
    loader_generator: torch.Generator,
) -> FitResult:
    kwargs: dict[str, Any] = {}
    if resume is not None and _accepts_keyword(fit_fn, "resume"):
        kwargs["resume"] = resume
    if warm_start is not None and _accepts_keyword(fit_fn, "warm_start"):
        kwargs["warm_start"] = warm_start
    if _accepts_keyword(fit_fn, "loader_generator"):
        kwargs["loader_generator"] = loader_generator
    return fit_fn(*args, **kwargs)


def _loader_parts(result: Any) -> tuple[Any, Any, Any, Mapping[str, Any]]:
    if isinstance(result, LoaderBundle):
        return result[0], result[1], result[2], result.metadata
    if isinstance(result, (tuple, list)) and len(result) == 4:
        metadata = result[3]
        if not isinstance(metadata, Mapping):
            raise TypeError("loader metadata must be a mapping")
        return result[0], result[1], result[2], metadata
    if isinstance(result, (tuple, list)) and len(result) == 3:
        if isinstance(result[2], Mapping):
            metadata = getattr(result, "metadata", result[2])
            if not isinstance(metadata, Mapping):
                raise TypeError("loader metadata must be a mapping")
            return result[0], result[1], getattr(result, "test_loader", None), metadata
        metadata = getattr(result, "metadata", {})
        if not isinstance(metadata, Mapping):
            raise TypeError("loader metadata must be a mapping")
        return result[0], result[1], result[2], metadata
    if isinstance(result, (tuple, list)) and len(result) == 2:
        metadata = getattr(result, "metadata", {})
        if not isinstance(metadata, Mapping):
            raise TypeError("loader metadata must be a mapping")
        return result[0], result[1], getattr(result, "test_loader", None), metadata
    raise TypeError("baseline loader builder must return train, val, and test loaders")


def _result_with_test_metrics(
    result: FitResult,
    test_metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> FitResult:
    if not isinstance(result, FitResult):
        raise TypeError("shared training engine must return FitResult")
    merged_metadata = dict(result.metadata or {})
    merged_metadata.update(metadata)
    return FitResult(
        result.best_metric,
        result.best_epoch,
        result.history,
        _numeric_metrics(test_metrics),
        merged_metadata,
    )


def run_3d_baseline(
    cfg: Mapping[str, Any],
    *,
    architecture: str,
    model_builder: Callable[[Mapping[str, Any]], nn.Module],
    loader_builder: Callable[[Mapping[str, Any], torch.Generator], Any],
    evaluator_builder: Callable[[Mapping[str, Any], torch.device], Callable[..., Mapping[str, Any]]],
    metadata_extra: Mapping[str, Any] | None = None,
    seed_fn: Callable[..., torch.Generator] = seed_everything,
    loss_builder: Callable[[Mapping[str, Any]], Callable[..., torch.Tensor]] = build_loss,
    phases_builder: Callable[[Mapping[str, Any]], Sequence[PhaseSpec]] = build_phases,
    checkpoint_builder: Callable[[Mapping[str, Any]], CheckpointManager | None] = build_checkpoints,
    tracker_builder: Callable[[Mapping[str, Any], Mapping[str, Any]], Tracker] = create_tracker,
    fit_fn: Callable[..., FitResult] = fit,
) -> FitResult:
    device = effective_device(cfg)
    spacing = explicit_spacing(cfg)
    seed = int(
        _first_value(cfg, (("seed",), ("reproducibility", "seed"), ("run", "seed")), default=0)
    )
    deterministic = bool(
        _first_value(
            cfg,
            (("deterministic",), ("reproducibility", "deterministic")),
            default=True,
        )
    )
    generator = seed_fn(seed, deterministic=deterministic)
    checkpoints = checkpoint_builder(cfg)
    if checkpoints is None:
        raise ValueError("3-D baseline pipelines require enabled checkpointing")
    resume = resume_path(cfg, checkpoints)
    warm_start = warm_start_path(cfg, checkpoints)
    if resume is not None and warm_start is not None:
        raise ValueError("resume and warm_start are mutually exclusive")

    model = model_builder(cfg)
    if not isinstance(getattr(model, "encoder", None), nn.Module):
        raise ValueError("baseline model must expose an nn.Module named 'encoder'")
    loader_result = loader_builder(cfg, generator)
    train_loader, val_loader, test_loader, loader_metadata = _loader_parts(loader_result)
    if not loader_metadata and _path_value(cfg, ("paths", "manifest"), _MISSING) is not _MISSING:
        raise ValueError("configured split manifest requires loader metadata from build_loaders")

    metadata = architecture_metadata(
        cfg,
        architecture,
        "3-D",
        extra={**dict(metadata_extra or {}), "spacing": spacing},
    )
    metadata.update(loader_metadata)
    metadata["execution_device"] = str(device)
    if resume is not None:
        metadata["source_checkpoint"] = str(resume)
        metadata["resume_mode"] = "exact"
    elif warm_start is not None:
        metadata["source_checkpoint"] = str(warm_start)
        metadata["resume_mode"] = "warm_start"
    run_config = _engine_config(cfg, metadata, device)
    tracker = tracker_builder(_tracking_config(cfg), run_config)
    evaluator = evaluator_builder(cfg, device)
    loss_fn = loss_builder(cfg)
    phases = phases_builder(cfg)
    result = _invoke_fit(
        fit_fn,
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
    if resume is not None:
        checkpoints.copy_best_from(resume)
    _restore_best(
        model,
        checkpoints,
        expected_metadata=_checkpoint_metadata(run_config, phases),
    )
    test_evaluator = _first_value(cfg, (("test_evaluator",), ("evaluation", "test_evaluator")))
    if not callable(test_evaluator):
        test_evaluator = evaluator
    test_metrics = test_evaluator(model, test_loader)
    if not isinstance(test_metrics, Mapping):
        raise TypeError("test evaluator must return a mapping of scalar metrics")
    return _result_with_test_metrics(result, test_metrics, metadata)


def run_2d_baseline(
    cfg: Mapping[str, Any],
    *,
    architecture: str,
    model_builder: Callable[[Mapping[str, Any]], nn.Module],
    loader_builder: Callable[[Mapping[str, Any], torch.Generator], Any],
    evaluator_builder: Callable[[Mapping[str, Any], torch.device], Callable[..., Mapping[str, Any]]],
    metadata_extra: Mapping[str, Any] | None = None,
    seed_fn: Callable[..., torch.Generator] = seed_everything,
    loss_builder: Callable[[Mapping[str, Any]], Callable[..., torch.Tensor]] = build_loss,
    phases_builder: Callable[[Mapping[str, Any]], Sequence[PhaseSpec]] = build_phases,
    checkpoint_builder: Callable[[Mapping[str, Any]], CheckpointManager | None] = build_checkpoints,
    tracker_builder: Callable[[Mapping[str, Any], Mapping[str, Any]], Tracker] = create_tracker,
    fit_fn: Callable[..., FitResult] = fit,
) -> FitResult:
    device = effective_device(cfg)
    seed = int(
        _first_value(cfg, (("seed",), ("reproducibility", "seed"), ("run", "seed")), default=0)
    )
    deterministic = bool(
        _first_value(
            cfg,
            (("deterministic",), ("reproducibility", "deterministic")),
            default=True,
        )
    )
    generator = seed_fn(seed, deterministic=deterministic)
    checkpoints = checkpoint_builder(cfg)
    if checkpoints is None:
        raise ValueError("2-D baseline pipelines require enabled checkpointing")
    resume = resume_path(cfg, checkpoints)
    warm_start = warm_start_path(cfg, checkpoints)
    if resume is not None and warm_start is not None:
        raise ValueError("resume and warm_start are mutually exclusive")

    model = model_builder(cfg)
    if not isinstance(getattr(model, "encoder", None), nn.Module):
        raise ValueError("baseline model must expose an nn.Module named 'encoder'")
    loader_result = loader_builder(cfg, generator)
    train_loader, val_loader, test_loader, loader_metadata = _loader_parts(loader_result)
    if not loader_metadata and _path_value(cfg, ("paths", "manifest"), _MISSING) is not _MISSING:
        raise ValueError("configured split manifest requires loader metadata from build_loaders")

    metadata = architecture_metadata(cfg, architecture, "2-D", extra=metadata_extra)
    metadata.update(loader_metadata)
    metadata["execution_device"] = str(device)
    if resume is not None:
        metadata["source_checkpoint"] = str(resume)
        metadata["resume_mode"] = "exact"
    elif warm_start is not None:
        metadata["source_checkpoint"] = str(warm_start)
        metadata["resume_mode"] = "warm_start"
    run_config = _engine_config(cfg, metadata, device)
    tracker = tracker_builder(_tracking_config(cfg), run_config)
    evaluator = evaluator_builder(cfg, device)
    loss_fn = loss_builder(cfg)
    phases = phases_builder(cfg)
    result = _invoke_fit(
        fit_fn,
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
    if resume is not None:
        checkpoints.copy_best_from(resume)
    _restore_best(
        model,
        checkpoints,
        expected_metadata=_checkpoint_metadata(run_config, phases),
    )
    test_evaluator = _first_value(cfg, (("test_evaluator",), ("evaluation", "test_evaluator")))
    if not callable(test_evaluator):
        test_evaluator = evaluator
    test_metrics = test_evaluator(model, test_loader)
    if not isinstance(test_metrics, Mapping):
        raise TypeError("test evaluator must return a mapping of scalar metrics")
    return _result_with_test_metrics(result, test_metrics, metadata)


__all__ = [
    "LoaderBundle",
    "architecture_metadata",
    "build_checkpoints",
    "build_loss",
    "build_phases",
    "build_slice_evaluator",
    "build_slice_loaders",
    "build_volume_evaluator",
    "build_volume_loaders",
    "effective_device",
    "evaluate_slices",
    "explicit_spacing",
    "resume_path",
    "warm_start_path",
    "run_2d_baseline",
    "run_3d_baseline",
]
