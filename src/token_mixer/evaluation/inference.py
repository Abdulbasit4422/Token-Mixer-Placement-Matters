from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from token_mixer.data.labels import multiclass_to_regions
from token_mixer.training.tracking import tracking_image_logging_enabled

from .metrics import (
    REGION_NAMES,
    dice_by_region,
    hd95_by_region,
    hd95_excluded_by_region,
    logits_to_regions,
)


_MONAI_IMPORT_ATTEMPTED = False


def _get_sliding_window_inference() -> Callable[..., torch.Tensor]:
    global _MONAI_IMPORT_ATTEMPTED
    _MONAI_IMPORT_ATTEMPTED = True
    try:
        from monai.inferers import sliding_window_inference
    except ImportError as exc:
        raise ImportError(
            "MONAI is required for full-volume evaluation; install the imaging extra"
        ) from exc
    return sliding_window_inference


def _device(value: torch.device | str) -> torch.device:
    if str(value).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(value)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested, but CUDA is unavailable")
    return resolved


def _mapping_value(batch: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in batch:
            return batch[key]
    return None


def _unpack_batch(batch: Any) -> tuple[Any, Any, Any, Any]:
    if isinstance(batch, Mapping):
        image = _mapping_value(batch, "image", "images", "input", "inputs", "x")
        target = _mapping_value(batch, "label", "labels", "target", "targets", "y")
        case_ids = _mapping_value(batch, "case_id", "case_ids", "id", "ids")
        spacing = _mapping_value(batch, "spacing", "spacings")
        if image is None or target is None:
            raise KeyError("evaluation batch must contain image and label/target values")
        return image, target, case_ids, spacing

    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        extras = list(batch[2:])
        case_ids = extras[0] if extras else None
        spacing = extras[1] if len(extras) > 1 else None
        return batch[0], batch[1], case_ids, spacing

    raise TypeError("evaluation loader must yield a mapping or (image, target, ...) tuple")


def _as_batched_tensor(value: Any, name: str) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.ndim == 4:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 5:
        raise ValueError(f"{name} must have shape (B, C, D, H, W), got {tuple(tensor.shape)}")
    return tensor


def _as_case_ids(value: Any, batch_size: int, offset: int) -> list[str]:
    if value is None:
        return [str(offset + index) for index in range(batch_size)]
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, torch.Tensor):
        values = value.detach().cpu().tolist()
    elif isinstance(value, np.ndarray):
        values = value.tolist()
    elif isinstance(value, Sequence):
        values = list(value)
    else:
        values = [value]
    if len(values) != batch_size:
        if batch_size == 1 and len(values) == 1:
            return [str(values[0])]
        raise ValueError(
            f"evaluation batch contains {batch_size} samples but {len(values)} case IDs"
        )
    return [str(case_id) for case_id in values]


def _validate_spacing(value: Any) -> tuple[float, float, float]:
    try:
        spacing = tuple(float(part) for part in value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("spacing must contain three positive finite values") from exc
    if len(spacing) != 3 or any(
        not math.isfinite(part) or part <= 0 for part in spacing
    ):
        raise ValueError("spacing must contain three positive finite values")
    return spacing


def _case_spacings(
    value: Any,
    batch_size: int,
    *,
    default_spacing: Sequence[float] | None = None,
) -> list[tuple[float, float, float]]:
    """Normalize one spacing or per-case spacing values to batch order.

    PyTorch's default collator represents per-case triples as ``[3, B]``;
    explicit batch spacing is also accepted as ``[B, 3]``. When a batch omits
    spacing, ``default_spacing`` must supply it. Unit voxel spacing is never
    inferred implicitly because doing so could make HD95 appear physical.
    """
    if value is None:
        if default_spacing is None:
            raise ValueError(
                "spacing is required; provide per-batch spacing or explicit "
                "default_spacing"
            )
        fallback = _validate_spacing(default_spacing)
        return [fallback] * batch_size

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    elif isinstance(value, np.ndarray):
        value = value.tolist()

    try:
        values = list(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("spacing must contain three positive finite values") from exc

    if batch_size == 3 and len(values) == 3:
        try:
            nested_lengths = [len(item) for item in values]
        except TypeError:
            nested_lengths = None
        if nested_lengths == [3, 3, 3]:
            raise ValueError(
                "ambiguous 3x3 spacing layout when batch_size is 3; provide "
                "one shared triple or spacing with an unambiguous batch size"
            )

    try:
        single_spacing = _validate_spacing(values)
    except ValueError:
        single_spacing = None
    if single_spacing is not None:
        return [single_spacing] * batch_size

    if len(values) == batch_size:
        return [_validate_spacing(item) for item in values]

    if len(values) == 3:
        try:
            collated_values = list(zip(*values))
        except TypeError as exc:
            raise ValueError("spacing must contain three positive finite values") from exc
        if len(collated_values) == batch_size:
            return [_validate_spacing(item) for item in collated_values]

    raise ValueError("spacing must contain three positive finite values")


def _positive_integer_sequence(value: Any, name: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{name} must contain positive integers")
    try:
        values = tuple(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain positive integers") from exc
    if not values:
        raise ValueError(f"{name} must contain positive integers")
    normalized: list[int] = []
    for item in values:
        if isinstance(item, bool) or not isinstance(item, Integral):
            raise ValueError(f"{name} must contain positive integers")
        item = int(item)
        if item <= 0:
            raise ValueError(f"{name} must contain positive integers")
        normalized.append(item)
    return tuple(normalized)


def _validated_overlap(overlap: Any) -> float:
    if isinstance(overlap, bool):
        raise ValueError("overlap must satisfy 0 <= overlap < 1")
    try:
        value = float(overlap)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("overlap must satisfy 0 <= overlap < 1") from exc
    if not math.isfinite(value) or not 0.0 <= value < 1.0:
        raise ValueError("overlap must satisfy 0 <= overlap < 1")
    return value


def sliding_window_count(
    spatial_shape: Sequence[int],
    roi_size: Sequence[int],
    overlap: float,
) -> int:
    """Count MONAI dense sliding-window patches for one spatial shape.

    MONAI pads dimensions smaller than the ROI to the ROI size, uses an
    integer scan interval of ``int(roi * (1 - overlap))`` (with a minimum of
    one), and adds a final patch when needed to cover the far boundary.
    """
    image_size = _positive_integer_sequence(spatial_shape, "spatial_shape")
    patch_size = _positive_integer_sequence(roi_size, "roi_size")
    if len(image_size) != len(patch_size):
        raise ValueError("spatial_shape and roi_size must have the same length")
    validated_overlap = _validated_overlap(overlap)

    count = 1
    for image_dimension, patch_dimension in zip(image_size, patch_size):
        effective_image_dimension = max(image_dimension, patch_dimension)
        if effective_image_dimension <= patch_dimension:
            dimension_count = 1
        else:
            scan_interval = max(int(patch_dimension * (1.0 - validated_overlap)), 1)
            dimension_count = (
                math.ceil(
                    (effective_image_dimension - patch_dimension) / scan_interval
                )
                + 1
            )
        count *= dimension_count
    return int(count)


def _restore_training_states(model: torch.nn.Module, states: list[bool]) -> None:
    for module, was_training in zip(model.modules(), states):
        module.training = was_training


def _nanmean(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or np.all(np.isnan(array)):
        return float("nan")
    return float(np.nanmean(array))


def evaluate_full_volumes(
    model: torch.nn.Module,
    loader: Any,
    roi_size: Sequence[int],
    sw_batch_size: int,
    overlap: float,
    device: torch.device | str,
    *,
    default_spacing: Sequence[float] | None = None,
    collect_case_records: bool = False,
    measure_latency: bool = False,
) -> dict[str, Any]:
    """Run MONAI sliding-window inference and aggregate full-volume metrics.

    Batches must provide spacing after case IDs, or callers must pass
    ``default_spacing`` explicitly. No implicit unit spacing is used: missing
    physical spacing cannot safely produce physical-looking HD95 values.
    Case records are collected only when ``collect_case_records`` is true.
    ``measure_latency`` adds per-case timing fields and requires batch size one.
    """
    sliding_window_inference = _get_sliding_window_inference()
    resolved_device = _device(device)
    normalized_roi_size = _positive_integer_sequence(roi_size, "roi_size")
    if len(normalized_roi_size) != 3:
        raise ValueError("roi_size must contain three positive integers")
    normalized_overlap = _validated_overlap(overlap)
    collect_case_records = bool(collect_case_records)
    measure_latency = bool(measure_latency)
    if measure_latency and not collect_case_records:
        # Timing has no public aggregate field.  Avoid doing hidden timing
        # work when callers did not request case records.
        measure_latency = False
    model_states = [module.training for module in model.modules()]
    model.eval()

    case_ids: list[str] = []
    case_records: list[dict[str, Any]] = []
    dice_values: dict[str, list[float]] = {region: [] for region in REGION_NAMES}
    hd95_values: dict[str, list[float]] = {region: [] for region in REGION_NAMES}
    excluded: dict[str, int] = {region: 0 for region in REGION_NAMES}
    excluded_case_ids: set[str] = set()

    try:
        with torch.no_grad():
            sample_offset = 0
            iterator = iter(loader)
            while True:
                batch_started = time.perf_counter() if measure_latency else None
                load_started = time.perf_counter() if measure_latency else None
                try:
                    batch = next(iterator)
                except StopIteration:
                    break
                load_seconds = (
                    time.perf_counter() - load_started
                    if load_started is not None
                    else None
                )
                preprocess_started = time.perf_counter() if measure_latency else None
                image, target, batch_case_ids, batch_spacing = _unpack_batch(batch)
                image_tensor = _as_batched_tensor(image, "image").to(resolved_device)
                target_tensor = _as_batched_tensor(target, "target").to(resolved_device)
                if image_tensor.shape[0] != target_tensor.shape[0]:
                    raise ValueError("image and target batch sizes must match")
                if target_tensor.shape[1] != len(REGION_NAMES):
                    raise ValueError(
                        f"target must contain {len(REGION_NAMES)} region channels, "
                        f"got shape {tuple(target_tensor.shape)}"
                    )

                if measure_latency and image_tensor.shape[0] != 1:
                    raise ValueError(
                        "case-level latency measurement requires evaluation batch size 1"
                    )

                batch_case_ids_list = _as_case_ids(
                    batch_case_ids, image_tensor.shape[0], sample_offset
                )
                spacings = _case_spacings(
                    batch_spacing,
                    image_tensor.shape[0],
                    default_spacing=default_spacing,
                )
                if measure_latency and resolved_device.type == "cuda":
                    torch.cuda.synchronize(resolved_device)
                preprocess_seconds = (
                    time.perf_counter() - preprocess_started
                    if preprocess_started is not None
                    else None
                )

                model_started = time.perf_counter() if measure_latency else None
                raw_logits = sliding_window_inference(
                    inputs=image_tensor,
                    roi_size=normalized_roi_size,
                    sw_batch_size=int(sw_batch_size),
                    predictor=model,
                    overlap=normalized_overlap,
                )
                if measure_latency and resolved_device.type == "cuda":
                    torch.cuda.synchronize(resolved_device)
                model_compute_seconds = (
                    time.perf_counter() - model_started
                    if model_started is not None
                    else None
                )
                if not isinstance(raw_logits, torch.Tensor):
                    raise TypeError("sliding-window predictor must return a torch.Tensor")
                postprocess_started = time.perf_counter() if measure_latency else None
                predictions = logits_to_regions(raw_logits)
                postprocess_seconds = (
                    time.perf_counter() - postprocess_started
                    if postprocess_started is not None
                    else None
                )

                metric_started = time.perf_counter() if measure_latency else None
                for index, case_id in enumerate(batch_case_ids_list):
                    case_prediction = predictions[index]
                    case_target = target_tensor[index]
                    case_dice = dice_by_region(case_prediction, case_target)
                    case_hd95 = hd95_by_region(
                        case_prediction,
                        case_target,
                        spacing=spacings[index],
                    )
                    case_excluded = hd95_excluded_by_region(case_prediction, case_target)
                    if any(case_excluded.values()):
                        excluded_case_ids.add(case_id)
                    case_ids.append(case_id)
                    for region in REGION_NAMES:
                        dice_values[region].append(case_dice[region])
                        hd95_values[region].append(case_hd95[region])
                        excluded[region] += case_excluded[region]
                    if collect_case_records:
                        spatial_shape = tuple(
                            int(size) for size in image_tensor.shape[-3:]
                        )
                        record: dict[str, Any] = {
                            "case_id": case_id,
                            "dice_by_region": dict(case_dice),
                            "hd95_by_region": dict(case_hd95),
                            "hd95_excluded_by_region": dict(case_excluded),
                            "hd95_excluded_count": sum(case_excluded.values()),
                            "voxel_count": int(math.prod(spatial_shape)),
                            "spacing": spacings[index],
                            "sliding_window_count": sliding_window_count(
                                spatial_shape,
                                normalized_roi_size,
                                normalized_overlap,
                            ),
                            "exclusion_flags": [
                                region
                                for region in REGION_NAMES
                                if case_excluded[region]
                            ],
                        }
                        if measure_latency:
                            assert model_compute_seconds is not None
                            assert load_seconds is not None
                            assert preprocess_seconds is not None
                            assert postprocess_seconds is not None
                            assert metric_started is not None
                            metric_seconds = time.perf_counter() - metric_started
                            end_to_end_seconds = (
                                time.perf_counter() - batch_started
                                if batch_started is not None
                                else None
                            )
                            record.update(
                                {
                                    "latency_ms": float(end_to_end_seconds * 1000.0),
                                    "load_seconds": float(load_seconds),
                                    "preprocess_seconds": float(preprocess_seconds),
                                    "model_compute_seconds": float(model_compute_seconds),
                                    "postprocess_seconds": float(postprocess_seconds),
                                    "metric_seconds": float(metric_seconds),
                                    "end_to_end_seconds": float(end_to_end_seconds),
                                }
                            )
                        case_records.append(record)
                sample_offset += image_tensor.shape[0]
    finally:
        _restore_training_states(model, model_states)

    result: dict[str, Any] = {"case_ids": case_ids}
    region_dice: list[float] = []
    region_hd95: list[float] = []
    for region in REGION_NAMES:
        mean_dice = float(np.mean(dice_values[region])) if dice_values[region] else float("nan")
        mean_hd95 = _nanmean(hd95_values[region])
        region_dice.append(mean_dice)
        region_hd95.append(mean_hd95)
        result[f"{region}_dice"] = mean_dice
        result[f"{region}_hd95"] = mean_hd95
        result[f"dice_{region}"] = mean_dice
        result[f"hd95_{region}"] = mean_hd95
        result[f"hd95_excluded_{region}"] = excluded[region]

    result["mean_dice"] = float(np.mean(region_dice)) if region_dice else float("nan")
    result["mean_hd95"] = _nanmean(region_hd95)
    result["hd95_excluded_cases"] = len(excluded_case_ids)
    if collect_case_records:
        result["case_records"] = case_records
    return result


def _as_2d_batched_tensor(value: Any, name: str) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4:
        raise ValueError(f"{name} must have shape (B, C, H, W), got {tuple(tensor.shape)}")
    return tensor


def _as_snapshot_input(value: Any, *, native_3d: bool, name: str) -> torch.Tensor:
    return (
        _as_batched_tensor(value, name)
        if native_3d
        else _as_2d_batched_tensor(value, name)
    )


def infer_regions(
    model: torch.nn.Module,
    image: Any,
    *,
    device: torch.device | str = "cpu",
    native_3d: bool = True,
    roi_size: Sequence[int] | None = None,
    sw_batch_size: int = 1,
    overlap: float = 0.25,
) -> torch.Tensor:
    """Run preview inference and return thresholded ET/TC/WT masks.

    Native 3-D models use the same MONAI sliding-window boundary as full-volume
    evaluation. TransUNet-style 2-D models call their ordinary forward path.
    Model training flags are restored before returning.
    """
    resolved_device = _device(device)
    image_tensor = _as_snapshot_input(image, native_3d=native_3d, name="image")
    image_tensor = image_tensor.to(resolved_device)
    model_states = [module.training for module in model.modules()]
    model.eval()
    try:
        with torch.no_grad():
            if native_3d:
                if roi_size is None:
                    roi_size = tuple(int(size) for size in image_tensor.shape[-3:])
                sliding_window_inference = _get_sliding_window_inference()
                raw_logits = sliding_window_inference(
                    inputs=image_tensor,
                    roi_size=tuple(int(size) for size in roi_size),
                    sw_batch_size=int(sw_batch_size),
                    predictor=model,
                    overlap=float(overlap),
                )
            else:
                raw_logits = model(image_tensor)
            if not isinstance(raw_logits, torch.Tensor):
                raise TypeError("preview predictor must return a torch.Tensor")
            return logits_to_regions(raw_logits).detach().cpu()
    finally:
        _restore_training_states(model, model_states)


def _plain_snapshot_config(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_snapshot_config(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_plain_snapshot_config(item) for item in value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain_snapshot_config(item) for item in value]
    return value


def _mapping_snapshot_config(value: Any) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("snapshot configuration must be a mapping")
    return value


def _bool_snapshot_config(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "on"}:
            return True
        if normalized in {"false", "no", "0", "off"}:
            return False
    return bool(value)


def _snapshot_settings(config: Mapping[str, Any]) -> Mapping[str, Any]:
    visualization = config.get("visualization", {})
    if not isinstance(visualization, Mapping):
        raise TypeError("visualization configuration must be a mapping")
    configured = visualization.get("segmentation_snapshots", {})
    if configured is None:
        return {}
    return _mapping_snapshot_config(configured)


def _tracker_images_enabled(config: Mapping[str, Any], tracker: Any) -> bool:
    return tracking_image_logging_enabled(config, tracker)


def _snapshot_device(config: Mapping[str, Any], device: torch.device | str | None) -> Any:
    if device is not None:
        return device
    configured = config.get("device", "auto")
    return configured


def _snapshot_native_3d(config: Mapping[str, Any], image: Any | None = None) -> bool:
    configured = config.get("native_3d")
    if configured is not None:
        return _bool_snapshot_config(configured, True)
    dimensionality = config.get("dimensionality")
    if dimensionality is not None:
        return str(dimensionality).lower() in {"3-d", "3d", "native_3d"}
    if _bool_snapshot_config(config.get("slice_based"), False):
        return False
    if image is not None:
        try:
            return np.asarray(image).ndim >= 4
        except Exception:
            pass
    return True


def _snapshot_int(value: Any, name: str, default: int) -> int:
    configured = default if value is None else value
    if isinstance(configured, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = int(configured)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if result < 1 or (isinstance(configured, float) and result != configured):
        raise ValueError(f"{name} must be a positive integer")
    return result


def _snapshot_nonnegative_int(value: Any, name: str, default: int) -> int:
    configured = default if value is None else value
    if isinstance(configured, bool):
        raise ValueError(f"{name} must be a non-negative integer")
    try:
        result = int(configured)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if result < 0 or (isinstance(configured, float) and result != configured):
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _snapshot_size3(value: Any, name: str) -> tuple[int, int, int]:
    if isinstance(value, Integral) and not isinstance(value, bool):
        values = (int(value),) * 3
    else:
        if isinstance(value, (str, bytes, bytearray)):
            raise ValueError(f"{name} must contain three positive integers")
        try:
            values = tuple(int(item) for item in value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must contain three positive integers") from exc
    if len(values) != 3 or any(item <= 0 for item in values):
        raise ValueError(f"{name} must contain three positive integers")
    return values


def _snapshot_axis(value: Any) -> int:
    try:
        axis = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("snapshot axis must be 0, 1, or 2") from exc
    if axis not in (0, 1, 2):
        raise ValueError("snapshot axis must be 0, 1, or 2")
    return axis


def _snapshot_case_hash(case_id: Any, image: Any, target: Any) -> str:
    digest = hashlib.sha256()
    if case_id is not None:
        digest.update(b"case-id:")
        digest.update(str(case_id).encode("utf-8", errors="replace"))
    else:
        digest.update(b"sample:")
        for value in (image, target):
            array = np.ascontiguousarray(np.asarray(value))
            digest.update(str(array.shape).encode("ascii"))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class SnapshotExample:
    image: Any
    target: Any
    case_hash: str


def _unwrap_snapshot_dataset(loader: Any) -> Any:
    current = loader
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        dataset = getattr(current, "dataset", None)
        if dataset is None:
            wrapped = getattr(current, "loader", None)
            if wrapped is None:
                break
            current = wrapped
            continue
        current = dataset
    return current


def _without_slice_crops(config: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(_plain_snapshot_config(config))
    for key in ("patch_size", "crop_size", "roi_size"):
        result.pop(key, None)
    for section_name in ("data", "dataset", "preprocessing", "transform"):
        section = result.get(section_name)
        if isinstance(section, Mapping):
            section = dict(section)
            for key in ("patch_size", "crop_size", "roi_size"):
                section.pop(key, None)
            result[section_name] = section
    return result


def _canonical_snapshot_target(target: Any) -> np.ndarray:
    array = np.asarray(target)
    if array.ndim == 4 and array.shape[0] == len(REGION_NAMES):
        return np.asarray(array, dtype=np.float32).copy()
    if array.ndim == 3 and array.shape[0] == len(REGION_NAMES):
        return np.asarray(array, dtype=np.float32).copy()
    if array.ndim == 2:
        label = array[None, ...]
        if not np.isfinite(label).all() or not np.equal(label, np.round(label)).all():
            raise ValueError("snapshot class targets must contain finite integer labels")
        return multiclass_to_regions(label)[:, 0]
    if array.ndim == 3:
        if not np.isfinite(array).all() or not np.equal(array, np.round(array)).all():
            raise ValueError("snapshot class targets must contain finite integer labels")
        return multiclass_to_regions(array)
    raise ValueError(f"snapshot target has unsupported shape {array.shape}")


def _sample_batch_size(image: Any, target: Any, case_ids: Any) -> int:
    if isinstance(case_ids, str) or case_ids is None:
        case_count = None
    else:
        try:
            case_count = len(case_ids)
        except TypeError:
            case_count = None
    if case_count is not None and case_count > 0:
        return int(case_count)
    image_array = np.asarray(image)
    target_array = np.asarray(target)
    if image_array.ndim == 5:
        return int(image_array.shape[0])
    if image_array.ndim == 4 and target_array.ndim == 5:
        return int(target_array.shape[0])
    if (
        image_array.ndim == 4
        and target_array.ndim == 4
        and target_array.shape[1] == len(REGION_NAMES)
        and image_array.shape[0] != len(REGION_NAMES)
    ):
        return int(image_array.shape[0])
    return 1


def _take_snapshot_sample(value: Any, index: int, batch_size: int) -> Any:
    array = np.asarray(value)
    if batch_size == 1:
        if array.ndim >= 4 and array.shape[0] == 1:
            return array[0]
        return array
    return array[index]


def _case_snapshot_examples(
    dataset: Any, sample_count: int, axis: int
) -> list[SnapshotExample]:
    cases = getattr(dataset, "cases", None)
    raw_config = getattr(dataset, "config", None)
    if not isinstance(cases, Sequence) or not isinstance(raw_config, Mapping):
        return []
    # Import data transforms only after effective snapshot gating has succeeded.
    from token_mixer.data.datasets import load_case_arrays
    from token_mixer.data.labels import regions_to_multiclass
    from token_mixer.data.transforms import crop_slice, preprocess_volume
    from token_mixer.evaluation.visualization import region_aware_slice_indices

    config = _plain_snapshot_config(raw_config)
    if not isinstance(config, Mapping):
        return []
    dataset_name = type(dataset).__name__.lower()
    examples: list[SnapshotExample] = []
    for case_index, case in enumerate(cases):
        if len(examples) >= sample_count:
            break
        image, label = load_case_arrays(case)
        if "slice" in dataset_name:
            processed_image, regions = preprocess_volume(
                image,
                label,
                _without_slice_crops(config),
                training=False,
            )
            configured_axis = getattr(dataset, "slice_axis", axis)
            try:
                configured_axis = int(configured_axis)
            except (TypeError, ValueError):
                configured_axis = axis
            configured_axis = _snapshot_axis(configured_axis)
            selected = region_aware_slice_indices(regions, slice_axis=configured_axis)
            selected_index = selected[0] if selected else processed_image.shape[configured_axis + 1] // 2
            image_slice = np.take(processed_image, selected_index, axis=configured_axis + 1)
            region_slice = np.take(regions, selected_index, axis=configured_axis + 1)
            slice_size = config.get("slice_size")
            if slice_size is None:
                data_config = config.get("data")
                if isinstance(data_config, Mapping):
                    slice_size = data_config.get("slice_size")
            if slice_size is not None:
                class_slice = regions_to_multiclass(region_slice[:, None, ...])[0]
                image_slice, class_slice = crop_slice(
                    image_slice,
                    class_slice,
                    slice_size,
                    training=False,
                    rng=np.random.default_rng(0),
                )
                region_slice = multiclass_to_regions(class_slice[None, ...])[:, 0]
            examples.append(
                SnapshotExample(
                    image_slice,
                    np.asarray(region_slice, dtype=np.float32),
                    _snapshot_case_hash(getattr(case, "case_id", case_index), image_slice, region_slice),
                )
            )
        else:
            processed_image, regions = preprocess_volume(
                image, label, config, training=False
            )
            examples.append(
                SnapshotExample(
                    processed_image,
                    np.asarray(regions, dtype=np.float32),
                    _snapshot_case_hash(getattr(case, "case_id", case_index), processed_image, regions),
                )
            )
    return examples


def _generic_snapshot_examples(loader: Any, sample_count: int) -> list[SnapshotExample]:
    dataset = _unwrap_snapshot_dataset(loader)
    examples: list[SnapshotExample] = []
    if dataset is not None and hasattr(dataset, "__getitem__"):
        try:
            dataset_length = len(dataset)
        except (TypeError, ValueError):
            dataset_length = sample_count
        for index in range(min(sample_count, int(dataset_length))):
            try:
                batch = dataset[index]
            except (IndexError, KeyError):
                break
            image, target, case_ids, _spacing = _unpack_batch(batch)
            batch_size = _sample_batch_size(image, target, case_ids)
            sample_image = _take_snapshot_sample(image, 0, batch_size)
            sample_target = _take_snapshot_sample(target, 0, batch_size)
            case_id = case_ids[0] if isinstance(case_ids, Sequence) and not isinstance(case_ids, str) else case_ids
            examples.append(
                SnapshotExample(
                    sample_image,
                    _canonical_snapshot_target(sample_target),
                    _snapshot_case_hash(case_id, sample_image, sample_target),
                )
            )
        return examples

    for batch in loader:
        image, target, case_ids, _spacing = _unpack_batch(batch)
        batch_size = _sample_batch_size(image, target, case_ids)
        for index in range(batch_size):
            sample_image = _take_snapshot_sample(image, index, batch_size)
            sample_target = _take_snapshot_sample(target, index, batch_size)
            case_id = (
                case_ids[index]
                if isinstance(case_ids, Sequence) and not isinstance(case_ids, str)
                else case_ids
            )
            examples.append(
                SnapshotExample(
                    sample_image,
                    _canonical_snapshot_target(sample_target),
                    _snapshot_case_hash(case_id, sample_image, sample_target),
                )
            )
            if len(examples) >= sample_count:
                return examples
        if len(examples) >= sample_count:
            break
    return examples


def fixed_snapshot_examples(loader: Any, sample_count: int, *, axis: int = 0) -> list[SnapshotExample]:
    """Select deterministic, non-augmented examples without touching model state."""
    dataset = _unwrap_snapshot_dataset(loader)
    examples = _case_snapshot_examples(dataset, sample_count, axis)
    if examples:
        return examples
    return _generic_snapshot_examples(loader, sample_count)


def _snapshot_roi_size(config: Mapping[str, Any]) -> tuple[int, int, int] | None:
    paths = (
        ("inference", "roi_size"),
        ("evaluation", "roi_size"),
        ("roi_size",),
        ("data", "patch_size"),
        ("data", "roi_size"),
        ("patch_size",),
    )
    for path in paths:
        current: Any = config
        for key in path:
            if not isinstance(current, Mapping) or key not in current:
                current = None
                break
            current = current[key]
        if current is not None:
            return _snapshot_size3(current, "roi_size")
    return (96, 96, 96)


def _snapshot_sw_batch_size(config: Mapping[str, Any]) -> int:
    for section_name in ("inference", "evaluation"):
        section = config.get(section_name)
        if isinstance(section, Mapping) and section.get("sw_batch_size") is not None:
            return _snapshot_int(section["sw_batch_size"], "sw_batch_size", 1)
    return _snapshot_int(config.get("sw_batch_size", 1), "sw_batch_size", 1)


def _snapshot_overlap(config: Mapping[str, Any]) -> float:
    value: Any = None
    for section_name in ("inference", "evaluation"):
        section = config.get(section_name)
        if isinstance(section, Mapping) and section.get("overlap") is not None:
            value = section["overlap"]
            break
    if value is None:
        value = config.get("overlap", 0.25)
    try:
        overlap = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("snapshot overlap must satisfy 0 <= overlap < 1") from exc
    if not math.isfinite(overlap) or not 0.0 <= overlap < 1.0:
        raise ValueError("snapshot overlap must satisfy 0 <= overlap < 1")
    return overlap


class SegmentationSnapshotter:
    """Tracker-gated deterministic train/validation preview collector."""

    def __init__(
        self,
        config: Mapping[str, Any],
        tracker: Any,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        self.config = _plain_snapshot_config(config)
        if not isinstance(self.config, Mapping):
            raise TypeError("snapshotter configuration must be a mapping")
        settings = _snapshot_settings(self.config)
        self.enabled = _bool_snapshot_config(settings.get("enabled", False), False)
        self.interval_epochs = _snapshot_int(
            settings.get("snapshot_interval_epochs", 10),
            "snapshot_interval_epochs",
            10,
        )
        self.include_best = _bool_snapshot_config(settings.get("include_best", True), True)
        self.include_final = _bool_snapshot_config(settings.get("include_final", True), True)
        raw_splits = settings.get("splits", ("train", "val"))
        if isinstance(raw_splits, str):
            raw_splits = (raw_splits,)
        try:
            self.splits = tuple(str(split) for split in raw_splits)
        except TypeError as exc:
            raise ValueError("snapshot splits must be a sequence") from exc
        if not self.splits or any(split not in {"train", "val"} for split in self.splits):
            raise ValueError("snapshot splits must contain only train and val")
        self.sample_count = _snapshot_int(settings.get("sample_count", 1), "sample_count", 1)
        self.axis = _snapshot_axis(settings.get("axis", 0))
        self.image_channel = _snapshot_nonnegative_int(
            settings.get("image_channel", 3),
            "image_channel",
            3,
        )
        self.local_enabled = _bool_snapshot_config(settings.get("local_enabled", False), False)
        output_dir = settings.get("output_dir", "segmentation_snapshots")
        self.output_dir = Path(output_dir)
        self.tracker = tracker
        self.wandb_enabled = _tracker_images_enabled(self.config, tracker) and callable(
            getattr(tracker, "log_images", None)
        )
        self.device = _snapshot_device(self.config, device)
        self.native_3d = (
            None
            if self.config.get("native_3d") is None and self.config.get("dimensionality") is None
            else _snapshot_native_3d(self.config)
        )
        self.final_validation = True
        self._examples: dict[str, list[SnapshotExample]] = {}
        self._emitted: set[tuple[str, int, str, str]] = set()
        self._errors: list[str] = []
        self._ledger_path = self.output_dir / ".snapshot_ledger.json"
        self._load_ledger()

    @property
    def image_work_enabled(self) -> bool:
        return self.enabled and (self.local_enabled or self.wandb_enabled)

    @property
    def errors(self) -> tuple[str, ...]:
        return tuple(self._errors)

    def _record_error(self, error: BaseException) -> None:
        error_type = type(error).__name__
        if not error_type.isidentifier():
            error_type = "SnapshotError"
        self._errors.append(f"{error_type[:64]}: snapshot_failed")
        del self._errors[:-32]

    def _load_ledger(self) -> None:
        if not self._ledger_path.is_file():
            return
        try:
            payload = json.loads(self._ledger_path.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                return
            for item in payload:
                if not isinstance(item, list) or len(item) != 4:
                    continue
                split, epoch, kind, case_hash = item
                if isinstance(split, str) and isinstance(kind, str) and isinstance(case_hash, str):
                    self._emitted.add((split, int(epoch), kind, case_hash))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            self._record_error(error)

    def _save_ledger(self) -> None:
        try:
            self._ledger_path.parent.mkdir(parents=True, exist_ok=True)
            payload = [list(item) for item in sorted(self._emitted)]
            self._ledger_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        except OSError as error:
            self._record_error(error)

    def state_dict(self) -> dict[str, Any]:
        return {
            "emitted": [list(item) for item in sorted(self._emitted)],
            "errors": list(self._errors),
        }

    def load_state_dict(self, state: Mapping[str, Any] | None) -> None:
        if not isinstance(state, Mapping):
            return
        emitted = state.get("emitted", ())
        if isinstance(emitted, Sequence):
            for item in emitted:
                if not isinstance(item, Sequence) or len(item) != 4:
                    continue
                split, epoch, kind, case_hash = item
                if not isinstance(split, str) or not isinstance(kind, str) or not isinstance(case_hash, str):
                    continue
                try:
                    self._emitted.add((split, int(epoch), kind, case_hash))
                except (TypeError, ValueError):
                    continue
        errors = state.get("errors")
        if isinstance(errors, Sequence) and not isinstance(errors, (str, bytes, bytearray)):
            self._errors = [self._redact_error_text(error) for error in errors][-32:]

    @staticmethod
    def _redact_error_text(error: Any) -> str:
        known_types = {
            "AssertionError",
            "AttributeError",
            "FileNotFoundError",
            "ImportError",
            "IndexError",
            "KeyError",
            "MemoryError",
            "OSError",
            "RuntimeError",
            "TimeoutError",
            "TypeError",
            "ValueError",
        }
        prefix = str(error).split(":", 1)[0].strip()
        error_type = prefix if prefix in known_types else "SnapshotError"
        return f"{error_type}: snapshot_failed"

    def _examples_for(self, split: str, loader: Any) -> list[SnapshotExample]:
        if split not in self._examples:
            self._examples[split] = fixed_snapshot_examples(
                loader,
                self.sample_count,
                axis=self.axis,
            )
        return self._examples[split]

    def _image_key(self, split: str, epoch: int, kind: str, index: int, count: int) -> str:
        if kind == "epoch":
            label = f"epoch_{epoch:04d}"
        else:
            label = f"{kind}_epoch_{epoch:04d}"
        if count > 1:
            label = f"{label}_{index:02d}"
        return f"segmentation/{split}/{label}"

    def _local_path(self, split: str, epoch: int, kind: str, index: int, count: int) -> Path:
        key = self._image_key(split, epoch, kind, index, count)
        return self.output_dir / split / f"{key.rsplit('/', 1)[-1]}.png"

    def _render_example(
        self,
        model: torch.nn.Module,
        example: SnapshotExample,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, ...], int]:
        native_3d = (
            _snapshot_native_3d(self.config, example.image)
            if self.native_3d is None
            else self.native_3d
        )
        prediction = infer_regions(
            model,
            example.image,
            device=self.device,
            native_3d=native_3d,
            roi_size=_snapshot_roi_size(self.config) if native_3d else None,
            sw_batch_size=_snapshot_sw_batch_size(self.config),
            overlap=_snapshot_overlap(self.config),
        )
        prediction_array = prediction[0].numpy()
        target_array = _canonical_snapshot_target(example.target)
        image_array = np.asarray(example.image)
        if native_3d:
            render_image = image_array
            render_target = target_array
            render_prediction = prediction_array
            render_axis = self.axis
        else:
            if image_array.ndim == 4 and image_array.shape[0] == 1:
                image_array = image_array[0]
            render_image = image_array[:, None, ...]
            render_target = target_array[:, None, ...]
            render_prediction = prediction_array[:, None, ...]
            render_axis = 0
        from token_mixer.evaluation.visualization import region_aware_slice_indices

        slice_indices = region_aware_slice_indices(render_target, slice_axis=render_axis)
        return (
            render_image,
            render_target,
            render_prediction,
            slice_indices,
            render_axis,
        )

    def snapshot(
        self,
        *,
        model: torch.nn.Module,
        train_loader: Any,
        val_loader: Any,
        epoch: int,
        global_step: int,
        kind: str = "epoch",
    ) -> list[str]:
        """Collect one scheduled, best, or final event.

        Effective gates are evaluated before accessing either loader. Rendering
        and tracker errors are recorded locally and never raised to the engine.
        """
        if not self.image_work_enabled:
            return []
        if kind not in {"epoch", "best", "final"}:
            raise ValueError("snapshot kind must be epoch, best, or final")
        if kind == "best" and not self.include_best:
            return []
        if kind == "final" and not self.include_final:
            return []

        loaders = {"train": train_loader, "val": val_loader}
        images: dict[str, Any] = {}
        captions: dict[str, str] = {}
        emitted_keys: list[tuple[str, int, str, str]] = []
        for split in self.splits:
            try:
                examples = self._examples_for(split, loaders[split])
            except BaseException as error:
                self._record_error(error)
                continue
            for index, example in enumerate(examples):
                ledger_key = (split, int(epoch), kind, example.case_hash)
                scheduled_key = (split, int(epoch), "epoch", example.case_hash)
                if ledger_key in self._emitted:
                    continue
                if kind == "best" and scheduled_key in self._emitted:
                    continue
                try:
                    render_image, render_target, render_prediction, slice_indices, render_axis = self._render_example(
                        model,
                        example,
                    )
                    image_key = self._image_key(split, int(epoch), kind, index, len(examples))
                    local_path = self._local_path(split, int(epoch), kind, index, len(examples))
                    if self.local_enabled:
                        from token_mixer.evaluation.visualization import save_slice_visualization

                        save_slice_visualization(
                            render_image,
                            render_target,
                            render_prediction,
                            local_path,
                            slice_index=slice_indices,
                            image_channel=self.image_channel,
                            slice_axis=render_axis,
                        )
                    if self.wandb_enabled:
                        if self.local_enabled:
                            images[image_key] = local_path
                        else:
                            from token_mixer.evaluation.visualization import render_slice_visualization

                            images[image_key] = render_slice_visualization(
                                render_image,
                                render_target,
                                render_prediction,
                                slice_index=slice_indices,
                                image_channel=self.image_channel,
                                slice_axis=render_axis,
                            )
                        captions[image_key] = f"case_hash={example.case_hash}"
                    self._emitted.add(ledger_key)
                    emitted_keys.append(ledger_key)
                except BaseException as error:
                    self._record_error(error)

        if images and self.wandb_enabled:
            try:
                self.tracker.log_images(
                    images,
                    step=int(global_step),
                    captions=captions,
                )
            except BaseException as error:
                self._record_error(error)
        if emitted_keys:
            self._save_ledger()
        return [
            f"{split}/{epoch}/{kind}/{case_hash}"
            for split, epoch, kind, case_hash in emitted_keys
        ]


def build_segmentation_snapshotter(
    config: Mapping[str, Any],
    tracker: Any,
    *,
    device: torch.device | str | None = None,
) -> SegmentationSnapshotter | None:
    """Build snapshot collection only when at least one output gate is open."""
    if not isinstance(config, Mapping):
        raise TypeError("snapshotter configuration must be a mapping")
    settings = _snapshot_settings(config)
    if not _bool_snapshot_config(settings.get("enabled", False), False):
        return None
    local_enabled = _bool_snapshot_config(settings.get("local_enabled", False), False)
    wandb_enabled = _tracker_images_enabled(config, tracker) and callable(
        getattr(tracker, "log_images", None)
    )
    if not local_enabled and not wandb_enabled:
        return None
    return SegmentationSnapshotter(config, tracker, device=device)
