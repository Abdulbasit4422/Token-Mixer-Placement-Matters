from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any, Callable

import numpy as np
import torch

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
) -> dict[str, Any]:
    """Run MONAI sliding-window inference and aggregate full-volume metrics.

    Batches must provide spacing after case IDs, or callers must pass
    ``default_spacing`` explicitly. No implicit unit spacing is used: missing
    physical spacing cannot safely produce physical-looking HD95 values.
    """
    sliding_window_inference = _get_sliding_window_inference()
    resolved_device = _device(device)
    model_states = [module.training for module in model.modules()]
    model.eval()

    case_ids: list[str] = []
    dice_values: dict[str, list[float]] = {region: [] for region in REGION_NAMES}
    hd95_values: dict[str, list[float]] = {region: [] for region in REGION_NAMES}
    excluded: dict[str, int] = {region: 0 for region in REGION_NAMES}
    excluded_case_ids: set[str] = set()

    try:
        with torch.no_grad():
            sample_offset = 0
            for batch in loader:
                image, target, batch_case_ids, batch_spacing = _unpack_batch(batch)
                image_tensor = _as_batched_tensor(image, "image").to(resolved_device)
                target_tensor = _as_batched_tensor(target, "target")
                if image_tensor.shape[0] != target_tensor.shape[0]:
                    raise ValueError("image and target batch sizes must match")
                if target_tensor.shape[1] != len(REGION_NAMES):
                    raise ValueError(
                        f"target must contain {len(REGION_NAMES)} region channels, "
                        f"got shape {tuple(target_tensor.shape)}"
                    )

                raw_logits = sliding_window_inference(
                    inputs=image_tensor,
                    roi_size=tuple(int(size) for size in roi_size),
                    sw_batch_size=int(sw_batch_size),
                    predictor=model,
                    overlap=float(overlap),
                )
                if not isinstance(raw_logits, torch.Tensor):
                    raise TypeError("sliding-window predictor must return a torch.Tensor")
                predictions = logits_to_regions(raw_logits)
                target_tensor = target_tensor.to(device=predictions.device)
                batch_case_ids_list = _as_case_ids(
                    batch_case_ids, image_tensor.shape[0], sample_offset
                )
                spacings = _case_spacings(
                    batch_spacing,
                    image_tensor.shape[0],
                    default_spacing=default_spacing,
                )

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
    return result
