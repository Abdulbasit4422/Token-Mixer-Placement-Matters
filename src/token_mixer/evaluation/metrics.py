from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from scipy import ndimage

from token_mixer.data.labels import REGION_NAMES


def _validate_threshold(threshold: float) -> float:
    threshold = float(threshold)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be a finite probability in [0, 1]")
    return threshold


def logits_to_regions(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Apply sigmoid and threshold to channel-first logits."""
    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits must be a torch.Tensor")
    if logits.ndim not in (4, 5):
        raise ValueError(
            "logits must have shape (C, D, H, W) or (B, C, D, H, W), "
            f"got {tuple(logits.shape)}"
        )
    if not torch.isfinite(logits).all().item():
        raise ValueError("logits must contain only finite values")
    threshold = _validate_threshold(threshold)
    # Use a strict comparison so an exactly 0.5 probability remains background.
    return (torch.sigmoid(logits) > threshold).to(dtype=torch.float32)


def _as_region_tensor(value: Any, name: str) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.ndim == 4:
        if tensor.shape[0] != len(REGION_NAMES):
            raise ValueError(
                f"{name} must have three region channels with shape "
                "(3, D, H, W) or (B, 3, D, H, W), "
                f"got {tuple(tensor.shape)}"
            )
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 5:
        if tensor.shape[1] != len(REGION_NAMES):
            raise ValueError(
                f"{name} must have three region channels with shape "
                "(3, D, H, W) or (B, 3, D, H, W), "
                f"got {tuple(tensor.shape)}"
            )
    else:
        raise ValueError(
            f"{name} must have three region channels with shape "
            "(3, D, H, W) or (B, 3, D, H, W), "
            f"got {tuple(tensor.shape)}"
        )
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"{name} must contain only finite values")
    return tensor.detach().to(device="cpu") > 0.5


def _validate_metric_inputs(
    pred: Any, target: Any
) -> tuple[torch.Tensor, torch.Tensor]:
    prediction = _as_region_tensor(pred, "pred")
    reference = _as_region_tensor(target, "target")
    if prediction.shape != reference.shape:
        raise ValueError(
            f"pred and target must have identical shapes, got "
            f"{tuple(prediction.shape)} and {tuple(reference.shape)}"
        )
    return prediction, reference


def _dice_values(prediction: torch.Tensor, reference: torch.Tensor) -> list[list[float]]:
    values: list[list[float]] = [[] for _ in REGION_NAMES]
    for channel in range(len(REGION_NAMES)):
        predicted = prediction[:, channel].reshape(prediction.shape[0], -1)
        target = reference[:, channel].reshape(reference.shape[0], -1)
        intersection = (predicted & target).sum(dim=1).numpy()
        sizes = predicted.sum(dim=1).numpy() + target.sum(dim=1).numpy()
        for overlap, size in zip(intersection, sizes):
            values[channel].append(1.0 if size == 0 else float(2.0 * overlap / size))
    return values


def dice_by_region(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Return macro Dice scores in canonical ``ET``, ``TC``, ``WT`` order."""
    prediction, reference = _validate_metric_inputs(pred, target)
    values = _dice_values(prediction, reference)
    return {
        region: float(np.mean(region_values))
        for region, region_values in zip(REGION_NAMES, values)
    }


def _validate_spacing(spacing: Sequence[float]) -> tuple[float, float, float]:
    try:
        values = tuple(float(value) for value in spacing)
    except (TypeError, ValueError) as exc:
        raise ValueError("spacing must contain three positive finite values") from exc
    if len(values) != 3 or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("spacing must contain three positive finite values")
    return values


def _surface(mask: np.ndarray) -> np.ndarray:
    structure = ndimage.generate_binary_structure(mask.ndim, 1)
    return mask & ~ndimage.binary_erosion(mask, structure=structure, border_value=0)


def _hd95_single(
    predicted: np.ndarray,
    target: np.ndarray,
    spacing: tuple[float, float, float],
) -> float:
    predicted = np.asarray(predicted, dtype=bool)
    target = np.asarray(target, dtype=bool)
    predicted_present = bool(predicted.any())
    target_present = bool(target.any())
    if not predicted_present and not target_present:
        return 0.0
    if predicted_present != target_present:
        return float("nan")

    predicted_surface = _surface(predicted)
    target_surface = _surface(target)
    target_distance = ndimage.distance_transform_edt(~target_surface, sampling=spacing)
    predicted_distance = ndimage.distance_transform_edt(~predicted_surface, sampling=spacing)
    distances = np.concatenate(
        (
            target_distance[predicted_surface],
            predicted_distance[target_surface],
        )
    )
    return float(np.percentile(distances, 95))


def _hd95_values(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    spacing: tuple[float, float, float],
) -> list[list[float]]:
    values: list[list[float]] = [[] for _ in REGION_NAMES]
    for batch_index in range(prediction.shape[0]):
        for channel in range(len(REGION_NAMES)):
            values[channel].append(
                _hd95_single(
                    prediction[batch_index, channel].numpy(),
                    reference[batch_index, channel].numpy(),
                    spacing,
                )
            )
    return values


def _nanmean(values: Sequence[float]) -> float:
    values_array = np.asarray(values, dtype=np.float64)
    if np.all(np.isnan(values_array)):
        return float("nan")
    return float(np.nanmean(values_array))


def hd95_by_region(
    pred: torch.Tensor,
    target: torch.Tensor,
    spacing: tuple[float, float, float],
) -> dict[str, float]:
    """Return spacing-aware 95th-percentile Hausdorff scores per region.

    Cases with one empty mask contribute ``NaN`` and are omitted by the
    per-region ``nanmean`` aggregation. Both-empty masks contribute ``0``.
    """
    prediction, reference = _validate_metric_inputs(pred, target)
    validated_spacing = _validate_spacing(spacing)
    values = _hd95_values(prediction, reference, validated_spacing)
    return {
        region: _nanmean(region_values)
        for region, region_values in zip(REGION_NAMES, values)
    }


def hd95_excluded_by_region(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, int]:
    """Count one-empty cases excluded from each region's HD95 mean."""
    prediction, reference = _validate_metric_inputs(pred, target)
    excluded: dict[str, int] = {}
    for channel, region in enumerate(REGION_NAMES):
        predicted_present = prediction[:, channel].flatten(start_dim=1).any(dim=1)
        target_present = reference[:, channel].flatten(start_dim=1).any(dim=1)
        excluded[region] = int((predicted_present ^ target_present).sum().item())
    return excluded
