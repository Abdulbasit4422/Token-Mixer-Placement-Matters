from __future__ import annotations

import numpy as np


REGION_NAMES: tuple[str, str, str] = ("ET", "TC", "WT")


def _require_label_volume(seg: np.ndarray) -> np.ndarray:
    seg = np.asarray(seg)
    if seg.ndim != 3:
        raise ValueError(f"Expected 3-D segmentation, got shape {seg.shape}")
    return seg


def _require_region_masks(masks: np.ndarray) -> np.ndarray:
    masks = np.asarray(masks)
    if masks.ndim != 4 or masks.shape[0] != len(REGION_NAMES):
        raise ValueError(
            "Expected region masks with shape (3, D, H, W), "
            f"got {masks.shape}"
        )
    return masks


def detect_et_label(seg: np.ndarray) -> int:
    """Detect whether segmentation uses ET label 4 or 3."""
    seg = _require_label_volume(seg)
    if np.any(seg == 4):
        return 4
    if np.any(seg == 3):
        return 3
    raise ValueError("Cannot detect ET label: segmentation contains neither 3 nor 4")


def to_region_masks(seg: np.ndarray, et_label: int | None = None) -> np.ndarray:
    """Convert BraTS labels to binary masks ordered as ET, TC, and WT."""
    seg = _require_label_volume(seg)
    if et_label is None:
        et_label = detect_et_label(seg)
    elif et_label not in (3, 4):
        raise ValueError(f"ET label must be 3 or 4, got {et_label}")

    et = seg == et_label
    tc = (seg == 1) | et
    wt = seg > 0
    return np.stack((et, tc, wt), axis=0).astype(np.float32)


def regions_to_multiclass(masks: np.ndarray) -> np.ndarray:
    """Convert ET/TC/WT masks to canonical labels 0/background, 1/ET, 2/TC, 3/WT."""
    masks = _require_region_masks(masks)
    label = np.zeros(masks.shape[1:], dtype=np.uint8)

    # Write broad regions first so more specific regions take priority.
    label[masks[2] > 0] = 3
    label[masks[1] > 0] = 2
    label[masks[0] > 0] = 1
    return label


def multiclass_to_regions(label: np.ndarray) -> np.ndarray:
    """Convert canonical labels 0/background, 1/ET, 2/TC, 3/WT to ET/TC/WT masks."""
    label = _require_label_volume(label)
    et = label == 1
    tc = (label == 1) | (label == 2)
    wt = label > 0
    return np.stack((et, tc, wt), axis=0).astype(np.float32)
