from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from numbers import Number
from pathlib import Path
from typing import Any

import numpy as np


# Region masks follow the package-wide canonical order: ET, TC, WT.
REGION_NAMES = ("ET", "TC", "WT")
_REGION_NAMES = REGION_NAMES
_REGION_COLORS = {
    "TC": np.array([1.0, 0.2, 0.2], dtype=np.float32),
    "WT": np.array([1.0, 0.9, 0.1], dtype=np.float32),
    "ET": np.array([0.1, 0.9, 1.0], dtype=np.float32),
}
REGION_COLORS = _REGION_COLORS


def _array(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.size == 0:
        raise ValueError(f"{name} must not be empty")
    return result


def _validate_alpha(alpha: float) -> float:
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    return alpha


def blend_overlay(image: Any, masks: Any, alpha: float = 0.45) -> np.ndarray:
    """Blend a grayscale slice and three binary masks into an RGB uint8 image."""
    image_array = _array(image, "image")
    masks_array = _array(masks, "masks")
    if image_array.ndim != 2:
        raise ValueError(f"image must be a 2-D slice, got shape {image_array.shape}")
    if masks_array.ndim != 3 or masks_array.shape[0] != len(_REGION_NAMES):
        raise ValueError(
            "masks must have shape (3, H, W) in ET/TC/WT order, "
            f"got {masks_array.shape}"
        )
    if masks_array.shape[1:] != image_array.shape:
        raise ValueError(
            f"image and masks spatial shapes must match, got "
            f"{image_array.shape} and {masks_array.shape[1:]}"
        )
    alpha = _validate_alpha(alpha)

    image_float = np.asarray(image_array, dtype=np.float32)
    if not np.isfinite(image_float).all():
        raise ValueError("image must contain finite values")
    minimum = float(np.nanmin(image_float))
    maximum = float(np.nanmax(image_float))
    if not np.isfinite(minimum) or not np.isfinite(maximum):
        raise ValueError("image must contain finite values")
    if maximum == minimum:
        image_uint8 = np.zeros(image_float.shape, dtype=np.uint8)
    else:
        image_uint8 = np.clip((image_float - minimum) / (maximum - minimum) * 255.0, 0, 255).astype(
            np.uint8
        )
    rgb = np.repeat(image_uint8[..., None], 3, axis=-1).astype(np.float32)

    # Draw broad WT first so more specific TC and ET colors remain visible.
    for index in (2, 1, 0):
        mask = masks_array[index] > 0.5
        color = _REGION_COLORS[_REGION_NAMES[index]] * 255.0
        for channel in range(3):
            rgb[..., channel] = np.where(
                mask,
                (1.0 - alpha) * rgb[..., channel] + alpha * color[channel],
                rgb[..., channel],
            )
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _first_volume(value: Any, name: str, channels: int | None = None) -> np.ndarray:
    result = _array(value, name)
    if result.ndim == 5:
        if result.shape[0] != 1:
            raise ValueError(
                f"{name} batched input must have batch size 1, got shape {result.shape}"
            )
        result = result[0]
    if channels is not None and result.ndim == 4 and result.shape[0] != channels:
        raise ValueError(f"{name} must have {channels} channels, got shape {result.shape}")
    return result


def _validate_slice_axis(slice_axis: int) -> int:
    slice_axis = int(slice_axis)
    if slice_axis not in (0, 1, 2):
        raise ValueError("slice_axis must be 0, 1, or 2")
    return slice_axis


def _slice_indices(slice_index: int | Sequence[int], size: int) -> tuple[int, ...]:
    if isinstance(slice_index, Sequence) and not isinstance(
        slice_index, (str, bytes, bytearray)
    ):
        values = tuple(int(index) for index in slice_index)
    else:
        values = (int(slice_index),)
    if not values:
        raise ValueError("slice_index must contain at least one index")
    if any(index < 0 or index >= size for index in values):
        raise IndexError(f"slice_index is outside image axis with size {size}")
    return values


def _region_volume(masks: Any) -> np.ndarray:
    result = _array(masks, "masks")
    if result.ndim == 5:
        if result.shape[0] != 1:
            raise ValueError(
                f"masks batched input must have batch size 1, got shape {result.shape}"
            )
        result = result[0]
    if result.ndim == 3 and result.shape[0] == len(_REGION_NAMES):
        result = result[:, None, ...]
    if result.ndim != 4 or result.shape[0] != len(_REGION_NAMES):
        raise ValueError(
            "masks must have shape (3, D, H, W), or a single 2-D slice with "
            f"three channels; got {result.shape}"
        )
    return result


def region_aware_slice_indices(
    masks: Any, *, slice_axis: int = 0
) -> tuple[int, ...]:
    """Choose deterministic slices that expose every available ET/TC/WT region.

    A single slice is preferred when it contains all regions. If no such slice
    exists, the earliest slice containing each region is returned as a small
    region-aware montage. Ties are resolved by axis order, so selection does
    not depend on loader or model state.
    """
    slice_axis = _validate_slice_axis(slice_axis)
    volume = _region_volume(masks)
    spatial_axis = slice_axis + 1
    present_axes = tuple(
        axis for axis in range(1, volume.ndim) if axis != spatial_axis
    )
    present = np.any(volume > 0.5, axis=present_axes)
    coverage = np.sum(present, axis=0)
    best_coverage = int(np.max(coverage)) if coverage.size else 0
    if best_coverage == len(_REGION_NAMES):
        return (int(np.flatnonzero(coverage == best_coverage)[0]),)

    selected: set[int] = set()
    for region_index in range(len(_REGION_NAMES)):
        positions = np.flatnonzero(present[region_index])
        if positions.size:
            selected.add(int(positions[0]))
    if not selected and coverage.size:
        selected.add(int(np.flatnonzero(coverage == best_coverage)[0]))
    return tuple(sorted(selected))


def _image_slice(
    image: Any,
    image_channel: int,
    slice_index: int,
    slice_axis: int,
) -> np.ndarray:
    volume = _first_volume(image, "image")
    if volume.ndim == 4:
        if not 0 <= image_channel < volume.shape[0]:
            raise IndexError(f"image_channel {image_channel} is outside image shape {volume.shape}")
        volume = volume[image_channel]
    if volume.ndim != 3:
        raise ValueError(
            "image must have shape (D, H, W), (C, D, H, W), or a batched equivalent"
        )
    if not 0 <= slice_index < volume.shape[slice_axis]:
        raise IndexError(
            f"slice_index {slice_index} is outside image axis {slice_axis} "
            f"with size {volume.shape[slice_axis]}"
        )
    return np.take(volume, slice_index, axis=slice_axis)


def _mask_slice(masks: Any, name: str, slice_index: int, slice_axis: int) -> np.ndarray:
    volume = _region_volume(masks)
    spatial_axis = slice_axis + 1
    if not 0 <= slice_index < volume.shape[spatial_axis]:
        raise IndexError(
            f"slice_index {slice_index} is outside {name} axis {slice_axis} "
            f"with size {volume.shape[spatial_axis]}"
        )
    return np.take(volume, slice_index, axis=spatial_axis)


def _validate_visualization_inputs(
    image: Any,
    target: Any,
    prediction: Any,
    *,
    slice_index: int | Sequence[int],
    image_channel: int,
    slice_axis: int,
) -> tuple[int, ...]:
    slice_axis = _validate_slice_axis(slice_axis)
    image_volume = _first_volume(image, "image")
    target_volume = _region_volume(target)
    prediction_volume = _region_volume(prediction)
    if target_volume.shape != prediction_volume.shape:
        raise ValueError(
            "target and prediction must have matching region-mask shapes, got "
            f"{target_volume.shape} and {prediction_volume.shape}"
        )
    for name, value in (("target", target_volume), ("prediction", prediction_volume)):
        try:
            finite = np.isfinite(value).all()
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain finite values") from exc
        if not finite:
            raise ValueError(f"{name} must contain finite values")

    if image_volume.ndim == 4:
        if not 0 <= image_channel < image_volume.shape[0]:
            raise ValueError(
                f"image_channel {image_channel} is outside image shape {image_volume.shape}"
            )
        image_shape = image_volume.shape[1:]
    elif image_volume.ndim == 3:
        image_shape = image_volume.shape
    else:
        raise ValueError(
            "image must have shape (D, H, W), (C, D, H, W), or a batched equivalent"
        )
    try:
        image_finite = np.isfinite(image_volume).all()
    except (TypeError, ValueError) as exc:
        raise ValueError("image must contain finite values") from exc
    if not image_finite:
        raise ValueError("image must contain finite values")
    if tuple(image_shape) != tuple(target_volume.shape[1:]):
        raise ValueError(
            "image, target, and prediction spatial shapes must match, got "
            f"{image_shape} and {target_volume.shape[1:]}"
        )
    return _slice_indices(slice_index, int(image_shape[slice_axis]))


def _grayscale_rgb(image_slice: np.ndarray) -> np.ndarray:
    image_float = np.asarray(image_slice, dtype=np.float32)
    if image_float.ndim != 2 or image_float.size == 0:
        raise ValueError("image slice must be a non-empty 2-D array")
    if not np.isfinite(image_float).all():
        raise ValueError("image must contain finite values")
    minimum = float(np.min(image_float))
    maximum = float(np.max(image_float))
    if maximum == minimum:
        image_uint8 = np.zeros(image_float.shape, dtype=np.uint8)
    else:
        image_uint8 = np.clip(
            (image_float - minimum) / (maximum - minimum) * 255.0, 0, 255
        ).astype(np.uint8)
    return np.repeat(image_uint8[..., None], 3, axis=-1)


def render_slice_visualization(
    image: Any,
    target: Any,
    prediction: Any,
    *,
    slice_index: int | Sequence[int],
    image_channel: int = 0,
    slice_axis: int = 0,
    alpha: float = 0.45,
) -> np.ndarray:
    """Render one or more canonical input/target/prediction RGB montages.

    This is the in-memory counterpart of :func:`save_slice_visualization`.
    It deliberately performs no plotting-library import, which lets the W&B
    adapter forward image arrays without creating local files.
    """
    indices = _validate_visualization_inputs(
        image,
        target,
        prediction,
        slice_index=slice_index,
        image_channel=image_channel,
        slice_axis=slice_axis,
    )
    rows: list[np.ndarray] = []
    for index in indices:
        image_slice = _image_slice(image, image_channel, index, slice_axis)
        target_slice = _mask_slice(target, "target", index, slice_axis)
        prediction_slice = _mask_slice(prediction, "prediction", index, slice_axis)
        rows.append(
            np.concatenate(
                (
                    _grayscale_rgb(image_slice),
                    blend_overlay(image_slice, target_slice, alpha=alpha),
                    blend_overlay(image_slice, prediction_slice, alpha=alpha),
                ),
                axis=1,
            )
        )
    width = {row.shape[1] for row in rows}
    if len(width) != 1:
        raise ValueError("all rendered slices must have the same spatial width")
    return np.concatenate(rows, axis=0).astype(np.uint8, copy=False)


def _output_path(value: str | Path) -> Path:
    path = Path(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _load_pyplot():
    import matplotlib

    if "MPLBACKEND" not in os.environ:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_slice_visualization(
    image: Any,
    target: Any,
    prediction: Any,
    output_path: str | Path,
    *,
    slice_index: int | Sequence[int],
    image_channel: int = 0,
    slice_axis: int = 0,
    alpha: float = 0.45,
) -> Path:
    """Save one input/target/prediction slice without loading or evaluating data.

    Volumes use channel-first ``[C, D, H, W]`` and ``[3, D, H, W]`` layouts.
    ``slice_axis`` selects one of the spatial axes ``D/H/W`` by index ``0/1/2``.
    """
    indices = _validate_visualization_inputs(
        image,
        target,
        prediction,
        slice_index=slice_index,
        image_channel=image_channel,
        slice_axis=slice_axis,
    )
    path = _output_path(output_path)

    plt = _load_pyplot()
    import matplotlib.patches as patches

    figure, axes = plt.subplots(
        len(indices), 3, figsize=(15, 5 * len(indices)), squeeze=False
    )
    try:
        for row, index in enumerate(indices):
            image_slice = _image_slice(image, image_channel, index, slice_axis)
            target_slice = _mask_slice(target, "target", index, slice_axis)
            prediction_slice = _mask_slice(prediction, "prediction", index, slice_axis)
            axes[row, 0].imshow(image_slice, cmap="gray")
            axes[row, 0].set_title("Input MRI")
            axes[row, 1].imshow(blend_overlay(image_slice, target_slice, alpha=alpha))
            axes[row, 1].set_title("Ground Truth")
            axes[row, 2].imshow(
                blend_overlay(image_slice, prediction_slice, alpha=alpha)
            )
            axes[row, 2].set_title("Prediction")
        legend = [
            patches.Patch(color=_REGION_COLORS[name], label=name) for name in _REGION_NAMES
        ]
        axes[0, 1].legend(handles=legend, loc="lower right", fontsize=8)
        axes[0, 2].legend(handles=legend, loc="lower right", fontsize=8)
        for axis in axes.flat:
            axis.axis("off")
        figure.tight_layout()
        figure.savefig(path, bbox_inches="tight", dpi=150)
    finally:
        plt.close(figure)
    return path


def _history_rows(history: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    rows = list(history)
    if any(not isinstance(row, Mapping) for row in rows):
        raise TypeError("history must contain mapping rows")
    return rows


def plot_metric_history(
    history: Sequence[Mapping[str, Any]],
    output_path: str | Path,
) -> Path:
    """Plot scalar loss/metric history and save it to ``output_path``."""
    rows = _history_rows(history)
    path = _output_path(output_path)

    x_values = [row.get("epoch", index + 1) for index, row in enumerate(rows)]
    numeric_keys: list[str] = []
    for row in rows:
        for key, value in row.items():
            if key in numeric_keys or key in {"epoch", "phase", "phase_index", "phase_epoch"}:
                continue
            if isinstance(value, (Number, np.number)) and not isinstance(value, bool):
                numeric_keys.append(str(key))

    plt = _load_pyplot()

    figure, axis = plt.subplots(figsize=(9, 5))
    try:
        for key in numeric_keys:
            values = [row.get(key, np.nan) for row in rows]
            values = [float(value) if isinstance(value, (Number, np.number)) else np.nan for value in values]
            axis.plot(x_values, values, marker="o", linewidth=1.5, label=key)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Value")
        axis.set_title("Metric History")
        axis.grid(alpha=0.25)
        if numeric_keys:
            axis.legend()
        figure.tight_layout()
        figure.savefig(path, bbox_inches="tight", dpi=150)
    finally:
        plt.close(figure)
    return path
