from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np

from .cases import normalize_nonzero
from .labels import to_region_masks


def _config_value(config: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in config:
            return config[key]
        for section in ("data", "dataset", "preprocessing", "transform"):
            nested = config.get(section)
            if isinstance(nested, Mapping) and key in nested:
                return nested[key]
    return default


def _size(value: Any, dimensions: int, name: str) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        result = (int(value),) * dimensions
    else:
        try:
            result = tuple(int(item) for item in value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain {dimensions} positive integers") from exc
    if len(result) != dimensions or any(item <= 0 for item in result):
        raise ValueError(f"{name} must contain {dimensions} positive integers")
    return result


def _as_channel_first_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 4:
        raise ValueError(f"Expected image with four dimensions, got shape {image.shape}")
    has_first_channels = image.shape[0] == 4
    has_last_channels = image.shape[-1] == 4
    if has_first_channels and has_last_channels:
        raise ValueError(f"Ambiguous image channel layout for shape {image.shape}")
    if has_first_channels:
        return image.copy()
    if has_last_channels:
        return np.moveaxis(image, -1, 0).copy()
    raise ValueError(
        "Expected image with four modality channels in first or last dimension, "
        f"got shape {image.shape}"
    )


def _as_region_masks(label: np.ndarray, et_label: int | None = None) -> np.ndarray:
    label = np.asarray(label)
    if label.ndim == 3:
        return to_region_masks(label, et_label=et_label)
    has_first_channels = label.ndim == 4 and label.shape[0] == 3
    has_last_channels = label.ndim == 4 and label.shape[-1] == 3
    if has_first_channels and has_last_channels:
        raise ValueError(f"Ambiguous region-mask channel layout for shape {label.shape}")
    if has_first_channels:
        masks = label
    elif has_last_channels:
        masks = np.moveaxis(label, -1, 0)
    else:
        raise ValueError(
            "Expected label volume with shape (D, H, W) or three region channels, "
            f"got shape {label.shape}"
        )
    if not np.all(np.isin(masks, (0, 1))):
        raise ValueError("Pre-channelized region masks must be binary")
    return masks.astype(np.float32, copy=True)


def _padding(value: Any, name: str = "padding") -> tuple[tuple[int, int], ...]:
    if value is None:
        return ((0, 0),) * 3
    if isinstance(value, (int, np.integer)):
        values = (int(value),) * 3
        return tuple((item, item) for item in values)
    try:
        values = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer or a three-value sequence") from exc
    if len(values) != 3:
        raise ValueError(f"{name} must be an integer or a three-value sequence")
    result: list[tuple[int, int]] = []
    for item in values:
        if isinstance(item, (int, np.integer)):
            before = after = int(item)
        else:
            try:
                before, after = (int(part) for part in item)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} contains an invalid axis padding") from exc
        if before < 0 or after < 0:
            raise ValueError(f"{name} cannot contain negative values")
        result.append((before, after))
    return tuple(result)


def _pad_spatial(
    image: np.ndarray,
    masks: np.ndarray,
    padding: tuple[tuple[int, int], ...],
) -> tuple[np.ndarray, np.ndarray]:
    image_pad = ((0, 0),) + padding
    masks_pad = ((0, 0),) + padding
    return (
        np.pad(image, image_pad, mode="constant"),
        np.pad(masks, masks_pad, mode="constant"),
    )


def _pad_to_size(
    image: np.ndarray,
    masks: np.ndarray,
    target: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    padding = []
    for current, requested in zip(image.shape[1:], target):
        deficit = max(0, requested - current)
        before = deficit // 2
        padding.append((before, deficit - before))
    return _pad_spatial(image, masks, tuple(padding))


def _crop_bounds(
    shape: Sequence[int],
    target: Sequence[int],
    training: bool,
    rng: np.random.Generator,
) -> tuple[slice, ...]:
    bounds: list[slice] = []
    for current, requested in zip(shape, target):
        if requested > current:
            raise ValueError("crop target cannot exceed padded volume")
        maximum_start = current - requested
        start = int(rng.integers(0, maximum_start + 1)) if training else maximum_start // 2
        bounds.append(slice(start, start + requested))
    return tuple(bounds)


def _rng(config: Mapping[str, Any]) -> np.random.Generator:
    configured = _config_value(config, "rng", default=None)
    if isinstance(configured, np.random.Generator):
        return configured
    seed = _config_value(config, "seed", default=None)
    return np.random.default_rng(None if seed is None else int(seed))


def _sample_value(value: Any, rng: np.random.Generator, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    try:
        low, high = (float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError("intensity range must be numeric") from exc
    return float(rng.uniform(low, high))


def _apply_training_augmentation(
    image: np.ndarray,
    masks: np.ndarray,
    config: Mapping[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    flip_axes = _config_value(config, "flip_axes", "random_flip_axes", default=())
    if flip_axes is True:
        flip_axes = (0, 1, 2)
    elif flip_axes is False:
        flip_axes = ()
    elif isinstance(flip_axes, (int, np.integer)):
        flip_axes = (int(flip_axes),)
    else:
        flip_axes = tuple(int(axis) for axis in flip_axes)
    if any(axis not in (0, 1, 2) for axis in flip_axes):
        raise ValueError("flip_axes must contain spatial axes 0, 1, or 2")

    probability = _config_value(config, "flip_probability", default=0.5)
    probabilities = None
    if isinstance(probability, Sequence) and not isinstance(probability, (str, bytes)):
        probabilities = tuple(float(item) for item in probability)
        if len(probabilities) != 3:
            raise ValueError("flip_probability sequence must contain three values")
    for axis in flip_axes:
        axis_probability = (
            probabilities[axis]
            if probabilities is not None
            else float(cast(float, probability))
        )
        if not 0.0 <= axis_probability <= 1.0:
            raise ValueError("flip_probability must be between 0 and 1")
        if rng.random() < axis_probability:
            image = np.flip(image, axis=axis + 1)
            masks = np.flip(masks, axis=axis + 1)

    scale = _config_value(config, "intensity_scale", default=1.0)
    if _config_value(config, "intensity_scale_range", default=None) is not None:
        scale = _config_value(config, "intensity_scale_range")
    scale = _sample_value(scale, rng, 1.0)
    shift = _config_value(config, "intensity_shift", default=0.0)
    if _config_value(config, "intensity_shift_range", default=None) is not None:
        shift = _config_value(config, "intensity_shift_range")
    shift = _sample_value(shift, rng, 0.0)
    jitter = _config_value(config, "intensity_jitter", default=None)
    if jitter is not None:
        jitter = float(jitter)
        scale *= float(rng.uniform(1.0 - jitter, 1.0 + jitter))
    noise_std = float(_config_value(config, "intensity_noise_std", default=0.0))

    nonzero = image != 0
    image = image * scale
    if shift:
        image = np.where(nonzero, image + shift, image)
    if noise_std:
        noise = rng.normal(0.0, noise_std, size=image.shape).astype(np.float32)
        image = np.where(nonzero, image + noise, image)
    return image, masks


def preprocess_volume(
    image: np.ndarray,
    label: np.ndarray,
    config: Mapping[str, Any],
    training: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize, spatially prepare, and augment one BraTS volume.

    Images use ``[modalities, D, H, W]`` and labels use ``[ET, TC, WT, D, H, W]``.
    Raw BraTS labels are converted before any spatial operation.
    """
    image = _as_channel_first_image(image)
    masks = _as_region_masks(label, et_label=_config_value(config, "et_label"))
    if image.shape[1:] != masks.shape[1:]:
        raise ValueError(
            f"Image and label spatial shapes differ: {image.shape[1:]} vs {masks.shape[1:]}"
        )

    if bool(_config_value(config, "normalize", default=True)):
        image = np.stack([normalize_nonzero(channel) for channel in image], axis=0)

    padding = _padding(_config_value(config, "padding", "spatial_padding", default=None))
    image, masks = _pad_spatial(image, masks, padding)
    target = _size(
        _config_value(config, "patch_size", "crop_size", "roi_size", default=None),
        3,
        "patch_size",
    )
    if target is not None:
        image, masks = _pad_to_size(image, masks, target)
        rng = _rng(config)
        bounds = _crop_bounds(image.shape[1:], target, training, rng)
        image = image[(slice(None), *bounds)]
        masks = masks[(slice(None), *bounds)]
    elif training:
        rng = _rng(config)
    else:
        rng = np.random.default_rng(0)

    if training:
        image, masks = _apply_training_augmentation(image, masks, config, rng)

    return np.asarray(image, dtype=np.float32).copy(), np.asarray(masks, dtype=np.float32).copy()


def crop_slice(
    image: np.ndarray,
    label: np.ndarray,
    size: Any,
    training: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Pad and crop a channel-first image plus one 2-D label slice."""
    target = _size(size, 2, "slice_size")
    if target is None:
        return np.asarray(image).copy(), np.asarray(label).copy()
    image = np.asarray(image)
    label = np.asarray(label)
    if image.ndim != 3 or label.ndim != 2 or image.shape[1:] != label.shape:
        raise ValueError("Expected image [C, H, W] and label [H, W] for slice crop")
    image, label_channels = _pad_to_2d_size(image, label[None, ...], target)
    bounds = _crop_bounds(image.shape[1:], target, training, rng)
    return (
        image[(slice(None), *bounds)].astype(np.float32, copy=True),
        label_channels[(slice(None), *bounds)][0].astype(label.dtype, copy=True),
    )


def _pad_to_2d_size(
    image: np.ndarray,
    label: np.ndarray,
    target: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    padding = []
    for current, requested in zip(image.shape[1:], target):
        deficit = max(0, requested - current)
        before = deficit // 2
        padding.append((before, deficit - before))
    return (
        np.pad(image, ((0, 0), *padding), mode="constant"),
        np.pad(label, ((0, 0), *padding), mode="constant"),
    )


def build_monai_swinunetr_transform(
    config: Mapping[str, Any], training: bool = False
):
    """Build an optional dictionary transform for SwinUNETR.

    MONAI is imported only when this adapter is requested. The adapter accepts
    ``{"image": ..., "label": ...}`` or a dictionary containing a ``CaseRecord``
    under ``case`` and always emits canonical three-region labels.
    """
    try:
        from monai.transforms import Transform  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "MONAI is required for build_monai_swinunetr_transform; install the imaging extra"
        ) from exc

    class SwinUNETRDictionaryTransform(Transform):
        def __call__(self, data: Mapping[str, Any]) -> dict[str, Any]:
            if not isinstance(data, Mapping):
                raise TypeError("MONAI adapter expects a mapping")
            result = dict(data)
            if "case" in result:
                from .datasets import load_case_arrays

                image, label = load_case_arrays(result["case"])
            else:
                if "image" not in result or "label" not in result:
                    raise KeyError("MONAI adapter requires image and label keys")
                image, label = result["image"], result["label"]
            result["image"], result["label"] = preprocess_volume(
                image, label, config, training
            )
            return result

    return SwinUNETRDictionaryTransform()


monai_swinunetr_transform = build_monai_swinunetr_transform
