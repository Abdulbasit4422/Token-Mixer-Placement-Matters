from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .cases import CaseRecord, MODALITY_NAMES, load_nifti
from .labels import regions_to_multiclass
from .transforms import _config_value, _rng, crop_slice, preprocess_volume


def _load_source(source: Any) -> np.ndarray:
    if isinstance(source, np.ndarray):
        return np.asarray(source).copy()
    return load_nifti(Path(source))


def load_case_arrays(case: CaseRecord) -> tuple[np.ndarray, np.ndarray]:
    """Load one case into canonical image and raw-label arrays."""
    missing = [name for name in MODALITY_NAMES if name not in case.modalities]
    if missing:
        raise ValueError(f"Case '{case.case_id}' is missing modalities: {missing}")
    modalities = [_load_source(case.modalities[name]) for name in MODALITY_NAMES]
    if any(volume.ndim != 3 for volume in modalities):
        raise ValueError(f"Case '{case.case_id}' modalities must be 3-D volumes")
    if len({volume.shape for volume in modalities}) != 1:
        raise ValueError(f"Case '{case.case_id}' modalities have different spatial shapes")
    label = _load_source(case.segmentation)
    if label.ndim != 3:
        raise ValueError(f"Case '{case.case_id}' segmentation must be a 3-D volume")
    if label.shape != modalities[0].shape:
        raise ValueError(f"Case '{case.case_id}' image and segmentation shapes differ")
    return np.stack(modalities, axis=0).astype(np.float32), label


def _case_config(config: Mapping[str, Any], index: int) -> dict[str, Any]:
    result = dict(config)
    seed = _config_value(config, "seed", default=None)
    if seed is not None:
        result["seed"] = int(seed) + index
    return result


class BratsPatchDataset:
    """Patch samples with canonical ET/TC/WT region targets."""

    def __init__(
        self,
        cases: Sequence[CaseRecord],
        config: Mapping[str, Any],
        training: bool,
    ) -> None:
        self.cases = list(cases)
        self.config = config
        self.training = bool(training)

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        index = _normalize_index(index, len(self))
        image, label = load_case_arrays(self.cases[index])
        return preprocess_volume(image, label, _case_config(self.config, index), self.training)


class BratsVolumeDataset:
    """Full-volume samples with canonical ET/TC/WT region targets and case ID."""

    def __init__(self, cases: Sequence[CaseRecord], config: Mapping[str, Any]) -> None:
        self.cases = list(cases)
        self.config = config

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, str]:
        index = _normalize_index(index, len(self))
        case = self.cases[index]
        image, label = load_case_arrays(case)
        config = _case_config(self.config, index)
        if _config_value(config, "patch_size", "volume_size", default=None) is None:
            # No volume target means preserve the complete spatial extent.
            processed_image, masks = preprocess_volume(image, label, config, training=False)
        else:
            if _config_value(config, "patch_size", default=None) is None:
                config["patch_size"] = _config_value(config, "volume_size")
            processed_image, masks = preprocess_volume(image, label, config, training=False)
        return processed_image, masks, case.case_id


class BratsSliceDataset:
    """2-D slices with canonical targets and in-memory case identity."""

    def __init__(
        self,
        cases: Sequence[CaseRecord],
        config: Mapping[str, Any],
        training: bool,
    ) -> None:
        self.cases = list(cases)
        self.config = config
        self.training = bool(training)
        self.slice_axis = _slice_axis(_config_value(config, "slice_axis", default=0))
        self._offsets: list[int] | None = None

    def _ensure_offsets(self) -> list[int]:
        if self._offsets is None:
            offsets = [0]
            for case_index, case in enumerate(self.cases):
                image, label = load_case_arrays(case)
                no_crop_config = _slice_preprocess_config(self.config, case_index)
                processed_image, _ = preprocess_volume(
                    image, label, no_crop_config, training=False
                )
                offsets.append(offsets[-1] + processed_image.shape[self.slice_axis + 1])
            self._offsets = offsets
        return self._offsets

    def __len__(self) -> int:
        offsets = self._ensure_offsets()
        return offsets[-1]

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, str]:
        index = _normalize_index(index, len(self))
        offsets = self._ensure_offsets()
        case_index = bisect_right(offsets, index) - 1
        slice_index = index - offsets[case_index]
        image, label = load_case_arrays(self.cases[case_index])

        config = _slice_preprocess_config(self.config, index)
        processed_image, masks = preprocess_volume(image, label, config, self.training)
        image_slice = np.take(processed_image, slice_index, axis=self.slice_axis + 1)
        multiclass = regions_to_multiclass(masks)
        label_slice = np.take(multiclass, slice_index, axis=self.slice_axis)
        cropped_image, cropped_label = crop_slice(
            image_slice,
            label_slice,
            _config_value(self.config, "slice_size", default=None),
            self.training,
            _rng(config),
        )
        return cropped_image, cropped_label, self.cases[case_index].case_id


def _slice_axis(value: Any) -> int:
    if isinstance(value, str):
        names = {"d": 0, "depth": 0, "axial": 0, "h": 1, "height": 1, "coronal": 1, "w": 2, "width": 2, "sagittal": 2}
        try:
            value = names[value.lower()]
        except KeyError as exc:
            raise ValueError("slice_axis must be 0, 1, 2, or a known axis name") from exc
    value = int(value)
    if value not in (0, 1, 2):
        raise ValueError("slice_axis must be 0, 1, or 2")
    return value


def _slice_preprocess_config(config: Mapping[str, Any], index: int) -> dict[str, Any]:
    result = dict(_case_config(config, index))
    crop_keys = ("patch_size", "crop_size", "roi_size")
    for key in crop_keys:
        result.pop(key, None)
    for section in ("data", "dataset", "preprocessing", "transform"):
        nested = result.get(section)
        if isinstance(nested, Mapping):
            nested = dict(nested)
            for key in crop_keys:
                nested.pop(key, None)
            result[section] = nested
    return result


def _normalize_index(index: int, length: int) -> int:
    if not isinstance(index, (int, np.integer)):
        raise TypeError("dataset index must be an integer")
    index = int(index)
    if index < 0:
        index += length
    if not 0 <= index < length:
        raise IndexError("dataset index out of range")
    return index
