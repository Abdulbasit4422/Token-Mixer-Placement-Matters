from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


MODALITY_NAMES: tuple[str, str, str, str] = ("t1n", "t1c", "t2w", "t2f")


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    modalities: Mapping[str, Path]
    segmentation: Path


def discover_cases(root: Path) -> list[CaseRecord]:
    """Discover complete cases under ``root/cases`` in canonical order."""
    cases_root = Path(root) / "cases"
    if cases_root.is_symlink() or not cases_root.is_dir():
        return []

    cases: list[CaseRecord] = []
    for case_dir in sorted(
        (
            path
            for path in cases_root.iterdir()
            if not path.is_symlink() and path.is_dir()
        ),
        key=lambda path: path.name,
    ):
        modalities = {
            name: case_dir / f"{name}.nii.gz" for name in MODALITY_NAMES
        }
        segmentation = case_dir / "segmentation.nii.gz"
        required = (*modalities.values(), segmentation)
        if not all(
            not path.is_symlink()
            and path.is_file()
            and path.stat().st_size > 0
            for path in required
        ):
            continue
        cases.append(
            CaseRecord(
                case_id=case_dir.name,
                modalities=modalities,
                segmentation=segmentation,
            )
        )
    return cases


def load_nifti(path: Path) -> np.ndarray:
    """Load NIfTI data as float32 with a path-specific error."""
    path = Path(path)
    try:
        import nibabel as nib

        data = nib.load(str(path)).get_fdata(dtype=np.float32)
        return np.asarray(data, dtype=np.float32)
    except Exception as exc:
        raise RuntimeError(f"Cannot load NIfTI '{path}': {exc}") from exc


def normalize_nonzero(volume: np.ndarray) -> np.ndarray:
    """Z-score nonzero voxels while retaining zero background."""
    volume = np.asarray(volume)
    mask = volume != 0
    normalized = np.zeros_like(volume, dtype=np.float32)
    if not np.any(mask):
        return normalized

    values = volume[mask].astype(np.float32, copy=False)
    mean = values.mean()
    std = values.std()
    if std == 0:
        return normalized
    normalized[mask] = (values - mean) / std
    return normalized
