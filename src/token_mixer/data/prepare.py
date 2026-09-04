from __future__ import annotations

import os
import re
import shutil
import tempfile
import warnings
from pathlib import Path, PureWindowsPath
from typing import Callable

import numpy as np

from .cases import CaseRecord, MODALITY_NAMES, discover_cases


_NNUNET_IMAGE_RE = re.compile(
    r"^(?P<case_id>.+)_(?P<channel>000[0-3])\.nii(?:\.gz)?$",
    re.IGNORECASE,
)
_NNUNET_CHANNELS: dict[str, str] = dict(
    zip(("0000", "0001", "0002", "0003"), MODALITY_NAMES)
)
_LABEL_SUFFIX_RE = re.compile(r"(?:[._-](?:seg|label|mask))$", re.IGNORECASE)
_SAFE_CASE_ID_RE = re.compile(r"[A-Za-z0-9_-]+")
_WINDOWS_RESERVED_CASE_IDS = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "COM1",
        "COM2",
        "COM3",
        "COM4",
        "COM5",
        "COM6",
        "COM7",
        "COM8",
        "COM9",
        "LPT1",
        "LPT2",
        "LPT3",
        "LPT4",
        "LPT5",
        "LPT6",
        "LPT7",
        "LPT8",
        "LPT9",
    }
)


def prepare_brats(
    source_root: Path, destination_root: Path, overwrite: bool = False
) -> list[CaseRecord]:
    """Copy BraTS cases into the canonical ``cases/<case_id>`` layout.

    Source files are loaded before being saved and the saved files are loaded
    again. Labels are copied without changing their voxel values; label
    convention conversion belongs to :mod:`token_mixer.data.labels`.
    """
    source_root = Path(source_root)
    destination_root = Path(destination_root)
    if not source_root.is_dir():
        raise FileNotFoundError(f"BraTS source root does not exist: {source_root}")

    source_cases = _discover_source_cases(source_root)
    if not source_cases:
        raise ValueError(f"No complete BraTS cases found under {source_root}")

    # Validate all IDs before constructing any destination path.
    for case_id in source_cases:
        _validate_source_case_id(case_id)

    destination_root.mkdir(parents=True, exist_ok=True)
    destination_root = destination_root.resolve()
    cases_root = destination_root / "cases"
    cases_root.mkdir(parents=True, exist_ok=True)
    resolved_cases_root = cases_root.resolve()
    if resolved_cases_root != cases_root:
        raise ValueError(
            "Destination cases root must resolve inside destination root: "
            f"{cases_root}"
        )

    destinations = {
        case_id: _destination_case_path(cases_root, resolved_cases_root, case_id)
        for case_id in source_cases
    }
    conflicts = [
        case_id
        for case_id, destination_case in destinations.items()
        if _path_exists(destination_case)
    ]
    if conflicts and not overwrite:
        raise FileExistsError(
            "Canonical case destinations already exist: "
            + ", ".join(conflicts)
        )

    staging_root = Path(
        tempfile.mkdtemp(prefix=".prepare-", dir=str(destination_root))
    )
    backup_root: Path | None = None
    try:
        for case_id, files in source_cases.items():
            staged_case = staging_root / case_id
            staged_case.mkdir()
            for name in MODALITY_NAMES:
                _copy_validated_nifti(files[name], staged_case / f"{name}.nii.gz")
            _copy_validated_nifti(
                files["segmentation"], staged_case / "segmentation.nii.gz"
            )

        backup_root = Path(
            tempfile.mkdtemp(prefix=".prepare-backup-", dir=str(destination_root))
        )
        _commit_staged_cases(staging_root, backup_root, destinations)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)

    return discover_cases(destination_root)


def _discover_source_cases(source_root: Path) -> dict[str, dict[str, Path]]:
    images_root = source_root / "imagesTr"
    labels_root = source_root / "labelsTr"
    if images_root.exists() or labels_root.exists():
        if not images_root.is_dir() or not labels_root.is_dir():
            raise ValueError(
                "BraTS imagesTr/labelsTr source layout requires both directories "
                f"under {source_root}"
            )
        return _discover_nnunet_cases(images_root, labels_root)
    return _discover_nested_cases(source_root)


def _discover_nnunet_cases(
    images_root: Path, labels_root: Path
) -> dict[str, dict[str, Path]]:
    images: dict[str, dict[str, Path]] = {}
    for path in _nifti_files(images_root):
        match = _NNUNET_IMAGE_RE.fullmatch(path.name)
        if match is None:
            continue
        case_id = match.group("case_id")
        _validate_source_case_id(case_id)
        modality = _NNUNET_CHANNELS[match.group("channel")]
        case_files = images.setdefault(case_id, {})
        if modality in case_files:
            raise ValueError(f"Duplicate {modality} image for case '{case_id}'")
        case_files[modality] = path

    labels: dict[str, Path] = {}
    for path in _nifti_files(labels_root):
        case_id = _label_case_id_from_filename(path)
        _validate_source_case_id(case_id)
        if case_id in labels:
            raise ValueError(f"Duplicate label case ID '{case_id}'")
        labels[case_id] = path

    result: dict[str, dict[str, Path]] = {}
    incomplete_case_ids: list[str] = []
    for case_id in sorted(set(images) | set(labels)):
        case_files = images.get(case_id, {})
        missing = [name for name in MODALITY_NAMES if name not in case_files]
        if missing or case_id not in labels:
            incomplete_case_ids.append(case_id)
            continue
        result[case_id] = {
            **case_files,
            "segmentation": labels[case_id],
        }
    _warn_incomplete_case_ids(incomplete_case_ids)
    return result


def _discover_nested_cases(source_root: Path) -> dict[str, dict[str, Path]]:
    grouped: dict[Path, list[Path]] = {}
    for path in _nifti_files(source_root):
        grouped.setdefault(path.parent, []).append(path)

    subject_dirs = sorted(grouped, key=lambda path: str(path))
    subject_case_ids: dict[str, Path] = {}
    for subject_dir in subject_dirs:
        case_id = subject_dir.name
        _validate_source_case_id(case_id)
        if case_id in subject_case_ids:
            raise ValueError(
                f"Duplicate source case ID '{case_id}' in "
                f"{subject_case_ids[case_id]} and {subject_dir}"
            )
        subject_case_ids[case_id] = subject_dir

    result: dict[str, dict[str, Path]] = {}
    incomplete_case_ids: list[str] = []
    for subject_dir in subject_dirs:
        case_id = subject_dir.name
        files = sorted(grouped[subject_dir], key=lambda path: path.name.lower())
        matches = {
            "t1n": _pick_file(files, "t1n", _is_t1n),
            "t1c": _pick_file(files, "t1c", _is_t1c),
            "t2w": _pick_file(files, "t2w", _is_t2w),
            "t2f": _pick_file(files, "t2f", _is_t2f),
            "segmentation": _pick_file(files, "segmentation", _is_segmentation),
        }
        selected_roles: dict[Path, list[str]] = {}
        for role, path in matches.items():
            if path is not None:
                selected_roles.setdefault(path, []).append(role)
        for path, roles in selected_roles.items():
            if len(roles) > 1:
                raise ValueError(
                    f"Source file '{path.name}' matches multiple roles: "
                    + ", ".join(roles)
                )

        if any(path is None for path in matches.values()):
            incomplete_case_ids.append(subject_dir.name)
            continue

        if case_id in result:
            raise ValueError(f"Duplicate source case ID '{case_id}'")
        result[case_id] = {name: path for name, path in matches.items() if path is not None}
    _warn_incomplete_case_ids(incomplete_case_ids)
    return dict(sorted(result.items()))


def _pick_file(
    files: list[Path], label: str, predicate: Callable[[str], bool]
) -> Path | None:
    matches = [path for path in files if predicate(path.name.lower())]
    if len(matches) > 1:
        names = ", ".join(path.name for path in matches)
        raise ValueError(f"Ambiguous {label} files: {names}")
    return matches[0] if matches else None


def _is_t1n(name: str) -> bool:
    return "t1n" in name or (
        "t1" in name and not any(token in name for token in ("t1c", "t1ce", "t1gd"))
    )


def _is_t1c(name: str) -> bool:
    return any(token in name for token in ("t1c", "t1ce", "t1gd"))


def _is_t2w(name: str) -> bool:
    return "t2w" in name or (
        bool(re.search(r"(?<![a-z0-9])t2(?![a-z0-9])", name))
        and "t2f" not in name
        and "flair" not in name
    )


def _is_t2f(name: str) -> bool:
    return "t2f" in name or "flair" in name


def _is_segmentation(name: str) -> bool:
    return "seg" in name or "mask" in name


def _nifti_files(root: Path) -> list[Path]:
    if root.is_symlink():
        return []
    return sorted(
        (
            path
            for path in root.rglob("*")
            if not path.is_symlink()
            and path.is_file()
            and _is_nifti(path)
            and all(
                not parent.is_symlink() for parent in path.parents if parent != root
            )
        ),
        key=lambda path: str(path).lower(),
    )


def _is_nifti(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def _case_id_from_filename(path: Path) -> str:
    name = path.name
    if name.lower().endswith(".nii.gz"):
        return name[:-7]
    if name.lower().endswith(".nii"):
        return name[:-4]
    raise ValueError(f"Not a NIfTI file: {path}")


def _label_case_id_from_filename(path: Path) -> str:
    return _LABEL_SUFFIX_RE.sub("", _case_id_from_filename(path))


def _validate_source_case_id(case_id: str) -> None:
    if not isinstance(case_id, str) or not case_id:
        raise ValueError(f"Unsafe source case ID: {case_id!r}")

    if (
        case_id in {".", ".."}
        or "/" in case_id
        or "\\" in case_id
        or any(ord(char) < 32 or ord(char) == 127 for char in case_id)
        or case_id[-1] in ". "
        or _SAFE_CASE_ID_RE.fullmatch(case_id) is None
        or case_id.upper() in _WINDOWS_RESERVED_CASE_IDS
        or Path(case_id).is_absolute()
        or PureWindowsPath(case_id).is_absolute()
        or bool(PureWindowsPath(case_id).drive)
    ):
        raise ValueError(f"Unsafe source case ID: {case_id!r}")


def _destination_case_path(
    cases_root: Path, resolved_cases_root: Path, case_id: str
) -> Path:
    destination_case = cases_root / case_id
    resolved_destination = destination_case.resolve(strict=False)
    try:
        relative_destination = resolved_destination.relative_to(resolved_cases_root)
    except ValueError as exc:
        raise ValueError(
            "Destination case is outside destination cases root: "
            f"{destination_case}"
        ) from exc
    if not relative_destination.parts:
        raise ValueError(
            "Destination case must be below destination cases root: "
            f"{destination_case}"
        )
    return destination_case


def _path_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def _commit_staged_cases(
    staging_root: Path,
    backup_root: Path,
    destinations: dict[str, Path],
) -> None:
    backed_up: list[tuple[Path, Path]] = []
    installed: list[Path] = []
    try:
        for case_id, destination_case in destinations.items():
            if not _path_exists(destination_case):
                continue
            backup_case = backup_root / case_id
            os.replace(destination_case, backup_case)
            backed_up.append((destination_case, backup_case))

        for case_id, destination_case in destinations.items():
            os.replace(staging_root / case_id, destination_case)
            installed.append(destination_case)
    except Exception as commit_error:
        rollback_errors: list[Exception] = []
        for destination_case in reversed(installed):
            try:
                _remove_path(destination_case)
            except Exception as rollback_error:
                rollback_errors.append(rollback_error)
        for destination_case, backup_case in reversed(backed_up):
            try:
                os.replace(backup_case, destination_case)
            except Exception as rollback_error:
                rollback_errors.append(rollback_error)
        if rollback_errors:
            details = "; ".join(str(error) for error in rollback_errors)
            raise RuntimeError(
                "Preparation commit failed and rollback failed; "
                f"backup preserved at {backup_root}: {details}"
            ) from commit_error
        shutil.rmtree(backup_root, ignore_errors=True)
        raise
    else:
        shutil.rmtree(backup_root, ignore_errors=True)


def _warn_incomplete_case_ids(case_ids: list[str]) -> None:
    if case_ids:
        warnings.warn(
            f"Skipping {len(case_ids)} incomplete source case(s): "
            + ", ".join(case_ids),
            UserWarning,
            stacklevel=3,
        )


def _copy_validated_nifti(source: Path, destination: Path) -> None:
    try:
        import nibabel as nib  # type: ignore[import-not-found]

        image = nib.load(str(source))
        np.asarray(image.dataobj)
        nib.save(image, str(destination))
        saved = nib.load(str(destination))
        np.asarray(saved.dataobj)
    except Exception as exc:
        raise RuntimeError(f"Cannot validate NIfTI '{source}': {exc}") from exc
