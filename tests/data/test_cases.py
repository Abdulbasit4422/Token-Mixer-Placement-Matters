from pathlib import Path

import numpy as np
import pytest

from token_mixer.data.cases import discover_cases, load_nifti, normalize_nonzero


def _write_nifti(path: Path, data: np.ndarray) -> None:
    nib = pytest.importorskip("nibabel")
    image = nib.Nifti1Image(np.asarray(data), affine=np.eye(4))
    nib.save(image, str(path))


def _write_complete_case(
    root: Path, case_id: str, zero_byte_name: str | None = None
) -> Path:
    case_dir = root / "cases" / case_id
    case_dir.mkdir(parents=True)
    for name in (
        "t1n.nii.gz",
        "t1c.nii.gz",
        "t2w.nii.gz",
        "t2f.nii.gz",
        "segmentation.nii.gz",
    ):
        path = case_dir / name
        if name == zero_byte_name:
            path.touch()
        else:
            path.write_bytes(b"placeholder")
    return case_dir


def test_discover_cases_requires_all_modalities_and_label(tmp_path: Path):
    case_dir = tmp_path / "cases" / "CASE001"
    case_dir.mkdir(parents=True)
    for name in (
        "t1n.nii.gz",
        "t1c.nii.gz",
        "t2w.nii.gz",
        "t2f.nii.gz",
        "segmentation.nii.gz",
    ):
        (case_dir / name).write_bytes(b"placeholder")

    cases = discover_cases(tmp_path)

    assert len(cases) == 1
    assert cases[0].case_id == "CASE001"
    assert tuple(cases[0].modalities) == ("t1n", "t1c", "t2w", "t2f")


def test_discover_cases_returns_sorted_cases_and_excludes_zero_byte_inputs(
    tmp_path: Path,
):
    _write_complete_case(tmp_path, "CASE010")
    _write_complete_case(tmp_path, "CASE002", zero_byte_name="t2w.nii.gz")
    _write_complete_case(tmp_path, "CASE001")

    cases = discover_cases(tmp_path)

    assert [case.case_id for case in cases] == ["CASE001", "CASE010"]


def test_discover_cases_skips_symlinked_case_directory(tmp_path: Path):
    outside_case = _write_complete_case(tmp_path / "outside", "CASE001")
    cases_root = tmp_path / "cases"
    cases_root.mkdir()
    try:
        (cases_root / "CASE001").symlink_to(outside_case, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks unavailable")

    assert discover_cases(tmp_path) == []


@pytest.mark.parametrize("symlink_name", ["t1n.nii.gz", "segmentation.nii.gz"])
def test_discover_cases_skips_symlinked_required_file(
    tmp_path: Path, symlink_name: str
):
    case_dir = _write_complete_case(tmp_path, "CASE001")
    outside_file = tmp_path / "outside.nii.gz"
    outside_file.write_bytes(b"outside")
    linked_file = case_dir / symlink_name
    linked_file.unlink()
    try:
        linked_file.symlink_to(outside_file)
    except (OSError, NotImplementedError):
        pytest.skip("file symlinks unavailable")

    assert discover_cases(tmp_path) == []


def test_incomplete_case_is_excluded(tmp_path: Path):
    case_dir = tmp_path / "cases" / "CASE001"
    case_dir.mkdir(parents=True)
    (case_dir / "t1n.nii.gz").write_bytes(b"placeholder")
    assert discover_cases(tmp_path) == []


def test_load_nifti_returns_float32_from_real_nifti_file(tmp_path: Path):
    path = tmp_path / "volume.nii.gz"
    source = np.array([[[0.5, 2.25], [4.5, 8.75]]], dtype=np.float64)
    _write_nifti(path, source)

    loaded = load_nifti(path)

    assert loaded.dtype == np.float32
    np.testing.assert_allclose(loaded, source.astype(np.float32))


def test_load_nifti_error_includes_requested_path(tmp_path: Path):
    pytest.importorskip("nibabel")
    path = tmp_path / "missing.nii.gz"

    with pytest.raises(RuntimeError) as exc_info:
        load_nifti(path)

    assert str(path) in str(exc_info.value)


def test_normalize_nonzero_preserves_zeros_and_z_scores_nonzero_values():
    volume = np.array([[[0.0, 1.0, 2.0, 0.0, 3.0]]], dtype=np.float64)

    normalized = normalize_nonzero(volume)

    assert normalized.dtype == np.float32
    np.testing.assert_array_equal(normalized[volume == 0], np.array([0.0, 0.0]))
    np.testing.assert_allclose(
        normalized[volume != 0],
        np.array([-1.2247449, 0.0, 1.2247449], dtype=np.float32),
    )
    np.testing.assert_allclose(normalized[volume != 0].mean(), 0.0, atol=1e-7)
    np.testing.assert_allclose(normalized[volume != 0].std(), 1.0, atol=1e-7)
