import json
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

import token_mixer.data.prepare as prepare_module
import token_mixer.pipelines.prepare_data as pipeline_module
from token_mixer.data.cases import CaseRecord, MODALITY_NAMES, discover_cases, load_nifti
from token_mixer.data.labels import to_region_masks
from token_mixer.data.prepare import prepare_brats
from token_mixer.pipelines.prepare_data import run_prepare


def _write_nifti(path: Path, data: np.ndarray) -> None:
    nib = pytest.importorskip("nibabel")
    image = nib.Nifti1Image(np.asarray(data), affine=np.eye(4))
    nib.save(image, str(path))


def _write_nested_case(source_root: Path, case_id: str, label: np.ndarray) -> Path:
    subject_dir = source_root / "cohort" / case_id
    subject_dir.mkdir(parents=True)
    image = np.ones((2, 2, 2), dtype=np.float32)
    for name in MODALITY_NAMES:
        _write_nifti(subject_dir / f"{case_id}_{name}.nii.gz", image)
    _write_nifti(subject_dir / f"{case_id}_seg.nii.gz", label)
    return subject_dir


def _write_nnunet_case(
    source_root: Path,
    case_id: str,
    label: np.ndarray,
    label_name: str | None = None,
) -> None:
    images_root = source_root / "imagesTr"
    labels_root = source_root / "labelsTr"
    images_root.mkdir(parents=True, exist_ok=True)
    labels_root.mkdir(parents=True, exist_ok=True)
    image = np.ones((2, 2, 2), dtype=np.float32)
    for channel in range(4):
        _write_nifti(images_root / f"{case_id}_000{channel}.nii.gz", image + channel)
    _write_nifti(labels_root / (label_name or f"{case_id}.nii.gz"), label)


def _fake_case_record(data_root: Path, case_id: str = "CASE001") -> CaseRecord:
    case_dir = data_root / "cases" / case_id
    return CaseRecord(
        case_id=case_id,
        modalities={name: case_dir / f"{name}.nii.gz" for name in MODALITY_NAMES},
        segmentation=case_dir / "segmentation.nii.gz",
    )


def test_prepare_brats_creates_safe_canonical_case_from_nested_subject(
    tmp_path: Path,
):
    source_root = tmp_path / "source"
    subject_dir = source_root / "cohort" / "CASE001"
    destination_root = tmp_path / "prepared"
    subject_dir.mkdir(parents=True)

    image = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    label = np.array([0, 1, 2, 4, 0, 1, 2, 4], dtype=np.uint8).reshape(2, 2, 2)
    for name in ("t1n", "t1c", "t2w", "t2f"):
        _write_nifti(subject_dir / f"CASE001_{name}.nii.gz", image)
    _write_nifti(subject_dir / "CASE001_seg.nii.gz", label)

    records = prepare_brats(source_root, destination_root)

    case_dir = destination_root / "cases" / "CASE001"
    assert [record.case_id for record in records] == ["CASE001"]
    assert sorted(path.name for path in case_dir.iterdir()) == [
        "segmentation.nii.gz",
        "t1c.nii.gz",
        "t1n.nii.gz",
        "t2f.nii.gz",
        "t2w.nii.gz",
    ]

    discovered = discover_cases(destination_root)
    assert [record.case_id for record in discovered] == ["CASE001"]
    loaded_label = load_nifti(discovered[0].segmentation)
    np.testing.assert_array_equal(loaded_label, label)
    assert to_region_masks(loaded_label)[0, 0, 1, 1] == 1.0

    marker = case_dir / "do-not-delete.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="CASE001"):
        prepare_brats(source_root, destination_root)
    assert marker.read_text(encoding="utf-8") == "keep"


def test_nifti_discovery_skips_symlinked_files_and_directory_components(
    tmp_path: Path,
):
    source_root = tmp_path / "source"
    local_dir = source_root / "cohort"
    local_dir.mkdir(parents=True)
    local_file = local_dir / "local.nii.gz"
    local_file.write_bytes(b"local")

    outside_file = tmp_path / "outside.nii.gz"
    outside_file.write_bytes(b"outside")
    outside_dir = tmp_path / "outside-dir"
    outside_dir.mkdir()
    (outside_dir / "nested.nii.gz").write_bytes(b"outside")

    try:
        (local_dir / "linked-file.nii.gz").symlink_to(outside_file)
        (source_root / "linked-dir").symlink_to(
            outside_dir, target_is_directory=True
        )
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation unavailable")

    assert prepare_module._nifti_files(source_root) == [local_file]


def test_run_prepare_writes_portable_relative_case_index(tmp_path: Path):
    source_root = tmp_path / "source"
    subject_dir = source_root / "cohort" / "CASE001"
    data_root = tmp_path / "prepared"
    subject_dir.mkdir(parents=True)

    volume = np.ones((2, 2, 2), dtype=np.float32)
    label = np.zeros((2, 2, 2), dtype=np.uint8)
    for name in ("t1n", "t1c", "t2w", "t2f"):
        _write_nifti(subject_dir / f"CASE001_{name}.nii.gz", volume)
    _write_nifti(subject_dir / "CASE001_seg.nii.gz", label)

    run_prepare(
        OmegaConf.create(
            {"paths": {"source_root": str(source_root), "data_root": str(data_root)}}
        )
    )

    index = json.loads((data_root / "case_index.json").read_text(encoding="utf-8"))
    assert index[0]["segmentation"] == "cases/CASE001/segmentation.nii.gz"


def test_prepare_brats_maps_nnunet_modalities_and_normalizes_seg_label(
    tmp_path: Path,
):
    source_root = tmp_path / "source"
    destination_root = tmp_path / "prepared"
    label = np.array([0, 1, 2, 4, 0, 1, 2, 4], dtype=np.uint8).reshape(2, 2, 2)
    _write_nnunet_case(source_root, "CASE001", label, "CASE001.seg.nii.gz")

    records = prepare_brats(source_root, destination_root)

    assert [record.case_id for record in records] == ["CASE001"]
    assert tuple(records[0].modalities) == MODALITY_NAMES
    assert [path.name for path in records[0].modalities.values()] == [
        "t1n.nii.gz",
        "t1c.nii.gz",
        "t2w.nii.gz",
        "t2f.nii.gz",
    ]
    np.testing.assert_array_equal(
        load_nifti(records[0].segmentation), label.astype(np.float32)
    )


def test_prepare_brats_rejects_duplicate_normalized_nnunet_labels(tmp_path: Path):
    source_root = tmp_path / "source"
    destination_root = tmp_path / "prepared"
    label = np.zeros((2, 2, 2), dtype=np.uint8)
    _write_nnunet_case(source_root, "CASE001", label, "CASE001.seg.nii.gz")
    _write_nifti(source_root / "labelsTr" / "CASE001_label.nii.gz", label)

    with pytest.raises(ValueError, match="Duplicate label.*CASE001"):
        prepare_brats(source_root, destination_root)


def test_prepare_brats_warns_about_incomplete_source_case_ids(tmp_path: Path):
    source_root = tmp_path / "source"
    destination_root = tmp_path / "prepared"
    label = np.zeros((2, 2, 2), dtype=np.uint8)
    _write_nested_case(source_root, "CASE001", label)
    incomplete_dir = source_root / "cohort" / "CASE002"
    incomplete_dir.mkdir(parents=True)
    _write_nifti(incomplete_dir / "CASE002_t1n.nii.gz", np.ones((2, 2, 2)))

    with pytest.warns(UserWarning, match="CASE002"):
        records = prepare_brats(source_root, destination_root)

    assert [record.case_id for record in records] == ["CASE001"]


def test_prepare_brats_rejects_nested_file_selected_for_multiple_roles(
    tmp_path: Path,
):
    source_root = tmp_path / "source"
    destination_root = tmp_path / "prepared"
    subject_dir = source_root / "cohort" / "CASE001"
    subject_dir.mkdir(parents=True)
    image = np.ones((2, 2, 2), dtype=np.float32)

    _write_nifti(subject_dir / "CASE001_t1n_seg.nii.gz", image)
    for name in ("t1c", "t2w", "t2f"):
        _write_nifti(subject_dir / f"CASE001_{name}.nii.gz", image)

    with pytest.raises(ValueError, match="matches multiple roles"):
        prepare_brats(source_root, destination_root)

    assert not destination_root.exists()


def test_prepare_brats_rejects_duplicate_nested_case_ids_before_completeness_check(
    tmp_path: Path,
):
    source_root = tmp_path / "source"
    destination_root = tmp_path / "prepared"
    image = np.ones((2, 2, 2), dtype=np.float32)
    incomplete_dir = source_root / "cohort-a" / "CASE001"
    incomplete_dir.mkdir(parents=True)
    _write_nifti(incomplete_dir / "CASE001_t1n.nii.gz", image)

    complete_dir = source_root / "cohort-b" / "CASE001"
    complete_dir.mkdir(parents=True)
    for name in MODALITY_NAMES:
        _write_nifti(complete_dir / f"CASE001_{name}.nii.gz", image)
    _write_nifti(complete_dir / "CASE001_seg.nii.gz", image)

    with pytest.raises(ValueError, match="Duplicate source case ID.*CASE001"):
        prepare_brats(source_root, destination_root)

    assert not destination_root.exists()


@pytest.mark.parametrize(
    "case_id",
    [
        "",
        ".",
        "..",
        "/absolute",
        "C:\\absolute",
        "nested/case",
        "nested\\case",
        "CASE\x00bad",
        "CASE\tbad",
        "CASE 001",
        "CASE:001",
        "CASE.001",
        "CASE001.",
        "CASE001 ",
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "COM1",
        "COM9",
        "LPT1",
        "LPT9",
    ],
)
def test_prepare_brats_rejects_unsafe_source_case_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case_id: str
):
    source_root = tmp_path / "source"
    source_root.mkdir()
    source_files = {
        name: source_root / f"{name}.nii.gz"
        for name in (*MODALITY_NAMES, "segmentation")
    }
    monkeypatch.setattr(
        prepare_module,
        "_discover_source_cases",
        lambda _source_root: {case_id: source_files},
    )

    with pytest.raises(ValueError, match="Unsafe source case ID"):
        prepare_brats(source_root, tmp_path / "prepared")

    assert not (tmp_path / "prepared").exists()


def test_prepare_brats_rejects_destination_that_resolves_outside_cases(
    tmp_path: Path,
):
    source_root = tmp_path / "source"
    destination_root = tmp_path / "prepared"
    _write_nested_case(source_root, "CASE001", np.zeros((2, 2, 2), dtype=np.uint8))
    destination_root.mkdir()
    outside_root = tmp_path / "outside"
    outside_root.mkdir()
    try:
        (destination_root / "cases").symlink_to(outside_root, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks unavailable")

    with pytest.raises(ValueError, match="Destination.*cases"):
        prepare_brats(source_root, destination_root)

    assert not (outside_root / "CASE001").exists()


def test_prepare_brats_preflights_all_conflicts_before_committing_any_case(
    tmp_path: Path,
):
    source_root = tmp_path / "source"
    destination_root = tmp_path / "prepared"
    label = np.zeros((2, 2, 2), dtype=np.uint8)
    _write_nested_case(source_root, "CASE001", label)
    _write_nested_case(source_root, "CASE002", label)
    existing_case = destination_root / "cases" / "CASE002"
    existing_case.mkdir(parents=True)
    marker = existing_case / "old.txt"
    marker.write_text("old", encoding="utf-8")

    with pytest.raises(FileExistsError, match="CASE002"):
        prepare_brats(source_root, destination_root)

    assert not (destination_root / "cases" / "CASE001").exists()
    assert marker.read_text(encoding="utf-8") == "old"


def test_prepare_brats_does_not_commit_earlier_case_when_later_staging_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source_root = tmp_path / "source"
    destination_root = tmp_path / "prepared"
    label = np.zeros((2, 2, 2), dtype=np.uint8)
    _write_nested_case(source_root, "CASE001", label)
    _write_nested_case(source_root, "CASE002", label)
    original_copy = prepare_module._copy_validated_nifti

    def fail_case_two(source: Path, destination: Path) -> None:
        if source.parent.name == "CASE002":
            raise RuntimeError("synthetic staging failure")
        original_copy(source, destination)

    monkeypatch.setattr(prepare_module, "_copy_validated_nifti", fail_case_two)

    with pytest.raises(RuntimeError, match="synthetic staging failure"):
        prepare_brats(source_root, destination_root)

    assert not (destination_root / "cases" / "CASE001").exists()
    assert not (destination_root / "cases" / "CASE002").exists()
    assert not list(destination_root.glob(".prepare-*"))


def test_prepare_brats_rolls_back_overwrite_when_commit_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source_root = tmp_path / "source"
    destination_root = tmp_path / "prepared"
    label = np.zeros((2, 2, 2), dtype=np.uint8)
    _write_nested_case(source_root, "CASE001", label)
    _write_nested_case(source_root, "CASE002", label)
    old_cases = {}
    for case_id in ("CASE001", "CASE002"):
        case_dir = destination_root / "cases" / case_id
        case_dir.mkdir(parents=True)
        marker = case_dir / "old.txt"
        marker.write_text(f"old-{case_id}", encoding="utf-8")
        old_cases[case_id] = marker

    real_replace = prepare_module.os.replace
    failed = False

    def fail_install_of_case_two(source: str | Path, destination: str | Path) -> None:
        nonlocal failed
        destination_path = Path(destination)
        if not failed and destination_path == destination_root / "cases" / "CASE002":
            failed = True
            raise OSError("synthetic commit failure")
        real_replace(source, destination)

    monkeypatch.setattr(prepare_module.os, "replace", fail_install_of_case_two)

    with pytest.raises(OSError, match="synthetic commit failure"):
        prepare_brats(source_root, destination_root, overwrite=True)

    for case_id, marker in old_cases.items():
        assert marker.read_text(encoding="utf-8") == f"old-{case_id}"
        assert sorted(path.name for path in marker.parent.iterdir()) == ["old.txt"]
    assert not list(destination_root.glob(".prepare-*"))
    assert not list(destination_root.glob(".prepare-backup-*"))


def test_prepare_brats_preserves_backup_when_rollback_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source_root = tmp_path / "source"
    destination_root = tmp_path / "prepared"
    label = np.zeros((2, 2, 2), dtype=np.uint8)
    _write_nested_case(source_root, "CASE001", label)
    _write_nested_case(source_root, "CASE002", label)
    cases_root = destination_root / "cases"
    for case_id in ("CASE001", "CASE002"):
        case_dir = cases_root / case_id
        case_dir.mkdir(parents=True)
        (case_dir / "old.txt").write_text(f"old-{case_id}", encoding="utf-8")

    real_replace = prepare_module.os.replace
    failed_install = False

    def fail_commit_and_rollback(source: str | Path, destination: str | Path) -> None:
        nonlocal failed_install
        source_path = Path(source)
        destination_path = Path(destination)
        if not failed_install and destination_path == cases_root / "CASE002":
            failed_install = True
            raise OSError("synthetic commit failure")
        if (
            failed_install
            and source_path.parent.name.startswith(".prepare-backup-")
            and source_path.name == "CASE001"
            and destination_path == cases_root / "CASE001"
        ):
            raise OSError("synthetic rollback failure")
        real_replace(source, destination)

    monkeypatch.setattr(prepare_module.os, "replace", fail_commit_and_rollback)

    with pytest.raises(RuntimeError, match="rollback failed") as exc_info:
        prepare_brats(source_root, destination_root, overwrite=True)

    backup_roots = list(destination_root.glob(".prepare-backup-*"))
    assert len(backup_roots) == 1
    backup_root = backup_roots[0]
    assert str(backup_root) in str(exc_info.value)
    assert (backup_root / "CASE001" / "old.txt").read_text(encoding="utf-8") == (
        "old-CASE001"
    )


def test_run_prepare_replaces_case_index_atomically_and_portably(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    data_root = tmp_path / "prepared"
    data_root.mkdir()
    index_path = data_root / "case_index.json"
    index_path.write_text("old index", encoding="utf-8")
    monkeypatch.setattr(
        pipeline_module,
        "prepare_brats",
        lambda _source_root, _data_root: [_fake_case_record(data_root)],
    )
    real_replace = pipeline_module.os.replace
    replacements: list[tuple[Path, Path]] = []

    def record_replace(source: str | Path, destination: str | Path) -> None:
        replacements.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(pipeline_module.os, "replace", record_replace)

    run_prepare(
        OmegaConf.create(
            {"paths": {"source_root": "unused", "data_root": str(data_root)}}
        )
    )

    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert index[0]["segmentation"] == "cases/CASE001/segmentation.nii.gz"
    assert replacements
    assert replacements[-1][1] == index_path
    assert replacements[-1][0].parent == data_root
    assert not list(data_root.glob(".case_index.json.*"))


def test_run_prepare_accepts_relative_data_root_for_portable_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    data_root = tmp_path / "prepared"
    data_root.mkdir()
    monkeypatch.setattr(
        pipeline_module,
        "prepare_brats",
        lambda _source_root, _data_root: [_fake_case_record(data_root)],
    )

    run_prepare(
        OmegaConf.create(
            {"paths": {"source_root": "unused", "data_root": "prepared"}}
        )
    )

    index = json.loads((data_root / "case_index.json").read_text(encoding="utf-8"))
    assert index[0]["segmentation"] == "cases/CASE001/segmentation.nii.gz"
