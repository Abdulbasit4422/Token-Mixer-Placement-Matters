from pathlib import Path

import numpy as np
import pytest

from token_mixer.data.cases import CaseRecord, MODALITY_NAMES
from token_mixer.data.datasets import (
    BratsPatchDataset,
    BratsSliceDataset,
    BratsVolumeDataset,
)
from token_mixer.data.transforms import (
    build_monai_swinunetr_transform,
    preprocess_volume,
)


def _make_fixture_case(tmp_path: Path) -> CaseRecord:
    modalities = {
        name: tmp_path / f"{name}.nii.gz" for name in MODALITY_NAMES
    }
    return CaseRecord(
        case_id="CASE001",
        modalities=modalities,
        segmentation=tmp_path / "segmentation.nii.gz",
    )


def _make_fixture_volumes() -> dict[str, np.ndarray]:
    shape = (5, 6, 7)
    grid = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    label = np.zeros(shape, dtype=np.uint8)
    label[1, 1, 1] = 4
    label[2, 2, 2] = 1
    label[3, 3, 3] = 2
    return {
        "t1n.nii.gz": grid,
        "t1c.nii.gz": np.flip(grid, axis=0).copy(),
        "t2w.nii.gz": np.flip(grid, axis=1).copy(),
        "t2f.nii.gz": np.flip(grid, axis=2).copy(),
        "segmentation.nii.gz": label,
    }


def _patch_loader(monkeypatch: pytest.MonkeyPatch, volumes: dict[str, np.ndarray]):
    def load(path: Path) -> np.ndarray:
        return volumes[path.name].copy()

    monkeypatch.setattr("token_mixer.data.datasets.load_nifti", load)


def test_preprocess_volume_uses_canonical_order_and_deterministic_center_crop():
    volumes = _make_fixture_volumes()
    image = np.stack([volumes[f"{name}.nii.gz"] for name in MODALITY_NAMES], axis=0)
    label = volumes["segmentation.nii.gz"]
    config = {"patch_size": (3, 4, 5), "normalize": False}

    first_image, first_masks = preprocess_volume(image, label, config, training=False)
    second_image, second_masks = preprocess_volume(image, label, config, training=False)

    assert first_image.shape == (4, 3, 4, 5)
    assert first_masks.shape == (3, 3, 4, 5)
    np.testing.assert_array_equal(first_image, second_image)
    np.testing.assert_array_equal(first_masks, second_masks)
    np.testing.assert_array_equal(first_image[0], image[0, 1:4, 1:5, 1:6])
    assert first_masks[:, 0, 0, 0].tolist() == [1.0, 1.0, 1.0]


def test_preprocess_volume_uses_configured_et_label_for_et_negative_volume():
    image = np.ones((4, 2, 2, 2), dtype=np.float32)
    label = np.zeros((2, 2, 2), dtype=np.uint8)
    label[0, 0, 0] = 1

    _, masks = preprocess_volume(
        image,
        label,
        {"normalize": False, "et_label": 3},
        training=False,
    )

    assert masks[:, 0, 0, 0].tolist() == [0.0, 1.0, 1.0]


def test_preprocess_volume_applies_configured_training_flip_and_intensity_shift():
    image = np.arange(4 * 3 * 4 * 5, dtype=np.float32).reshape(4, 3, 4, 5)
    label = np.zeros((3, 4, 5), dtype=np.uint8)
    label[0, 0, 0] = 4
    config = {
        "patch_size": (3, 4, 5),
        "normalize": False,
        "flip_axes": (0,),
        "flip_probability": 1.0,
        "intensity_shift": 2.0,
    }

    transformed_image, transformed_masks = preprocess_volume(
        image, label, config, training=True
    )

    flipped_image = np.flip(image, axis=1)
    expected_image = np.where(flipped_image != 0, flipped_image + 2.0, 0.0)
    np.testing.assert_array_equal(transformed_image, expected_image)
    assert transformed_masks[0, -1, 0, 0] == 1.0


def test_preprocess_volume_false_flip_axes_disables_all_flips():
    image = np.arange(4 * 2 * 2 * 2, dtype=np.float32).reshape(4, 2, 2, 2)
    label = np.zeros((2, 2, 2), dtype=np.uint8)
    label[0, 0, 0] = 4
    config = {
        "normalize": False,
        "flip_axes": False,
        "flip_probability": 1.0,
    }

    transformed_image, transformed_masks = preprocess_volume(
        image, label, config, training=True
    )

    np.testing.assert_array_equal(transformed_image, image)
    np.testing.assert_array_equal(transformed_masks[0], label == 4)


def test_preprocess_volume_rejects_ambiguous_image_channel_layout():
    image = np.zeros((4, 2, 2, 4), dtype=np.float32)
    label = np.zeros((2, 2, 4), dtype=np.uint8)
    label[0, 0, 0] = 4

    with pytest.raises(ValueError, match="Ambiguous"):
        preprocess_volume(image, label, {"normalize": False}, training=False)


def test_preprocess_volume_rejects_ambiguous_region_mask_channel_layout():
    image = np.zeros((4, 2, 2, 3), dtype=np.float32)
    masks = np.zeros((3, 2, 2, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="Ambiguous"):
        preprocess_volume(image, masks, {"normalize": False}, training=False)


def test_preprocess_volume_rejects_non_binary_region_masks():
    image = np.zeros((4, 2, 2, 2), dtype=np.float32)
    masks = np.zeros((3, 2, 2, 2), dtype=np.float32)
    masks[0, 0, 0, 0] = 0.5

    with pytest.raises(ValueError, match="binary"):
        preprocess_volume(image, masks, {"normalize": False}, training=False)


def test_brats_patch_and_volume_datasets_return_expected_shapes_and_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    case = _make_fixture_case(tmp_path)
    volumes = _make_fixture_volumes()
    _patch_loader(monkeypatch, volumes)
    config = {"patch_size": (3, 4, 5), "normalize": False}

    patch_image, patch_masks = BratsPatchDataset([case], config, training=False)[0]
    volume_image, volume_masks, case_id = BratsVolumeDataset([case], config)[0]

    assert patch_image.shape == (4, 3, 4, 5)
    assert patch_masks.shape == (3, 3, 4, 5)
    assert volume_image.shape == (4, 3, 4, 5)
    assert volume_masks.shape == (3, 3, 4, 5)
    assert case_id == "CASE001"
    np.testing.assert_array_equal(patch_masks, volume_masks)
    assert patch_masks[:, 0, 0, 0].tolist() == [1.0, 1.0, 1.0]


def test_brats_slice_dataset_returns_four_class_labels_and_configured_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    case = _make_fixture_case(tmp_path)
    volumes = _make_fixture_volumes()
    _patch_loader(monkeypatch, volumes)
    config = {
        "slice_size": (4, 5),
        "slice_axis": 0,
        "normalize": False,
    }

    dataset = BratsSliceDataset([case], config, training=False)
    image, label, case_id = dataset[1]

    assert len(dataset) == 5
    assert image.shape == (4, 4, 5)
    assert label.shape == (4, 5)
    assert label.dtype == np.uint8
    assert set(np.unique(label)).issubset({0, 1, 2, 3})
    assert label[0, 0] == 1
    assert case_id == "CASE001"


def test_brats_slice_dataset_exposes_case_id_without_changing_training_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    case = _make_fixture_case(tmp_path)
    _patch_loader(monkeypatch, _make_fixture_volumes())

    sample = BratsSliceDataset(
        [case],
        {"slice_axis": 0, "normalize": False},
        training=False,
    )[0]

    assert len(sample) == 3
    assert sample[2] == "CASE001"
    assert sample[0].shape == (4, 6, 7)
    assert sample[1].shape == (6, 7)


def test_brats_slice_offsets_use_no_crop_padded_shape_on_configured_axis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    case = _make_fixture_case(tmp_path)
    _patch_loader(monkeypatch, _make_fixture_volumes())
    config = {
        "patch_size": (2, 2, 2),
        "padding": ((1, 2), (2, 3), (4, 5)),
        "slice_axis": 1,
        "normalize": False,
    }

    dataset = BratsSliceDataset([case], config, training=False)

    assert len(dataset) == 11
    image, label, case_id = dataset[10]
    assert image.shape == (4, 8, 16)
    assert label.shape == (8, 16)
    assert case_id == "CASE001"


def test_monai_adapter_is_optional():
    pytest.importorskip("monai")
    transform = build_monai_swinunetr_transform({"patch_size": (3, 4, 5)})
    assert callable(transform)
