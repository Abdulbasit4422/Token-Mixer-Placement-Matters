import numpy as np
import pytest

from token_mixer.data.labels import (
    REGION_NAMES,
    detect_et_label,
    multiclass_to_regions,
    regions_to_multiclass,
    to_region_masks,
)


def test_label_four_uses_et_tc_wt_order():
    seg = np.array([[[0, 1, 2, 4]]], dtype=np.uint8)
    masks = to_region_masks(seg)
    assert REGION_NAMES == ("ET", "TC", "WT")
    assert masks[:, 0, 0, :].tolist() == [
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0, 1.0],
    ]


def test_label_three_uses_et_tc_wt_semantics():
    seg = np.array([[[0, 1, 2, 3]]], dtype=np.uint8)
    masks = to_region_masks(seg)
    np.testing.assert_array_equal(
        masks[:, 0, 0, :],
        np.array(
            [
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )


def test_label_four_takes_precedence_when_both_et_conventions_occur():
    seg = np.array([[[0, 1, 2, 3, 4]]], dtype=np.uint8)

    assert detect_et_label(seg) == 4
    np.testing.assert_array_equal(
        to_region_masks(seg)[:, 0, 0, :],
        np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )


def test_detect_et_label_rejects_segmentation_without_et_marker():
    seg = np.array([[[0, 1, 2]]], dtype=np.uint8)

    with pytest.raises(ValueError, match="neither 3 nor 4"):
        detect_et_label(seg)


@pytest.mark.parametrize(
    "converter",
    [detect_et_label, multiclass_to_regions, to_region_masks],
)
def test_raw_label_converters_require_three_dimensions(converter):
    with pytest.raises(ValueError, match="Expected 3-D segmentation"):
        converter(np.zeros((2, 2), dtype=np.uint8))


@pytest.mark.parametrize("shape", [(2, 2, 2, 2), (3, 2, 2)])
def test_region_mask_converter_requires_three_region_channels(shape):
    with pytest.raises(ValueError, match="Expected region masks with shape"):
        regions_to_multiclass(np.zeros(shape, dtype=np.float32))


def test_region_round_trip_uses_transunet_priority():
    masks = np.array([
        [[[0, 0, 0, 1]]],
        [[[0, 1, 0, 1]]],
        [[[0, 1, 1, 1]]],
    ], dtype=np.uint8)
    label = regions_to_multiclass(masks)
    assert label[0, 0].tolist() == [0, 2, 3, 1]
    np.testing.assert_array_equal(multiclass_to_regions(label), masks)
