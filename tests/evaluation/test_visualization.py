from pathlib import Path

import numpy as np
import pytest

import token_mixer.evaluation.visualization as visualization
from token_mixer.evaluation.visualization import (
    blend_overlay,
    plot_metric_history,
    save_slice_visualization,
)


def test_blend_overlay_returns_rgb_uint8_array_without_loading_data():
    image = np.zeros((2, 2), dtype=np.float32)
    image[0, 0] = 1.0
    masks = np.zeros((3, 2, 2), dtype=np.float32)
    masks[0, 0, 1] = 1.0

    result = blend_overlay(image, masks, alpha=0.5)

    assert result.shape == (2, 2, 3)
    assert result.dtype == np.uint8
    assert result[0, 0].tolist() == [255, 255, 255]
    assert result[0, 1, 2] > result[0, 1, 0]


def test_blend_overlay_uses_canonical_region_order_and_legacy_colors():
    image = np.zeros((1, 3), dtype=np.float32)
    masks = np.zeros((3, 1, 3), dtype=np.float32)
    masks[0, 0, 0] = 1.0  # ET: cyan
    masks[1, 0, 1] = 1.0  # TC: red
    masks[2, 0, 2] = 1.0  # WT: yellow

    result = blend_overlay(image, masks, alpha=1.0)

    np.testing.assert_array_equal(result[0, 0], [25, 229, 255])
    np.testing.assert_array_equal(result[0, 1], [255, 51, 51])
    np.testing.assert_array_equal(result[0, 2], [255, 229, 25])


def test_blend_overlay_rejects_mixed_nonfinite_image_values():
    image = np.array([[np.nan, 1.0]], dtype=np.float32)
    masks = np.zeros((3, 1, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="finite"):
        blend_overlay(image, masks)


@pytest.mark.parametrize("batched_input", ["image", "target", "prediction"])
def test_save_slice_visualization_rejects_multi_item_batches(
    tmp_path: Path, batched_input: str
):
    image = np.zeros((1, 2, 2, 2), dtype=np.float32)
    target = np.zeros((3, 2, 2, 2), dtype=np.float32)
    prediction = np.zeros_like(target)
    if batched_input == "image":
        image = np.zeros((2, 1, 2, 2, 2), dtype=np.float32)
    elif batched_input == "target":
        target = np.zeros((2, 3, 2, 2, 2), dtype=np.float32)
    else:
        prediction = np.zeros((2, 3, 2, 2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="batch size 1"):
        save_slice_visualization(
            image,
            target,
            prediction,
            tmp_path / "multi_item_batch.png",
            slice_index=0,
        )


@pytest.mark.parametrize(
    ("slice_axis", "slice_index"),
    [(0, 1), (1, 2), (2, 3)],
)
def test_save_slice_visualization_selects_canonical_spatial_axis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    slice_axis: int,
    slice_index: int,
):
    image = np.arange(2 * 2 * 3 * 4, dtype=np.float32).reshape(2, 2, 3, 4)
    target = np.arange(3 * 2 * 3 * 4, dtype=np.float32).reshape(3, 2, 3, 4)
    prediction = target + 1000.0
    calls: list[tuple[np.ndarray, np.ndarray]] = []

    def fake_blend_overlay(
        image_slice: np.ndarray, masks_slice: np.ndarray, alpha: float = 0.45
    ) -> np.ndarray:
        del alpha
        calls.append((image_slice.copy(), masks_slice.copy()))
        return np.zeros((*image_slice.shape, 3), dtype=np.uint8)

    monkeypatch.setattr(visualization, "blend_overlay", fake_blend_overlay)

    save_slice_visualization(
        image,
        target,
        prediction,
        tmp_path / f"slice_axis_{slice_axis}.png",
        slice_index=slice_index,
        image_channel=1,
        slice_axis=slice_axis,
    )

    expected_image = np.take(image[1], slice_index, axis=slice_axis)
    expected_target = np.take(target, slice_index, axis=slice_axis + 1)
    expected_prediction = np.take(prediction, slice_index, axis=slice_axis + 1)
    assert len(calls) == 2
    np.testing.assert_array_equal(calls[0][0], expected_image)
    np.testing.assert_array_equal(calls[0][1], expected_target)
    np.testing.assert_array_equal(calls[1][0], expected_image)
    np.testing.assert_array_equal(calls[1][1], expected_prediction)


def test_visualization_functions_write_only_requested_output_paths(tmp_path: Path):
    image = np.ones((4, 4, 3), dtype=np.float32)
    target = np.zeros((3, 4, 4, 3), dtype=np.float32)
    prediction = np.zeros_like(target)
    target[2, 1, 1, 1] = 1.0
    prediction[2, 1, 1, 1] = 1.0

    overlay_path = tmp_path / "case_overlay.png"
    metrics_path = tmp_path / "metrics.png"
    save_slice_visualization(
        image,
        target,
        prediction,
        overlay_path,
        slice_index=1,
        image_channel=0,
    )
    plot_metric_history(
        [{"epoch": 1, "train_loss": 1.0, "val_loss": 0.8, "mean_dice": 0.5}],
        metrics_path,
    )

    assert overlay_path.is_file()
    assert metrics_path.is_file()
    assert overlay_path.stat().st_size > 0
    assert metrics_path.stat().st_size > 0
