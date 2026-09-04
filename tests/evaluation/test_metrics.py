import math

import pytest
import torch

from token_mixer.evaluation.metrics import (
    dice_by_region,
    hd95_by_region,
    logits_to_regions,
)


def test_threshold_uses_probability_half():
    logits = torch.tensor([[[[[-1.0, 0.0, 1.0, 0.2]]]]])
    result = logits_to_regions(logits, threshold=0.5)
    assert result.flatten().tolist() == [0, 0, 1, 1]


def test_logits_to_regions_rejects_nonfinite_logits():
    logits = torch.tensor([[[[[float("nan"), float("inf"), float("-inf")]]]]])

    with pytest.raises(ValueError, match="finite"):
        logits_to_regions(logits)


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("invalid_input", ["pred", "target"])
def test_region_metrics_reject_nonfinite_pred_and_target_values(
    invalid_value: float, invalid_input: str
):
    prediction = torch.zeros((3, 1, 1, 1))
    target = torch.zeros_like(prediction)
    (prediction if invalid_input == "pred" else target)[0, 0, 0, 0] = invalid_value

    with pytest.raises(ValueError, match="finite"):
        dice_by_region(prediction, target)


def test_dice_uses_canonical_region_order_and_absent_mask_policy():
    target = torch.zeros((3, 2, 2, 2))
    prediction = torch.zeros_like(target)

    target[0, 0, 0, 0] = 1
    prediction[0, 0, 0, 0] = 1
    target[1, 0, 0, 1] = 1

    assert dice_by_region(prediction, target) == {
        "ET": 1.0,
        "TC": 0.0,
        "WT": 1.0,
    }


def test_hd95_uses_voxel_spacing_and_returns_nan_for_one_empty_mask():
    target = torch.zeros((3, 3, 3, 3))
    prediction = torch.zeros_like(target)
    target[0, 1, 1, 1] = 1
    prediction[0, 1, 1, 2] = 1
    target[1, 1, 1, 1] = 1

    scores = hd95_by_region(prediction, target, spacing=(2.0, 3.0, 4.0))

    assert scores["ET"] == 4.0
    assert math.isnan(scores["TC"])
    assert scores["WT"] == 0.0


def test_hd95_uses_opposing_foreground_surfaces():
    target = torch.zeros((3, 9, 9, 9))
    prediction = torch.zeros_like(target)
    target[0, 1:8, 1:8, 1:8] = 1
    prediction[0, 1:8, 1:8, 1:8] = 1
    prediction[0, 3:6, 3:6, 3:6] = 0

    scores = hd95_by_region(prediction, target, spacing=(1.0, 1.0, 1.0))

    assert scores["ET"] == 1.0
    assert scores["TC"] == 0.0
    assert scores["WT"] == 0.0


def test_region_metrics_reject_noncanonical_channel_count():
    masks = torch.zeros((2, 2, 2, 2))

    with pytest.raises(ValueError, match="three region channels"):
        dice_by_region(masks, masks)
