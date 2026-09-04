from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import token_mixer.evaluation.inference as inference_module
from token_mixer.evaluation.inference import evaluate_full_volumes


class _ZeroLogitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv3d(1, 3, kernel_size=1)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, image):
        return self.projection(image)


def test_inference_import_does_not_require_monai(monkeypatch):
    import token_mixer.evaluation.inference as inference_module

    monkeypatch.setattr(inference_module, "_MONAI_IMPORT_ATTEMPTED", False)
    assert callable(inference_module.evaluate_full_volumes)


def test_evaluate_full_volumes_uses_sliding_window_and_restores_model_state(
    monkeypatch: pytest.MonkeyPatch,
):
    pytest.importorskip("monai")
    import monai.inferers

    calls = []
    real_inference = monai.inferers.sliding_window_inference

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real_inference(*args, **kwargs)

    monkeypatch.setattr(monai.inferers, "sliding_window_inference", spy)

    model = _ZeroLogitModel().train()
    image = np.ones((1, 1, 4, 4, 4), dtype=np.float32)
    target = np.zeros((1, 3, 4, 4, 4), dtype=np.float32)

    result = evaluate_full_volumes(
        model,
        [(image, target, ["CASE001"])],
        roi_size=(2, 2, 2),
        sw_batch_size=2,
        overlap=0.25,
        device="cpu",
        default_spacing=(1.0, 1.0, 1.0),
    )

    assert model.training is True
    assert result["case_ids"] == ["CASE001"]
    assert result["mean_dice"] == 1.0
    assert result["mean_hd95"] == 0.0
    assert calls
    assert calls[0][1]["roi_size"] == (2, 2, 2)
    assert calls[0][1]["sw_batch_size"] == 2
    assert calls[0][1]["overlap"] == 0.25


def test_evaluate_full_volumes_restores_nested_training_flags(
    monkeypatch: pytest.MonkeyPatch,
):
    def fake_sliding_window(*, inputs, predictor, **_kwargs):
        return predictor(inputs)

    monkeypatch.setattr(
        inference_module,
        "_get_sliding_window_inference",
        lambda: fake_sliding_window,
    )

    model = _ZeroLogitModel().train()
    model.projection.eval()
    image = np.ones((1, 1, 2, 2, 2), dtype=np.float32)
    target = np.zeros((1, 3, 2, 2, 2), dtype=np.float32)

    evaluate_full_volumes(
        model,
        [(image, target, ["CASE001"])],
        roi_size=(2, 2, 2),
        sw_batch_size=1,
        overlap=0.0,
        device="cpu",
        default_spacing=(1.0, 1.0, 1.0),
    )

    assert model.training is True
    assert model.projection.training is False


def _patch_inference_for_spacing(
    monkeypatch: pytest.MonkeyPatch,
    observed_spacings: list[tuple[float, float, float]],
) -> None:
    def fake_sliding_window(*, inputs, predictor, **_kwargs):
        return predictor(inputs)

    def capture_hd95(_prediction, _target, *, spacing):
        observed_spacings.append(spacing)
        return {region: 0.0 for region in inference_module.REGION_NAMES}

    monkeypatch.setattr(inference_module, "_get_sliding_window_inference", lambda: fake_sliding_window)
    monkeypatch.setattr(inference_module, "hd95_by_region", capture_hd95)


@pytest.mark.parametrize(
    ("batch_spacing", "expected"),
    [
        (
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)],
        ),
        (
            torch.tensor([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]),
            [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)],
        ),
    ],
)
def test_evaluate_full_volumes_normalizes_batched_spacing_layouts(
    monkeypatch: pytest.MonkeyPatch,
    batch_spacing: torch.Tensor,
    expected: list[tuple[float, float, float]],
):
    observed_spacings: list[tuple[float, float, float]] = []
    _patch_inference_for_spacing(monkeypatch, observed_spacings)

    image = np.ones((2, 1, 2, 2, 2), dtype=np.float32)
    target = np.zeros((2, 3, 2, 2, 2), dtype=np.float32)
    evaluate_full_volumes(
        _ZeroLogitModel(),
        [(image, target, ["CASE001", "CASE002"], batch_spacing)],
        roi_size=(2, 2, 2),
        sw_batch_size=1,
        overlap=0.0,
        device="cpu",
    )

    assert observed_spacings == expected


@pytest.mark.parametrize(
    "spacing",
    [
        (1.0, 0.0, 1.0),
        (1.0, -1.0, 1.0),
        (1.0, float("nan"), 1.0),
        (1.0, float("inf"), 1.0),
        (1.0, 2.0),
    ],
)
def test_case_spacings_rejects_nonpositive_nonfinite_or_wrong_length(spacing):
    with pytest.raises(ValueError, match="three positive finite values"):
        inference_module._case_spacings(spacing, batch_size=1)


def test_case_spacings_rejects_ambiguous_three_by_three_layout():
    spacing = torch.tensor(
        [[1.0, 4.0, 7.0], [2.0, 5.0, 8.0], [3.0, 6.0, 9.0]]
    )

    with pytest.raises(ValueError, match="ambiguous.*3x3"):
        inference_module._case_spacings(spacing, batch_size=3)


def test_batches_without_spacing_require_explicit_default_spacing(
    monkeypatch: pytest.MonkeyPatch,
):
    observed_spacings: list[tuple[float, float, float]] = []
    _patch_inference_for_spacing(monkeypatch, observed_spacings)

    image = np.ones((1, 1, 2, 2, 2), dtype=np.float32)
    target = np.zeros((1, 3, 2, 2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="spacing.*default_spacing"):
        evaluate_full_volumes(
            _ZeroLogitModel(),
            [(image, target, ["CASE001"])],
            roi_size=(2, 2, 2),
            sw_batch_size=1,
            overlap=0.0,
            device="cpu",
        )


def test_default_spacing_applies_to_batches_without_spacing(monkeypatch: pytest.MonkeyPatch):
    observed_spacings: list[tuple[float, float, float]] = []
    _patch_inference_for_spacing(monkeypatch, observed_spacings)

    image = np.ones((1, 1, 2, 2, 2), dtype=np.float32)
    target = np.zeros((1, 3, 2, 2, 2), dtype=np.float32)
    evaluate_full_volumes(
        _ZeroLogitModel(),
        [(image, target, ["CASE001"])],
        roi_size=(2, 2, 2),
        sw_batch_size=1,
        overlap=0.0,
        device="cpu",
        default_spacing=(2.0, 3.0, 4.0),
    )

    assert observed_spacings == [(2.0, 3.0, 4.0)]


def test_hd95_excluded_cases_counts_unique_cases_across_regions(
    monkeypatch: pytest.MonkeyPatch,
):
    def fake_sliding_window(*, inputs, predictor, **_kwargs):
        return predictor(inputs)

    monkeypatch.setattr(
        inference_module,
        "_get_sliding_window_inference",
        lambda: fake_sliding_window,
    )

    image = np.ones((1, 1, 2, 2, 2), dtype=np.float32)
    target = np.zeros((1, 3, 2, 2, 2), dtype=np.float32)
    target[0, 0, 0, 0, 0] = 1.0
    target[0, 1, 0, 0, 0] = 1.0

    result = evaluate_full_volumes(
        _ZeroLogitModel(),
        [(image, target, ["CASE001"])],
        roi_size=(2, 2, 2),
        sw_batch_size=1,
        overlap=0.0,
        device="cpu",
        default_spacing=(1.0, 1.0, 1.0),
    )

    assert result["hd95_excluded_ET"] == 1
    assert result["hd95_excluded_TC"] == 1
    assert result["hd95_excluded_WT"] == 0
    assert result["hd95_excluded_cases"] == 1

def test_array_only_batches_use_explicit_unit_spacing_default(
    monkeypatch: pytest.MonkeyPatch,
):
    observed_spacings: list[tuple[float, float, float]] = []
    _patch_inference_for_spacing(monkeypatch, observed_spacings)

    image = np.ones((1, 1, 2, 2, 2), dtype=np.float32)
    target = np.zeros((1, 3, 2, 2, 2), dtype=np.float32)
    evaluate_full_volumes(
        _ZeroLogitModel(),
        [(image, target, ["CASE001"])],
        roi_size=(2, 2, 2),
        sw_batch_size=1,
        overlap=0.0,
        device="cpu",
        default_spacing=(1.0, 1.0, 1.0),
    )

    assert observed_spacings == [(1.0, 1.0, 1.0)]
