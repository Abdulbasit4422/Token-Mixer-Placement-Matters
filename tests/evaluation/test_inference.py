import importlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import token_mixer.evaluation.inference as inference_module
from token_mixer.evaluation.inference import (
    build_segmentation_snapshotter,
    evaluate_full_volumes,
    infer_regions,
    sliding_window_count,
)


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


def test_snapshotter_gate_disables_image_work_before_loader_access():
    class _ExplodingLoader:
        def __iter__(self):
            raise AssertionError("disabled snapshots iterated loader")

    snapshotter = build_segmentation_snapshotter(
        {
            "visualization": {
                "segmentation_snapshots": {
                    "enabled": True,
                    "local_enabled": False,
                }
            },
            "tracking": {"log_images": False},
        },
        tracker=object(),
        device="cpu",
    )

    assert snapshotter is None


def test_disabled_tracking_mode_blocks_wandb_snapshot_work(tmp_path):
    config = _snapshot_config(tmp_path, local_enabled=False, log_images=True)
    config["tracking"] = {
        "enabled": False,
        "mode": "disabled",
        "log_images": True,
    }

    assert (
        build_segmentation_snapshotter(
            config,
            tracker=_ImageTracker(),
            device="cpu",
        )
        is None
    )


def test_infer_regions_uses_direct_two_dimensional_model_path():
    class _TwoDModel(nn.Module):
        def forward(self, image):
            return torch.ones(image.shape[0], 3, *image.shape[-2:])

    image = torch.zeros(1, 4, 4, 4)
    prediction = infer_regions(
        _TwoDModel(),
        image,
        device="cpu",
        native_3d=False,
    )

    assert tuple(prediction.shape) == (1, 3, 4, 4)
    assert prediction.all()


class _ImageTracker:
    def __init__(self):
        self.calls = []

    def log_images(self, images, *, step, captions=None):
        self.calls.append((dict(images), step, dict(captions or {})))


def _snapshot_loader():
    image = torch.ones(1, 4, 4, 4)
    target = torch.zeros(1, 3, 4, 4)
    target[:, :, 1, 1] = 1.0
    return DataLoader(TensorDataset(image, target), batch_size=1, shuffle=False)


def _snapshot_volume_loader():
    image = torch.ones(1, 4, 4, 4, 4)
    target = torch.zeros(1, 3, 4, 4, 4)
    target[:, :, 1, 1, 1] = 1.0
    return DataLoader(TensorDataset(image, target), batch_size=1, shuffle=False)


def _snapshot_config(tmp_path, *, local_enabled, log_images):
    return {
        "device": "cpu",
        "visualization": {
            "segmentation_snapshots": {
                "enabled": True,
                "snapshot_interval_epochs": 10,
                "include_best": True,
                "include_final": True,
                "splits": ["train", "val"],
                "sample_count": 1,
                "axis": 0,
                "image_channel": 3,
                "local_enabled": local_enabled,
                "output_dir": str(tmp_path),
            }
        },
        "tracking": {
            "enabled": bool(log_images),
            "mode": "offline" if log_images else "disabled",
            "log_images": log_images,
        },
    }


def test_local_only_snapshots_write_pngs_without_importing_wandb(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    real_import = importlib.import_module

    def fail_wandb(name, *args, **kwargs):
        if name == "wandb":
            raise AssertionError("local-only snapshots imported wandb")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("importlib.import_module", fail_wandb)
    loader = _snapshot_loader()
    snapshotter = build_segmentation_snapshotter(
        _snapshot_config(tmp_path, local_enabled=True, log_images=False),
        tracker=_ImageTracker(),
        device="cpu",
    )
    assert snapshotter is not None

    snapshotter.snapshot(
        model=nn.Conv2d(4, 3, kernel_size=1),
        train_loader=loader,
        val_loader=loader,
        epoch=10,
        global_step=7,
    )

    assert (tmp_path / "train" / "epoch_0010.png").is_file()
    assert (tmp_path / "val" / "epoch_0010.png").is_file()


def test_wandb_only_snapshots_forward_arrays_and_hashed_captions_without_plotting(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    real_import = __import__

    def fail_matplotlib(name, *args, **kwargs):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise AssertionError("W&B-only snapshots imported plotting code")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fail_matplotlib)
    tracker = _ImageTracker()
    loader = _snapshot_loader()
    snapshotter = build_segmentation_snapshotter(
        _snapshot_config(tmp_path, local_enabled=False, log_images=True),
        tracker=tracker,
        device="cpu",
    )
    assert snapshotter is not None

    snapshotter.snapshot(
        model=nn.Conv2d(4, 3, kernel_size=1),
        train_loader=loader,
        val_loader=loader,
        epoch=10,
        global_step=7,
    )

    assert len(tracker.calls) == 1
    images, step, captions = tracker.calls[0]
    assert step == 7
    assert set(images) == {
        "segmentation/train/epoch_0010",
        "segmentation/val/epoch_0010",
    }
    assert all(isinstance(image, np.ndarray) for image in images.values())
    assert all("case_hash=" in caption for caption in captions.values())
    assert all("CASE" not in caption for caption in captions.values())


def test_native_3d_snapshot_uses_nested_data_patch_size_for_sliding_window(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    calls = []

    def fake_sliding_window(*, inputs, predictor, **kwargs):
        calls.append(kwargs)
        return predictor(inputs)

    monkeypatch.setattr(
        inference_module,
        "_get_sliding_window_inference",
        lambda: fake_sliding_window,
    )
    config = _snapshot_config(tmp_path, local_enabled=False, log_images=True)
    config["native_3d"] = True
    config["data"] = {"patch_size": [2, 3, 4]}
    config["visualization"]["segmentation_snapshots"]["splits"] = ["train"]
    snapshotter = build_segmentation_snapshotter(
        config,
        tracker=_ImageTracker(),
        device="cpu",
    )
    assert snapshotter is not None

    snapshotter.snapshot(
        model=nn.Conv3d(4, 3, kernel_size=1),
        train_loader=_snapshot_volume_loader(),
        val_loader=_snapshot_volume_loader(),
        epoch=10,
        global_step=7,
    )

    assert calls
    assert calls[0]["roi_size"] == (2, 3, 4)


def test_transunet_fixed_examples_resolve_nested_slice_size_and_keep_regions(
    monkeypatch: pytest.MonkeyPatch,
):
    class SliceFixture:
        cases = [SimpleNamespace(case_id="CASE-2D")]
        config = {"data": {"slice_size": [2, 3]}}
        slice_axis = 0

    image = np.zeros((4, 2, 4, 5), dtype=np.float32)
    label = np.zeros((2, 4, 5), dtype=np.uint8)
    label[0, 2, 1] = 1
    label[0, 2, 2] = 4

    monkeypatch.setattr(
        "token_mixer.data.datasets.load_case_arrays",
        lambda _case: (image, label),
    )

    examples = inference_module._case_snapshot_examples(
        SliceFixture(),
        sample_count=1,
        axis=0,
    )

    assert len(examples) == 1
    assert examples[0].image.shape == (4, 2, 3)
    assert examples[0].target.shape == (3, 2, 3)
    assert np.all(examples[0].target.sum(axis=(1, 2)) > 0)


def test_snapshot_errors_redact_exception_messages_case_ids_and_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    config = _snapshot_config(tmp_path, local_enabled=True, log_images=False)
    snapshotter = build_segmentation_snapshotter(
        config,
        tracker=_ImageTracker(),
        device="cpu",
    )
    assert snapshotter is not None

    def fail_render(*_args, **_kwargs):
        raise RuntimeError("CASE-SECRET C:\\private\\case.nii.gz")

    monkeypatch.setattr(snapshotter, "_render_example", fail_render)
    snapshotter.snapshot(
        model=nn.Conv2d(4, 3, kernel_size=1),
        train_loader=_snapshot_loader(),
        val_loader=_snapshot_loader(),
        epoch=10,
        global_step=7,
    )

    assert snapshotter.errors
    assert "RuntimeError" in snapshotter.errors[0]
    assert "CASE-SECRET" not in snapshotter.errors[0]
    assert "private" not in snapshotter.errors[0]


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


def test_sliding_window_count_matches_monai_scan_convention():
    assert sliding_window_count((10, 11, 12), (4, 4, 4), 0.25) == 48
    assert sliding_window_count((2, 5, 8), (4, 4, 4), 0.25) == 6
    assert sliding_window_count((10, 11, 12), (4, 4, 4), 0.25) == sliding_window_count(
        (10, 11, 12), (4, 4, 4), 0.25
    )


def test_full_volume_case_records_are_opt_in_and_include_protocol_fields(
    monkeypatch: pytest.MonkeyPatch,
):
    def fake_sliding_window(*, inputs, predictor, **_kwargs):
        return predictor(inputs)

    monkeypatch.setattr(
        inference_module,
        "_get_sliding_window_inference",
        lambda: fake_sliding_window,
    )

    image = np.ones((1, 1, 5, 6, 7), dtype=np.float32)
    target = np.ones((1, 3, 5, 6, 7), dtype=np.float32)
    result = evaluate_full_volumes(
        _ZeroLogitModel(),
        [(image, target, ["CASE001"])],
        roi_size=(4, 4, 4),
        sw_batch_size=1,
        overlap=0.25,
        device="cpu",
        default_spacing=(1.0, 1.0, 2.0),
        collect_case_records=True,
    )

    row = result["case_records"][0]
    assert row["case_id"] == "CASE001"
    assert row["spacing"] == (1.0, 1.0, 2.0)
    assert row["voxel_count"] == 5 * 6 * 7
    assert row["sliding_window_count"] == 8
    assert set(row["dice_by_region"]) == {"ET", "TC", "WT"}
    assert set(row["hd95_by_region"]) == {"ET", "TC", "WT"}
    assert set(row["exclusion_flags"]) == {"ET", "TC", "WT"}
    assert row["hd95_excluded_by_region"] == {"ET": 1, "TC": 1, "WT": 1}
    assert row["hd95_excluded_count"] == 3
    assert "latency_ms" not in row


def test_full_volume_default_result_does_not_collect_case_records(
    monkeypatch: pytest.MonkeyPatch,
):
    def fake_sliding_window(*, inputs, predictor, **_kwargs):
        return predictor(inputs)

    monkeypatch.setattr(
        inference_module,
        "_get_sliding_window_inference",
        lambda: fake_sliding_window,
    )
    image = np.ones((1, 1, 4, 4, 4), dtype=np.float32)
    target = np.zeros((1, 3, 4, 4, 4), dtype=np.float32)

    result = evaluate_full_volumes(
        _ZeroLogitModel(),
        [(image, target, ["CASE001"])],
        roi_size=(4, 4, 4),
        sw_batch_size=1,
        overlap=0.25,
        device="cpu",
        default_spacing=(1.0, 1.0, 1.0),
    )

    assert "case_records" not in result


def test_full_volume_case_latency_requires_explicit_timing_and_single_case_batch(
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

    result = evaluate_full_volumes(
        _ZeroLogitModel(),
        [(image, target, ["CASE001"])],
        roi_size=(2, 2, 2),
        sw_batch_size=1,
        overlap=0.0,
        device="cpu",
        default_spacing=(1.0, 1.0, 1.0),
        collect_case_records=True,
        measure_latency=True,
    )

    row = result["case_records"][0]
    assert isinstance(row["latency_ms"], float)
    assert row["latency_ms"] >= 0.0
    for key in (
        "load_seconds",
        "preprocess_seconds",
        "model_compute_seconds",
        "postprocess_seconds",
        "metric_seconds",
        "end_to_end_seconds",
    ):
        assert key in row
        assert row[key] >= 0.0
    assert row["end_to_end_seconds"] >= row["model_compute_seconds"]
    assert row["latency_ms"] == pytest.approx(
        row["end_to_end_seconds"] * 1000.0
    )

    batched_image = np.ones((2, 1, 2, 2, 2), dtype=np.float32)
    batched_target = np.zeros((2, 3, 2, 2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="batch size 1"):
        evaluate_full_volumes(
            _ZeroLogitModel(),
            [
                (
                    batched_image,
                    batched_target,
                    ["CASE001", "CASE002"],
                )
            ],
            roi_size=(2, 2, 2),
            sw_batch_size=1,
            overlap=0.0,
            device="cpu",
            default_spacing=(1.0, 1.0, 1.0),
            collect_case_records=True,
            measure_latency=True,
        )
