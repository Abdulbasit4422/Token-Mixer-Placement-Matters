from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import token_mixer.training.engine as engine_module
from token_mixer.evaluation.inference import build_segmentation_snapshotter
from token_mixer.evaluation.metrics import REGION_NAMES
from token_mixer.privacy import hash_case_id
from token_mixer.training.checkpoints import CheckpointManager
from token_mixer.training.engine import EpochTrainResult, FitResult, _optimizer_step, fit
from token_mixer.training.phases import PhaseSpec


class _TinyEncoderDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(1, 1, bias=False)
        self.decoder = nn.Linear(1, 1, bias=False)

    def forward(self, image):
        return self.decoder(self.encoder(image))


class _SnapshotOutputModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Identity()
        self.decoder = nn.Identity()
        self.value = nn.Parameter(torch.tensor(0.03))

    def forward(self, image):
        return self.value.expand(image.shape[0], 3, *image.shape[-2:])


class _RecordingTracker:
    def __init__(self):
        self.logs = []
        self.summaries = []
        self.finished = False
        self.defined_metrics = []

    def log(self, metrics, step):
        self.logs.append((dict(metrics), step))

    def log_summary(self, metrics):
        self.summaries.append(dict(metrics))

    def define_metric(self, name, *, step_metric=None):
        self.defined_metrics.append((name, step_metric))

    def finish(self):
        self.finished = True


class _ImageRecordingTracker(_RecordingTracker):
    image_logging_enabled = True

    def __init__(self):
        super().__init__()
        self.image_calls = []

    def log_images(self, images, *, step, captions=None):
        del captions
        self.image_calls.append((dict(images), step))


class _FailingFinishTracker(_RecordingTracker):
    def finish(self):
        self.finished = True
        raise RuntimeError("tracker finish failed")


class _RecordingSnapshotter:
    interval_epochs = 10
    include_best = True
    include_final = True

    def __init__(self):
        self.events = []

    def snapshot(
        self,
        *,
        model,
        train_loader,
        val_loader,
        epoch,
        global_step,
        kind,
    ):
        del model, train_loader, val_loader
        self.events.append((kind, epoch, global_step))


class _StateRecordingSnapshotter(_RecordingSnapshotter):
    def __init__(self):
        super().__init__()
        self.states = {}

    def snapshot(self, *, model, **kwargs):
        self.states[kwargs["kind"]] = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
        super().snapshot(model=model, **kwargs)


class _BatchNormEncoderDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.BatchNorm1d(1), nn.Dropout(p=0.5))
        self.decoder = nn.Linear(1, 1, bias=False)

    def forward(self, image):
        return self.decoder(self.encoder(image))


def _loader():
    images = torch.ones(3, 1)
    targets = torch.ones(3, 1)
    return DataLoader(TensorDataset(images, targets), batch_size=1, shuffle=False)


def _config():
    return {
        "device": "cpu",
        "use_amp": False,
        "gradient_accumulation_steps": 2,
        "max_grad_norm": 0.25,
        "validation_interval": 1,
        "monitor": "mean_dice",
        "maximize": True,
        "optimizer": {"name": "sgd", "momentum": 0.0},
        "scheduler": {"name": "step", "step_size": 1, "gamma": 0.5},
    }


def test_fit_handles_accumulation_phases_validation_tracking_and_checkpoints(tmp_path: Path):
    model = _TinyEncoderDecoder()
    tracker = _RecordingTracker()
    checkpoints = CheckpointManager(tmp_path)
    phase_seen = []

    def evaluator(current_model, _loader):
        phase_seen.append(
            all(not parameter.requires_grad for parameter in current_model.encoder.parameters())
        )
        return {"mean_dice": float(len(phase_seen)) / 2.0}

    result = fit(
        model,
        _loader(),
        _loader(),
        nn.MSELoss(),
        evaluator,
        [
            PhaseSpec("phase1", 1, True, 0.0, 0.1),
            PhaseSpec("phase2", 1, False, 0.1, 0.1),
        ],
        _config(),
        tracker,
        checkpoints,
    )

    assert isinstance(result, FitResult)
    assert result.best_metric == 1.0
    assert result.best_epoch == 2
    assert len(result.history) == 2
    assert phase_seen == [True, False]
    assert result.history[-1]["global_step"] == 4
    assert (tmp_path / "best.pt").is_file()
    assert (tmp_path / "last.pt").is_file()
    assert tracker.logs
    assert tracker.summaries[-1]["best_metric"] == 1.0
    assert tracker.finished is True


def test_fit_runs_validation_inside_cpu_amp_autocast(tmp_path: Path):
    config = _config()
    config["use_amp"] = True
    observed_autocast: list[bool] = []

    def evaluator(current_model, _loader):
        observed_autocast.append(torch.is_autocast_enabled("cpu"))
        return {"mean_dice": 0.0}

    fit(
        _TinyEncoderDecoder(),
        _loader(),
        _loader(),
        nn.MSELoss(),
        evaluator,
        [PhaseSpec("phase1", 1, False, 0.1, 0.1)],
        config,
        _RecordingTracker(),
        None,
    )

    assert observed_autocast == [True]


def test_fit_rejects_non_finite_loss_before_backward(tmp_path: Path):
    model = _TinyEncoderDecoder()

    def non_finite_loss(_prediction, _target):
        return torch.tensor(float("inf"), requires_grad=True)

    with pytest.raises(FloatingPointError, match="finite"):
        fit(
            model,
            _loader(),
            _loader(),
            non_finite_loss,
            lambda *_args: {"mean_dice": 0.0},
            [PhaseSpec("phase1", 1, False, 0.1, 0.1)],
            _config(),
            _RecordingTracker(),
            CheckpointManager(tmp_path),
        )


class _LoaderWithoutLength:
    def __init__(self, batches):
        self._batches = batches

    def __iter__(self):
        yield from self._batches


def test_fit_flushes_partial_accumulation_for_iterable_without_length(tmp_path: Path):
    model = _TinyEncoderDecoder()
    loader = _LoaderWithoutLength([(torch.ones(1, 1), torch.ones(1, 1))])

    result = fit(
        model,
        loader,
        loader,
        nn.MSELoss(),
        lambda *_args: {"mean_dice": 0.0},
        [PhaseSpec("phase1", 1, False, 0.1, 0.1)],
        _config(),
        _RecordingTracker(),
        CheckpointManager(tmp_path),
    )

    assert result.history[0]["global_step"] == 1


def test_fit_preserves_zero_max_grad_norm_as_enabled_clipping():
    model = _TinyEncoderDecoder()
    with torch.no_grad():
        model.encoder.weight.fill_(1.0)
        model.decoder.weight.fill_(1.0)
    loader = DataLoader(
        TensorDataset(torch.ones(1, 1), torch.zeros(1, 1)), batch_size=1, shuffle=False
    )
    config = _config()
    config["gradient_accumulation_steps"] = 1
    config["max_grad_norm"] = 0.0
    initial_weights = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }

    fit(
        model,
        loader,
        loader,
        nn.MSELoss(),
        lambda *_args: {"mean_dice": 0.0},
        [PhaseSpec("phase1", 1, False, 0.1, 0.1)],
        config,
        _RecordingTracker(),
        None,
    )

    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, initial_weights[name])


def test_fit_keeps_frozen_encoder_in_eval_mode_during_training():
    model = _BatchNormEncoderDecoder()
    images = torch.full((4, 1), 3.0)
    targets = torch.zeros(4, 1)
    loader = DataLoader(TensorDataset(images, targets), batch_size=4, shuffle=False)

    fit(
        model,
        loader,
        loader,
        nn.MSELoss(),
        lambda *_args: {"mean_dice": 0.0},
        [PhaseSpec("frozen", 1, True, 0.0, 0.1)],
        _config(),
        _RecordingTracker(),
        None,
    )

    assert model.encoder.training is False
    assert torch.equal(model.encoder[0].running_mean, torch.zeros(1))
    assert torch.equal(model.encoder[0].running_var, torch.ones(1))


def test_optimizer_step_rejects_nonfinite_gradients_before_step():
    model = nn.Linear(1, 1)
    parameter = next(model.parameters())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    scaler = torch.amp.GradScaler(device="cpu", enabled=False)
    initial_value = parameter.detach().clone()
    parameter.grad = torch.full_like(parameter, float("nan"))

    with pytest.raises(FloatingPointError, match="gradient"):
        _optimizer_step(
            model,
            optimizer,
            scheduler,
            scaler,
            None,
            global_step=4,
            scheduler_interval="update",
        )

    assert torch.equal(parameter, initial_value)
    assert scheduler.last_epoch == 0


def test_fit_uses_update_count_for_default_cosine_t_max():
    config = _config()
    config["scheduler"] = {"name": "cosine", "interval": "update"}
    config["max_grad_norm"] = None

    result = fit(
        _TinyEncoderDecoder(),
        _loader(),
        _loader(),
        nn.MSELoss(),
        lambda *_args: {"mean_dice": 0.0},
        [PhaseSpec("phase1", 2, False, 0.1, 0.1)],
        config,
        _RecordingTracker(),
        None,
    )

    assert result.history[0]["global_step"] == 2
    assert result.history[0]["learning_rate"] == pytest.approx(0.05)


def test_fit_requires_explicit_cosine_t_max_for_unknown_length_update_loader():
    loader = _LoaderWithoutLength([(torch.ones(1, 1), torch.ones(1, 1))])
    config = _config()
    config["scheduler"] = {"name": "cosine", "interval": "update"}

    with pytest.raises(ValueError, match="t_max"):
        fit(
            _TinyEncoderDecoder(),
            loader,
            loader,
            nn.MSELoss(),
            lambda *_args: {"mean_dice": 0.0},
            [PhaseSpec("phase1", 1, False, 0.1, 0.1)],
            config,
            _RecordingTracker(),
            None,
        )


def test_fit_rejects_nonfinite_monitored_metric():
    tracker = _RecordingTracker()

    with pytest.raises(FloatingPointError, match="monitored metric.*finite"):
        fit(
            _TinyEncoderDecoder(),
            _loader(),
            _loader(),
            nn.MSELoss(),
            lambda *_args: {"mean_dice": float("nan")},
            [PhaseSpec("phase1", 1, False, 0.1, 0.1)],
            _config(),
            tracker,
            None,
        )

    assert tracker.finished is True


def test_fit_finishes_tracker_when_setup_fails_and_preserves_setup_error():
    tracker = _FailingFinishTracker()
    config = _config()
    config["gradient_accumulation_steps"] = 0

    with pytest.raises(ValueError, match="gradient_accumulation_steps"):
        fit(
            _TinyEncoderDecoder(),
            _loader(),
            _loader(),
            nn.MSELoss(),
            lambda *_args: {"mean_dice": 0.0},
            [PhaseSpec("phase1", 1, False, 0.1, 0.1)],
            config,
            tracker,
            None,
        )

    assert tracker.finished is True


def test_fit_preserves_evaluator_error_when_tracker_finish_fails():
    def evaluator(_model, _loader):
        raise RuntimeError("evaluator failed")

    with pytest.raises(RuntimeError, match="evaluator failed"):
        fit(
            _TinyEncoderDecoder(),
            _loader(),
            _loader(),
            nn.MSELoss(),
            evaluator,
            [PhaseSpec("phase1", 1, False, 0.1, 0.1)],
            _config(),
            _FailingFinishTracker(),
            None,
        )


def test_fit_preserves_training_error_when_tracker_finish_fails():
    def loss_fn(_prediction, _target):
        raise RuntimeError("training failed")

    with pytest.raises(RuntimeError, match="training failed"):
        fit(
            _TinyEncoderDecoder(),
            _loader(),
            _loader(),
            loss_fn,
            lambda *_args: {"mean_dice": 0.0},
            [PhaseSpec("phase1", 1, False, 0.1, 0.1)],
            _config(),
            _FailingFinishTracker(),
            None,
        )


def test_fit_logs_encoder_and_decoder_learning_rates_separately():
    config = _config()
    config["scheduler"] = None

    result = fit(
        _TinyEncoderDecoder(),
        _loader(),
        _loader(),
        nn.MSELoss(),
        lambda *_args: {"mean_dice": 0.0},
        [PhaseSpec("phase1", 1, False, 0.01, 0.1)],
        config,
        _RecordingTracker(),
        None,
    )

    assert result.history[0]["encoder_lr"] == pytest.approx(0.01)
    assert result.history[0]["decoder_lr"] == pytest.approx(0.1)


def test_fit_propagates_manifest_hash_into_checkpoint_state(tmp_path: Path):
    config = _config()
    config["manifest_hash"] = "manifest-sha256"
    checkpoints = CheckpointManager(tmp_path)

    fit(
        _TinyEncoderDecoder(),
        _loader(),
        _loader(),
        nn.MSELoss(),
        lambda *_args: {"mean_dice": 0.0},
        [PhaseSpec("phase1", 1, False, 0.1, 0.1)],
        config,
        _RecordingTracker(),
        checkpoints,
    )

    payload = torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=False)
    assert payload["manifest_hash"] == "manifest-sha256"
    assert payload["state"]["manifest_hash"] == "manifest-sha256"


def test_epoch_train_result_is_immutable():
    result = EpochTrainResult(train_loss=0.5, global_step=2, samples=3, voxels=24)

    assert result.train_loss == 0.5
    with pytest.raises((AttributeError, TypeError)):
        result.samples = 99


def test_fit_records_observed_efficiency_fields_without_changing_steps():
    tracker = _RecordingTracker()
    images = torch.ones(3, 2, 2, 4, 1)
    targets = torch.ones(3, 2, 2, 4, 1)
    loader = DataLoader(TensorDataset(images, targets), batch_size=1, shuffle=False)

    result = fit(
        _TinyEncoderDecoder(),
        loader,
        loader,
        nn.MSELoss(),
        lambda *_args: {"mean_dice": 0.5},
        [PhaseSpec("phase1", 1, False, 0.1, 0.1)],
        _config(),
        tracker,
        None,
    )

    record = result.history[0]
    assert record["global_step"] == 2
    assert record["train/epoch"] == 1
    assert record["train/optimizer_steps"] == 2
    assert record["train/samples"] == 3
    assert record["train/voxels"] == 24
    assert record["train/samples_per_second"] > 0
    assert record["train/voxels_per_second"] > 0
    assert record["val/epoch"] == 1
    assert record["val/epoch_seconds"] >= 0
    assert record["run/elapsed_seconds"] >= record["train/epoch_seconds"]
    assert "train/time_to_best_seconds" in record
    assert ("train/epoch_seconds", "train/epoch") in tracker.defined_metrics
    assert ("train/time_to_best_seconds", "train/epoch") in tracker.defined_metrics
    assert ("val/epoch_seconds", "val/epoch") in tracker.defined_metrics


def test_fit_omits_validation_fields_on_non_validation_epochs():
    config = _config()
    config["validation_interval"] = 2

    result = fit(
        _TinyEncoderDecoder(),
        _loader(),
        _loader(),
        nn.MSELoss(),
        lambda *_args: {"mean_dice": 0.5},
        [PhaseSpec("phase1", 3, False, 0.1, 0.1)],
        config,
        _RecordingTracker(),
        None,
    )

    assert "val/epoch" not in result.history[0]
    assert "val/epoch_seconds" not in result.history[0]
    assert result.history[1]["val/epoch"] == 2


def test_fit_snapshotter_uses_absolute_interval_and_best_final_order(tmp_path: Path):
    snapshotter = _RecordingSnapshotter()
    values = iter([0.0, *([0.0] * 9), 1.0, *([1.0] * 10)])

    result = fit(
        _TinyEncoderDecoder(),
        _loader(),
        _loader(),
        nn.MSELoss(),
        lambda *_args: {"mean_dice": next(values)},
        [PhaseSpec("phase1", 21, False, 0.1, 0.1)],
        _config(),
        _RecordingTracker(),
        CheckpointManager(tmp_path),
        snapshotter=snapshotter,
    )

    assert [event[:2] for event in snapshotter.events] == [
        ("best", 1),
        ("epoch", 10),
        ("best", 11),
        ("epoch", 20),
        ("final", 21),
    ]
    assert result.history[-1]["global_step"] > result.history[0]["global_step"]


def test_fit_disabled_tracking_mode_does_not_call_injected_snapshotter():
    snapshotter = _RecordingSnapshotter()
    config = {
        **_config(),
        "visualization": {
            "segmentation_snapshots": {
                "enabled": True,
                "local_enabled": False,
            }
        },
        "tracking": {
            "enabled": False,
            "mode": "disabled",
            "log_images": True,
        },
    }

    fit(
        _TinyEncoderDecoder(),
        _loader(),
        _loader(),
        nn.MSELoss(),
        lambda *_args: {"mean_dice": 0.5},
        [PhaseSpec("phase1", 1, False, 0.1, 0.1)],
        config,
        _RecordingTracker(),
        None,
        snapshotter=snapshotter,
    )

    assert snapshotter.events == []


def test_fit_does_not_auto_enable_segmentation_snapshots_for_cnn_denoising(
    monkeypatch: pytest.MonkeyPatch,
):
    import token_mixer.evaluation.inference as inference_module

    builder_calls = []

    def forbidden_builder(*_args, **_kwargs):
        builder_calls.append("called")
        raise AssertionError("snapshot_failed: CNN denoising pairs are not segmentation data")

    monkeypatch.setattr(
        inference_module,
        "build_segmentation_snapshotter",
        forbidden_builder,
    )
    config = {
        **_config(),
        "experiment": {"model_name": "cnn_pretrain"},
        "data_name": "imagenet",
        "tracking": {
            "enabled": True,
            "mode": "offline",
            "log_images": True,
        },
        "visualization": {
            "segmentation_snapshots": {
                "enabled": True,
                "local_enabled": False,
            }
        },
    }

    result = fit(
        _TinyEncoderDecoder(),
        _loader(),
        _loader(),
        nn.MSELoss(),
        lambda *_args: {"mean_dice": 0.5},
        [PhaseSpec("phase1", 1, False, 0.1, 0.1)],
        config,
        _ImageRecordingTracker(),
        None,
    )

    assert builder_calls == []
    assert "snapshot_failed" not in repr(result.history)


@pytest.mark.parametrize(
    ("marker_name", "marker_config"),
    [
        ("model_name", {"model": {"name": "cnn_pretrain"}}),
        ("data_name", {"data_name": "imagenet"}),
        ("dataset_id", {"dataset_id": "imagenet"}),
        ("imagefolder", {"data": {"dataset": "ImageFolder"}}),
        ("denoising_architecture", {"architecture": "denoising_autoencoder"}),
    ],
)
@pytest.mark.parametrize("injected", [False, True], ids=["automatic", "injected"])
def test_fit_blocks_cnn_imagefolder_snapshotters_for_all_config_markers(
    monkeypatch: pytest.MonkeyPatch,
    marker_name,
    marker_config,
    injected,
):
    import token_mixer.evaluation.inference as inference_module

    builder_calls = []

    def forbidden_builder(*_args, **_kwargs):
        builder_calls.append(marker_name)
        raise AssertionError("snapshotter must not be built for CNN/ImageFolder data")

    monkeypatch.setattr(
        inference_module,
        "build_segmentation_snapshotter",
        forbidden_builder,
    )
    config = {
        **_config(),
        **marker_config,
        "tracking": {
            "enabled": True,
            "mode": "offline",
            "log_images": True,
        },
        "visualization": {
            "segmentation_snapshots": {
                "enabled": True,
                "local_enabled": False,
            }
        },
    }
    snapshotter = _RecordingSnapshotter() if injected else None

    fit(
        _TinyEncoderDecoder(),
        _loader(),
        _loader(),
        nn.MSELoss(),
        lambda *_args: {"mean_dice": 0.5},
        [PhaseSpec("phase1", 1, False, 0.1, 0.1)],
        config,
        _ImageRecordingTracker(),
        None,
        snapshotter=snapshotter,
    )

    assert builder_calls == []
    if snapshotter is not None:
        assert snapshotter.events == []


def test_fit_final_snapshot_uses_best_weights_without_checkpoints():
    torch.manual_seed(0)
    snapshotter = _StateRecordingSnapshotter()
    values = iter((1.0, 0.0))

    fit(
        _TinyEncoderDecoder(),
        _loader(),
        _loader(),
        nn.MSELoss(),
        lambda *_args: {"mean_dice": next(values)},
        [PhaseSpec("phase1", 2, False, 0.1, 0.1)],
        _config(),
        _RecordingTracker(),
        None,
        snapshotter=snapshotter,
    )

    assert [event[:2] for event in snapshotter.events] == [
        ("best", 1),
        ("final", 2),
    ]
    assert snapshotter.events[1][2] > snapshotter.events[0][2]
    assert snapshotter.states["final"]
    for name, value in snapshotter.states["best"].items():
        torch.testing.assert_close(snapshotter.states["final"][name], value)


def test_fit_redacts_evaluator_case_aliases_before_tracker_log():
    raw_ids = ["CASE-ENGINE-001", "BraTS-ENGINE-002"]
    tracker = _RecordingTracker()

    def evaluator(_model, _loader):
        return {
            "mean_dice": 0.5,
            "case_ids": raw_ids,
            "nested": {"caseId": raw_ids[0]},
            "ids": [raw_ids[1]],
        }

    result = fit(
        _TinyEncoderDecoder(),
        _loader(),
        _loader(),
        nn.MSELoss(),
        evaluator,
        [PhaseSpec("phase1", 1, False, 0.1, 0.1)],
        _config(),
        tracker,
        None,
    )

    logged = tracker.logs[0][0]
    assert logged["case_id_hashes"] == [hash_case_id(raw_id) for raw_id in raw_ids]
    assert logged["nested"]["case_id_hash"] == hash_case_id(raw_ids[0])
    assert logged["case_id_hashes"] == [hash_case_id(raw_ids[0]), hash_case_id(raw_ids[1])]
    assert all(raw_id not in repr(logged) for raw_id in raw_ids)
    assert result.history[0]["case_ids"] == raw_ids
    assert result.history[0]["nested"]["caseId"] == raw_ids[0]


def test_fit_keeps_final_snapshot_when_final_epoch_is_scheduled_and_restores_best_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    image = torch.ones(1, 4, 4, 4)
    target = torch.zeros(1, 3, 4, 4)
    target[0, 0, 0, 0] = 1.0
    target[0, 1, 1, 1] = 1.0
    target[0, 2, 2, 2] = 1.0
    loader = DataLoader(
        TensorDataset(image, target),
        batch_size=1,
        shuffle=False,
    )
    tracker = _ImageRecordingTracker()
    config = {
        **_config(),
        "tracking": {
            "enabled": True,
            "mode": "offline",
            "log_images": True,
        },
        "visualization": {
            "segmentation_snapshots": {
                "enabled": True,
                "snapshot_interval_epochs": 2,
                "include_best": True,
                "include_final": True,
                "splits": ["train", "val"],
                "sample_count": 1,
                "axis": 0,
                "image_channel": 3,
                "local_enabled": False,
                "output_dir": str(tmp_path),
            }
        },
    }
    rendered: list[tuple[np.ndarray, np.ndarray]] = []
    values = iter((1.0, 0.0, 1.0))

    def fake_render(image, target, prediction, **kwargs):
        del image, kwargs
        rendered.append((np.asarray(target).copy(), np.asarray(prediction).copy()))
        return np.zeros((4, 4, 3), dtype=np.uint8)

    monkeypatch.setattr(
        "token_mixer.evaluation.visualization.render_slice_visualization",
        fake_render,
    )
    snapshotter = build_segmentation_snapshotter(config, tracker, device="cpu")
    assert snapshotter is not None

    result = fit(
        _SnapshotOutputModel(),
        loader,
        loader,
        lambda prediction, _target: prediction.mean(),
        lambda _model, _loader: {"mean_dice": next(values)},
        [PhaseSpec("train", 2, False, 0.1, 0.1)],
        config,
        tracker,
        None,
        snapshotter=snapshotter,
    )

    assert result.best_epoch == 1
    assert [sorted(images) for images, _step in tracker.image_calls] == [
        [
            "segmentation/train/best_epoch_0001",
            "segmentation/val/best_epoch_0001",
        ],
        [
            "segmentation/train/epoch_0002",
            "segmentation/val/epoch_0002",
        ],
        [
            "segmentation/train/final_epoch_0002",
            "segmentation/val/final_epoch_0002",
        ],
    ]
    assert len(rendered) == 6
    assert all(
        target_array.shape[0] == len(REGION_NAMES)
        and prediction.shape[0] == len(REGION_NAMES)
        and np.all(target_array.sum(axis=tuple(range(1, target_array.ndim))) > 0)
        for target_array, prediction in rendered
    )
    assert not np.array_equal(rendered[0][1], rendered[2][1])
    np.testing.assert_array_equal(rendered[0][1], rendered[4][1])
    np.testing.assert_array_equal(rendered[1][1], rendered[5][1])


def test_fit_leaves_tracker_open_when_requested():
    tracker = _RecordingTracker()

    fit(
        _TinyEncoderDecoder(),
        _loader(),
        _loader(),
        nn.MSELoss(),
        lambda *_args: {"mean_dice": 0.5},
        [PhaseSpec("phase1", 1, False, 0.1, 0.1)],
        _config(),
        tracker,
        None,
        finish_tracker=False,
    )

    assert tracker.finished is False


def test_fit_early_stopping_patience_counts_validation_evaluations():
    tracker = _RecordingTracker()
    values = iter((1.0, 0.9, 0.8, 0.7, 0.6))

    result = fit(
        _TinyEncoderDecoder(),
        _loader(),
        _loader(),
        nn.MSELoss(),
        lambda *_args: {"mean_dice": next(values)},
        [PhaseSpec("phase1", 5, False, 0.1, 0.1)],
        _config(),
        tracker,
        None,
        early_stopping={
            "enabled": True,
            "monitor": "mean_dice",
            "mode": "max",
            "patience": 2,
            "min_delta": 0.0,
        },
    )

    assert [record["epoch"] for record in result.history] == [1, 2, 3]
    assert result.history[-1]["global_step"] == 6
    assert result.history[-1]["early_stopping/stopped"] is True
    assert result.history[-1]["early_stopping/stop_epoch"] == 3
    assert tracker.summaries[-1]["early_stopping/stopped"] is True
    assert tracker.summaries[-1]["early_stopping/stop_epoch"] == 3
    assert tracker.summaries[-1]["early_stopping/best_epoch"] == 1


def test_fit_early_stopping_honors_min_delta_min_epochs_and_min_mode():
    values = iter((10.0, 9.95, 9.0, 9.0, 9.0))

    result = fit(
        _TinyEncoderDecoder(),
        _loader(),
        _loader(),
        nn.MSELoss(),
        lambda *_args: {"mean_dice": next(values)},
        [PhaseSpec("phase1", 5, False, 0.1, 0.1)],
        _config(),
        _RecordingTracker(),
        None,
        early_stopping={
            "enabled": True,
            "monitor": "mean_dice",
            "mode": "min",
            "patience": 1,
            "min_delta": 0.1,
            "min_epochs": 3,
        },
    )

    assert [record["epoch"] for record in result.history] == [1, 2, 3, 4]
    assert result.history[-1]["early_stopping/stopped"] is True
    assert result.history[-1]["early_stopping/stop_epoch"] == 4
    assert result.history[-1]["early_stopping/best_epoch"] == 3


def test_fit_copies_nvml_boundary_snapshot_without_changing_optimizer_work(monkeypatch):
    class _FakeCudaModel(_TinyEncoderDecoder):
        def to(self, _device):
            return self

    class _FakeSampler:
        started = 0
        snapshots = 0
        stopped = 0

        def __init__(self, device_index, interval_seconds):
            assert device_index == 0
            assert interval_seconds == pytest.approx(0.1)

        def start(self):
            type(self).started += 1
            return self

        def snapshot(self):
            type(self).snapshots += 1
            return SimpleNamespace(
                average_watts=10.0,
                max_watts=12.0,
                joules=1.5,
                samples=3,
                interval_seconds=0.1,
                status="ok",
            )

        def stop(self):
            type(self).stopped += 1
            return self.snapshot()

    monkeypatch.setattr(engine_module, "NvmlPowerSampler", _FakeSampler)
    monkeypatch.setattr(engine_module, "_resolve_device", lambda _value: torch.device("cuda"))
    monkeypatch.setattr(engine_module, "_synchronize", lambda _device: None)
    monkeypatch.setattr(engine_module, "reset_peak_memory", lambda _device: None)
    monkeypatch.setattr(
        engine_module,
        "peak_memory_gb",
        lambda _device: {"allocated_gb": 0.25, "reserved_gb": 0.5},
    )
    monkeypatch.setattr(
        engine_module,
        "_move_to_device",
        lambda value, _device, _name: value,
    )

    config = _config()
    config["efficiency"] = {
        "nvml": {
            "enabled": True,
            "device_index": 0,
            "sample_interval_seconds": 0.1,
        }
    }
    result = fit(
        _FakeCudaModel(),
        _loader(),
        _loader(),
        nn.MSELoss(),
        lambda *_args: {"mean_dice": 0.5},
        [PhaseSpec("phase1", 1, False, 0.1, 0.1)],
        config,
        _RecordingTracker(),
        None,
    )

    record = result.history[0]
    assert record["global_step"] == 2
    assert record["train/peak_memory_allocated_gb"] == 0.25
    assert record["power/average_watts"] == 10.0
    assert record["power/max_watts"] == 12.0
    assert record["power/energy_joules"] == 1.5
    assert record["power/sample_count"] == 3
    assert record["power/sample_interval_ms"] == pytest.approx(100.0)
    assert record["power/status"] == "ok"
    assert _FakeSampler.started == 1
    assert _FakeSampler.snapshots >= 2
    assert _FakeSampler.stopped == 1
