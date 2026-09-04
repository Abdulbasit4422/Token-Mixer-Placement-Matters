from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from token_mixer.training.checkpoints import CheckpointManager
from token_mixer.training.engine import FitResult, _optimizer_step, fit
from token_mixer.training.phases import PhaseSpec


class _TinyEncoderDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(1, 1, bias=False)
        self.decoder = nn.Linear(1, 1, bias=False)

    def forward(self, image):
        return self.decoder(self.encoder(image))


class _RecordingTracker:
    def __init__(self):
        self.logs = []
        self.summaries = []
        self.finished = False

    def log(self, metrics, step):
        self.logs.append((dict(metrics), step))

    def log_summary(self, metrics):
        self.summaries.append(dict(metrics))

    def finish(self):
        self.finished = True


class _FailingFinishTracker(_RecordingTracker):
    def finish(self):
        self.finished = True
        raise RuntimeError("tracker finish failed")


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
