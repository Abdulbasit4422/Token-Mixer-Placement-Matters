from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from token_mixer.training.checkpoints import CheckpointManager
from token_mixer.training.engine import FitResult, fit
from token_mixer.training.phases import PhaseSpec


class _TinyEncoderDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(1, 1, bias=False)
        self.decoder = nn.Linear(1, 1, bias=False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(image))


class _Tracker:
    def __init__(self) -> None:
        self.summaries = []

    def log(self, _metrics, step):
        del step

    def log_summary(self, metrics):
        self.summaries.append(dict(metrics))

    def finish(self):
        return None


class _SnapshotLedger:
    interval_epochs = 10
    include_best = True
    include_final = True
    image_work_enabled = True

    def __init__(self):
        self.events = []
        self.emitted = set()

    def snapshot(self, **kwargs):
        key = (kwargs["epoch"], kwargs["kind"])
        if key not in self.emitted:
            self.emitted.add(key)
            self.events.append(key)

    def state_dict(self):
        return {"emitted": [list(item) for item in sorted(self.emitted)]}

    def load_state_dict(self, state):
        self.emitted = {tuple(item) for item in state.get("emitted", [])}


def _loader() -> DataLoader:
    values = torch.ones(1, 1)
    return DataLoader(TensorDataset(values, values), batch_size=1, shuffle=False)


def _config() -> dict[str, object]:
    return {
        "device": "cpu",
        "use_amp": False,
        "optimizer": {"name": "sgd", "momentum": 0.0},
        "scheduler": {"name": "step", "step_size": 1, "gamma": 0.5},
        "monitor": "mean_dice",
        "maximize": True,
    }


def test_fit_resume_restores_checkpoint_model_and_continues_epoch(tmp_path: Path):
    first_model = _TinyEncoderDecoder()
    first = fit(
        first_model,
        _loader(),
        _loader(),
        nn.MSELoss(),
        lambda *_args: {"mean_dice": 0.5},
        [PhaseSpec("train", 1, False, 0.1, 0.1)],
        _config(),
        _Tracker(),
        CheckpointManager(tmp_path),
    )
    assert isinstance(first, FitResult)
    checkpoint = tmp_path / "last.pt"

    resumed_model = _TinyEncoderDecoder()
    with torch.no_grad():
        for parameter in resumed_model.parameters():
            parameter.fill_(99.0)

    resumed = fit(
        resumed_model,
        _loader(),
        _loader(),
        nn.MSELoss(),
        lambda *_args: {"mean_dice": 0.5},
        [PhaseSpec("train", 2, False, 0.1, 0.1)],
        _config(),
        _Tracker(),
        CheckpointManager(tmp_path),
        resume=checkpoint,
    )

    assert [record["epoch"] for record in resumed.history] == [1, 2]
    assert [record["global_step"] for record in resumed.history] == [1, 2]
    assert all(
        not torch.equal(parameter.detach(), torch.full_like(parameter, 99.0))
        for parameter in resumed_model.parameters()
    )


def test_fit_resume_starts_new_process_segment_and_preserves_history(tmp_path: Path):
    manager = CheckpointManager(tmp_path)
    generator = torch.Generator().manual_seed(7)

    first = fit(
        _TinyEncoderDecoder(),
        _loader(),
        _loader(),
        nn.MSELoss(),
        lambda *_args: {"mean_dice": 0.5},
        [PhaseSpec("train", 1, False, 0.1, 0.1)],
        _config(),
        _Tracker(),
        manager,
        loader_generator=generator,
    )

    resumed = fit(
        _TinyEncoderDecoder(),
        _loader(),
        _loader(),
        nn.MSELoss(),
        lambda *_args: {"mean_dice": 0.5},
        [PhaseSpec("train", 2, False, 0.1, 0.1)],
        _config(),
        _Tracker(),
        manager,
        resume=tmp_path / "last.pt",
        loader_generator=torch.Generator().manual_seed(99),
    )

    assert isinstance(first, FitResult)
    assert [record["epoch"] for record in resumed.history] == [1, 2]
    assert [record["global_step"] for record in resumed.history] == [1, 2]
    assert resumed.metadata["timing_scope"] == "process_segment"
    assert resumed.metadata["segment_elapsed_seconds"] >= 0
    assert resumed.metadata["cumulative_train_seconds"] >= first.metadata[
        "cumulative_train_seconds"
    ]


def test_fit_resume_without_loader_generator_preserves_history_and_global_step(
    tmp_path: Path,
):
    manager = CheckpointManager(tmp_path)
    fit(
        _TinyEncoderDecoder(),
        _loader(),
        _loader(),
        nn.MSELoss(),
        lambda *_args: {"mean_dice": 0.5},
        [PhaseSpec("train", 1, False, 0.1, 0.1)],
        _config(),
        _Tracker(),
        manager,
    )

    resumed = fit(
        _TinyEncoderDecoder(),
        _loader(),
        _loader(),
        nn.MSELoss(),
        lambda *_args: {"mean_dice": 0.5},
        [PhaseSpec("train", 2, False, 0.1, 0.1)],
        _config(),
        _Tracker(),
        manager,
        resume=tmp_path / "last.pt",
    )

    assert [record["epoch"] for record in resumed.history] == [1, 2]
    assert [record["global_step"] for record in resumed.history] == [1, 2]


def test_fit_resume_after_early_stop_does_not_train_extra_epochs(tmp_path: Path):
    manager = CheckpointManager(tmp_path)
    first_values = iter((1.0, 0.9, 0.8, 0.7))
    source_generator = torch.Generator().manual_seed(7)
    early_config = {
        **_config(),
        "early_stopping": {
            "enabled": True,
            "monitor": "mean_dice",
            "mode": "max",
            "patience": 2,
            "min_delta": 0.0,
        },
    }
    first = fit(
        _TinyEncoderDecoder(),
        _loader(),
        _loader(),
        nn.MSELoss(),
        lambda *_args: {"mean_dice": next(first_values)},
        [PhaseSpec("train", 5, False, 0.1, 0.1)],
        early_config,
        _Tracker(),
        manager,
        loader_generator=source_generator,
    )
    payload = torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=False)
    assert payload["state"]["early_stopping_stopped"] is True
    assert payload["state"]["early_stopping_stop_epoch"] == 3
    assert payload["state"]["early_stopping_best_epoch"] == 1
    saved_rng_state = payload["rng_state"]["torch"]
    saved_loader_state = payload["state"]["loader_generator"]

    def fail_if_training_or_validation_runs(*_args):
        raise AssertionError("resumed stopped run performed extra work")

    resumed_tracker = _Tracker()
    torch.manual_seed(12345)
    resumed_model = _TinyEncoderDecoder()
    torch.manual_seed(54321)
    resumed_generator = torch.Generator().manual_seed(99)
    resumed = fit(
        resumed_model,
        _loader(),
        _loader(),
        nn.MSELoss(),
        fail_if_training_or_validation_runs,
        [PhaseSpec("train", 5, False, 0.1, 0.1)],
        early_config,
        resumed_tracker,
        manager,
        resume=tmp_path / "last.pt",
        loader_generator=resumed_generator,
    )

    assert [record["epoch"] for record in first.history] == [1, 2, 3]
    assert [record["epoch"] for record in resumed.history] == [1, 2, 3]
    assert resumed.history[-1]["global_step"] == first.history[-1]["global_step"]
    assert resumed.metadata["early_stopping/stopped"] is True
    assert resumed.metadata["early_stopping/stop_epoch"] == 3
    assert resumed.metadata["early_stopping/best_epoch"] == 1
    assert resumed_tracker.summaries[-1]["early_stopping/stopped"] is True
    assert torch.equal(torch.get_rng_state(), saved_rng_state)
    assert torch.equal(resumed_generator.get_state(), saved_loader_state)


def test_fit_resume_deduplicates_final_snapshot_state(tmp_path: Path):
    manager = CheckpointManager(tmp_path)
    first_snapshotter = _SnapshotLedger()

    fit(
        _TinyEncoderDecoder(),
        _loader(),
        _loader(),
        nn.MSELoss(),
        lambda *_args: {"mean_dice": 0.5},
        [PhaseSpec("train", 2, False, 0.1, 0.1)],
        _config(),
        _Tracker(),
        manager,
        snapshotter=first_snapshotter,
    )

    resumed_snapshotter = _SnapshotLedger()
    fit(
        _TinyEncoderDecoder(),
        _loader(),
        _loader(),
        nn.MSELoss(),
        lambda *_args: {"mean_dice": 0.5},
        [PhaseSpec("train", 2, False, 0.1, 0.1)],
        _config(),
        _Tracker(),
        manager,
        resume=tmp_path / "last.pt",
        snapshotter=resumed_snapshotter,
    )

    assert first_snapshotter.events == [(1, "best"), (2, "final")]
    assert resumed_snapshotter.events == []
