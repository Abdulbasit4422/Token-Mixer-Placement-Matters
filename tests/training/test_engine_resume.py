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
    def log(self, _metrics, step):
        del step

    def log_summary(self, _metrics):
        return None

    def finish(self):
        return None


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

    assert resumed.history[0]["epoch"] == 2
    assert resumed.history[0]["global_step"] == 2
    assert all(
        not torch.equal(parameter.detach(), torch.full_like(parameter, 99.0))
        for parameter in resumed_model.parameters()
    )
