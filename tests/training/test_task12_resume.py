from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from token_mixer.pipelines._baseline_common import (
    resume_path,
    run_3d_baseline,
    warm_start_path,
)
from token_mixer.training.artifacts import write_run_artifacts
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


def _loader(generator: torch.Generator | None = None) -> DataLoader:
    values = torch.ones(2, 1)
    return DataLoader(
        TensorDataset(values, values),
        batch_size=1,
        shuffle=generator is not None,
        generator=generator,
    )


def _config(checkpoint_dir: Path | None = None) -> dict[str, object]:
    config: dict[str, object] = {
        "device": "cpu",
        "use_amp": False,
        "gradient_accumulation_steps": 1,
        "validation_interval": 1,
        "monitor": "mean_dice",
        "maximize": True,
        "architecture": "TinyEncoderDecoder",
        "model_config": {"width": 1},
        "manifest_hash": "manifest-sha256",
        "loss": "mse",
        "optimizer": {"name": "adam", "weight_decay": 0.0},
        "scheduler": {"name": "step", "step_size": 1, "gamma": 0.5},
    }
    if checkpoint_dir is not None:
        config["paths"] = {"checkpoint_dir": str(checkpoint_dir)}
    return config


def _phases(epochs: int = 1) -> list[PhaseSpec]:
    return [PhaseSpec("train", epochs, False, 0.01, 0.01)]


def _fit(
    model: nn.Module,
    manager: CheckpointManager | None,
    config: dict[str, object],
    *,
    epochs: int = 1,
    resume: Path | None = None,
    warm_start: Path | None = None,
    generator: torch.Generator | None = None,
) -> FitResult:
    train_generator = generator
    train_loader = _loader(train_generator)
    val_loader = _loader()
    return fit(
        model,
        train_loader,
        val_loader,
        nn.MSELoss(),
        lambda *_args: {"mean_dice": 0.5},
        _phases(epochs),
        config,
        _Tracker(),
        manager,
        resume=resume,
        warm_start=warm_start,
        loader_generator=generator,
    )


def test_resume_path_uses_nested_run_resume_path(tmp_path: Path):
    configured = tmp_path / "source" / "last.pt"

    assert resume_path({"run": {"resume": str(configured)}}, None) == configured


def test_warm_start_path_uses_nested_run_warm_start_path(tmp_path: Path):
    configured = tmp_path / "source" / "best.pt"

    assert (
        warm_start_path({"run": {"warm_start": str(configured)}}, None)
        == configured
    )


def test_nested_run_warm_start_mode_falls_back_to_nested_run_resume_path(
    tmp_path: Path,
):
    configured = tmp_path / "source" / "resume.pt"
    config = {
        "run": {
            "resume": str(configured),
            "resume_mode": "warm_start",
        }
    }

    assert resume_path(config, None) is None
    assert warm_start_path(config, None) == configured


def test_checkpoint_round_trip_restores_rng_and_validates_metadata(tmp_path: Path):
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    generator = torch.Generator().manual_seed(11)
    model = nn.Linear(1, 1)
    manager = CheckpointManager(tmp_path)
    metadata = {
        "architecture": "Tiny",
        "model_config": {"width": 1},
        "manifest_hash": "manifest-sha256",
        "loss": "mse",
        "optimizer": {"name": "adam"},
        "scheduler": {"name": "step"},
        "phase_plan": [{"name": "train", "epochs": 1}],
        "monitor": "mean_dice",
        "direction": "maximize",
    }
    path = manager.save(
        "last",
        model,
        None,
        None,
        None,
        {"epoch": 1, "checkpoint_metadata": metadata, "loader_generator": generator},
    )

    expected_python = random.random()
    expected_numpy = np.random.random()
    expected_torch = torch.rand(3)
    expected_loader = torch.rand(3, generator=generator)

    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    restored_generator = torch.Generator().manual_seed(99)
    manager.load(
        path,
        nn.Linear(1, 1),
        generator=restored_generator,
        expected_metadata=metadata,
    )

    assert random.random() == expected_python
    assert np.random.random() == expected_numpy
    assert torch.equal(torch.rand(3), expected_torch)
    assert torch.equal(torch.rand(3, generator=restored_generator), expected_loader)

    with pytest.raises(ValueError, match="architecture"):
        manager.load(
            path,
            nn.Linear(1, 1),
            expected_metadata={**metadata, "architecture": "Other"},
        )


def test_model_only_load_does_not_restore_rng_or_optimizer_state(tmp_path: Path):
    torch.manual_seed(4)
    source = nn.Linear(1, 1)
    manager = CheckpointManager(tmp_path)
    path = manager.save("best", source, None, None, None, {"epoch": 1})

    target = nn.Linear(1, 1)
    with torch.no_grad():
        target.weight.zero_()
        target.bias.zero_()
    torch.manual_seed(123)
    before = torch.get_rng_state().clone()
    loaded = manager.load_model(path, target)

    assert torch.equal(torch.get_rng_state(), before)
    assert torch.equal(target.weight, source.weight)
    assert torch.equal(target.bias, source.bias)
    assert loaded["model_state_dict"]


def test_fit_exact_resume_keeps_history_and_training_counters(tmp_path: Path):
    source_manager = CheckpointManager(tmp_path / "source")
    source_generator = torch.Generator().manual_seed(7)
    source = _fit(
        _TinyEncoderDecoder(),
        source_manager,
        _config(),
        generator=source_generator,
    )
    assert source.history[0]["global_step"] == 2

    destination_manager = CheckpointManager(tmp_path / "continued")
    resumed = _fit(
        _TinyEncoderDecoder(),
        destination_manager,
        _config(),
        epochs=2,
        resume=source_manager.root / "last.pt",
        generator=torch.Generator().manual_seed(99),
    )

    assert [record["epoch"] for record in resumed.history] == [1, 2]
    assert resumed.history[-1]["global_step"] == 4
    assert resumed.best_metric == pytest.approx(source.best_metric)
    assert resumed.best_epoch == source.best_epoch


def test_fit_warm_start_loads_weights_with_fresh_history_and_rng(tmp_path: Path):
    source_manager = CheckpointManager(tmp_path / "source")
    _fit(
        _TinyEncoderDecoder(),
        source_manager,
        _config(),
        generator=torch.Generator().manual_seed(7),
    )

    model = _TinyEncoderDecoder()
    torch.manual_seed(123)
    expected_rng = torch.get_rng_state().clone()
    result = _fit(
        model,
        CheckpointManager(tmp_path / "warm"),
        _config(),
        warm_start=source_manager.root / "best.pt",
        generator=torch.Generator().manual_seed(99),
    )

    assert [record["epoch"] for record in result.history] == [1]
    assert result.history[0]["global_step"] == 2
    assert torch.equal(torch.get_rng_state(), expected_rng)


def test_exact_resume_copies_source_best_into_new_output(tmp_path: Path):
    source_root = tmp_path / "source" / "checkpoints"
    target_root = tmp_path / "target" / "checkpoints"
    source_manager = CheckpointManager(source_root)
    source_model = _TinyEncoderDecoder()
    with torch.no_grad():
        source_model.encoder.weight.fill_(1.0)
    source_manager.save("best", source_model, None, None, None, {"epoch": 1})
    with torch.no_grad():
        source_model.encoder.weight.fill_(2.0)
    source_last = source_manager.save("last", source_model, None, None, None, {"epoch": 1})
    source_best_bytes = (source_root / "best.pt").read_bytes()

    target_model = _TinyEncoderDecoder()

    def fake_fit(*_args, **_kwargs):
        return FitResult(0.5, 1, [])

    result = run_3d_baseline(
        _config(target_root) | {"resume": str(source_last), "spacing": (1.0, 1.0, 1.0)},
        architecture="Tiny",
        model_builder=lambda _cfg: target_model,
        loader_builder=lambda _cfg, _generator: ([], [], []),
        evaluator_builder=lambda _cfg, _device: lambda _model, _loader: {"mean_dice": 0.5},
        loss_builder=lambda _cfg: nn.MSELoss(),
        fit_fn=fake_fit,
    )

    assert result.test_metrics == {"mean_dice": 0.5}
    assert torch.equal(target_model.encoder.weight, torch.ones_like(target_model.encoder.weight))
    assert (target_root / "best.pt").read_bytes() == source_best_bytes
    assert (source_root / "best.pt").read_bytes() == source_best_bytes


def test_run_artifacts_include_reproducibility_contract(tmp_path: Path):
    result = FitResult(
        0.75,
        2,
        [{"epoch": 1, "mean_dice": 0.5}],
        {"mean_dice": 0.8},
        {
            "architecture": "MetaUNETR",
            "variant": "mod_a",
            "model_config": {"base_channels": 2},
            "manifest_hash": "manifest-sha256",
            "execution_device": "cpu",
            "code_version": "0.1.0",
            "monitor": "mean_dice",
            "maximize": True,
            "source_checkpoint": "old/checkpoints/last.pt",
        },
    )
    paths = write_run_artifacts(
        tmp_path,
        {
            "experiment": {"name": "mod_a"},
            "seed": 42,
            "device": "cpu",
            "tracking": {"enabled": False, "mode": "disabled"},
        },
        result,
    )

    provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    assert provenance["architecture"] == "MetaUNETR"
    assert provenance["variant"] == "mod_a"
    assert provenance["model_config"] == {"base_channels": 2}
    assert provenance["seed"] == 42
    assert provenance["device"] == "cpu"
    assert provenance["manifest_hash"] == "manifest-sha256"
    assert provenance["monitor"] == "mean_dice"
    assert provenance["direction"] == "maximize"
    assert provenance["source_checkpoint"] == "old/checkpoints/last.pt"
