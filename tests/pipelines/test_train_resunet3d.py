from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from token_mixer.models import weight_transfer
from token_mixer.training.artifacts import write_run_artifacts
from token_mixer.training.engine import FitResult


def _source_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Build deterministic synthetic ResNet-shaped state for CPU injection."""
    target = model.encoder.state_dict()
    source: dict[str, torch.Tensor] = {}
    for index, (target_key, source_key) in enumerate(
        weight_transfer._resnet18_target_sources().items()
    ):
        if target_key not in target:
            continue
        target_tensor = target[target_key]
        if target_key == "stages.0.blocks.0.conv1.weight":
            shape = (64, 3, 7, 7)
        elif target_tensor.ndim == 5:
            shape = (
                int(target_tensor.shape[0]),
                int(target_tensor.shape[1]),
                int(target_tensor.shape[3]),
                int(target_tensor.shape[4]),
            )
        else:
            shape = tuple(int(size) for size in target_tensor.shape)
        source[source_key] = torch.full(shape, float(index + 1))
    return source


def _config(tmp_path: Path) -> Any:
    return OmegaConf.create(
        {
            "device": "cpu",
            "spacing": [1.0, 1.0, 1.0],
            "seed": 17,
            "deterministic": True,
            "model": {
                "in_channels": 4,
                "out_channels": 3,
                "base_features": 2,
                "depths": [2, 2, 2, 2, 2],
                "normalization": "group",
                "norm_num_groups": 1,
            },
            "imagenet_transfer": {
                "enabled": True,
                "download": False,
                "cache_dir": str(tmp_path / "cache"),
                "source": "synthetic-pretraining",
            },
            "paths": {"checkpoint_dir": str(tmp_path / "checkpoints")},
            "phases": [
                {
                    "name": "train",
                    "epochs": 1,
                    "freeze_encoder": False,
                    "encoder_lr": 0.001,
                    "decoder_lr": 0.001,
                }
            ],
            "tracking": {"enabled": False, "mode": "disabled"},
        }
    )


def test_resunet_pipeline_invokes_injected_transfer_and_records_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    import token_mixer.pipelines.train_resunet3d as pipeline

    model = pipeline.build_resunet3d(_config(tmp_path))
    source = _source_state(model)
    calls: list[dict[str, object]] = []
    real_loader = pipeline.load_imagenet_resnet18_weights

    def load_transfer(*args: object, **kwargs: object) -> dict[str, int | float]:
        calls.append({"args": args, **kwargs})
        return real_loader(*args, **kwargs)

    monkeypatch.setattr(pipeline, "load_imagenet_resnet18_weights", load_transfer)
    monkeypatch.setattr(
        pipeline,
        "build_loaders",
        lambda _cfg, _generator: ([], [], []),
    )
    monkeypatch.setattr(
        pipeline,
        "_build_evaluator",
        lambda _cfg, _device: lambda _model, _loader: {"mean_dice": 0.5},
    )
    monkeypatch.setattr(pipeline, "create_tracker", lambda *_args: object())

    def fake_fit(*args: object, **_kwargs: object) -> FitResult:
        checkpoints = args[-1]
        checkpoints.save(
            "best",
            args[0],
            None,
            None,
            None,
            {"epoch": 1, "metric": 0.5, "config": args[6]},
        )
        return FitResult(0.5, 1, [{"mean_dice": 0.5}])

    monkeypatch.setattr(pipeline, "fit", fake_fit)

    result = pipeline.run_resunet3d(_config(tmp_path), source_model=source)

    assert len(calls) == 1
    assert calls[0]["source_model"] is source
    assert calls[0]["download"] is False
    assert result.metadata is not None
    assert result.metadata["transfer_requested"] is True
    assert result.metadata["transfer_status"] == "loaded"
    assert result.metadata["transfer_source"] == "synthetic-pretraining"
    assert result.metadata["transfer_count"] == result.metadata["transfer_counts"]["copied"]

    artifact_paths = write_run_artifacts(tmp_path / "run", _config(tmp_path), result)
    provenance = json.loads(
        artifact_paths["provenance"].read_text(encoding="utf-8")
    )
    metadata = provenance["metadata"]
    assert metadata["transfer_requested"] is True
    assert metadata["transfer_status"] == "loaded"
    assert metadata["transfer_count"] > 0
    assert metadata["transfer_source"] == "synthetic-pretraining"


def test_resunet_pipeline_reports_requested_transfer_failure_before_training(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    import token_mixer.pipelines.train_resunet3d as pipeline

    fit_called = False

    def fail_fit(*_args: object, **_kwargs: object) -> None:
        nonlocal fit_called
        fit_called = True

    monkeypatch.setattr(pipeline, "fit", fail_fit)
    monkeypatch.setattr(pipeline, "build_loaders", lambda *_args: ([], [], []))

    with pytest.raises(RuntimeError, match="ImageNet transfer.*requested|download=True"):
        pipeline.run_resunet3d(_config(tmp_path))

    assert fit_called is False


def test_cli_writes_failed_transfer_provenance_without_metrics_or_training(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    import token_mixer.cli as cli
    import token_mixer.pipelines.train_resunet3d as pipeline

    config = _config(tmp_path)
    config["experiment"] = {"name": "resunet3d"}
    output_dir = tmp_path / "failed-run"
    fit_called = False

    def fail_fit(*_args: object, **_kwargs: object) -> None:
        nonlocal fit_called
        fit_called = True

    monkeypatch.setattr(cli, "_hydra_output_dir", lambda: output_dir)
    monkeypatch.setattr(pipeline, "fit", fail_fit)

    with pytest.raises(RuntimeError, match="ImageNet transfer.*requested"):
        cli._run(config)

    provenance = json.loads(
        (output_dir / "provenance.json").read_text(encoding="utf-8")
    )
    transfer = provenance["metadata"]["weight_transfer"]
    assert provenance["status"] == "failed"
    assert "error" in provenance
    assert "ImageNet transfer" in provenance["error"]
    assert transfer["requested"] is True
    assert transfer["status"] == "failed"
    assert transfer["count"] == 0
    assert transfer["source"] == "synthetic-pretraining"
    assert "best_metric" not in provenance
    assert not (output_dir / "metrics.json").exists()
    assert fit_called is False


def test_cli_reraises_transfer_failure_if_failure_artifact_write_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    import token_mixer.cli as cli

    config = _config(tmp_path)
    config["experiment"] = {"name": "resunet3d"}
    original = RuntimeError("original transfer failure")
    setattr(
        original,
        "transfer_report",
        {
            "requested": True,
            "status": "failed",
            "count": 0,
            "counts": {},
            "source": "synthetic-pretraining",
        },
    )

    def fail_dispatch(_cfg: object) -> None:
        raise original

    def fail_artifact(*_args: object, **_kwargs: object) -> None:
        raise OSError("artifact filesystem failure")

    monkeypatch.setattr(cli, "_hydra_output_dir", lambda: tmp_path / "failed-run")
    monkeypatch.setattr(cli, "_dispatch", fail_dispatch)
    monkeypatch.setattr(cli, "write_failed_run_artifact", fail_artifact)

    with pytest.raises(RuntimeError) as caught:
        cli._run(config)

    assert caught.value is original
