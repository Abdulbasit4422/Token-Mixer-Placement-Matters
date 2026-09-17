from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from token_mixer.models.weight_transfer import inflate_encoder_state_dict
from token_mixer.training.checkpoints import CheckpointManager
from token_mixer.training.engine import FitResult


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(3, 3, kernel_size=1)
        self.decoder = nn.Conv2d(3, 3, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def _config(tmp_path: Path) -> Any:
    return OmegaConf.create(
        {
            "seed": 17,
            "deterministic": True,
            "device": "cpu",
            "model": {
                "in_channels": 3,
                "feature_size": 4,
                "depths": [0, 0, 0, 0],
            },
            "data": {
                "root": str(tmp_path / "images"),
                "image_size": 32,
                "noise_std": 0.0,
                "val_fraction": 0.5,
                "pin_memory": False,
            },
            "training": {
                "epochs": 1,
                "batch_size": 1,
                "num_workers": 0,
                "learning_rate": 0.001,
                "optimizer": {"name": "sgd", "momentum": 0.0},
                "scheduler": None,
                "max_grad_norm": None,
            },
            "paths": {
                "output_dir": str(tmp_path / "experiment"),
                "checkpoint_dir": str(tmp_path / "checkpoints"),
            },
            "tracking": {"enabled": False},
        }
    )


def _fit_result_with_best(result: FitResult):
    def fake_fit(*args):
        args[-1].save(
            "best",
            args[0],
            None,
            None,
            None,
            {"epoch": result.best_epoch, "metric": result.best_metric, "config": args[6]},
        )
        return result

    return fake_fit


def test_pipeline_defaults_preserve_root_rgb_imagenet_model_settings():
    import token_mixer.pipelines.pretrain_cnn as pipeline

    model_config = pipeline._model_config(OmegaConf.create({}))

    assert model_config["in_channels"] == 3
    assert model_config["feature_size"] == 32
    assert model_config["depths"] == (1, 1, 1, 1)
    assert model_config["image_size"] == 96


def test_pipeline_defaults_preserve_supported_root_training_settings():
    import token_mixer.pipelines.pretrain_cnn as pipeline

    cfg = OmegaConf.create({})
    phases = pipeline._build_phases(cfg)
    engine_config = pipeline._engine_config(cfg, pipeline._model_config(cfg), "cpu")

    assert len(phases) == 1
    assert phases[0].epochs == 3
    assert phases[0].encoder_lr == pytest.approx(1e-3)
    assert phases[0].decoder_lr == pytest.approx(1e-3)
    assert engine_config["use_amp"] is True
    assert engine_config["weight_decay"] == pytest.approx(0.05)
    assert engine_config["max_grad_norm"] == pytest.approx(1.0)


def test_pipeline_defaults_configure_root_cosine_scheduler_floor():
    import token_mixer.pipelines.pretrain_cnn as pipeline

    engine_config = pipeline._engine_config(
        OmegaConf.create({}), pipeline._model_config(OmegaConf.create({})), "cpu"
    )

    assert engine_config["scheduler"]["name"] == "cosine"
    assert engine_config["scheduler"]["interval"] == "update"
    assert engine_config["scheduler"]["eta_min"] == pytest.approx(1e-6)


@pytest.mark.parametrize(
    "cfg",
    [
        {"min_lr": 2e-5},
        {"training": {"min_lr": 2e-5}},
        {"engine": {"min_lr": 2e-5}},
    ],
)
def test_pipeline_uses_configured_min_lr_for_default_cosine_floor(cfg: dict[str, Any]):
    import token_mixer.pipelines.pretrain_cnn as pipeline

    engine_config = pipeline._engine_config(
        OmegaConf.create(cfg), pipeline._model_config(OmegaConf.create({})), "cpu"
    )

    assert engine_config["scheduler"]["eta_min"] == pytest.approx(2e-5)


@pytest.mark.parametrize(
    ("location", "key"),
    [
        (location, key)
        for location in ("root", "training", "engine")
        for key in ("warmup_epochs", "warmup_lr", "save_every", "early_stop", "log_every")
    ],
)
def test_pipeline_rejects_unsupported_training_controls(location: str, key: str):
    import token_mixer.pipelines.pretrain_cnn as pipeline

    cfg = OmegaConf.create({key: 1} if location == "root" else {location: {key: 1}})

    with pytest.raises(ValueError, match=f"unsupported.*{key}"):
        pipeline._engine_config(cfg, pipeline._model_config(cfg), "cpu")


def test_pipeline_maps_root_epoch_and_gradient_clip_aliases():
    import token_mixer.pipelines.pretrain_cnn as pipeline

    cfg = OmegaConf.create({"num_epochs": 7, "grad_clip": 0.25})

    phases = pipeline._build_phases(cfg)
    engine_config = pipeline._engine_config(
        cfg, pipeline._model_config(cfg), "cpu"
    )

    assert phases[0].epochs == 7
    assert engine_config["max_grad_norm"] == pytest.approx(0.25)


def test_pipeline_maps_nested_epoch_and_gradient_clip_aliases():
    import token_mixer.pipelines.pretrain_cnn as pipeline

    cfg = OmegaConf.create({"training": {"num_epochs": 7, "grad_clip": 0.25}})

    phases = pipeline._build_phases(cfg)
    engine_config = pipeline._engine_config(
        cfg, pipeline._model_config(cfg), "cpu"
    )

    assert phases[0].epochs == 7
    assert engine_config["max_grad_norm"] == pytest.approx(0.25)


def test_pipeline_consumes_nested_experiment_training_settings():
    import token_mixer.pipelines.pretrain_cnn as pipeline

    cfg = OmegaConf.create(
        {
            "experiment": {
                "training": {
                    "loss": "mse",
                    "optimizer": {"name": "sgd", "momentum": 0.25},
                    "scheduler": {"name": "step", "step_size": 4, "interval": "epoch"},
                    "phases": [
                        {
                            "name": "nested_phase",
                            "epochs": 7,
                            "freeze_encoder": False,
                            "encoder_lr": 0.001,
                            "decoder_lr": 0.002,
                        }
                    ],
                    "validation_interval": 3,
                    "use_amp": False,
                    "max_grad_norm": 0.25,
                }
            }
        }
    )

    phases = pipeline._build_phases(cfg)
    engine_config = pipeline._engine_config(cfg, pipeline._model_config(cfg), "cpu")

    assert phases[0].name == "nested_phase"
    assert phases[0].epochs == 7
    assert engine_config["optimizer"] == {"name": "sgd", "momentum": 0.25}
    assert engine_config["scheduler"] == {
        "name": "step",
        "step_size": 4,
        "interval": "epoch",
    }
    assert engine_config["validation_interval"] == 3
    assert engine_config["use_amp"] is False
    assert engine_config["max_grad_norm"] == pytest.approx(0.25)
    assert pipeline._build_loss(cfg) is pipeline.mse_loss


def test_pipeline_preserves_explicit_scheduler_none():
    import token_mixer.pipelines.pretrain_cnn as pipeline

    cfg = OmegaConf.create({"training": {"scheduler": None}})

    engine_config = pipeline._engine_config(cfg, pipeline._model_config(cfg), "cpu")

    assert engine_config["scheduler"] is None


def test_pipeline_preserves_explicit_custom_scheduler():
    import token_mixer.pipelines.pretrain_cnn as pipeline

    custom_scheduler = {"name": "step", "step_size": 4, "gamma": 0.5}
    cfg = OmegaConf.create({"training": {"scheduler": custom_scheduler}})

    engine_config = pipeline._engine_config(cfg, pipeline._model_config(cfg), "cpu")

    assert engine_config["scheduler"] == custom_scheduler


def test_pipeline_dataloader_defaults_preserve_root_imagenet_loader_settings(
    tmp_path: Path,
):
    image_module = pytest.importorskip("PIL.Image")
    import token_mixer.pipelines.pretrain_cnn as pipeline

    root = tmp_path / "images"
    for class_index, class_name in enumerate(("class_a", "class_b")):
        class_dir = root / class_name
        class_dir.mkdir(parents=True)
        for image_index in range(5):
            image_module.new(
                "RGB",
                (40, 40),
                (class_index * 100, image_index * 10, 25),
            ).save(class_dir / f"sample_{image_index}.png")

    train_loader, val_loader = pipeline.build_dataloaders(
        OmegaConf.create({"data": {"root": str(root)}}),
        torch.Generator().manual_seed(42),
    )

    assert train_loader.batch_size == 5
    assert train_loader.num_workers == 4
    assert train_loader.pin_memory is True
    assert train_loader.persistent_workers is True
    assert train_loader.drop_last is True
    assert val_loader.batch_size == 5
    assert val_loader.drop_last is False


def test_pipeline_dataloaders_apply_run_max_cases_before_train_val_split(
    tmp_path: Path,
):
    image_module = pytest.importorskip("PIL.Image")
    import token_mixer.pipelines.pretrain_cnn as pipeline

    root = tmp_path / "images"
    for class_index, class_name in enumerate(("class_a", "class_b")):
        class_dir = root / class_name
        class_dir.mkdir(parents=True)
        for image_index in range(2):
            image_module.new(
                "RGB",
                (40, 40),
                (class_index * 100, image_index * 10, 25),
            ).save(class_dir / f"sample_{image_index}.png")

    cfg = OmegaConf.create(
        {
            "model": {"in_channels": 3, "image_size": 32},
            "data": {
                "root": str(root),
                "noise_std": 0.0,
                "val_fraction": 0.5,
                "pin_memory": False,
            },
            "training": {
                "batch_size": 1,
                "num_workers": 0,
                "persistent_workers": False,
                "drop_last": False,
            },
            "run": {"max_cases": 2},
        }
    )

    train_loader, val_loader = pipeline.build_dataloaders(
        cfg, torch.Generator().manual_seed(42)
    )

    assert len(train_loader.dataset) == 1
    assert len(val_loader.dataset) == 1


def test_validation_noise_is_reused_for_model_and_case_level_access(
    tmp_path: Path,
):
    image_module = pytest.importorskip("PIL.Image")
    import token_mixer.pipelines.pretrain_cnn as pipeline

    root = tmp_path / "images"
    for class_name in ("class_a", "class_b"):
        class_dir = root / class_name
        class_dir.mkdir(parents=True)
        image_module.new("RGB", (40, 40), (100, 50, 25)).save(
            class_dir / "sample.png"
        )

    cfg = OmegaConf.create(
        {
            "model": {"in_channels": 3, "image_size": 32},
            "data": {
                "root": str(root),
                "noise_std": 0.5,
                "val_fraction": 0.5,
            },
            "training": {
                "batch_size": 1,
                "num_workers": 0,
                "pin_memory": False,
                "persistent_workers": False,
                "drop_last": False,
            },
            "seed": 17,
        }
    )

    _train_loader, val_loader = pipeline.build_dataloaders(
        cfg,
        torch.Generator().manual_seed(17),
    )
    model_level_noisy = next(iter(val_loader))[0]
    case_level_noisy = next(iter(val_loader))[0]

    torch.testing.assert_close(model_level_noisy, case_level_noisy)


def test_run_pipeline_seeds_before_building_model_or_loaders_and_delegates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    import token_mixer.pipelines.pretrain_cnn as pipeline

    events: list[str] = []
    generator = object()
    model = _TinyModel()
    expected = FitResult(0.25, 1, [{"mse": 0.25}])

    def seed_everything(seed: int, deterministic: bool = True):
        events.append(f"seed:{seed}:{deterministic}")
        return generator

    def build_model(cfg):
        events.append("model")
        assert cfg["in_channels"] == 3
        return model

    def build_loaders(cfg, received_generator):
        events.append("loaders")
        assert received_generator is generator
        return "train-loader", "val-loader"

    class Tracker:
        pass

    def create_tracker(config, run_config):
        events.append("tracker")
        assert config["enabled"] is False
        assert run_config["source_model_config"]["feature_size"] == 4
        return Tracker()

    fit_calls: list[tuple[Any, ...]] = []

    def fit(*args):
        events.append("fit")
        fit_calls.append(args)
        args[-1].save(
            "best",
            args[0],
            None,
            None,
            None,
            {"epoch": expected.best_epoch, "metric": expected.best_metric, "config": args[6]},
        )
        return expected

    monkeypatch.setattr(pipeline, "seed_everything", seed_everything)
    monkeypatch.setattr(pipeline, "build_denoising_model", build_model)
    monkeypatch.setattr(pipeline, "build_dataloaders", build_loaders)
    monkeypatch.setattr(pipeline, "create_tracker", create_tracker)
    monkeypatch.setattr(pipeline, "fit", fit)

    assert pipeline.run_cnn_denoising_pretrain(_config(tmp_path)) is expected
    assert events == ["seed:17:True", "model", "loaders", "tracker", "fit"]
    assert fit_calls[0][0] is model
    assert fit_calls[0][1:3] == ("train-loader", "val-loader")
    assert callable(fit_calls[0][3])
    assert callable(fit_calls[0][4])
    assert fit_calls[0][6]["monitor"] == "mse"
    assert fit_calls[0][6]["maximize"] is False


def test_pipeline_finishes_tracker_after_final_validation_and_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    import token_mixer.pipelines.pretrain_cnn as pipeline

    cfg = _config(tmp_path)
    model = _TinyModel()
    expected = FitResult(0.25, 1, [{"mse": 0.25}])
    events: list[str] = []

    class Tracker:
        def log_summary(self, metrics):
            assert (tmp_path / "experiment" / "metrics.json").is_file()
            assert metrics["val/mse"] == pytest.approx(0.1)
            events.append("summary")

        def log_table(self, *_args, **_kwargs):
            events.append("table")

        def log_artifact(self, name, files, **kwargs):
            assert name == "model"
            assert kwargs["aliases"] == ("best", "latest")
            assert {"config.yaml", "metrics.json", "provenance.json", "encoder_best.pth"} <= set(
                files
            )
            assert all(Path(path).is_file() for path in files.values())
            events.append("artifact")
            return "entity/project/model:v0"

        def finish(self):
            events.append("finish")

    monkeypatch.setattr(pipeline, "seed_everything", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(pipeline, "build_denoising_model", lambda _cfg: model)
    monkeypatch.setattr(
        pipeline,
        "build_dataloaders",
        lambda _cfg, _generator: ("train-loader", "val-loader"),
    )
    monkeypatch.setattr(pipeline, "create_tracker", lambda *_args: Tracker())

    def fake_fit(*_args, **kwargs):
        assert kwargs["finish_tracker"] is False
        events.append("fit")
        return expected

    def restore_best(*_args, **_kwargs):
        events.append("restore_best")
        return {"epoch": 1, "metric": 0.25}

    def evaluate(_model, _loader):
        events.append("validation")
        return {"mse": 0.1}

    def export_encoder(*_args):
        events.append("export")
        path = tmp_path / "experiment" / "encoder_best.pth"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"encoder")
        return path

    monkeypatch.setattr(pipeline, "fit", fake_fit)
    monkeypatch.setattr(pipeline, "_restore_best_checkpoint", restore_best)
    monkeypatch.setattr(pipeline, "evaluate_denoising", evaluate)
    monkeypatch.setattr(pipeline, "_export_encoder", export_encoder)

    result = pipeline.run_cnn_denoising_pretrain(cfg)

    assert result.test_metrics == {"mse": 0.1}
    assert events == [
        "fit",
        "restore_best",
        "validation",
        "export",
        "summary",
        "table",
        "artifact",
        "finish",
    ]


def test_pipeline_requires_checkpointing_before_training(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    import token_mixer.pipelines.pretrain_cnn as pipeline

    cfg = _config(tmp_path)
    cfg["checkpoints"] = {"enabled": False}
    fit_calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(pipeline, "build_denoising_model", lambda _cfg: _TinyModel())
    monkeypatch.setattr(
        pipeline,
        "build_dataloaders",
        lambda _cfg, _generator: ("train-loader", "val-loader"),
    )

    monkeypatch.setattr(
        pipeline,
        "fit",
        lambda *args: fit_calls.append(args) or FitResult(0.0, 1, []),
    )

    with pytest.raises(ValueError, match="best checkpoint|checkpointing"):
        pipeline.run_cnn_denoising_pretrain(cfg)

    assert fit_calls == []


def test_pipeline_fails_before_export_when_best_checkpoint_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    import token_mixer.pipelines.pretrain_cnn as pipeline

    model = _TinyModel()
    result = FitResult(0.125, 1, [{"mse": 0.125}])
    monkeypatch.setattr(pipeline, "build_denoising_model", lambda _cfg: model)
    monkeypatch.setattr(
        pipeline,
        "build_dataloaders",
        lambda _cfg, _generator: ("train-loader", "val-loader"),
    )
    monkeypatch.setattr(pipeline, "fit", lambda *_args: result)

    with pytest.raises(FileNotFoundError, match="best checkpoint"):
        pipeline.run_cnn_denoising_pretrain(_config(tmp_path))

    assert not (tmp_path / "experiment" / "encoder_best.pth").exists()


def test_pipeline_records_resolved_auto_device_in_engine_and_visualization(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    import token_mixer.pipelines.pretrain_cnn as pipeline

    model = _TinyModel()
    result = FitResult(0.125, 1, [{"mse": 0.125}])
    cfg = _config(tmp_path)
    cfg["device"] = "auto"
    cfg["visualization"] = {"reconstruction_grid": True, "num_images": 1}
    observed: dict[str, Any] = {}
    monkeypatch.setattr(pipeline.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(pipeline, "build_denoising_model", lambda _cfg: model)
    monkeypatch.setattr(
        pipeline,
        "build_dataloaders",
        lambda _cfg, _generator: ("train-loader", "val-loader"),
    )

    def create_tracker(_config, run_config):
        observed["tracker_config"] = run_config
        return object()

    def fake_fit(*args):
        observed["fit_config"] = args[6]
        args[-1].save(
            "best",
            model,
            None,
            None,
            None,
            {"epoch": 1, "metric": 0.125, "config": args[6]},
        )
        return result

    def save_grid(*args, **kwargs):
        observed["grid_device"] = kwargs.get("device", args[3] if len(args) > 3 else None)

    monkeypatch.setattr(pipeline, "create_tracker", create_tracker)
    monkeypatch.setattr(pipeline, "fit", fake_fit)
    monkeypatch.setattr(pipeline, "_save_reconstruction_grid", save_grid)

    pipeline.run_cnn_denoising_pretrain(cfg)

    assert observed["tracker_config"]["device"] == "cpu"
    assert observed["fit_config"]["device"] == "cpu"
    assert observed["grid_device"] == torch.device("cpu")


def test_pipeline_keeps_encoder_export_device_agnostic(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    import token_mixer.pipelines.pretrain_cnn as pipeline

    model = _TinyModel()
    result = FitResult(0.125, 1, [{"mse": 0.125}])
    observed: dict[str, Any] = {}
    monkeypatch.setattr(pipeline, "build_denoising_model", lambda _cfg: model)
    monkeypatch.setattr(
        pipeline,
        "build_dataloaders",
        lambda _cfg, _generator: ("train-loader", "val-loader"),
    )

    def fake_fit(*args):
        args[-1].save(
            "best",
            model,
            None,
            None,
            None,
            {"epoch": 1, "metric": 0.125, "config": args[6]},
        )
        return result

    def fake_export(*args, **kwargs):
        assert len(args) == 5
        assert kwargs == {}
        observed["device"] = kwargs.get("device")

    monkeypatch.setattr(pipeline, "fit", fake_fit)
    monkeypatch.setattr(pipeline, "_export_encoder", fake_export)

    pipeline.run_cnn_denoising_pretrain(_config(tmp_path))

    assert observed["device"] is None


def test_effective_device_rejects_explicit_cuda_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
):
    import token_mixer.pipelines.pretrain_cnn as pipeline

    monkeypatch.setattr(pipeline.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA.*unavailable"):
        pipeline._effective_device({"device": "cuda"})


def test_pipeline_exports_encoder_from_best_checkpoint_with_source_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    import token_mixer.pipelines.pretrain_cnn as pipeline

    model = _TinyModel()
    result = FitResult(0.125, 2, [{"mse": 0.125}])
    monkeypatch.setattr(pipeline, "build_denoising_model", lambda _cfg: model)
    monkeypatch.setattr(
        pipeline,
        "build_dataloaders",
        lambda _cfg, _generator: ("train-loader", "val-loader"),
    )

    def fake_fit(*args):
        checkpoints = args[-1]
        run_config = args[6]
        with torch.no_grad():
            model.encoder.weight.fill_(1.0)
        checkpoints.save(
            "best",
            model,
            None,
            None,
            None,
            {"epoch": 2, "metric": 0.125, "config": run_config},
        )
        with torch.no_grad():
            model.encoder.weight.fill_(2.0)
        return result

    monkeypatch.setattr(pipeline, "fit", fake_fit)

    assert pipeline.run_cnn_denoising_pretrain(_config(tmp_path)) == result

    exported = torch.load(
        tmp_path / "experiment" / "encoder_best.pth",
        map_location="cpu",
        weights_only=False,
    )
    assert set(exported) == {
        "encoder_state_dict",
        "source_model_config",
        "epoch",
        "val_loss",
    }
    assert torch.all(exported["encoder_state_dict"]["weight"] == 1.0)
    assert exported["source_model_config"]["feature_size"] == 4
    assert exported["epoch"] == 2
    assert exported["val_loss"] == pytest.approx(0.125)

    transferred, counts = inflate_encoder_state_dict(
        exported["encoder_state_dict"],
        {
            "weight": torch.zeros(3, 3, 1, 1, 1),
            "bias": torch.zeros(3),
        },
    )
    assert set(transferred) == {"weight", "bias"}
    assert counts["direct"] == 1
    assert counts["inflated"] == 1
    assert counts["skipped"] == 0


def test_pipeline_records_resolved_resume_source_in_result_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    import token_mixer.pipelines.pretrain_cnn as pipeline

    model = _TinyModel()
    source_manager = CheckpointManager(tmp_path / "source-checkpoints")
    source_path = source_manager.save("best", model, None, None, None, {"epoch": 1})
    result = FitResult(0.125, 1, [{"mse": 0.125}])
    cfg = _config(tmp_path)
    cfg["resume"] = str(source_path)

    monkeypatch.setattr(pipeline, "build_denoising_model", lambda _cfg: model)
    monkeypatch.setattr(
        pipeline,
        "build_dataloaders",
        lambda _cfg, _generator: ("train-loader", "val-loader"),
    )
    monkeypatch.setattr(pipeline, "fit", _fit_result_with_best(result))

    resumed = pipeline.run_cnn_denoising_pretrain(cfg)

    assert resumed.metadata is not None
    assert resumed.metadata["source_checkpoint"] == str(source_path)
    assert resumed.metadata["resume_mode"] == "exact"


def test_pipeline_reconstruction_grid_is_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    import token_mixer.pipelines.pretrain_cnn as pipeline

    model = _TinyModel()
    result = FitResult(0.125, 1, [{"mse": 0.125}])
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    monkeypatch.setattr(pipeline, "build_denoising_model", lambda _cfg: model)
    monkeypatch.setattr(
        pipeline,
        "build_dataloaders",
        lambda _cfg, _generator: ("train-loader", "val-loader"),
    )
    monkeypatch.setattr(pipeline, "fit", _fit_result_with_best(result))
    monkeypatch.setattr(
        pipeline,
        "_save_reconstruction_grid",
        lambda *args, **kwargs: calls.append((args, kwargs)),
        raising=False,
    )

    pipeline.run_cnn_denoising_pretrain(_config(tmp_path))

    assert calls == []


def test_pipeline_reconstruction_grid_runs_only_when_explicitly_enabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    import token_mixer.pipelines.pretrain_cnn as pipeline

    model = _TinyModel()
    result = FitResult(0.125, 1, [{"mse": 0.125}])
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    monkeypatch.setattr(pipeline, "build_denoising_model", lambda _cfg: model)
    monkeypatch.setattr(
        pipeline,
        "build_dataloaders",
        lambda _cfg, _generator: ("train-loader", "val-loader"),
    )
    monkeypatch.setattr(pipeline, "fit", _fit_result_with_best(result))
    monkeypatch.setattr(
        pipeline,
        "_save_reconstruction_grid",
        lambda *args, **kwargs: calls.append((args, kwargs)),
        raising=False,
    )
    cfg = _config(tmp_path)
    cfg["visualization"] = {"reconstruction_grid": True, "num_images": 2}

    pipeline.run_cnn_denoising_pretrain(cfg)

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[:3] == (model, "val-loader", tmp_path / "experiment")
    assert kwargs["n"] == 2


def test_run_pipeline_smokes_real_imagefolder_and_shared_engine(tmp_path: Path):
    image_module = pytest.importorskip("PIL.Image")
    import token_mixer.pipelines.pretrain_cnn as pipeline

    root = tmp_path / "images"
    for index, class_name in enumerate(("class_a", "class_b")):
        class_dir = root / class_name
        class_dir.mkdir(parents=True)
        image_module.new("RGB", (40, 40), (index * 100, 50, 25)).save(
            class_dir / "sample.png"
        )

    result = pipeline.run_cnn_denoising_pretrain(_config(tmp_path))

    assert isinstance(result, FitResult)
    assert result.best_epoch == 1
    assert (tmp_path / "experiment" / "encoder_best.pth").is_file()
