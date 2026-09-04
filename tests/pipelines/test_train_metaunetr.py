from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from token_mixer.data.cases import CaseRecord
from token_mixer.data.splits import SplitManifest, save_split_manifest
from token_mixer.training.engine import FitResult
from token_mixer.training.phases import PhaseSpec


def _config(variant: str = "mod_a") -> dict[str, object]:
    return {
        "variant": variant,
        "seed": 17,
        "deterministic": True,
        "model": {
            "in_channels": 4,
            "num_classes": 3,
            "base_channels": 4,
            "depths": (1, 1, 1, 1),
            "axis_fusion": "cat",
            "scan_direction": "forward",
        },
        "tracking": {"enabled": False},
        "spacing": (1.0, 1.0, 1.0),
        "phases": [
            {
                "name": "train",
                "epochs": 1,
                "freeze_encoder": False,
                "encoder_lr": 0.001,
                "decoder_lr": 0.001,
            }
        ],
        "device": "cpu",
    }


class _Tracker:
    pass


def _patch_runtime(monkeypatch: pytest.MonkeyPatch, pipeline, result: FitResult):
    events: list[tuple[str, Any]] = []
    tracker_configs: list[Mapping[str, object]] = []
    loader_generator = object()
    model = object()

    def seed_everything(seed: int, deterministic: bool = True):
        events.append(("seed", (seed, deterministic)))
        return loader_generator

    def build_loaders(cfg, generator):
        events.append(("loaders", (cfg, generator)))
        return "train-loader", "val-loader"

    def build_metaunetr(cfg, variant: str):
        events.append(("model", (cfg, variant)))
        return model

    def create_tracker(config, run_config):
        tracker_configs.append(dict(run_config))
        events.append(("tracker", config))
        return _Tracker()

    fit_calls: list[tuple[Any, ...]] = []

    def fit(*args):
        fit_calls.append(args)
        events.append(("fit", args))
        return result

    monkeypatch.setattr(pipeline, "seed_everything", seed_everything)
    monkeypatch.setattr(pipeline, "build_loaders", build_loaders)
    monkeypatch.setattr(pipeline, "build_metaunetr", build_metaunetr)
    monkeypatch.setattr(pipeline, "create_tracker", create_tracker)
    monkeypatch.setattr(pipeline, "fit", fit)
    return events, tracker_configs, fit_calls, model, loader_generator


def test_run_metaunetr_selects_each_allowed_variant_and_rejects_other_values(
    monkeypatch: pytest.MonkeyPatch,
):
    import token_mixer.pipelines.train_metaunetr as pipeline

    result = FitResult(0.0, 0, [])
    for variant in ("metaunetr_mamba", "mod_a", "mod_b"):
        events, _, _, _, _ = _patch_runtime(monkeypatch, pipeline, result)

        assert pipeline.run_metaunetr(_config(variant)) is result
        assert next(event for event in events if event[0] == "model")[1][1] == variant

    with pytest.raises(ValueError, match="variant"):
        pipeline.run_metaunetr(_config("unknown"))

    with pytest.raises(ValueError, match="exactly one"):
        pipeline.run_metaunetr({**_config(), "variant": ("mod_a", "mod_b")})


def test_run_metaunetr_propagates_variant_model_metadata_to_tracker(
    monkeypatch: pytest.MonkeyPatch,
):
    import token_mixer.pipelines.train_metaunetr as pipeline

    result = FitResult(0.0, 0, [])
    _, tracker_configs, _, _, _ = _patch_runtime(monkeypatch, pipeline, result)

    pipeline.run_metaunetr(_config())

    metadata = tracker_configs[0]
    assert metadata["variant"] == "mod_a"
    assert metadata["base_channels"] == 4
    assert metadata["base_widths"] == (4, 8, 16, 32)
    assert metadata["depths"] == (1, 1, 1, 1)
    assert metadata["scan_direction"] == "forward"
    assert metadata["axis_fusion"] == "cat"


def test_run_metaunetr_seeds_before_constructing_model_or_loaders(
    monkeypatch: pytest.MonkeyPatch,
):
    import token_mixer.pipelines.train_metaunetr as pipeline

    result = FitResult(0.0, 0, [])
    events, _, _, _, loader_generator = _patch_runtime(monkeypatch, pipeline, result)

    pipeline.run_metaunetr(_config())

    assert ("seed", (17, True)) in events
    assert events.index(next(event for event in events if event[0] == "seed")) < events.index(
        next(event for event in events if event[0] == "loaders")
    )
    assert events.index(next(event for event in events if event[0] == "seed")) < events.index(
        next(event for event in events if event[0] == "model")
    )
    loader_event = next(event for event in events if event[0] == "loaders")
    assert loader_event[1][1] is loader_generator


def test_run_metaunetr_returns_shared_engine_result_and_passes_contract_inputs(
    monkeypatch: pytest.MonkeyPatch,
):
    import token_mixer.pipelines.train_metaunetr as pipeline

    expected = FitResult(0.75, 3, [{"mean_dice": 0.75}])
    _, _, fit_calls, model, _ = _patch_runtime(monkeypatch, pipeline, expected)

    assert pipeline.run_metaunetr(_config()) is expected

    call = fit_calls[0]
    assert call[0] is model
    assert call[1:3] == ("train-loader", "val-loader")
    assert callable(call[3])
    assert callable(call[4])
    assert len(call[5]) == 1
    assert call[6]["variant"] == "mod_a"


def test_run_metaunetr_reloads_best_and_evaluates_held_out_test_loader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    import token_mixer.pipelines.train_metaunetr as pipeline

    model = nn.Linear(1, 1, bias=False)
    model.encoder = nn.Identity()
    expected = FitResult(0.75, 1, [{"mean_dice": 0.75}])
    cfg = _config()
    cfg["paths"] = {"checkpoint_dir": str(tmp_path / "checkpoints")}
    events: list[tuple[str, object]] = []
    checkpoints = pipeline._build_checkpoints(cfg)
    assert checkpoints is not None

    class _Bundle(tuple):
        metadata = {"manifest_hash": "manifest-sha256"}
        test_loader = "test-loader"

        def __new__(cls):
            return super().__new__(cls, ("train-loader", "val-loader"))

    monkeypatch.setattr(pipeline, "build_metaunetr", lambda *_args: model)
    monkeypatch.setattr(
        pipeline,
        "build_loaders",
        lambda *_args: _Bundle(),
    )
    monkeypatch.setattr(pipeline, "seed_everything", lambda *_args, **_kwargs: torch.Generator())
    monkeypatch.setattr(pipeline, "create_tracker", lambda *_args: object())
    monkeypatch.setattr(
        pipeline,
        "_build_evaluator",
        lambda *_args: lambda current_model, loader: events.append(
            (
                "evaluate",
                (current_model, loader, float(current_model.weight[0, 0].detach())),
            )
        )
        or {"mean_dice": 0.4},
    )

    def fake_fit(*args):
        with torch.no_grad():
            model.weight.fill_(1.0)
        args[-1].save(
            "best",
            model,
            None,
            None,
            None,
            {"epoch": 1, "metric": 0.75, "config": args[6]},
        )
        with torch.no_grad():
            model.weight.fill_(2.0)
        return expected

    monkeypatch.setattr(pipeline, "fit", fake_fit)

    result = pipeline.run_metaunetr(cfg)

    assert result.test_metrics == {"mean_dice": 0.4}
    assert events == [("evaluate", (model, "test-loader", 1.0))]


def test_metaunetr_consumes_nested_experiment_training_settings():
    import token_mixer.pipelines.train_metaunetr as pipeline

    cfg = OmegaConf.create(
        {
            "experiment": {
                "training": {
                    "loss": "dice_ce",
                    "optimizer": {"name": "sgd", "momentum": 0.25},
                    "scheduler": {"name": "step", "step_size": 4, "interval": "epoch"},
                    "phases": [
                        {
                            "name": "nested_phase",
                            "epochs": 7,
                            "freeze_encoder": True,
                            "encoder_lr": 0.0,
                            "decoder_lr": 0.002,
                        }
                    ],
                    "validation_interval": 3,
                    "use_amp": True,
                }
            }
        }
    )

    phases = pipeline._build_phases(cfg)
    engine_config = pipeline._engine_config(cfg, {})

    assert phases[0].name == "nested_phase"
    assert phases[0].epochs == 7
    assert engine_config["optimizer"] == {"name": "sgd", "momentum": 0.25}
    assert engine_config["scheduler"] == {
        "name": "step",
        "step_size": 4,
        "interval": "epoch",
    }
    assert engine_config["validation_interval"] == 3
    assert engine_config["use_amp"] is True


def test_metaunetr_resolves_nested_dice_ce_loss():
    import token_mixer.pipelines.train_metaunetr as pipeline

    cfg = OmegaConf.create({"experiment": {"training": {"loss": "dice_ce"}}})
    logits = torch.zeros(1, 3, 2, 2)
    target = torch.ones_like(logits)

    loss = pipeline._build_loss(cfg)(logits, target)
    binary_cross_entropy = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, target
    )

    assert loss > binary_cross_entropy


class _FakePatchDataset:
    def __init__(self, cases, config, training):
        self.cases = list(cases)
        self.config = config
        self.training = training

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        del index
        return (
            np.zeros((4, 32, 32, 32), dtype=np.float32),
            np.zeros((3, 32, 32, 32), dtype=np.float32),
        )


class _FakeVolumeDataset:
    def __init__(self, cases, config):
        self.cases = list(cases)
        self.config = config

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        case_id = self.cases[index].case_id
        depth = 32 if case_id.endswith("A") else 64
        return (
            np.zeros((4, depth, 32, 32), dtype=np.float32),
            np.zeros((3, depth, 32, 32), dtype=np.float32),
            case_id,
        )


def _loader_manifest(path: Path) -> SplitManifest:
    manifest = SplitManifest(
        seed=17,
        dataset_id="fixture-v1",
        train=["TRAIN"],
        val=["VALA", "VALB"],
        test=["TEST"],
        val_fraction=0.15,
        test_fraction=0.10,
    )
    save_split_manifest(manifest, path)
    return manifest


def _loader_config(manifest_path: Path) -> Any:
    return OmegaConf.create(
        {
            "paths": {
                "data_root": "unused-data-root",
                "manifest": str(manifest_path),
            },
            "dataset_id": "fixture-v1",
            "split_seed": 17,
            "val_fraction": 0.15,
            "test_fraction": 0.10,
            "batch_size": 2,
            "num_workers": 0,
            "data": {
                "patch_size": [32, 32, 32],
                "volume_size": [64, 64, 64],
                "nested": {
                    "crop_size": [32, 64, 32],
                    "deeper": {"roi_size": [32, 32, 64]},
                    "list": [[{"volume_size": [32, 32, 32]}]],
                },
            },
            "spacing": [1.0, 1.5, 2.0],
        }
    )


def _patch_fake_loader_dependencies(monkeypatch: pytest.MonkeyPatch, pipeline) -> None:
    cases = [
        CaseRecord(case_id, {}, Path("segmentation.nii.gz"))
        for case_id in ("TRAIN", "VALA", "VALB", "TEST")
    ]
    monkeypatch.setattr(pipeline, "discover_cases", lambda _root: cases)
    monkeypatch.setattr(pipeline, "BratsPatchDataset", _FakePatchDataset)
    monkeypatch.setattr(pipeline, "BratsVolumeDataset", _FakeVolumeDataset)


def test_build_loaders_uses_single_case_validation_batches_and_preserves_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    import token_mixer.pipelines.train_metaunetr as pipeline

    manifest_path = tmp_path / "split.json"
    manifest = _loader_manifest(manifest_path)
    _patch_fake_loader_dependencies(monkeypatch, pipeline)
    load_calls: list[Path] = []
    real_load_manifest = pipeline.load_split_manifest

    def load_manifest(path: Path):
        load_calls.append(path)
        return real_load_manifest(path)

    monkeypatch.setattr(pipeline, "load_split_manifest", load_manifest)

    loaders = pipeline.build_loaders(_loader_config(manifest_path), torch.Generator())
    train_loader, val_loader = loaders

    assert train_loader.batch_size == 2
    assert val_loader.batch_size == 1
    assert [batch[0].shape[0] for batch in val_loader] == [1, 1]

    crop_keys = {"patch_size", "crop_size", "roi_size", "volume_size"}

    def assert_no_crop_keys(value: Any) -> None:
        if isinstance(value, Mapping):
            assert not crop_keys.intersection(value)
            for nested in value.values():
                assert_no_crop_keys(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                assert_no_crop_keys(nested)

    assert_no_crop_keys(val_loader.dataset.config)

    metadata = loaders.metadata
    assert metadata["manifest_path"] == str(manifest_path)
    assert metadata["manifest_hash"] == hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    assert metadata["dataset_id"] == manifest.dataset_id
    assert metadata["split_seed"] == manifest.seed
    assert metadata["val_fraction"] == manifest.val_fraction
    assert metadata["test_fraction"] == manifest.test_fraction
    assert metadata["split_counts"] == {"train": 1, "val": 2, "test": 1}
    assert load_calls == [manifest_path]


def test_build_loaders_applies_run_max_cases_to_train_and_validation_splits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    import token_mixer.pipelines.train_metaunetr as pipeline

    manifest_path = tmp_path / "split.json"
    save_split_manifest(
        SplitManifest(
            seed=17,
            dataset_id="fixture-v1",
            train=["TRAIN1", "TRAIN2"],
            val=["VALA", "VALB"],
            test=["TEST1", "TEST2"],
            val_fraction=0.15,
            test_fraction=0.10,
        ),
        manifest_path,
    )
    cases = [
        CaseRecord(case_id, {}, Path("segmentation.nii.gz"))
        for case_id in ("TRAIN1", "TRAIN2", "VALA", "VALB", "TEST1", "TEST2")
    ]
    monkeypatch.setattr(pipeline, "discover_cases", lambda _root: cases)
    monkeypatch.setattr(pipeline, "BratsPatchDataset", _FakePatchDataset)
    monkeypatch.setattr(pipeline, "BratsVolumeDataset", _FakeVolumeDataset)
    config = _loader_config(manifest_path)
    config["run"] = {"max_cases": 1}

    train_loader, val_loader = pipeline.build_loaders(config, torch.Generator())

    assert [case.case_id for case in train_loader.dataset.cases] == ["TRAIN1"]
    assert [case.case_id for case in val_loader.dataset.cases] == ["VALA"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dataset_id", "other-dataset"),
        ("split_seed", 99),
        ("val_fraction", 0.20),
        ("test_fraction", 0.20),
    ],
)
def test_build_loaders_rejects_configured_manifest_metadata_mismatches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    value: Any,
):
    import token_mixer.pipelines.train_metaunetr as pipeline

    manifest_path = tmp_path / "split.json"
    _loader_manifest(manifest_path)
    _patch_fake_loader_dependencies(monkeypatch, pipeline)
    config = _loader_config(manifest_path)
    config[field] = value

    with pytest.raises(ValueError, match="does not match"):
        pipeline.build_loaders(config, torch.Generator())


def test_build_loaders_rejects_duplicate_discovered_case_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    import token_mixer.pipelines.train_metaunetr as pipeline

    manifest_path = tmp_path / "split.json"
    _loader_manifest(manifest_path)
    duplicate_cases = [
        CaseRecord("TRAIN", {}, Path("segmentation.nii.gz")),
        CaseRecord("TRAIN", {}, Path("segmentation.nii.gz")),
    ]
    monkeypatch.setattr(pipeline, "discover_cases", lambda _root: duplicate_cases)

    with pytest.raises(ValueError, match="duplicate"):
        pipeline.build_loaders(_loader_config(manifest_path), torch.Generator())


def test_build_loaders_rejects_missing_test_case_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    import token_mixer.pipelines.train_metaunetr as pipeline

    manifest_path = tmp_path / "split.json"
    _loader_manifest(manifest_path)
    cases = [
        CaseRecord(case_id, {}, Path("segmentation.nii.gz"))
        for case_id in ("TRAIN", "VALA", "VALB")
    ]
    monkeypatch.setattr(pipeline, "discover_cases", lambda _root: cases)

    with pytest.raises(ValueError, match="missing"):
        pipeline.build_loaders(_loader_config(manifest_path), torch.Generator())


def test_run_metaunetr_accepts_dictconfig_and_propagates_resolved_execution_device(
    monkeypatch: pytest.MonkeyPatch,
):
    import token_mixer.pipelines.train_metaunetr as pipeline

    result = FitResult(0.0, 0, [])
    events, _, _, _, _ = _patch_runtime(monkeypatch, pipeline, result)
    config = _config()
    config.pop("device")
    config["training"] = {"device": "cpu"}

    assert pipeline.run_metaunetr(OmegaConf.create(config)) is result
    model_event = next(event for event in events if event[0] == "model")
    assert model_event[1][0]["execution_device"] == torch.device("cpu")


def test_effective_auto_device_keeps_cpu_fallback_without_backend_side_effects(
    monkeypatch: pytest.MonkeyPatch,
):
    import token_mixer.pipelines.train_metaunetr as pipeline

    monkeypatch.setattr(pipeline.torch.cuda, "is_available", lambda: False)
    deterministic = pipeline.torch.backends.cudnn.deterministic
    benchmark = pipeline.torch.backends.cudnn.benchmark

    assert pipeline._effective_device({"device": "auto"}) == torch.device("cpu")
    assert pipeline.torch.backends.cudnn.deterministic is deterministic
    assert pipeline.torch.backends.cudnn.benchmark is benchmark


@pytest.mark.parametrize(
    "spacing",
    [(1.0, 0.0, 1.0), (1.0, float("nan"), 1.0), (1.0, 2.0)],
)
def test_build_evaluator_rejects_invalid_configured_spacing(spacing):
    import token_mixer.pipelines.train_metaunetr as pipeline

    config = _config()
    config["spacing"] = spacing

    with pytest.raises(ValueError, match="spacing"):
        pipeline._build_evaluator(config)


def test_build_evaluator_requires_explicit_spacing_for_volume_batches():
    import token_mixer.pipelines.train_metaunetr as pipeline

    config = _config()
    config.pop("spacing")

    with pytest.raises(ValueError, match="spacing"):
        pipeline._build_evaluator(config)


def test_build_evaluator_passes_canonical_spacing_to_full_volume_evaluation(
    monkeypatch: pytest.MonkeyPatch,
):
    import token_mixer.pipelines.train_metaunetr as pipeline

    calls: list[dict[str, Any]] = []

    def evaluate(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return {"mean_dice": 0.0}

    monkeypatch.setattr(pipeline, "evaluate_full_volumes", evaluate)
    evaluator = pipeline._build_evaluator(_config())

    assert evaluator("model", "loader") == {"mean_dice": 0.0}
    assert calls[0]["kwargs"]["default_spacing"] == (1.0, 1.0, 1.0)
    assert calls[0]["args"][2] == (96, 96, 96)
    assert calls[0]["args"][5] == torch.device("cpu")


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("roi_size", (30, 32, 32)),
        ("sw_batch_size", 0),
        ("overlap", 1.0),
    ],
)
def test_build_evaluator_rejects_invalid_inference_geometry(key: str, value: Any):
    import token_mixer.pipelines.train_metaunetr as pipeline

    config = _config()
    config[key] = value

    with pytest.raises(ValueError):
        pipeline._build_evaluator(config)


def test_run_metaunetr_rejects_nondivisible_training_spatial_size_before_model(
    monkeypatch: pytest.MonkeyPatch,
):
    import token_mixer.pipelines.train_metaunetr as pipeline

    result = FitResult(0.0, 0, [])
    events, _, _, _, _ = _patch_runtime(monkeypatch, pipeline, result)
    config = _config()
    config["data"] = {"patch_size": (48, 32, 32)}

    with pytest.raises(ValueError, match="divisible"):
        pipeline.run_metaunetr(config)

    assert not any(event[0] == "model" for event in events)


def test_run_metaunetr_accepts_only_forward_scan_direction_and_keeps_it_metadata_only(
    monkeypatch: pytest.MonkeyPatch,
):
    import token_mixer.pipelines.train_metaunetr as pipeline

    result = FitResult(0.0, 0, [])
    events, tracker_configs, _, _, _ = _patch_runtime(monkeypatch, pipeline, result)
    config = _config()
    config["model"]["scan_direction"] = "reverse"

    with pytest.raises(ValueError, match="scan_direction"):
        pipeline.run_metaunetr(config)
    assert not any(event[0] == "model" for event in events)

    config["model"]["scan_direction"] = "forward"
    pipeline.run_metaunetr(config)
    model_cfg = next(event for event in events if event[0] == "model")[1][0]
    assert "scan_direction" not in model_cfg
    assert tracker_configs[-1]["scan_direction"] == "forward"


def test_run_metaunetr_rejects_nested_experiment_model_scan_direction(
    monkeypatch: pytest.MonkeyPatch,
):
    import token_mixer.pipelines.train_metaunetr as pipeline

    result = FitResult(0.0, 0, [])
    events, _, _, _, _ = _patch_runtime(monkeypatch, pipeline, result)
    config = _config()
    config.pop("model")
    config["experiment"] = {"model": {"scan_direction": "reverse"}}

    with pytest.raises(ValueError, match="scan_direction"):
        pipeline.run_metaunetr(config)

    assert not any(event[0] == "model" for event in events)


def test_run_metaunetr_passes_manifest_provenance_and_spacing_to_tracker(
    monkeypatch: pytest.MonkeyPatch,
):
    import token_mixer.pipelines.train_metaunetr as pipeline

    result = FitResult(0.0, 0, [])
    events, tracker_configs, _, _, _ = _patch_runtime(monkeypatch, pipeline, result)

    class _LoaderResult(tuple):
        def __new__(cls):
            value = super().__new__(cls, ("train-loader", "val-loader"))
            value.metadata = {
                "manifest_path": "split.json",
                "manifest_hash": "sha256",
                "dataset_id": "fixture-v1",
                "split_seed": 17,
                "val_fraction": 0.15,
                "test_fraction": 0.10,
                "split_counts": {"train": 1, "val": 1, "test": 1},
            }
            return value

    monkeypatch.setattr(
        pipeline,
        "build_loaders",
        lambda _cfg, _generator: _LoaderResult(),
    )

    pipeline.run_metaunetr(_config())

    metadata = tracker_configs[-1]
    assert metadata["manifest_path"] == "split.json"
    assert metadata["manifest_hash"] == "sha256"
    assert metadata["dataset_id"] == "fixture-v1"
    assert metadata["split_seed"] == 17
    assert metadata["split_counts"] == {"train": 1, "val": 1, "test": 1}
    assert metadata["spacing"] == (1.0, 1.0, 1.0)
    assert any(event[0] == "fit" for event in events)


def test_run_metaunetr_rejects_missing_loader_metadata_without_reloading_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    import token_mixer.pipelines.train_metaunetr as pipeline

    result = FitResult(0.0, 0, [])
    _patch_runtime(monkeypatch, pipeline, result)
    manifest_path = tmp_path / "split.json"
    _loader_manifest(manifest_path)
    real_load_manifest = pipeline.load_split_manifest
    load_calls: list[Path] = []

    def load_manifest(path: Path):
        load_calls.append(path)
        return real_load_manifest(path)

    def build_loaders(_cfg, _generator):
        pipeline.load_split_manifest(manifest_path)
        return "train-loader", "val-loader"

    monkeypatch.setattr(pipeline, "load_split_manifest", load_manifest)
    monkeypatch.setattr(pipeline, "build_loaders", build_loaders)
    config = _config()
    config["paths"] = {"manifest": str(manifest_path)}

    with pytest.raises(ValueError, match="loader metadata"):
        pipeline.run_metaunetr(config)

    assert load_calls == [manifest_path]


def test_run_metaunetr_builds_dependencies_before_tracker(
    monkeypatch: pytest.MonkeyPatch,
):
    import token_mixer.pipelines.train_metaunetr as pipeline

    result = FitResult(0.0, 0, [])
    events, _, _, _, _ = _patch_runtime(monkeypatch, pipeline, result)
    dependency_events: list[str] = []
    order: list[str] = []
    real_create_tracker = pipeline.create_tracker

    def create_tracker(config, run_config):
        order.append("tracker")
        return real_create_tracker(config, run_config)

    monkeypatch.setattr(pipeline, "create_tracker", create_tracker)

    def build_loss(_cfg):
        dependency_events.append("loss")
        order.append("loss")
        return lambda *_args: torch.tensor(0.0)

    def build_evaluator(_cfg, _device=None):
        dependency_events.append("evaluator")
        order.append("evaluator")
        return lambda *_args: {"mean_dice": 0.0}

    def build_phases(_cfg):
        dependency_events.append("phases")
        order.append("phases")
        return [PhaseSpec("train", 1, False, 0.001, 0.001)]

    def build_checkpoints(_cfg):
        dependency_events.append("checkpoints")
        order.append("checkpoints")
        return None

    monkeypatch.setattr(
        pipeline,
        "_build_loss",
        build_loss,
    )
    monkeypatch.setattr(pipeline, "_build_evaluator", build_evaluator)
    monkeypatch.setattr(pipeline, "_build_phases", build_phases)
    monkeypatch.setattr(pipeline, "_build_checkpoints", build_checkpoints)

    pipeline.run_metaunetr(_config())

    assert dependency_events == ["loss", "evaluator", "phases", "checkpoints"]
    assert order == ["loss", "evaluator", "phases", "checkpoints", "tracker"]
    assert any(event[0] == "tracker" for event in events)


def test_run_metaunetr_does_not_create_tracker_when_dependency_configuration_fails(
    monkeypatch: pytest.MonkeyPatch,
):
    import token_mixer.pipelines.train_metaunetr as pipeline

    result = FitResult(0.0, 0, [])
    events, _, _, _, _ = _patch_runtime(monkeypatch, pipeline, result)

    def fail_phases(_cfg):
        raise ValueError("invalid phases")

    monkeypatch.setattr(pipeline, "_build_phases", fail_phases)

    with pytest.raises(ValueError, match="invalid phases"):
        pipeline.run_metaunetr(_config())
    assert not any(event[0] == "tracker" for event in events)


class _TinyPipelineModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv3d(4, 1, kernel_size=1)
        self.decoder = nn.Conv3d(1, 3, kernel_size=1)

    def forward(self, image):
        return self.decoder(self.encoder(image))


def test_run_metaunetr_smokes_real_dataloaders_and_engine_without_dataset_files(
    monkeypatch: pytest.MonkeyPatch,
):
    import token_mixer.pipelines.train_metaunetr as pipeline

    images = torch.zeros((1, 4, 32, 32, 32), dtype=torch.float32)
    targets = torch.zeros((1, 3, 32, 32, 32), dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(images, targets), batch_size=1, shuffle=False)
    val_loader = DataLoader(TensorDataset(images, targets), batch_size=1, shuffle=False)
    monkeypatch.setattr(
        pipeline,
        "build_metaunetr",
        lambda _cfg, _variant: _TinyPipelineModel(),
    )
    monkeypatch.setattr(
        pipeline,
        "build_loaders",
        lambda _cfg, _generator: (train_loader, val_loader),
    )

    config = _config()
    config["loss_fn"] = nn.MSELoss()
    config["evaluator"] = lambda _model, _loader: {"mean_dice": 0.0}
    result = pipeline.run_metaunetr(config)

    assert isinstance(result, FitResult)
    assert result.best_epoch == 1
