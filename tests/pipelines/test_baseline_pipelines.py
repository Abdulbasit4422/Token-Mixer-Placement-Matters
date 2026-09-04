from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from token_mixer.data.cases import CaseRecord
from token_mixer.data.splits import SplitManifest, save_split_manifest
from token_mixer.pipelines import _baseline_common as common
from token_mixer.training.engine import FitResult


class _Tiny3D(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Conv3d(4, 2, kernel_size=1)
        self.decoder = nn.Conv3d(2, 3, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(image))


class _Tiny2D(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Conv2d(4, 2, kernel_size=1)
        self.decoder = nn.Conv2d(2, 3, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(image))


class _RegionLogitModel(nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return image[:, :3]


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
            torch.zeros(4, 8, 8, 8),
            torch.zeros(3, 8, 8, 8),
        )


class _FakeVolumeDataset:
    def __init__(self, cases, config):
        self.cases = list(cases)
        self.config = config

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        case_id = self.cases[index].case_id
        return (
            torch.zeros(4, 8, 8, 8),
            torch.zeros(3, 8, 8, 8),
            case_id,
        )


class _FakeSliceDataset(_FakeVolumeDataset):
    def __init__(self, cases, config, training):
        super().__init__(cases, config)
        self.training = training


def _manifest(path: Path) -> SplitManifest:
    manifest = SplitManifest(
        seed=17,
        dataset_id="fixture-v1",
        train=["train"],
        val=["val"],
        test=["test"],
        val_fraction=0.2,
        test_fraction=0.2,
    )
    save_split_manifest(manifest, path)
    return manifest


def _volume_config(tmp_path: Path, manifest_path: Path) -> Any:
    return OmegaConf.create(
        {
            "paths": {
                "data_root": str(tmp_path / "data"),
                "manifest": str(manifest_path),
                "checkpoint_dir": str(tmp_path / "checkpoints"),
            },
            "dataset_id": "fixture-v1",
            "split_seed": 17,
            "val_fraction": 0.2,
            "test_fraction": 0.2,
            "batch_size": 1,
            "num_workers": 0,
            "spacing": [1.0, 1.5, 2.0],
            "device": "cpu",
            "seed": 17,
            "deterministic": True,
            "phases": [
                {
                    "name": "train",
                    "epochs": 1,
                    "freeze_encoder": False,
                    "encoder_lr": 0.001,
                    "decoder_lr": 0.001,
                }
            ],
            "tracking": {"enabled": False},
        }
    )


def test_build_volume_loaders_uses_persisted_train_val_test_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    manifest_path = tmp_path / "split.json"
    manifest = _manifest(manifest_path)
    cases = [
        CaseRecord(case_id, {}, Path("segmentation.nii.gz"))
        for case_id in ("train", "val", "test")
    ]
    monkeypatch.setattr(common, "discover_cases", lambda _root: cases)
    monkeypatch.setattr(common, "BratsPatchDataset", _FakePatchDataset)
    monkeypatch.setattr(common, "BratsVolumeDataset", _FakeVolumeDataset)

    bundle = common.build_volume_loaders(_volume_config(tmp_path, manifest_path), torch.Generator())
    train_loader, val_loader, test_loader = bundle

    assert [case.case_id for case in train_loader.dataset.cases] == manifest.train
    assert [case.case_id for case in val_loader.dataset.cases] == manifest.val
    assert [case.case_id for case in test_loader.dataset.cases] == manifest.test
    assert val_loader.batch_size == 1
    assert test_loader.batch_size == 1
    assert bundle.metadata["manifest_hash"]
    assert bundle.metadata["split_counts"] == {"train": 1, "val": 1, "test": 1}


def test_baseline_loaders_apply_run_max_cases_to_all_splits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    manifest_path = tmp_path / "split.json"
    manifest = SplitManifest(
        seed=17,
        dataset_id="fixture-v1",
        train=["train-a", "train-b"],
        val=["val-a", "val-b"],
        test=["test-a", "test-b"],
        val_fraction=0.2,
        test_fraction=0.2,
    )
    save_split_manifest(manifest, manifest_path)
    cases = [
        CaseRecord(case_id, {}, Path("segmentation.nii.gz"))
        for case_id in ("train-a", "train-b", "val-a", "val-b", "test-a", "test-b")
    ]
    monkeypatch.setattr(common, "discover_cases", lambda _root: cases)
    monkeypatch.setattr(common, "BratsPatchDataset", _FakePatchDataset)
    monkeypatch.setattr(common, "BratsVolumeDataset", _FakeVolumeDataset)
    monkeypatch.setattr(common, "BratsSliceDataset", _FakeSliceDataset)
    config = _volume_config(tmp_path, manifest_path)
    config["run"] = {"max_cases": 1}

    volume_bundle = common.build_volume_loaders(config, torch.Generator())
    slice_bundle = common.build_slice_loaders(config, torch.Generator())

    for bundle in (volume_bundle, slice_bundle):
        train_loader, val_loader, test_loader = bundle
        assert [case.case_id for case in train_loader.dataset.cases] == ["train-a"]
        assert [case.case_id for case in val_loader.dataset.cases] == ["val-a"]
        assert [case.case_id for case in test_loader.dataset.cases] == ["test-a"]


def test_baseline_resolves_nested_experiment_training_settings_and_dice_ce():
    from monai.losses import DiceCELoss

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
                            "freeze_encoder": False,
                            "encoder_lr": 0.001,
                            "decoder_lr": 0.002,
                        }
                    ],
                    "validation_interval": 3,
                    "use_amp": True,
                }
            }
        }
    )

    phases = common.build_phases(cfg)
    engine_config = common._engine_config(cfg, {}, torch.device("cpu"))
    logits = torch.zeros(1, 3, 2, 2)
    target = torch.ones_like(logits)
    loss = common.build_loss(cfg)(logits, target)
    binary_cross_entropy = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, target
    )

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
    assert isinstance(common.build_loss(cfg), DiceCELoss)
    assert loss > binary_cross_entropy


def test_fit_result_keeps_old_positional_constructor_and_exposes_pipeline_fields():
    result = FitResult(0.5, 2, [{"mean_dice": 0.5}])
    assert result.test_metrics is None
    assert result.metadata is None

    enriched = FitResult(
        0.5,
        2,
        result.history,
        {"mean_dice": 0.4},
        {"architecture": "fixture"},
    )
    assert enriched.test_metrics == {"mean_dice": 0.4}
    assert enriched.metadata == {"architecture": "fixture"}


@pytest.mark.parametrize(
    ("module_name", "run_name", "builder_name", "model"),
    [
        ("train_resunet3d", "run_resunet3d", "build_resunet3d", _Tiny3D()),
        ("train_swinunetr", "run_swinunetr", "build_swinunetr", _Tiny3D()),
        ("train_transunet", "run_transunet", "build_transunet", _Tiny2D()),
    ],
)
def test_baseline_entrypoints_reload_best_before_test_and_return_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    module_name: str,
    run_name: str,
    builder_name: str,
    model: nn.Module,
):
    import importlib

    pipeline = importlib.import_module(f"token_mixer.pipelines.{module_name}")
    cfg = _volume_config(tmp_path, tmp_path / "unused-manifest.json")
    cfg["third_party"] = {"transunet_root": str(tmp_path / "TransUNet")}
    cfg["third_party"]["pretrained_path"] = str(tmp_path / "weights.npz")

    loader_metadata = {
        "manifest_path": "split.json",
        "manifest_hash": "manifest-sha256",
        "dataset_id": "fixture-v1",
        "split_seed": 17,
        "val_fraction": 0.2,
        "test_fraction": 0.2,
        "split_counts": {"train": 1, "val": 1, "test": 1},
    }
    monkeypatch.setattr(pipeline, builder_name, lambda _cfg: model)
    monkeypatch.setattr(
        pipeline,
        "build_loaders",
        lambda _cfg, _generator: ("train-loader", "val-loader", "test-loader", loader_metadata),
    )
    monkeypatch.setattr(pipeline, "seed_everything", lambda *_args, **_kwargs: torch.Generator())
    monkeypatch.setattr(pipeline, "create_tracker", lambda *_args: object())
    if module_name == "train_transunet":
        monkeypatch.setattr(pipeline, "validate_transunet_config", lambda _cfg: {})

    seen: list[tuple[Any, Any, float]] = []

    def evaluator(current_model, loader):
        weight = float(next(current_model.parameters()).detach().flatten()[0])
        seen.append((current_model, loader, weight))
        return {"mean_dice": 0.25, "ET_dice": 0.25}

    monkeypatch.setattr(pipeline, "_build_evaluator", lambda *_args: evaluator)

    expected = FitResult(0.25, 1, [{"mean_dice": 0.25}])

    def fake_fit(*args, **kwargs):
        del kwargs
        checkpoints = args[-1]
        current_model = args[0]
        with torch.no_grad():
            next(current_model.parameters()).fill_(1.0)
        checkpoints.save(
            "best",
            current_model,
            None,
            None,
            None,
            {"epoch": 1, "metric": 0.25, "config": args[6]},
        )
        with torch.no_grad():
            next(current_model.parameters()).fill_(2.0)
        return expected

    monkeypatch.setattr(pipeline, "fit", fake_fit)

    result = getattr(pipeline, run_name)(cfg)

    assert result.test_metrics["mean_dice"] == pytest.approx(0.25)
    assert result.metadata["architecture"]
    assert result.metadata["manifest_hash"] == "manifest-sha256"
    if module_name != "train_transunet":
        assert result.metadata["spacing"] == (1.0, 1.5, 2.0)
    else:
        assert result.metadata["dimensionality"] == "2-D"
    assert seen[-1][1] == "test-loader"
    assert seen[-1][2] == pytest.approx(1.0)


def test_transunet_rejects_missing_external_checkout_before_model_or_fit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    import token_mixer.pipelines.train_transunet as pipeline

    cfg = _volume_config(tmp_path, tmp_path / "unused-manifest.json")
    cfg["third_party"] = {
        "transunet_root": str(tmp_path / "missing-transunet"),
        "pretrained_path": str(tmp_path / "missing-weights.npz"),
    }
    events: list[str] = []
    monkeypatch.setattr(pipeline, "seed_everything", lambda *_args, **_kwargs: events.append("seed"))
    monkeypatch.setattr(pipeline, "build_transunet", lambda _cfg: events.append("model"))
    monkeypatch.setattr(pipeline, "fit", lambda *_args, **_kwargs: events.append("fit"))

    with pytest.raises(FileNotFoundError, match="TransUNet"):
        pipeline.run_transunet(cfg)

    assert events == []


def test_slice_evaluator_converts_multiclass_targets_to_canonical_regions():
    model = _RegionLogitModel().eval()
    labels = torch.zeros(1, 4, 4, dtype=torch.long)
    labels[:, :2, :2] = 1
    labels[:, 2:, :2] = 2
    labels[:, :, 2:] = 3
    regions = torch.zeros(1, 3, 4, 4)
    regions[:, 0] = labels == 1
    regions[:, 1] = (labels == 1) | (labels == 2)
    regions[:, 2] = labels > 0
    images = torch.zeros(1, 4, 4, 4)
    images[:, :3] = regions * 10.0 - 5.0

    metrics = common.evaluate_slices(model, [(images, labels)])

    assert set(("ET_dice", "TC_dice", "WT_dice", "mean_dice")) <= metrics.keys()
    assert metrics["ET_dice"] == pytest.approx(1.0)
    assert metrics["TC_dice"] == pytest.approx(1.0)
    assert metrics["WT_dice"] == pytest.approx(1.0)


def test_slice_evaluator_averages_metrics_per_slice_across_uneven_batches():
    model = _RegionLogitModel().eval()
    bad_labels = torch.zeros(1, 4, 4, dtype=torch.long)
    bad_labels[:, 0, 0] = 1
    bad_images = torch.full((1, 4, 4, 4), -5.0)
    good_labels = torch.zeros(2, 4, 4, dtype=torch.long)
    good_images = torch.full((2, 4, 4, 4), -5.0)

    metrics = common.evaluate_slices(
        model,
        [
            (bad_images, bad_labels),
            (good_images, good_labels),
        ],
    )

    assert metrics["ET_dice"] == pytest.approx(2.0 / 3.0)
