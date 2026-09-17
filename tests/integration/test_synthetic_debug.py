from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from token_mixer.cli import _save_composed_config
from token_mixer.data.splits import SplitManifest, save_split_manifest
from token_mixer.evaluation.benchmark import (
    BenchmarkResult,
    hash_case_id,
    run_model_protocol,
    serialize_benchmark,
)
from token_mixer.evaluation.inference import (
    build_segmentation_snapshotter,
    evaluate_full_volumes,
)
from token_mixer.evaluation.metrics import REGION_NAMES
from token_mixer.models import weight_transfer
from token_mixer.models.cnn_pretrain import build_denoising_model, mse_loss
from token_mixer.models.metaunetr.variants import build_metaunetr
from token_mixer.models.resunet3d import build_resunet3d
from token_mixer.models.transunet import build_transunet
from token_mixer.pipelines._baseline_common import build_slice_loaders, build_volume_loaders
from token_mixer.pipelines.prepare_data import run_prepare
from token_mixer.pipelines.train_resunet3d import run_resunet3d
from token_mixer.reproducibility import seed_everything
from token_mixer.training.artifacts import write_run_artifacts
from token_mixer.training.checkpoints import CheckpointManager
from token_mixer.training.engine import FitResult
from token_mixer.training.tracking import Tracker, create_tracker


class _TinyTransUNetBackend(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Conv2d(4, 4, kernel_size=1)
        self.decoder = nn.Conv2d(4, 4, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(image))


class _RegionIdentity(nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return image[:, :3]


def _synthetic_resnet_state(model: nn.Module) -> dict[str, torch.Tensor]:
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


def _write_source_fixture(root: Path) -> list[str]:
    nib = pytest.importorskip("nibabel")
    case_ids = [f"case-{index:02d}" for index in range(4)]
    shape = (32, 32, 32)
    for case_index, case_id in enumerate(case_ids):
        case_root = root / case_id
        case_root.mkdir(parents=True)
        segmentation = np.zeros(shape, dtype=np.uint8)
        start = 6 + case_index
        segmentation[start : start + 4, 8:12, 8:12] = 4
        segmentation[start + 4 : start + 8, 8:12, 8:12] = 1
        segmentation[start + 8 : start + 12, 8:12, 8:12] = 2
        image = np.stack(
            (
                np.where(segmentation == 4, 5.0, -5.0),
                np.where(np.isin(segmentation, (1, 4)), 5.0, -5.0),
                np.where(segmentation > 0, 5.0, -5.0),
                np.full(shape, 0.25, dtype=np.float32),
            )
        ).astype(np.float32)
        affine = np.eye(4, dtype=np.float32)
        for channel, name in enumerate(("t1n", "t1c", "t2w", "t2f")):
            nib.save(nib.Nifti1Image(image[channel], affine), case_root / f"{name}.nii.gz")
        nib.save(
            nib.Nifti1Image(segmentation, affine),
            case_root / "segmentation.nii.gz",
        )
    return case_ids


def _segmentation_config(
    data_root: Path,
    manifest_path: Path,
    checkpoint_root: Path,
) -> object:
    return OmegaConf.create(
        {
            "paths": {
                "data_root": str(data_root),
                "manifest": str(manifest_path),
                "checkpoint_dir": str(checkpoint_root),
            },
            "dataset_id": "synthetic-v1",
            "split_seed": 42,
            "val_fraction": 0.25,
            "test_fraction": 0.25,
            "et_label": 4,
            "normalize": False,
            "patch_size": [32, 32, 32],
            "spacing": [1.0, 1.0, 1.0],
            "batch_size": 1,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "drop_last": False,
            "device": "cpu",
            "seed": 42,
            "deterministic": True,
            "tracking": {"enabled": False, "mode": "disabled"},
        }
    )


def _assert_forward_loss_backward(
    model: nn.Module,
    image: torch.Tensor,
    target: torch.Tensor,
    expected_shape: tuple[int, ...],
) -> None:
    model.train()
    model.zero_grad(set_to_none=True)
    model_input = image.detach().clone().requires_grad_(True)
    logits = model(model_input)
    assert tuple(logits.shape) == expected_shape
    loss = F.binary_cross_entropy_with_logits(logits, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert model_input.grad is not None
    assert torch.isfinite(model_input.grad).all()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert gradients
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)


def test_four_case_debug_harness_covers_prepare_models_checkpoint_metrics_and_artifacts(
    tmp_path: Path,
):
    pytest.importorskip("monai")
    seed_everything(42, deterministic=True)

    source_root = tmp_path / "source"
    case_ids = _write_source_fixture(source_root)
    data_root = tmp_path / "prepared"
    run_prepare(
        OmegaConf.create(
            {
                "paths": {
                    "source_root": str(source_root),
                    "data_root": str(data_root),
                }
            }
        )
    )
    assert (data_root / "case_index.json").is_file()

    manifest_path = tmp_path / "data" / "manifests" / "brats_seed42.json"
    manifest = SplitManifest(
        seed=42,
        dataset_id="synthetic-v1",
        train=case_ids[:2],
        val=case_ids[2:3],
        test=case_ids[3:],
        val_fraction=0.25,
        test_fraction=0.25,
    )
    save_split_manifest(manifest, manifest_path)
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["train"] == case_ids[:2]

    output_dir = tmp_path / "outputs" / "synthetic_debug" / "run-42"
    checkpoint_root = output_dir / "checkpoints"
    segmentation_cfg = _segmentation_config(data_root, manifest_path, checkpoint_root)
    volume_bundle = build_volume_loaders(segmentation_cfg, torch.Generator().manual_seed(42))
    train_loader, val_loader, test_loader = volume_bundle
    image, target = next(iter(train_loader))[:2]
    image = image.float()
    target = target.float()
    assert tuple(image.shape) == (1, 4, 32, 32, 32)
    assert tuple(target.shape) == (1, 3, 32, 32, 32)

    model_cfg = OmegaConf.create(
        {
            "in_channels": 4,
            "num_classes": 3,
            "base_channels": 2,
            "depths": [1, 1, 1, 1],
            "window_size": 1,
            "num_heads": [1, 1, 1, 1],
            "d_state": 1,
            "d_conv": 1,
            "mamba_expand": 1,
            "axis_fusion": "sum",
            "execution_device": "cpu",
            "norm_name": "group",
            "norm_num_groups": 1,
        }
    )
    for variant in ("metaunetr_mamba", "mod_a", "mod_b"):
        _assert_forward_loss_backward(
            build_metaunetr(model_cfg, variant),
            image,
            target,
            (1, 3, 32, 32, 32),
        )

    res_cfg = OmegaConf.create(
        {
            "model": {
                "in_channels": 4,
                "out_channels": 3,
                "base_features": 1,
                "depths": [1, 1, 1, 1, 1],
                "normalization": "group",
                "norm_num_groups": 1,
                "spatial_size": [32, 32, 32],
            }
        }
    )
    res_model = build_resunet3d(res_cfg)
    _assert_forward_loss_backward(res_model, image, target, (1, 3, 32, 32, 32))

    pipeline_output = output_dir / "resunet_pipeline"
    pipeline_cfg = _segmentation_config(
        data_root,
        manifest_path,
        pipeline_output / "checkpoints",
    )
    pipeline_cfg["experiment"] = {"name": "resunet3d"}
    pipeline_cfg["model"] = {
        "in_channels": 4,
        "out_channels": 3,
        "base_features": 1,
        "depths": [2, 2, 2, 2, 2],
        "normalization": "group",
        "norm_num_groups": 1,
    }
    pipeline_cfg["imagenet_transfer"] = {
        "enabled": True,
        "download": False,
        "source": "synthetic-pretraining",
    }
    source_model = build_resunet3d(pipeline_cfg)
    pipeline_result = run_resunet3d(
        pipeline_cfg,
        source_model=_synthetic_resnet_state(source_model),
    )
    assert isinstance(pipeline_result, FitResult)
    assert pipeline_result.best_epoch == 1
    assert pipeline_result.metadata is not None
    transfer_report = pipeline_result.metadata["weight_transfer"]
    assert transfer_report["status"] == "loaded"
    assert transfer_report["count"] > 0
    _save_composed_config(pipeline_cfg, pipeline_output)
    pipeline_artifacts = write_run_artifacts(
        pipeline_output,
        pipeline_cfg,
        pipeline_result,
    )
    assert (pipeline_output / "checkpoints" / "best.pt").is_file()
    pipeline_provenance = json.loads(
        pipeline_artifacts["provenance"].read_text(encoding="utf-8")
    )
    pipeline_metrics = json.loads(
        pipeline_artifacts["metrics"].read_text(encoding="utf-8")
    )
    history = pipeline_metrics["history"]
    assert len(history) == 1
    assert [record["train/epoch"] for record in history] == [1]
    assert len({record["train/epoch"] for record in history}) == len(history)
    for field in (
        "train/epoch_seconds",
        "train/optimizer_steps",
        "train/samples_per_second",
        "train/voxels_per_second",
        "train/peak_memory_allocated_gb",
        "train/peak_memory_reserved_gb",
        "power/average_watts",
        "power/max_watts",
        "power/energy_joules",
        "power/sample_count",
        "power/sample_interval_ms",
        "power/status",
        "run/elapsed_seconds",
    ):
        assert field in history[0]
    assert pipeline_metrics["efficiency"]["power/status"] in {
        "disabled",
        "unavailable",
        "ok",
    }
    assert pipeline_provenance["timing"]["timing_scope"] == "process_segment"
    assert pipeline_provenance["protocol"] == "native_3d_full_volume"
    assert pipeline_provenance["early_stopping"] == {
        "stopped": False,
        "stop_epoch": None,
        "best_epoch": 1,
    }
    assert pipeline_provenance["metadata"]["weight_transfer"]["count"] == transfer_report["count"]
    transfer_counts = transfer_report["counts"]

    benchmark_model = nn.Conv3d(4, 3, kernel_size=1)
    benchmark_output = run_model_protocol(
        benchmark_model,
        torch.zeros(1, 4, 4, 4, 4),
        protocol="native_3d_full_volume",
        warmup_iterations=0,
        repetitions=1,
        batch_sizes=[1],
    )
    benchmark_case_id = case_ids[-1]
    benchmark_path = serialize_benchmark(
        tmp_path / "benchmark",
        BenchmarkResult(
            summary=benchmark_output,
            rows=[
                {
                    "row_type": "case",
                    "case_id_hash": hash_case_id(benchmark_case_id),
                    "model": "ResUNet3D",
                    "protocol": "native_3d_full_volume",
                    "dice_by_region": {region: None for region in REGION_NAMES},
                    "hd95_by_region": {region: None for region in REGION_NAMES},
                    "exclusion_flags": list(REGION_NAMES),
                }
            ],
            provenance={
                "data_status": "synthetic_fixture",
                "protocol": "native_3d_full_volume",
            },
        ),
    )
    benchmark_text = benchmark_path.read_text(encoding="utf-8")
    benchmark_payload = json.loads(benchmark_text)
    assert "model/macs" in benchmark_payload["summary"]
    assert "model/flops" in benchmark_payload["summary"]
    assert "power/status" in benchmark_payload["summary"]
    assert benchmark_payload["summary"]["power/status"] in {
        "unavailable",
        "ok",
        "partial",
    }
    benchmark_row = benchmark_payload["rows"][0]
    assert "case_id_hash" in benchmark_row
    assert "case_id" not in benchmark_row
    assert benchmark_case_id not in benchmark_text

    slice_bundle = build_slice_loaders(
        segmentation_cfg,
        torch.Generator().manual_seed(42),
    )
    slice_image, slice_target = next(iter(slice_bundle[0]))[:2]
    trans_model = build_transunet(
        OmegaConf.create(
            {
                "model": {
                    "in_channels": 4,
                    "num_classes": 3,
                    "image_size": [32, 32],
                    "vit_name": "R50-ViT-B_16",
                }
            }
        ),
        external_model=_TinyTransUNetBackend(),
    )
    _assert_forward_loss_backward(
        trans_model,
        slice_image.float(),
        slice_target.float(),
        (slice_image.shape[0], 3, 32, 32),
    )

    optional_skips: dict[str, str] = {}
    if importlib.util.find_spec("einops") is None:
        optional_skips["swinunetr"] = "einops is unavailable"
    else:
        from token_mixer.models.swinunetr import build_swinunetr

        swin_cfg = OmegaConf.create(
            {
                "model": {
                    "feature_size": 12,
                    "depths": [1, 1, 1, 1],
                    "num_heads": [1, 1, 1, 1],
                    "window_size": 1,
                    "use_checkpoint": False,
                }
            }
        )
        # Keep other fixtures at 32^3; SwinUNETR needs 64^3 to avoid a 1^3 bottleneck.
        swin_image = F.pad(image, (16, 16, 16, 16, 16, 16))
        swin_target = F.pad(target, (16, 16, 16, 16, 16, 16))
        _assert_forward_loss_backward(
            build_swinunetr(swin_cfg),
            swin_image,
            swin_target,
            (1, 3, 64, 64, 64),
        )

    image_root = tmp_path / "imagenet"
    image_module = pytest.importorskip("PIL.Image")
    for class_index, class_name in enumerate(("class-a", "class-b")):
        class_root = image_root / class_name
        class_root.mkdir(parents=True)
        for image_index in range(2):
            image_module.new(
                "RGB",
                (32, 32),
                (class_index * 100, image_index * 25, 20),
            ).save(class_root / f"image-{image_index}.png")
    cnn_cfg = OmegaConf.create(
        {
            "model": {
                "in_channels": 3,
                "feature_size": 2,
                "depths": [0, 0, 0, 0],
                "image_size": 32,
            },
            "data": {
                "root": str(image_root),
                "noise_std": 0.0,
                "val_fraction": 0.5,
            },
            "training": {
                "batch_size": 1,
                "num_workers": 0,
                "pin_memory": False,
                "persistent_workers": False,
                "drop_last": False,
            },
            "run": {"max_cases": 4},
        }
    )
    from token_mixer.pipelines.pretrain_cnn import build_dataloaders

    cnn_train_loader, _cnn_val_loader = build_dataloaders(
        cnn_cfg,
        torch.Generator().manual_seed(42),
    )
    cnn_image, cnn_target = next(iter(cnn_train_loader))
    cnn_model = build_denoising_model(cnn_cfg)
    cnn_model.zero_grad(set_to_none=True)
    cnn_logits = cnn_model(cnn_image.float())
    cnn_loss = mse_loss(cnn_logits, cnn_target.float())
    assert torch.isfinite(cnn_loss)
    cnn_loss.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in cnn_model.parameters()
        if parameter.requires_grad
    )

    checkpoint_model = nn.Linear(2, 2)
    before = {name: value.detach().clone() for name, value in checkpoint_model.state_dict().items()}
    checkpoints = CheckpointManager(checkpoint_root)
    checkpoints.save(
        "best",
        checkpoint_model,
        None,
        None,
        None,
        {
            "epoch": 1,
            "metric": 1.0,
            "manifest_hash": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        },
    )
    with torch.no_grad():
        for parameter in checkpoint_model.parameters():
            parameter.zero_()
    checkpoints.load(checkpoint_root / "best.pt", checkpoint_model)
    assert all(torch.equal(value, checkpoint_model.state_dict()[name]) for name, value in before.items())

    full_metrics = evaluate_full_volumes(
        _RegionIdentity(),
        test_loader,
        roi_size=(32, 32, 32),
        sw_batch_size=1,
        overlap=0.0,
        device="cpu",
        default_spacing=(1.0, 1.0, 1.0),
    )
    assert full_metrics["case_ids"] == manifest.test
    assert full_metrics["mean_dice"] == pytest.approx(1.0)
    assert all(f"{region}_dice" in full_metrics for region in REGION_NAMES)

    tracker = create_tracker({"enabled": False, "mode": "disabled"}, {})
    assert isinstance(tracker, Tracker)
    tracker.finish()

    config = OmegaConf.create(
        {
            "runtime": "local",
            "experiment": {"name": "synthetic_debug"},
            "tracking": {"enabled": False, "mode": "disabled"},
            "paths": {"manifest": str(manifest_path)},
        }
    )
    result = FitResult(
        best_metric=float(full_metrics["mean_dice"]),
        best_epoch=1,
        history=[{"epoch": 1, "mean_dice": float(full_metrics["mean_dice"])}],
        test_metrics={
            key: value
            for key, value in full_metrics.items()
            if isinstance(value, (int, float))
        },
        metadata={
            "architecture": "synthetic-debug",
            "execution_device": "cpu",
            "manifest_path": str(manifest_path),
            "manifest_hash": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "case_ids": manifest.test,
            "weight_transfer": transfer_counts,
            "optional_skips": optional_skips,
        },
    )
    _save_composed_config(config, output_dir)
    write_run_artifacts(output_dir, config, result)

    assert {
        path.name for path in output_dir.iterdir()
    } >= {"config.yaml", "metrics.json", "provenance.json", "checkpoints"}
    assert (output_dir / "checkpoints" / "best.pt").is_file()
    assert result.metadata["case_ids"] == manifest.test
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    provenance = json.loads((output_dir / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["metadata"]["weight_transfer"]["total"] == transfer_counts["total"]
    expected_case_hashes = [hash_case_id(case_id) for case_id in manifest.test]
    assert metrics["metadata"]["case_id_hashes"] == expected_case_hashes
    assert provenance["metadata"]["case_id_hashes"] == expected_case_hashes
    assert "case_id" not in metrics["metadata"]
    assert "case_ids" not in metrics["metadata"]
    assert "case_id" not in provenance["metadata"]
    assert "case_ids" not in provenance["metadata"]
    persisted_text = (output_dir / "provenance.json").read_text(encoding="utf-8")
    assert all(case_id not in persisted_text for case_id in manifest.test)


def test_synthetic_efficiency_fixtures_keep_non_hardware_statuses(
    monkeypatch: pytest.MonkeyPatch,
):
    import token_mixer.evaluation.benchmark as benchmark_module

    class _UnavailablePowerFixture:
        def __init__(self, device_index, interval_seconds):
            self.device_index = device_index
            self.interval_seconds = interval_seconds

        def start(self):
            return self

        def stop(self):
            return {
                "average_watts": None,
                "max_watts": None,
                "joules": None,
                "samples": 0,
                "interval_seconds": self.interval_seconds,
                "device_index": self.device_index,
                "status": "unavailable",
                "reason": "synthetic fixture: NVML unavailable",
            }

    monkeypatch.setattr(benchmark_module, "NvmlPowerSampler", _UnavailablePowerFixture)
    monkeypatch.setattr(
        benchmark_module,
        "static_model_cost",
        lambda *_args, **_kwargs: {
            "parameters": 0,
            "trainable_parameters": 0,
            "macs": None,
            "flops": None,
            "mac_tool": "thop",
            "mac_tool_version": "fixture",
            "mac_convention": "fixture counter convention",
            "mac_status": "unavailable",
            "mac_error": "synthetic fixture: THOP unavailable",
            "flop_tool": "fvcore",
            "flop_tool_version": "fixture",
            "flop_convention": "fixture counter convention",
            "flop_status": "unavailable",
            "flop_error": "synthetic fixture: fvcore unavailable",
            "unsupported_ops": {},
        },
    )
    monkeypatch.setattr(
        benchmark_module,
        "measure_forward",
        lambda *_args, **kwargs: {
            "status": "ok",
            "latency_mean_ms": 1.0,
            "latency_median_ms": 1.0,
            "latency_p95_ms": 1.0,
            "latency_std_ms": 0.0,
            "batch_size": 1,
            "input_shape": [1, 1],
            "warmup_iterations": kwargs["warmup_iterations"],
            "repetitions": kwargs["repetitions"],
            "protocol": kwargs["protocol"],
        },
    )
    monkeypatch.setattr(
        benchmark_module,
        "measure_batch_sweep",
        lambda *_args, **kwargs: {
            "status": "ok",
            "sweep_status": "ok",
            "rows": [],
            "batch_sizes": list(_args[2]),
            "largest_passing_batch": 1,
            "first_failing_batch": None,
            "protocol": kwargs["protocol"],
        },
    )

    output = run_model_protocol(
        nn.Identity(),
        torch.ones(1, 1),
        protocol="synthetic_fixture",
        warmup_iterations=0,
        repetitions=1,
        batch_sizes=[1],
    )

    assert output["model/mac_status"] == "unavailable"
    assert output["model/flop_status"] == "unavailable"
    assert output["model/macs"] is None
    assert output["model/flops"] is None
    assert output["model/mac_tool"] == "thop"
    assert output["model/flop_tool"] == "fvcore"
    assert output["power/status"] == "unavailable"
    assert output["power/reason"] == "synthetic fixture: NVML unavailable"


def test_synthetic_snapshots_keep_interval_and_best_all_regions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    class _SnapshotTracker:
        image_logging_enabled = True

        def __init__(self):
            self.calls: list[tuple[dict[str, object], int]] = []

        def log_images(self, images, *, step, captions=None):
            del captions
            self.calls.append((dict(images), step))

    image = torch.ones(1, 4, 4, 4)
    target = torch.zeros(1, 3, 4, 4)
    target[:, 0, 0, 0] = 1.0
    target[:, 1, 1, 1] = 1.0
    target[:, 2, 2, 2] = 1.0
    loader = DataLoader(
        TensorDataset(image, target),
        batch_size=1,
        shuffle=False,
    )
    tracker = _SnapshotTracker()
    config = {
        "device": "cpu",
        "native_3d": False,
        "tracking": {
            "enabled": True,
            "mode": "offline",
            "log_images": True,
        },
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
                "local_enabled": False,
                "output_dir": str(tmp_path),
            }
        },
    }
    rendered_masks: list[tuple[np.ndarray, np.ndarray]] = []

    def fake_render(image, target, prediction, **kwargs):
        del image, kwargs
        rendered_masks.append((np.asarray(target), np.asarray(prediction)))
        return np.zeros((4, 4, 3), dtype=np.uint8)

    monkeypatch.setattr(
        "token_mixer.evaluation.visualization.render_slice_visualization",
        fake_render,
    )
    snapshotter = build_segmentation_snapshotter(config, tracker, device="cpu")
    assert snapshotter is not None

    class _RegionIdentity(nn.Module):
        def forward(self, value):
            return value[:, :3]

    model = _RegionIdentity()
    snapshotter.snapshot(
        model=model,
        train_loader=loader,
        val_loader=loader,
        epoch=10,
        global_step=10,
        kind="epoch",
    )
    snapshotter.snapshot(
        model=model,
        train_loader=loader,
        val_loader=loader,
        epoch=11,
        global_step=11,
        kind="best",
    )

    assert [sorted(images) for images, _step in tracker.calls] == [
        [
            "segmentation/train/epoch_0010",
            "segmentation/val/epoch_0010",
        ],
        [
            "segmentation/train/best_epoch_0011",
            "segmentation/val/best_epoch_0011",
        ],
    ]
    assert len(rendered_masks) == 4
    assert all(mask.shape[0] == len(REGION_NAMES) for mask, _prediction in rendered_masks)
    assert all(np.all(mask.sum(axis=tuple(range(1, mask.ndim))) > 0) for mask, _prediction in rendered_masks)
    assert all(
        prediction.shape[0] == len(REGION_NAMES)
        for _mask, prediction in rendered_masks
    )


def test_synthetic_disabled_tracking_constructs_no_wandb_or_images(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    real_import = importlib.import_module

    def fail_wandb_import(name, *args, **kwargs):
        if name == "wandb":
            raise AssertionError("disabled synthetic tracking imported wandb")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fail_wandb_import)
    tracker = create_tracker(
        {"enabled": False, "mode": "disabled", "log_images": True},
        {},
    )
    snapshotter = build_segmentation_snapshotter(
        {
            "device": "cpu",
            "tracking": {
                "enabled": False,
                "mode": "disabled",
                "log_images": True,
            },
            "visualization": {
                "segmentation_snapshots": {
                    "enabled": True,
                    "local_enabled": False,
                    "output_dir": str(tmp_path),
                }
            },
        },
        tracker,
        device="cpu",
    )

    assert snapshotter is None
    tracker.log_images({"segmentation/fixture": tmp_path / "unused.png"}, step=1)


def test_synthetic_debug_profile_keeps_snapshots_effectively_disabled_by_default():
    config = _segmentation_config(Path("data"), Path("manifest.json"), Path("checkpoints"))
    config["visualization"] = {
        "segmentation_snapshots": {
            "enabled": True,
            "local_enabled": False,
        }
    }

    assert config.tracking.get("log_images", False) is False
    assert config.visualization.segmentation_snapshots.local_enabled is False
