from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F

from token_mixer.cli import _save_composed_config
from token_mixer.data.splits import SplitManifest, save_split_manifest
from token_mixer.evaluation.inference import evaluate_full_volumes
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
    assert pipeline_provenance["metadata"]["weight_transfer"]["count"] == transfer_report["count"]
    transfer_counts = transfer_report["counts"]

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
    provenance = json.loads((output_dir / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["metadata"]["weight_transfer"]["total"] == transfer_counts["total"]
    assert provenance["metadata"]["case_ids"] == manifest.test
