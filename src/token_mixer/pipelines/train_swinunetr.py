"""Training entrypoint for the MONAI SwinUNETR 3-D baseline."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from omegaconf import DictConfig

from token_mixer.models.swinunetr import build_swinunetr
from token_mixer.pipelines._baseline_common import (
    LoaderBundle,
    build_checkpoints,
    build_loss,
    build_phases,
    build_volume_evaluator,
    build_volume_loaders,
    run_3d_baseline,
)
from token_mixer.reproducibility import seed_everything
from token_mixer.training.engine import FitResult, fit
from token_mixer.training.tracking import create_tracker


def build_loaders(cfg: DictConfig | Mapping[str, Any], generator: Any) -> LoaderBundle:
    return build_volume_loaders(cfg, generator)


def _build_evaluator(cfg: DictConfig | Mapping[str, Any], device: Any):
    return build_volume_evaluator(cfg, device)


def _build_checkpoints(cfg: DictConfig | Mapping[str, Any]):
    return build_checkpoints(cfg)


def _build_loss(cfg: DictConfig | Mapping[str, Any]):
    return build_loss(cfg)


def _build_phases(cfg: DictConfig | Mapping[str, Any]):
    return build_phases(cfg)


def run_swinunetr(cfg: DictConfig | Mapping[str, Any]) -> FitResult:
    """Run manifest-backed 3-D training and evaluate the held-out test split."""
    return run_3d_baseline(
        cfg,
        architecture="SwinUNETR",
        model_builder=build_swinunetr,
        loader_builder=build_loaders,
        evaluator_builder=_build_evaluator,
        seed_fn=seed_everything,
        loss_builder=_build_loss,
        phases_builder=_build_phases,
        checkpoint_builder=_build_checkpoints,
        tracker_builder=create_tracker,
        fit_fn=fit,
    )


__all__ = ["build_loaders", "run_swinunetr"]
