"""Training entrypoint for the external, 2-D slice-based TransUNet adapter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from omegaconf import DictConfig
from torch import nn

from token_mixer.models.transunet import (
    build_transunet,
    get_transunet_metadata,
    validate_transunet_config,
)
from token_mixer.pipelines._baseline_common import (
    LoaderBundle,
    _first_value,
    build_checkpoints,
    build_loss,
    build_phases,
    build_slice_evaluator,
    build_slice_loaders,
    run_2d_baseline,
)
from token_mixer.reproducibility import seed_everything
from token_mixer.training.engine import FitResult, fit
from token_mixer.training.tracking import create_tracker


def build_loaders(cfg: DictConfig | Mapping[str, Any], generator: Any) -> LoaderBundle:
    return build_slice_loaders(cfg, generator)


def _build_evaluator(cfg: DictConfig | Mapping[str, Any], device: Any):
    return build_slice_evaluator(cfg, device)


def _build_checkpoints(cfg: DictConfig | Mapping[str, Any]):
    return build_checkpoints(cfg)


def _build_loss(cfg: DictConfig | Mapping[str, Any]):
    return build_loss(cfg)


def _build_phases(cfg: DictConfig | Mapping[str, Any]):
    return build_phases(cfg)


def _injected_external_model(cfg: Mapping[str, Any]) -> nn.Module | None:
    configured = _first_value(
        cfg,
        (
            ("external_model",),
            ("external_network",),
            ("model", "external_model"),
            ("model", "external_network"),
        ),
    )
    return configured if isinstance(configured, nn.Module) else None


def _build_model(cfg: DictConfig | Mapping[str, Any]) -> nn.Module:
    injected = _injected_external_model(cfg)
    if injected is None:
        return build_transunet(cfg)
    return build_transunet(cfg, external_model=injected)


def run_transunet(cfg: DictConfig | Mapping[str, Any]) -> FitResult:
    """Validate external files, train 2-D slices, and score held-out slices."""
    if _injected_external_model(cfg) is None:
        validate_transunet_config(cfg)
    return run_2d_baseline(
        cfg,
        architecture="TransUNet",
        model_builder=_build_model,
        loader_builder=build_loaders,
        evaluator_builder=_build_evaluator,
        metadata_extra=get_transunet_metadata(),
        seed_fn=seed_everything,
        loss_builder=_build_loss,
        phases_builder=_build_phases,
        checkpoint_builder=_build_checkpoints,
        tracker_builder=create_tracker,
        fit_fn=fit,
    )


__all__ = ["build_loaders", "run_transunet"]
