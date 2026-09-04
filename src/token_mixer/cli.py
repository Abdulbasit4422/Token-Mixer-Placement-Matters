"""Hydra entrypoint for selecting a configured research pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from token_mixer.training.artifacts import (
    write_failed_run_artifact,
    write_run_artifacts,
)
from token_mixer.training.engine import FitResult


CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def _hydra_output_dir() -> Path:
    return Path(HydraConfig.get().runtime.output_dir)


def _save_composed_config(cfg: DictConfig, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=str(output_dir / "config.yaml"), resolve=False)


def _experiment_name(cfg: DictConfig | Mapping[str, Any]) -> str:
    experiment = cfg.get("experiment")
    if not isinstance(experiment, Mapping) or experiment.get("name") is None:
        raise ValueError("unknown experiment: configuration has no experiment.name")
    return str(experiment["name"])


def _dispatch(cfg: DictConfig) -> Any:
    name = _experiment_name(cfg)

    runner: Callable[[DictConfig], Any]
    if name == "cnn_denoising_pretrain":
        from token_mixer.pipelines.pretrain_cnn import run_cnn_denoising_pretrain

        runner = run_cnn_denoising_pretrain
    elif name in {"metaunetr_mamba", "mod_a", "mod_b"}:
        from token_mixer.pipelines.train_metaunetr import run_metaunetr

        runner = run_metaunetr
    elif name == "resunet3d":
        from token_mixer.pipelines.train_resunet3d import run_resunet3d

        runner = run_resunet3d
    elif name == "swinunetr":
        from token_mixer.pipelines.train_swinunetr import run_swinunetr

        runner = run_swinunetr
    elif name == "transunet":
        from token_mixer.pipelines.train_transunet import run_transunet

        runner = run_transunet
    else:
        raise ValueError(f"unknown experiment: {name}")
    return runner(cfg)


def _run(cfg: DictConfig) -> None:
    output_dir = _hydra_output_dir()
    _save_composed_config(cfg, output_dir)
    try:
        result = _dispatch(cfg)
    except Exception as exc:
        transfer_report = getattr(exc, "transfer_report", None)
        if isinstance(transfer_report, Mapping) and transfer_report.get("requested"):
            try:
                write_failed_run_artifact(
                    output_dir,
                    cfg,
                    exc,
                    {
                        "architecture": "ResUNet3D",
                        "dimensionality": "3-D",
                        "weight_transfer": transfer_report,
                        "imagenet_transfer": transfer_report,
                        "transfer_requested": transfer_report.get("requested"),
                        "transfer_status": transfer_report.get("status"),
                        "transfer_count": transfer_report.get("count"),
                        "transfer_counts": transfer_report.get("counts"),
                        "transfer_source": transfer_report.get("source"),
                    },
                )
            except Exception as artifact_error:
                exc.add_note(
                    "failed to persist failed-run provenance: "
                    f"{artifact_error}"
                )
        raise
    if isinstance(result, FitResult):
        write_run_artifacts(output_dir, cfg, result)


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="local")
def main(cfg: DictConfig) -> None:
    _run(cfg)


__all__ = ["main"]
