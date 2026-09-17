"""Hydra entrypoint for selecting a configured research pipeline."""

from __future__ import annotations

from collections.abc import Mapping
import shutil
import tempfile
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


def _run_output_dir(cfg: DictConfig) -> Path:
    try:
        return _hydra_output_dir()
    except Exception:
        from token_mixer.pipelines._baseline_common import _output_dir

        return _output_dir(cfg)


def _save_composed_config(cfg: DictConfig, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=str(output_dir / "config.yaml"), resolve=False)


def _experiment_name(cfg: DictConfig | Mapping[str, Any]) -> str:
    experiment = cfg.get("experiment")
    if not isinstance(experiment, Mapping) or experiment.get("name") is None:
        raise ValueError("unknown experiment: configuration has no experiment.name")
    return str(experiment["name"])


def _write_fallback_completion_artifacts(
    output_dir: Path, cfg: DictConfig, result: FitResult
) -> None:
    """Fill missing CLI artifacts without replacing pipeline-owned evidence."""
    metrics_path = output_dir / "metrics.json"
    provenance_path = output_dir / "provenance.json"
    missing = {
        "metrics": not metrics_path.exists(),
        "provenance": not provenance_path.exists(),
    }
    if not any(missing.values()):
        return
    if all(missing.values()):
        write_run_artifacts(output_dir, cfg, result)
        return

    # ``write_run_artifacts`` intentionally writes its pair atomically.  Use a
    # sibling temporary directory for the partial-artifact fallback so an
    # existing pipeline file is never rewritten.
    with tempfile.TemporaryDirectory(prefix=".completion-", dir=str(output_dir)) as name:
        generated = write_run_artifacts(Path(name), cfg, result)
        if missing["metrics"]:
            shutil.copyfile(generated["metrics"], metrics_path)
        if missing["provenance"]:
            shutil.copyfile(generated["provenance"], provenance_path)


def _dispatch(cfg: DictConfig) -> Any:
    if cfg.get("command") == "benchmark":
        from token_mixer.pipelines.benchmark import run_benchmark

        return run_benchmark(cfg)

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
    output_dir = _run_output_dir(cfg)
    _save_composed_config(cfg, output_dir)
    try:
        result = _dispatch(cfg)
    except Exception as exc:
        transfer_report = getattr(exc, "transfer_report", None)
        failure_metadata: dict[str, Any] = {}
        if isinstance(transfer_report, Mapping) and transfer_report.get("requested"):
            failure_metadata = {
                "architecture": "ResUNet3D",
                "dimensionality": "3-D",
                "weight_transfer": transfer_report,
                "imagenet_transfer": transfer_report,
                "transfer_requested": transfer_report.get("requested"),
                "transfer_status": transfer_report.get("status"),
                "transfer_count": transfer_report.get("count"),
                "transfer_counts": transfer_report.get("counts"),
                "transfer_source": transfer_report.get("source"),
            }
        try:
            write_failed_run_artifact(
                output_dir,
                cfg,
                exc,
                failure_metadata,
            )
        except Exception as artifact_error:
            exc.add_note(
                "failed to persist failed-run provenance: "
                f"{artifact_error}"
            )
        raise
    if isinstance(result, FitResult):
        _write_fallback_completion_artifacts(output_dir, cfg, result)


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="local")
def main(cfg: DictConfig) -> None:
    _run(cfg)


__all__ = ["main"]
