from __future__ import annotations

import importlib
import os
from collections.abc import Mapping
from typing import Any


class Tracker:
    """Small tracking interface with no-op default behavior."""

    def log(self, metrics: Mapping[str, float], step: int) -> None:
        del metrics, step

    def log_summary(self, metrics: Mapping[str, float]) -> None:
        del metrics

    def finish(self) -> None:
        return None


class _WandbTracker(Tracker):
    def __init__(self, run: Any):
        self._run = run

    def log(self, metrics: Mapping[str, float], step: int) -> None:
        self._run.log(dict(metrics), step=step)

    def log_summary(self, metrics: Mapping[str, float]) -> None:
        summary = getattr(self._run, "summary", None)
        if summary is not None and hasattr(summary, "update"):
            summary.update(dict(metrics))
        else:
            self._run.log(dict(metrics))

    def finish(self) -> None:
        self._run.finish()


def _wandb_error(exc: BaseException) -> RuntimeError:
    error = RuntimeError(
        "W&B tracking is enabled, but the 'wandb' package is unavailable; "
        "install the project dependencies before training"
    )
    error.__cause__ = exc
    return error


def create_tracker(
    config: Mapping[str, Any], run_config: Mapping[str, Any]
) -> Tracker:
    """Create disabled, offline, or online tracking before training starts."""
    if not bool(config.get("enabled", False)):
        return Tracker()

    mode = str(config.get("mode", "online")).lower()
    if mode == "disabled":
        return Tracker()
    if mode not in {"online", "offline"}:
        raise ValueError("tracking mode must be one of: online, offline, disabled")

    if mode == "online" and not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("W&B online tracking requires WANDB_API_KEY before training")

    try:
        wandb = importlib.import_module("wandb")
    except (ImportError, ModuleNotFoundError) as exc:
        raise _wandb_error(exc)

    init_kwargs: dict[str, Any] = {
        "project": config.get("project"),
        "entity": config.get("entity"),
        "config": dict(run_config),
        "mode": mode,
        "dir": config.get("directory", config.get("dir")),
    }
    run_name = config.get("run_name", config.get("name"))
    if run_name is not None:
        init_kwargs["name"] = run_name

    return _WandbTracker(wandb.init(**init_kwargs))
