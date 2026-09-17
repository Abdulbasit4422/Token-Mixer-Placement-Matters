from __future__ import annotations

import importlib
import netrc
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from token_mixer.privacy import (
    redact_case_identifiers,
    redact_image_metadata,
    sanitize_namespace_key,
)


_WANDB_NETRC_HOSTS = (
    "api.wandb.ai",
    "api.wandb.com",
    "wandb.ai",
    "wandb.com",
)


class Tracker:
    """Small tracking interface with no-op default behavior."""

    @property
    def run_id(self) -> str | None:
        return None

    @property
    def image_logging_enabled(self) -> bool:
        return False

    def log(self, metrics: Mapping[str, Any], step: int) -> None:
        del metrics, step

    def log_summary(self, metrics: Mapping[str, Any]) -> None:
        del metrics

    def log_table(
        self,
        name: str,
        columns: Sequence[str],
        rows: Sequence[Sequence[Any]],
    ) -> None:
        del name, columns, rows

    def log_images(
        self,
        images: Mapping[str, Path | Any],
        *,
        step: int,
        captions: Mapping[str, str] | None = None,
    ) -> None:
        del images, step, captions

    def define_metric(self, name: str, *, step_metric: str | None = None) -> None:
        del name, step_metric

    def log_artifact(
        self,
        name: str,
        files: Mapping[str, Path],
        *,
        artifact_type: str = "model",
        aliases: Sequence[str] = (),
    ) -> str | None:
        del name, files, artifact_type, aliases
        return None

    def restore_artifact(self, reference: str, destination: Path) -> Path:
        del reference, destination
        raise RuntimeError("cannot restore W&B artifact: tracking is disabled")

    def finish(self) -> None:
        return None


def _redact_table_payload(
    columns: Sequence[str], rows: Sequence[Sequence[Any]]
) -> tuple[list[str], list[list[Any]]]:
    source_columns = [str(column) for column in columns]
    sanitized_rows: list[Mapping[str, Any]] = []
    output_columns: list[str] = []

    for column in source_columns:
        redacted_column = redact_case_identifiers({column: None})
        names = list(redacted_column) if isinstance(redacted_column, Mapping) else []
        canonical = names[0] if names else column
        if canonical not in output_columns:
            output_columns.append(canonical)

    for row in rows:
        if isinstance(row, Mapping):
            row_mapping = dict(row)
        else:
            row_mapping = dict(zip(source_columns, row))
        redacted_row = redact_case_identifiers(row_mapping)
        if not isinstance(redacted_row, Mapping):
            redacted_row = {}
        sanitized_rows.append(redacted_row)
        for column in redacted_row:
            if column not in output_columns:
                output_columns.append(str(column))

    return output_columns, [
        [row.get(column) for column in output_columns] for row in sanitized_rows
    ]


class _WandbTracker(Tracker):
    def __init__(self, run: Any, wandb: Any, *, log_images: bool = False):
        self._run = run
        self._wandb = wandb
        self._log_images_enabled = bool(log_images)

    @property
    def run_id(self) -> str | None:
        run_id = getattr(self._run, "id", None)
        return None if run_id is None else str(run_id)

    @property
    def image_logging_enabled(self) -> bool:
        return self._log_images_enabled

    def log(self, metrics: Mapping[str, Any], step: int) -> None:
        self._run.log(redact_case_identifiers(dict(metrics)), step=step)

    def log_summary(self, metrics: Mapping[str, Any]) -> None:
        summary = getattr(self._run, "summary", None)
        sanitized_metrics = redact_case_identifiers(dict(metrics))
        if summary is not None and hasattr(summary, "update"):
            summary.update(sanitized_metrics)
        else:
            self._run.log(sanitized_metrics)

    def log_table(
        self,
        name: str,
        columns: Sequence[str],
        rows: Sequence[Sequence[Any]],
    ) -> None:
        sanitized_columns, sanitized_rows = _redact_table_payload(columns, rows)
        table = self._wandb.Table(
            columns=sanitized_columns,
            data=sanitized_rows,
        )
        self._run.log({sanitize_namespace_key(name): table})

    def log_images(
        self,
        images: Mapping[str, Path | Any],
        *,
        step: int,
        captions: Mapping[str, str] | None = None,
    ) -> None:
        if not self._log_images_enabled:
            return
        captions = captions or {}
        payload: dict[str, Any] = {}
        for name, image in images.items():
            caption = captions.get(name)
            payload[sanitize_namespace_key(name)] = self._wandb.Image(
                image,
                caption=(
                    None
                    if caption is None
                    else redact_image_metadata(caption)
                ),
            )
        self._run.log(payload, step=step)

    def define_metric(self, name: str, *, step_metric: str | None = None) -> None:
        name = sanitize_namespace_key(name)
        if step_metric is not None:
            step_metric = sanitize_namespace_key(step_metric)
        if step_metric is None:
            if name.startswith("train/"):
                step_metric = "train/epoch"
            elif name.startswith("val/"):
                step_metric = "val/epoch"
        if step_metric is None:
            self._run.define_metric(name)
        else:
            self._run.define_metric(name, step_metric=step_metric)

    def log_artifact(
        self,
        name: str,
        files: Mapping[str, Path],
        *,
        artifact_type: str = "model",
        aliases: Sequence[str] = (),
    ) -> str | None:
        artifact = self._wandb.Artifact(name, type=artifact_type)
        for artifact_name, path in files.items():
            artifact.add_file(str(path), name=artifact_name)
        self._run.log_artifact(artifact, aliases=list(aliases))
        waited_artifact = artifact.wait()
        version = getattr(artifact, "version", None)
        if version is None and waited_artifact is not None:
            version = getattr(waited_artifact, "version", None)
        if version is None:
            return None
        return f"{getattr(artifact, 'name', name)}:{version}"

    def restore_artifact(self, reference: str, destination: Path) -> Path:
        downloaded = Path(
            self._run.use_artifact(reference).download(root=str(destination))
        )
        if downloaded.is_dir():
            best_checkpoint = downloaded / "best.pt"
            if best_checkpoint.is_file():
                return best_checkpoint
        return downloaded

    def finish(self) -> None:
        self._run.finish()


def tracking_image_logging_enabled(
    config: Mapping[str, Any], tracker: Any | None = None
) -> bool:
    """Return whether configured, active tracking may receive W&B images."""
    tracking = config.get("tracking") if isinstance(config, Mapping) else None
    if not isinstance(tracking, Mapping):
        return False
    def as_bool(value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"true", "yes", "1", "on"}
        return bool(value)

    if not as_bool(tracking.get("enabled", False)):
        return False
    if str(tracking.get("mode", "disabled")).lower() not in {"online", "offline"}:
        return False
    if not as_bool(tracking.get("log_images", False)):
        return False
    configured = getattr(tracker, "image_logging_enabled", None)
    if configured is not None and not as_bool(configured):
        return False
    return True


def _wandb_error(exc: BaseException) -> RuntimeError:
    error = RuntimeError(
        "W&B tracking is enabled, but the 'wandb' package is unavailable; "
        "install the project dependencies before training"
    )
    error.__cause__ = exc
    return error


def _wandb_netrc_has_credentials() -> bool:
    try:
        credentials = netrc.netrc()
    except (OSError, netrc.NetrcParseError):
        return False
    return any(credentials.authenticators(host) is not None for host in _WANDB_NETRC_HOSTS)


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

    if mode == "online" and not (
        os.environ.get("WANDB_API_KEY") or _wandb_netrc_has_credentials()
    ):
        raise RuntimeError(
            "W&B online tracking requires credentials from WANDB_API_KEY or "
            "a matching standard .netrc entry before training"
        )

    try:
        wandb = importlib.import_module("wandb")
    except (ImportError, ModuleNotFoundError) as exc:
        raise _wandb_error(exc)

    init_kwargs: dict[str, Any] = {
        "project": config.get("project"),
        "entity": config.get("entity"),
        "config": redact_case_identifiers(dict(run_config)),
        "mode": mode,
        "dir": config.get("directory", config.get("dir")),
    }
    run_name = config.get("run_name", config.get("name"))
    if run_name is not None:
        init_kwargs["name"] = redact_case_identifiers({"run_name": run_name})[
            "run_name"
        ]
    for key in ("group", "job_type", "tags"):
        value = config.get(key)
        if value is not None:
            init_kwargs[key] = redact_case_identifiers({key: value})[key]

    return _WandbTracker(
        wandb.init(**init_kwargs),
        wandb,
        log_images=bool(config.get("log_images", False)),
    )
