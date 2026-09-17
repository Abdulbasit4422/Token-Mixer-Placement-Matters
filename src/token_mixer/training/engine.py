from __future__ import annotations

import math
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Number
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

from token_mixer.evaluation.efficiency import (
    NvmlPowerSampler,
    peak_memory_gb,
    reset_peak_memory,
)
from token_mixer.privacy import redact_case_identifiers

from .checkpoints import CheckpointManager
from .phases import PhaseSpec, apply_phase
from .tracking import Tracker, tracking_image_logging_enabled


_MISSING = object()


@dataclass(frozen=True)
class FitResult:
    """Summary returned by the model-independent training engine."""

    best_metric: float
    best_epoch: int
    history: list[dict[str, Any]]
    test_metrics: dict[str, float] | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class EpochTrainResult:
    """Measured result returned by one training epoch."""

    train_loss: float
    global_step: int
    samples: int
    voxels: int


def _config_value(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _section_value(config: Any, key: str, default: Any = None) -> Any:
    value = _config_value(config, key, _MISSING)
    if value is not _MISSING:
        return value
    run_config = _config_value(config, "run", None)
    if run_config is not None:
        return _config_value(run_config, key, default)
    return default


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str):
        return {"name": value}
    raise TypeError("configuration section must be a mapping or name string")


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "on"}:
            return True
        if normalized in {"false", "no", "0", "off"}:
            return False
    return bool(value)


def _resolve_device(value: torch.device | str | None) -> torch.device:
    requested = "auto" if value is None else value
    if str(requested).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested, but CUDA is unavailable")
    return device


def _to_plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_plain(item) for item in value)
    return value


def _callable_name(value: Any) -> str:
    return f"{getattr(value, '__module__', '')}.{getattr(value, '__qualname__', repr(value))}"


def _phase_plan(phases: Sequence[PhaseSpec]) -> list[dict[str, Any]]:
    return [
        {
            "name": phase.name,
            "epochs": phase.epochs,
            "freeze_encoder": phase.freeze_encoder,
            "encoder_lr": phase.encoder_lr,
            "decoder_lr": phase.decoder_lr,
        }
        for phase in phases
    ]


def _checkpoint_metadata(
    config: Any, phases: Sequence[PhaseSpec]
) -> dict[str, Any]:
    """Build compatibility metadata without requiring a model-specific schema."""
    explicit = _config_value(config, "checkpoint_metadata", {})
    if explicit is None:
        explicit = {}
    if not isinstance(explicit, Mapping):
        raise TypeError("checkpoint_metadata configuration must be a mapping")

    metadata: dict[str, Any] = {
        "phase_plan": _phase_plan(phases),
        "monitor": str(_config_value(config, "monitor", "mean_dice")),
        "direction": (
            "maximize"
            if _as_bool(_config_value(config, "maximize", True), True)
            else "minimize"
        ),
    }
    for key in ("architecture", "variant", "manifest_hash"):
        value = _config_value(config, key, None)
        if value is not None:
            metadata[key] = _to_plain(value)

    model_config = _config_value(config, "model_config", None)
    if model_config is None:
        model_config = _config_value(config, "source_model_config", None)
    if model_config is None:
        model_config = _config_value(config, "model", None)
    if model_config is not None:
        metadata["model_config"] = _to_plain(model_config)

    loss = _config_value(config, "loss", _config_value(config, "loss_fn", None))
    if loss is not None:
        metadata["loss"] = _callable_name(loss) if callable(loss) else _to_plain(loss)

    optimizer = _config_value(config, "optimizer", None)
    if optimizer is not None:
        metadata["optimizer"] = _to_plain(optimizer)
    scheduler = _config_value(config, "scheduler", None)
    if scheduler is not None or _config_value(config, "scheduler", _MISSING) is not _MISSING:
        metadata["scheduler"] = _to_plain(scheduler)

    metadata.update({str(key): _to_plain(value) for key, value in explicit.items()})
    return metadata


def _first_present(batch: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in batch:
            return batch[key]
    return None


def _unpack_training_batch(batch: Any) -> tuple[Any, Any]:
    if isinstance(batch, Mapping):
        image = _first_present(batch, ("image", "images", "input", "inputs", "x"))
        target = _first_present(batch, ("label", "labels", "target", "targets", "y"))
        if image is None or target is None:
            raise KeyError("training batch must contain image and label/target values")
        return image, target

    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]

    raise TypeError("training loader must yield a mapping or (image, target, ...) tuple")


def _move_to_device(value: Any, device: torch.device, name: str) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} could not be converted to a tensor")
    return tensor.to(device)


def _parameter_groups(model: nn.Module, phase: PhaseSpec) -> list[dict[str, Any]]:
    encoder = getattr(model, "encoder", None)
    if not isinstance(encoder, nn.Module):
        raise ValueError("model must expose an nn.Module named 'encoder'")
    encoder_module = encoder

    encoder_parameter_ids = {id(parameter) for parameter in encoder_module.parameters()}
    encoder_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) in encoder_parameter_ids and parameter.requires_grad
    ]
    other_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in encoder_parameter_ids and parameter.requires_grad
    ]

    groups: list[dict[str, Any]] = []
    if encoder_parameters:
        groups.append({"params": encoder_parameters, "lr": float(phase.encoder_lr)})
    if other_parameters:
        groups.append({"params": other_parameters, "lr": float(phase.decoder_lr)})
    if not groups:
        raise ValueError(f"phase '{phase.name}' has no trainable parameters")
    return groups


def _build_optimizer(model: nn.Module, phase: PhaseSpec, config: Any) -> torch.optim.Optimizer:
    options = dict(_as_mapping(_config_value(config, "optimizer", {})))
    name = str(options.pop("name", "adamw")).lower().replace("-", "").replace("_", "")
    options.pop("lr", None)
    if "weight_decay" not in options:
        configured_weight_decay = _config_value(config, "weight_decay", None)
        if configured_weight_decay is not None:
            options["weight_decay"] = configured_weight_decay

    optimizers: dict[str, type[torch.optim.Optimizer]] = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop,
    }
    try:
        optimizer_type = optimizers[name]
    except KeyError as exc:
        supported = ", ".join(sorted(optimizers))
        raise ValueError(f"unsupported optimizer '{name}'; choose from {supported}") from exc

    return optimizer_type(
        _parameter_groups(model, phase),
        lr=float(phase.decoder_lr),
        **options,
    )


def _loader_length(loader: Iterable[Any]) -> int | None:
    try:
        return len(loader)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    phase: PhaseSpec,
    config: Any,
    updates_per_epoch: int | None = None,
) -> tuple[Any, str]:
    raw_config = _config_value(config, "scheduler", None)
    if raw_config is None or raw_config is False:
        return None, "epoch"

    options = dict(_as_mapping(raw_config))
    name = str(options.pop("name", "none")).lower().replace("-", "").replace("_", "")
    interval = str(
        options.pop(
            "interval",
            "update" if _as_bool(options.pop("step_per_update", False), False) else "epoch",
        )
    ).lower()
    if interval not in {"epoch", "update", "step", "batch"}:
        raise ValueError("scheduler interval must be 'epoch' or 'update'")
    if interval in {"step", "batch"}:
        interval = "update"

    if name in {"none", "off", "disabled"}:
        return None, interval
    if name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(options.pop("step_size", 1)),
            gamma=float(options.pop("gamma", 0.1)),
            **options,
        )
    elif name in {"cosine", "cosineannealing"}:
        configured_t_max = options.pop("t_max", None)
        if configured_t_max is None:
            configured_t_max = options.pop("T_max", None)
        if configured_t_max is None:
            if interval == "update":
                if updates_per_epoch is None:
                    raise ValueError(
                        "cosine scheduler with interval='update' requires explicit "
                        "t_max for unknown-length training loaders"
                    )
                configured_t_max = max(1, phase.epochs * updates_per_epoch)
            else:
                configured_t_max = max(1, phase.epochs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(configured_t_max),
            eta_min=float(options.pop("eta_min", 0.0)),
            **options,
        )
    elif name in {"exponential", "explr"}:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=float(options.pop("gamma", 0.99)),
            **options,
        )
    elif name in {"multistep", "multisteplr"}:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(options.pop("milestones", [])),
            gamma=float(options.pop("gamma", 0.1)),
            **options,
        )
    else:
        raise ValueError(
            "unsupported scheduler "
            f"'{name}'; choose from cosine, exponential, multistep, none, or step"
        )
    return scheduler, interval


def _scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("training metrics must contain scalar tensors")
        return value.detach().cpu().item()
    if isinstance(value, Number):
        return float(value)
    return value


def _metric_is_better(value: float, best: float, maximize: bool) -> bool:
    if not math.isfinite(value):
        return False
    return value > best if maximize else value < best


def _metric_is_better_by_delta(
    value: float,
    best: float,
    maximize: bool,
    min_delta: float,
) -> bool:
    if not math.isfinite(value):
        return False
    if not math.isfinite(best):
        return True
    if maximize:
        return value > best + min_delta
    return value < best - min_delta


def _synchronize(device: torch.device) -> None:
    """Synchronize only at measured section boundaries."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _voxel_count(tensor: torch.Tensor) -> int:
    if tensor.ndim == 0:
        return 1
    batch = int(tensor.shape[0])
    spatial_shape = tensor.shape[2:] if tensor.ndim >= 2 else ()
    return batch * math.prod(int(dimension) for dimension in spatial_shape)


def _early_stopping_options(
    config: Any,
    configured: Mapping[str, Any] | Any | None,
    monitor: str,
    maximize: bool,
) -> dict[str, Any]:
    raw = configured
    if raw is None:
        raw = _section_value(config, "early_stopping", None)
    if raw is None or raw is False:
        raw = {}
    if not isinstance(raw, Mapping):
        raise TypeError("early_stopping configuration must be a mapping")

    mode = str(raw.get("mode", "max" if maximize else "min")).strip().lower()
    if mode in {"max", "maximize"}:
        early_maximize = True
    elif mode in {"min", "minimize"}:
        early_maximize = False
    else:
        raise ValueError("early_stopping mode must be 'max' or 'min'")

    patience = raw.get("patience", 10)
    if isinstance(patience, bool) or not isinstance(patience, int) or patience < 0:
        raise ValueError("early_stopping patience must be a non-negative integer")

    min_epochs = raw.get("min_epochs", 0)
    if isinstance(min_epochs, bool) or not isinstance(min_epochs, int) or min_epochs < 0:
        raise ValueError("early_stopping min_epochs must be a non-negative integer")

    min_delta = raw.get("min_delta", 0.0)
    try:
        min_delta = float(min_delta)
    except (TypeError, ValueError) as exc:
        raise ValueError("early_stopping min_delta must be finite and non-negative") from exc
    if not math.isfinite(min_delta) or min_delta < 0:
        raise ValueError("early_stopping min_delta must be finite and non-negative")

    configured_monitor = raw.get("monitor", monitor)
    if configured_monitor is None or not str(configured_monitor).strip():
        raise ValueError("early_stopping monitor must be non-empty")

    return {
        "enabled": _as_bool(raw.get("enabled", False), False),
        "monitor": str(configured_monitor),
        "maximize": early_maximize,
        "patience": int(patience),
        "min_delta": min_delta,
        "min_epochs": int(min_epochs),
    }


def _measured_seconds(state: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = state.get(key, default)
    if isinstance(value, bool) or not isinstance(value, Number):
        return default
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value) or value < 0:
        return default
    return value


def _power_value(snapshot: Any, key: str, default: Any = None) -> Any:
    if isinstance(snapshot, Mapping):
        return snapshot.get(key, default)
    return getattr(snapshot, key, default)


def _power_fields(snapshot: Any, *, status: str = "disabled") -> dict[str, Any]:
    if snapshot is None:
        return {
            "power/average_watts": None,
            "power/max_watts": None,
            "power/energy_joules": None,
            "power/sample_count": 0,
            "power/sample_interval_ms": None,
            "power/status": status,
        }
    interval_seconds = _power_value(snapshot, "interval_seconds")
    interval_ms = (
        None
        if interval_seconds is None
        else float(interval_seconds) * 1000.0
    )
    return {
        "power/average_watts": _power_value(snapshot, "average_watts"),
        "power/max_watts": _power_value(snapshot, "max_watts"),
        "power/energy_joules": _power_value(snapshot, "joules"),
        "power/sample_count": int(_power_value(snapshot, "samples", 0) or 0),
        "power/sample_interval_ms": interval_ms,
        "power/status": str(_power_value(snapshot, "status", status)),
    }


def _save_checkpoint(
    checkpoints: CheckpointManager | None,
    tag: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    state: Mapping[str, Any],
) -> None:
    if checkpoints is not None:
        checkpoints.save(tag, model, optimizer, scheduler, scaler, state)


def _optimizer_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    max_grad_norm: float | None,
    global_step: int,
    scheduler_interval: str,
    gradient_rescale: float = 1.0,
) -> int:
    scaler.unscale_(optimizer)
    if gradient_rescale != 1.0:
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(gradient_rescale)

    trainable_gradients = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    for parameter in trainable_gradients:
        if not torch.isfinite(parameter.grad).all():
            raise FloatingPointError("gradients must be finite before optimizer step")
    if max_grad_norm is not None and trainable_gradients:
        gradient_norm = nn.utils.clip_grad_norm_(trainable_gradients, max_grad_norm)
        if not math.isfinite(float(gradient_norm)):
            raise FloatingPointError("gradient norm must be finite")
    scaler.step(optimizer)
    scaler.update()
    global_step += 1
    if scheduler is not None and scheduler_interval == "update":
        scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return global_step


def _snapshot_config_is_cnn_imagefolder_incompatible(config: Any) -> bool:
    """Identify configurations whose targets are not segmentation masks."""

    candidates: list[str] = []
    sections = (
        "experiment",
        "model",
        "model_config",
        "source_model_config",
        "data",
        "dataset",
    )
    section_keys = (
        "name",
        "model_name",
        "family",
        "protocol",
        "dataset",
        "dataset_name",
        "dataset_id",
        "dataset_type",
        "dataset_class",
        "dataset_cls",
        "architecture",
        "loader",
        "type",
        "class",
    )
    for section_name in sections:
        section = _config_value(config, section_name, None)
        if isinstance(section, Mapping):
            candidates.extend(
                str(section[key]).strip().lower()
                for key in section_keys
                if section.get(key) is not None
            )
        elif section is not None:
            candidates.append(str(section).strip().lower())

    for key in (
        "model_name",
        "data_name",
        "dataset_id",
        "dataset_name",
        "dataset_type",
        "dataset_class",
        "architecture",
        "protocol",
        "protocol_id",
        "dataset",
        "imagefolder",
        "image_folder",
    ):
        value = _config_value(config, key, None)
        if value is not None:
            candidates.append(str(value).strip().lower())

    explicit_imagefolder_markers = (
        "imagefolder",
        "image_folder",
        "is_imagefolder",
        "is_image_folder",
        "uses_imagefolder",
        "uses_image_folder",
    )
    for marker_name in explicit_imagefolder_markers:
        value = _config_value(config, marker_name, None)
        if _as_bool(value, False):
            return True
        for section_name in ("data", "dataset"):
            section = _config_value(config, section_name, None)
            if isinstance(section, Mapping) and _as_bool(
                section.get(marker_name), False
            ):
                return True

    return any(
        "cnn" in value
        or "denois" in value
        or "imagenet" in value
        or "imagefolder" in value
        for value in candidates
    )


def _snapshot_config_requests_work(config: Any, tracker: Any) -> bool:
    if _snapshot_config_is_cnn_imagefolder_incompatible(config):
        return False

    visualization = _config_value(config, "visualization", {})
    if not isinstance(visualization, Mapping):
        return False
    settings = visualization.get("segmentation_snapshots", {})
    if not isinstance(settings, Mapping) or not _as_bool(
        settings.get("enabled", False), False
    ):
        return False
    local_enabled = _as_bool(settings.get("local_enabled", False), False)
    wandb_enabled = tracking_image_logging_enabled(config, tracker)
    return local_enabled or wandb_enabled


def _snapshotter_enabled(snapshotter: Any) -> bool:
    configured = getattr(snapshotter, "image_work_enabled", _MISSING)
    return True if configured is _MISSING else bool(configured)


def _snapshotter_option(snapshotter: Any, name: str, default: Any) -> Any:
    value = getattr(snapshotter, name, _MISSING)
    return default if value is _MISSING else value


def _snapshotter_state(snapshotter: Any) -> Any:
    state_dict = getattr(snapshotter, "state_dict", None)
    if not callable(state_dict):
        return None
    try:
        return state_dict()
    except BaseException:
        return None


def _restore_snapshotter_state(snapshotter: Any, state: Any) -> None:
    if state is None:
        return
    load_state_dict = getattr(snapshotter, "load_state_dict", None)
    if callable(load_state_dict):
        load_state_dict(state)


def _call_snapshotter(
    snapshotter: Any,
    *,
    model: nn.Module,
    train_loader: Iterable[Any],
    val_loader: Iterable[Any],
    epoch: int,
    global_step: int,
    kind: str,
) -> None:
    callback = getattr(snapshotter, "snapshot", None)
    if not callable(callback):
        if callable(snapshotter):
            callback = snapshotter
        else:
            raise TypeError("snapshotter must expose callable snapshot()")
    callback(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epoch=epoch,
        global_step=global_step,
        kind=kind,
    )


def _safe_snapshot(
    snapshotter: Any,
    *,
    model: nn.Module,
    train_loader: Iterable[Any],
    val_loader: Iterable[Any],
    epoch: int,
    global_step: int,
    kind: str,
) -> None:
    if not _snapshotter_enabled(snapshotter):
        return
    try:
        _call_snapshotter(
            snapshotter,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epoch=epoch,
            global_step=global_step,
            kind=kind,
        )
    except BaseException as error:
        record_error = getattr(snapshotter, "_record_error", None)
        if callable(record_error):
            try:
                record_error(error)
            except BaseException:
                pass


def _train_epoch(
    model: nn.Module,
    loader: Iterable[Any],
    loss_fn: Callable[[Any, Any], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    device: torch.device,
    accumulation_steps: int,
    max_grad_norm: float | None,
    global_step: int,
    scheduler_interval: str,
) -> EpochTrainResult:
    model.train()
    encoder = getattr(model, "encoder", None)
    if isinstance(encoder, nn.Module) and not any(
        parameter.requires_grad for parameter in encoder.parameters()
    ):
        encoder.eval()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    batches = 0
    samples = 0
    voxels = 0
    total_batches = _loader_length(loader)

    for batch_index, batch in enumerate(loader):
        image, target = _unpack_training_batch(batch)
        image_tensor = _move_to_device(image, device, "image")
        target_tensor = _move_to_device(target, device, "target")
        samples += int(image_tensor.shape[0]) if image_tensor.ndim > 0 else 1
        voxels += _voxel_count(image_tensor)
        with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            loss = loss_fn(model(image_tensor), target_tensor)
        if not isinstance(loss, torch.Tensor):
            loss = torch.as_tensor(loss, device=device)
        if loss.numel() != 1:
            raise ValueError(f"loss_fn must return a scalar tensor, got shape {tuple(loss.shape)}")

        loss_value = float(loss.detach().item())
        if not math.isfinite(loss_value):
            raise FloatingPointError(f"loss must be finite, got {loss_value}")

        window_start = batch_index - batch_index % accumulation_steps
        if total_batches is None:
            window_size = accumulation_steps
        else:
            window_size = min(accumulation_steps, total_batches - window_start)
        scaler.scale(loss / window_size).backward()
        total_loss += loss_value
        batches += 1

        is_update = (
            (batch_index + 1) % accumulation_steps == 0
            or (total_batches is not None and batch_index + 1 == total_batches)
        )
        if not is_update:
            continue

        global_step = _optimizer_step(
            model,
            optimizer,
            scheduler,
            scaler,
            max_grad_norm,
            global_step,
            scheduler_interval,
        )

    if total_batches is None and batches % accumulation_steps:
        # Unknown-length iterables cannot identify the final window in advance.
        # Rescale its gradients so a short window still represents its mean loss.
        remainder = batches % accumulation_steps
        global_step = _optimizer_step(
            model,
            optimizer,
            scheduler,
            scaler,
            max_grad_norm,
            global_step,
            scheduler_interval,
            gradient_rescale=accumulation_steps / remainder,
        )

    if batches == 0:
        raise ValueError("training loader yielded no batches")
    if scheduler is not None and scheduler_interval == "epoch":
        scheduler.step()
    return EpochTrainResult(total_loss / batches, global_step, samples, voxels)


def fit(
    model: nn.Module,
    train_loader: Iterable[Any],
    val_loader: Iterable[Any],
    loss_fn: Callable[[Any, Any], torch.Tensor],
    evaluator: Callable[[nn.Module, Iterable[Any]], Mapping[str, Any]],
    phases: Sequence[PhaseSpec],
    config: Mapping[str, Any] | Any,
    tracker: Tracker | Any,
    checkpoints: CheckpointManager | None,
    *,
    resume: Path | str | None = None,
    loader_generator: torch.Generator | None = None,
    warm_start: Path | str | None = None,
    finish_tracker: bool = True,
    snapshotter: Any | None = None,
    early_stopping: Mapping[str, Any] | Any | None = None,
) -> FitResult:
    """Train any encoder/decoder model through configured freezing phases."""
    tracker = tracker or Tracker()
    try:
        phase_specs = tuple(phases)
        if not phase_specs:
            raise ValueError("fit requires at least one phase")
        if any(phase.epochs < 0 for phase in phase_specs):
            raise ValueError("phase epochs must be non-negative")

        device = _resolve_device(_config_value(config, "device", "auto"))
        model.to(device)
        use_amp = _as_bool(_config_value(config, "use_amp", False), False)
        accumulation_steps = int(_config_value(config, "gradient_accumulation_steps", 1))
        if accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be at least one")

        configured_clip = _config_value(
            config,
            "max_grad_norm",
            _config_value(config, "grad_clip", None),
        )
        max_grad_norm = None if configured_clip is None else float(configured_clip)
        if max_grad_norm is not None and max_grad_norm < 0:
            raise ValueError("max_grad_norm must be non-negative")

        validation_interval = int(_config_value(config, "validation_interval", 1))
        if validation_interval < 1:
            raise ValueError("validation_interval must be at least one")
        monitor = str(_config_value(config, "monitor", "mean_dice"))
        maximize = _as_bool(_config_value(config, "maximize", True), True)
        manifest_hash = _config_value(config, "manifest_hash", None)
        run_config = _to_plain(config)
        checkpoint_metadata = _checkpoint_metadata(config, phase_specs)
        snapshot_config_incompatible = _snapshot_config_is_cnn_imagefolder_incompatible(
            config
        )
        if (
            snapshotter is None
            and not snapshot_config_incompatible
            and _snapshot_config_requests_work(config, tracker)
        ):
            # Keep snapshot imports behind the effective output gate. In
            # particular, a local profile with image logging disabled must not
            # import inference or plotting helpers merely to train.
            from token_mixer.evaluation.inference import build_segmentation_snapshotter

            snapshotter = build_segmentation_snapshotter(
                run_config,
                tracker,
                device=device,
            )
        elif snapshotter is not None:
            configured_visualization = _config_value(config, "visualization", _MISSING)
            configured_snapshots = (
                configured_visualization.get("segmentation_snapshots", _MISSING)
                if isinstance(configured_visualization, Mapping)
                else _MISSING
            )
            if (
                snapshot_config_incompatible
                or (
                    isinstance(configured_snapshots, Mapping)
                    and not _snapshot_config_requests_work(config, tracker)
                )
            ):
                # An injected callback cannot bypass an explicit feature/output
                # gate. The callback seam remains available to callers whose
                # configuration has no snapshot section at all.
                snapshotter = None
        scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp)
        total_train_batches = _loader_length(train_loader)
        updates_per_epoch = (
            None
            if total_train_batches is None
            else max(1, (total_train_batches + accumulation_steps - 1) // accumulation_steps)
        )

        best_metric = float("-inf") if maximize else float("inf")
        best_epoch = 0
        found_best = False
        best_model_state: dict[str, torch.Tensor] | None = None
        global_step = 0
        history: list[dict[str, Any]] = []
        absolute_epoch = 0
        total_epochs = sum(item.epochs for item in phase_specs)
        early_options = _early_stopping_options(
            config, early_stopping, monitor, maximize
        )
        early_best_metric = (
            float("-inf") if early_options["maximize"] else float("inf")
        )
        early_best_epoch = 0
        early_bad_evaluations = 0
        early_stopped = False
        early_stop_epoch: int | None = None
        timing_scope = "process_segment"
        segment_start: float | None = None
        segment_elapsed_seconds = 0.0
        cumulative_train_seconds = 0.0
        cumulative_validation_seconds = 0.0
        cumulative_elapsed_seconds = 0.0
        segment_base_elapsed_seconds = 0.0
        time_to_best_seconds: float | None = None
        time_to_best_scope = "process_segment"
        power_sampler: Any | None = None
        power_setup_status = "disabled"
        final_power_snapshot: Any | None = None
        latest_checkpoint_state: dict[str, Any] | None = None
        efficiency_config = _as_mapping(_section_value(config, "efficiency", {}))
        nvml_config = _as_mapping(efficiency_config.get("nvml", {}))
        nvml_enabled = _as_bool(nvml_config.get("enabled", False), False)
        try:
            nvml_device_index = int(
                nvml_config.get("device_index", device.index or 0)
            )
        except (TypeError, ValueError):
            nvml_device_index = -1
        try:
            nvml_interval_seconds = float(
                nvml_config.get("sample_interval_seconds", 0.25)
            )
        except (TypeError, ValueError):
            nvml_interval_seconds = 0.0
        define_metric = getattr(tracker, "define_metric", None)
        if callable(define_metric):
            for metric_name in (
                "train/epoch_seconds",
                "train/optimizer_steps",
                "train/samples",
                "train/voxels",
                "train/samples_per_second",
                "train/voxels_per_second",
                "train/peak_memory_allocated_gb",
                "train/peak_memory_reserved_gb",
                "train/time_to_best_seconds",
                "val/epoch_seconds",
                "power/average_watts",
                "power/max_watts",
                "power/energy_joules",
                "power/sample_count",
                "power/sample_interval_ms",
                "power/status",
                "run/elapsed_seconds",
            ):
                if metric_name.startswith("train/"):
                    try:
                        define_metric(metric_name, step_metric="train/epoch")
                    except TypeError:
                        define_metric(metric_name)
                elif metric_name.startswith("val/"):
                    try:
                        define_metric(metric_name, step_metric="val/epoch")
                    except TypeError:
                        define_metric(metric_name)
                else:
                    try:
                        define_metric(metric_name, step_metric="train/epoch")
                    except TypeError:
                        define_metric(metric_name)
        resume_path = None if resume is None else Path(resume)
        warm_start_path = None if warm_start is None else Path(warm_start)
        if resume_path is not None and warm_start_path is not None:
            raise ValueError("resume and warm_start are mutually exclusive")
        if (resume_path is not None or warm_start_path is not None) and checkpoints is None:
            raise ValueError("resume and warm_start require a checkpoint manager")
        resume_phase_index = 1
        resume_phase_epoch = 0
        resume_loaded = False
        if warm_start_path is not None:
            checkpoints.load_model(
                warm_start_path,
                model,
                expected_metadata=checkpoint_metadata,
            )
        if resume_path is not None:
            resume_payload = checkpoints.read(resume_path)
            checkpoints.validate(resume_payload, checkpoint_metadata)
            resume_state = resume_payload.get("state", resume_payload)
            if not isinstance(resume_state, Mapping):
                raise ValueError("resume checkpoint state must be a mapping")
            try:
                resume_phase_index = int(resume_state.get("phase_index", 1))
                resume_phase_epoch = int(resume_state.get("phase_epoch", 0))
                absolute_epoch = int(resume_state.get("epoch", 0))
                global_step = int(resume_state.get("global_step", 0))
            except (TypeError, ValueError) as exc:
                raise ValueError("resume checkpoint contains invalid training state") from exc
            if resume_phase_index < 1 or resume_phase_index > len(phase_specs):
                raise ValueError("resume checkpoint phase_index is outside configured phases")
            if resume_phase_epoch < 0 or absolute_epoch < 0 or global_step < 0:
                raise ValueError("resume checkpoint contains negative training state")
            saved_phase_name = resume_state.get("phase")
            if (
                saved_phase_name is not None
                and str(saved_phase_name) != phase_specs[resume_phase_index - 1].name
            ):
                raise ValueError(
                    "resume checkpoint phase is incompatible with configured phases: "
                    f"checkpoint={saved_phase_name!r}, "
                    f"configured={phase_specs[resume_phase_index - 1].name!r}"
                )
            if resume_phase_epoch > phase_specs[resume_phase_index - 1].epochs:
                raise ValueError(
                    "resume checkpoint phase_epoch exceeds configured phase epochs"
                )
            saved_best_metric = resume_state.get("best_metric")
            saved_best_epoch = resume_state.get("best_epoch")
            if saved_best_metric is not None:
                try:
                    best_metric = float(saved_best_metric)
                except (TypeError, ValueError) as exc:
                    raise ValueError("resume checkpoint contains invalid best_metric") from exc
                found_best = math.isfinite(best_metric)
            if saved_best_epoch is not None:
                try:
                    best_epoch = int(saved_best_epoch)
                except (TypeError, ValueError) as exc:
                    raise ValueError("resume checkpoint contains invalid best_epoch") from exc
            if best_epoch < 0:
                raise ValueError("resume checkpoint contains negative best_epoch")
            saved_early_stopped = resume_state.get("early_stopping_stopped")
            if isinstance(saved_early_stopped, bool):
                early_stopped = saved_early_stopped
            saved_early_stop_epoch = resume_state.get("early_stopping_stop_epoch")
            if isinstance(saved_early_stop_epoch, int) and not isinstance(
                saved_early_stop_epoch, bool
            ):
                early_stop_epoch = max(0, saved_early_stop_epoch)
            saved_early_best = resume_state.get("early_stopping_best_metric")
            if isinstance(saved_early_best, Number) and not isinstance(
                saved_early_best, bool
            ):
                saved_early_best = float(saved_early_best)
                if math.isfinite(saved_early_best):
                    early_best_metric = saved_early_best
            saved_early_epoch = resume_state.get("early_stopping_best_epoch")
            if isinstance(saved_early_epoch, int) and not isinstance(
                saved_early_epoch, bool
            ):
                early_best_epoch = max(0, saved_early_epoch)
            saved_bad_evaluations = resume_state.get(
                "early_stopping_bad_evaluations", 0
            )
            if isinstance(saved_bad_evaluations, int) and not isinstance(
                saved_bad_evaluations, bool
            ):
                early_bad_evaluations = max(0, saved_bad_evaluations)
            if early_stopped:
                early_options["enabled"] = True
                if early_stop_epoch is None:
                    early_stop_epoch = absolute_epoch
                if early_best_epoch == 0:
                    early_best_epoch = best_epoch
            cumulative_train_seconds = _measured_seconds(
                resume_state,
                "cumulative_train_seconds",
                _measured_seconds(resume_state, "measured_train_seconds"),
            )
            cumulative_validation_seconds = _measured_seconds(
                resume_state,
                "cumulative_validation_seconds",
                _measured_seconds(resume_state, "measured_validation_seconds"),
            )
            cumulative_elapsed_seconds = _measured_seconds(
                resume_state,
                "cumulative_elapsed_seconds",
                _measured_seconds(
                    resume_state,
                    "measured_elapsed_seconds",
                    _measured_seconds(resume_state, "segment_elapsed_seconds"),
                ),
            )
            time_to_best_seconds = resume_state.get("time_to_best_seconds")
            if not isinstance(time_to_best_seconds, Number) or isinstance(
                time_to_best_seconds, bool
            ):
                time_to_best_seconds = None
            else:
                time_to_best_seconds = float(time_to_best_seconds)
                if not math.isfinite(time_to_best_seconds) or time_to_best_seconds < 0:
                    time_to_best_seconds = None
            time_to_best_scope = "cumulative_measured"
            segment_base_elapsed_seconds = cumulative_elapsed_seconds
            saved_history = resume_state.get("history", [])
            if saved_history is None:
                saved_history = []
            if not isinstance(saved_history, (list, tuple)):
                raise ValueError("resume checkpoint history must be a sequence")
            for record in saved_history:
                if not isinstance(record, Mapping):
                    raise ValueError("resume checkpoint history entries must be mappings")
            history = [dict(_to_plain(record)) for record in saved_history]
            saved_snapshot_state = resume_state.get(
                "snapshot_state", resume_state.get("snapshotter_state")
            )
            if saved_snapshot_state is not None and snapshotter is not None:
                try:
                    _restore_snapshotter_state(snapshotter, saved_snapshot_state)
                except BaseException as error:
                    record_error = getattr(snapshotter, "_record_error", None)
                    if callable(record_error):
                        try:
                            record_error(error)
                        except BaseException:
                            pass
    except BaseException:
        if finish_tracker:
            try:
                tracker.finish()
            except BaseException:
                pass
        raise

    try:
        if resume_path is not None and early_stopped:
            checkpoints.load(
                resume_path,
                model,
                generator=loader_generator,
                expected_metadata=checkpoint_metadata,
            )
            resume_loaded = True
        for phase_index, phase in enumerate(phase_specs, start=1):
            if resume_path is not None and early_stopped:
                break
            if phase.epochs == 0:
                continue
            if resume_path is not None and phase_index < resume_phase_index:
                continue
            apply_phase(model, phase)
            optimizer = _build_optimizer(model, phase, config)
            scheduler, scheduler_interval = _build_scheduler(
                optimizer,
                phase,
                config,
                updates_per_epoch=updates_per_epoch,
            )

            phase_epoch_start = 1
            if resume_path is not None and not resume_loaded:
                checkpoints.load(
                    resume_path,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    generator=loader_generator,
                    expected_metadata=checkpoint_metadata,
                )
                resume_loaded = True
                phase_epoch_start = resume_phase_epoch + 1
                if phase_epoch_start > phase.epochs:
                    continue
                if early_stopped:
                    break

            if segment_start is None:
                segment_start = time.perf_counter()
                if nvml_enabled and device.type == "cuda":
                    try:
                        power_sampler = NvmlPowerSampler(
                            device_index=nvml_device_index,
                            interval_seconds=nvml_interval_seconds,
                        ).start()
                        power_setup_status = "ok"
                    except BaseException:
                        power_sampler = None
                        power_setup_status = "unavailable"

            last_phase_epoch = phase_epoch_start - 1
            for phase_epoch in range(phase_epoch_start, phase.epochs + 1):
                last_phase_epoch = phase_epoch
                absolute_epoch += 1
                previous_global_step = global_step
                reset_peak_memory(device)
                _synchronize(device)
                train_started = time.perf_counter()
                train_result = _train_epoch(
                    model,
                    train_loader,
                    loss_fn,
                    optimizer,
                    scheduler,
                    scaler,
                    device,
                    accumulation_steps,
                    max_grad_norm,
                    global_step,
                    scheduler_interval,
                )
                _synchronize(device)
                train_seconds = time.perf_counter() - train_started
                global_step = train_result.global_step
                cumulative_train_seconds += train_seconds

                record: dict[str, Any] = {
                    "epoch": absolute_epoch,
                    "phase": phase.name,
                    "phase_index": phase_index,
                    "phase_epoch": phase_epoch,
                    "train_loss": train_result.train_loss,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "global_step": global_step,
                    "train/epoch": absolute_epoch,
                    "train/epoch_seconds": train_seconds,
                    "train/optimizer_steps": global_step - previous_global_step,
                    "train/samples": train_result.samples,
                    "train/voxels": train_result.voxels,
                    "train/samples_per_second": (
                        train_result.samples / train_seconds
                        if train_seconds > 0
                        else 0.0
                    ),
                    "train/voxels_per_second": (
                        train_result.voxels / train_seconds
                        if train_seconds > 0
                        else 0.0
                    ),
                }
                encoder = getattr(model, "encoder", None)
                encoder_parameter_ids = (
                    {id(parameter) for parameter in encoder.parameters()}
                    if isinstance(encoder, nn.Module)
                    else set()
                )
                for group in optimizer.param_groups:
                    group_parameters = group.get("params", ())
                    if any(id(parameter) in encoder_parameter_ids for parameter in group_parameters):
                        record["encoder_lr"] = float(group["lr"])
                    else:
                        record["decoder_lr"] = float(group["lr"])
                memory = peak_memory_gb(device)
                record["train/peak_memory_allocated_gb"] = memory["allocated_gb"]
                record["train/peak_memory_reserved_gb"] = memory["reserved_gb"]
                metric_value: float | None = None
                validation_seconds: float | None = None
                new_best = False
                should_validate = (
                    absolute_epoch % validation_interval == 0
                    or absolute_epoch == total_epochs
                )
                if should_validate:
                    _synchronize(device)
                    validation_started = time.perf_counter()
                    model.eval()
                    with torch.no_grad(), torch.amp.autocast(
                        device_type=device.type,
                        enabled=use_amp,
                    ):
                        validation_metrics = evaluator(model, val_loader)
                    _synchronize(device)
                    validation_seconds = time.perf_counter() - validation_started
                    cumulative_validation_seconds += validation_seconds
                    if not isinstance(validation_metrics, Mapping):
                        raise TypeError("evaluator must return a mapping of scalar metrics")
                    for key, value in validation_metrics.items():
                        record[str(key)] = _scalar(value)
                    if monitor not in validation_metrics:
                        raise KeyError(f"evaluator did not return monitored metric '{monitor}'")
                    raw_metric = _scalar(validation_metrics[monitor])
                    try:
                        metric_value = float(raw_metric)
                    except (TypeError, ValueError) as exc:
                        raise TypeError(f"monitored metric '{monitor}' must be numeric") from exc
                    if not math.isfinite(metric_value):
                        raise FloatingPointError(
                            f"monitored metric '{monitor}' must be finite, got {metric_value}"
                        )

                    if _metric_is_better(metric_value, best_metric, maximize):
                        best_metric = metric_value
                        best_epoch = absolute_epoch
                        found_best = True
                        new_best = True
                        best_model_state = {
                            name: value.detach().cpu().clone()
                            for name, value in model.state_dict().items()
                        }

                    if early_options["enabled"]:
                        early_monitor = early_options["monitor"]
                        if early_monitor not in validation_metrics:
                            raise KeyError(
                                "evaluator did not return early-stopping metric "
                                f"'{early_monitor}'"
                            )
                        raw_early_metric = _scalar(validation_metrics[early_monitor])
                        try:
                            early_metric = float(raw_early_metric)
                        except (TypeError, ValueError) as exc:
                            raise TypeError(
                                f"early-stopping metric '{early_monitor}' must be numeric"
                            ) from exc
                        if not math.isfinite(early_metric):
                            raise FloatingPointError(
                                "early-stopping metric "
                                f"'{early_monitor}' must be finite, got {early_metric}"
                            )
                        early_improved = _metric_is_better_by_delta(
                            early_metric,
                            early_best_metric,
                            early_options["maximize"],
                            early_options["min_delta"],
                        )
                        if early_improved:
                            early_best_metric = early_metric
                            early_best_epoch = absolute_epoch
                            early_bad_evaluations = 0
                        else:
                            early_bad_evaluations += 1
                        if (
                            not early_improved
                            and absolute_epoch >= early_options["min_epochs"]
                            and early_bad_evaluations >= early_options["patience"]
                        ):
                            early_stopped = True
                            early_stop_epoch = absolute_epoch

                    record["val/epoch"] = absolute_epoch
                    record["val/epoch_seconds"] = validation_seconds

                if segment_start is None:
                    raise RuntimeError("training timer was not initialized")
                segment_elapsed_seconds = max(
                    0.0, time.perf_counter() - segment_start
                )
                cumulative_elapsed_seconds = (
                    segment_base_elapsed_seconds + segment_elapsed_seconds
                )
                if new_best:
                    time_to_best_seconds = cumulative_elapsed_seconds
                    time_to_best_scope = (
                        "cumulative_measured"
                        if resume_path is not None
                        else "process_segment"
                    )
                    record["train/time_to_best_seconds"] = time_to_best_seconds
                    record["train/time_to_best_scope"] = time_to_best_scope

                if power_sampler is not None:
                    try:
                        power_snapshot = power_sampler.snapshot()
                        power_record = _power_fields(power_snapshot)
                    except BaseException:
                        power_record = _power_fields(None, status="unavailable")
                else:
                    power_record = _power_fields(None, status=power_setup_status)
                record.update(power_record)
                record["run/elapsed_seconds"] = segment_elapsed_seconds
                record["timing_scope"] = timing_scope
                record["early_stopping/stopped"] = early_stopped
                record["early_stopping/stop_epoch"] = early_stop_epoch
                record["early_stopping/best_epoch"] = (
                    early_best_epoch if early_options["enabled"] else best_epoch
                )

                history.append(record)
                tracker.log(redact_case_identifiers(record), step=global_step)

                checkpoint_state: dict[str, Any] = {
                    "epoch": absolute_epoch,
                    "phase": phase.name,
                    "phase_index": phase_index,
                    "phase_epoch": phase_epoch,
                    "global_step": global_step,
                    "metric": metric_value,
                    "monitored_metric": metric_value,
                    "best_metric": best_metric if found_best else None,
                    "best_epoch": best_epoch,
                    "timing_scope": timing_scope,
                    "segment_elapsed_seconds": segment_elapsed_seconds,
                    "cumulative_train_seconds": cumulative_train_seconds,
                    "cumulative_validation_seconds": cumulative_validation_seconds,
                    "cumulative_elapsed_seconds": cumulative_elapsed_seconds,
                    "measured_train_seconds": cumulative_train_seconds,
                    "measured_validation_seconds": cumulative_validation_seconds,
                    "measured_elapsed_seconds": cumulative_elapsed_seconds,
                    "time_to_best_seconds": time_to_best_seconds,
                    "time_to_best_scope": time_to_best_scope,
                    "early_stopping_stopped": early_stopped,
                    "early_stopping_stop_epoch": early_stop_epoch,
                    "early_stopping_best_epoch": (
                        early_best_epoch if early_options["enabled"] else best_epoch
                    ),
                    "early_stopping_best_metric": (
                        early_best_metric if early_options["enabled"] else None
                    ),
                    "early_stopping_bad_evaluations": early_bad_evaluations,
                    "manifest_hash": manifest_hash,
                    "config": run_config,
                    "checkpoint_metadata": checkpoint_metadata,
                    "history": history,
                }
                if loader_generator is not None:
                    checkpoint_state["loader_generator"] = loader_generator
                scheduled_snapshot = False
                if snapshotter is not None and _snapshotter_enabled(snapshotter):
                    try:
                        snapshot_interval = int(
                            _snapshotter_option(snapshotter, "interval_epochs", 10)
                        )
                    except (TypeError, ValueError):
                        snapshot_interval = 10
                    scheduled_snapshot = (
                        snapshot_interval > 0
                        and absolute_epoch % snapshot_interval == 0
                    )
                    if scheduled_snapshot:
                        _safe_snapshot(
                            snapshotter,
                            model=model,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            epoch=absolute_epoch,
                            global_step=global_step,
                            kind="epoch",
                        )
                    if (
                        new_best
                        and _as_bool(
                            _snapshotter_option(snapshotter, "include_best", True),
                            True,
                        )
                        and not scheduled_snapshot
                    ):
                        _safe_snapshot(
                            snapshotter,
                            model=model,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            epoch=absolute_epoch,
                            global_step=global_step,
                            kind="best",
                        )
                    snapshot_state = _snapshotter_state(snapshotter)
                    if snapshot_state is not None:
                        checkpoint_state["snapshot_state"] = snapshot_state
                    snapshot_errors = getattr(snapshotter, "errors", None)
                    if snapshot_errors is not None:
                        checkpoint_state["snapshot_errors"] = list(snapshot_errors)
                latest_checkpoint_state = dict(checkpoint_state)
                if metric_value is not None and found_best and best_epoch == absolute_epoch:
                    _save_checkpoint(
                        checkpoints,
                        "best",
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        checkpoint_state,
                    )
                _save_checkpoint(
                    checkpoints,
                    "last",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    checkpoint_state,
                )

                if early_stopped:
                    break

            phase_state = {
                "epoch": absolute_epoch,
                "phase": phase.name,
                "phase_index": phase_index,
                "global_step": global_step,
                "metric": history[-1].get(monitor),
                "monitored_metric": history[-1].get(monitor),
                "best_metric": best_metric if found_best else None,
                "best_epoch": best_epoch,
                "timing_scope": timing_scope,
                "segment_elapsed_seconds": segment_elapsed_seconds,
                "cumulative_train_seconds": cumulative_train_seconds,
                "cumulative_validation_seconds": cumulative_validation_seconds,
                "cumulative_elapsed_seconds": cumulative_elapsed_seconds,
                "measured_train_seconds": cumulative_train_seconds,
                "measured_validation_seconds": cumulative_validation_seconds,
                "measured_elapsed_seconds": cumulative_elapsed_seconds,
                "time_to_best_seconds": time_to_best_seconds,
                "time_to_best_scope": time_to_best_scope,
                "early_stopping_stopped": early_stopped,
                "early_stopping_stop_epoch": early_stop_epoch,
                "early_stopping_best_epoch": (
                    early_best_epoch if early_options["enabled"] else best_epoch
                ),
                "early_stopping_best_metric": (
                    early_best_metric if early_options["enabled"] else None
                ),
                "early_stopping_bad_evaluations": early_bad_evaluations,
                "manifest_hash": manifest_hash,
                "config": run_config,
                "phase_epoch": (
                    last_phase_epoch if early_stopped else phase.epochs
                ),
                "checkpoint_metadata": checkpoint_metadata,
                "history": history,
            }
            if loader_generator is not None:
                phase_state["loader_generator"] = loader_generator
            if snapshotter is not None:
                snapshot_state = _snapshotter_state(snapshotter)
                if snapshot_state is not None:
                    phase_state["snapshot_state"] = snapshot_state
                snapshot_errors = getattr(snapshotter, "errors", None)
                if snapshot_errors is not None:
                    phase_state["snapshot_errors"] = list(snapshot_errors)
            _save_checkpoint(
                checkpoints,
                f"{phase.name}_resume",
                model,
                optimizer,
                scheduler,
                scaler,
                phase_state,
            )
            if early_stopped:
                break

        if resume_path is not None and not resume_loaded:
            raise ValueError(
                "resume checkpoint phase could not be applied to configured phases"
            )
        if (
            snapshotter is not None
            and _snapshotter_enabled(snapshotter)
            and _as_bool(
                _snapshotter_option(snapshotter, "include_final", True), True
            )
            and absolute_epoch > 0
        ):
            model_state_before_final = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            final_best_model_state = best_model_state
            best_path = (
                None
                if checkpoints is None
                else checkpoints.root / "best.pt"
            )
            if best_path is not None and best_path.is_file():
                try:
                    checkpoints.load_model(
                        best_path,
                        model,
                        expected_metadata=checkpoint_metadata,
                    )
                    final_best_model_state = {
                        name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items()
                    }
                except BaseException as error:
                    record_error = getattr(snapshotter, "_record_error", None)
                    if callable(record_error):
                        try:
                            record_error(error)
                        except BaseException:
                            pass
            if final_best_model_state is not None:
                try:
                    model.load_state_dict(final_best_model_state)
                except BaseException as error:
                    record_error = getattr(snapshotter, "_record_error", None)
                    if callable(record_error):
                        try:
                            record_error(error)
                        except BaseException:
                            pass
            if "val" in _snapshotter_option(snapshotter, "splits", ("train", "val")) and _as_bool(
                _snapshotter_option(snapshotter, "final_validation", False), False
            ):
                try:
                    model.eval()
                    with torch.no_grad(), torch.amp.autocast(
                        device_type=device.type,
                        enabled=use_amp,
                    ):
                        evaluator(model, val_loader)
                except BaseException as error:
                    record_error = getattr(snapshotter, "_record_error", None)
                    if callable(record_error):
                        try:
                            record_error(error)
                        except BaseException:
                            pass
            _safe_snapshot(
                snapshotter,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epoch=absolute_epoch,
                global_step=global_step,
                kind="final",
            )
            if (
                checkpoints is not None
                and latest_checkpoint_state is not None
                and callable(getattr(snapshotter, "state_dict", None))
            ):
                model.load_state_dict(model_state_before_final)
                final_checkpoint_state = dict(latest_checkpoint_state)
                snapshot_state = _snapshotter_state(snapshotter)
                if snapshot_state is not None:
                    final_checkpoint_state["snapshot_state"] = snapshot_state
                snapshot_errors = getattr(snapshotter, "errors", None)
                if snapshot_errors is not None:
                    final_checkpoint_state["snapshot_errors"] = list(snapshot_errors)
                final_checkpoint_state["history"] = history
                _save_checkpoint(
                    checkpoints,
                    "last",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    final_checkpoint_state,
                )
                if final_best_model_state is not None:
                    model.load_state_dict(final_best_model_state)
        summary_metric = best_metric if found_best else float("nan")
        tracker.log_summary(
            redact_case_identifiers(
                {
                    "best_metric": summary_metric,
                    "best_epoch": best_epoch,
                    "global_step": global_step,
                    "early_stopping/stopped": early_stopped,
                    "early_stopping/stop_epoch": early_stop_epoch,
                    "early_stopping/best_epoch": (
                        early_best_epoch if early_options["enabled"] else best_epoch
                    ),
                }
            )
        )
    except BaseException:
        if finish_tracker:
            try:
                tracker.finish()
            except BaseException:
                pass
        raise
    finally:
        if power_sampler is not None:
            try:
                final_power_snapshot = power_sampler.stop()
            except BaseException:
                final_power_snapshot = None

    if finish_tracker:
        tracker.finish()

    try:
        final_power_fields = _power_fields(
            final_power_snapshot,
            status=power_setup_status if power_sampler is None else "unavailable",
        )
    except BaseException:
        final_power_fields = _power_fields(None, status="unavailable")
    metadata = {
        "timing_scope": timing_scope,
        "segment_elapsed_seconds": segment_elapsed_seconds,
        "cumulative_train_seconds": cumulative_train_seconds,
        "cumulative_validation_seconds": cumulative_validation_seconds,
        "cumulative_elapsed_seconds": cumulative_elapsed_seconds,
        "measured_train_seconds": cumulative_train_seconds,
        "measured_validation_seconds": cumulative_validation_seconds,
        "measured_elapsed_seconds": cumulative_elapsed_seconds,
        "time_to_best_seconds": time_to_best_seconds,
        "time_to_best_scope": time_to_best_scope,
        "train/time_to_best_scope": time_to_best_scope,
        "early_stopping/stopped": early_stopped,
        "early_stopping/stop_epoch": early_stop_epoch,
        "early_stopping/best_epoch": (
            early_best_epoch if early_options["enabled"] else best_epoch
        ),
        "snapshot_errors": list(getattr(snapshotter, "errors", ()))
        if snapshotter is not None
        else [],
        **final_power_fields,
    }
    return FitResult(summary_metric, best_epoch, history, metadata=metadata)
