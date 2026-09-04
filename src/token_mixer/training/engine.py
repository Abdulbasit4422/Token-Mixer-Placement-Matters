from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Number
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

from .checkpoints import CheckpointManager
from .phases import PhaseSpec, apply_phase
from .tracking import Tracker


_MISSING = object()


@dataclass(frozen=True)
class FitResult:
    """Summary returned by the model-independent training engine."""

    best_metric: float
    best_epoch: int
    history: list[dict[str, Any]]
    test_metrics: dict[str, float] | None = None
    metadata: dict[str, Any] | None = None


def _config_value(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


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
) -> tuple[float, int]:
    model.train()
    encoder = getattr(model, "encoder", None)
    if isinstance(encoder, nn.Module) and not any(
        parameter.requires_grad for parameter in encoder.parameters()
    ):
        encoder.eval()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    batches = 0
    total_batches = _loader_length(loader)

    for batch_index, batch in enumerate(loader):
        image, target = _unpack_training_batch(batch)
        image_tensor = _move_to_device(image, device, "image")
        target_tensor = _move_to_device(target, device, "target")
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
    return total_loss / batches, global_step


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
        global_step = 0
        history: list[dict[str, Any]] = []
        absolute_epoch = 0
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
            if loader_generator is not None:
                saved_history = resume_state.get("history", [])
                if saved_history is None:
                    saved_history = []
                if not isinstance(saved_history, (list, tuple)):
                    raise ValueError("resume checkpoint history must be a sequence")
                for record in saved_history:
                    if not isinstance(record, Mapping):
                        raise ValueError("resume checkpoint history entries must be mappings")
                history = [dict(_to_plain(record)) for record in saved_history]
    except BaseException:
        try:
            tracker.finish()
        except BaseException:
            pass
        raise

    try:
        for phase_index, phase in enumerate(phase_specs, start=1):
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

            for phase_epoch in range(phase_epoch_start, phase.epochs + 1):
                absolute_epoch += 1
                train_loss, global_step = _train_epoch(
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

                record: dict[str, Any] = {
                    "epoch": absolute_epoch,
                    "phase": phase.name,
                    "phase_index": phase_index,
                    "phase_epoch": phase_epoch,
                    "train_loss": train_loss,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "global_step": global_step,
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
                metric_value: float | None = None
                should_validate = (
                    absolute_epoch % validation_interval == 0
                    or absolute_epoch == sum(item.epochs for item in phase_specs)
                )
                if should_validate:
                    model.eval()
                    with torch.no_grad(), torch.amp.autocast(
                        device_type=device.type,
                        enabled=use_amp,
                    ):
                        validation_metrics = evaluator(model, val_loader)
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

                history.append(record)
                tracker.log(record, step=global_step)

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
                    "manifest_hash": manifest_hash,
                    "config": run_config,
                    "checkpoint_metadata": checkpoint_metadata,
                    "history": history,
                }
                if loader_generator is not None:
                    checkpoint_state["loader_generator"] = loader_generator
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

            phase_state = {
                "epoch": absolute_epoch,
                "phase": phase.name,
                "phase_index": phase_index,
                "global_step": global_step,
                "metric": history[-1].get(monitor),
                "monitored_metric": history[-1].get(monitor),
                "best_metric": best_metric if found_best else None,
                "best_epoch": best_epoch,
                "manifest_hash": manifest_hash,
                "config": run_config,
                "phase_epoch": phase.epochs,
                "checkpoint_metadata": checkpoint_metadata,
                "history": history,
            }
            if loader_generator is not None:
                phase_state["loader_generator"] = loader_generator
            _save_checkpoint(
                checkpoints,
                f"{phase.name}_resume",
                model,
                optimizer,
                scheduler,
                scaler,
                phase_state,
            )

        if resume_path is not None and not resume_loaded:
            raise ValueError(
                "resume checkpoint phase could not be applied to configured phases"
            )
        summary_metric = best_metric if found_best else float("nan")
        tracker.log_summary(
            {
                "best_metric": summary_metric,
                "best_epoch": best_epoch,
                "global_step": global_step,
            }
        )
    except BaseException:
        try:
            tracker.finish()
        except BaseException:
            pass
        raise

    tracker.finish()
    return FitResult(summary_metric, best_epoch, history)
