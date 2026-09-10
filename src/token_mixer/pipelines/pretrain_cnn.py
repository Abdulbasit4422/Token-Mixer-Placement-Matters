"""Pipeline boundary for 2-D ImageNet denoising pretraining."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Integral
from pathlib import Path
from typing import Any, cast

import torch
import torchvision.transforms as T
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from token_mixer.models.cnn_pretrain import (
    DenoisingDataset,
    build_denoising_model,
    evaluate_denoising,
    mse_loss,
)
from token_mixer.pipelines._baseline_common import (
    _copy_resume_best,
    _flatten_training_config,
    _invoke_fit,
    _max_cases,
    resume_path,
    warm_start_path,
)
from token_mixer.reproducibility import seed_everything, seed_worker
from token_mixer.training.checkpoints import CheckpointManager
from token_mixer.training.engine import FitResult, _checkpoint_metadata as _engine_checkpoint_metadata, _resolve_device, fit
from token_mixer.training.phases import PhaseSpec
from token_mixer.training.tracking import create_tracker


_MISSING = object()
_MODEL_DEFAULTS: dict[str, Any] = {
    "in_channels": 3,
    "feature_size": 32,
    "depths": (1, 1, 1, 1),
    "image_size": 96,
    "mlp_ratio": 4.0,
}
# Supported root-script defaults. The shared engine supports cosine decay, but
# warmup, periodic saves, and early stopping remain unsupported and are not
# emulated here.
_ROOT_SUPPORTED_DEFAULTS: dict[str, Any] = {
    "seed": 42,
    "batch_size": 5,
    "num_workers": 4,
    "pin_memory": True,
    "persistent_workers": True,
    "drop_last": True,
    "epochs": 3,
    "learning_rate": 1e-3,
    "min_lr": 1e-6,
    "noise_std": 0.15,
    "val_fraction": 0.05,
    "weight_decay": 0.05,
    "max_grad_norm": 1.0,
    "use_amp": True,
}
_UNSUPPORTED_ENGINE_CONTROLS = frozenset(
    {"warmup_epochs", "warmup_lr", "save_every", "early_stop", "log_every"}
)


def _path_value(config: Any, path: Sequence[str], default: Any = _MISSING) -> Any:
    current = config
    for key in path:
        if isinstance(current, Mapping):
            if key not in current:
                return default
            current = current[key]
        else:
            try:
                current = getattr(current, key)
            except AttributeError:
                return default
    return current


def _first_value(config: Any, paths: Sequence[Sequence[str]], default: Any = None) -> Any:
    for path in paths:
        value = _path_value(config, path)
        if value is not _MISSING and value is not None:
            return value
    return default


def _first_configured(
    config: Any, paths: Sequence[Sequence[str]], default: Any = _MISSING
) -> Any:
    """Return first configured value, retaining an explicit ``None``."""
    for path in paths:
        value = _path_value(config, path)
        if value is not _MISSING:
            return value
    return default


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_plain(item) for item in value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain(item) for item in value]
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} configuration must be a mapping")
    return value


def _loader_value(
    cfg: DictConfig | Mapping[str, Any], key: str, default: Any
) -> Any:
    return _first_value(
        cfg,
        (
            ("loader", key),
            ("experiment", "training", key),
            ("data", key),
            ("training", key),
            ("run", key),
            (key,),
        ),
        default=default,
    )


def _model_config(cfg: DictConfig | Mapping[str, Any]) -> dict[str, Any]:
    configured = _first_value(cfg, (("model",),), default={})
    if configured is None:
        configured = {}
    if not isinstance(configured, Mapping):
        raise TypeError("model configuration must be a mapping")
    result = dict(_plain(configured))
    candidates: dict[str, tuple[tuple[str, ...], ...]] = {
        "in_channels": (("model", "in_channels"), ("in_channels",)),
        "feature_size": (("model", "feature_size"), ("feature_size",)),
        "depths": (("model", "depths"), ("depths",)),
        "image_size": (
            ("model", "image_size"),
            ("image_size",),
            ("data", "image_size"),
            ("data", "img_size"),
            ("img_size",),
        ),
        "mlp_ratio": (("model", "mlp_ratio"), ("mlp_ratio",)),
        "norm_num_groups": (("model", "norm_num_groups"), ("norm_num_groups",)),
    }
    for key, paths in candidates.items():
        if key not in result:
            value = _first_value(cfg, paths, default=_MISSING)
            if value is not _MISSING:
                result[key] = _plain(value)
        if key in _MODEL_DEFAULTS:
            result.setdefault(key, _MODEL_DEFAULTS[key])
    return result


def _image_size(value: Any) -> tuple[int, int] | int:
    if isinstance(value, bool):
        raise ValueError("image_size must be a positive integer or two-value sequence")
    if isinstance(value, Integral):
        result = int(value)
        if result <= 0:
            raise ValueError("image_size must be a positive integer")
        return result
    if isinstance(value, (str, bytes, bytearray)):
        raise ValueError("image_size must be a positive integer or two-value sequence")
    try:
        values = tuple(value)
    except TypeError as exc:
        raise ValueError("image_size must be a positive integer or two-value sequence") from exc
    if len(values) != 2 or any(
        isinstance(item, bool) or not isinstance(item, Integral) or int(item) <= 0
        for item in values
    ):
        raise ValueError("image_size must be a positive integer or two-value sequence")
    return int(values[0]), int(values[1])


def _build_transforms(image_size: int | tuple[int, int], channels: int):
    if channels == 1:
        to_channels = T.Grayscale(num_output_channels=1)
    elif channels == 3:
        to_channels = T.Lambda(
            lambda image: image.convert("RGB") if image.mode != "RGB" else image
        )
    else:
        raise ValueError("ImageFolder denoising supports exactly 1 or 3 input channels")

    train_transform = T.Compose(
        [
            to_channels,
            T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(15),
            T.ToTensor(),
        ]
    )
    val_transform = T.Compose(
        [
            to_channels,
            T.Resize(
                tuple(int(size * 256 / 224) for size in image_size)
                if isinstance(image_size, tuple)
                else int(image_size * 256 / 224)
            ),
            T.CenterCrop(image_size),
            T.ToTensor(),
        ]
    )
    return train_transform, val_transform


def build_dataloaders(
    cfg: DictConfig | Mapping[str, Any], generator: torch.Generator
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    """Build ImageFolder denoising loaders at pipeline execution time."""
    model_cfg = _model_config(cfg)
    channels = int(model_cfg["in_channels"])
    image_size = _image_size(model_cfg["image_size"])
    root_value = _first_value(
        cfg,
        (
            ("paths", "image_root"),
            ("paths", "data_root"),
            ("data", "image_root"),
            ("data", "root"),
            ("data", "data_dir"),
            ("image_root",),
            ("data_dir",),
        ),
    )
    if root_value is None:
        raise ValueError("ImageFolder denoising requires a configured image root")
    root = Path(root_value)
    if not root.is_dir():
        raise FileNotFoundError(f"ImageFolder root does not exist: {root}")

    noise_std = float(
        _loader_value(cfg, "noise_std", _ROOT_SUPPORTED_DEFAULTS["noise_std"])
    )
    val_fraction = float(
        _loader_value(cfg, "val_fraction", _ROOT_SUPPORTED_DEFAULTS["val_fraction"])
    )
    if not math.isfinite(noise_std) or noise_std < 0:
        raise ValueError("noise_std must be a non-negative finite number")
    if not math.isfinite(val_fraction) or not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be strictly between 0 and 1")

    train_transform, val_transform = _build_transforms(image_size, channels)
    full_dataset = ImageFolder(root, transform=train_transform)
    max_cases = _max_cases(cfg)
    dataset_size = len(full_dataset) if max_cases is None else min(max_cases, len(full_dataset))
    if dataset_size < 2:
        raise ValueError("ImageFolder denoising requires at least two images")
    n_val = max(1, int(dataset_size * val_fraction))
    n_val = min(n_val, dataset_size - 1)
    indices = torch.randperm(len(full_dataset), generator=generator).tolist()[:dataset_size]
    train_indices = indices[n_val:]
    val_indices = indices[:n_val]
    train_dataset = DenoisingDataset(
        Subset(full_dataset, train_indices), noise_std=noise_std, channels=channels
    )
    val_dataset = DenoisingDataset(
        Subset(ImageFolder(root, transform=val_transform), val_indices),
        noise_std=noise_std,
        channels=channels,
    )

    batch_size = int(
        _loader_value(cfg, "batch_size", _ROOT_SUPPORTED_DEFAULTS["batch_size"])
    )
    num_workers = int(
        _loader_value(cfg, "num_workers", _ROOT_SUPPORTED_DEFAULTS["num_workers"])
    )
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers cannot be negative")
    pin_memory = bool(
        _loader_value(cfg, "pin_memory", _ROOT_SUPPORTED_DEFAULTS["pin_memory"])
    )
    persistent_workers = bool(
        _loader_value(
            cfg, "persistent_workers", _ROOT_SUPPORTED_DEFAULTS["persistent_workers"]
        )
    )
    drop_last = bool(
        _loader_value(cfg, "drop_last", _ROOT_SUPPORTED_DEFAULTS["drop_last"])
    )
    common: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "generator": generator,
    }
    if num_workers:
        common["worker_init_fn"] = seed_worker
        common["persistent_workers"] = persistent_workers
        prefetch_factor = _loader_value(cfg, "prefetch_factor", None)
        if prefetch_factor is not None:
            common["prefetch_factor"] = int(prefetch_factor)

    return (
        DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=drop_last,
            **common,
        ),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, **common),
    )


def _build_phases(cfg: DictConfig | Mapping[str, Any]) -> list[PhaseSpec]:
    raw_phases = _first_value(
        cfg,
        (
            ("experiment", "training", "phases"),
            ("phases",),
            ("training", "phases"),
            ("experiment", "phases"),
        ),
    )
    if raw_phases is None:
        learning_rate = float(
            _first_value(
                cfg,
                (
                    ("experiment", "training", "learning_rate"),
                    ("experiment", "training", "lr"),
                    ("learning_rate",),
                    ("lr",),
                    ("training", "learning_rate"),
                    ("training", "lr"),
                ),
                default=_ROOT_SUPPORTED_DEFAULTS["learning_rate"],
            )
        )
        raw_phases = [
            {
                "name": "train",
                "epochs": _first_value(
                    cfg,
                    (
                        ("epochs",),
                        ("num_epochs",),
                        ("experiment", "training", "epochs"),
                        ("experiment", "training", "num_epochs"),
                        ("training", "epochs"),
                        ("training", "num_epochs"),
                    ),
                    default=_ROOT_SUPPORTED_DEFAULTS["epochs"],
                ),
                "freeze_encoder": False,
                "encoder_lr": learning_rate,
                "decoder_lr": learning_rate,
            }
        ]
    if isinstance(raw_phases, Mapping):
        if "name" in raw_phases or "epochs" in raw_phases:
            raw_phases = [raw_phases]
        else:
            raw_phases = [
                {"name": name, **dict(spec)}
                for name, spec in raw_phases.items()
                if isinstance(spec, Mapping)
            ]
    if not isinstance(raw_phases, Sequence) or isinstance(raw_phases, (str, bytes)):
        raise TypeError("phases configuration must be a sequence of mappings")

    phases: list[PhaseSpec] = []
    for index, raw_phase in enumerate(raw_phases, start=1):
        phase = _mapping(raw_phase, f"phase {index}")
        decoder_lr = float(
            phase.get(
                "decoder_lr",
                phase.get("lr", _ROOT_SUPPORTED_DEFAULTS["learning_rate"]),
            )
        )
        phases.append(
            PhaseSpec(
                name=str(phase.get("name", f"phase{index}")),
                epochs=int(phase.get("epochs", 0)),
                freeze_encoder=bool(phase.get("freeze_encoder", False)),
                encoder_lr=float(phase.get("encoder_lr", decoder_lr)),
                decoder_lr=decoder_lr,
            )
        )
    return phases


def _effective_device(cfg: DictConfig | Mapping[str, Any]) -> torch.device:
    requested = _first_value(
        cfg,
        (
            ("device",),
            ("experiment", "training", "device"),
            ("training", "device"),
            ("run", "device"),
        ),
        default="auto",
    )
    return _resolve_device(requested)


def _build_checkpoints(
    cfg: DictConfig | Mapping[str, Any], output_dir: Path | None = None
) -> CheckpointManager | None:
    configured = _first_value(cfg, (("checkpoints",),))
    if configured is False:
        return None
    checkpoint_cfg = _mapping(configured, "checkpoints")
    if checkpoint_cfg and not bool(checkpoint_cfg.get("enabled", True)):
        return None
    root = _first_value(
        cfg,
        (
            ("checkpoints", "root"),
            ("checkpoints", "directory"),
            ("paths", "checkpoint_dir"),
            ("paths", "checkpoint_root"),
            ("experiment", "checkpoint_dir"),
            ("checkpoint_dir",),
        ),
    )
    if root is None and output_dir is not None:
        root = output_dir / "checkpoints"
    return None if root is None else CheckpointManager(Path(root))


def _output_dir(cfg: DictConfig | Mapping[str, Any]) -> Path:
    value = _first_value(
        cfg,
        (
            ("paths", "experiment_output"),
            ("paths", "output_dir"),
            ("experiment", "output_dir"),
            ("experiment", "output_root"),
            ("experiment_output",),
            ("output_dir",),
            ("paths", "output_root"),
            ("output_root",),
        ),
    )
    if value is None:
        raise ValueError("CNN denoising pretraining requires a configured experiment output path")
    return Path(value)


def _engine_config(
    cfg: DictConfig | Mapping[str, Any], model_cfg: Mapping[str, Any], device: Any
) -> dict[str, Any]:
    _reject_unsupported_engine_controls(cfg)
    result = _flatten_training_config(cfg)
    result["source_model_config"] = _plain(model_cfg)
    result["model_config"] = _plain(model_cfg)
    result.setdefault("architecture", model_cfg.get("architecture", "denoising_autoencoder"))
    result["device"] = str(device)
    scheduler = _first_configured(
        cfg,
        (
            ("experiment", "training", "scheduler"),
            ("scheduler",),
            ("training", "scheduler"),
            ("engine", "scheduler"),
        ),
    )
    if scheduler is _MISSING:
        min_lr = _first_value(
            cfg,
            (
                ("experiment", "training", "min_lr"),
                ("min_lr",),
                ("training", "min_lr"),
                ("engine", "min_lr"),
            ),
            default=_ROOT_SUPPORTED_DEFAULTS["min_lr"],
        )
        result["scheduler"] = {
            "name": "cosine",
            "interval": "update",
            "eta_min": _plain(min_lr),
        }
    else:
        result["scheduler"] = _plain(scheduler)
    configured_loss = _first_value(
        cfg,
        (
            ("experiment", "training", "loss_fn"),
            ("experiment", "training", "loss"),
            ("loss_fn",),
            ("loss",),
            ("training", "loss_fn"),
            ("training", "loss"),
        ),
    )
    result["loss"] = "mse" if configured_loss is None else _plain(configured_loss)
    result.setdefault("monitor", "mse")
    result.setdefault("maximize", False)
    result.setdefault("weight_decay", _ROOT_SUPPORTED_DEFAULTS["weight_decay"])
    max_grad_norm = _first_configured(
        cfg,
        (
            ("max_grad_norm",),
            ("grad_clip",),
            ("experiment", "training", "max_grad_norm"),
            ("experiment", "training", "grad_clip"),
            ("training", "max_grad_norm"),
            ("training", "grad_clip"),
            ("engine", "max_grad_norm"),
            ("engine", "grad_clip"),
        ),
    )
    result["max_grad_norm"] = (
        _ROOT_SUPPORTED_DEFAULTS["max_grad_norm"]
        if max_grad_norm is _MISSING
        else _plain(max_grad_norm)
    )
    result.setdefault("use_amp", _ROOT_SUPPORTED_DEFAULTS["use_amp"])
    return result


def _reject_unsupported_engine_controls(cfg: Any) -> None:
    locations: list[tuple[str, Mapping[str, Any]]] = []
    if isinstance(cfg, Mapping):
        locations.append(("root", cfg))
        for section_name in ("training", "engine"):
            section = cfg.get(section_name)
            if isinstance(section, Mapping):
                locations.append((section_name, section))
        experiment = cfg.get("experiment")
        if isinstance(experiment, Mapping):
            section = experiment.get("training")
            if isinstance(section, Mapping):
                locations.append(("experiment.training", section))

    for location, section in locations:
        for key in sorted(_UNSUPPORTED_ENGINE_CONTROLS):
            if key in section:
                raise ValueError(
                    f"unsupported {location} training control '{key}'; "
                    "shared engine does not implement it"
                )


def _build_loss(cfg: DictConfig | Mapping[str, Any]):
    configured = _first_value(
        cfg,
        (
            ("experiment", "training", "loss_fn"),
            ("experiment", "training", "loss"),
            ("loss_fn",),
            ("loss",),
            ("training", "loss_fn"),
            ("training", "loss"),
        ),
    )
    if configured is None:
        return mse_loss
    if callable(configured):
        return configured
    if isinstance(configured, str) and configured.lower().replace("-", "_") in {
        "mse",
        "mean_squared_error",
    }:
        return mse_loss
    raise ValueError(f"unsupported denoising loss '{configured}'; choose mse")


def _restore_best_checkpoint(
    model: nn.Module,
    checkpoints: CheckpointManager | None,
    *,
    expected_metadata: Mapping[str, Any] | None = None,
) -> Mapping[str, Any] | None:
    if checkpoints is None:
        return None
    path = checkpoints.root / "best.pt"
    if not path.is_file():
        return None
    return checkpoints.load_model(path, model, expected_metadata=expected_metadata)


def _export_encoder(
    model: nn.Module,
    output_dir: Path,
    model_cfg: Mapping[str, Any],
    result: FitResult,
    best_payload: Mapping[str, Any] | None,
) -> Path:
    """Export CPU-normalized encoder weights; execution device is not needed."""
    encoder = getattr(model, "encoder", None)
    if not isinstance(encoder, nn.Module):
        raise ValueError("denoising model must expose an nn.Module named 'encoder'")
    encoder = cast(nn.Module, encoder)
    epoch = result.best_epoch
    val_loss: Any = result.best_metric
    if best_payload is not None:
        if best_payload.get("epoch") is not None:
            epoch = int(best_payload["epoch"])
        checkpoint_metric = best_payload.get(
            "metric", best_payload.get("monitored_metric", best_payload.get("best_metric"))
        )
        if checkpoint_metric is not None:
            val_loss = float(checkpoint_metric)
    state_dict = {
        name: value.detach().cpu().clone() for name, value in encoder.state_dict().items()
    }
    payload = {
        "encoder_state_dict": state_dict,
        "source_model_config": _plain(model_cfg),
        "epoch": epoch,
        "val_loss": val_loss,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "encoder_best.pth"
    torch.save(payload, path)
    return path


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "on"}:
            return True
        if normalized in {"false", "no", "0", "off"}:
            return False
    return bool(value)


def _reconstruction_grid_options(
    cfg: DictConfig | Mapping[str, Any],
) -> tuple[bool, int]:
    visualization = _mapping(
        _first_value(cfg, (("visualization",),), default={}), "visualization"
    )
    configured = visualization.get(
        "reconstruction_grid",
        _first_value(cfg, (("reconstruction_grid",),), default=False),
    )
    if isinstance(configured, Mapping):
        enabled = _as_bool(configured.get("enabled", True), True)
        count = configured.get(
            "num_images",
            configured.get("n", visualization.get("num_images", 4)),
        )
    else:
        enabled = _as_bool(configured)
        count = visualization.get("num_images", 4)
    if not enabled:
        return False, 4
    try:
        count = int(count)
    except (TypeError, ValueError) as exc:
        raise ValueError("reconstruction grid num_images must be positive") from exc
    if count < 1:
        raise ValueError("reconstruction grid num_images must be positive")
    return True, count


@torch.no_grad()
def _save_reconstruction_grid(
    model: nn.Module,
    val_loader: Any,
    output_dir: Path,
    device: Any,
    *,
    n: int = 4,
    use_amp: bool = False,
) -> Path:
    """Save opt-in noisy, denoised, and clean validation examples."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    batch = next(iter(val_loader))
    if isinstance(batch, Mapping):
        noisy = batch.get("noisy", batch.get("input"))
        clean = batch.get("clean", batch.get("target"))
    elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
        noisy, clean = batch[0], batch[1]
    else:
        raise TypeError("reconstruction grid loader must yield (noisy, clean) pairs")
    if noisy is None or clean is None:
        raise KeyError("reconstruction grid batch must contain noisy/input and clean/target")

    noisy = noisy if isinstance(noisy, torch.Tensor) else torch.as_tensor(noisy)
    clean = clean if isinstance(clean, torch.Tensor) else torch.as_tensor(clean)
    if noisy.ndim != 4 or clean.ndim != 4:
        raise ValueError("reconstruction grid images must have shape [B, C, H, W]")
    count = min(int(n), int(noisy.shape[0]))
    if count < 1:
        raise ValueError("reconstruction grid loader yielded no images")

    try:
        runtime_device = next(model.parameters()).device
    except StopIteration:
        runtime_device = torch.device(device)
    was_training = model.training
    model.eval()
    figure = None
    try:
        noisy = noisy[:count].to(runtime_device)
        clean = clean[:count].to(runtime_device)
        with torch.amp.autocast(
            device_type=runtime_device.type,
            enabled=_as_bool(use_amp),
        ):
            prediction = model(noisy)

        figure, axes = plt.subplots(count, 3, figsize=(12, 4 * count), squeeze=False)
        for index in range(count):
            images = (noisy[index], prediction[index], clean[index])
            for column, (image, title) in enumerate(
                zip(images, ("Noisy", "Denoised", "Clean"))
            ):
                image = image.detach().cpu().float().clamp(0.0, 1.0)
                rendered = (
                    image[0].numpy()
                    if image.shape[0] == 1
                    else image.permute(1, 2, 0).numpy()
                )
                axes[index, column].imshow(rendered)
                axes[index, column].axis("off")
                if index == 0:
                    axes[index, column].set_title(title)
        figure.tight_layout()
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "cnn_reconstruction_grid.png"
        figure.savefig(path, dpi=120)
    finally:
        if figure is not None:
            plt.close(figure)
        if was_training:
            model.train()
    return path


def run_cnn_denoising_pretrain(cfg: DictConfig) -> FitResult:
    """Run configured 2-D denoising pretraining through the shared engine."""
    model_cfg = _model_config(cfg)
    output_dir = _output_dir(cfg)
    checkpoints = _build_checkpoints(cfg, output_dir)
    if checkpoints is None:
        raise ValueError(
            "CNN denoising pretraining requires enabled checkpointing to produce "
            "encoder_best.pth"
        )
    device = _effective_device(cfg)
    seed = int(
        _first_value(
            cfg,
            (("seed",), ("reproducibility", "seed"), ("run", "seed")),
            default=_ROOT_SUPPORTED_DEFAULTS["seed"],
        )
    )
    deterministic = bool(
        _first_value(
            cfg,
            (("deterministic",), ("reproducibility", "deterministic")),
            default=True,
        )
    )
    generator = seed_everything(seed, deterministic=deterministic)

    model = build_denoising_model(model_cfg)
    train_loader, val_loader = build_dataloaders(cfg, generator)
    phases = _build_phases(cfg)
    loss_fn = _build_loss(cfg)
    run_config = _engine_config(cfg, model_cfg, device)
    resume = resume_path(cfg, checkpoints)
    warm_start = warm_start_path(cfg, checkpoints)
    if resume is not None and warm_start is not None:
        raise ValueError("resume and warm_start are mutually exclusive")
    if resume is not None:
        run_config["source_checkpoint"] = str(resume)
        run_config["resume_mode"] = "exact"
    elif warm_start is not None:
        run_config["source_checkpoint"] = str(warm_start)
        run_config["resume_mode"] = "warm_start"
    if resume is not None:
        _copy_resume_best(checkpoints, resume)
    tracking_config = _plain(_first_value(cfg, (("tracking",),), default={}))
    tracker = create_tracker(_mapping(tracking_config, "tracking"), run_config)
    result = _invoke_fit(
        fit,
        (
            model,
            train_loader,
            val_loader,
            loss_fn,
            evaluate_denoising,
            phases,
            run_config,
            tracker,
            checkpoints,
        ),
        resume=resume,
        warm_start=warm_start,
        loader_generator=generator,
    )
    if not isinstance(result, FitResult):
        raise TypeError("shared training engine must return FitResult")
    best_payload = _restore_best_checkpoint(
        model,
        checkpoints,
        expected_metadata=_engine_checkpoint_metadata(run_config, phases),
    )
    if best_payload is None:
        best_path = checkpoints.root / "best.pt"
        raise FileNotFoundError(
            f"best checkpoint '{best_path}' is absent; cannot export encoder_best.pth"
        )
    if resume is not None or warm_start is not None:
        result_metadata = dict(result.metadata or {})
        result_metadata.update(
            {
                "source_checkpoint": str(resume or warm_start),
                "resume_mode": "exact" if resume is not None else "warm_start",
                "architecture": run_config.get("architecture"),
                "model_config": run_config.get("model_config"),
                "execution_device": run_config.get("device"),
            }
        )
        result = FitResult(
            result.best_metric,
            result.best_epoch,
            result.history,
            result.test_metrics,
            result_metadata,
        )
    _export_encoder(model, output_dir, model_cfg, result, best_payload)
    grid_enabled, grid_count = _reconstruction_grid_options(cfg)
    if grid_enabled:
        _save_reconstruction_grid(
            model,
            val_loader,
            output_dir,
            device,
            n=grid_count,
            use_amp=run_config["use_amp"],
        )
    return result


__all__ = ["build_dataloaders", "run_cnn_denoising_pretrain"]
