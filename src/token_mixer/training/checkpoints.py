from __future__ import annotations

import os
import random
import shutil
import tempfile
from pathlib import Path
from pathlib import PureWindowsPath
from typing import Any, Mapping

import numpy as np
import torch

from token_mixer import __version__


_CHECKPOINT_SCHEMA_VERSION = 2
_MISSING = object()


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    python_state = state.get("python", state.get("random"))
    if python_state is not None:
        random.setstate(python_state)

    numpy_state = state.get("numpy")
    if numpy_state is not None:
        np.random.set_state(numpy_state)

    torch_state = state.get("torch")
    if torch_state is not None:
        torch.set_rng_state(torch_state)

    cuda_state = state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def _component_state(component: Any) -> dict[str, Any] | None:
    return None if component is None else component.state_dict()


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _first_present(payload: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return value
    return None


def _comparison_value(value: Any) -> Any:
    """Convert checkpoint metadata to a stable, dependency-free comparison value."""
    if isinstance(value, Mapping):
        return {str(key): _comparison_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_comparison_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_comparison_value(item) for item in value), key=repr)
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return _comparison_value(value.item() if value.numel() == 1 else value.tolist())
    if isinstance(value, np.ndarray):
        return _comparison_value(value.item() if value.ndim == 0 else value.tolist())
    if isinstance(value, np.generic):
        return _comparison_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if callable(value):
        return f"{getattr(value, '__module__', '')}.{getattr(value, '__qualname__', repr(value))}"
    return value


def _payload_schema_version(payload: Mapping[str, Any]) -> int | None:
    value = payload.get("schema_version")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("checkpoint schema_version must be a positive integer")
    if value > _CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"checkpoint schema_version {value} is newer than supported version "
            f"{_CHECKPOINT_SCHEMA_VERSION}"
        )
    return value


def _payload_metadata(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata: dict[str, Any] = {}
    for key in ("checkpoint_metadata", "metadata"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            metadata.update(value)
    state = payload.get("state")
    if isinstance(state, Mapping):
        for key in ("checkpoint_metadata", "metadata"):
            value = state.get(key)
            if isinstance(value, Mapping):
                metadata.update(value)

    # Legacy checkpoints kept compatibility information only in ``config``.
    # Pull its top-level fields without making missing fields mandatory.
    config_sources = [payload.get("config")]
    if isinstance(state, Mapping):
        config_sources.append(state.get("config"))
    for config in config_sources:
        if isinstance(config, Mapping):
            for key, value in config.items():
                metadata.setdefault(str(key), value)
            model_config = config.get("model_config")
            if model_config is None:
                model_config = config.get("model")
            if model_config is not None:
                metadata.setdefault("model_config", model_config)
    return metadata


def _phase_plan_matches(actual: Any, expected: Any) -> bool:
    if not isinstance(actual, (list, tuple)) or not isinstance(expected, (list, tuple)):
        return _comparison_value(actual) == _comparison_value(expected)
    if len(actual) > len(expected):
        return False
    for saved_phase, configured_phase in zip(actual, expected):
        if not isinstance(saved_phase, Mapping) or not isinstance(configured_phase, Mapping):
            if _comparison_value(saved_phase) != _comparison_value(configured_phase):
                return False
            continue
        for key, saved_value in saved_phase.items():
            if key not in configured_phase:
                continue
            configured_value = configured_phase[key]
            if key == "epochs":
                try:
                    if int(configured_value) < int(saved_value):
                        return False
                except (TypeError, ValueError):
                    return False
            elif _comparison_value(saved_value) != _comparison_value(configured_value):
                return False
    return True


def _validate_metadata(
    payload: Mapping[str, Any], expected_metadata: Mapping[str, Any] | None
) -> None:
    if expected_metadata is None:
        return
    saved_metadata = _payload_metadata(payload)
    state = payload.get("state")
    mismatches: list[str] = []
    for key, expected_value in expected_metadata.items():
        if key == "checkpoint_metadata":
            continue
        actual_value: Any = saved_metadata.get(key, _MISSING)
        if actual_value is _MISSING:
            actual_value = payload.get(key, _MISSING)
            if actual_value is None:
                actual_value = _MISSING
        if actual_value is _MISSING and isinstance(state, Mapping):
            actual_value = state.get(key, _MISSING)
        if actual_value is _MISSING:
            # Older checkpoints do not carry all metadata. They remain loadable,
            # but any metadata they do carry is checked strictly.
            continue
        matches = (
            _phase_plan_matches(actual_value, expected_value)
            if key == "phase_plan"
            else _comparison_value(actual_value) == _comparison_value(expected_value)
        )
        if not matches:
            mismatches.append(
                f"{key} (checkpoint={_comparison_value(actual_value)!r}, "
                f"configured={_comparison_value(expected_value)!r})"
            )
    if mismatches:
        raise ValueError("checkpoint metadata is incompatible: " + "; ".join(mismatches))


def _loader_generator_state(payload: Mapping[str, Any]) -> Any:
    state = _first_present(payload, "loader_generator_state", "loader_generator")
    if state is not None:
        return state

    nested_state = payload.get("state")
    if isinstance(nested_state, Mapping):
        return _first_present(nested_state, "loader_generator_state", "loader_generator")
    return None


class CheckpointManager:
    """Save and restore pipeline-owned training checkpoints."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def _path_for_tag(self, tag: str | Path) -> Path:
        try:
            raw_tag = os.fspath(tag)
        except TypeError as exc:
            raise ValueError("checkpoint tag must be a simple filename") from exc
        if isinstance(raw_tag, bytes):
            raw_tag = os.fsdecode(raw_tag)

        windows_tag = PureWindowsPath(raw_tag)
        if (
            not raw_tag.strip()
            or raw_tag in {".", ".."}
            or "/" in raw_tag
            or "\\" in raw_tag
            or Path(raw_tag).is_absolute()
            or Path(raw_tag).anchor
            or windows_tag.drive
            or windows_tag.root
        ):
            raise ValueError("checkpoint tag must be a simple filename")

        filename = raw_tag if raw_tag.endswith(".pt") else f"{raw_tag}.pt"
        if filename.startswith("."):
            raise ValueError("checkpoint tag must be a non-hidden filename")

        root = self.root.resolve()
        path = (self.root / filename).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("checkpoint tag escapes checkpoint root") from exc
        return self.root / filename

    def save(
        self,
        tag: str | Path,
        model: torch.nn.Module,
        optimizer: Any,
        scheduler: Any,
        scaler: Any,
        state: Mapping[str, Any],
    ) -> Path:
        """Persist model state, training state, provenance, and RNG state."""
        if not isinstance(state, Mapping):
            raise TypeError("checkpoint state must be a mapping")

        metadata = dict(state)
        loader_generator = metadata.get("loader_generator")
        loader_generator_state = None
        if loader_generator is not None:
            if not isinstance(loader_generator, torch.Generator):
                raise TypeError("checkpoint state 'loader_generator' must be a torch.Generator")
            loader_generator_state = loader_generator.get_state()
            metadata["loader_generator"] = loader_generator_state
        else:
            loader_generator_state = metadata.get("loader_generator_state")

        metric = metadata.get("monitored_metric", metadata.get("metric"))
        config = metadata.get("config", metadata.get("composed_config"))
        composed_config = metadata.get("composed_config", config)
        checkpoint_metadata = metadata.get("checkpoint_metadata", {})
        if not isinstance(checkpoint_metadata, Mapping):
            raise TypeError("checkpoint state 'checkpoint_metadata' must be a mapping")
        checkpoint_metadata = dict(checkpoint_metadata)
        metadata["checkpoint_metadata"] = checkpoint_metadata

        payload: dict[str, Any] = {
            **metadata,
            "schema_version": _CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_type": "training",
            "state": metadata,
            "checkpoint_metadata": checkpoint_metadata,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": _component_state(optimizer),
            "scheduler_state_dict": _component_state(scheduler),
            "scaler_state_dict": _component_state(scaler),
            "rng_state": _capture_rng_state(),
            "loader_generator_state": loader_generator_state,
            "config": config,
            "composed_config": composed_config,
            "manifest_hash": metadata.get("manifest_hash"),
            "code_version": metadata.get("code_version", __version__),
            "epoch": metadata.get("epoch"),
            "global_step": metadata.get("global_step"),
            "monitored_metric": metric,
            "metric": metadata.get("metric", metric),
        }
        for key, value in checkpoint_metadata.items():
            payload.setdefault(key, value)

        path = self._path_for_tag(tag)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary_path = Path(temporary_name)
        try:
            os.close(descriptor)
            descriptor = -1
            torch.save(payload, temporary_path)
            os.replace(temporary_path, path)
        except BaseException:
            if descriptor != -1:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
            raise
        return path

    def read(self, path: Path) -> dict[str, Any]:
        """Read and validate checkpoint container without mutating runtime state."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")
        payload = _torch_load(path)
        if not isinstance(payload, Mapping):
            raise ValueError(f"Checkpoint '{path}' must contain a mapping")
        _payload_schema_version(payload)
        return dict(payload)

    def validate(
        self,
        payload: Mapping[str, Any],
        expected_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(payload, Mapping):
            raise ValueError("checkpoint payload must be a mapping")
        _payload_schema_version(payload)
        _validate_metadata(payload, expected_metadata)

    @staticmethod
    def _best_source(path: Path) -> Path:
        path = Path(path)
        return path if path.name == "best.pt" else path.parent / "best.pt"

    def copy_best_from(self, checkpoint: Path) -> Path | None:
        """Copy source run's best checkpoint into this run without changing source."""
        source = self._best_source(Path(checkpoint))
        if not source.is_file():
            return None
        destination = self.root / "best.pt"
        if source.resolve() == destination.resolve():
            return destination

        self.root.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        temporary_path = Path(temporary_name)
        try:
            os.close(descriptor)
            descriptor = -1
            shutil.copy2(source, temporary_path)
            os.replace(temporary_path, destination)
        finally:
            if descriptor != -1:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            if temporary_path.exists():
                temporary_path.unlink()
        return destination

    def load_model(
        self,
        path: Path,
        model: torch.nn.Module,
        *,
        expected_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Load only model weights; never restore optimizer, loader, or RNG state."""
        payload = self.read(path)
        self.validate(payload, expected_metadata)
        model_state = _first_present(payload, "model_state_dict", "model_state", "model")
        if model_state is None:
            raise ValueError(f"Checkpoint '{path}' has no model state")
        model.load_state_dict(model_state)
        return payload

    def load(
        self,
        path: Path,
        model: torch.nn.Module,
        optimizer: Any = None,
        scheduler: Any = None,
        scaler: Any = None,
        generator: torch.Generator | None = None,
        *,
        expected_metadata: Mapping[str, Any] | None = None,
        restore_rng: bool = True,
    ) -> dict[str, Any]:
        """Load checkpoint state into supplied objects and restore RNG state."""
        path = Path(path)
        payload = self.read(path)
        self.validate(payload, expected_metadata)

        model_state = _first_present(payload, "model_state_dict", "model_state", "model")
        if model_state is None:
            raise ValueError(f"Checkpoint '{path}' has no model state")

        component_states: list[tuple[str, Any, Any]] = []
        if optimizer is not None:
            optimizer_state = _first_present(
                payload, "optimizer_state_dict", "optimizer_state", "optimizer"
            )
            if optimizer_state is None:
                raise ValueError(
                    f"Checkpoint '{path}' has no optimizer state, but an optimizer was supplied"
                )
            component_states.append(("optimizer", optimizer, optimizer_state))
        if scheduler is not None:
            scheduler_state = _first_present(
                payload, "scheduler_state_dict", "scheduler_state", "scheduler"
            )
            if scheduler_state is None:
                raise ValueError(
                    f"Checkpoint '{path}' has no scheduler state, but a scheduler was supplied"
                )
            component_states.append(("scheduler", scheduler, scheduler_state))
        if scaler is not None:
            scaler_state = _first_present(payload, "scaler_state_dict", "scaler_state", "scaler")
            if scaler_state is None:
                raise ValueError(
                    f"Checkpoint '{path}' has no scaler state, but a scaler was supplied"
                )

            component_states.append(("scaler", scaler, scaler_state))

        generator_state = _loader_generator_state(payload)
        if generator is not None:
            if generator_state is None:
                if _payload_schema_version(payload) is not None:
                    raise ValueError(
                        f"Checkpoint '{path}' has no loader generator state, but a generator was supplied"
                    )
            if not isinstance(generator_state, torch.Tensor):
                if generator_state is not None:
                    raise ValueError(f"Checkpoint '{path}' has invalid loader generator state")

        model.load_state_dict(model_state)

        for _name, component, component_state in component_states:
            component.load_state_dict(component_state)

        if generator is not None and generator_state is not None:
            generator.set_state(generator_state)

        rng_state = payload.get("rng_state")
        if restore_rng and isinstance(rng_state, Mapping):
            _restore_rng_state(rng_state)

        return dict(payload)
