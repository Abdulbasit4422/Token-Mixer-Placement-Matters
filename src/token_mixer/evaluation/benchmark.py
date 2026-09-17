"""Pure benchmark result and fixed-input protocol helpers.

This module owns measurement composition and local serialization only.  Model
construction, checkpoint restoration, case selection, and tracking remain in
``token_mixer.pipelines.benchmark``.  The counters and timing primitives are
implemented in :mod:`token_mixer.evaluation.efficiency`; keeping this module
free of W&B calls makes model-level protocol tests deterministic and reusable.
"""

from __future__ import annotations

import json
import inspect
import math
import os
import platform
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from token_mixer.evaluation.efficiency import (
    NvmlPowerSampler,
    checkpoint_size_bytes,
    measure_batch_sweep,
    measure_forward,
    model_parameter_counts,
    static_model_cost,
)
from token_mixer.privacy import hash_case_id, redact_case_identifiers


BENCHMARK_SCHEMA_VERSION = 1
DEFAULT_POWER_DEVICE_INDEX = 0
DEFAULT_POWER_INTERVAL_SECONDS = 0.01
_MAX_POWER_REASON_LENGTH = 240
_WINDOWS_RESERVED_DEVICE_STEMS = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}
_WINDOWS_INVALID_FILENAME_CHARACTERS = set('<>:"/\\|?*')


def _copy_value(value: Any) -> Any:
    """Copy ordinary benchmark containers without copying model/tensor state."""

    if isinstance(value, Mapping):
        return {key: _copy_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_value(item) for item in value)
    return value


@dataclass(frozen=True)
class BenchmarkResult:
    """Immutable top-level result contract for one benchmark run."""

    summary: dict[str, Any]
    rows: list[dict[str, Any]]
    provenance: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.summary, Mapping):
            raise TypeError("benchmark summary must be a mapping")
        if not isinstance(self.rows, Sequence) or isinstance(
            self.rows, (str, bytes, bytearray)
        ):
            raise TypeError("benchmark rows must be a sequence of mappings")
        if not isinstance(self.provenance, Mapping):
            raise TypeError("benchmark provenance must be a mapping")
        normalized_rows: list[dict[str, Any]] = []
        for row in self.rows:
            if not isinstance(row, Mapping):
                raise TypeError("each benchmark row must be a mapping")
            normalized_rows.append(_copy_value(dict(row)))
        object.__setattr__(self, "summary", _copy_value(dict(self.summary)))
        object.__setattr__(self, "rows", normalized_rows)
        object.__setattr__(self, "provenance", _copy_value(dict(self.provenance)))


def _json_safe(value: Any) -> Any:
    """Convert tensors, NumPy values, paths, and non-finite numbers to JSON."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return _json_safe(value.item() if value.ndim == 0 else value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        values = list(value)
        if isinstance(value, (set, frozenset)):
            values.sort(key=repr)
        return [_json_safe(item) for item in values]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            listed = tolist()
        except (TypeError, ValueError, RuntimeError):
            listed = None
        if listed is not None and listed is not value:
            return _json_safe(listed)
    item = getattr(value, "item", None)
    if callable(item):
        try:
            scalar = item()
        except (TypeError, ValueError, RuntimeError):
            scalar = None
        if scalar is not None and scalar is not value:
            return _json_safe(scalar)
    return str(value)


def _redact_case_ids(value: Any) -> Any:
    """Recursively replace raw case fields with stable hashes."""

    return redact_case_identifiers(value)


def _write_atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if descriptor != -1:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if temporary_path.exists():
            temporary_path.unlink()


def _validated_output_name(output_name: str | os.PathLike[str] | None) -> str:
    if output_name is None:
        return "benchmark.json"
    try:
        name = os.fspath(output_name)
    except TypeError as exc:
        raise ValueError(
            "benchmark output_name must be a JSON filename inside output_dir"
        ) from exc
    if not isinstance(name, str):
        raise ValueError("benchmark output_name must be a JSON filename inside output_dir")
    reserved_stem = name.split(".", 1)[0].rstrip(" .").upper()
    if (
        not name
        or name != name.strip()
        or "/" in name
        or "\\" in name
        or any(
            ord(character) < 32
            or ord(character) == 127
            or character in _WINDOWS_INVALID_FILENAME_CHARACTERS
            for character in name
        )
        or name[-1] in " ."
        or reserved_stem in _WINDOWS_RESERVED_DEVICE_STEMS
        or Path(name).is_absolute()
        or not name.lower().endswith(".json")
    ):
        raise ValueError("benchmark output_name must be a JSON filename inside output_dir")
    return name


def serialize_benchmark(
    output_dir: Path,
    result: BenchmarkResult,
    *,
    output_name: str | os.PathLike[str] | None = None,
) -> Path:
    """Write one atomic, JSON-safe benchmark file and return its path."""

    if not isinstance(result, BenchmarkResult):
        raise TypeError("result must be a BenchmarkResult")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "summary": _json_safe(_redact_case_ids(result.summary)),
        "rows": _json_safe(_redact_case_ids(result.rows)),
        "provenance": _json_safe(_redact_case_ids(result.provenance)),
    }
    path = output_dir / _validated_output_name(output_name)
    _write_atomic_json(path, payload)
    return path


def _first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, Mapping):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _input_batch_size(value: Any) -> int:
    tensor = _first_tensor(value)
    if tensor is None or tensor.ndim == 0:
        return 1
    if int(tensor.shape[0]) < 1:
        raise ValueError("benchmark inputs must have a positive batch dimension")
    return int(tensor.shape[0])


def _selected_cuda_device(
    model: torch.nn.Module, inputs: Any
) -> torch.device | None:
    tensors = [_first_tensor(inputs)]
    try:
        tensors.append(next(model.parameters()))
    except StopIteration:
        pass
    for tensor in tensors:
        if tensor is None or tensor.device.type != "cuda":
            continue
        return tensor.device
    return None


def _execution_device(model: torch.nn.Module, inputs: Any) -> torch.device:
    tensor = _first_tensor(inputs)
    if tensor is not None:
        return tensor.device
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _precision_name(model: torch.nn.Module, inputs: Any) -> str:
    try:
        dtype = next(model.parameters()).dtype
    except StopIteration:
        tensor = _first_tensor(inputs)
        dtype = torch.float32 if tensor is None else getattr(tensor, "dtype", torch.float32)
    return str(dtype).removeprefix("torch.")


def _runtime_metadata(model: torch.nn.Module, inputs: Any) -> dict[str, Any]:
    device = _execution_device(model, inputs)
    hardware: dict[str, Any] = {
        "device": str(device),
        "device_type": device.type,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if device.type == "cuda":
        try:
            hardware["device_index"] = (
                int(device.index)
                if device.index is not None
                else int(torch.cuda.current_device())
            )
        except (RuntimeError, AssertionError, ValueError):
            hardware["device_index"] = None
        try:
            hardware["device_name"] = torch.cuda.get_device_name(device)
        except (RuntimeError, AssertionError, ValueError):
            hardware["device_name"] = None
    return {
        "precision": _precision_name(model, inputs),
        "timing_boundary": "model_forward_only",
        "hardware": hardware,
        "software": {
            "pytorch": str(torch.__version__),
            "python": platform.python_version(),
        },
    }


def _default_power_device_index(model: torch.nn.Module, inputs: Any) -> int:
    """Resolve NVML's default index from the selected CUDA execution device."""

    selected = _selected_cuda_device(model, inputs)
    if selected is not None:
        if selected.index is not None:
            return int(selected.index)
        return int(torch.cuda.current_device())
    return DEFAULT_POWER_DEVICE_INDEX


def _cuda_visible_devices() -> list[str] | None:
    configured = os.environ.get("CUDA_VISIBLE_DEVICES")
    if configured is None:
        return None
    return [item.strip() for item in configured.split(",")]


def _power_device_resolution(
    model: torch.nn.Module,
    inputs: Any,
    configured_index: int | None,
) -> dict[str, Any]:
    selected = _selected_cuda_device(model, inputs)
    logical_index = None
    if selected is not None:
        logical_index = (
            int(selected.index)
            if selected.index is not None
            else int(torch.cuda.current_device())
        )
    visible_devices = _cuda_visible_devices()

    if configured_index is not None:
        return {
            "physical_index": int(configured_index),
            "logical_index": logical_index,
            "status": "explicit_override",
            "reason": None,
            "visible_devices": visible_devices,
        }

    if logical_index is None:
        return {
            "physical_index": DEFAULT_POWER_DEVICE_INDEX,
            "logical_index": None,
            "status": "not_cuda",
            "reason": None,
            "visible_devices": visible_devices,
        }

    if visible_devices is None:
        return {
            "physical_index": logical_index,
            "logical_index": logical_index,
            "status": "identity",
            "reason": None,
            "visible_devices": None,
        }

    if not visible_devices or any(
        not item.isdigit() or int(item) < 0 for item in visible_devices
    ):
        reason = (
            "CUDA_VISIBLE_DEVICES contains UUID or unresolved entries; "
            "NVML physical device index cannot be resolved safely"
        )
        return {
            "physical_index": None,
            "logical_index": logical_index,
            "status": "unavailable",
            "reason": reason,
            "visible_devices": visible_devices,
        }

    if logical_index >= len(visible_devices):
        reason = (
            f"logical CUDA device {logical_index} is not present in "
            "CUDA_VISIBLE_DEVICES"
        )
        return {
            "physical_index": None,
            "logical_index": logical_index,
            "status": "unavailable",
            "reason": reason,
            "visible_devices": visible_devices,
        }

    return {
        "physical_index": int(visible_devices[logical_index]),
        "logical_index": logical_index,
        "status": "resolved",
        "reason": None,
        "visible_devices": visible_devices,
    }


def _repeat_input(value: Any, batch_size: int, source_batch_size: int) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value
        if source_batch_size == batch_size:
            return value
        if int(value.shape[0]) < 1:
            raise ValueError("benchmark inputs must have a positive batch dimension")
        sample = value[:1]
        repeats = (batch_size, *([1] * (value.ndim - 1)))
        return sample.repeat(*repeats)
    if isinstance(value, Mapping):
        return {
            key: _repeat_input(item, batch_size, source_batch_size)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_repeat_input(item, batch_size, source_batch_size) for item in value)
    if isinstance(value, list):
        return [_repeat_input(item, batch_size, source_batch_size) for item in value]
    return value


def _input_factory(inputs: Any):
    source_batch_size = _input_batch_size(inputs)

    def factory(batch_size: int) -> Any:
        return _repeat_input(inputs, int(batch_size), source_batch_size)

    return factory


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _throughput(latency_ms: Any, batch_size: Any, input_shape: Any) -> tuple[float | None, float | None]:
    latency = _number(latency_ms)
    try:
        batch = int(batch_size)
    except (TypeError, ValueError, OverflowError):
        batch = 0
    if latency is None or latency <= 0 or batch <= 0:
        return None, None
    samples_per_second = batch * 1000.0 / latency
    try:
        shape = tuple(int(size) for size in input_shape)
        # Model inputs are channel-first: exclude batch and channel axes when
        # reporting voxel throughput so multi-channel volumes are not counted
        # multiple times.
        voxels_per_sample = int(math.prod(shape[2:])) if len(shape) > 2 else 1
    except (TypeError, ValueError, OverflowError):
        voxels_per_sample = 1
    return samples_per_second, samples_per_second * voxels_per_sample


def _stat_value(stats: Any, name: str, default: Any = None) -> Any:
    if stats is None:
        return default
    if isinstance(stats, Mapping):
        return stats.get(name, default)
    return getattr(stats, name, default)


def _bounded_power_reason(error: object) -> str:
    reason = " ".join(str(error).split())
    if not reason:
        reason = (
            error.__class__.__name__
            if isinstance(error, BaseException)
            else "unknown error"
        )
    return reason[:_MAX_POWER_REASON_LENGTH]


def _power_fields(stats: Any) -> dict[str, Any]:
    interval_seconds = _stat_value(stats, "interval_seconds")
    if interval_seconds is None:
        interval_ms = _stat_value(stats, "sample_interval_ms")
    else:
        interval_ms = _number(interval_seconds)
        interval_ms = None if interval_ms is None else interval_ms * 1000.0
    return {
        "power/average_watts": _stat_value(stats, "average_watts"),
        "power/max_watts": _stat_value(stats, "max_watts"),
        "power/energy_joules": _stat_value(
            stats, "joules", _stat_value(stats, "energy_joules")
        ),
        "power/sample_count": _stat_value(stats, "samples", _stat_value(stats, "sample_count", 0)),
        "power/sample_interval_ms": interval_ms,
        "power/status": _stat_value(stats, "status", "unavailable"),
        "power/reason": _stat_value(stats, "reason"),
        "power/device_index": _stat_value(stats, "device_index"),
        "power/nvml_version": _stat_value(stats, "nvml_version"),
    }


def _start_power_sampler(
    device_index: int, interval_seconds: float
) -> tuple[Any | None, Any | None]:
    try:
        sampler = NvmlPowerSampler(device_index=device_index, interval_seconds=interval_seconds)
        started = sampler.start()
        return sampler, started
    except BaseException as error:
        # Power is optional telemetry.  Counter/timing failures still retain
        # their original status and are never replaced by NVML setup errors.
        return None, error


def _stop_power_sampler(
    sampler: Any | None,
    start_error: Any | None,
    *,
    device_index: int | None,
    interval_seconds: float = DEFAULT_POWER_INTERVAL_SECONDS,
    disabled: bool = False,
) -> Any:
    if sampler is None:
        return {
            "average_watts": None,
            "max_watts": None,
            "joules": None,
            "samples": 0,
            "interval_seconds": interval_seconds,
            "status": "disabled" if disabled else "unavailable",
            "reason": (
                "power measurement disabled"
                if disabled
                else None if start_error is None else _bounded_power_reason(start_error)
            ),
            "device_index": device_index,
            "nvml_version": None,
        }
    try:
        return sampler.stop()
    except BaseException as error:
        return {
            "average_watts": None,
            "max_watts": None,
            "joules": None,
            "samples": 0,
            "interval_seconds": interval_seconds,
            "status": "unavailable",
            "reason": _bounded_power_reason(error),
            "device_index": getattr(sampler, "device_index", device_index),
            "nvml_version": getattr(
                sampler,
                "nvml_version",
                getattr(sampler, "_nvml_version", None),
            ),
        }


def _prefixed(prefix: str, values: Mapping[str, Any]) -> dict[str, Any]:
    return {f"{prefix}/{key}": value for key, value in values.items()}


def _accepts_keyword(function: Any, name: str) -> bool:
    try:
        parameters = inspect.signature(function).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters) or any(
        parameter.name == name
        and parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        for parameter in parameters
    )


def _disabled_static_model_cost(
    model: torch.nn.Module,
    checkpoint_path: str | Path | None,
    *,
    mac_tool: str,
    flop_tool: str,
) -> dict[str, Any]:
    """Return explicit non-measurement fields when profiling is disabled."""

    return {
        **model_parameter_counts(model),
        "checkpoint_bytes": checkpoint_size_bytes(checkpoint_path),
        "macs": None,
        "flops": None,
        "mac_tool": mac_tool,
        "mac_tool_version": None,
        "flop_tool": flop_tool,
        "flop_tool_version": None,
        "mac_convention": None,
        "flop_convention": None,
        "mac_status": "disabled",
        "flop_status": "disabled",
        "mac_error": None,
        "flop_error": None,
        "unsupported_ops": {},
        "uncalled_modules": [],
        "status": "disabled",
    }


def _static_model_cost(
    model: torch.nn.Module,
    inputs: Any,
    *,
    checkpoint_path: str | Path | None,
    profiler_enabled: bool,
    mac_tool: str,
    flop_tool: str,
) -> dict[str, Any]:
    if not profiler_enabled:
        return _disabled_static_model_cost(
            model,
            checkpoint_path,
            mac_tool=mac_tool,
            flop_tool=flop_tool,
        )

    kwargs: dict[str, Any] = {
        "inputs": inputs,
    }
    if checkpoint_path is not None:
        kwargs["checkpoint_path"] = checkpoint_path
    for name, value, default in (
        ("mac_tool", mac_tool, "thop"),
        ("flop_tool", flop_tool, "fvcore"),
    ):
        if _accepts_keyword(static_model_cost, name):
            kwargs[name] = value
        elif value != default:
            raise ValueError(
                f"configured {name}={value!r} is unsupported by the active profiler"
            )
    result = dict(static_model_cost(model, **kwargs))
    # Keep configured tool identity explicit even when the shared helper uses
    # its built-in default signature.
    result["mac_tool"] = mac_tool
    result["flop_tool"] = flop_tool
    return result


def run_model_protocol(
    model: torch.nn.Module,
    inputs: Any,
    *,
    protocol: str,
    warmup_iterations: int,
    repetitions: int,
    batch_sizes: Sequence[int],
    checkpoint_path: str | Path | None = None,
    power_device_index: int | None = None,
    power_interval_seconds: float = DEFAULT_POWER_INTERVAL_SECONDS,
    power_enabled: bool = True,
    profiler_enabled: bool = True,
    mac_tool: str = "thop",
    flop_tool: str = "fvcore",
) -> dict[str, Any]:
    """Run static, repeated, and batch-sweep measurements on fixed inputs.

    ``inputs`` is passed unchanged to static and repeated measurements.  Sweep
    inputs are deterministic replicas of its first sample and are allocated by
    the factory before each measured batch size, never inside a timed loop.
    """

    if not isinstance(protocol, str) or not protocol.strip():
        raise ValueError("protocol must be a non-empty string")
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")

    if not isinstance(mac_tool, str) or not mac_tool.strip():
        raise ValueError("mac_tool must be a non-empty string")
    if not isinstance(flop_tool, str) or not flop_tool.strip():
        raise ValueError("flop_tool must be a non-empty string")
    static = _static_model_cost(
        model,
        inputs,
        checkpoint_path=checkpoint_path,
        profiler_enabled=bool(profiler_enabled),
        mac_tool=mac_tool,
        flop_tool=flop_tool,
    )

    if bool(power_enabled):
        power_resolution = _power_device_resolution(
            model,
            inputs,
            power_device_index,
        )
        resolved_power_device_index = power_resolution["physical_index"]
        if resolved_power_device_index is None:
            sampler = None
            start_error = RuntimeError(power_resolution["reason"])
        else:
            sampler, start_error = _start_power_sampler(
                resolved_power_device_index,
                power_interval_seconds,
            )
    else:
        power_resolution = {
            "physical_index": power_device_index,
            "logical_index": None,
            "status": "disabled",
            "reason": "power measurement disabled",
            "visible_devices": _cuda_visible_devices(),
        }
        resolved_power_device_index = power_device_index
        sampler = None
        start_error = None
    forward: dict[str, Any]
    sweep: dict[str, Any]
    power_stats: Any
    try:
        # The efficiency helpers own eval/inference_mode and preserve original
        # nested training flags.  This outer section names exactly what NVML
        # samples cover: model inference, not checkpoint restoration/static I/O.
        forward = dict(
            measure_forward(
                model,
                inputs,
                warmup_iterations=warmup_iterations,
                repetitions=repetitions,
                protocol=protocol,
            )
        )
        sweep = dict(
            measure_batch_sweep(
                model,
                _input_factory(inputs),
                batch_sizes,
                warmup_iterations=warmup_iterations,
                repetitions=repetitions,
                protocol=protocol,
            )
        )
    finally:
        power_stats = _stop_power_sampler(
            sampler,
            start_error,
            device_index=resolved_power_device_index,
            interval_seconds=power_interval_seconds,
            disabled=not bool(power_enabled),
        )

    result: dict[str, Any] = {
        "static_model_cost": static,
        "measure_forward": forward,
        "measure_batch_sweep": sweep,
        # Short aliases make the pure contract convenient for callers while
        # retaining the explicit helper names for diagnostics.
        "static": static,
        "forward": forward,
        "sweep": sweep,
    }
    result.update(_prefixed("model", static))
    result.update(_prefixed("inference", forward))
    result.update(
        _prefixed("model", _runtime_metadata(model, inputs))
    )
    result.update(_power_fields(power_stats))
    result.update(
        {
            "power/logical_device_index": power_resolution["logical_index"],
            "power/device_mapping_status": power_resolution["status"],
            "power/device_mapping_reason": power_resolution["reason"],
            "power/cuda_visible_devices": power_resolution["visible_devices"],
            "power/device_mapping": {
                "logical_index": power_resolution["logical_index"],
                "physical_index": power_resolution["physical_index"],
                "status": power_resolution["status"],
                "visible_devices": power_resolution["visible_devices"],
                "reason": power_resolution["reason"],
            },
        }
    )

    samples_per_second, voxels_per_second = _throughput(
        forward.get("latency_mean_ms"),
        forward.get("batch_size"),
        forward.get("input_shape"),
    )
    result["inference/throughput_samples_per_second"] = samples_per_second
    result["inference/throughput_voxels_per_second"] = voxels_per_second
    result.update(
        {
            "inference/sweep_status": sweep.get(
                "sweep_status", sweep.get("status", "unavailable")
            ),
            "inference/sweep_rows": sweep.get("rows", []),
            "inference/largest_passing_batch": sweep.get("largest_passing_batch"),
            "inference/first_failing_batch": sweep.get("first_failing_batch"),
        }
    )
    # The protocol and fixed-input metadata are canonical even when a custom
    # measurement fixture omits those optional helper fields.
    result.setdefault("inference/warmup_iterations", warmup_iterations)
    result.setdefault("inference/repetitions", repetitions)
    result.setdefault("inference/batch_size", _input_batch_size(inputs))
    result["inference/protocol"] = protocol
    return result


__all__ = [
    "BENCHMARK_SCHEMA_VERSION",
    "BenchmarkResult",
    "hash_case_id",
    "run_model_protocol",
    "serialize_benchmark",
]
