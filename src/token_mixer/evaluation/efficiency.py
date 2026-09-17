"""Small, reproducible measurement helpers for model efficiency experiments.

This module deliberately owns measurements only.  It does not start training,
write tracking artifacts, or make policy decisions about benchmark protocols.
"""

from __future__ import annotations

import math
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import fvcore
import numpy as np
import thop
import torch
from fvcore.nn import FlopCountAnalysis
from torch import nn


# NVML stays lazy so importing the evaluation package does not require a
# functioning NVIDIA driver.  Tests and callers may replace this module-level
# value with a fixture before starting a sampler.
pynvml: Any | None = None

_BYTES_PER_GIB = float(1024**3)
_MAX_REASON_LENGTH = 240


@dataclass(frozen=True)
class TimingSummary:
    mean_ms: float
    median_ms: float
    p95_ms: float
    std_ms: float
    samples: int


@dataclass(frozen=True)
class PowerStats:
    average_watts: float | None
    max_watts: float | None
    joules: float | None
    samples: int
    interval_seconds: float
    device_index: int
    nvml_version: str | None
    status: str
    reason: str | None = None


def _bounded_reason(error: object) -> str:
    reason = " ".join(str(error).split())
    if not reason:
        reason = error.__class__.__name__ if isinstance(error, BaseException) else "unknown error"
    return reason[:_MAX_REASON_LENGTH]


def _validate_iterations(warmup_iterations: int, repetitions: int) -> None:
    if isinstance(warmup_iterations, bool) or not isinstance(warmup_iterations, int):
        raise ValueError("warmup_iterations must be a non-negative integer")
    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be a non-negative integer")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int):
        raise ValueError("repetitions must be a positive integer")
    if repetitions <= 0:
        raise ValueError("repetitions must be a positive integer")


def summarize_timings(samples: Iterable[float]) -> TimingSummary:
    """Summarize finite latency samples in milliseconds.

    NumPy's default percentile interpolation is intentional: it gives the
    conventional linear p95 for small repeated-sample fixtures as well as
    larger benchmark runs.
    """

    values = np.asarray(list(samples), dtype=np.float64)
    if values.size == 0:
        raise ValueError("timing samples cannot be empty")
    if not np.all(np.isfinite(values)):
        raise ValueError("timing samples must be finite")

    return TimingSummary(
        mean_ms=float(np.mean(values)),
        median_ms=float(np.median(values)),
        p95_ms=float(np.percentile(values, 95)),
        std_ms=float(np.std(values)),
        samples=int(values.size),
    )


def model_parameter_counts(model: nn.Module) -> dict[str, int]:
    """Return total and trainable parameter counts without relying on a tool."""

    parameters = list(model.parameters())
    return {
        "parameters": sum(parameter.numel() for parameter in parameters),
        "trainable_parameters": sum(
            parameter.numel() for parameter in parameters if parameter.requires_grad
        ),
    }


def checkpoint_size_bytes(checkpoint_path: str | Path | None) -> int | None:
    """Return checkpoint file size, or ``None`` when no file is supplied."""

    if checkpoint_path is None:
        return None
    try:
        path = Path(checkpoint_path)
        if not path.is_file():
            return None
        return int(path.stat().st_size)
    except (OSError, TypeError, ValueError):
        return None


def _as_cuda_device(device: torch.device | str | int | None) -> torch.device | None:
    if device is None:
        if not torch.cuda.is_available():
            return None
        return torch.device("cuda")
    resolved = torch.device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        return None
    return resolved


def reset_peak_memory(device: torch.device | str | int | None = None) -> None:
    """Reset PyTorch CUDA allocator peaks when CUDA is available."""

    resolved = _as_cuda_device(device)
    if resolved is None:
        return
    torch.cuda.reset_peak_memory_stats(resolved)


def peak_memory_gb(
    device: torch.device | str | int | None = None,
) -> dict[str, float | None]:
    """Return peak allocated and reserved CUDA memory in GiB.

    CPU measurements intentionally use ``None`` rather than zero: zero would
    falsely imply that a CUDA allocator measurement occurred.
    """

    resolved = _as_cuda_device(device)
    if resolved is None:
        return {"allocated_gb": None, "reserved_gb": None}
    return {
        "allocated_gb": float(torch.cuda.max_memory_allocated(resolved) / _BYTES_PER_GIB),
        "reserved_gb": float(torch.cuda.max_memory_reserved(resolved) / _BYTES_PER_GIB),
    }


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


def _input_metadata(inputs: Any) -> tuple[list[int] | None, int | None, torch.device | None]:
    tensor = _first_tensor(inputs)
    if tensor is None:
        return None, None, None
    shape = [int(dimension) for dimension in tensor.shape]
    batch_size = shape[0] if shape else None
    return shape, batch_size, tensor.device


def _model_call_parts(inputs: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Preserve the positional/keyword structure used by ``_call_model``."""

    if isinstance(inputs, Mapping):
        return (), dict(inputs)
    if isinstance(inputs, tuple):
        return inputs, {}
    if isinstance(inputs, list):
        return tuple(inputs), {}
    return (inputs,), {}


def _call_model(model: nn.Module, inputs: Any) -> Any:
    """Call common prepared-input forms without allocating replacement inputs."""

    args, kwargs = _model_call_parts(inputs)
    return model(*args, **kwargs)


class _KeywordInputAdapter(nn.Module):
    """Expose keyword inputs as profiler-compatible positional arguments."""

    def __init__(self, model: nn.Module, keyword_names: Sequence[str]):
        super().__init__()
        self.model = model
        self.keyword_names = tuple(keyword_names)

    def forward(self, *args: Any) -> Any:
        if len(args) != len(self.keyword_names):
            raise TypeError(
                "profiler input count does not match prepared keyword inputs"
            )
        return self.model(**dict(zip(self.keyword_names, args)))


def _counter_call(model: nn.Module, inputs: Any) -> tuple[nn.Module, tuple[Any, ...]]:
    """Adapt prepared calls to counters while retaining original call semantics."""

    args, kwargs = _model_call_parts(inputs)
    if not kwargs:
        return model, args
    return _KeywordInputAdapter(model, tuple(kwargs)), tuple(kwargs.values())


@contextmanager
def _evaluation_mode(model: nn.Module):
    states = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        yield
    finally:
        # Assign flags directly.  Calling train() recursively would overwrite
        # intentionally mixed nested training states.
        for module, training in states:
            module.training = training


def _cuda_timing_device(inputs: Any) -> torch.device | None:
    _, _, input_device = _input_metadata(inputs)
    if input_device is None or input_device.type != "cuda":
        return None
    if not torch.cuda.is_available():
        return None
    return input_device


def measure_forward(
    model: nn.Module,
    inputs: Any,
    *,
    warmup_iterations: int,
    repetitions: int,
    protocol: str,
) -> dict[str, Any]:
    """Measure model-only forward latency for one prepared input object."""

    _validate_iterations(warmup_iterations, repetitions)
    input_shape, batch_size, _ = _input_metadata(inputs)
    cuda_device = _cuda_timing_device(inputs)
    latencies: list[float] = []

    with _evaluation_mode(model), torch.inference_mode():
        for _ in range(warmup_iterations):
            _call_model(model, inputs)

        # Warmups are excluded from allocator peaks and timed samples.
        reset_peak_memory(cuda_device)

        if cuda_device is None:
            for _ in range(repetitions):
                started = time.perf_counter()
                _call_model(model, inputs)
                latencies.append((time.perf_counter() - started) * 1000.0)
        else:
            for _ in range(repetitions):
                torch.cuda.synchronize(cuda_device)
                stream = torch.cuda.current_stream(cuda_device)
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record(stream)
                _call_model(model, inputs)
                end_event.record(stream)
                end_event.synchronize()
                latencies.append(float(start_event.elapsed_time(end_event)))

        summary = summarize_timings(latencies)

    memory = peak_memory_gb(cuda_device)
    return {
        "status": "ok",
        "latency_mean_ms": summary.mean_ms,
        "latency_median_ms": summary.median_ms,
        "latency_p95_ms": summary.p95_ms,
        "latency_std_ms": summary.std_ms,
        "latency_samples_ms": latencies,
        "samples": summary.samples,
        "peak_memory_allocated_gb": memory["allocated_gb"],
        "peak_memory_reserved_gb": memory["reserved_gb"],
        "input_shape": input_shape,
        "batch_size": batch_size,
        "warmup_iterations": warmup_iterations,
        "repetitions": repetitions,
        "protocol": protocol,
    }


def _is_out_of_memory(error: BaseException) -> bool:
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    return "out of memory" in str(error).lower()


def _empty_cuda_cache() -> None:
    try:
        torch.cuda.empty_cache()
    except Exception:
        # Cache cleanup must not hide the original OOM boundary.
        pass


def _oom_row(
    batch_size: int,
    inputs: Any | None,
    *,
    warmup_iterations: int,
    repetitions: int,
    protocol: str,
    error: BaseException,
) -> dict[str, Any]:
    input_shape, _, _ = _input_metadata(inputs)
    return {
        "status": "oom",
        "error": _bounded_reason(error),
        "input_shape": input_shape,
        "batch_size": batch_size,
        "warmup_iterations": warmup_iterations,
        "repetitions": repetitions,
        "protocol": protocol,
        "peak_memory_allocated_gb": None,
        "peak_memory_reserved_gb": None,
    }


def measure_batch_sweep(
    model: nn.Module,
    input_factory: Callable[[int], Any],
    batch_sizes: Sequence[int],
    *,
    warmup_iterations: int,
    repetitions: int,
    protocol: str,
) -> dict[str, Any]:
    """Measure declared batch sizes until an explicit CUDA OOM boundary."""

    _validate_iterations(warmup_iterations, repetitions)
    declared_batches = list(batch_sizes)
    if not declared_batches:
        raise ValueError("batch_sizes cannot be empty")
    for batch_size in declared_batches:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_sizes must contain positive integers")

    rows: list[dict[str, Any]] = []
    largest_passing_batch: int | None = None
    first_failing_batch: int | None = None

    for batch_size in declared_batches:
        inputs = None
        try:
            inputs = input_factory(batch_size)
            row = measure_forward(
                model,
                inputs,
                warmup_iterations=warmup_iterations,
                repetitions=repetitions,
                protocol=protocol,
            )
        except Exception as error:
            if not _is_out_of_memory(error):
                raise
            first_failing_batch = batch_size
            rows.append(
                _oom_row(
                    batch_size,
                    inputs,
                    warmup_iterations=warmup_iterations,
                    repetitions=repetitions,
                    protocol=protocol,
                    error=error,
                )
            )
            _empty_cuda_cache()
            break
        rows.append(row)
        if row.get("status") == "ok":
            if largest_passing_batch is None or batch_size > largest_passing_batch:
                largest_passing_batch = batch_size

    status = "oom" if first_failing_batch is not None else "ok"
    return {
        "status": status,
        "sweep_status": status,
        "rows": rows,
        "largest_passing_batch": largest_passing_batch,
        "first_failing_batch": first_failing_batch,
        "batch_sizes": declared_batches,
        "protocol": protocol,
    }


def _package_version(package: Any, distribution: str) -> str:
    value = getattr(package, "__version__", None)
    if value is not None:
        return str(value)
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "unknown"


def _operator_counts(value: Any) -> dict[str, int]:
    if value is None:
        return {}
    try:
        return {str(name): int(count) for name, count in value.items()}
    except AttributeError:
        return {str(name): 1 for name in value}


def _uncalled_modules(value: Any) -> list[str]:
    if value is None:
        return []
    return sorted(str(name) for name in value)


def _numeric_value(value: Any) -> int | float | None:
    if isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(float(value)):
        number = float(value)
        return int(number) if number.is_integer() else number
    return None


THOP_MAC_CONVENTION = (
    "THOP MACs: one multiply-accumulate pair counted as one MAC; no conversion to FLOPs."
)
FVCORE_FLOP_CONVENTION = (
    "fvcore FlopCountAnalysis operator FLOPs, reported without conversion or substitution."
)


def static_model_cost(
    model: nn.Module,
    *,
    inputs: Any,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    """Collect parameters, checkpoint size, THOP MACs, and fvcore FLOPs.

    Both counters receive the same prepared input object.  Counter failures are
    isolated so a working counter and authoritative parameter counts survive an
    unsupported custom operator in the other tool.
    """

    result: dict[str, Any] = {
        **model_parameter_counts(model),
        "checkpoint_bytes": checkpoint_size_bytes(checkpoint_path),
        "macs": None,
        "flops": None,
        "mac_tool": "thop",
        "mac_tool_version": _package_version(thop, "ultralytics-thop"),
        "flop_tool": "fvcore",
        "flop_tool_version": _package_version(fvcore, "fvcore"),
        "mac_convention": THOP_MAC_CONVENTION,
        "flop_convention": FVCORE_FLOP_CONVENTION,
        "mac_status": "unavailable",
        "flop_status": "unavailable",
        "mac_error": None,
        "flop_error": None,
        "unsupported_ops": {},
        "uncalled_modules": [],
    }

    counter_model, counter_inputs = _counter_call(model, inputs)

    with _evaluation_mode(model), torch.inference_mode():
        try:
            macs, _ = thop.profile(counter_model, inputs=counter_inputs, verbose=False)
            result["macs"] = _numeric_value(macs)
            result["mac_status"] = "ok" if result["macs"] is not None else "unavailable"
            if result["macs"] is None:
                result["mac_error"] = "THOP returned a non-finite or non-numeric count"
        except Exception as error:
            result["mac_error"] = _bounded_reason(error)

        analysis: FlopCountAnalysis | None = None
        try:
            analysis = FlopCountAnalysis(counter_model, counter_inputs)
            result["flops"] = _numeric_value(analysis.total())
            diagnostic_error = False
            try:
                result["unsupported_ops"] = _operator_counts(analysis.unsupported_ops())
            except Exception as error:
                result["flop_error"] = _bounded_reason(error)
                diagnostic_error = True
            try:
                result["uncalled_modules"] = _uncalled_modules(analysis.uncalled_modules())
            except Exception as error:
                if result["flop_error"] is None:
                    result["flop_error"] = _bounded_reason(error)
                diagnostic_error = True

            if result["flops"] is None:
                result["flop_status"] = "unavailable"
                if result["flop_error"] is None:
                    result["flop_error"] = "fvcore returned a non-finite or non-numeric count"
            elif result["unsupported_ops"] or diagnostic_error:
                result["flop_status"] = "partial"
            else:
                result["flop_status"] = "ok"

            # THOP may omit unsupported operators while still counting other
            # modules. fvcore diagnostics therefore make the complete MAC
            # count non-authoritative, regardless of whether it is zero.
            if (
                result["mac_status"] == "ok"
                and result["unsupported_ops"]
            ):
                result["mac_status"] = "partial"
                result["mac_error"] = (
                    "THOP count may be incomplete because fvcore reported "
                    "unsupported operators"
                )
        except Exception as error:
            result["flop_error"] = _bounded_reason(error)

    statuses = (result["mac_status"], result["flop_status"])
    result["status"] = "ok" if statuses == ("ok", "ok") else "partial"
    if statuses == ("unavailable", "unavailable"):
        result["status"] = "unavailable"
    return result


def _load_pynvml() -> Any:
    global pynvml
    if pynvml is None:
        import pynvml as nvml_module

        pynvml = nvml_module
    return pynvml


def _version_from_nvml(module: Any) -> str | None:
    getter = getattr(module, "nvmlSystemGetNVMLVersion", None)
    if getter is None:
        value = getattr(module, "__version__", None)
    else:
        value = getter()
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return None if value is None else str(value)


class NvmlPowerSampler:
    """Best-effort daemon-thread sampler for board-level NVML power."""

    def __init__(self, device_index: int, interval_seconds: float):
        if isinstance(device_index, bool) or not isinstance(device_index, int) or device_index < 0:
            raise ValueError("device_index must be a non-negative integer")
        if not math.isfinite(float(interval_seconds)) or interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive and finite")

        self.device_index = device_index
        self.interval_seconds = float(interval_seconds)
        self._module: Any | None = None
        self._handle: Any | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._samples: list[tuple[float, float]] = []
        self._initialized = False
        self._started = False
        self._status = "not_started"
        self._reason: str | None = None
        self._nvml_version: str | None = None

    def _set_unavailable(self, error: object) -> None:
        with self._lock:
            self._status = "unavailable"
            self._reason = _bounded_reason(error)

    def _record_sample(self, watts: float, timestamp: float | None = None) -> None:
        watts = float(watts)
        if not math.isfinite(watts) or watts < 0:
            raise ValueError("power sample must be finite and non-negative")
        if timestamp is None:
            timestamp = time.monotonic()
        timestamp = float(timestamp)
        if not math.isfinite(timestamp):
            raise ValueError("power sample timestamp must be finite")
        with self._lock:
            self._samples.append((timestamp, watts))

    def _sample_once(self) -> bool:
        try:
            timestamp = time.monotonic()
            power_mw = self._module.nvmlDeviceGetPowerUsage(self._handle)
            self._record_sample(float(power_mw) / 1000.0, timestamp=timestamp)
            return True
        except Exception as error:
            self._set_unavailable(error)
            self._stop_event.set()
            return False

    def _run(self) -> None:
        try:
            while not self._stop_event.wait(self.interval_seconds):
                if not self._sample_once():
                    return
        finally:
            self._thread_exited()

    def _thread_exited(self) -> None:
        with self._lock:
            self._thread = None
        self._shutdown_nvml()

    def _shutdown_nvml(self) -> None:
        with self._lock:
            if not self._initialized or self._module is None:
                return
            module = self._module
            self._initialized = False
        try:
            module.nvmlShutdown()
        except Exception as error:
            self._set_unavailable(error)

    @staticmethod
    def _join_thread(thread: threading.Thread) -> None:
        thread.join()
        while thread.is_alive():
            thread.join()

    def start(self) -> "NvmlPowerSampler":
        if self._started:
            return self
        self._started = True
        try:
            self._module = _load_pynvml()
            self._module.nvmlInit()
            self._initialized = True
            self._nvml_version = _version_from_nvml(self._module)
            self._handle = self._module.nvmlDeviceGetHandleByIndex(self.device_index)
            self._status = "ok"
        except Exception as error:
            self._set_unavailable(error)
            return self

        # Capture one boundary sample synchronously, then continue sampling in
        # a daemon thread.  A first sample makes short benchmark sections
        # useful while preserving the non-blocking measured model loop.
        if not self._sample_once():
            return self
        try:
            thread = threading.Thread(target=self._run, daemon=True)
            self._thread = thread
            thread.start()
        except Exception as error:
            self._set_unavailable(error)
            self._stop_event.set()
            if self._thread is not None:
                thread = self._thread
                try:
                    if thread.is_alive():
                        self._join_thread(thread)
                finally:
                    self._thread = None
            self._shutdown_nvml()
        return self

    def _snapshot_values(self) -> tuple[float | None, float | None, float | None, int]:
        with self._lock:
            samples = list(self._samples)
        if not samples:
            return None, None, None, 0

        max_watts = max(watts for _, watts in samples)
        if len(samples) == 1:
            return samples[0][1], max_watts, 0.0, 1

        joules = 0.0
        for (left_time, left_watts), (right_time, right_watts) in zip(samples, samples[1:]):
            elapsed = right_time - left_time
            if elapsed < 0:
                continue
            joules += (left_watts + right_watts) * 0.5 * elapsed

        duration = samples[-1][0] - samples[0][0]
        if duration > 0:
            average_watts = joules / duration
        else:
            average_watts = float(np.mean([watts for _, watts in samples]))
        return average_watts, max_watts, joules, len(samples)

    def snapshot(self) -> PowerStats:
        average_watts, max_watts, joules, sample_count = self._snapshot_values()
        with self._lock:
            status = self._status
            reason = self._reason
            version = self._nvml_version
        if status == "not_started":
            status = "unavailable"
            reason = "sampler was not started"
        return PowerStats(
            average_watts=average_watts,
            max_watts=max_watts,
            joules=joules,
            samples=sample_count,
            interval_seconds=self.interval_seconds,
            device_index=self.device_index,
            nvml_version=version,
            status=status,
            reason=reason,
        )

    def stop(self) -> PowerStats:
        thread = self._thread
        if thread is not None:
            self._stop_event.set()
            self._join_thread(thread)
            self._thread = None
            self._shutdown_nvml()
        else:
            self._shutdown_nvml()
        return self.snapshot()


__all__ = [
    "TimingSummary",
    "PowerStats",
    "NvmlPowerSampler",
    "summarize_timings",
    "model_parameter_counts",
    "checkpoint_size_bytes",
    "reset_peak_memory",
    "peak_memory_gb",
    "measure_forward",
    "measure_batch_sweep",
    "static_model_cost",
]
