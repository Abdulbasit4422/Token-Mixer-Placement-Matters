from __future__ import annotations

import math
import sys

import pytest
import torch
from torch import nn

import token_mixer.evaluation.efficiency as efficiency
from token_mixer.evaluation.efficiency import (
    NvmlPowerSampler,
    checkpoint_size_bytes,
    measure_batch_sweep,
    measure_forward,
    model_parameter_counts,
    static_model_cost,
    summarize_timings,
)


def test_summarize_timings_returns_distribution_summary():
    summary = summarize_timings([1.0, 2.0, 3.0])

    assert summary.mean_ms == pytest.approx(2.0)
    assert summary.median_ms == pytest.approx(2.0)
    assert summary.p95_ms == pytest.approx(2.9)
    assert summary.std_ms == pytest.approx(math.sqrt(2.0 / 3.0))
    assert summary.samples == 3


@pytest.mark.parametrize("samples", [[], [1.0, float("nan")], [1.0, float("inf")]])
def test_summarize_timings_rejects_empty_or_nonfinite_samples(samples):
    with pytest.raises(ValueError, match="empty|finite"):
        summarize_timings(samples)


def test_parameter_counts_include_total_and_trainable_parameters():
    model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
    for parameter in model[1].parameters():
        parameter.requires_grad = False

    assert model_parameter_counts(model) == {
        "parameters": 26,
        "trainable_parameters": 16,
    }


def test_checkpoint_size_bytes_returns_file_size(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"0123456789")

    assert checkpoint_size_bytes(checkpoint) == 10
    assert checkpoint_size_bytes(None) is None


def test_measure_forward_reuses_prepared_input_and_returns_schema():
    seen = []

    class Model(nn.Module):
        def forward(self, value):
            seen.append(id(value))
            return value

    inputs = torch.ones(2, 3, 4, 4)
    result = measure_forward(
        Model(),
        inputs,
        warmup_iterations=2,
        repetitions=3,
        protocol="fixture",
    )

    assert len(seen) == 5
    assert set(seen) == {id(inputs)}
    assert result["status"] == "ok"
    assert result["latency_mean_ms"] >= 0.0
    assert result["batch_size"] == 2
    assert result["input_shape"] == [2, 3, 4, 4]
    assert result["warmup_iterations"] == 2
    assert result["repetitions"] == 3
    assert result["protocol"] == "fixture"
    assert result["peak_memory_allocated_gb"] is None
    assert result["peak_memory_reserved_gb"] is None


@pytest.mark.parametrize(
    ("warmup_iterations", "repetitions"),
    [(-1, 1), (0, 0), (1, -1)],
)
def test_measure_forward_rejects_invalid_iteration_counts(
    warmup_iterations, repetitions
):
    with pytest.raises(ValueError, match="warmup|repetitions"):
        measure_forward(
            nn.Identity(),
            torch.ones(1, 2),
            warmup_iterations=warmup_iterations,
            repetitions=repetitions,
            protocol="fixture",
        )


def test_static_model_cost_reports_thop_and_fvcore_for_same_input(tmp_path):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Conv2d(3, 4, kernel_size=3)

        def forward(self, value):
            return self.layer(value)

    model = Model()
    inputs = torch.ones(2, 3, 8, 8)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")

    result = static_model_cost(model, inputs=inputs, checkpoint_path=checkpoint)

    assert result["parameters"] == 112
    assert result["trainable_parameters"] == 112
    assert result["checkpoint_bytes"] == len(b"checkpoint")
    assert result["macs"] > 0
    assert result["flops"] > 0
    assert result["mac_status"] == "ok"
    assert result["flop_status"] == "ok"
    assert result["mac_tool"] == "thop"
    assert result["flop_tool"] == "fvcore"
    assert result["mac_tool_version"] == "2.1.6"
    assert result["flop_tool_version"] == "0.1.5.post20221221"
    assert result["mac_convention"]
    assert result["flop_convention"]
    assert result["unsupported_ops"] == {}
    assert result["uncalled_modules"] == []


def test_static_model_cost_marks_unsupported_operator_partial():
    class UnsupportedModel(nn.Module):
        def forward(self, value):
            return torch.sin(value)

    result = static_model_cost(
        UnsupportedModel(),
        inputs=torch.ones(1, 3, 4, 4),
    )

    assert result["unsupported_ops"]
    assert result["flop_status"] == "partial"
    assert result["flops"] is not None
    assert result["mac_status"] == "partial"


def test_static_model_cost_marks_mixed_unsupported_mac_count_partial():
    class MixedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Conv2d(3, 4, kernel_size=3)

        def forward(self, value):
            return torch.sin(self.layer(value))

    result = static_model_cost(
        MixedModel(),
        inputs=torch.ones(1, 3, 8, 8),
    )

    assert result["macs"] > 0
    assert result["unsupported_ops"]
    assert result["mac_status"] == "partial"


def test_static_model_cost_reuses_exact_fixed_input():
    seen = []

    class Model(nn.Module):
        def forward(self, value):
            seen.append(id(value))
            return value + 1

    inputs = torch.ones(1, 2)
    static_model_cost(Model(), inputs=inputs)

    assert seen
    assert set(seen) == {id(inputs)}


@pytest.mark.parametrize("input_style", ["tuple", "mapping"])
def test_static_model_cost_accepts_call_structure_used_by_forward(input_style):
    class PairModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(2, 2)

        def forward(self, left, right):
            return self.layer(left + right)

    left = torch.ones(1, 2)
    right = torch.ones(1, 2)
    inputs = (left, right) if input_style == "tuple" else {"left": left, "right": right}

    result = static_model_cost(PairModel(), inputs=inputs)

    assert result["macs"] is not None
    assert result["flops"] is not None
    assert result["mac_status"] != "unavailable"
    assert result["flop_status"] != "unavailable"


def test_nvml_sampler_integrates_power_with_trapezoidal_rule(monkeypatch):
    class FakeNvml:
        __version__ = "13.610.43"

        def __init__(self):
            self.shutdown_calls = 0

        def nvmlInit(self):
            pass

        def nvmlDeviceGetHandleByIndex(self, index):
            assert index == 0
            return "handle"

        def nvmlDeviceGetPowerUsage(self, handle):
            assert handle == "handle"
            return 100_000

        def nvmlSystemGetNVMLVersion(self):
            return b"550.1"

        def nvmlShutdown(self):
            self.shutdown_calls += 1

    class NoopThread:
        def __init__(self, target, daemon):
            self.target = target
            self.daemon = daemon

        def start(self):
            pass

        def join(self, timeout=None):
            pass

        def is_alive(self):
            return False

    fake_nvml = FakeNvml()
    monkeypatch.setattr(efficiency, "pynvml", fake_nvml)
    monkeypatch.setattr(efficiency.threading, "Thread", NoopThread)
    monkeypatch.setattr(efficiency.time, "monotonic", lambda: 0.0)
    sampler = NvmlPowerSampler(device_index=0, interval_seconds=0.01)
    sampler.start()
    sampler._record_sample(200.0, timestamp=1.0)
    sampler._record_sample(300.0, timestamp=2.0)

    stats = sampler.snapshot()
    sampler.stop()

    assert fake_nvml.shutdown_calls == 1
    assert stats.status == "ok"
    assert stats.samples == 3
    assert stats.max_watts == pytest.approx(300.0)
    assert stats.average_watts == pytest.approx(200.0)
    assert stats.joules == pytest.approx(400.0)
    assert stats.nvml_version == "550.1"


def test_nvml_sampler_reports_unavailable_without_nvml(monkeypatch):
    monkeypatch.setattr(efficiency, "pynvml", None)
    monkeypatch.setitem(sys.modules, "pynvml", None)

    sampler = NvmlPowerSampler(device_index=0, interval_seconds=0.01)
    sampler.start()
    stats = sampler.stop()

    assert stats.status == "unavailable"
    assert stats.reason
    assert stats.samples == 0


def test_batch_sweep_records_oom_boundary_and_reraises_unrelated_errors():
    class Model(nn.Module):
        def forward(self, value):
            if value.shape[0] == 2:
                raise RuntimeError("CUDA out of memory")
            if value.shape[0] == 3:
                raise RuntimeError("unrelated")
            return value

    def input_factory(batch_size):
        return torch.ones(batch_size, 2)

    result = measure_batch_sweep(
        Model(),
        input_factory,
        [1, 2, 4],
        warmup_iterations=0,
        repetitions=1,
        protocol="fixture",
    )

    assert result["status"] == "oom"
    assert result["sweep_status"] == "oom"
    assert result["largest_passing_batch"] == 1
    assert result["first_failing_batch"] == 2
    assert [row["batch_size"] for row in result["rows"]] == [1, 2]
    assert result["rows"][-1]["status"] == "oom"

    with pytest.raises(RuntimeError, match="unrelated"):
        measure_batch_sweep(
            Model(),
            input_factory,
            [1, 3],
            warmup_iterations=0,
            repetitions=1,
            protocol="fixture",
        )


def test_batch_sweep_accepts_explicit_oom_message_from_non_runtime_error():
    class Model(nn.Module):
        def forward(self, value):
            if value.shape[0] == 2:
                raise ValueError("out of memory")
            return value

    result = measure_batch_sweep(
        Model(),
        lambda batch_size: torch.ones(batch_size, 2),
        [1, 2],
        warmup_iterations=0,
        repetitions=1,
        protocol="fixture",
    )

    assert result["status"] == "oom"
    assert result["first_failing_batch"] == 2


def test_cuda_timing_uses_input_device_stream_and_allocator_boundaries(monkeypatch):
    cuda_device = torch.device("cuda:1")
    prepared_input = object()
    expected_stream = object()
    synchronize_calls = []
    current_stream_calls = []
    reset_calls = []
    allocated_calls = []
    reserved_calls = []

    class Model(nn.Module):
        def forward(self, value):
            assert value is prepared_input
            return value

    class FakeEvent:
        instances = []

        def __init__(self, *, enable_timing):
            assert enable_timing is True
            self.recorded_streams = []
            self.synchronized = False
            FakeEvent.instances.append(self)

        def record(self, stream=None):
            self.recorded_streams.append(stream)

        def synchronize(self):
            self.synchronized = True

        def elapsed_time(self, other):
            assert self.synchronized is False
            assert other.synchronized is True
            return 2.5

    monkeypatch.setattr(
        efficiency,
        "_input_metadata",
        lambda _inputs: ([4, 2], 4, cuda_device),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda device=None: synchronize_calls.append(device),
    )
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda device=None: current_stream_calls.append(device) or expected_stream,
    )
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(
        torch.cuda,
        "reset_peak_memory_stats",
        lambda device=None: reset_calls.append(device),
    )
    monkeypatch.setattr(
        torch.cuda,
        "max_memory_allocated",
        lambda device=None: allocated_calls.append(device) or 2 * 1024**3,
    )
    monkeypatch.setattr(
        torch.cuda,
        "max_memory_reserved",
        lambda device=None: reserved_calls.append(device) or 3 * 1024**3,
    )

    result = measure_forward(
        Model(),
        prepared_input,
        warmup_iterations=0,
        repetitions=1,
        protocol="cuda-fixture",
    )

    assert synchronize_calls == [cuda_device]
    assert current_stream_calls == [cuda_device]
    assert reset_calls == [cuda_device]
    assert allocated_calls == [cuda_device]
    assert reserved_calls == [cuda_device]
    assert [event.recorded_streams for event in FakeEvent.instances] == [
        [expected_stream],
        [expected_stream],
    ]
    assert result["latency_mean_ms"] == pytest.approx(2.5)
    assert result["peak_memory_allocated_gb"] == pytest.approx(2.0)
    assert result["peak_memory_reserved_gb"] == pytest.approx(3.0)


def test_measure_forward_restores_mixed_nested_training_states():
    class Child(nn.Module):
        def __init__(self):
            super().__init__()
            self.grandchild = nn.ReLU()

        def forward(self, value):
            return self.grandchild(value)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.child = Child()

        def forward(self, value):
            return self.child(value)

    model = Model().train()
    model.child.eval()
    model.child.grandchild.train()

    measure_forward(
        model,
        torch.ones(1, 2),
        warmup_iterations=1,
        repetitions=1,
        protocol="state-fixture",
    )

    assert model.training is True
    assert model.child.training is False
    assert model.child.grandchild.training is True


def test_static_model_cost_marks_diagnostic_failure_partial(monkeypatch):
    class FailingDiagnostics:
        def __init__(self, model, inputs):
            assert isinstance(model, nn.Module)
            assert len(inputs) == 1

        def total(self):
            return 123

        def unsupported_ops(self):
            raise RuntimeError("unsupported diagnostics unavailable")

        def uncalled_modules(self):
            raise RuntimeError("uncalled diagnostics unavailable")

    monkeypatch.setattr(efficiency, "FlopCountAnalysis", FailingDiagnostics)
    result = static_model_cost(nn.Linear(2, 2), inputs=torch.ones(1, 2))

    assert result["flops"] == 123
    assert result["flop_status"] == "partial"
    assert result["flop_error"]


def test_static_model_cost_preserves_independent_mac_failure_status(monkeypatch):
    def fail_profile(*args, **kwargs):
        raise RuntimeError("THOP fixture failure")

    monkeypatch.setattr(efficiency.thop, "profile", fail_profile)
    result = static_model_cost(nn.Linear(2, 2), inputs=torch.ones(1, 2))

    assert result["macs"] is None
    assert result["mac_status"] == "unavailable"
    assert result["mac_error"]
    assert result["flop_status"] == "ok"


def test_static_model_cost_preserves_independent_flop_failure_status(monkeypatch):
    class FailingFlops:
        def __init__(self, model, inputs):
            pass

        def total(self):
            raise RuntimeError("fvcore fixture failure")

    monkeypatch.setattr(efficiency, "FlopCountAnalysis", FailingFlops)
    result = static_model_cost(nn.Linear(2, 2), inputs=torch.ones(1, 2))

    assert result["mac_status"] == "ok"
    assert result["flop_status"] == "unavailable"
    assert result["flops"] is None
    assert result["flop_error"]


def test_nvml_stop_fully_joins_before_shutdown(monkeypatch):
    events = []

    class FakeNvml:
        def __init__(self):
            self.shutdown = False

        def nvmlInit(self):
            events.append("init")

        def nvmlDeviceGetHandleByIndex(self, index):
            return "handle"

        def nvmlDeviceGetPowerUsage(self, handle):
            assert not self.shutdown, "NVML used after shutdown"
            events.append("power")
            return 100_000

        def nvmlSystemGetNVMLVersion(self):
            return b"fixture"

        def nvmlShutdown(self):
            events.append("shutdown")
            self.shutdown = True

    class ControlledThread:
        instance = None

        def __init__(self, target, daemon):
            assert daemon is True
            self.target = target
            self.alive = True
            ControlledThread.instance = self

        def start(self):
            events.append("thread-start")

        def join(self, timeout=None):
            events.append(("join", timeout))
            if timeout is None:
                self.alive = False
                events.append("thread-exit")
                self.target()

        def is_alive(self):
            return self.alive

        def finish(self):
            self.alive = False
            events.append("thread-exit")
            self.target()

    fake_nvml = FakeNvml()
    monkeypatch.setattr(efficiency, "pynvml", fake_nvml)
    monkeypatch.setattr(efficiency.threading, "Thread", ControlledThread)
    sampler = NvmlPowerSampler(device_index=0, interval_seconds=0.01)
    sampler.start()
    thread = ControlledThread.instance

    sampler.stop()
    assert thread is not None
    assert thread.alive is False
    assert [event for event in events if isinstance(event, tuple)] == [("join", None)]
    assert events.index("thread-exit") < events.index("shutdown")
    assert fake_nvml.shutdown is True


def test_nvml_thread_failure_marks_unavailable_and_shuts_down_after_exit(monkeypatch):
    events = []

    class FailingNvml:
        def __init__(self):
            self.power_calls = 0

        def nvmlInit(self):
            pass

        def nvmlDeviceGetHandleByIndex(self, index):
            return "handle"

        def nvmlDeviceGetPowerUsage(self, handle):
            self.power_calls += 1
            if self.power_calls == 1:
                return 100_000
            events.append("power-failure")
            raise PermissionError("NVML permission denied")

        def nvmlSystemGetNVMLVersion(self):
            return b"fixture"

        def nvmlShutdown(self):
            events.append("shutdown")

    class ControlledThread:
        instance = None

        def __init__(self, target, daemon):
            self.target = target
            self.alive = True
            ControlledThread.instance = self

        def start(self):
            pass

        def join(self, timeout=None):
            pass

        def is_alive(self):
            return self.alive

        def finish(self):
            self.alive = False
            self.target()

    monkeypatch.setattr(efficiency, "pynvml", FailingNvml())
    monkeypatch.setattr(efficiency.threading, "Thread", ControlledThread)
    sampler = NvmlPowerSampler(device_index=0, interval_seconds=0.01)
    sampler.start()
    sampler._sample_once()
    assert sampler.snapshot().status == "unavailable"
    assert events == ["power-failure"]

    ControlledThread.instance.finish()
    assert events == ["power-failure", "shutdown"]


@pytest.mark.parametrize("failure_stage", ["construct", "start"])
def test_nvml_thread_start_failure_marks_unavailable_and_shuts_down(
    monkeypatch, failure_stage
):
    events = []

    class FakeNvml:
        def __init__(self):
            self.shutdown_calls = 0

        def nvmlInit(self):
            events.append("init")

        def nvmlDeviceGetHandleByIndex(self, index):
            return "handle"

        def nvmlDeviceGetPowerUsage(self, handle):
            events.append("power")
            return 100_000

        def nvmlSystemGetNVMLVersion(self):
            return b"fixture"

        def nvmlShutdown(self):
            events.append("shutdown")
            self.shutdown_calls += 1

    class FailingThread:
        def __init__(self, target, daemon):
            events.append("thread-construct")
            if failure_stage == "construct":
                raise RuntimeError("thread construction failed")
            self.target = target
            self.alive = False

        def start(self):
            events.append("thread-start")
            raise RuntimeError("thread start failed")

        def is_alive(self):
            return self.alive

        def join(self):
            events.append("join")

    fake_nvml = FakeNvml()
    monkeypatch.setattr(efficiency, "pynvml", fake_nvml)
    monkeypatch.setattr(efficiency.threading, "Thread", FailingThread)
    sampler = NvmlPowerSampler(device_index=0, interval_seconds=0.01)

    stats = sampler.start().snapshot()

    assert stats.status == "unavailable"
    assert stats.reason
    assert "shutdown" in events
    assert fake_nvml.shutdown_calls == 1
    assert sampler._thread is None
    assert sampler._initialized is False
    if failure_stage == "construct":
        assert events == ["init", "power", "thread-construct", "shutdown"]
    else:
        assert events == [
            "init",
            "power",
            "thread-construct",
            "thread-start",
            "shutdown",
        ]
