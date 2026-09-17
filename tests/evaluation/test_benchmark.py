from __future__ import annotations

import hashlib
import json
import re

import pytest
import torch
from torch import nn

import token_mixer.evaluation.benchmark as benchmark
from token_mixer.evaluation.benchmark import (
    BenchmarkResult,
    hash_case_id,
    run_model_protocol,
    serialize_benchmark,
)


def test_benchmark_result_is_frozen_and_copies_top_level_inputs():
    summary = {"inference/latency_mean_ms": 2.0}
    rows = [{"protocol": "fixture"}]
    provenance = {"manifest_hash": "sha256"}
    result = BenchmarkResult(summary=summary, rows=rows, provenance=provenance)

    summary["changed"] = True
    rows.append({"protocol": "other"})
    provenance["changed"] = True

    assert "changed" not in result.summary
    assert len(result.rows) == 1
    assert "changed" not in result.provenance
    with pytest.raises((AttributeError, TypeError)):
        result.summary = {}


def test_hash_case_id_uses_first_16_sha256_hex_characters():
    case_id = "BraTS-μ-001"

    assert hash_case_id(case_id) == hashlib.sha256(
        case_id.encode("utf-8")
    ).hexdigest()[:16]
    assert len(hash_case_id(case_id)) == 16


def test_serialize_benchmark_contains_safe_case_rows(tmp_path):
    result = BenchmarkResult(
        summary={"inference/latency_mean_ms": 2.0, "bad": float("inf")},
        rows=[
            {
                "case_id": "BraTS-001",
                "protocol": "native_3d_full_volume",
                "latency_ms": float("nan"),
            }
        ],
        provenance={"manifest_hash": "sha256"},
    )

    path = serialize_benchmark(tmp_path, result)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["rows"][0]["case_id_hash"] == hash_case_id("BraTS-001")
    assert "case_id" not in payload["rows"][0]
    assert payload["rows"][0]["latency_ms"] is None
    assert payload["summary"]["bad"] is None
    assert "BraTS-001" not in path.read_text(encoding="utf-8")


def test_serialize_benchmark_redacts_case_identifiers_in_all_sections(tmp_path):
    raw_ids = [
        "SUMMARY-CASE-001",
        "SUMMARY-CASE-002",
        "ROW-CASE-001",
        "ROW-CASE-002",
        "PROVENANCE-CASE-001",
        "PROVENANCE-CASE-002",
    ]
    result = BenchmarkResult(
        summary={
            "case_id": raw_ids[0],
            "case_ids": raw_ids[1:2],
            "nested": {"case_id": raw_ids[1], "keep": "summary"},
            "keep": {"count": 2},
        },
        rows=[
            {
                "case_id": raw_ids[2],
                "case_ids": [raw_ids[3]],
                "nested": {"case_id": raw_ids[3]},
                "case_id_hash": "existing-row-hash",
                "case_id_hashes": ["existing-row-plural-hash"],
                "keep": "row",
            }
        ],
        provenance={
            "case_id": raw_ids[4],
            "case_ids": [raw_ids[5]],
            "nested": {"case_id": raw_ids[5], "keep": "provenance"},
            "keep": {"status": "fixture"},
        },
    )

    path = serialize_benchmark(tmp_path, result)

    payload = json.loads(path.read_text(encoding="utf-8"))

    def assert_safe(value):
        if isinstance(value, dict):
            assert "case_id" not in value
            assert "case_ids" not in value
            for item in value.values():
                assert_safe(item)
        elif isinstance(value, list):
            for item in value:
                assert_safe(item)

    assert_safe(payload)
    text = path.read_text(encoding="utf-8")
    assert all(raw_id not in text for raw_id in raw_ids)
    assert payload["summary"]["case_id_hash"] == hash_case_id(raw_ids[0])
    assert payload["summary"]["case_id_hashes"] == [hash_case_id(raw_ids[1])]
    assert payload["summary"]["nested"]["case_id_hash"] == hash_case_id(raw_ids[1])
    assert payload["summary"]["keep"] == {"count": 2}
    assert payload["rows"][0]["case_id_hash"] == hash_case_id(raw_ids[2])
    assert payload["rows"][0]["case_id_hashes"] == [hash_case_id(raw_ids[3])]
    assert payload["rows"][0]["keep"] == "row"
    assert payload["provenance"]["case_id_hash"] == hash_case_id(raw_ids[4])
    assert payload["provenance"]["case_id_hashes"] == [hash_case_id(raw_ids[5])]
    assert payload["provenance"]["nested"]["case_id_hash"] == hash_case_id(raw_ids[5])
    assert payload["provenance"]["keep"] == {"status": "fixture"}


def test_serialize_benchmark_redacts_evaluator_aliases_in_nested_provenance(tmp_path):
    raw_ids = [
        "CASE-ALIAS-001",
        "CASE-ALIAS-002",
        "BraTS-ALIAS-003",
        "BraTS-ALIAS-004",
        "patient-ALIAS-005",
        "patient-ALIAS-006",
        "CASE-ALIAS-007",
    ]
    result = BenchmarkResult(
        summary={
            "id": raw_ids[0],
            "ids": [raw_ids[1]],
            "nested": {"caseId": raw_ids[2], "caseIds": (raw_ids[3],)},
        },
        rows=[],
        provenance={
            "caseId": raw_ids[6],
            "nested": {
                "id": raw_ids[4],
                "ids": [raw_ids[5]],
                "error": {"caseIds": [raw_ids[6]]},
            },
        },
    )

    path = serialize_benchmark(tmp_path, result)
    payload = json.loads(path.read_text(encoding="utf-8"))
    text = path.read_text(encoding="utf-8")

    def assert_safe(value):
        if isinstance(value, dict):
            for key in ("case_id", "case_ids", "caseId", "caseIds", "id", "ids"):
                assert key not in value
            for item in value.values():
                assert_safe(item)
        elif isinstance(value, list):
            for item in value:
                assert_safe(item)

    assert_safe(payload)
    assert all(raw_id not in text for raw_id in raw_ids)
    assert payload["summary"]["case_id_hash"] == hash_case_id(raw_ids[0])
    assert payload["summary"]["case_id_hashes"] == [hash_case_id(raw_ids[1])]
    assert payload["summary"]["nested"]["case_id_hash"] == hash_case_id(raw_ids[2])
    assert payload["summary"]["nested"]["case_id_hashes"] == [hash_case_id(raw_ids[3])]
    assert payload["provenance"]["case_id_hash"] == hash_case_id(raw_ids[6])
    assert payload["provenance"]["nested"]["case_id_hash"] == hash_case_id(raw_ids[4])
    assert payload["provenance"]["nested"]["case_id_hashes"] == [
        hash_case_id(raw_ids[5])
    ]
    assert payload["provenance"]["nested"]["error"]["case_id_hashes"] == [
        hash_case_id(raw_ids[6])
    ]


def test_serialize_benchmark_redacts_aliases_in_nested_rows_and_preserves_input(
    tmp_path,
):
    raw_id = "BraTS-BENCHMARK-008"
    rows = [{"nested": {"ids": [raw_id]}, "id": raw_id}]
    result = BenchmarkResult(summary={}, rows=rows, provenance={})

    serialize_benchmark(tmp_path, result)

    assert rows == [{"nested": {"ids": [raw_id]}, "id": raw_id}]
    payload = json.loads((tmp_path / "benchmark.json").read_text(encoding="utf-8"))
    assert payload["rows"] == [
        {
            "nested": {"case_id_hashes": [hash_case_id(raw_id)]},
            "case_id_hash": hash_case_id(raw_id),
        }
    ]


def test_serialize_benchmark_canonicalizes_supplied_case_hashes_and_aliases(tmp_path):
    raw_case_id = "CASE001"
    other_raw_case_id = "CASE002"
    supplied_valid_hash = hash_case_id("approved-case")
    result = BenchmarkResult(
        summary={},
        rows=[
            {
                "case_id": raw_case_id,
                "case_id_hash": hash_case_id("wrong-case"),
            },
            {
                "caseId": other_raw_case_id,
                "case_id_hash": supplied_valid_hash,
            },
            {
                "case_id_hash": "opaque-not-a-hash",
                "case_id_hashes": ["also-not-a-hash", supplied_valid_hash],
            },
        ],
        provenance={},
    )

    path = serialize_benchmark(tmp_path, result)
    payload = json.loads(path.read_text(encoding="utf-8"))

    first, second, third = payload["rows"]
    assert first["case_id_hash"] == hash_case_id(raw_case_id)
    assert second["case_id_hash"] == hash_case_id(other_raw_case_id)
    assert re.fullmatch(r"[0-9a-f]{16}", third["case_id_hash"])
    assert all(
        re.fullmatch(r"[0-9a-f]{16}", value)
        for value in third["case_id_hashes"]
    )
    assert raw_case_id not in path.read_text(encoding="utf-8")
    assert other_raw_case_id not in path.read_text(encoding="utf-8")


def test_serialize_benchmark_redacts_paths_and_normalizes_generic_identifier_context(
    tmp_path,
):
    valid_hash = hash_case_id("valid-case")
    result = BenchmarkResult(
        summary={
            "source_checkpoint": "C:/private/SUBJECT_001/best.pt",
            "data_root": "C:/data/SUBJECT_001",
            "subject_id": "SUBJECT_001",
            "source_artifact": "entity/project/model:v1",
            "project_name": "token-mixer-placement-matters",
        },
        rows=[
            {
                "case_id_hash": "SUBJECT_001",
                "case_id_hashes": [valid_hash, "SUBJECT_002"],
            }
        ],
        provenance={"checkpoint_path": "C:/private/SUBJECT_001/best.pt"},
    )

    path = serialize_benchmark(tmp_path, result)
    payload = json.loads(path.read_text(encoding="utf-8"))
    text = path.read_text(encoding="utf-8")

    assert "C:/private/SUBJECT_001/best.pt" not in text
    assert "C:/data/SUBJECT_001" not in text
    assert "SUBJECT_001" not in text
    assert payload["summary"]["source_artifact"] == "entity/project/model:v1"
    assert payload["summary"]["project_name"] == "token-mixer-placement-matters"
    assert payload["summary"]["subject_id"] == hash_case_id("SUBJECT_001")
    assert payload["rows"][0]["case_id_hash"] == hash_case_id("SUBJECT_001")
    assert payload["rows"][0]["case_id_hashes"] == [
        valid_hash,
        hash_case_id("SUBJECT_002"),
    ]


def test_serialize_benchmark_redacts_paths_in_namespaced_keys_and_patient_context(
    tmp_path,
):
    raw_path = r"C:\private\user\file.pt"
    raw_case = "CASE001"
    result = BenchmarkResult(
        summary={
            f"checkpoint/{raw_path}": raw_path,
            "train/loss": 0.5,
            "val/dice": 0.8,
            "patient": "A17",
            "patient_name": "A17",
            "case_note": raw_case,
            "source_artifact": "entity/project/model:v1",
        },
        rows=[],
        provenance={f"nested/path={raw_path}": raw_path},
    )

    path = serialize_benchmark(tmp_path, result)
    payload = json.loads(path.read_text(encoding="utf-8"))
    text = path.read_text(encoding="utf-8")

    assert raw_path not in text
    assert raw_case not in text
    assert payload["summary"]["train/loss"] == 0.5
    assert payload["summary"]["val/dice"] == 0.8
    assert payload["summary"]["patient"] == hash_case_id("A17")
    assert payload["summary"]["patient_name"] == hash_case_id("A17")
    assert payload["summary"]["source_artifact"] == "entity/project/model:v1"
    assert any("[REDACTED_PATH]" in key for key in payload["summary"])
    assert any("[REDACTED_PATH]" in key for key in payload["provenance"])


def test_serialize_benchmark_honors_custom_json_filename(tmp_path):
    result = BenchmarkResult(summary={}, rows=[], provenance={})

    path = serialize_benchmark(tmp_path, result, output_name="custom-benchmark.json")

    assert path == tmp_path / "custom-benchmark.json"
    assert path.is_file()
    assert not (tmp_path / "benchmark.json").exists()


@pytest.mark.parametrize(
    "output_name",
    [
        "",
        "nested/benchmark.json",
        "../benchmark.json",
        "benchmark.txt",
        "C:escape.json",
        "safe\x00.json",
        "safe\x01.json",
        "CON.json",
        "PRN.txt",
        "AUX.data.json",
        "NUL.json",
        "COM9.json",
        "LPT1.txt",
        "benchmark.json ",
        "benchmark.json.",
        "unsafe:name.json",
    ],
)
def test_serialize_benchmark_rejects_unsafe_output_filename(tmp_path, output_name):
    result = BenchmarkResult(summary={}, rows=[], provenance={})

    with pytest.raises(ValueError, match="output_name"):
        serialize_benchmark(tmp_path, result, output_name=output_name)


def test_run_model_protocol_combines_measurements_and_preserves_fixed_input(
    monkeypatch: pytest.MonkeyPatch,
):
    inputs = torch.ones(1, 2, 4, 4)
    calls: list[tuple[str, object]] = []

    class FakeSampler:
        def __init__(self, device_index, interval_seconds):
            calls.append(("sampler_init", (device_index, interval_seconds)))

        def start(self):
            calls.append(("sampler_start", None))
            return self

        def stop(self):
            calls.append(("sampler_stop", None))
            return type(
                "Stats",
                (),
                {
                    "average_watts": 100.0,
                    "max_watts": 120.0,
                    "joules": 2.5,
                    "samples": 3,
                    "interval_seconds": 0.01,
                    "status": "ok",
                },
            )()

    def fake_static(model, *, inputs, checkpoint_path=None):
        del model, checkpoint_path
        calls.append(("static", inputs))
        return {
            "parameters": 4,
            "trainable_parameters": 3,
            "macs": 10,
            "flops": 20,
            "mac_tool": "thop",
            "mac_tool_version": "fixture",
            "mac_convention": "mac",
            "mac_status": "ok",
            "flop_tool": "fvcore",
            "flop_tool_version": "fixture",
            "flop_convention": "flop",
            "flop_status": "ok",
            "unsupported_ops": {},
            "checkpoint_bytes": None,
        }

    def fake_forward(model, inputs, *, warmup_iterations, repetitions, protocol):
        del model
        calls.append(("forward", inputs))
        return {
            "status": "ok",
            "latency_mean_ms": 2.0,
            "latency_median_ms": 2.0,
            "latency_p95_ms": 2.0,
            "latency_std_ms": 0.0,
            "peak_memory_allocated_gb": None,
            "peak_memory_reserved_gb": None,
            "batch_size": 1,
            "input_shape": [1, 2, 4, 4],
            "warmup_iterations": warmup_iterations,
            "repetitions": repetitions,
            "protocol": protocol,
        }

    def fake_sweep(model, factory, batch_sizes, *, warmup_iterations, repetitions, protocol):
        del model, warmup_iterations, repetitions
        calls.append(("sweep_input", factory(2)))
        return {
            "status": "oom",
            "sweep_status": "oom",
            "rows": [{"status": "ok", "batch_size": 1}],
            "largest_passing_batch": 1,
            "first_failing_batch": 2,
            "batch_sizes": list(batch_sizes),
            "protocol": protocol,
        }

    monkeypatch.setattr(benchmark, "NvmlPowerSampler", FakeSampler)
    monkeypatch.setattr(benchmark, "static_model_cost", fake_static)
    monkeypatch.setattr(benchmark, "measure_forward", fake_forward)
    monkeypatch.setattr(benchmark, "measure_batch_sweep", fake_sweep)

    output = run_model_protocol(
        nn.Identity(),
        inputs,
        protocol="fixture",
        warmup_iterations=20,
        repetitions=7,
        batch_sizes=[1, 2],
    )

    assert output["model/parameters"] == 4
    assert output["model/macs"] == 10
    assert output["model/flops"] == 20
    assert output["inference/latency_mean_ms"] == pytest.approx(2.0)
    assert output["inference/warmup_iterations"] == 20
    assert output["inference/repetitions"] == 7
    assert output["inference/protocol"] == "fixture"
    assert output["power/energy_joules"] == pytest.approx(2.5)
    assert output["power/sample_interval_ms"] == pytest.approx(10.0)
    assert output["inference/largest_passing_batch"] == 1
    assert output["inference/first_failing_batch"] == 2
    assert calls[0] == ("static", inputs)
    assert calls[1] == ("sampler_init", (0, 0.01))
    assert calls[2] == ("sampler_start", None)
    assert calls[3] == ("forward", inputs)


@pytest.mark.parametrize("failure_stage", ["start", "stop"])
def test_run_model_protocol_preserves_power_failure_provenance_in_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    failure_stage: str,
):
    failure_reason = "NVML failure: " + ("detail " * 100)
    device_index = 6

    class FailingSampler:
        def __init__(self, device_index, interval_seconds):
            del interval_seconds
            self.device_index = device_index
            self._nvml_version = "fixture-nvml"

        def start(self):
            if failure_stage == "start":
                raise RuntimeError(failure_reason)
            return self

        def stop(self):
            if failure_stage == "stop":
                raise PermissionError(failure_reason)
            raise AssertionError("successful stop is not expected")

    monkeypatch.setattr(benchmark, "NvmlPowerSampler", FailingSampler)
    monkeypatch.setattr(
        benchmark,
        "static_model_cost",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        benchmark,
        "measure_forward",
        lambda *_args, **_kwargs: {
            "latency_mean_ms": 1.0,
            "batch_size": 1,
            "input_shape": [1, 1],
        },
    )
    monkeypatch.setattr(
        benchmark,
        "measure_batch_sweep",
        lambda *_args, **_kwargs: {"status": "ok", "rows": []},
    )

    output = run_model_protocol(
        nn.Identity(),
        torch.ones(1, 1),
        protocol="fixture",
        warmup_iterations=0,
        repetitions=1,
        batch_sizes=[1],
        power_device_index=device_index,
    )

    assert output["power/status"] == "unavailable"
    assert output["power/device_index"] == device_index
    assert output["power/nvml_version"] == (
        "fixture-nvml" if failure_stage == "stop" else None
    )
    assert output["power/reason"].startswith("NVML failure:")
    assert len(output["power/reason"]) <= 240

    path = benchmark.serialize_benchmark(
        tmp_path,
        BenchmarkResult(summary=output, rows=[], provenance={}),
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["summary"]["power/device_index"] == device_index
    assert payload["summary"]["power/nvml_version"] == (
        "fixture-nvml" if failure_stage == "stop" else None
    )
    assert payload["summary"]["power/reason"] == output["power/reason"]


def test_run_model_protocol_accepts_minimal_static_fixture_and_counts_spatial_voxels(
    monkeypatch: pytest.MonkeyPatch,
):
    inputs = torch.ones(2, 3, 4, 5)
    calls: list[object] = []

    class FakeSampler:
        def __init__(self, device_index, interval_seconds):
            del device_index, interval_seconds

        def start(self):
            return self

        def stop(self):
            return {"status": "unavailable", "samples": 0}

    def fake_static(model, *, inputs):
        del model
        calls.append(inputs)
        return {"parameters": 1, "trainable_parameters": 1}

    def fake_forward(model, inputs, *, warmup_iterations, repetitions, protocol):
        del model, inputs, warmup_iterations, repetitions, protocol
        return {
            "latency_mean_ms": 2.0,
            "batch_size": 2,
            "input_shape": [2, 3, 4, 5],
        }

    monkeypatch.setattr(benchmark, "NvmlPowerSampler", FakeSampler)
    monkeypatch.setattr(benchmark, "static_model_cost", fake_static)
    monkeypatch.setattr(benchmark, "measure_forward", fake_forward)
    monkeypatch.setattr(
        benchmark,
        "measure_batch_sweep",
        lambda *_args, **_kwargs: {"status": "ok", "rows": []},
    )

    output = run_model_protocol(
        nn.Identity(),
        inputs,
        protocol="fixture",
        warmup_iterations=0,
        repetitions=1,
        batch_sizes=[1],
    )

    assert calls == [inputs]
    assert output["inference/throughput_samples_per_second"] == pytest.approx(1000.0)
    assert output["inference/throughput_voxels_per_second"] == pytest.approx(20_000.0)


def test_run_model_protocol_derives_nvml_index_from_prepared_cuda_input(
    monkeypatch: pytest.MonkeyPatch,
):
    class PreparedCudaTensor:
        device = torch.device("cuda:3")
        ndim = 4
        shape = (1, 2, 4, 4)

    power_indices: list[int] = []

    class FakeSampler:
        def __init__(self, device_index, interval_seconds):
            del interval_seconds
            power_indices.append(device_index)

        def start(self):
            return self

        def stop(self):
            return {"status": "unavailable", "samples": 0}

    monkeypatch.setattr(benchmark, "_first_tensor", lambda _value: PreparedCudaTensor())
    monkeypatch.setattr(benchmark, "NvmlPowerSampler", FakeSampler)
    monkeypatch.setattr(
        benchmark,
        "static_model_cost",
        lambda _model, *, inputs: {"parameters": 1, "trainable_parameters": 1},
    )
    monkeypatch.setattr(
        benchmark,
        "measure_forward",
        lambda *_args, **_kwargs: {
            "status": "ok",
            "latency_mean_ms": 1.0,
            "batch_size": 1,
            "input_shape": [1, 2, 4, 4],
        },
    )
    monkeypatch.setattr(
        benchmark,
        "measure_batch_sweep",
        lambda *_args, **_kwargs: {"status": "ok", "rows": []},
    )

    model = nn.Identity()
    inputs = object()
    benchmark.run_model_protocol(
        model,
        inputs,
        protocol="fixture",
        warmup_iterations=0,
        repetitions=1,
        batch_sizes=[1],
    )
    benchmark.run_model_protocol(
        model,
        inputs,
        protocol="fixture",
        warmup_iterations=0,
        repetitions=1,
        batch_sizes=[1],
        power_device_index=7,
    )

    assert power_indices == [3, 7]


@pytest.mark.parametrize(
    ("visible_devices", "expected_physical_index"),
    [("2,5", 5), ("GPU-aaa,GPU-bbb", None)],
)
def test_run_model_protocol_resolves_cuda_visible_devices_safely(
    monkeypatch: pytest.MonkeyPatch,
    visible_devices: str,
    expected_physical_index: int | None,
):
    class PreparedCudaTensor:
        device = torch.device("cuda:1")
        ndim = 4
        shape = (1, 2, 4, 4)

    observed_indices: list[int] = []

    class FakeSampler:
        def __init__(self, device_index, interval_seconds):
            del interval_seconds
            observed_indices.append(device_index)
            self.device_index = device_index

        def start(self):
            return self

        def stop(self):
            return {
                "status": "ok",
                "samples": 1,
                "device_index": self.device_index,
                "nvml_version": "fixture",
            }

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible_devices)
    monkeypatch.setattr(benchmark, "_first_tensor", lambda _value: PreparedCudaTensor())
    monkeypatch.setattr(benchmark, "NvmlPowerSampler", FakeSampler)
    monkeypatch.setattr(
        benchmark,
        "static_model_cost",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        benchmark,
        "measure_forward",
        lambda *_args, **_kwargs: {
            "latency_mean_ms": 1.0,
            "batch_size": 1,
            "input_shape": [1, 2, 4, 4],
        },
    )
    monkeypatch.setattr(
        benchmark,
        "measure_batch_sweep",
        lambda *_args, **_kwargs: {"status": "ok", "rows": []},
    )

    output = run_model_protocol(
        nn.Identity(),
        object(),
        protocol="fixture",
        warmup_iterations=0,
        repetitions=1,
        batch_sizes=[1],
    )

    if expected_physical_index is None:
        assert observed_indices == []
        assert output["power/status"] == "unavailable"
        assert output["power/reason"]
        assert output["power/device_mapping_status"] == "unavailable"
    else:
        assert observed_indices == [expected_physical_index]
        assert output["power/status"] == "ok"
        assert output["power/device_index"] == expected_physical_index
        assert output["power/device_mapping_status"] == "resolved"
    assert output["power/logical_device_index"] == 1
    assert output["power/device_mapping"]["visible_devices"] == visible_devices.split(",")

    override = run_model_protocol(
        nn.Identity(),
        object(),
        protocol="fixture",
        warmup_iterations=0,
        repetitions=1,
        batch_sizes=[1],
        power_device_index=7,
    )
    assert observed_indices[-1] == 7
    assert override["power/device_mapping_status"] == "explicit_override"
    assert override["power/device_index"] == 7


def test_run_model_protocol_uses_real_static_counter_and_reports_unsupported_ops():
    class UnsupportedModel(nn.Module):
        def forward(self, value):
            return torch.sin(value)

    output = run_model_protocol(
        UnsupportedModel(),
        torch.ones(1, 2, 4, 4),
        protocol="fixture",
        warmup_iterations=0,
        repetitions=1,
        batch_sizes=[1],
    )

    assert output["model/mac_tool"] == "thop"
    assert output["model/flop_tool"] == "fvcore"
    assert output["model/mac_convention"]
    assert output["model/flop_convention"]
    assert output["model/unsupported_ops"]
    assert output["model/flop_status"] == "partial"


def test_run_model_protocol_preserves_oom_boundary():
    class OOMModel(nn.Module):
        def forward(self, value):
            if value.shape[0] == 2:
                raise RuntimeError("CUDA out of memory")
            return value

    output = run_model_protocol(
        OOMModel(),
        torch.ones(1, 2),
        protocol="fixture",
        warmup_iterations=0,
        repetitions=1,
        batch_sizes=[1, 2, 4],
    )

    assert output["inference/sweep_status"] == "oom"
    assert output["inference/largest_passing_batch"] == 1
    assert output["inference/first_failing_batch"] == 2
    assert output["inference/sweep_rows"][-1]["status"] == "oom"


def test_run_model_protocol_records_precision_timing_hardware_and_software_metadata(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        benchmark,
        "static_model_cost",
        lambda *_args, **_kwargs: {
            "parameters": 1,
            "trainable_parameters": 1,
            "macs": 1,
            "flops": 2,
            "mac_status": "ok",
            "flop_status": "ok",
        },
    )
    monkeypatch.setattr(
        benchmark,
        "measure_forward",
        lambda *_args, **_kwargs: {
            "status": "ok",
            "latency_mean_ms": 1.0,
            "batch_size": 1,
            "input_shape": [1, 1],
        },
    )
    monkeypatch.setattr(
        benchmark,
        "measure_batch_sweep",
        lambda *_args, **_kwargs: {"status": "ok", "rows": []},
    )

    output = run_model_protocol(
        nn.Identity(),
        torch.ones(1, 1),
        protocol="fixture",
        warmup_iterations=0,
        repetitions=1,
        batch_sizes=[1],
    )

    assert output["model/precision"] == "float32"
    assert output["model/timing_boundary"] == "model_forward_only"
    assert output["model/hardware"]["device"] == "cpu"
    assert output["model/software"]["pytorch"]
    assert output["model/software"]["python"]


def test_run_model_protocol_forwards_efficiency_controls_and_gates_power(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = {}

    def fake_static(model, *, inputs, checkpoint_path=None, **kwargs):
        del model, inputs, checkpoint_path
        calls["static"] = kwargs
        return {
            "parameters": 1,
            "trainable_parameters": 1,
            "macs": None,
            "flops": None,
            "mac_status": "unavailable",
            "flop_status": "unavailable",
        }

    def fake_forward(model, inputs, *, warmup_iterations, repetitions, protocol):
        del model, inputs
        return {
            "status": "ok",
            "latency_mean_ms": 1.0,
            "batch_size": 1,
            "input_shape": [1, 1],
            "warmup_iterations": warmup_iterations,
            "repetitions": repetitions,
            "protocol": protocol,
        }

    def fake_sweep(
        model,
        input_factory,
        batch_sizes,
        *,
        warmup_iterations,
        repetitions,
        protocol,
    ):
        del model, input_factory, warmup_iterations, repetitions
        return {"status": "ok", "rows": [], "batch_sizes": list(batch_sizes), "protocol": protocol}

    monkeypatch.setattr(benchmark, "static_model_cost", fake_static)
    monkeypatch.setattr(benchmark, "measure_forward", fake_forward)
    monkeypatch.setattr(benchmark, "measure_batch_sweep", fake_sweep)

    class UnexpectedSampler:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("disabled power measurement initialized NVML")

    monkeypatch.setattr(benchmark, "NvmlPowerSampler", UnexpectedSampler)

    output = run_model_protocol(
        nn.Identity(),
        torch.ones(1, 1),
        protocol="fixture",
        warmup_iterations=0,
        repetitions=1,
        batch_sizes=[1],
        power_enabled=False,
        profiler_enabled=True,
        mac_tool="thop",
        flop_tool="fvcore",
    )

    assert calls["static"] == {"mac_tool": "thop", "flop_tool": "fvcore"}
    assert output["power/status"] == "disabled"
