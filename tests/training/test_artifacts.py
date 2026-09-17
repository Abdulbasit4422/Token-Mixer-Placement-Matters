from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from token_mixer import __version__
from token_mixer.evaluation.benchmark import hash_case_id
from token_mixer.privacy import safe_error_message
from token_mixer.training.artifacts import (
    _json_safe,
    write_failed_run_artifact,
    write_run_artifacts,
)
from token_mixer.training.engine import FitResult


def _assert_no_raw_case_fields(value):
    if isinstance(value, dict):
        for key in ("case_id", "case_ids", "caseId", "caseIds", "id", "ids"):
            assert key not in value
        for item in value.values():
            _assert_no_raw_case_fields(item)
    elif isinstance(value, list):
        for item in value:
            _assert_no_raw_case_fields(item)


def test_write_run_artifacts_persists_metrics_and_provenance(tmp_path: Path):
    result = FitResult(
        best_metric=0.75,
        best_epoch=2,
        history=[{"epoch": 1, "mean_dice": 0.5}],
        test_metrics={"mean_dice": 0.8, "mean_hd95": float("nan")},
        metadata={
            "architecture": "fixture",
            "execution_device": "cpu",
            "manifest_hash": "manifest-sha256",
        },
    )
    config = {
        "runtime": "local",
        "experiment": {"name": "mod_a"},
        "tracking": {"enabled": False, "mode": "disabled"},
    }

    paths = write_run_artifacts(tmp_path, config, result)

    assert paths == {
        "metrics": tmp_path / "metrics.json",
        "provenance": tmp_path / "provenance.json",
    }
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["schema_version"] == 2
    assert metrics["best_epoch"] == 2
    assert metrics["best_metric"] == 0.75
    assert metrics["history"] == [{"epoch": 1, "mean_dice": 0.5}]
    assert metrics["test_metrics"] == {"mean_dice": 0.8, "mean_hd95": None}

    provenance = json.loads((tmp_path / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["code_version"] == __version__
    assert provenance["experiment"] == "mod_a"
    assert provenance["runtime"] == "local"
    assert provenance["metadata"] == result.metadata
    assert provenance["tracking"] == {"enabled": False, "mode": "disabled"}


def test_write_run_artifacts_rejects_non_fit_result(tmp_path: Path):
    try:
        write_run_artifacts(tmp_path, {}, object())
    except TypeError as exc:
        assert "FitResult" in str(exc)
    else:
        raise AssertionError("non-FitResult completion must be rejected")


def test_write_run_artifacts_serializes_efficiency_and_timing_metadata(tmp_path: Path):
    result = FitResult(
        0.5,
        1,
        [],
        {"mean_dice": 0.5},
        {
            "architecture": "fixture",
            "timing_scope": "process_segment",
            "cumulative_train_seconds": 1.25,
            "power/energy_joules": 12.5,
            "power/status": "ok",
            "train/peak_memory_allocated_gb": 0.75,
        },
    )

    paths = write_run_artifacts(tmp_path, {}, result)
    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))

    assert metrics["timing"]["timing_scope"] == "process_segment"
    assert metrics["timing"]["cumulative_train_seconds"] == pytest.approx(1.25)
    assert metrics["efficiency"]["power/energy_joules"] == pytest.approx(12.5)
    assert metrics["efficiency"]["train/peak_memory_allocated_gb"] == pytest.approx(0.75)
    assert provenance["schema_version"] == 2
    assert provenance["timing"]["cumulative_train_seconds"] == pytest.approx(1.25)


def test_write_failed_run_artifact_preserves_failure_without_metrics(tmp_path: Path):
    path = write_failed_run_artifact(tmp_path, {}, ValueError("fixture failure"))

    provenance = json.loads(path.read_text(encoding="utf-8"))
    assert provenance["schema_version"] == 2
    assert provenance["status"] == "failed"
    assert provenance["error"] == "fixture failure"
    assert not (tmp_path / "metrics.json").exists()


def test_json_safe_recursively_serializes_multi_element_tensors_and_arrays():
    value = _json_safe(
        {
            "tensor": torch.tensor([[1, 2], [3, 4]]),
            "array": np.array([[0.25, 0.5], [0.75, 1.0]], dtype=np.float32),
            "nested": [torch.tensor([5, 6]), np.array([7, 8])],
        }
    )

    assert value == {
        "tensor": [[1, 2], [3, 4]],
        "array": [[pytest.approx(0.25), pytest.approx(0.5)], [pytest.approx(0.75), pytest.approx(1.0)]],
        "nested": [[5, 6], [7, 8]],
    }


def test_failed_run_updates_existing_provenance_without_replacing_metrics(
    tmp_path: Path,
):
    result = FitResult(
        0.75,
        2,
        [{"epoch": 1, "mean_dice": 0.5}],
        {"mean_dice": 0.8},
        {
            "architecture": "fixture",
            "protocol": "native_3d_full_volume",
            "code_version": "fixture-v1",
        },
    )
    config = {"experiment": {"name": "mod_a"}}
    write_run_artifacts(tmp_path, config, result)
    before_metrics = json.loads((tmp_path / "metrics.json").read_text())

    write_failed_run_artifact(
        tmp_path,
        config,
        RuntimeError("tracker upload failed"),
        {"protocol": "native_3d_full_volume"},
    )

    after_metrics = json.loads((tmp_path / "metrics.json").read_text())
    provenance = json.loads((tmp_path / "provenance.json").read_text())
    assert after_metrics == before_metrics
    assert provenance["status"] == "failed"
    assert provenance["partial"] is True
    assert provenance["error"] == "tracker upload failed"
    assert provenance["architecture"] == "fixture"
    assert provenance["code_version"] == "fixture-v1"


def test_run_artifacts_redact_case_identifiers_recursively_and_preserve_hashes(
    tmp_path: Path,
):
    raw_ids = ["PATIENT-001", "PATIENT-002", "PATIENT-003", "PATIENT-004"]
    result = FitResult(
        0.75,
        1,
        [],
        {},
        {
            "case_id": raw_ids[0],
            "case_ids": raw_ids[1:3],
            "nested": {
                "case_id": raw_ids[3],
                "case_ids": (raw_ids[0],),
                "case_id_hash": "existing-hash",
                "case_id_hashes": ["existing-plural-hash"],
                "keep": {"kind": "fixture"},
            },
        },
    )

    paths = write_run_artifacts(tmp_path, {}, result)

    assert result.metadata["case_id"] == raw_ids[0]
    for path in paths.values():
        payload = json.loads(path.read_text(encoding="utf-8"))
        _assert_no_raw_case_fields(payload)
        text = path.read_text(encoding="utf-8")
        assert all(raw_id not in text for raw_id in raw_ids)
        metadata = payload["metadata"]
        assert metadata["case_id_hash"] == hash_case_id(raw_ids[0])
        assert metadata["case_id_hashes"] == [
            hash_case_id(raw_ids[1]),
            hash_case_id(raw_ids[2]),
        ]
        assert metadata["nested"]["case_id_hash"] == "existing-hash"
        assert metadata["nested"]["case_id_hashes"] == ["existing-plural-hash"]
        assert metadata["nested"]["keep"] == {"kind": "fixture"}


def test_failed_run_artifact_redacts_case_identifiers_recursively(tmp_path: Path):
    raw_ids = ["FAILED-PATIENT-001", "FAILED-PATIENT-002"]

    path = write_failed_run_artifact(
        tmp_path,
        {},
        RuntimeError("fixture failure"),
        {
            "case_id": raw_ids[0],
            "case_ids": raw_ids,
            "nested": {"case_id": raw_ids[1]},
            "keep": "intact",
        },
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    _assert_no_raw_case_fields(payload)
    text = path.read_text(encoding="utf-8")
    assert all(raw_id not in text for raw_id in raw_ids)
    assert payload["metadata"]["case_id_hash"] == hash_case_id(raw_ids[0])
    assert payload["metadata"]["case_id_hashes"] == [
        hash_case_id(raw_ids[0]),
        hash_case_id(raw_ids[1]),
    ]
    assert payload["metadata"]["nested"]["case_id_hash"] == hash_case_id(raw_ids[1])
    assert payload["metadata"]["keep"] == "intact"


def test_safe_error_message_preserves_ordinary_fixture_messages():
    assert safe_error_message(ValueError("fixture failure")) == "fixture failure"


def test_failed_run_artifact_redacts_aliases_and_sensitive_error_fragments(
    tmp_path: Path,
):
    raw_ids = ["CASE-ORACLE-001", "BraTS-ORACLE-002", "patient-ORACLE-003"]
    error = RuntimeError(
        "unable to read CASE-ERROR-001 for patient-ORACLE-003 at "
        "C:\\private\\BraTS-ORACLE-002\\scan.nii.gz; api_key=secret-value"
    )

    path = write_failed_run_artifact(
        tmp_path,
        {},
        error,
        {
            "caseIds": raw_ids,
            "nested": {"id": raw_ids[0], "keep": "fixture"},
        },
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    text = path.read_text(encoding="utf-8")
    _assert_no_raw_case_fields(payload)
    assert all(raw_id not in text for raw_id in raw_ids)
    assert "RuntimeError" in payload["error"]
    assert "[REDACTED_CASE_ID]" in payload["error"]
    assert "[REDACTED_PATH]" in payload["error"]
    assert "[REDACTED_SECRET]" in payload["error"]
    assert payload["metadata"]["case_id_hashes"] == [
        hash_case_id(raw_id) for raw_id in raw_ids
    ]
    assert payload["metadata"]["nested"]["case_id_hash"] == hash_case_id(raw_ids[0])
    assert payload["metadata"]["nested"]["keep"] == "fixture"


def test_safe_error_message_bounds_untrusted_exception_text_and_keeps_type():
    error = ValueError("CASE-BOUND-001 " + ("detail " * 500))

    message = safe_error_message(error)

    assert message.startswith("ValueError: ")
    assert "CASE-BOUND-001" not in message
    assert len(message) <= 512
