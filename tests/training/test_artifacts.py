from __future__ import annotations

import json
from pathlib import Path

from token_mixer import __version__
from token_mixer.training.artifacts import write_run_artifacts
from token_mixer.training.engine import FitResult


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
    assert metrics == {
        "best_epoch": 2,
        "best_metric": 0.75,
        "history": [{"epoch": 1, "mean_dice": 0.5}],
        "test_metrics": {"mean_dice": 0.8, "mean_hd95": None},
    }

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
