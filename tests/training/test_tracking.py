import builtins
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import token_mixer.training.tracking as tracking_module
from token_mixer.privacy import hash_case_id, redact_case_identifiers
from token_mixer.training.tracking import create_tracker


def test_disabled_tracker_does_not_import_or_initialize_wandb(monkeypatch):
    imports = []
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "wandb":
            imports.append(name)
            raise AssertionError("disabled tracking imported wandb")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    tracker = create_tracker({"enabled": False}, {})

    tracker.log({"loss": 0.5}, step=1)
    tracker.log_summary({"val_loss": 0.4})
    tracker.log_table("metrics", ["epoch"], [[1]])
    tracker.log_images({"preview": object()}, step=1)
    tracker.define_metric("train/loss")
    assert tracker.log_artifact("model", {}) is None
    assert tracker.run_id is None
    with pytest.raises(RuntimeError, match="tracking is disabled"):
        tracker.restore_artifact("model:v1", Path("restore"))
    tracker.finish()

    assert imports == []


def test_disabled_mode_is_noop_even_when_tracking_is_enabled(monkeypatch):
    def fail_import(name, *args, **kwargs):
        if name == "wandb":
            raise AssertionError("disabled mode imported wandb")
        return original_import(name, *args, **kwargs)

    original_import = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", fail_import)
    tracker = create_tracker({"enabled": True, "mode": "disabled"}, {})

    tracker.finish()


def test_offline_tracker_initializes_lazily_and_forwards_logs(monkeypatch, tmp_path):
    calls = []

    class FakeRun:
        def log(self, metrics, step=None):
            calls.append(("log", metrics, step))

        def finish(self):
            calls.append(("finish",))

    def init(**kwargs):
        calls.append(("init", kwargs))
        return FakeRun()

    fake_wandb = SimpleNamespace(init=init)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    config = {
        "enabled": True,
        "mode": "offline",
        "project": "fixture-project",
        "entity": "fixture-entity",
        "run_name": "fixture-run",
        "directory": str(tmp_path),
    }
    run_config = {"seed": 42, "phase": "phase1"}

    tracker = create_tracker(config, run_config)
    tracker.log({"train/loss": 0.5}, step=3)
    tracker.log_summary({"val/dice": 0.8})
    tracker.finish()

    assert calls[0] == (
        "init",
        {
            "project": "fixture-project",
            "entity": "fixture-entity",
            "config": run_config,
            "mode": "offline",
            "dir": str(tmp_path),
            "name": "fixture-run",
        },
    )
    assert calls[1] == ("log", {"train/loss": 0.5}, 3)
    assert calls[2] == ("log", {"val/dice": 0.8}, None)
    assert calls[3] == ("finish",)


def test_online_tracker_requires_api_key_before_initializing(monkeypatch):
    init_called = False

    def init(**kwargs):
        nonlocal init_called
        init_called = True
        return object()

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init))
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setattr(tracking_module, "_wandb_netrc_has_credentials", lambda: False)

    with pytest.raises(RuntimeError, match="WANDB_API_KEY"):
        create_tracker({"enabled": True, "mode": "online"}, {})

    assert init_called is False


def test_online_tracker_accepts_netrc_without_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setattr(tracking_module, "_wandb_netrc_has_credentials", lambda: True)
    fake_run = SimpleNamespace(
        id="abc123",
        log=lambda *args, **kwargs: None,
        finish=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=lambda **_: fake_run))

    tracker = create_tracker(
        {"enabled": True, "mode": "online", "directory": str(tmp_path)},
        {"seed": 42},
    )

    assert tracker.run_id == "abc123"


def test_online_tracker_rejects_missing_api_key_and_netrc(monkeypatch):
    init_called = False

    def init(**kwargs):
        nonlocal init_called
        init_called = True
        return object()

    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setattr(tracking_module, "_wandb_netrc_has_credentials", lambda: False)
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init))

    with pytest.raises(RuntimeError) as exc_info:
        create_tracker({"enabled": True, "mode": "online"}, {})

    message = str(exc_info.value)
    assert "WANDB_API_KEY" in message
    assert ".netrc" in message
    assert "fixture-key" not in message
    assert init_called is False


def test_enabled_tracker_reports_missing_wandb_package(monkeypatch):
    imports = []
    real_import = tracking_module.importlib.import_module

    def isolated_import(name, *args, **kwargs):
        if name == "wandb":
            imports.append(name)
            raise ModuleNotFoundError("No module named 'wandb'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setenv("WANDB_API_KEY", "fixture-key")
    monkeypatch.setattr(tracking_module.importlib, "import_module", isolated_import)

    with pytest.raises(RuntimeError, match="wandb.*unavailable"):
        create_tracker({"enabled": True, "mode": "offline"}, {})

    assert imports == ["wandb"]


def test_online_tracker_initializes_with_api_key_without_network(monkeypatch, tmp_path):
    calls = []

    class FakeRun:
        def log(self, metrics, step=None):
            calls.append(("log", metrics, step))

        def finish(self):
            calls.append(("finish",))

    def init(**kwargs):
        calls.append(("init", kwargs))
        return FakeRun()

    monkeypatch.setenv("WANDB_API_KEY", "fixture-key")
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init))

    config = {
        "enabled": True,
        "mode": "online",
        "project": "fixture-project",
        "entity": "fixture-entity",
        "directory": str(tmp_path),
    }
    run_config = {"seed": 42}

    tracker = create_tracker(config, run_config)
    tracker.log({"train/loss": 0.5}, step=3)
    tracker.finish()

    assert calls == [
        (
            "init",
            {
                "project": "fixture-project",
                "entity": "fixture-entity",
                "config": run_config,
                "mode": "online",
                "dir": str(tmp_path),
            },
        ),
        ("log", {"train/loss": 0.5}, 3),
        ("finish",),
    ]


def test_wandb_init_forwards_group_job_type_and_tags(monkeypatch):
    calls = []

    class FakeRun:
        id = "run-with-metadata"

        def log(self, *_args, **_kwargs):
            return None

        def finish(self):
            return None

    def init(**kwargs):
        calls.append(kwargs)
        return FakeRun()

    monkeypatch.setenv("WANDB_API_KEY", "fixture-key")
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init))

    create_tracker(
        {
            "enabled": True,
            "mode": "online",
            "project": "fixture-project",
            "entity": "fixture-entity",
            "group": "fixture-group",
            "job_type": "train",
            "tags": ["baseline", "seed42"],
        },
        {"seed": 42},
    )

    assert calls == [
        {
            "project": "fixture-project",
            "entity": "fixture-entity",
            "config": {"seed": 42},
            "mode": "online",
            "dir": None,
            "group": "fixture-group",
            "job_type": "train",
            "tags": ["baseline", "seed42"],
        }
    ]


def test_wandb_tracker_forwards_epoch_metric_axes(monkeypatch):
    calls = []

    class FakeRun:
        def define_metric(self, name, **kwargs):
            calls.append((name, kwargs))

        def log(self, *_args, **_kwargs):
            return None

        def finish(self):
            return None

    monkeypatch.setenv("WANDB_API_KEY", "fixture-key")
    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(init=lambda **_: FakeRun()),
    )

    tracker = create_tracker({"enabled": True, "mode": "online"}, {})
    tracker.define_metric("train/loss")
    tracker.define_metric("val/dice")
    tracker.define_metric("global_step")

    assert calls == [
        ("train/loss", {"step_metric": "train/epoch"}),
        ("val/dice", {"step_metric": "val/epoch"}),
        ("global_step", {}),
    ]


def test_wandb_tracker_constructs_and_logs_table(monkeypatch):
    table_calls = []
    log_calls = []

    class FakeTable:
        def __init__(self, *, columns, data):
            table_calls.append((columns, data))
            self.value = "table"

    class FakeRun:
        def log(self, metrics, step=None):
            log_calls.append((metrics, step))

        def finish(self):
            return None

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(init=lambda **_: FakeRun(), Table=FakeTable),
    )

    tracker = create_tracker({"enabled": True, "mode": "offline"}, {})
    tracker.log_table("epoch_metrics", ("epoch", "loss"), ((1, 0.5), (2, 0.25)))

    assert table_calls == [(["epoch", "loss"], [[1, 0.5], [2, 0.25]])]
    assert log_calls[0][1] is None
    assert log_calls[0][0]["epoch_metrics"].value == "table"


def test_wandb_tracker_redacts_case_aliases_in_config_logs_summary_and_tables(
    monkeypatch,
):
    raw_ids = ["CASE-WANDB-001", "BraTS-WANDB-002", "patient-WANDB-003"]
    init_calls = []
    log_calls = []
    summary_updates = []
    table_calls = []

    class FakeSummary:
        def update(self, metrics):
            summary_updates.append(dict(metrics))

    class FakeTable:
        def __init__(self, *, columns, data):
            table_calls.append((list(columns), [list(row) for row in data]))

    class FakeRun:
        summary = FakeSummary()

        def log(self, metrics, step=None):
            log_calls.append((metrics, step))

        def finish(self):
            return None

    def init(**kwargs):
        init_calls.append(kwargs)
        return FakeRun()

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(init=init, Table=FakeTable),
    )
    run_config = {
        "caseIds": raw_ids,
        "nested": {"id": raw_ids[0]},
        "run_id": "unrelated-run-id",
    }

    tracker = create_tracker({"enabled": True, "mode": "offline"}, run_config)
    tracker.log(
        {
            "caseId": raw_ids[0],
            "ids": raw_ids[1:],
            "nested": {"case_ids": [raw_ids[2]]},
            "loss": 0.5,
        },
        step=4,
    )
    tracker.log_summary({"caseIds": raw_ids[:2], "metric": 0.8})
    tracker.log_table(
        "history",
        ("epoch", "caseIds", "id"),
        ((1, [raw_ids[0]], raw_ids[1]),),
    )

    assert init_calls[0]["config"] == {
        "case_id_hashes": [hash_case_id(case_id) for case_id in raw_ids],
        "nested": {"case_id_hash": hash_case_id(raw_ids[0])},
        "run_id": "unrelated-run-id",
    }
    assert log_calls[0] == (
        {
            "case_id_hash": hash_case_id(raw_ids[0]),
            "case_id_hashes": [hash_case_id(raw_ids[1]), hash_case_id(raw_ids[2])],
            "nested": {"case_id_hashes": [hash_case_id(raw_ids[2])]},
            "loss": 0.5,
        },
        4,
    )
    assert summary_updates == [
        {"case_id_hashes": [hash_case_id(raw_ids[0]), hash_case_id(raw_ids[1])], "metric": 0.8}
    ]
    assert table_calls == [
        (
            ["epoch", "case_id_hashes", "case_id_hash"],
            [[1, [hash_case_id(raw_ids[0])], hash_case_id(raw_ids[1])]],
        ),
    ]
    captured = repr((init_calls, log_calls, summary_updates, table_calls))
    assert all(raw_id not in captured for raw_id in raw_ids)


def test_wandb_payloads_redact_local_paths_and_generic_subject_ids_but_keep_artifact_refs(
    monkeypatch,
):
    init_calls = []
    log_calls = []

    class FakeRun:
        def log(self, metrics, step=None):
            log_calls.append((metrics, step))

        def finish(self):
            return None

    def init(**kwargs):
        init_calls.append(kwargs)
        return FakeRun()

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init))
    run_config = {
        "source_checkpoint": "C:/private/SUBJECT_001/best.pt",
        "data_root": "C:/data/SUBJECT_001",
        "subject_id": "SUBJECT_001",
        "source_artifact": "entity/project/model:v1",
        "project_name": "token-mixer-placement-matters",
    }

    tracker = create_tracker({"enabled": True, "mode": "offline"}, run_config)
    tracker.log(
        {
            "checkpoint_path": "C:/private/SUBJECT_001/best.pt",
            "subject_id": "SUBJECT_001",
            "source_artifact": "entity/project/model:v1",
        },
        step=1,
    )

    config_payload = init_calls[0]["config"]
    log_payload = log_calls[0][0]
    assert "C:/private/SUBJECT_001/best.pt" not in repr(init_calls)
    assert "C:/data/SUBJECT_001" not in repr(init_calls)
    assert "SUBJECT_001" not in repr(init_calls)
    assert config_payload["source_artifact"] == "entity/project/model:v1"
    assert config_payload["project_name"] == "token-mixer-placement-matters"
    assert log_payload["source_artifact"] == "entity/project/model:v1"
    assert "SUBJECT_001" not in repr(log_payload)


def test_wandb_image_captions_are_redacted_before_image_construction(
    monkeypatch,
):
    image_calls = []

    class FakeImage:
        def __init__(self, image, caption=None):
            image_calls.append((image, caption))

    class FakeRun:
        def log(self, *_args, **_kwargs):
            return None

        def finish(self):
            return None

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(init=lambda **_: FakeRun(), Image=FakeImage),
    )
    tracker = create_tracker(
        {"enabled": True, "mode": "offline", "log_images": True}, {}
    )

    tracker.log_images(
        {"preview": object()},
        step=1,
        captions={"preview": "C:/data/SUBJECT_001/scan.png"},
    )

    assert image_calls[0][1] != "C:/data/SUBJECT_001/scan.png"
    assert "SUBJECT_001" not in image_calls[0][1]


def test_wandb_tracker_logs_images_in_offline_and_online_modes(monkeypatch, tmp_path):
    image_calls = []
    log_calls = []

    class FakeImage:
        def __init__(self, image, caption=None):
            image_calls.append((image, caption))
            self.value = len(image_calls)

    class FakeRun:
        def log(self, metrics, step=None):
            log_calls.append((metrics, step))

        def finish(self):
            return None

    def init(**kwargs):
        log_calls.append(("init", kwargs))
        return FakeRun()

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(init=init, Image=FakeImage),
    )
    image_path = tmp_path / "preview.png"

    for mode in ("offline", "online"):
        image_calls.clear()
        log_calls.clear()
        if mode == "online":
            monkeypatch.setenv("WANDB_API_KEY", "fixture-key")
        else:
            monkeypatch.delenv("WANDB_API_KEY", raising=False)

        tracker = create_tracker(
            {"enabled": True, "mode": mode, "log_images": True}, {}
        )
        tracker.log_images(
            {"preview": image_path, "mask": object()},
            step=7,
            captions={"preview": "input preview"},
        )

        assert log_calls[0][1]["mode"] == mode
        assert image_calls[0] == (image_path, "input preview")
        assert image_calls[1][1] is None
        assert log_calls[1][1] == 7
        logged_images = log_calls[1][0]
        assert set(logged_images) == {"preview", "mask"}
        assert logged_images["preview"].value == 1
        assert logged_images["mask"].value == 2


def test_wandb_image_gate_skips_image_construction(monkeypatch):
    image_calls = []

    class FakeImage:
        def __init__(self, *_args, **_kwargs):
            image_calls.append(True)

    class FakeRun:
        def log(self, *_args, **_kwargs):
            raise AssertionError("image gate logged disabled images")

        def finish(self):
            return None

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(init=lambda **_: FakeRun(), Image=FakeImage),
    )

    tracker = create_tracker(
        {"enabled": True, "mode": "offline", "log_images": False}, {}
    )
    tracker.log_images({"preview": object()}, step=3)

    assert image_calls == []


def test_wandb_hash_fields_normalize_untrusted_identifier_values(monkeypatch):
    init_calls = []
    log_calls = []

    class FakeRun:
        def log(self, metrics, step=None):
            log_calls.append((metrics, step))

        def finish(self):
            return None

    def init(**kwargs):
        init_calls.append(kwargs)
        return FakeRun()

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init))
    raw_ids = ["RAW_IDENTIFIER_001", "RAW_IDENTIFIER_002"]
    tracker = create_tracker(
        {"enabled": True, "mode": "offline"},
        {
            "case_id_hash": raw_ids[0],
            "case_id_hashes": [raw_ids[1], hash_case_id("approved-case")],
        },
    )

    tracker.log(
        {
            "case_id_hash": raw_ids[0],
            "case_id_hashes": [raw_ids[1], hash_case_id("approved-case")],
        },
        step=1,
    )

    expected = {
        "case_id_hash": hash_case_id(raw_ids[0]),
        "case_id_hashes": [
            hash_case_id(raw_ids[1]),
            hash_case_id("approved-case"),
        ],
    }
    assert init_calls[0]["config"] == expected
    assert log_calls == [(expected, 1)]
    assert all(raw_id not in repr((init_calls, log_calls)) for raw_id in raw_ids)


def test_tracker_reports_wandb_image_gate(monkeypatch):
    class FakeRun:
        def finish(self):
            return None

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(init=lambda **_: FakeRun()),
    )

    disabled = create_tracker({"enabled": True, "mode": "offline"}, {})
    enabled = create_tracker(
        {"enabled": True, "mode": "offline", "log_images": True}, {}
    )

    assert disabled.image_logging_enabled is False
    assert enabled.image_logging_enabled is True


def test_wandb_tracker_logs_artifact_files_waits_and_returns_reference(
    monkeypatch, tmp_path
):
    events = []

    class FakeArtifact:
        def __init__(self, name, type):
            self.name = name
            self.version = "v3"
            events.append(("artifact", name, type))

        def add_file(self, path, name):
            events.append(("file", path, name))

        def wait(self):
            events.append("wait")

    class FakeRun:
        def log_artifact(self, artifact, aliases):
            events.append(("log_artifact", artifact.name, aliases))

        def log(self, *_args, **_kwargs):
            return None

        def finish(self):
            return None

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(init=lambda **_: FakeRun(), Artifact=FakeArtifact),
    )
    first = tmp_path / "metrics.json"
    second = tmp_path / "best.pt"

    tracker = create_tracker({"enabled": True, "mode": "offline"}, {})
    reference = tracker.log_artifact(
        "model",
        {"metrics.json": first, "best.pt": second},
        artifact_type="checkpoint",
        aliases=("best", "latest"),
    )

    assert reference == "model:v3"
    assert events == [
        ("artifact", "model", "checkpoint"),
        ("file", str(first), "metrics.json"),
        ("file", str(second), "best.pt"),
        ("log_artifact", "model", ["best", "latest"]),
        "wait",
    ]


def test_wandb_tracker_restores_artifact_and_resolves_best_checkpoint(
    monkeypatch, tmp_path
):
    calls = []
    destination = tmp_path / "restored"
    destination.mkdir()
    best = destination / "best.pt"
    best.write_bytes(b"checkpoint")

    class UsedArtifact:
        def download(self, *, root):
            calls.append(("download", root))
            return destination

    class FakeRun:
        def use_artifact(self, reference):
            calls.append(("use_artifact", reference))
            return UsedArtifact()

        def log(self, *_args, **_kwargs):
            return None

        def finish(self):
            return None

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(init=lambda **_: FakeRun()),
    )

    tracker = create_tracker({"enabled": True, "mode": "offline"}, {})

    assert tracker.restore_artifact("entity/project/model:v3", destination) == best
    assert calls == [
        ("use_artifact", "entity/project/model:v3"),
        ("download", str(destination)),
    ]


def test_redaction_sanitizes_identifier_content_inside_namespaced_keys():
    raw_payload = {
        "metric/CASE001": 1.0,
        r"image\A17": 2.0,
        "power/sample_count": 3,
    }

    redacted = redact_case_identifiers(raw_payload)

    assert redacted == {
        "metric/[REDACTED_CASE_ID]": 1.0,
        r"image\[REDACTED_CASE_ID]": 2.0,
        "power/sample_count": 3,
    }


def test_wandb_redacts_generic_context_and_image_metadata_before_construction(
    monkeypatch,
):
    init_calls = []
    log_calls = []
    image_calls = []

    class FakeImage:
        def __init__(self, image, caption=None):
            image_calls.append((image, caption))

    class FakeRun:
        def log(self, metrics, step=None):
            log_calls.append((metrics, step))

        def finish(self):
            return None

    def init(**kwargs):
        init_calls.append(kwargs)
        return FakeRun()

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(init=init, Image=FakeImage),
    )
    tracker = create_tracker(
        {"enabled": True, "mode": "offline", "log_images": True},
        {
            "subject_name": "A17",
            "source_artifact": "entity/PROJECT/CASE001:v1",
        },
    )

    tracker.log(
        {"subject_name": "A17", "metric/CASE001": 1.0},
        step=1,
    )
    tracker.log_images(
        {"CASE001": object()},
        step=1,
        captions={"CASE001": "CASE001-caption"},
    )

    assert init_calls[0]["config"]["subject_name"] == hash_case_id("A17")
    assert init_calls[0]["config"]["source_artifact"] == "entity/PROJECT/CASE001:v1"
    logged_metrics = log_calls[0][0]
    assert logged_metrics["subject_name"] == hash_case_id("A17")
    assert "metric/[REDACTED_CASE_ID]" in logged_metrics
    image_payload = log_calls[1][0]
    assert "CASE001" not in repr(image_payload)
    assert image_calls[0][1] != "CASE001-caption"


def test_redaction_removes_local_paths_from_namespaced_mapping_keys_only():
    forward_path = "C:/private/user/file.pt"
    backward_path = r"C:\private\user\file.pt"

    redacted = redact_case_identifiers(
        {
            f"checkpoint/{forward_path}": forward_path,
            f"nested/checkpoint={backward_path}": backward_path,
            "train/loss": 0.5,
            "val/dice": 0.8,
            "source_artifact": "entity/project/model:v1",
        }
    )

    assert redacted["train/loss"] == 0.5
    assert redacted["val/dice"] == 0.8
    assert redacted["source_artifact"] == "entity/project/model:v1"
    assert "C:/private/user/file.pt" not in repr(redacted)
    assert r"C:\private\user\file.pt" not in repr(redacted)
    assert any("[REDACTED_PATH]" in key for key in redacted)


def test_wandb_sanitizes_all_submitted_names_and_payload_contexts(monkeypatch):
    raw_path = r"C:\private\user\file.pt"
    raw_case = "CASE001"
    init_calls = []
    log_calls = []
    summary_updates = []
    table_calls = []
    image_calls = []
    metric_calls = []

    class FakeSummary:
        def update(self, metrics):
            summary_updates.append(dict(metrics))

    class FakeTable:
        def __init__(self, *, columns, data):
            table_calls.append((list(columns), [list(row) for row in data]))
            self.value = "table"

    class FakeImage:
        def __init__(self, image, caption=None):
            image_calls.append((image, caption))
            self.value = "image"

    class FakeRun:
        summary = FakeSummary()

        def log(self, metrics, step=None):
            log_calls.append((metrics, step))

        def define_metric(self, name, **kwargs):
            metric_calls.append((name, kwargs))

        def finish(self):
            return None

    def init(**kwargs):
        init_calls.append(kwargs)
        return FakeRun()

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(init=init, Table=FakeTable, Image=FakeImage),
    )
    tracker = create_tracker(
        {"enabled": True, "mode": "offline", "log_images": True},
        {
            "checkpoint_path": raw_path,
            "patient": "A17",
            "patient_name": "A17",
            "case_note": raw_case,
            "ordinary_label": "train/loss",
            "source_artifact": "entity/project/model:v1",
        },
    )

    tracker.log(
        {
            f"metric/{raw_case}": raw_path,
            "train/loss": 0.5,
        },
        step=1,
    )
    tracker.log_summary({f"summary/{raw_case}": raw_path, "val/dice": 0.8})
    tracker.log_table(
        f"history/{raw_case}",
        (f"column/{raw_case}", "source_path", "train/loss"),
        ((raw_case, raw_path, 0.5),),
    )
    image_name = f"preview/{raw_path}"
    tracker.log_images(
        {image_name: object()},
        step=1,
        captions={image_name: f"{raw_case}: {raw_path}"},
    )
    tracker.define_metric(f"val/{raw_case}", step_metric=f"step/{raw_path}")

    captured = repr(
        (init_calls, log_calls, summary_updates, table_calls, image_calls, metric_calls)
    )
    assert raw_path not in captured
    assert raw_case not in captured
    assert init_calls[0]["config"]["patient"] == hash_case_id("A17")
    assert init_calls[0]["config"]["patient_name"] == hash_case_id("A17")
    assert init_calls[0]["config"]["ordinary_label"] == "train/loss"
    assert init_calls[0]["config"]["source_artifact"] == "entity/project/model:v1"
    assert log_calls[0][0]["train/loss"] == 0.5
    assert "metric/[REDACTED_CASE_ID]" in log_calls[0][0]
    assert summary_updates[0]["val/dice"] == 0.8
    assert "summary/[REDACTED_CASE_ID]" in summary_updates[0]
    assert list(log_calls[1][0]) == ["history/[REDACTED_CASE_ID]"]
    assert table_calls[0][0][-1] == "train/loss"
    assert "column/[REDACTED_CASE_ID]" in table_calls[0][0]
    assert table_calls[0][1][0][1] == "[REDACTED_PATH]"
    assert list(log_calls[2][0]) == ["preview/[REDACTED_PATH]"]
    assert image_calls[0][1] != f"{raw_case}: {raw_path}"
    assert metric_calls[0][0] != f"val/{raw_case}"
    assert metric_calls[0][1]["step_metric"] != f"step/{raw_path}"
