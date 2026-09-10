import builtins
import sys
from types import SimpleNamespace

import pytest

import token_mixer.training.tracking as tracking_module
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

    with pytest.raises(RuntimeError, match="WANDB_API_KEY"):
        create_tracker({"enabled": True, "mode": "online"}, {})

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
