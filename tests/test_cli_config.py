from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import importlib
from pathlib import Path
import subprocess
import sys

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


@contextmanager
def _compose(config_name: str, overrides: list[str]) -> Iterator[DictConfig]:
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(
            config_name=config_name,
            overrides=overrides,
            return_hydra_config=True,
        )
        hydra_config = HydraConfig.instance()
        previous = hydra_config.cfg
        hydra_config.set_config(cfg)
        try:
            yield cfg
        finally:
            hydra_config.cfg = previous


def test_local_profile_composes_debug_experiment():
    with _compose("local", ["experiment=mod_a"]) as cfg:
        assert cfg.runtime == "local"
        assert cfg.run.name == "debug"
        assert cfg.experiment.name == "mod_a"
        assert cfg.paths.data_root.endswith("data/local/brats")
        assert OmegaConf.select(cfg, "tracking.enabled") is False
        assert cfg.visualization.segmentation_snapshots.snapshot_interval_epochs == 10
        assert cfg.visualization.segmentation_snapshots.image_channel == 3
        assert cfg.visualization.segmentation_snapshots.local_enabled is False


def test_cloud_profile_composes_full_experiment():
    with _compose("cloud", ["experiment=mod_b"]) as cfg:
        assert cfg.runtime == "cloud"
        assert cfg.run.name == "full"
        assert cfg.device == "cuda"
        assert cfg.paths.data_root.endswith("data/cloud/brats")
        assert cfg.tracking.log_images is True
        assert cfg.visualization.segmentation_snapshots.enabled is True
        assert cfg.visualization.segmentation_snapshots.splits == ["train", "val"]


def test_benchmark_profile_composes_without_training_dispatch():
    with _compose("benchmark", []) as cfg:
        assert cfg.command == "benchmark"
        assert cfg.runtime == "benchmark"
        assert cfg.run.name == "full"
        assert cfg.experiment.name == "mod_a"
        assert cfg.paths.data_root.endswith("data/cloud/brats")
        assert cfg.data.name == "brats"
        assert cfg.data.image_root is None
        assert "root" not in cfg.data
        assert cfg.paths.output_root.endswith("outputs")
        assert cfg.tracking.enabled is True
        assert cfg.tracking.mode == "online"
        assert cfg.tracking.entity == "aniekanetimudo"
        assert cfg.tracking.project == "token-mixer-placement-matters"
        assert cfg.tracking.group == "token-mixer-brats-seed42"
        assert cfg.tracking.job_type == "benchmark"
        assert cfg.efficiency.enabled is True
        assert cfg.efficiency.profiler.mac_tool == "thop"
        assert cfg.efficiency.profiler.flop_tool == "fvcore"
        assert cfg.benchmark.source_checkpoint is None
        assert cfg.benchmark.source_artifact is None


def test_benchmark_cnn_composition_uses_cloud_imagenet_root():
    with _compose("benchmark", ["experiment=cnn_denoising_pretrain"]) as cfg:
        assert cfg.paths.image_root.endswith("data/cloud/imagenet")
        assert cfg.data.image_root.endswith("data/benchmark/imagenet")
        assert cfg.data.root == cfg.data.image_root


@pytest.mark.parametrize(
    ("experiment", "model", "data"),
    [
        ("cnn_denoising_pretrain", "cnn_pretrain", "imagenet"),
        ("metaunetr_mamba", "metaunetr", "brats"),
        ("mod_a", "metaunetr", "brats"),
        ("mod_b", "metaunetr", "brats"),
        ("resunet3d", "resunet3d", "brats"),
        ("swinunetr", "swinunetr", "brats"),
        ("transunet", "transunet", "brats"),
    ],
)
def test_experiment_groups_select_model_and_data(
    experiment: str, model: str, data: str
):
    with _compose("local", [f"experiment={experiment}"]) as cfg:
        assert cfg.model.name == model
        assert cfg.data.name == data
        assert cfg.experiment.name == experiment


def test_pretraining_group_uses_runtime_image_root_and_transunet_paths_are_unset():
    with _compose("cloud", ["experiment=cnn_denoising_pretrain"]) as cfg:
        assert cfg.paths.image_root.endswith("data/cloud/imagenet")
        assert cfg.third_party.transunet_root is None
        assert cfg.third_party.pretrained_path is None


@pytest.mark.parametrize(
    ("config_name", "runtime"),
    [("local", "local"), ("cloud", "cloud")],
)
def test_cnn_denoising_compositions_resolve_imagenet_root_alias(
    config_name: str, runtime: str
):
    with _compose(config_name, ["experiment=cnn_denoising_pretrain"]) as cfg:
        resolved = OmegaConf.to_container(cfg.data, resolve=True)

    assert resolved["root"] == resolved["image_root"]
    assert resolved["root"].endswith(f"data/{runtime}/imagenet")


@pytest.mark.parametrize(
    ("experiment", "module_name", "runner_name"),
    [
        (
            "cnn_denoising_pretrain",
            "token_mixer.pipelines.pretrain_cnn",
            "run_cnn_denoising_pretrain",
        ),
        ("metaunetr_mamba", "token_mixer.pipelines.train_metaunetr", "run_metaunetr"),
        ("mod_a", "token_mixer.pipelines.train_metaunetr", "run_metaunetr"),
        ("mod_b", "token_mixer.pipelines.train_metaunetr", "run_metaunetr"),
        ("resunet3d", "token_mixer.pipelines.train_resunet3d", "run_resunet3d"),
        ("swinunetr", "token_mixer.pipelines.train_swinunetr", "run_swinunetr"),
        ("transunet", "token_mixer.pipelines.train_transunet", "run_transunet"),
    ],
)
def test_dispatches_experiment_to_pipeline_without_constructing_models(
    monkeypatch: pytest.MonkeyPatch,
    experiment: str,
    module_name: str,
    runner_name: str,
):
    import token_mixer.cli as cli

    pipeline = importlib.import_module(module_name)
    expected = object()
    calls = []

    def runner(cfg):
        calls.append(cfg)
        return expected

    monkeypatch.setattr(pipeline, runner_name, runner)
    cfg = OmegaConf.create({"experiment": {"name": experiment}})

    assert cli._dispatch(cfg) is expected
    assert calls == [cfg]


def test_dispatches_benchmark_command_before_experiment(
    monkeypatch: pytest.MonkeyPatch,
):
    import token_mixer.cli as cli
    import token_mixer.pipelines.benchmark as benchmark

    expected = object()
    calls = []

    def runner(cfg):
        calls.append(cfg)
        return expected

    monkeypatch.setattr(benchmark, "run_benchmark", runner)
    cfg = OmegaConf.create(
        {"command": "benchmark", "experiment": {"name": "mod_a"}}
    )

    assert cli._dispatch(cfg) is expected
    assert calls == [cfg]


def test_run_saves_composed_config_dispatches_and_preserves_working_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    import token_mixer.cli as cli

    cfg = OmegaConf.create(
        {
            "runtime": "local",
            "experiment": {"name": "mod_a"},
        }
    )
    calls = []
    cwd = Path.cwd()
    monkeypatch.setattr(cli, "_hydra_output_dir", lambda: tmp_path)
    monkeypatch.setattr(cli, "_dispatch", lambda received: calls.append(received))

    cli._run(cfg)

    assert Path.cwd() == cwd
    assert calls == [cfg]
    saved = OmegaConf.load(tmp_path / "config.yaml")
    assert saved.runtime == "local"
    assert saved.experiment.name == "mod_a"


def test_run_writes_completion_artifacts_after_successful_dispatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    import token_mixer.cli as cli
    from token_mixer.training.engine import FitResult

    cfg = OmegaConf.create(
        {
            "runtime": "local",
            "experiment": {"name": "mod_a"},
            "tracking": {"enabled": False, "mode": "disabled"},
        }
    )
    monkeypatch.setattr(cli, "_hydra_output_dir", lambda: tmp_path)
    monkeypatch.setattr(
        cli,
        "_dispatch",
        lambda _cfg: FitResult(0.5, 1, [{"mean_dice": 0.5}]),
    )

    cli._run(cfg)

    assert (tmp_path / "config.yaml").is_file()
    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "provenance.json").is_file()


def test_run_does_not_rewrite_pipeline_completion_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    import token_mixer.cli as cli
    from token_mixer.training.engine import FitResult

    cfg = OmegaConf.create(
        {
            "runtime": "local",
            "experiment": {"name": "mod_a"},
            "tracking": {"enabled": False, "mode": "disabled"},
        }
    )
    monkeypatch.setattr(cli, "_hydra_output_dir", lambda: tmp_path)
    (tmp_path / "metrics.json").write_text("pipeline metrics\n", encoding="utf-8")
    (tmp_path / "provenance.json").write_text("pipeline provenance\n", encoding="utf-8")
    monkeypatch.setattr(cli, "_dispatch", lambda _cfg: FitResult(0.5, 1, []))

    cli._run(cfg)

    assert (tmp_path / "metrics.json").read_text(encoding="utf-8") == "pipeline metrics\n"
    assert (tmp_path / "provenance.json").read_text(encoding="utf-8") == "pipeline provenance\n"


def test_run_writes_failed_provenance_for_non_transfer_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    import token_mixer.cli as cli

    cfg = OmegaConf.create(
        {
            "runtime": "local",
            "experiment": {"name": "mod_a"},
            "tracking": {"enabled": False, "mode": "disabled"},
        }
    )
    failure = ValueError("fixture training failure")
    monkeypatch.setattr(cli, "_hydra_output_dir", lambda: tmp_path)
    monkeypatch.setattr(cli, "_dispatch", lambda _cfg: (_ for _ in ()).throw(failure))

    with pytest.raises(ValueError) as caught:
        cli._run(cfg)

    assert caught.value is failure
    provenance = OmegaConf.load(tmp_path / "provenance.json")
    assert provenance.status == "failed"
    assert provenance.error == "fixture training failure"
    assert not (tmp_path / "metrics.json").exists()


def test_dispatch_rejects_unknown_experiment():
    import token_mixer.cli as cli

    with pytest.raises(ValueError, match="unknown experiment"):
        cli._dispatch(OmegaConf.create({"experiment": {"name": "unknown"}}))


@pytest.mark.parametrize(
    "arguments",
    [
        ["--config-name", "local", "experiment=mod_a", "run=debug", "--help"],
        ["--config-name", "cloud", "experiment=mod_b", "run=full", "--cfg", "job"],
    ],
)
def test_cli_inspection_commands_do_not_train_or_emit_machine_paths(
    arguments: list[str],
):
    result = subprocess.run(
        [sys.executable, "-m", "token_mixer", *arguments],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert str(PROJECT_ROOT) not in output
    assert "Traceback" not in output


def test_default_composed_metaunetr_model_builds_with_reduced_dimensions():
    from torch import nn

    from token_mixer.models.metaunetr.variants import build_metaunetr

    with _compose(
        "local",
        [
            "experiment=metaunetr_mamba",
            "model.base_channels=2",
            "model.depths=[1,1,1,1]",
            "model.num_heads=[1,1,1,1]",
            "model.window_size=1",
            "model.d_state=2",
            "model.d_conv=1",
            "model.mamba_expand=1",
        ],
    ) as cfg:
        model = build_metaunetr(cfg.model, cfg.experiment.variant)

    assert isinstance(model, nn.Module)
    assert model.base_channels == 2
