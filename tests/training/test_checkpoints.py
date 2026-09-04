import os
from pathlib import Path

import pytest
import torch
from torch import nn

import token_mixer.training.checkpoints as checkpoint_module
from token_mixer.training.checkpoints import CheckpointManager
from token_mixer.training.phases import PhaseSpec, apply_phase


def _make_scaler():
    return torch.amp.GradScaler("cpu", enabled=False)


def _assert_nested_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        actual_value = actual[key]
        expected_value = expected[key]
        if isinstance(actual_value, torch.Tensor):
            assert torch.equal(actual_value, expected_value)
        elif isinstance(actual_value, dict):
            _assert_nested_equal(actual_value, expected_value)
        else:
            assert actual_value == expected_value


def test_checkpoint_round_trip_restores_training_state(tmp_path: Path):
    torch.manual_seed(5)
    model = nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    scaler = _make_scaler()

    loss = model(torch.ones(1, 2)).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()

    expected_model = {name: value.detach().clone() for name, value in model.state_dict().items()}
    expected_optimizer = optimizer.state_dict()
    expected_scheduler = scheduler.state_dict()
    expected_scaler = scaler.state_dict()
    state = {
        "epoch": 1,
        "global_step": 1,
        "config": {"seed": 5},
        "manifest_hash": "manifest-sha256",
        "code_version": "0.1.0",
        "monitored_metric": 0.75,
    }

    manager = CheckpointManager(tmp_path)
    path = manager.save("last", model, optimizer, scheduler, scaler, state)

    restored_model = nn.Linear(2, 2)
    restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=0.5)
    restored_scheduler = torch.optim.lr_scheduler.StepLR(restored_optimizer, step_size=1)
    restored_scaler = _make_scaler()
    loaded = manager.load(
        path,
        restored_model,
        restored_optimizer,
        restored_scheduler,
        restored_scaler,
    )

    assert path == tmp_path / "last.pt"
    for name, value in restored_model.state_dict().items():
        assert torch.equal(value, expected_model[name])
    _assert_nested_equal(restored_optimizer.state_dict(), expected_optimizer)
    _assert_nested_equal(restored_scheduler.state_dict(), expected_scheduler)
    _assert_nested_equal(restored_scaler.state_dict(), expected_scaler)
    assert loaded["epoch"] == 1
    assert loaded["global_step"] == 1
    assert loaded["config"] == {"seed": 5}
    assert loaded["manifest_hash"] == "manifest-sha256"
    assert loaded["code_version"] == "0.1.0"
    assert loaded["monitored_metric"] == 0.75


def test_checkpoint_manager_writes_named_resume_files(tmp_path: Path):
    model = nn.Linear(1, 1)
    manager = CheckpointManager(tmp_path)

    best = manager.save("best", model, None, None, None, {"epoch": 2})
    phase = manager.save("phase1_resume", model, None, None, None, {"epoch": 1})

    assert best == tmp_path / "best.pt"
    assert phase == tmp_path / "phase1_resume.pt"
    assert best.is_file()
    assert phase.is_file()


def test_checkpoint_load_requires_existing_path(tmp_path: Path):
    manager = CheckpointManager(tmp_path)

    try:
        manager.load(tmp_path / "missing.pt", nn.Linear(1, 1))
    except FileNotFoundError as exc:
        assert "missing.pt" in str(exc)
    else:
        raise AssertionError("missing checkpoint should fail explicitly")


@pytest.mark.parametrize(
    "tag",
    ["", ".", "..", ".hidden", ".hidden.pt", "nested/tag", r"nested\tag", r"C:\escape", "C:relative"],
)
def test_checkpoint_tag_rejects_paths_and_invalid_names(tmp_path: Path, tag: str):
    manager = CheckpointManager(tmp_path)

    with pytest.raises(ValueError, match="checkpoint tag"):
        manager._path_for_tag(tag)


def test_checkpoint_tag_rejects_absolute_path(tmp_path: Path):
    manager = CheckpointManager(tmp_path)

    with pytest.raises(ValueError, match="checkpoint tag"):
        manager._path_for_tag(tmp_path / "outside")


def test_checkpoint_tag_keeps_simple_names_and_pt_suffix(tmp_path: Path):
    manager = CheckpointManager(tmp_path)

    assert manager._path_for_tag("last") == tmp_path / "last.pt"
    assert manager._path_for_tag("last.pt") == tmp_path / "last.pt"


def test_checkpoint_save_uses_unique_same_directory_temporary_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    manager = CheckpointManager(tmp_path)
    temporary_paths: list[Path] = []
    real_replace = os.replace

    def record_replace(source, destination):
        temporary_paths.append(Path(source))
        return real_replace(source, destination)

    monkeypatch.setattr(checkpoint_module.os, "replace", record_replace)

    model = nn.Linear(1, 1)
    manager.save("first", model, None, None, None, {})
    manager.save("second", model, None, None, None, {})

    assert len(temporary_paths) == 2
    assert len(set(temporary_paths)) == 2
    assert all(path.parent == tmp_path for path in temporary_paths)
    assert all(not path.exists() for path in temporary_paths)


def test_checkpoint_save_preserves_target_and_cleans_temp_on_save_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    manager = CheckpointManager(tmp_path)
    target = tmp_path / "last.pt"
    target.write_bytes(b"previous checkpoint")

    def partial_save(_payload, path):
        Path(path).write_bytes(b"partial checkpoint")
        raise OSError("simulated write failure")

    monkeypatch.setattr(checkpoint_module.torch, "save", partial_save)

    with pytest.raises(OSError, match="simulated write failure"):
        manager.save("last", nn.Linear(1, 1), None, None, None, {})

    assert target.read_bytes() == b"previous checkpoint"
    assert list(tmp_path.glob("*.tmp")) == []


def test_checkpoint_load_restores_loader_generator_without_serializing_object(tmp_path: Path):
    manager = CheckpointManager(tmp_path)
    generator = torch.Generator().manual_seed(123)
    torch.rand(4, generator=generator)

    path = manager.save(
        "loader",
        nn.Linear(1, 1),
        None,
        None,
        None,
        {"loader_generator": generator},
    )
    expected = torch.rand(6, generator=generator)

    restored_generator = torch.Generator().manual_seed(999)
    loaded = manager.load(path, nn.Linear(1, 1), generator=restored_generator)
    actual = torch.rand(6, generator=restored_generator)

    assert torch.equal(actual, expected)
    assert isinstance(loaded["loader_generator_state"], torch.Tensor)
    assert not isinstance(loaded["state"]["loader_generator"], torch.Generator)


@pytest.mark.parametrize("component", ["optimizer", "scheduler", "scaler"])
def test_checkpoint_load_rejects_missing_supplied_component_state(
    tmp_path: Path, component: str
):
    manager = CheckpointManager(tmp_path)
    path = manager.save("model-only", nn.Linear(1, 1), None, None, None, {})
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    scaler = _make_scaler()

    with pytest.raises(ValueError, match=component):
        manager.load(path, model, **{component: locals()[component]})


def test_model_only_checkpoint_loads_when_optional_components_are_omitted(tmp_path: Path):
    manager = CheckpointManager(tmp_path)
    path = manager.save("model-only", nn.Linear(1, 1), None, None, None, {})

    manager.load(path, nn.Linear(1, 1))


class _EncoderDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(2, 2)
        self.decoder = nn.Linear(2, 1)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def test_apply_phase_freezes_then_unfreezes_encoder():
    model = _EncoderDecoder()

    apply_phase(
        model,
        PhaseSpec("phase1", epochs=1, freeze_encoder=True, encoder_lr=0.0, decoder_lr=0.1),
    )
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.decoder.parameters())

    apply_phase(
        model,
        PhaseSpec("phase2", epochs=1, freeze_encoder=False, encoder_lr=0.01, decoder_lr=0.1),
    )
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_apply_phase_preserves_non_encoder_freezes():
    model = _EncoderDecoder()
    for parameter in model.decoder.parameters():
        parameter.requires_grad = False

    apply_phase(
        model,
        PhaseSpec("phase1", epochs=1, freeze_encoder=True, encoder_lr=0.0, decoder_lr=0.1),
    )

    assert all(not parameter.requires_grad for parameter in model.decoder.parameters())
