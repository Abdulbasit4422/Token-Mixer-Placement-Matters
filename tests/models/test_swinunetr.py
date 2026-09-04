from __future__ import annotations

import builtins
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from token_mixer.models import swinunetr
from token_mixer.training.engine import _parameter_groups
from token_mixer.training.phases import PhaseSpec


class _FakeSwinUNETR(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.swinViT = nn.Conv3d(4, 4, kernel_size=1, bias=False)
        self.decoder = nn.Conv3d(4, 3, kernel_size=1, bias=False)

    def forward(self, inputs):
        return self.decoder(self.swinViT(inputs))


def test_module_import_does_not_load_monai():
    project_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(project_root / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import token_mixer.models.swinunetr; "
                "assert not any(name == 'monai' or name.startswith('monai.') "
                "for name in sys.modules)"
            ),
        ],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_builder_reads_nested_config_and_maps_drop_to_monai(monkeypatch):
    class FakeSwinUNETR(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.kwargs = kwargs
            self.swinViT = nn.Identity()

    monkeypatch.setattr(swinunetr, "_load_swinunetr", lambda: FakeSwinUNETR)
    cfg = OmegaConf.create(
        {
            "model": {
                "in_channels": 4,
                "out_channels": 3,
                "feature_size": 24,
                "use_checkpoint": False,
                "use_v2": True,
                "drop": 0.1,
                "attn_drop_rate": 0.2,
                "dropout_path_rate": 0.3,
                "spatial_dims": 3,
            }
        }
    )

    model = swinunetr.build_swinunetr(cfg)

    assert model.network.kwargs == {
        "in_channels": 4,
        "out_channels": 3,
        "feature_size": 24,
        "use_checkpoint": False,
        "use_v2": True,
        "drop_rate": 0.1,
        "attn_drop_rate": 0.2,
        "dropout_path_rate": 0.3,
        "spatial_dims": 3,
    }


def test_builder_defaults_match_legacy_brat_constructor(monkeypatch):
    class FakeSwinUNETR(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.kwargs = kwargs
            self.swinViT = nn.Identity()

    monkeypatch.setattr(swinunetr, "_load_swinunetr", lambda: FakeSwinUNETR)

    model = swinunetr.build_swinunetr(OmegaConf.create({}))

    assert model.network.kwargs == {
        "in_channels": 4,
        "out_channels": 3,
        "feature_size": 48,
        "use_checkpoint": True,
        "use_v2": False,
        "drop_rate": 0.0,
        "attn_drop_rate": 0.0,
        "dropout_path_rate": 0.0,
        "spatial_dims": 3,
    }


def test_builder_exposes_unique_encoder_boundary_for_shared_parameter_groups(monkeypatch):
    monkeypatch.setattr(swinunetr, "_load_swinunetr", lambda: _FakeSwinUNETR)

    model = swinunetr.build_swinunetr(OmegaConf.create({}))

    assert isinstance(model.encoder, nn.Module)
    named_parameters = list(model.named_parameters(remove_duplicate=False))
    assert len(named_parameters) == len({id(parameter) for _, parameter in named_parameters})

    groups = _parameter_groups(
        model,
        PhaseSpec("test", epochs=1, freeze_encoder=False, encoder_lr=0.1, decoder_lr=0.2),
    )
    encoder_ids = {id(parameter) for parameter in model.encoder.parameters()}
    grouped_ids = [{id(parameter) for parameter in group["params"]} for group in groups]
    assert grouped_ids[0] == encoder_ids
    assert grouped_ids[0].isdisjoint(grouped_ids[1])
    assert set.union(*grouped_ids) == {id(parameter) for parameter in model.parameters()}


@pytest.mark.parametrize(
    ("name", "value"),
    [("spatial_dims", 2), ("in_channels", 3), ("out_channels", 2)],
)
def test_builder_rejects_noncanonical_contract_values(monkeypatch, name, value):
    monkeypatch.setattr(
        swinunetr,
        "_load_swinunetr",
        lambda: pytest.fail("MONAI constructor should not run after validation"),
    )

    with pytest.raises(ValueError, match=name):
        swinunetr.build_swinunetr(OmegaConf.create({name: value}))


def test_builder_forwards_supported_architecture_options(monkeypatch):
    class ConfigurableSwinUNETR(nn.Module):
        def __init__(
            self,
            in_channels,
            out_channels,
            patch_size,
            depths,
            num_heads,
            window_size,
            feature_size,
            use_checkpoint,
            use_v2,
            drop_rate,
            attn_drop_rate,
            dropout_path_rate,
            spatial_dims,
        ):
            super().__init__()
            self.kwargs = {
                "in_channels": in_channels,
                "out_channels": out_channels,
                "patch_size": patch_size,
                "depths": depths,
                "num_heads": num_heads,
                "window_size": window_size,
                "feature_size": feature_size,
                "use_checkpoint": use_checkpoint,
                "use_v2": use_v2,
                "drop_rate": drop_rate,
                "attn_drop_rate": attn_drop_rate,
                "dropout_path_rate": dropout_path_rate,
                "spatial_dims": spatial_dims,
            }
            self.swinViT = nn.Identity()

        def forward(self, inputs):
            return inputs[:, :3]

    monkeypatch.setattr(swinunetr, "_load_swinunetr", lambda: ConfigurableSwinUNETR)
    cfg = OmegaConf.create(
        {
            "model": {
                "patch_size": 4,
                "depths": [1, 2, 3, 4],
                "num_heads": [2, 4, 8, 16],
                "window_size": 5,
                "feature_size": 24,
                "checkpoint": False,
                "use_v2": True,
                "drop": 0.1,
                "attn_drop_rate": 0.2,
                "dropout_path_rate": 0.3,
            }
        }
    )

    model = swinunetr.build_swinunetr(cfg)

    assert model.network.kwargs == {
        "in_channels": 4,
        "out_channels": 3,
        "patch_size": 4,
        "depths": (1, 2, 3, 4),
        "num_heads": (2, 4, 8, 16),
        "window_size": 5,
        "feature_size": 24,
        "use_checkpoint": False,
        "use_v2": True,
        "drop_rate": 0.1,
        "attn_drop_rate": 0.2,
        "dropout_path_rate": 0.3,
        "spatial_dims": 3,
    }


def test_builder_rejects_explicit_option_missing_from_installed_signature(monkeypatch):
    class LegacySwinUNETR(nn.Module):
        def __init__(self, in_channels, out_channels, feature_size, spatial_dims):
            super().__init__()
            self.swinViT = nn.Identity()

        def forward(self, inputs):
            return inputs[:, :3]

    monkeypatch.setattr(swinunetr, "_load_swinunetr", lambda: LegacySwinUNETR)

    with pytest.raises(TypeError, match="patch_size.*not supported"):
        swinunetr.build_swinunetr(
            OmegaConf.create({"model": {"patch_size": 4}})
        )


def test_wrapper_validates_input_rank_and_channels(monkeypatch):
    monkeypatch.setattr(swinunetr, "_load_swinunetr", lambda: _FakeSwinUNETR)
    model = swinunetr.build_swinunetr(OmegaConf.create({}))

    with pytest.raises(ValueError, match="5 dimensions"):
        model(torch.randn(1, 4, 16, 16))
    with pytest.raises(ValueError, match="4 input channels"):
        model(torch.randn(1, 3, 16, 16, 16))


def test_builder_omits_unsupported_implicit_defaults_for_older_monai(monkeypatch):
    class LegacySwinUNETR(nn.Module):
        def __init__(
            self,
            in_channels,
            out_channels,
            feature_size,
            use_checkpoint,
            spatial_dims,
        ):
            super().__init__()
            self.kwargs = {
                "in_channels": in_channels,
                "out_channels": out_channels,
                "feature_size": feature_size,
                "use_checkpoint": use_checkpoint,
                "spatial_dims": spatial_dims,
            }
            self.swinViT = nn.Identity()

    monkeypatch.setattr(swinunetr, "_load_swinunetr", lambda: LegacySwinUNETR)
    cfg = OmegaConf.create(
        {
            "feature_size": 24,
        }
    )

    model = swinunetr.build_swinunetr(cfg)

    assert model.network.kwargs == {
        "in_channels": 4,
        "out_channels": 3,
        "feature_size": 24,
        "use_checkpoint": True,
        "spatial_dims": 3,
    }


def test_builder_supplies_img_size_only_to_legacy_monai(monkeypatch):
    class LegacySwinUNETR(nn.Module):
        def __init__(
            self,
            img_size,
            in_channels,
            out_channels,
            feature_size,
            use_checkpoint,
            spatial_dims,
        ):
            super().__init__()
            self.kwargs = {
                "img_size": img_size,
                "in_channels": in_channels,
                "out_channels": out_channels,
                "feature_size": feature_size,
                "use_checkpoint": use_checkpoint,
                "spatial_dims": spatial_dims,
            }
            self.swinViT = nn.Identity()

    monkeypatch.setattr(swinunetr, "_load_swinunetr", lambda: LegacySwinUNETR)

    model = swinunetr.build_swinunetr(
        OmegaConf.create({"model": {"roi_size": [64, 96, 128]}})
    )

    assert model.network.kwargs == {
        "img_size": (64, 96, 128),
        "in_channels": 4,
        "out_channels": 3,
        "feature_size": 48,
        "use_checkpoint": True,
        "spatial_dims": 3,
    }


@pytest.mark.parametrize("feature_size", [0, -1, 1.5, True])
def test_builder_rejects_nonpositive_or_noninteger_feature_size(
    monkeypatch, feature_size
):
    monkeypatch.setattr(
        swinunetr,
        "_load_swinunetr",
        lambda: pytest.fail("MONAI constructor should not run after validation"),
    )

    with pytest.raises(ValueError, match="feature_size"):
        swinunetr.build_swinunetr(OmegaConf.create({"feature_size": feature_size}))


@pytest.mark.parametrize("spatial_dims", [0, -1, 1, 4, True])
def test_builder_rejects_invalid_spatial_dims(monkeypatch, spatial_dims):
    monkeypatch.setattr(
        swinunetr,
        "_load_swinunetr",
        lambda: pytest.fail("MONAI constructor should not run after validation"),
    )

    with pytest.raises(ValueError, match="spatial_dims"):
        swinunetr.build_swinunetr(OmegaConf.create({"spatial_dims": spatial_dims}))


def test_builder_reports_optional_monai_dependency(monkeypatch):
    real_import = builtins.__import__

    def missing_monai(name, *args, **kwargs):
        if name == "monai.networks.nets" or name.startswith("monai."):
            raise ModuleNotFoundError("No module named 'monai'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_monai)

    with pytest.raises(ImportError, match="MONAI.*imaging"):
        swinunetr.build_swinunetr(OmegaConf.create({}))


def test_monai_cpu_forward_preserves_brat_logits_shape():
    pytest.importorskip("monai", reason="SwinUNETR CPU integration requires MONAI")
    pytest.importorskip("einops", reason="MONAI SwinUNETR forward requires einops")

    model = swinunetr.build_swinunetr(
        OmegaConf.create(
            {
                "model": {
                    "feature_size": 12,
                    "use_checkpoint": False,
                    "spatial_dims": 3,
                }
            }
        )
    ).eval()

    with torch.no_grad():
        # 64^3 is minimum valid synthetic size; 32^3 reaches MONAI's 1^3 bottleneck.
        logits = model(torch.randn(1, 4, 64, 64, 64))

    assert logits.shape == (1, 3, 64, 64, 64)
    assert torch.isfinite(logits).all()
    assert model.output_regions == ("ET", "TC", "WT")
    assert model.in_channels == 4
    assert model.out_channels == 3
