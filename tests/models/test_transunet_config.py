from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from token_mixer.models import transunet
from token_mixer.models.transunet import (
    TransUNetSliceAdapter,
    adapt_transunet_output,
    get_transunet_metadata,
    resize_slice,
    build_transunet,
)
from token_mixer.training.engine import _parameter_groups
from token_mixer.training.phases import PhaseSpec


def _config(tmp_path: Path) -> object:
    return OmegaConf.create(
        {
            "model": {
                "in_channels": 4,
                "num_classes": 3,
                "image_size": 224,
                "vit_name": "R50-ViT-B_16",
            },
            "third_party": {
                "transunet_root": str(tmp_path / "TransUNet"),
                "pretrained_path": str(tmp_path / "R50+ViT-B_16.npz"),
            },
        }
    )


def _injected_config(*, image_size: int | tuple[int, int] = 4) -> object:
    return OmegaConf.create(
        {
            "model": {
                "in_channels": 4,
                "num_classes": 3,
                "image_size": image_size,
                "vit_name": "R50-ViT-B_16",
            }
        }
    )


class _InjectedBackend(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Conv2d(4, 4, kernel_size=1)
        self.decoder = nn.Conv2d(4, 4, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(image))


class _FixedFourClassBackend(_InjectedBackend):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        values = torch.arange(4, dtype=image.dtype, device=image.device)
        return values.view(1, 4, 1, 1).expand(
            image.shape[0], 4, image.shape[-2], image.shape[-1]
        )


def _write_fake_checkout(root: Path, marker: str) -> None:
    networks = root / "networks"
    networks.mkdir(parents=True)
    (networks / "__init__.py").write_text("", encoding="utf-8")
    (networks / "vit_seg_configs.py").write_text(
        "from types import SimpleNamespace\n"
        f"CONFIGS = {{'R50-ViT-B_16': SimpleNamespace(marker={marker!r})}}\n",
        encoding="utf-8",
    )
    (networks / "vit_seg_modeling_resnet_skip.py").write_text(
        "from torch import nn\n"
        "class StdConv2d(nn.Conv2d):\n"
        "    pass\n",
        encoding="utf-8",
    )
    (networks / "vit_seg_modeling.py").write_text(
        "from . import vit_seg_configs as configs\n"
        "from .vit_seg_modeling_resnet_skip import StdConv2d\n"
        "CONFIGS = configs.CONFIGS\n"
        "class VisionTransformer:\n"
        "    pass\n",
        encoding="utf-8",
    )


def test_build_transunet_rejects_missing_external_root_before_import(
    tmp_path: Path,
):
    with pytest.raises(FileNotFoundError, match="third_party.transunet_root"):
        build_transunet(_config(tmp_path))


def test_build_transunet_rejects_missing_pretrained_file_before_import(
    tmp_path: Path,
):
    (tmp_path / "TransUNet").mkdir()

    with pytest.raises(FileNotFoundError, match="pretrained"):
        build_transunet(_config(tmp_path))


def test_build_transunet_rejects_missing_required_configuration():
    with pytest.raises(ValueError, match="third_party.transunet_root"):
        build_transunet(OmegaConf.create({}))


def test_build_transunet_rejects_checkout_without_required_sources(
    tmp_path: Path,
):
    root = tmp_path / "TransUNet"
    root.mkdir()
    (tmp_path / "R50+ViT-B_16.npz").touch()

    with pytest.raises(FileNotFoundError, match="required external source"):
        build_transunet(_config(tmp_path))


def test_transunet_metadata_declares_slice_based_canonical_contract():
    metadata = get_transunet_metadata()

    assert metadata["architecture"] == "TransUNet"
    assert metadata["dimensionality"] == "2-D"
    assert metadata["slice_based"] is True
    assert metadata["native_3d"] is False
    assert metadata["canonical_regions"] == ("ET", "TC", "WT")
    assert metadata["external_class_mapping"] == {
        0: "background",
        1: "ET",
        2: "TC",
        3: "WT",
    }


def test_four_class_logits_become_canonical_region_logits():
    external_logits = torch.tensor(
        [[[[0.0]], [[1.0]], [[2.0]], [[3.0]]]], dtype=torch.float32
    )
    probabilities = torch.softmax(external_logits, dim=1)
    expected = torch.cat(
        (
            torch.log(probabilities[:, 1:2] / probabilities[:, (0, 2, 3)].sum(dim=1, keepdim=True)),
            torch.log(probabilities[:, (1, 2)].sum(dim=1, keepdim=True) / probabilities[:, (0, 3)].sum(dim=1, keepdim=True)),
            torch.log(probabilities[:, (1, 2, 3)].sum(dim=1, keepdim=True) / probabilities[:, 0:1]),
        ),
        dim=1,
    )

    result = adapt_transunet_output(external_logits, output_kind="logits")

    assert result.shape == (1, 3, 1, 1)
    torch.testing.assert_close(result, expected)


def test_four_class_softmax_becomes_canonical_region_logits():
    probabilities = torch.tensor(
        [[[[0.1]], [[0.2]], [[0.3]], [[0.4]]]], dtype=torch.float32
    )

    result = adapt_transunet_output(probabilities, output_kind="softmax")

    expected = torch.tensor(
        [[[[np.log(0.2 / 0.8)]], [[np.log(0.5 / 0.5)]], [[np.log(0.9 / 0.1)]]]],
        dtype=torch.float32,
    )
    torch.testing.assert_close(result, expected)


def test_output_adapter_rejects_scalar_class_ids_without_probabilities():
    with pytest.raises(ValueError, match="raw region logits"):
        adapt_transunet_output(torch.zeros(1, 4, 4), output_kind="logits")


def test_resize_slice_uses_numpy_compatible_torch_interpolation():
    image = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
    label = np.array([[0, 1], [2, 3]], dtype=np.uint8)

    resized_image = resize_slice(image, (4, 4), mode="bilinear")
    resized_label = resize_slice(label, (4, 4), mode="nearest")

    assert isinstance(resized_image, np.ndarray)
    assert resized_image.shape == (3, 4, 4)
    assert resized_image.dtype == image.dtype
    assert isinstance(resized_label, np.ndarray)
    assert resized_label.shape == (4, 4)
    assert resized_label.dtype == label.dtype
    assert set(np.unique(resized_label)).issubset(set(np.unique(label)))


def test_slice_adapter_returns_three_canonical_raw_logits_without_checkout():
    class FakeExternalModel(nn.Module):
        def forward(self, image: torch.Tensor) -> torch.Tensor:
            return torch.zeros(
                image.shape[0],
                4,
                image.shape[-2],
                image.shape[-1],
                device=image.device,
            )

    adapter = TransUNetSliceAdapter(
        FakeExternalModel(),
        image_size=(4, 4),
        in_channels=4,
    )
    output = adapter(torch.zeros(2, 4, 2, 3))

    assert output.shape == (2, 3, 2, 3)
    assert adapter.metadata["slice_based"] is True
    assert adapter.metadata["canonical_regions"] == ("ET", "TC", "WT")


def test_slice_adapter_exposes_encoder_boundary_for_shared_fit():
    backend = _InjectedBackend()
    adapter = TransUNetSliceAdapter(
        backend,
        image_size=(4, 4),
        in_channels=4,
    )

    assert isinstance(adapter.encoder, nn.Module)
    assert adapter.encoder is backend.encoder
    groups = _parameter_groups(
        adapter,
        PhaseSpec("test", epochs=1, freeze_encoder=False, encoder_lr=0.01, decoder_lr=0.1),
    )
    assert len(groups) == 2
    assert {id(parameter) for parameter in groups[0]["params"]} == {
        id(parameter) for parameter in backend.encoder.parameters()
    }


def test_builder_accepts_injected_backend_and_state_without_external_paths(monkeypatch):
    backend = _InjectedBackend()
    state = {
        name: torch.ones_like(parameter)
        for name, parameter in backend.state_dict().items()
    }
    monkeypatch.setattr(
        transunet.np,
        "load",
        lambda *_args, **_kwargs: pytest.fail("injected backend must not load a file"),
    )

    model = build_transunet(
        _injected_config(image_size=(4, 4)),
        external_network=backend,
        external_state=state,
    )

    assert torch.equal(backend.encoder.weight, torch.ones_like(backend.encoder.weight))
    output = model(torch.zeros(2, 4, 2, 3))
    assert output.shape == (2, 3, 2, 3)


def test_injected_backend_preserves_four_class_to_canonical_conversion():
    model = build_transunet(
        _injected_config(image_size=(2, 3)),
        external_model=_FixedFourClassBackend(),
    )
    logits = model(torch.zeros(1, 4, 2, 3))

    expected = adapt_transunet_output(
        torch.arange(4, dtype=torch.float32).view(1, 4, 1, 1).expand(1, 4, 2, 3),
        output_kind="logits",
    )
    assert logits.shape == (1, 3, 2, 3)
    torch.testing.assert_close(logits, expected)


def test_external_import_isolated_and_cleans_up_module_cache(tmp_path: Path):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    _write_fake_checkout(first_root, "first")
    _write_fake_checkout(second_root, "second")
    original_sys_path = list(sys.path)
    original_external_modules = {
        name
        for name in sys.modules
        if name == "networks" or name.startswith("networks.")
    }

    _, _, first_configs = transunet._import_external_api(first_root)
    _, _, second_configs = transunet._import_external_api(second_root)

    assert first_configs["R50-ViT-B_16"].marker == "first"
    assert second_configs["R50-ViT-B_16"].marker == "second"
    assert sys.path == original_sys_path
    assert {
        name
        for name in sys.modules
        if name == "networks" or name.startswith("networks.")
    } == original_external_modules


def test_external_integration_runs_only_when_checkout_and_weights_are_configured():
    root_value = os.environ.get("TRANSUNET_ROOT")
    pretrained_value = os.environ.get("TRANSUNET_PRETRAINED")
    if not root_value or not pretrained_value:
        pytest.skip("set TRANSUNET_ROOT and TRANSUNET_PRETRAINED for external integration")

    root = Path(root_value)
    pretrained = Path(pretrained_value)
    if not root.is_dir() or not pretrained.is_file():
        pytest.skip("configured TransUNet checkout or pretrained file is unavailable")

    cfg = OmegaConf.create(
        {
            "model": {
                "in_channels": 4,
                "num_classes": 3,
                "image_size": 224,
                "vit_name": "R50-ViT-B_16",
            },
            "third_party": {
                "transunet_root": str(root),
                "pretrained_path": str(pretrained),
            },
        }
    )
    model = build_transunet(cfg)
    output = model(torch.zeros(1, 4, 224, 224))

    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 3, 224, 224)
