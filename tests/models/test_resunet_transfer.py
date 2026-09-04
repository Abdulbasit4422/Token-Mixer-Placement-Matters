from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from token_mixer.models.resunet3d import build_resunet3d
from token_mixer.models import weight_transfer


def _model():
    return build_resunet3d(
        OmegaConf.create(
            {
                "model": {
                    "in_channels": 4,
                    "out_channels": 3,
                    "base_features": 32,
                    "depths": [2, 2, 2, 2, 2],
                    "normalization": "instance",
                }
            }
        )
    )


def _bn_state(
    source: dict[str, torch.Tensor],
    prefix: str,
    channels: int,
    value: float,
) -> None:
    source[f"{prefix}.weight"] = torch.full((channels,), value, dtype=torch.float32)
    source[f"{prefix}.bias"] = torch.full(
        (channels,), value + 1, dtype=torch.float32
    )


def _resnet_block_state(
    source: dict[str, torch.Tensor],
    prefix: str,
    in_channels: int,
    out_channels: int,
    value: float,
    downsample: bool = False,
) -> None:
    source[f"{prefix}.conv1.weight"] = torch.full(
        (out_channels, in_channels, 3, 3), value
    )
    _bn_state(source, f"{prefix}.bn1", out_channels, value + 10)
    source[f"{prefix}.conv2.weight"] = torch.full(
        (out_channels, out_channels, 3, 3), value + 1
    )
    _bn_state(source, f"{prefix}.bn2", out_channels, value + 20)
    if downsample:
        source[f"{prefix}.downsample.0.weight"] = torch.full(
            (out_channels, in_channels, 1, 1), value + 2
        )
        _bn_state(source, f"{prefix}.downsample.1", out_channels, value + 30)


def _source_state() -> dict[str, torch.Tensor]:
    source: dict[str, torch.Tensor] = {
        "conv1.weight": torch.arange(64 * 3 * 7 * 7, dtype=torch.float32).reshape(
            64, 3, 7, 7
        )
    }
    _bn_state(source, "bn1", 64, 100)
    _resnet_block_state(source, "layer1.1", 64, 64, 200)
    for stage, (in_channels, out_channels, value) in {
        2: (64, 128, 300),
        3: (128, 256, 400),
        4: (256, 512, 500),
    }.items():
        _resnet_block_state(
            source,
            f"layer{stage}.0",
            in_channels,
            out_channels,
            value,
            downsample=True,
        )
        _resnet_block_state(
            source,
            f"layer{stage}.1",
            out_channels,
            out_channels,
            value + 50,
        )
    return source


def test_resnet18_transfer_maps_blocks_norm_affine_and_reports_coverage():
    model = _model()

    counts = weight_transfer.load_imagenet_resnet18_weights(
        model,
        source_model=_source_state(),
    )

    state = model.encoder.state_dict()
    expected_layer2 = torch.full((128, 64, 3, 3), 300.0)
    expected_layer2 = expected_layer2.unsqueeze(2).repeat(1, 1, 3, 1, 1) / 3

    assert torch.equal(state["stages.2.blocks.0.conv1.weight"], expected_layer2)
    assert torch.equal(
        state["stages.2.blocks.0.norm1.weight"], torch.full((128,), 310.0)
    )
    assert torch.equal(
        state["stages.2.blocks.0.skip.0.weight"],
        torch.full((128, 64, 1, 1, 1), 302.0),
    )
    assert counts["copied"] == (
        counts["direct"] + counts["inflated"] + counts["adapted"]
    )
    assert counts["coverage"] == pytest.approx(
        counts["copied"] / counts["total"]
    )
    assert counts["coverage"] > 0.5


def test_resnet18_transfer_adapts_rgb_stem_output_input_spatial_and_depth():
    model = _model()
    source = _source_state()

    counts = weight_transfer.load_imagenet_resnet18_weights(
        model,
        source_model=source,
    )

    cropped = source["conv1.weight"][:, :, 2:5, 2:5]
    reduced_output = torch.stack(
        [cropped[index : index + 2].mean(dim=0) for index in range(0, 64, 2)]
    )
    mean_input = reduced_output.mean(dim=1, keepdim=True)
    expected = torch.cat((reduced_output, mean_input), dim=1)
    expected = expected.unsqueeze(2).repeat(1, 1, 3, 1, 1) / 3

    assert torch.equal(model.encoder.state_dict()["stages.0.blocks.0.conv1.weight"], expected)
    assert counts["adapted"] >= 1


def test_resnet18_transfer_refuses_implicit_pretrained_download():
    with pytest.raises(RuntimeError, match="download=True"):
        weight_transfer.load_imagenet_resnet18_weights(_model())


def test_resnet18_transfer_forwards_explicit_download_request_lazily(monkeypatch):
    source = _source_state()
    calls: list[dict[str, object]] = []

    class FakeSource(nn.Module):
        def state_dict(self):
            return source

    def create_model(name: str, **kwargs: object) -> FakeSource:
        calls.append({"name": name, **kwargs})
        return FakeSource()

    monkeypatch.setitem(sys.modules, "timm", SimpleNamespace(create_model=create_model))

    counts = weight_transfer.load_imagenet_resnet18_weights(
        _model(),
        cache_dir="cache",
        download=True,
    )

    assert calls == [
        {
            "name": "resnet18.a1_in1k",
            "pretrained": True,
            "num_classes": 0,
            "cache_dir": "cache",
        }
    ]
    assert counts["coverage"] > 0.5
