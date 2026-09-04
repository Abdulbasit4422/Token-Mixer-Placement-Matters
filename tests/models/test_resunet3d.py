from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from token_mixer.models.resunet3d import build_resunet3d
from token_mixer.models.weight_transfer import inflate_encoder_state_dict


def _config(**model_overrides: object):
    model = {
        "in_channels": 4,
        "out_channels": 3,
        "base_features": 2,
        "depths": [1, 1, 1, 1, 1],
        "normalization": "group",
        "norm_num_groups": 1,
        "spatial_size": [16, 16, 16],
    }
    model.update(model_overrides)
    return OmegaConf.create({"model": model})


def test_resunet3d_emits_canonical_raw_logits_and_preserves_spatial_shape():
    model = build_resunet3d(_config()).eval()
    inputs = torch.randn(1, 4, 16, 16, 16)

    with torch.no_grad():
        logits = model(inputs)

    assert logits.shape == (1, 3, 16, 16, 16)
    assert model.output_regions == ("ET", "TC", "WT")
    assert model.out_channels == len(model.output_regions)


def test_resunet3d_uses_configured_widths_depths_and_normalization():
    model = build_resunet3d(
        _config(
            base_features=3,
            depths=[2, 1, 3, 1, 2],
            normalization="instance",
            spatial_size=[32, 32, 32],
        )
    )

    assert model.widths == (3, 6, 12, 24, 48)
    assert tuple(len(stage.blocks) for stage in model.encoder.stages) == (2, 1, 3, 1, 2)
    assert all(
        isinstance(module, nn.InstanceNorm3d)
        for module in model.encoder.modules()
        if isinstance(module, (nn.InstanceNorm3d, nn.GroupNorm))
    )


def test_resunet3d_encoder_state_is_public_and_transferable():
    model = build_resunet3d(_config()).eval()
    target = model.encoder.state_dict()
    source = {key: value.clone() for key, value in target.items()}

    transferred, counts = inflate_encoder_state_dict(source, target)

    assert isinstance(model.encoder, nn.Module)
    assert set(transferred) == set(target)
    assert counts["direct"] == len(target)
    assert counts["inflated"] == 0
    model.encoder.load_state_dict(transferred)


def test_resunet3d_parameter_group_helpers_return_legacy_compatible_lists():
    model = build_resunet3d(_config())

    assert model.encoder_params() == list(model.encoder.parameters())
    assert model.decoder_params() == [
        *model.decoder.parameters(),
        *model.head.parameters(),
    ]


def test_resunet3d_tiny_cpu_backward_has_finite_input_and_parameter_gradients():
    model = build_resunet3d(_config())
    inputs = torch.randn(1, 4, 16, 16, 16, requires_grad=True)

    loss = model(inputs).square().mean()
    loss.backward()

    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert gradients
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"base_features": 0}, "positive"),
        ({"depths": [1, 1, 0, 1, 1]}, "depths"),
        ({"depths": [1, 1, 1, 1]}, "depths"),
        ({"spatial_size": [16, 16]}, "spatial"),
        ({"out_channels": 2}, "out_channels"),
    ],
)
def test_resunet3d_rejects_invalid_model_configuration(
    overrides: dict[str, object], message: str
):
    with pytest.raises(ValueError, match=message):
        build_resunet3d(_config(**overrides))


def test_resunet3d_rejects_input_channel_and_rank_mismatches():
    model = build_resunet3d(_config()).eval()

    with pytest.raises(ValueError, match="input channels"):
        model(torch.randn(1, 3, 16, 16, 16))

    with pytest.raises(ValueError, match="expects"):
        model(torch.randn(1, 4, 16, 16))
