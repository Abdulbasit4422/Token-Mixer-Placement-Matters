from __future__ import annotations

import pytest
import torch

from token_mixer.models.weight_transfer import (
    WeightTransferWarning,
    inflate_encoder_state_dict,
)


def test_copies_compatible_normalization_and_linear_tensors_directly():
    source = {
        "stem.weight": torch.arange(2.0).reshape(2, 1, 1, 1, 1),
        "norm.weight": torch.arange(3.0),
        "norm.bias": torch.arange(3.0) + 10,
        "projection.weight": torch.arange(6.0).reshape(2, 3),
        "projection.bias": torch.arange(2.0) + 20,
    }
    target = {key: torch.zeros_like(value) for key, value in source.items()}

    transferred, counts = inflate_encoder_state_dict(source, target)

    assert list(transferred) == list(target)
    for key in target:
        assert torch.equal(transferred[key], source[key])
        assert transferred[key] is not source[key]
    assert counts["direct"] == 5
    assert counts["inflated"] == 0
    assert counts["skipped"] == 0
    assert counts["total"] == 5


def test_maps_pretrain_names_to_current_encoder_names():
    source = {
        "stem.2.weight": torch.arange(2 * 3 * 3 * 3.0).reshape(2, 3, 3, 3),
        "stage1.0.norm1.weight": torch.arange(2.0),
        "stage1.0.mlp.0.weight": torch.arange(8.0).reshape(4, 2),
        "down1.conv.weight": torch.arange(4 * 2 * 2 * 2.0).reshape(4, 2, 2, 2),
    }
    target = {
        "encoder.stem.weight": torch.zeros(2, 4, 2, 2, 2),
        "encoder.stages.0.0.norm1.weight": torch.zeros(2),
        "encoder.stages.0.0.mlp.fc1.weight": torch.zeros(4, 2),
        "encoder.downsamples.0.reduction.weight": torch.zeros(4, 2, 2, 2, 2),
    }

    transferred, counts = inflate_encoder_state_dict(source, target)

    assert set(transferred) == set(target)
    assert counts["direct"] == 2
    assert counts["inflated"] == 2
    assert counts["skipped"] == 0
    assert counts["total"] == 4


def test_inflates_standard_two_dimensional_kernel_across_depth():
    kernel = torch.arange(2 * 3 * 2 * 2.0).reshape(2, 3, 2, 2)
    source = {"block.conv.weight": kernel}
    target = {"block.conv.weight": torch.zeros(2, 3, 3, 2, 2)}

    transferred, counts = inflate_encoder_state_dict(source, target)

    expected = kernel.unsqueeze(2).repeat(1, 1, 3, 1, 1) / 3
    assert torch.equal(transferred["block.conv.weight"], expected)
    assert counts["direct"] == 0
    assert counts["inflated"] == 1
    assert counts["skipped"] == 0
    assert counts["total"] == 1


def test_adapts_mri_stem_channels_and_center_crops_spatial_kernel():
    kernel = torch.arange(2 * 3 * 3 * 3.0).reshape(2, 3, 3, 3)
    source = {"stem.2.weight": kernel}
    target = {"stem.weight": torch.zeros(2, 4, 2, 2, 2)}

    transferred, counts = inflate_encoder_state_dict(source, target)

    cropped = kernel[:, :, :2, :2]
    extra_channel = cropped.mean(dim=1, keepdim=True)
    adapted = torch.cat((cropped, extra_channel), dim=1)
    expected = adapted.unsqueeze(2).repeat(1, 1, 2, 1, 1) / 2
    assert torch.equal(transferred["stem.weight"], expected)
    assert counts["inflated"] == 1
    assert counts["direct"] == 0
    assert counts["skipped"] == 0


def test_adapts_generic_stem_by_averaging_all_source_channel_bins():
    kernel = torch.arange(2 * 16.0).reshape(2, 16, 1, 1)
    source = {"stem.weight": kernel}
    target = {"stem.weight": torch.zeros(2, 4, 1, 1, 1)}

    transferred, counts = inflate_encoder_state_dict(source, target)

    expected_channels = torch.stack(
        [kernel[:, start : start + 4].mean(dim=1) for start in range(0, 16, 4)],
        dim=1,
    )
    expected = expected_channels.unsqueeze(2)
    assert torch.equal(transferred["stem.weight"], expected)
    assert counts["inflated"] == 1
    assert counts["skipped"] == 0


def test_adapts_uneven_generic_stem_bins_without_dropping_channels():
    kernel = torch.arange(5.0).reshape(1, 5, 1, 1)
    source = {"stem.weight": kernel}
    target = {"stem.weight": torch.zeros(1, 2, 1, 1, 1)}

    transferred, _ = inflate_encoder_state_dict(source, target)

    expected_channels = torch.tensor([1.0, 3.5]).reshape(1, 2, 1, 1)
    assert torch.equal(transferred["stem.weight"].squeeze(2), expected_channels)


def test_rejects_standard_to_depthwise_convolution_transfer():
    source = {
        "stage1.0.conv1.weight": torch.ones(4, 4, 3, 3),
        "norm.weight": torch.ones(4),
    }
    target = {
        "stages.0.0.dwconv.weight": torch.zeros(4, 1, 2, 3, 3),
        "norm.weight": torch.zeros(4),
    }

    with pytest.warns(WeightTransferWarning, match=r"stages\.0\.0\.dwconv\.weight"):
        transferred, counts = inflate_encoder_state_dict(source, target)

    assert set(transferred) == {"norm.weight"}
    assert counts["direct"] == 1
    assert counts["inflated"] == 0
    assert counts["skipped"] == 1
    assert counts["missing_source"] == 0
    assert counts["incompatible"] == 1
    assert counts["total"] == 2


def test_skips_incompatible_shapes_without_reshaping():
    source = {
        "stem.weight": torch.ones(2, 1, 1, 1, 1),
        "linear.weight": torch.arange(6.0).reshape(2, 3),
    }
    target = {
        "stem.weight": torch.zeros(2, 1, 1, 1, 1),
        "linear.weight": torch.zeros(3, 2),
    }

    with pytest.warns(WeightTransferWarning, match=r"linear\.weight"):
        transferred, counts = inflate_encoder_state_dict(source, target)

    assert set(transferred) == {"stem.weight"}
    assert counts["direct"] == 1
    assert counts["inflated"] == 0
    assert counts["skipped"] == 1
    assert counts["missing_source"] == 0
    assert counts["incompatible"] == 1


def test_reports_missing_keys_in_sorted_warning_and_counts():
    source = {
        "stem.weight": torch.ones(1, 1, 1, 1, 1),
        "present.weight": torch.ones(1),
    }
    target = {
        "z_missing.weight": torch.zeros(1),
        "stem.weight": torch.zeros(1, 1, 1, 1, 1),
        "present.weight": torch.zeros(1),
        "a_missing.weight": torch.zeros(1),
    }

    with pytest.warns(WeightTransferWarning) as caught:
        transferred, counts = inflate_encoder_state_dict(source, target)

    message = str(caught[0].message)
    assert message.index("a_missing.weight") < message.index("z_missing.weight")
    assert set(transferred) == {"stem.weight", "present.weight"}
    assert counts["direct"] == 2
    assert counts["inflated"] == 0
    assert counts["skipped"] == 2
    assert counts["missing_source"] == 2
    assert counts["incompatible"] == 0
    assert counts["total"] == 4


def test_raises_when_critical_stem_entry_point_is_not_transferred():
    source = {"unrelated.weight": torch.ones(2)}
    target = {"stem.weight": torch.zeros(2, 4, 1, 1, 1)}

    with pytest.warns(WeightTransferWarning, match=r"stem\.weight"):
        with pytest.raises(RuntimeError, match=r"critical.*stem\.weight"):
            inflate_encoder_state_dict(source, target)


def test_raises_when_target_coverage_is_below_fifty_percent():
    source = {"stem.weight": torch.ones(1, 1, 1, 1, 1)}
    target = {
        "stem.weight": torch.zeros(1, 1, 1, 1, 1),
        "missing_a.weight": torch.zeros(1),
        "missing_b.weight": torch.zeros(1),
    }

    with pytest.warns(WeightTransferWarning, match=r"missing_a\.weight"):
        with pytest.raises(RuntimeError, match="coverage.*1/3.*50%"):
            inflate_encoder_state_dict(source, target)
