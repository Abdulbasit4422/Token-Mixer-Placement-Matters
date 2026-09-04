from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from token_mixer.models.resunet3d import build_resunet3d
from token_mixer.models.swinunetr import build_swinunetr


def test_tiny_resunet3d_cpu_forward_preserves_canonical_shape():
    model = build_resunet3d(
        OmegaConf.create(
            {
                "model": {
                    "in_channels": 4,
                    "out_channels": 3,
                    "base_features": 2,
                    "depths": [1, 1, 1, 1, 1],
                    "normalization": "group",
                    "norm_num_groups": 1,
                }
            }
        )
    ).eval()

    with torch.no_grad():
        logits = model(torch.randn(1, 4, 16, 16, 16))

    assert logits.shape == (1, 3, 16, 16, 16)
    assert torch.isfinite(logits).all()
    assert model.output_regions == ("ET", "TC", "WT")


def test_swinunetr_cpu_forward_preserves_canonical_shape_when_imaging_deps_exist():
    pytest.importorskip("monai")
    pytest.importorskip("einops")

    model = build_swinunetr(
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
