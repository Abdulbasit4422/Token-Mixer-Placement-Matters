from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from token_mixer.models.cnn_pretrain import (
    DenoisingAutoencoder,
    DenoisingDataset,
    build_denoising_model,
    compute_psnr,
    evaluate_denoising,
    mse_loss,
)


def test_denoising_model_reads_nested_config_and_preserves_image_shape():
    cfg = OmegaConf.create(
        {
            "model": {
                "in_channels": 2,
                "feature_size": 4,
                "depths": [1, 0, 0, 0],
                "norm_num_groups": 4,
            },
            "data": {"image_size": 32},
        }
    )

    model = build_denoising_model(cfg)
    output = model(torch.rand(2, 2, 32, 32))

    assert isinstance(model, DenoisingAutoencoder)
    assert output.shape == (2, 2, 32, 32)
    assert torch.all((0.0 <= output) & (output <= 1.0))


def test_denoising_model_defaults_match_root_rgb_imagenet_pretraining():
    model = build_denoising_model(OmegaConf.create({}))

    assert model.config["in_channels"] == 3
    assert model.config["feature_size"] == 32
    assert model.config["depths"] == (1, 1, 1, 1)
    assert model.config["image_size"] == (96, 96)


def test_denoising_model_rejects_input_channel_and_normalization_shape_mismatches():
    model = build_denoising_model(
        OmegaConf.create(
            {
                "in_channels": 2,
                "feature_size": 4,
                "depths": [0, 0, 0, 0],
                "image_size": 32,
            }
        )
    )

    with pytest.raises(ValueError, match="channels"):
        model(torch.rand(1, 3, 32, 32))

    with pytest.raises(ValueError, match="normalization"):
        build_denoising_model(
            OmegaConf.create(
                {
                    "in_channels": 2,
                    "feature_size": 4,
                    "depths": [0, 0, 0, 0],
                    "norm_num_groups": 3,
                    "image_size": 32,
                }
            )
        )


def test_denoising_dataset_returns_clean_target_and_bounded_noisy_input():
    images = torch.tensor(
        [
            [[[0.0, 0.5], [1.0, 0.25]]],
            [[[0.2, 0.4], [0.6, 0.8]]],
        ]
    )
    dataset = DenoisingDataset(TensorDataset(images, torch.zeros(2)), noise_std=1.0)

    torch.manual_seed(7)
    noisy, clean = dataset[0]

    assert torch.equal(clean, images[0])
    assert noisy.shape == clean.shape
    assert torch.all((0.0 <= noisy) & (noisy <= 1.0))
    assert not torch.equal(noisy, clean)


def test_mse_and_psnr_match_unit_range_behavior():
    prediction = torch.zeros(1, 1, 2, 2)
    target = torch.ones_like(prediction)

    assert mse_loss(prediction, target).item() == pytest.approx(1.0)
    assert compute_psnr(1.0) == pytest.approx(0.0)
    assert compute_psnr(0.0) == pytest.approx(100.0)


def test_denoising_evaluator_reports_mean_mse_and_psnr():
    class Identity(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    noisy = torch.zeros(2, 1, 2, 2)
    clean = torch.ones_like(noisy)
    metrics = evaluate_denoising(
        Identity(), DataLoader(TensorDataset(noisy, clean), batch_size=1)
    )

    assert metrics == {"mse": pytest.approx(1.0), "psnr": pytest.approx(0.0)}
