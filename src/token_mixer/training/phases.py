from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from torch import nn


@dataclass(frozen=True)
class PhaseSpec:
    """Training settings shared by encoder-freezing phases."""

    name: str
    epochs: int
    freeze_encoder: bool
    encoder_lr: float
    decoder_lr: float


def apply_phase(model: nn.Module, phase: PhaseSpec) -> None:
    """Apply encoder freezing while leaving optimizer grouping to the pipeline."""
    encoder = getattr(model, "encoder", None)
    if not isinstance(encoder, nn.Module):
        raise ValueError("model must expose an nn.Module named 'encoder'")
    encoder_module = cast(nn.Module, encoder)
    for parameter in encoder_module.parameters():
        parameter.requires_grad = not phase.freeze_encoder
