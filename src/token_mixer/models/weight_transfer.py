"""Pure 2-D CNN encoder to 3-D MetaUNETR weight transfer.

The source experiment is the repository's ImageNet denoising pre-training run
in ``pretrain_cnn.py``; its source model is ``PretrainCNNEncoder``, whose
hierarchy mirrors the CNN token-mixer path.
The target naming follows ``token_mixer.models.metaunetr.encoder.Encoder3D``.
The supported inflation semantics retain the explicit procedure in the legacy
``finetune_brats_mod_b.py`` transfer code while refusing ambiguous reshapes.

Provenance:

* Oyetunji et al., *Token Mixer Placement Matters: A Systematic Encoder-Decoder
  Ablation Study of Mamba for Brain Tumour Segmentation on BraTS-Africa*, the
  supplied project manuscript.
* Lyu et al., *MetaUNETR: Rethinking Token Mixer Encoding for Efficient
  Multi-Organ Segmentation*, MICCAI 2024,
  https://papers.miccai.org/miccai-2024/paper/2749_paper.pdf.
* Official MetaUNETR implementation, https://github.com/lyupengju/MetaUNETR.
* He et al., *Deep Residual Learning for Image Recognition*,
  https://arxiv.org/abs/1512.03385, and the official timm implementation,
  https://github.com/huggingface/pytorch-image-models.
* Source-model and legacy-transfer references: ``pretrain_cnn.py`` and
  ``finetune_brats_mod_b.py`` and ``finetune_nnunet_brats.py`` in this
  repository.

Only target keys that receive a direct or explicitly supported inflated copy
are returned. Missing and incompatible target keys are reported in a stable,
sorted warning and in the returned counts. The warning is intentional: callers
can capture it for an inspectable transfer report without relying on stdout.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F


MIN_COVERAGE = 0.5
TIMM_RESNET18_NAME = "resnet18.a1_in1k"
_RESUNET_STEM_KEY = "stages.0.blocks.0.conv1.weight"


class WeightTransferWarning(UserWarning):
    """A target tensor was not copied or supported for inflation."""


def inflate_encoder_state_dict(
    source: Mapping[str, Tensor],
    target: Mapping[str, Tensor],
) -> tuple[dict[str, Tensor], dict[str, int]]:
    """Map a 2-D encoder state dictionary into a 3-D encoder state dictionary.

    Rules are deliberately shape-based and name-assisted, never reshape-based:

    * Exact-shape tensors are detached and cloned directly. This covers
      compatible normalization parameters, linear weights/biases, and already
      compatible tensors reached through a known encoder-name alias.
    * A 4-D convolution kernel ``[out, in, height, width]`` becomes a 5-D
      kernel ``[out, in, depth, height, width]`` by repeating along depth and
      dividing by the depth size. Output/input channels and spatial kernel
      dimensions must match.
    * MRI stem kernels additionally adapt input channels by copying the three
      source channels and filling a fourth channel with their deterministic
      mean. Generic stems with more source channels than the target average
      deterministic contiguous source-channel bins, so no source filters are
      discarded. Extra target channels still use the deterministic source
      mean. Stem spatial kernels use an explicit centered crop or zero-pad;
      this is the only supported spatial-size adaptation.
    * Depthwise and standard convolution kernels are not interchangeable.
      A mismatch is skipped instead of copying a tensor with incompatible
      grouped-convolution semantics.

    The returned counts always contain ``direct``, ``inflated``, ``skipped``,
    and ``total``. They also contain ``missing_source`` and ``incompatible``
    so callers can distinguish absent source keys from rejected shapes.

    A target stem weight is a critical encoder entry point. If a recognizable
    target stem exists but is not transferred, this function raises. It also
    raises when copied target-key coverage is below ``MIN_COVERAGE`` (50%).
    Targets without a recognizable stem remain useful for isolated synthetic
    shape tests; the coverage guard still applies to them.
    """
    _validate_state_dict("source", source)
    _validate_state_dict("target", target)
    if not target:
        raise RuntimeError("target state dictionary is empty")

    transferred: dict[str, Tensor] = {}
    skipped: dict[str, str] = {}
    direct = 0
    inflated = 0
    missing_source = 0
    incompatible = 0

    for target_key, target_tensor in target.items():
        candidates = [
            candidate
            for candidate in _candidate_source_keys(target_key)
            if candidate in source
        ]
        if not candidates:
            missing_source += 1
            skipped[target_key] = "no matching source key"
            continue

        result, kind, reason = _resolve_transfer(
            target_key,
            target_tensor,
            candidates,
            source,
        )
        if result is None:
            incompatible += 1
            skipped[target_key] = reason
            continue

        transferred[target_key] = result
        if kind == "direct":
            direct += 1
        else:
            inflated += 1

    total = len(target)
    skipped_count = missing_source + incompatible
    counts = {
        "direct": direct,
        "inflated": inflated,
        "skipped": skipped_count,
        "total": total,
        "missing_source": missing_source,
        "incompatible": incompatible,
    }

    _warn_skipped(skipped)

    critical_missing = sorted(
        key for key in _critical_target_keys(target) if key not in transferred
    )
    copied = direct + inflated
    if critical_missing:
        raise RuntimeError(
            "critical encoder entry point(s) were not transferred: "
            f"{', '.join(critical_missing)}; copied {copied}/{total} target keys"
        )

    if copied / total < MIN_COVERAGE:
        raise RuntimeError(
            f"weight-transfer coverage {copied}/{total} is below required "
            f"{int(MIN_COVERAGE * 100)}%; skipped target keys: "
            f"{', '.join(sorted(skipped))}"
        )

    return transferred, counts


def _validate_state_dict(name: str, state: Mapping[str, Tensor]) -> None:
    if not isinstance(state, Mapping):
        raise TypeError(f"{name} must be a mapping of string keys to tensors")
    for key, value in state.items():
        if not isinstance(key, str):
            raise TypeError(f"{name} key {key!r} is not a string")
        if not isinstance(value, Tensor):
            raise TypeError(f"{name}[{key!r}] is not a torch.Tensor")


def _candidate_source_keys(target_key: str) -> list[str]:
    """Return deterministic aliases for current and legacy encoder names."""
    base = target_key.removeprefix("encoder.")
    candidates: list[str] = []

    def add(key: str) -> None:
        if key not in candidates:
            candidates.append(key)

    add(target_key)
    add(base)
    if not target_key.startswith("encoder."):
        add(f"encoder.{base}")

    for variant in _structural_key_variants(base):
        add(variant)
        add(f"encoder.{variant}")

    return candidates


def _structural_key_variants(key: str) -> list[str]:
    parts = key.split(".")
    bases = [parts]

    if len(parts) >= 4 and parts[0] == "stages":
        stage_index, block_index = parts[1], parts[2]
        if stage_index.isdigit():
            bases.append(
                [f"stage{int(stage_index) + 1}", block_index, *parts[3:]]
            )

    if len(parts) >= 3 and parts[0] == "downsamples":
        downsample_index = parts[1]
        if downsample_index.isdigit():
            component = "conv" if parts[2] == "reduction" else parts[2]
            bases.append([f"down{int(downsample_index) + 1}", component, *parts[3:]])

    if len(parts) >= 2 and parts[-2] == "stem" and parts[-1] in {"weight", "bias"}:
        bases.append([*parts[:-2], "stem", "2", parts[-1]])

    variants: list[str] = []
    for base in bases:
        for alias in _component_aliases(base):
            candidate = ".".join(alias)
            if candidate not in variants:
                variants.append(candidate)
    return variants


def _component_aliases(parts: list[str]) -> list[list[str]]:
    aliases = [parts]

    for old, new in (("dwconv", "conv1"), ("dw_conv", "conv1")):
        if old in parts:
            aliases.append([new if part == old else part for part in parts])

    if len(parts) >= 2:
        for old, new in (("fc1", "0"), ("fc2", "2")):
            for index in range(len(parts) - 1):
                if parts[index : index + 2] == ["mlp", old]:
                    aliases.append(
                        [
                            *parts[:index],
                            "mlp",
                            new,
                            *parts[index + 2 :],
                        ]
                    )
                    break

    return aliases


def _resolve_transfer(
    target_key: str,
    target_tensor: Tensor,
    candidates: list[str],
    source: Mapping[str, Tensor],
) -> tuple[Tensor | None, str | None, str]:
    first_inflated: Tensor | None = None
    reasons: list[str] = []

    for source_key in candidates:
        result, kind, reason = _try_transfer(
            source_key,
            source[source_key],
            target_key,
            target_tensor,
        )
        if result is None:
            reasons.append(f"{source_key}: {reason}")
        elif kind == "direct":
            return result, kind, ""
        elif first_inflated is None:
            first_inflated = result

    if first_inflated is not None:
        return first_inflated, "inflated", ""
    return None, None, "; ".join(reasons)


def _try_transfer(
    source_key: str,
    source_tensor: Tensor,
    target_key: str,
    target_tensor: Tensor,
) -> tuple[Tensor | None, str | None, str]:
    if source_tensor.shape == target_tensor.shape:
        mismatch = _convolution_kind_mismatch(
            source_key,
            source_tensor,
            target_key,
            target_tensor,
        )
        if mismatch is not None:
            return None, None, mismatch
        return source_tensor.detach().clone(), "direct", ""

    if source_tensor.ndim == 4 and target_tensor.ndim == 5:
        if _is_stem_weight(target_key):
            result, reason = _inflate_stem(source_tensor, target_tensor)
        else:
            result, reason = _inflate_convolution(
                source_key,
                source_tensor,
                target_key,
                target_tensor,
            )
        if result is not None:
            return result, "inflated", ""
        return None, None, reason

    return (
        None,
        None,
        f"shape {tuple(source_tensor.shape)} cannot map to "
        f"{tuple(target_tensor.shape)} without a supported rule",
    )


def _inflate_convolution(
    source_key: str,
    source_tensor: Tensor,
    target_key: str,
    target_tensor: Tensor,
) -> tuple[Tensor | None, str]:
    mismatch = _convolution_kind_mismatch(
        source_key,
        source_tensor,
        target_key,
        target_tensor,
    )
    if mismatch is not None:
        return None, mismatch

    if source_tensor.shape[:2] != target_tensor.shape[:2]:
        return (
            None,
            "convolution output/input channels differ: "
            f"{tuple(source_tensor.shape[:2])} vs {tuple(target_tensor.shape[:2])}",
        )
    if source_tensor.shape[-2:] != target_tensor.shape[-2:]:
        return (
            None,
            "non-stem convolution spatial kernels differ: "
            f"{tuple(source_tensor.shape[-2:])} vs "
            f"{tuple(target_tensor.shape[-2:])}",
        )
    depth = int(target_tensor.shape[2])
    if depth < 1:
        return None, "target convolution has non-positive depth kernel"

    return (
        source_tensor.detach().clone().unsqueeze(2).repeat(1, 1, depth, 1, 1)
        / depth,
        "",
    )


def _inflate_stem(
    source_tensor: Tensor,
    target_tensor: Tensor,
) -> tuple[Tensor | None, str]:
    if source_tensor.shape[0] != target_tensor.shape[0]:
        return (
            None,
            "stem output channels differ: "
            f"{source_tensor.shape[0]} vs {target_tensor.shape[0]}",
        )
    if source_tensor.shape[1] < 1:
        return None, "stem source has no input channels"
    if target_tensor.shape[1] < 1:
        return None, "stem target has no input channels"

    source_channels = source_tensor.detach().clone()
    target_channels = int(target_tensor.shape[1])
    source_count = int(source_channels.shape[1])
    if source_count == 3 and target_channels == 4:
        # Preserve the approved RGB-to-MRI adaptation rule exactly.
        mean_channel = source_channels.mean(dim=1, keepdim=True)
        adapted = torch.cat(
            (source_channels, mean_channel),
            dim=1,
        )
    elif source_count > target_channels:
        # tensor_split partitions every source channel into stable,
        # left-to-right bins, including the non-divisible case.
        source_bins = torch.tensor_split(source_channels, target_channels, dim=1)
        adapted = torch.cat(
            [source_bin.mean(dim=1, keepdim=True) for source_bin in source_bins],
            dim=1,
        )
    else:
        extra = target_channels - source_count
        mean_channel = source_channels.mean(dim=1, keepdim=True)
        adapted = torch.cat(
            (source_channels, mean_channel.expand(-1, extra, -1, -1)),
            dim=1,
        )

    adapted = _adapt_stem_spatial(
        adapted,
        int(target_tensor.shape[3]),
        int(target_tensor.shape[4]),
    )
    depth = int(target_tensor.shape[2])
    if depth < 1:
        return None, "stem target has non-positive depth kernel"

    return adapted.unsqueeze(2).repeat(1, 1, depth, 1, 1) / depth, ""


def _adapt_stem_spatial(kernel: Tensor, target_height: int, target_width: int) -> Tensor:
    """Center-crop or symmetrically zero-pad stem spatial dimensions."""
    height, width = kernel.shape[-2:]
    if height > target_height:
        start = (height - target_height) // 2
        kernel = kernel[..., start : start + target_height, :]
    if width > target_width:
        start = (width - target_width) // 2
        kernel = kernel[..., :, start : start + target_width]

    pad_height = target_height - kernel.shape[-2]
    pad_width = target_width - kernel.shape[-1]
    if pad_height > 0 or pad_width > 0:
        top = pad_height // 2
        bottom = pad_height - top
        left = pad_width // 2
        right = pad_width - left
        kernel = F.pad(kernel, (left, right, top, bottom))
    return kernel


def _is_stem_weight(key: str) -> bool:
    return bool(re.search(r"(?:^|\.)stem\.weight$", key))


def _critical_target_keys(target: Mapping[str, Tensor]) -> list[str]:
    return sorted(key for key in target if _is_stem_weight(key))


def _convolution_kind_mismatch(
    source_key: str,
    source_tensor: Tensor,
    target_key: str,
    target_tensor: Tensor,
) -> str | None:
    if _is_stem_weight(target_key):
        return None
    source_kind = _convolution_kind(source_key, source_tensor)
    target_kind = _convolution_kind(target_key, target_tensor)
    if source_kind is not None and target_kind is not None and source_kind != target_kind:
        return (
            "depthwise/standard convolution mismatch: "
            f"source={source_kind}, target={target_kind}"
        )
    return None


def _convolution_kind(key: str, tensor: Tensor) -> str | None:
    if tensor.ndim not in {4, 5} or not key.endswith(".weight"):
        return None
    components = set(key.lower().split("."))
    if {"dwconv", "dw_conv", "depthwise"} & components:
        return "depthwise"
    if "conv" in components or "conv1" in components or "reduction" in components:
        return "standard"
    if tensor.shape[1] == 1:
        return "depthwise"
    return None


def _resnet18_target_sources() -> dict[str, str]:
    """Return the explicit legacy ResNet-18 to ResUNet encoder map."""
    mapping = {
        _RESUNET_STEM_KEY: "conv1.weight",
    }
    for suffix in (
        "weight",
        "bias",
        "running_mean",
        "running_var",
        "num_batches_tracked",
    ):
        mapping[f"stages.0.blocks.0.norm1.{suffix}"] = f"bn1.{suffix}"

    def add_block(source_prefix: str, target_prefix: str) -> None:
        for component in ("conv1", "conv2"):
            mapping[f"{target_prefix}.{component}.weight"] = (
                f"{source_prefix}.{component}.weight"
            )
        for component in ("norm1", "norm2"):
            source_norm = component.replace("norm", "bn")
            for suffix in (
                "weight",
                "bias",
                "running_mean",
                "running_var",
                "num_batches_tracked",
            ):
                mapping[f"{target_prefix}.{component}.{suffix}"] = (
                    f"{source_prefix}.{source_norm}.{suffix}"
                )

    add_block("layer1.1", "stages.1.blocks.1")
    for stage in (2, 3, 4):
        for block in (0, 1):
            add_block(
                f"layer{stage}.{block}",
                f"stages.{stage}.blocks.{block}",
            )
        mapping[f"stages.{stage}.blocks.0.skip.0.weight"] = (
            f"layer{stage}.0.downsample.0.weight"
        )
        for suffix in (
            "weight",
            "bias",
            "running_mean",
            "running_var",
            "num_batches_tracked",
        ):
            mapping[f"stages.{stage}.blocks.0.skip.1.{suffix}"] = (
                f"layer{stage}.0.downsample.1.{suffix}"
            )
    return mapping


def _adapt_resnet_stem(
    source: Tensor,
    target_shape: torch.Size,
) -> Tensor | None:
    """Adapt ResNet-18's RGB 7x7 kernel to a ResUNet 3-D stem."""
    if source.ndim != 4 or len(target_shape) != 5:
        return None
    source_channels = source.detach().clone()
    if not (
        torch.is_floating_point(source_channels)
        or torch.is_complex(source_channels)
    ):
        source_channels = source_channels.float()
    target_out, target_in, target_depth, target_height, target_width = map(
        int, target_shape
    )
    if target_depth < 1 or target_height < 1 or target_width < 1:
        return None

    if source_channels.shape[0] > target_out:
        output_bins = torch.tensor_split(source_channels, target_out, dim=0)
        source_channels = torch.cat(
            [output_bin.mean(dim=0, keepdim=True) for output_bin in output_bins],
            dim=0,
        )
    elif source_channels.shape[0] < target_out:
        extra = target_out - source_channels.shape[0]
        mean_output = source_channels.mean(dim=0, keepdim=True)
        source_channels = torch.cat(
            (source_channels, mean_output.expand(extra, -1, -1, -1)),
            dim=0,
        )

    source_count = source_channels.shape[1]
    if source_count > target_in:
        input_bins = torch.tensor_split(source_channels, target_in, dim=1)
        source_channels = torch.cat(
            [input_bin.mean(dim=1, keepdim=True) for input_bin in input_bins],
            dim=1,
        )
    elif source_count < target_in:
        extra = target_in - source_count
        mean_input = source_channels.mean(dim=1, keepdim=True)
        source_channels = torch.cat(
            (source_channels, mean_input.expand(-1, extra, -1, -1)),
            dim=1,
        )

    source_height, source_width = source_channels.shape[-2:]
    if source_height < target_height or source_width < target_width:
        return None
    height_start = (source_height - target_height) // 2
    width_start = (source_width - target_width) // 2
    source_channels = source_channels[
        ...,
        height_start : height_start + target_height,
        width_start : width_start + target_width,
    ]
    return (
        source_channels.unsqueeze(2).repeat(1, 1, target_depth, 1, 1)
        / target_depth
    )


def _adapt_resnet_vector(source: Tensor, target_shape: torch.Size) -> Tensor | None:
    if source.ndim != 1 or len(target_shape) != 1:
        return None
    target_size = int(target_shape[0])
    if target_size < 1:
        return None
    values = source.detach().clone()
    if not (torch.is_floating_point(values) or torch.is_complex(values)):
        values = values.float()
    if values.shape[0] > target_size:
        bins = torch.tensor_split(values, target_size, dim=0)
        return torch.stack([item.mean() for item in bins])
    if values.shape[0] < target_size:
        mean = values.mean().reshape(1)
        return torch.cat((values, mean.expand(target_size - values.shape[0])))
    return values


def _resnet18_transfer_value(
    source_key: str,
    source: Tensor,
    target_key: str,
    target: Tensor,
) -> tuple[Tensor | None, str | None, str]:
    if source.shape == target.shape:
        return source.detach().clone(), "direct", ""

    if target_key == _RESUNET_STEM_KEY:
        result = _adapt_resnet_stem(source, target.shape)
        if result is not None and result.shape == target.shape:
            return result, "adapted", ""
        return None, None, "ResNet-18 stem cannot adapt to target shape"

    if target_key.startswith("stages.0.blocks.0.norm1."):
        result = _adapt_resnet_vector(source, target.shape)
        if result is not None and result.shape == target.shape:
            return result, "adapted", ""
        return None, None, "ResNet-18 stem normalization cannot adapt to target shape"

    if source.ndim == 4 and target.ndim == 5:
        result, reason = _inflate_convolution(
            source_key,
            source,
            target_key,
            target,
        )
        if result is not None:
            return result, "inflated", ""
        return None, None, reason

    return (
        None,
        None,
        f"shape {tuple(source.shape)} cannot map to "
        f"{tuple(target.shape)} without a supported ResNet-18 rule",
    )


def transfer_resnet18_encoder_state_dict(
    source: Mapping[str, Tensor],
    target: Mapping[str, Tensor],
) -> tuple[dict[str, Tensor], dict[str, int | float]]:
    """Transfer explicit timm ResNet-18 encoder tensors into a ResUNet encoder."""
    _validate_state_dict("source", source)
    _validate_state_dict("target", target)
    if not target:
        raise RuntimeError("target ResUNet encoder state dictionary is empty")

    target_sources = _resnet18_target_sources()
    transferred: dict[str, Tensor] = {}
    skipped: dict[str, str] = {}
    direct = 0
    inflated = 0
    adapted = 0
    missing_source = 0
    incompatible = 0
    unmapped = 0

    for target_key, target_tensor in target.items():
        source_key = target_sources.get(target_key)
        if source_key is None:
            unmapped += 1
            skipped[target_key] = "target key is outside the legacy ResNet-18 map"
            continue
        if source_key not in source:
            missing_source += 1
            skipped[target_key] = f"missing source key {source_key}"
            continue

        result, kind, reason = _resnet18_transfer_value(
            source_key,
            source[source_key],
            target_key,
            target_tensor,
        )
        if result is None:
            incompatible += 1
            skipped[target_key] = reason
            continue
        transferred[target_key] = result
        if kind == "direct":
            direct += 1
        elif kind == "inflated":
            inflated += 1
        else:
            adapted += 1

    copied = direct + inflated + adapted
    total = len(target)
    counts: dict[str, int | float] = {
        "direct": direct,
        "inflated": inflated,
        "adapted": adapted,
        "copied": copied,
        "skipped": len(skipped),
        "total": total,
        "coverage": copied / total,
        "missing_source": missing_source,
        "incompatible": incompatible,
        "unmapped": unmapped,
    }
    _warn_skipped(
        {
            key: reason
            for key, reason in skipped.items()
            if key in target_sources
        }
    )

    if _RESUNET_STEM_KEY in target and _RESUNET_STEM_KEY not in transferred:
        raise RuntimeError(
            "critical ResUNet stem was not transferred; "
            f"copied {copied}/{total} target keys"
        )
    if copied / total < MIN_COVERAGE:
        raise RuntimeError(
            f"ResNet-18 transfer coverage {copied}/{total} is below required "
            f"{int(MIN_COVERAGE * 100)}%"
        )
    return transferred, counts


def _create_timm_resnet18(cache_dir: str | Path | None = None) -> nn.Module:
    try:
        import timm
    except ImportError as exc:
        raise ImportError(
            "ImageNet ResNet-18 transfer requires optional timm; "
            "install the research extra with `uv sync --extra research`"
        ) from exc

    kwargs: dict[str, Any] = {
        "pretrained": True,
        "num_classes": 0,
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    return timm.create_model(TIMM_RESNET18_NAME, **kwargs)


def load_imagenet_resnet18_weights(
    model: nn.Module,
    cache_dir: str | Path | None = None,
    *,
    source_model: nn.Module | Mapping[str, Tensor] | None = None,
    download: bool = False,
) -> dict[str, int | float]:
    """Load or inject ImageNet ResNet-18 weights into ``model.encoder``.

    ``source_model`` keeps tests and offline runs network-free. When omitted,
    timm is imported and pretrained weights are created only with
    ``download=True``; the default never contacts the network.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("ResUNet3D model must be a torch.nn.Module")
    encoder = getattr(model, "encoder", None)
    if not isinstance(encoder, nn.Module):
        raise TypeError("ResUNet3D model must expose an nn.Module encoder")
    encoder = cast(nn.Module, encoder)
    if not isinstance(download, bool):
        raise TypeError("download must be a boolean")

    if source_model is None:
        if not download:
            raise RuntimeError(
                "ImageNet ResNet-18 loading is disabled by default; "
                "pass download=True to allow timm checkpoint loading"
            )
        source_model = _create_timm_resnet18(cache_dir)

    if isinstance(source_model, Mapping):
        source_state = source_model
    else:
        source_object: Any = source_model
        state_dict: Any = getattr(source_object, "state_dict", None)
        if not callable(state_dict):
            raise TypeError("source_model must be an nn.Module or tensor mapping")
        if isinstance(source_object, nn.Module):
            source_object.eval()
        source_state = cast(Mapping[str, Tensor], state_dict())

    transferred, counts = transfer_resnet18_encoder_state_dict(
        source_state,
        encoder.state_dict(),
    )
    encoder.load_state_dict(transferred, strict=False)
    return counts


def _warn_skipped(skipped: Mapping[str, str]) -> None:
    if not skipped:
        return
    details = "; ".join(
        f"{key}: {skipped[key]}" for key in sorted(skipped)
    )
    warnings.warn(
        f"weight transfer skipped {len(skipped)} target key(s): {details}",
        WeightTransferWarning,
        stacklevel=2,
    )


__all__ = [
    "MIN_COVERAGE",
    "TIMM_RESNET18_NAME",
    "WeightTransferWarning",
    "inflate_encoder_state_dict",
    "load_imagenet_resnet18_weights",
    "transfer_resnet18_encoder_state_dict",
]
