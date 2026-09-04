"""2-D slice-based adapter for the official TransUNet implementation.

This module preserves the legacy R50-ViT-B/16 TransUNet path from this project
without presenting it as a native 3-D model. It accepts ``[B, C, H, W]`` slices
and returns canonical raw region logits ``[B, 3, H, W]`` in ``ET, TC, WT``
order. The external API is the official ``Beckschen/TransUNet`` checkout:
``networks.vit_seg_modeling.VisionTransformer``, its ``CONFIGS`` mapping,
``VisionTransformer.load_from(weights)``, and
``networks.vit_seg_modeling_resnet_skip.StdConv2d``.

Reference: Chen et al., *TransUNet: Transformers Make Strong Encoders for
Medical Image Segmentation*, https://arxiv.org/abs/2102.04306; official
implementation: https://github.com/Beckschen/TransUNet.

External modules are imported only inside :func:`build_transunet`, after
configuration, checkout, pretrained-file, and source-file validation.
"""

from __future__ import annotations

import copy
import importlib
import importlib.util
import sys
import types
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast

import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.nn import functional as F

from token_mixer.data.labels import REGION_NAMES


CANONICAL_REGIONS: tuple[str, str, str] = REGION_NAMES
EXTERNAL_CLASS_MAPPING: dict[int, str] = {
    0: "background",
    1: "ET",
    2: "TC",
    3: "WT",
}
EXTERNAL_NUM_CLASSES = 4
CANONICAL_NUM_CLASSES = len(CANONICAL_REGIONS)

_MISSING = object()
_DEFAULTS: dict[str, Any] = {
    "in_channels": 4,
    "image_size": (224, 224),
    "patch_size": 16,
    "vit_name": "R50-ViT-B_16",
    "n_skip": 3,
    "num_classes": CANONICAL_NUM_CLASSES,
    "canonical_num_classes": CANONICAL_NUM_CLASSES,
    "external_num_classes": EXTERNAL_NUM_CLASSES,
    "output_kind": "logits",
}
_REQUIRED_EXTERNAL_SOURCES = (
    Path("networks") / "vit_seg_modeling.py",
    Path("networks") / "vit_seg_modeling_resnet_skip.py",
    Path("networks") / "vit_seg_configs.py",
)
_EXTERNAL_PACKAGE_PREFIX = "_token_mixer_transunet_external_"

OutputKind: TypeAlias = Literal["logits", "softmax", "probabilities"]


def _path_value(config: Any, path: Sequence[str], default: Any = _MISSING) -> Any:
    current = config
    for key in path:
        if isinstance(current, Mapping):
            if key not in current:
                return default
            current = current[key]
        else:
            try:
                current = getattr(current, key)
            except AttributeError:
                return default
    return current


def _first_value(config: Any, paths: Sequence[Sequence[str]], default: Any = None) -> Any:
    for path in paths:
        value = _path_value(config, path)
        if value is not _MISSING and value is not None:
            return value
    return default


def _as_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None or value is _MISSING:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} configuration must be a mapping")
    return value


def _as_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if result <= 0 or (isinstance(value, float) and result != value):
        raise ValueError(f"{name} must be a positive integer")
    return result


def _as_nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if result < 0 or (isinstance(value, float) and result != value):
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _as_size2(value: Any, name: str) -> tuple[int, int]:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be one or two positive integers")
    if isinstance(value, (str, bytes, bytearray)):
        try:
            result = (_as_positive_int(value, name),) * 2
        except ValueError as exc:
            raise ValueError(f"{name} must be one or two positive integers") from exc
    elif isinstance(value, (int, np.integer)):
        result = (int(value), int(value))
    else:
        try:
            values = tuple(value)
        except TypeError as exc:
            raise ValueError(f"{name} must be one or two positive integers") from exc
        if len(values) == 1:
            result = (_as_positive_int(values[0], name),) * 2
        elif len(values) == 2:
            result = (
                _as_positive_int(values[0], name),
                _as_positive_int(values[1], name),
            )
        else:
            raise ValueError(f"{name} must be one or two positive integers")
    if any(size <= 0 for size in result):
        raise ValueError(f"{name} must be one or two positive integers")
    return cast(tuple[int, int], result)


def _as_path(value: Any, name: str) -> Path:
    if (
        value is None
        or value is _MISSING
        or isinstance(value, bytes)
        or isinstance(value, str)
        and not value.strip()
    ):
        raise ValueError(f"TransUNet configuration requires {name}")
    try:
        path = Path(value).expanduser()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"TransUNet configuration requires {name} as a path") from exc
    if not str(path):
        raise ValueError(f"TransUNet configuration requires {name}")
    return path


def _resolve_pretrained_value(cfg: Mapping[str, Any]) -> Any:
    return _first_value(
        cfg,
        (
            ("third_party", "pretrained_path"),
            ("third_party", "transunet_pretrained"),
            ("third_party", "transunet_pretrained_path"),
            ("model", "pretrained_path"),
            ("pretrained_path",),
        ),
        default=_MISSING,
    )


def _model_values(
    cfg: Mapping[str, Any],
    *,
    validate_official_geometry: bool = True,
) -> dict[str, Any]:
    model = _as_mapping(_path_value(cfg, ("model",), default={}), "model")

    def value(name: str, *fallback_paths: tuple[str, ...]) -> Any:
        return _first_value(
            cfg,
            (("model", name), *fallback_paths),
            default=_DEFAULTS.get(name, _MISSING),
        )

    image_size = _as_size2(
        value("image_size", ("model", "img_size"), ("image_size",), ("img_size",)),
        "image_size",
    )
    patch_size = _as_size2(
        value("patch_size", ("model", "vit_patches_size"), ("patch_size",)),
        "patch_size",
    )
    if validate_official_geometry:
        if patch_size[0] != patch_size[1]:
            raise ValueError("TransUNet patch_size must be square")
        if patch_size[0] != 16:
            raise ValueError(
                "TransUNet R50-ViT-B_16 adapter requires patch_size=16"
            )
        if image_size[0] != image_size[1]:
            raise ValueError(
                "TransUNet image_size must be square because official decoder reshapes a square token grid"
            )
        if image_size[0] % patch_size[0] != 0:
            raise ValueError("TransUNet image_size must be divisible by patch_size")

    in_channels = _as_positive_int(
        value("in_channels", ("in_channels",)),
        "in_channels",
    )
    if in_channels not in (3, 4):
        raise ValueError(
            "TransUNet adapter supports three RGB or four MRI input channels; "
            f"got {in_channels}"
        )

    configured_num_classes = _as_positive_int(
        value("num_classes", ("n_classes",), ("model", "n_classes")),
        "num_classes",
    )
    if configured_num_classes not in (CANONICAL_NUM_CLASSES, EXTERNAL_NUM_CLASSES):
        raise ValueError(
            "TransUNet num_classes must be canonical 3 or external 4; "
            f"got {configured_num_classes}"
        )

    canonical_num_classes = _as_positive_int(
        value("canonical_num_classes", ("canonical_num_classes",)),
        "canonical_num_classes",
    )
    if canonical_num_classes != CANONICAL_NUM_CLASSES:
        raise ValueError(
            f"TransUNet canonical_num_classes must be {CANONICAL_NUM_CLASSES}"
        )

    external_num_classes = _as_positive_int(
        value("external_num_classes", ("external_num_classes",)),
        "external_num_classes",
    )
    if external_num_classes != EXTERNAL_NUM_CLASSES:
        raise ValueError(
            f"TransUNet external_num_classes must be {EXTERNAL_NUM_CLASSES} "
            "for the supported official four-class API"
        )

    n_skip = _as_nonnegative_int(value("n_skip", ("n_skip",)), "n_skip")
    if n_skip > 4:
        raise ValueError("TransUNet n_skip must be between 0 and 4")

    vit_name = value("vit_name", ("model", "variant"), ("vit_name",), ("variant",))
    if not isinstance(vit_name, str) or not vit_name:
        raise ValueError("TransUNet vit_name must be a non-empty string")
    if vit_name != "R50-ViT-B_16":
        raise ValueError(
            "TransUNet adapter only supports official R50-ViT-B_16; "
            f"got {vit_name!r}"
        )

    output_kind = value(
        "output_kind",
        ("model", "external_output_kind"),
        ("external_output_kind",),
    )
    if output_kind == "probability":
        output_kind = "probabilities"
    if output_kind not in ("logits", "softmax", "probabilities"):
        raise ValueError(
            "TransUNet output_kind must be 'logits', 'softmax', or 'probabilities'"
        )

    # Keep model section material available for future pipeline provenance while
    # ensuring only validated, supported architecture values drive construction.
    result = dict(model)
    result.update(
        {
            "in_channels": in_channels,
            "image_size": image_size,
            "patch_size": patch_size[0],
            "vit_name": vit_name,
            "n_skip": n_skip,
            "num_classes": configured_num_classes,
            "canonical_num_classes": canonical_num_classes,
            "external_num_classes": external_num_classes,
            "output_kind": output_kind,
        }
    )
    return result


def _config_values(
    cfg: DictConfig | Mapping[str, Any],
    *,
    require_external_paths: bool = True,
    validate_official_geometry: bool = True,
) -> dict[str, Any]:
    if not isinstance(cfg, Mapping):
        raise TypeError("TransUNet configuration must be a mapping")

    third_party = _as_mapping(
        _path_value(cfg, ("third_party",), default=_MISSING),
        "third_party",
    )
    root_value = _path_value(third_party, ("transunet_root",), default=_MISSING)
    if require_external_paths and (root_value is _MISSING or root_value is None):
        raise ValueError(
            "TransUNet configuration requires third_party.transunet_root"
        )
    root = (
        _as_path(root_value, "third_party.transunet_root")
        if root_value is not _MISSING and root_value is not None
        else None
    )

    pretrained_value = _resolve_pretrained_value(cfg)
    if require_external_paths and (pretrained_value is _MISSING or pretrained_value is None):
        raise ValueError(
            "TransUNet configuration requires a configured pretrained file via "
            "third_party.pretrained_path or model.pretrained_path"
        )
    pretrained_path = (
        _as_path(pretrained_value, "the TransUNet pretrained file")
        if pretrained_value is not _MISSING and pretrained_value is not None
        else None
    )

    values = _model_values(
        cfg,
        validate_official_geometry=validate_official_geometry,
    )
    values.update(
        {
            "transunet_root": root,
            "pretrained_path": pretrained_path,
        }
    )
    return values


def _validate_external_paths(values: Mapping[str, Any]) -> None:
    root = values["transunet_root"]
    pretrained_path = values["pretrained_path"]
    if not root.is_dir():
        raise FileNotFoundError(
            "TransUNet external checkout configured by third_party.transunet_root "
            f"does not exist or is not a directory: {root}"
        )
    if not pretrained_path.is_file():
        raise FileNotFoundError(
            "TransUNet pretrained file configured in third_party.pretrained_path "
            f"does not exist or is not a file: {pretrained_path}"
        )

    missing = [root / relative for relative in _REQUIRED_EXTERNAL_SOURCES if not (root / relative).is_file()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "TransUNet checkout is missing required external source file(s): "
            f"{missing_text}"
        )


def validate_transunet_config(cfg: DictConfig) -> dict[str, Any]:
    """Validate TransUNet configuration and required external files.

    This function performs no imports from the external checkout. It is useful
    for failing before a training pipeline attempts model construction.
    """
    values = _config_values(cfg)
    _validate_external_paths(values)
    return dict(values)


def get_transunet_metadata() -> dict[str, Any]:
    """Return JSON-friendly metadata for the non-native 3-D model boundary."""
    return {
        "architecture": "TransUNet",
        "dimensionality": "2-D",
        "spatial_dims": 2,
        "slice_based": True,
        "native_3d": False,
        "input_layout": "[B, C, H, W]",
        "output_layout": "[B, ET/TC/WT, H, W]",
        "canonical_regions": CANONICAL_REGIONS,
        "external_num_classes": EXTERNAL_NUM_CLASSES,
        "external_class_mapping": dict(EXTERNAL_CLASS_MAPPING),
    }


transunet_metadata = get_transunet_metadata


def _validate_output_kind(output_kind: str) -> OutputKind:
    if output_kind == "probability":
        output_kind = "probabilities"
    if output_kind not in ("logits", "softmax", "probabilities"):
        raise ValueError(
            "output_kind must be 'logits', 'softmax', or 'probabilities'"
        )
    return output_kind  # type: ignore[return-value]


def _class_event_logit(log_prob: Tensor, event: tuple[int, ...]) -> Tensor:
    event_indices = torch.tensor(event, device=log_prob.device)
    complement = tuple(index for index in range(EXTERNAL_NUM_CLASSES) if index not in event)
    complement_indices = torch.tensor(complement, device=log_prob.device)
    return torch.logsumexp(log_prob.index_select(1, event_indices), dim=1, keepdim=True) - torch.logsumexp(
        log_prob.index_select(1, complement_indices), dim=1, keepdim=True
    )


def adapt_transunet_output(
    output: Tensor,
    *,
    output_kind: str = "logits",
) -> Tensor:
    """Convert official four-class output into canonical raw region logits.

    Four-class logits are converted to class log-probabilities, then each
    nested region event is represented as a binary log-odds logit:
    ``ET={1}``, ``TC={1,2}``, and ``WT={1,2,3}``. For a softmax tensor, the
    tensor must already contain normalized non-negative probabilities. Scalar
    class IDs and arbitrary tuples/dictionaries cannot recover raw logits and
    are rejected instead of being guessed at.
    """
    if not isinstance(output, Tensor):
        raise TypeError(
            "TransUNet output adapter expects a torch.Tensor with shape "
            "[B, 4, H, W]"
        )
    if output.ndim == 3:
        raise ValueError(
            "TransUNet output adapter cannot recover raw region logits from "
            "scalar class IDs; expected four class channels"
        )
    if output.ndim != 4 or output.shape[1] != EXTERNAL_NUM_CLASSES:
        raise ValueError(
            "TransUNet output adapter expects four class channels with shape "
            f"[B, 4, H, W], got {tuple(output.shape)}"
        )
    if not torch.isfinite(output).all().item():
        raise ValueError("TransUNet output must contain only finite values")

    kind = _validate_output_kind(output_kind)
    output_float = output.float()
    if kind == "logits":
        log_prob = F.log_softmax(output_float, dim=1)
    else:
        if (output_float < 0).any().item():
            raise ValueError("TransUNet softmax output must be non-negative")
        probabilities_sum = output_float.sum(dim=1, keepdim=True)
        if not torch.allclose(
            probabilities_sum,
            torch.ones_like(probabilities_sum),
            atol=1e-4,
            rtol=1e-4,
        ):
            raise ValueError("TransUNet softmax output must sum to one across classes")
        tiny = torch.finfo(output_float.dtype).tiny
        log_prob = torch.log(output_float.clamp_min(tiny))

    return torch.cat(
        (
            _class_event_logit(log_prob, (1,)),
            _class_event_logit(log_prob, (1, 2)),
            _class_event_logit(log_prob, (1, 2, 3)),
        ),
        dim=1,
    )


four_class_to_region_logits = adapt_transunet_output
transunet_output_to_region_logits = adapt_transunet_output


def resize_slice(
    array: np.ndarray | Tensor,
    size: int | Sequence[int],
    *,
    mode: Literal["bilinear", "nearest"] = "bilinear",
) -> np.ndarray | Tensor:
    """Resize 2-D or channel-first slices using PyTorch interpolation only.

    NumPy inputs return NumPy arrays with nearest-neighbor labels retaining
    their original dtype. Tensor inputs remain tensors on their original
    device. Accepted layouts are ``[H, W]``, ``[C, H, W]``, and
    ``[B, C, H, W]``.
    """
    if mode not in ("bilinear", "nearest"):
        raise ValueError("mode must be 'bilinear' or 'nearest'")
    target_size = _as_size2(size, "size")
    is_numpy = isinstance(array, np.ndarray)
    tensor = cast(Tensor, array if isinstance(array, Tensor) else torch.as_tensor(array))
    if tensor.ndim not in (2, 3, 4):
        raise ValueError(
            "slice must have shape [H, W], [C, H, W], or [B, C, H, W], "
            f"got {tuple(tensor.shape)}"
        )

    original_ndim = tensor.ndim
    original_dtype = tensor.dtype
    if original_ndim == 2:
        batched = tensor.unsqueeze(0).unsqueeze(0)
    elif original_ndim == 3:
        batched = tensor.unsqueeze(0)
    else:
        batched = tensor

    if not (torch.is_floating_point(batched) or torch.is_complex(batched)):
        batched = batched.float()
    if torch.is_complex(batched):
        raise TypeError("complex slices are not supported by 2-D interpolation")

    kwargs: dict[str, Any] = {"size": target_size, "mode": mode}
    if mode == "bilinear":
        kwargs["align_corners"] = False
    resized = F.interpolate(batched, **kwargs)
    if original_ndim == 2:
        resized = resized[0, 0]
    elif original_ndim == 3:
        resized = resized[0]

    if is_numpy:
        if mode == "nearest" and not torch.is_floating_point(tensor):
            resized = resized.to(dtype=original_dtype)
        return resized.detach().cpu().numpy()
    if torch.is_floating_point(tensor):
        return resized.to(dtype=original_dtype)
    if mode == "nearest":
        return resized.to(dtype=original_dtype)
    return resized


class TransUNetSliceAdapter(nn.Module):
    """Wrap an external four-class TransUNet as an explicit 2-D slice model."""

    def __init__(
        self,
        external_model: nn.Module,
        *,
        image_size: int | Sequence[int],
        in_channels: int = 4,
        output_kind: str = "logits",
        encoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(external_model, nn.Module):
            raise TypeError("external_model must be an nn.Module")
        self.external_model = external_model
        if encoder is None:
            encoder = next(
                (
                    candidate
                    for name in ("encoder", "transformer")
                    if isinstance(candidate := getattr(external_model, name, None), nn.Module)
                ),
                external_model,
            )
        if not isinstance(encoder, nn.Module):
            raise TypeError("encoder must be an nn.Module")
        self.encoder = encoder
        self.image_size = _as_size2(image_size, "image_size")
        self.in_channels = _as_positive_int(in_channels, "in_channels")
        if self.in_channels != 4:
            raise ValueError("TransUNetSliceAdapter requires four MRI input channels")
        self.num_classes = CANONICAL_NUM_CLASSES
        self.external_num_classes = EXTERNAL_NUM_CLASSES
        self.output_kind = _validate_output_kind(output_kind)
        self.metadata = get_transunet_metadata()
        self.metadata.update(
            {
                "in_channels": self.in_channels,
                "image_size": self.image_size,
                "output_kind": self.output_kind,
            }
        )

    def forward(self, image: Tensor) -> Tensor:
        if not isinstance(image, Tensor) or image.ndim != 4:
            shape = getattr(image, "shape", None)
            raise ValueError(f"TransUNet adapter expects [B, C, H, W], got {shape}")
        if image.shape[1] != self.in_channels:
            raise ValueError(
                f"TransUNet adapter expects {self.in_channels} input channels, "
                f"got {image.shape[1]}"
            )
        if any(size <= 0 for size in image.shape[-2:]):
            raise ValueError(
                f"TransUNet adapter requires positive spatial dimensions, got {tuple(image.shape)}"
            )

        original_size = tuple(image.shape[-2:])
        model_input = image
        if original_size != self.image_size:
            model_input = F.interpolate(
                image,
                size=self.image_size,
                mode="bilinear",
                align_corners=False,
            )
        external_output = self.external_model(model_input)
        region_logits = adapt_transunet_output(
            external_output,
            output_kind=self.output_kind,
        )
        if tuple(region_logits.shape[-2:]) != original_size:
            region_logits = F.interpolate(
                region_logits,
                size=original_size,
                mode="bilinear",
                align_corners=False,
            )
        expected_shape = (image.shape[0], CANONICAL_NUM_CLASSES, *original_size)
        if tuple(region_logits.shape) != expected_shape:
            raise RuntimeError(
                f"TransUNet adapter returned {tuple(region_logits.shape)}, "
                f"expected {expected_shape}"
            )
        return region_logits


def _import_external_api(root: Path) -> tuple[Any, Any, Mapping[str, Any]]:
    networks_root = root / "networks"
    package_name = f"{_EXTERNAL_PACKAGE_PREFIX}{uuid.uuid4().hex}"

    def load_module(name: str, path: Path) -> types.ModuleType:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot create import spec for external source: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    package_init = networks_root / "__init__.py"
    package_spec = importlib.util.spec_from_file_location(
        package_name,
        package_init if package_init.is_file() else None,
        submodule_search_locations=[str(networks_root)],
    )
    if package_spec is None:
        package = types.ModuleType(package_name)
        package.__path__ = [str(networks_root)]
        package.__package__ = package_name
    else:
        package = importlib.util.module_from_spec(package_spec)
        package.__path__ = [str(networks_root)]
    sys.modules[package_name] = package

    try:
        importlib.invalidate_caches()
        if package_spec is not None and package_spec.loader is not None:
            package_spec.loader.exec_module(package)
        try:
            load_module(
                f"{package_name}.vit_seg_configs",
                networks_root / "vit_seg_configs.py",
            )
            resnet_skip = load_module(
                f"{package_name}.vit_seg_modeling_resnet_skip",
                networks_root / "vit_seg_modeling_resnet_skip.py",
            )
            modeling = load_module(
                f"{package_name}.vit_seg_modeling",
                networks_root / "vit_seg_modeling.py",
            )
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError(
                "TransUNet checkout must provide importable official modules under "
                "networks/; isolated loading does not mutate sys.path"
            ) from exc
    finally:
        for name in tuple(sys.modules):
            if name == package_name or name.startswith(f"{package_name}."):
                sys.modules.pop(name, None)

    vision_transformer = getattr(modeling, "VisionTransformer", None)
    std_conv2d = getattr(resnet_skip, "StdConv2d", None)
    configs = getattr(modeling, "CONFIGS", None)
    if not callable(vision_transformer) or not callable(std_conv2d) or not isinstance(configs, Mapping):
        raise ImportError(
            "TransUNet checkout does not expose supported official API: "
            "VisionTransformer, CONFIGS, and StdConv2d"
        )
    return vision_transformer, std_conv2d, configs


def _load_pretrained(model: nn.Module, path: Path) -> None:
    load_from = getattr(model, "load_from", None)
    if not callable(load_from):
        raise ImportError(
            "TransUNet external model must expose load_from(weights) for the "
            "configured pretrained .npz file"
        )
    try:
        weights = np.load(str(path), allow_pickle=False)
        try:
            load_from(weights)
        finally:
            close = getattr(weights, "close", None)
            if callable(close):
                close()
    except Exception as exc:
        raise RuntimeError(
            "TransUNet pretrained file is incompatible with the official "
            f"load_from(weights) API: {path}"
        ) from exc


def load_transunet_state_dict(
    model: nn.Module,
    state: Mapping[str, Tensor],
) -> None:
    """Load pipeline-provided weights into an injected external network."""
    if not isinstance(model, nn.Module):
        raise TypeError("TransUNet network must be an nn.Module")
    if not isinstance(state, Mapping):
        raise TypeError("TransUNet state must be a mapping of parameter names to tensors")
    try:
        model.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError("injected TransUNet state dictionary is incompatible") from exc


def _adapt_input_stem(model: nn.Module, std_conv2d: Any, in_channels: int) -> None:
    try:
        root = model.transformer.embeddings.hybrid_model.root
        old_conv = root.conv
    except AttributeError as exc:
        raise ImportError(
            "Supported TransUNet API must expose "
            "transformer.embeddings.hybrid_model.root.conv for the R50 stem"
        ) from exc
    if not isinstance(old_conv, nn.Conv2d) or old_conv.in_channels != 3:
        raise ImportError(
            "Supported TransUNet R50 stem must be a three-channel nn.Conv2d"
        )
    if in_channels == 3:
        return
    if in_channels != 4:
        raise ValueError("R50 TransUNet stem adaptation supports only four MRI channels")

    new_conv = std_conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=old_conv.groups,
        bias=old_conv.bias is not None,
    ).to(device=old_conv.weight.device, dtype=old_conv.weight.dtype)
    with torch.no_grad():
        new_conv.weight[:, :3].copy_(old_conv.weight)
        new_conv.weight[:, 3:4].copy_(old_conv.weight.mean(dim=1, keepdim=True))
        if old_conv.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)
    root.conv = new_conv


def build_transunet(
    cfg: DictConfig,
    *,
    external_model: nn.Module | None = None,
    external_network: nn.Module | None = None,
    external_state: Mapping[str, Tensor] | None = None,
) -> nn.Module:
    """Build or wrap TransUNet while keeping external loading explicit.

    With no injected network, ``cfg`` must identify one validated official
    checkout and pretrained file. Pipelines that own construction and weight
    loading can inject an already-created network and optional state dictionary;
    that path performs no filesystem or external-module loading.
    """
    if external_model is not None and external_network is not None:
        raise ValueError("provide only one of external_model or external_network")
    injected_model = external_model if external_model is not None else external_network
    if injected_model is not None and not isinstance(injected_model, nn.Module):
        raise TypeError("injected TransUNet network must be an nn.Module")
    if external_state is not None and injected_model is None:
        raise ValueError(
            "external_state requires an injected external_model or external_network"
        )

    values = _config_values(
        cfg,
        require_external_paths=injected_model is None,
        validate_official_geometry=injected_model is None,
    )
    if injected_model is None:
        _validate_external_paths(values)

        vision_transformer, std_conv2d, configs = _import_external_api(
            values["transunet_root"]
        )
        try:
            external_config = copy.deepcopy(configs[values["vit_name"]])
        except KeyError as exc:
            raise ImportError(
                "TransUNet external CONFIGS must contain R50-ViT-B_16"
            ) from exc

        external_config.n_classes = EXTERNAL_NUM_CLASSES
        external_config.n_skip = values["n_skip"]
        external_config.pretrained_path = str(values["pretrained_path"])
        if hasattr(external_config, "patches"):
            external_config.patches.grid = tuple(
                size // values["patch_size"] for size in values["image_size"]
            )

        try:
            injected_model = vision_transformer(
                external_config,
                img_size=values["image_size"],
                num_classes=EXTERNAL_NUM_CLASSES,
            )
        except Exception as exc:
            raise RuntimeError(
                "TransUNet external VisionTransformer(config, img_size, num_classes) "
                "construction failed"
            ) from exc

        _load_pretrained(injected_model, values["pretrained_path"])
        _adapt_input_stem(injected_model, std_conv2d, values["in_channels"])
    elif external_state is not None:
        load_transunet_state_dict(injected_model, external_state)

    return TransUNetSliceAdapter(
        injected_model,
        image_size=values["image_size"],
        in_channels=values["in_channels"],
        output_kind=values["output_kind"],
    )


__all__ = [
    "CANONICAL_NUM_CLASSES",
    "CANONICAL_REGIONS",
    "EXTERNAL_CLASS_MAPPING",
    "EXTERNAL_NUM_CLASSES",
    "TransUNetSliceAdapter",
    "adapt_transunet_output",
    "build_transunet",
    "four_class_to_region_logits",
    "get_transunet_metadata",
    "load_transunet_state_dict",
    "resize_slice",
    "transunet_metadata",
    "transunet_output_to_region_logits",
    "validate_transunet_config",
]
