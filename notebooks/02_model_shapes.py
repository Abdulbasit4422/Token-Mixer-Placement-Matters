# %% [markdown]
# # Model Shape Smoke Test
#
# Run tiny CPU forwards against every packaged model track. The debug run file
# supplies reproducibility settings; explicit tiny model configs keep this
# notebook fast and prevent accidental full training. MONAI and the external
# TransUNet checkout are optional and are reported as skips when unavailable.

# %%
from pathlib import Path
import importlib.util
import os
import sys

import pandas as pd
import torch
from omegaconf import OmegaConf


configured_root = os.environ.get("TOKEN_MIXER_REPO_ROOT")
REPO_ROOT = Path(configured_root).expanduser() if configured_root else Path.cwd()
if not (REPO_ROOT / "src" / "token_mixer").is_dir():
    REPO_ROOT = Path.cwd()
SOURCE_ROOT = REPO_ROOT / "src"
if SOURCE_ROOT.is_dir() and str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from token_mixer.models.cnn_pretrain import build_denoising_model
from token_mixer.models.metaunetr.variants import build_metaunetr
from token_mixer.models.resunet3d import build_resunet3d
from token_mixer.models.swinunetr import build_swinunetr
from token_mixer.models.transunet import build_transunet
from token_mixer.reproducibility import seed_everything

DEBUG_CONFIG_PATH = REPO_ROOT / "configs" / "run" / "debug.yaml"
DEBUG_RUN = OmegaConf.load(str(DEBUG_CONFIG_PATH))
DEBUG_SEED = int(DEBUG_RUN.seed)
DEBUG_VALUES = OmegaConf.to_container(DEBUG_RUN, resolve=True)


def with_debug_config(model_config: dict, **extra: object):
    return OmegaConf.create(
        {
            "device": "cpu",
            "run": DEBUG_VALUES,
            "model": model_config,
            **extra,
        }
    )


def run_shape_check(
    name: str,
    builder,
    config,
    input_shape: tuple[int, ...],
    expected_shape: tuple[int, ...],
) -> dict[str, object]:
    seed_everything(DEBUG_SEED)
    model = builder(config).to("cpu").eval()
    inputs = torch.randn(input_shape)
    with torch.no_grad():
        outputs = model(inputs)
    actual_shape = tuple(int(size) for size in outputs.shape)
    if actual_shape != expected_shape:
        raise RuntimeError(
            f"{name} returned {actual_shape}; expected {expected_shape}"
        )
    if not torch.isfinite(outputs).all().item():
        raise RuntimeError(f"{name} returned non-finite outputs")
    result = {
        "model": name,
        "status": "PASS",
        "input_shape": input_shape,
        "output_shape": actual_shape,
        "notes": "tiny CPU forward",
    }
    print(f"PASS {name}: {input_shape} -> {actual_shape}")
    return result


def skipped_shape_check(name: str, reason: str) -> dict[str, object]:
    print(f"SKIP {name}: {reason}")
    return {
        "model": name,
        "status": "SKIP",
        "input_shape": None,
        "output_shape": None,
        "notes": reason,
    }


def optional_dependency_skip_reason(*dependencies: str) -> str | None:
    missing = [
        dependency
        for dependency in dependencies
        if importlib.util.find_spec(dependency) is None
    ]
    if not missing:
        return None
    return f"optional dependencies unavailable: {', '.join(missing)}; install the imaging extra"


results: list[dict[str, object]] = []

# The three paper variants share the same tiny CPU contract and differ only in
# the configured mixer placement.
meta_config = {
    "in_channels": 4,
    "num_classes": 3,
    "base_channels": 4,
    "depths": [1, 1, 1, 1],
    "window_size": 2,
    "num_heads": 2,
    "d_state": 2,
    "d_conv": 2,
    "mamba_expand": 1,
    "axis_fusion": "sum",
    "norm_name": "group",
    "norm_num_groups": 1,
}
# MetaUNETR uses MONAI blocks, but its CPU path does not require einops.
metaunetr_skip_reason = optional_dependency_skip_reason("monai")
for variant in ("metaunetr_mamba", "mod_a", "mod_b"):
    if metaunetr_skip_reason:
        results.append(skipped_shape_check(variant, metaunetr_skip_reason))
        continue
    try:
        results.append(
            run_shape_check(
                variant,
                lambda config, selected_variant=variant: build_metaunetr(
                    config, selected_variant
                ),
                OmegaConf.create(meta_config),
                (1, 4, 32, 32, 32),
                (1, 3, 32, 32, 32),
            )
        )
    except ImportError as exc:
        results.append(
            skipped_shape_check(
                variant,
                f"optional MONAI dependency unavailable during construction/forward: {exc}",
            )
        )

# %% [markdown]
# ## Baselines

# %%
results.append(
    run_shape_check(
        "resunet3d",
        build_resunet3d,
        with_debug_config(
            {
                "in_channels": 4,
                "out_channels": 3,
                "base_features": 2,
                "depths": [1, 1, 1, 1, 1],
                "normalization": "group",
                "norm_num_groups": 1,
            }
        ),
        (1, 4, 16, 16, 16),
        (1, 3, 16, 16, 16),
    )
)

results.append(
    run_shape_check(
        "cnn_denoising_pretrain",
        build_denoising_model,
        with_debug_config(
            {
                "in_channels": 3,
                "feature_size": 2,
                "depths": [1, 1, 1, 1],
                "image_size": 32,
                "norm_num_groups": 1,
            }
        ),
        (1, 3, 32, 32),
        (1, 3, 32, 32),
    )
)

# SwinUNETR requires both MONAI and its einops dependency.
swinunetr_skip_reason = optional_dependency_skip_reason("monai", "einops")
if swinunetr_skip_reason:
    results.append(
        skipped_shape_check("swinunetr", swinunetr_skip_reason)
    )
else:
    try:
        results.append(
            run_shape_check(
                "swinunetr",
                build_swinunetr,
                with_debug_config(
                    {
                        "in_channels": 4,
                        "out_channels": 3,
                        "feature_size": 12,
                        "spatial_dims": 3,
                        "patch_size": 2,
                        "depths": [1, 1, 1, 1],
                        "num_heads": [3, 6, 12, 24],
                        "window_size": 2,
                        "img_size": [32, 32, 32],
                        "use_checkpoint": False,
                    }
                ),
                (1, 4, 32, 32, 32),
                (1, 3, 32, 32, 32),
            )
        )
    except ImportError as exc:
        results.append(
            skipped_shape_check(
                "swinunetr",
                f"optional MONAI dependency unavailable during construction/forward: {exc}",
            )
        )

# TransUNet requires both an external checkout and pretrained weights. Paths
# are supplied through environment variables, never embedded in this notebook.
transunet_root_value = os.environ.get("TRANSUNET_ROOT")
transunet_pretrained_value = os.environ.get("TRANSUNET_PRETRAINED")
if not transunet_root_value or not transunet_pretrained_value:
    results.append(
        skipped_shape_check(
            "transunet",
            "set TRANSUNET_ROOT and TRANSUNET_PRETRAINED for external integration",
        )
    )
else:
    transunet_root = Path(transunet_root_value).expanduser()
    transunet_pretrained = Path(transunet_pretrained_value).expanduser()
    if not transunet_root.is_dir() or not transunet_pretrained.is_file():
        results.append(
            skipped_shape_check(
                "transunet",
                "configured external checkout or pretrained file is unavailable",
            )
        )
    else:
        try:
            results.append(
                run_shape_check(
                    "transunet",
                    build_transunet,
                    with_debug_config(
                        {
                            "in_channels": 4,
                            "num_classes": 4,
                            "canonical_num_classes": 3,
                            "external_num_classes": 4,
                            "image_size": [32, 32],
                            "patch_size": 16,
                            "vit_name": "R50-ViT-B_16",
                            "n_skip": 3,
                            "output_kind": "logits",
                        },
                        third_party={
                            "transunet_root": str(transunet_root),
                            "pretrained_path": str(transunet_pretrained),
                        },
                    ),
                    (1, 4, 32, 32),
                    (1, 3, 32, 32),
                )
            )
        except (FileNotFoundError, ImportError) as exc:
            results.append(skipped_shape_check("transunet", str(exc)))

# %% [markdown]
# ## Shape table

# %%
shape_table = pd.DataFrame(
    results,
    columns=["model", "status", "input_shape", "output_shape", "notes"],
)
print(shape_table.to_string(index=False))
shape_table
