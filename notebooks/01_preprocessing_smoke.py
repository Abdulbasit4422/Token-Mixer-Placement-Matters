# %% [markdown]
# # Preprocessing Smoke Test
#
# Load one complete case from the configured local data root and exercise the
# package preprocessing path. This notebook never downloads data or trains a
# model. When no fixture is present, it reports the expected layout and exits
# normally.

# %%
from pathlib import Path
import importlib.util
import os
import sys

import torch


configured_root = os.environ.get("TOKEN_MIXER_REPO_ROOT")
REPO_ROOT = Path(configured_root).expanduser() if configured_root else Path.cwd()
if not (REPO_ROOT / "src" / "token_mixer").is_dir():
    REPO_ROOT = Path.cwd()
SOURCE_ROOT = REPO_ROOT / "src"
if SOURCE_ROOT.is_dir() and str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from token_mixer.data.cases import discover_cases
from token_mixer.data.datasets import BratsPatchDataset
from token_mixer.data.labels import REGION_NAMES
from token_mixer.reproducibility import seed_everything

# TOKEN_MIXER_DATA_ROOT may be absolute or relative to the notebook's working
# directory. The default follows the repository's local-data convention.
configured_data_root = os.environ.get("TOKEN_MIXER_DATA_ROOT")
DATA_ROOT = (
    Path(configured_data_root).expanduser()
    if configured_data_root
    else REPO_ROOT / "data" / "local" / "brats"
)
if not DATA_ROOT.is_absolute():
    DATA_ROOT = (Path.cwd() / DATA_ROOT).resolve()

_ = seed_everything(42)
cases = discover_cases(DATA_ROOT)

# %% [markdown]
# ## Discover and preprocess

# %%
if not cases:
    print(f"No complete local fixture found under: {DATA_ROOT}")
    print(
        "Expected <data-root>/cases/<case-id>/ with t1n.nii.gz, t1c.nii.gz, "
        "t2w.nii.gz, t2f.nii.gz, and segmentation.nii.gz."
    )
    print(
        "Set TOKEN_MIXER_DATA_ROOT to a local fixture root. "
        "No download or training was performed."
    )
else:
    smoke_config = {
        "patch_size": (32, 32, 32),
        "normalize": True,
        "flip_axes": False,
        "seed": 42,
    }
    dataset = BratsPatchDataset(cases[:1], smoke_config, training=False)
    try:
        image, masks = dataset[0]
    except RuntimeError as exc:
        message = str(exc)
        if importlib.util.find_spec("nibabel") is None:
            print(
                "Local fixture discovered, but optional nibabel dependency is "
                "unavailable. Install the imaging extra to load NIfTI files."
            )
        else:
            print(f"Local fixture could not be loaded: {message}")
        print("No download or training was performed.")
    else:
        image_tensor = torch.as_tensor(image)
        mask_tensor = torch.as_tensor(masks)
        assert image_tensor.shape[0] == 4
        assert mask_tensor.shape[0] == len(REGION_NAMES)
        print(f"Case: {cases[0].case_id}")
        print(f"Image tensor shape: {tuple(image_tensor.shape)}")
        print(f"Target tensor shape: {tuple(mask_tensor.shape)}")
        print(f"Target region order: {REGION_NAMES}")
        print(f"Image dtype: {image_tensor.dtype}; target dtype: {mask_tensor.dtype}")
