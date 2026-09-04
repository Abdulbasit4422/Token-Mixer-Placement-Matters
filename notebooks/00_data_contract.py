# %% [markdown]
# # Data Contract
#
# Exercise the package-wide segmentation and case contracts without requiring
# a downloaded dataset. Raw BraTS labels are converted to canonical
# `[ET, TC, WT]` masks, while `CaseRecord` keeps modality paths explicit.

# %%
from pathlib import Path
import os
import sys

import numpy as np


configured_root = os.environ.get("TOKEN_MIXER_REPO_ROOT")
REPO_ROOT = Path(configured_root).expanduser() if configured_root else Path.cwd()
if not (REPO_ROOT / "src" / "token_mixer").is_dir():
    REPO_ROOT = Path.cwd()
SOURCE_ROOT = REPO_ROOT / "src"
if SOURCE_ROOT.is_dir() and str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from token_mixer.data.cases import CaseRecord, MODALITY_NAMES, discover_cases
from token_mixer.data.labels import (
    REGION_NAMES,
    detect_et_label,
    multiclass_to_regions,
    regions_to_multiclass,
    to_region_masks,
)

# %% [markdown]
# ## Canonical labels
#
# Both common enhancing-tumor conventions are accepted. The package exposes
# one region order and one canonical multiclass representation at all model
# boundaries.

# %%
assert REGION_NAMES == ("ET", "TC", "WT")

raw_labels = {
    "BraTS label 4": np.array([[[0, 1, 2, 4]]], dtype=np.uint8),
    "BraTS label 3": np.array([[[0, 1, 2, 3]]], dtype=np.uint8),
}

for description, raw_label in raw_labels.items():
    et_label = detect_et_label(raw_label)
    region_masks = to_region_masks(raw_label)
    canonical_label = regions_to_multiclass(region_masks)
    restored_masks = multiclass_to_regions(canonical_label)

    np.testing.assert_array_equal(restored_masks, region_masks)
    assert canonical_label[0, 0].tolist() == [0, 2, 3, 1]
    print(
        f"{description}: detected ET={et_label}; "
        f"raw={raw_label.shape}, regions={region_masks.shape}, "
        f"canonical={canonical_label.shape}"
    )

# %% [markdown]
# ## Case record
#
# `CaseRecord` stores canonical modality names and paths. Discovery is metadata
# only here, so this cell remains safe when local data is absent.

# %%
case_root = Path("data") / "local" / "brats" / "cases" / "CONTRACT_CASE"
case = CaseRecord(
    case_id="CONTRACT_CASE",
    modalities={
        modality: case_root / f"{modality}.nii.gz" for modality in MODALITY_NAMES
    },
    segmentation=case_root / "segmentation.nii.gz",
)

assert case.case_id == "CONTRACT_CASE"
assert tuple(case.modalities) == MODALITY_NAMES
assert case.segmentation.name == "segmentation.nii.gz"
print("CaseRecord:", case)
print("Canonical modalities:", tuple(case.modalities))
print(
    "Complete local cases discovered:",
    len(discover_cases(REPO_ROOT / "data" / "local" / "brats")),
)
