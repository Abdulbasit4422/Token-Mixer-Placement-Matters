"""
dataset.py  -  MONAI dataloader factory for BraTS-Africa NIfTI volumes.

Expected directory layout (standard BraTS / nnUNet raw format):
    <data_root>/
        imagesTr/
            BraTS-SSA-00001-000_0000.nii.gz   # T1n
            BraTS-SSA-00001-000_0001.nii.gz   # T1c
            BraTS-SSA-00001-000_0002.nii.gz   # T2w
            BraTS-SSA-00001-000_0003.nii.gz   # T2f (FLAIR)
            ...
        labelsTr/
            BraTS-SSA-00001-000.nii.gz        # segmentation mask
            ...

Label convention -- BraTS-Africa uses the BraTS 2023 scheme:
    0 -> background
    1 -> NCR  (necrotic core)
    2 -> ED   (peritumoral edema)
    3 -> ET   (enhancing tumour)      <-- NOT 4, unlike BraTS 2018-2022!

Output mask channels after ConvertToMultiChannelBasedOnBratsClassesd(et_label=3):
    channel 0 -> TC  (tumour core  = labels 1 + 3)
    channel 1 -> WT  (whole tumour = labels 1 + 2 + 3)
    channel 2 -> ET  (enhancing    = label  3)

This matches the SwinUNETR head order used in train_swinunetr_new_.py
(SUBREGIONS = ["TC", "WT", "ET"]).

==============================================================================
FIX (this version): MONAI's ConvertToMultiChannelBasedOnBratsClassesd defaults
to et_label=4 (the BraTS 2018-2022 convention). BraTS-Africa labels never
contain a 4 -- they use 3 for ET (BraTS 2023 convention, confirmed by the
"Label convention: ET=3 (BraTS 2023)" message printed during data loading).
With the old default, `img == 4` was always False, so:
  - the ET channel was silently all-zero for every case (ground truth AND
    prediction), explaining the ET Dice/HD95 = 0.00 and the missing cyan
    region in visualizations
  - the TC channel only counted NCR (label 1), silently missing the ET
    voxels that should also count toward tumour core
Fix: pass et_label=3 explicitly to both get_train_transforms() and
get_val_transforms() below. This is the ONLY change from the previous
version of this file.
==============================================================================
"""

import os
import glob
import json
from typing import Tuple

from monai.data import Dataset, DataLoader, CacheDataset, partition_dataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ConvertToMultiChannelBasedOnBratsClassesd,
    CropForegroundd,
    RandSpatialCropd,
    RandFlipd,
    NormalizeIntensityd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Spacingd,
    Orientationd,
    SpatialPadd,
    ToTensord,
)

# ── Patch size must match the ROI_SIZE in train_swinunetr_new_.py ────────────
ROI_SIZE = (96, 96, 96)
TARGET_SPACING = (1.0, 1.0, 1.0)  # isotropic 1 mm³ - standard BraTS space

# Suffix patterns for the four input modalities (nnUNet _000X convention)
MODALITY_SUFFIXES = ["_0000.nii.gz", "_0001.nii.gz", "_0002.nii.gz", "_0003.nii.gz"]

# FIX: BraTS-Africa uses the BraTS 2023 label convention (ET = label 3, not 4).
ET_LABEL = 3

from monai.transforms import MapTransform
import numpy as np


class ConvertToBraTSAfricaClassesd(MapTransform):
    """
    BraTS-Africa:
        0 = background
        1 = NCR
        2 = ED
        3 = ET

    Output channels:
        0 = TC = 1 OR 3
        1 = WT = 1 OR 2 OR 3
        2 = ET = 3
    """

    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):

        d = dict(data)

        for key in self.keys:

            label = d[key]

            tc = np.logical_or(
                label == 1,
                label == 3
            )

            wt = np.logical_or(
                tc,
                label == 2
            )

            et = label == 3

            d[key] = np.stack(
                [tc, wt, et],
                axis=0
            ).astype(np.float32)

        return d
# ── 1. DATA LIST BUILDER ──────────────────────────────────────────────────────
def build_data_list(data_root: str) -> list[dict]:
    """
    Scan imagesTr / labelsTr and pair each case's four modality files
    with its segmentation label.  Returns a list of dicts ready for
    MONAI's Dataset.
    """
    images_dir = os.path.join(data_root, "imagesTr")
    labels_dir = os.path.join(data_root, "labelsTr")

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(
            f"imagesTr not found under '{data_root}'.  "
            "Check that DATA_DST points to the raw NIfTI root, "
            "not the nnUNet_preprocessed folder (which stores .npy arrays)."
        )

    all_image_files = sorted(glob.glob(os.path.join(images_dir, "*_0000.nii.gz")))
    if not all_image_files:
        raise FileNotFoundError(
            f"No files matching '*_0000.nii.gz' found in '{images_dir}'."
        )

    data_list = []
    skipped = 0
    for t1n_path in all_image_files:
        base = t1n_path.replace("_0000.nii.gz", "")
        case_id = os.path.basename(base)

        image_paths = [base + suffix for suffix in MODALITY_SUFFIXES]

        label_path = os.path.join(labels_dir, case_id + ".nii.gz")
        if not os.path.exists(label_path):
            label_path = os.path.join(labels_dir, case_id + ".seg.nii.gz")

        missing = [p for p in image_paths + [label_path] if not os.path.exists(p)]
        if missing:
            print(f"  [WARN] Skipping {case_id} - missing files: {missing}")
            skipped += 1
            continue

        data_list.append({"image": image_paths, "label": label_path})

    print(
        f"[dataset] Found {len(data_list)} complete cases "
        f"({skipped} skipped) under '{data_root}'."
    )
    return data_list


# ── 2. MONAI TRANSFORMS ───────────────────────────────────────────────────────
def get_train_transforms() -> Compose:
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image"]),
            # FIX: et_label=3 (BraTS 2023 / BraTS-Africa convention).
            # Previously defaulted to et_label=4, which never matched this
            # dataset's labels and silently zeroed out the ET channel.
            ConvertToBraTSAfricaClassesd(
                 keys="label"
            ),
            Spacingd(
                keys=["image", "label"],
                pixdim=TARGET_SPACING,
                mode=("bilinear", "nearest"),
            ),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            SpatialPadd(
                keys=["image", "label"],
                spatial_size=ROI_SIZE,
            ),
            RandSpatialCropd(
                keys=["image", "label"],
                roi_size=ROI_SIZE,
                random_size=False,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
            ToTensord(keys=["image", "label"]),
        ]
    )


def get_val_transforms() -> Compose:
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image"]),
            # FIX: same et_label=3 correction as get_train_transforms above.
            ConvertToBraTSAfricaClassesd(
              keys="label"
             ),
            Spacingd(
                keys=["image", "label"],
                pixdim=TARGET_SPACING,
                mode=("bilinear", "nearest"),
            ),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ToTensord(keys=["image", "label"]),
        ]
    )

def get_test_transforms():
    """
    Test-time transforms.
    Uses exactly the same preprocessing as validation,
    but kept separate for final evaluation and visualization.
    """

    return Compose([
        LoadImaged(
            keys=["image", "label"]
        ),

        EnsureChannelFirstd(
            keys="image"
        ),

        ConvertToBraTSAfricaClassesd(
            keys="label"
        ),

        Orientationd(
            keys=["image", "label"],
            axcodes="RAS"
        ),

        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest")
        ),

        NormalizeIntensityd(
            keys="image",
            nonzero=True,
            channel_wise=True
        ),

        ToTensord(
            keys=["image", "label"]
        )
    ])
# ── 3. DATALOADER FACTORY ─────────────────────────────────────────────────────
def get_dataloaders(
    data_root: str,
    batch_size: int = 2,
    val_split: float = 0.2,
    num_workers: int = 8,
    cache_rate: float = 0.0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Build and return (train_loader, val_loader)."""
    data_list = build_data_list(data_root)

    partitions = partition_dataset(
        data=data_list,
        ratios=[1.0 - val_split, val_split],
        shuffle=True,
        seed=seed,
    )
    train_files, val_files = partitions[0], partitions[1]
    print(f"[dataset] Train: {len(train_files)} cases  |  Val: {len(val_files)} cases")

    DatasetClass = CacheDataset if cache_rate > 0 else Dataset

    train_ds_kwargs = dict(data=train_files, transform=get_train_transforms())
    val_ds_kwargs   = dict(data=val_files,   transform=get_val_transforms())
    if cache_rate > 0:
        train_ds_kwargs["cache_rate"] = cache_rate
        val_ds_kwargs["cache_rate"]   = cache_rate

    train_ds = DatasetClass(**train_ds_kwargs)
    val_ds   = DatasetClass(**val_ds_kwargs)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


# ── 4. OPTIONAL: dedicated held-out test split ────────────────────────────────
def get_test_loader(
    data_root: str,
    batch_size: int = 1,
    num_workers: int = 8,
    val_split: float = 0.2,
    test_split: float = 0.5,
    seed: int = 42,
) -> DataLoader:
    """
    Carves a dedicated test split out of the same val pool used by
    get_dataloaders(), using the SAME seed so train cases never leak into
    it. test_split=0.5 means: of the val_split fraction, half becomes the
    final val set (used for periodic in-training checks) and half becomes
    a held-out test set (used only for the final evaluate_test/visualize_test
    pass). Auto-detected by train_swinunetr_new_.py if present.
    """
    data_list = build_data_list(data_root)

    partitions = partition_dataset(
        data=data_list,
        ratios=[1.0 - val_split, val_split],
        shuffle=True,
        seed=seed,
    )
    val_pool = partitions[1]

    test_partitions = partition_dataset(
        data=val_pool,
        ratios=[1.0 - test_split, test_split],
        shuffle=True,
        seed=seed,
    )
    test_files = test_partitions[1]
    print(f"[dataset] Test: {len(test_files)} cases (held out from val pool)")

    test_ds = Dataset(data=test_files, transform=get_test_transforms())
    return DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )