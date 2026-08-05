import os
import json
import nibabel as nib
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil

USER = os.environ["USER"]
BRATS_SRC = Path(f"/scratch/{USER}/brats-mamba/data")
NNUNET_RAW = Path(f"/scratch/{USER}/nnUNet_raw")

DATASET_NAME = "Dataset100_BraTSAfrica"
OUT_DIR = NNUNET_RAW / DATASET_NAME
IMAGES_TR = OUT_DIR / "imagesTr"
LABELS_TR = OUT_DIR / "labelsTr"

def safe_copy_nifti(in_path, out_path):
    """Safely loads and saves the NIfTI file. Throws an error if the file is corrupted."""
    img = nib.load(str(in_path))
    nib.save(img, str(out_path))

def map_brats_labels(label_path, out_path):
    """nnUNet requires consecutive labels. BraTS uses 1, 2, 4. We map 4 -> 3."""
    img = nib.load(str(label_path))
    data = img.get_fdata().astype(np.uint8)
    data[data == 4] = 3
    new_img = nib.Nifti1Image(data, img.affine, img.header)
    nib.save(new_img, str(out_path))

def main():
    print(f"Creating nnUNet structure in: {OUT_DIR}")
    
    # Wipe the broken folder to start completely fresh
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
        
    IMAGES_TR.mkdir(parents=True, exist_ok=True)
    LABELS_TR.mkdir(parents=True, exist_ok=True)
    
    print(f"Scanning for NIfTI files recursively in: {BRATS_SRC}")
    all_niftis = list(BRATS_SRC.rglob('*.nii*'))
    
    if not all_niftis:
        print("CRITICAL ERROR: No NIfTI files found!")
        return
        
    subject_dirs = set(f.parent for f in all_niftis)
    print(f"Found {len(subject_dirs)} potential subject folders. Converting and validating compression...")
    
    valid_count = 0
    
    for subj in tqdm(list(subject_dirs)):
        subj_id = subj.name
        niftis = list(subj.glob('*.nii*'))
        
        t1 = next((f for f in niftis if 't1n' in f.name.lower() or ('t1' in f.name.lower() and 'ce' not in f.name.lower() and 'gd' not in f.name.lower() and 't1c' not in f.name.lower())), None)
        t1c = next((f for f in niftis if 't1c' in f.name.lower() or 't1ce' in f.name.lower() or 't1gd' in f.name.lower()), None)
        t2 = next((f for f in niftis if 't2w' in f.name.lower() or ('t2' in f.name.lower() and 'f' not in f.name.lower() and 'flair' not in f.name.lower())), None)
        flair = next((f for f in niftis if 'flair' in f.name.lower() or 't2f' in f.name.lower()), None)
        seg = next((f for f in niftis if 'seg' in f.name.lower() or 'mask' in f.name.lower()), None)
        
        if all([t1, t1c, t2, flair, seg]):
            try:
                # Use our safe copy function. If any file is 0 bytes, this triggers the except block.
                safe_copy_nifti(t1, IMAGES_TR / f"{subj_id}_0000.nii.gz")
                safe_copy_nifti(t1c, IMAGES_TR / f"{subj_id}_0001.nii.gz")
                safe_copy_nifti(t2, IMAGES_TR / f"{subj_id}_0002.nii.gz")
                safe_copy_nifti(flair, IMAGES_TR / f"{subj_id}_0003.nii.gz")
                
                map_brats_labels(seg, LABELS_TR / f"{subj_id}.nii.gz")
                valid_count += 1
            except Exception as e:
                # If a file is corrupted, warn the user, clean up partial files for this subject, and skip it
                tqdm.write(f"\n[WARNING] Skipping {subj_id} due to corrupted file: {e}")
                for i in range(4):
                    partial_file = IMAGES_TR / f"{subj_id}_000{i}.nii.gz"
                    if partial_file.exists(): partial_file.unlink()
                partial_label = LABELS_TR / f"{subj_id}.nii.gz"
                if partial_label.exists(): partial_label.unlink()
            
    dataset_info = {
        "channel_names": {
            "0": "T1",
            "1": "T1ce",
            "2": "T2",
            "3": "FLAIR"
        },
        "labels": {
            "background": 0,
            "necrotic_tumor_core": 1,
            "peritumoral_edema": 2,
            "enhancing_tumor": 3
        },
        "numTraining": valid_count,
        "file_ending": ".nii.gz"
    }
    
    with open(OUT_DIR / "dataset.json", "w") as f:
        json.dump(dataset_info, f, indent=4)
        
    print(f"\nnnUNet Dataset conversion complete! Successfully processed {valid_count} subjects.")

if __name__ == "__main__":
    main()