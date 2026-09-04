# Data Directory

Data roots hold runtime inputs and generated preparation files. They are not
part of the source distribution and their contents are ignored by Git.

## Runtime Roots

- `data/local/` is for a small, approved debug fixture. The default local
  profile expects prepared BraTS data at `data/local/brats` and uses only the
  debug case limit from `configs/run/debug.yaml`.
- `data/cloud/` is for the full dataset supplied by an approved cloud or GPU
  job. The default cloud profile expects prepared BraTS data at
  `data/cloud/brats`; stage or mount it before invoking the package.
- `configs/data/imagenet.yaml` uses the corresponding runtime path
  `data/<runtime>/imagenet` when the optional 2-D CNN pretraining path is used.

No dataset download, cloud mount, or machine-specific path is encoded here.
Keep source data in approved access-controlled storage and provide it to a job
through the selected runtime root.

## Prepared BraTS Layout

`token_mixer.data.cases.discover_cases` expects complete cases below a `cases/`
directory:

```text
data/<runtime>/brats/
|-- cases/
    `-- <case_id>/
        |-- t1n.nii.gz
        |-- t1c.nii.gz
        |-- t2w.nii.gz
        |-- t2f.nii.gz
        `-- segmentation.nii.gz
```

Each modality and `segmentation.nii.gz` must be a non-empty 3-D NIfTI file. The
four image files are loaded in `[t1n, t1c, t2w, t2f]` order. The segmentation
file retains source BraTS voxel values; `token_mixer.data.labels` converts raw
ET label `3` or `4` into canonical region masks in `[ET, TC, WT]` order.
`configs/data/brats.yaml` defaults `data.et_label: 4`. A label-3 source dataset
requires an explicit override `data.et_label=3`.
Without that override, ET labels are not selected.

`token_mixer.data.prepare.prepare_brats` accepts the supported nested BraTS
source layout and the nnU-Net-style `imagesTr/` plus `labelsTr/` layout. It
validates files and writes the canonical case directories without silently
rewriting label values. Preparation refuses existing case destinations unless
an explicit overwrite option is supplied.

## Manifests

`data/manifests/` contains small, tracked split metadata. The shared BraTS
manifest path configured by `configs/local.yaml` and `configs/cloud.yaml` is:

```text
data/manifests/brats_seed42.json
```

`token_mixer.data.splits.save_split_manifest` writes `dataset_id`, seed,
validation/test fractions, and sorted `train`, `val`, and `test` case IDs.
`load_split_manifest` validates the same fields before a pipeline uses them.
All BraTS segmentation experiments should consume this one manifest so model
comparisons use the same cases. CNN denoising/ImageNet pretraining is excluded
from this persisted BraTS manifest; `configs/experiment/cnn_denoising_pretrain.yaml`
uses a seeded ImageFolder split rooted at `data/<runtime>/imagenet`. A manifest
contains no voxel arrays, but case IDs can still be sensitive; share them only
when approved.

## Ignored Files

`.gitignore` ignores all contents below `data/local/` and `data/cloud/`, not
only known extensions. This covers raw NIfTI data, archives, generated
`case_index.json` files, temporary preparation files, and optional ImageNet
data. It also ignores repository `outputs/`, checkpoint directories and weight
files, W&B offline run files, Hydra runtime directories, and tool caches.

The directory markers `data/local/.gitkeep`, `data/cloud/.gitkeep`, and
`data/manifests/.gitkeep` remain visible, as do manifest files and this README.
Do not force-add raw data to bypass these rules.

## Safe Handling

Git ignore rules reduce accidental staging; they do not encrypt, delete, or
control access to files. Do not commit PHI, raw medical images, credentials, or
provider tokens. Use approved storage and access controls, inspect generated
manifests before sharing, and check `git status` and `git check-ignore` before
staging changes. Keep generated outputs and checkpoints local to the run or in
approved artifact storage.
