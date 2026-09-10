# Data Flow And Contracts

[Data-flow diagram](../assets/codebase/data-flow.svg)

This guide documents the active data boundary. Runtime behavior is defined by
the package under `src/`, the Hydra values under `configs/`, and the contract
tests under `tests/data/`. The diagram is a separate hand-authored asset; it
explains this guide but is not an implementation source of truth.

## Source Map

The claims in this guide are grounded in these active symbols:

- [`src/token_mixer/data/prepare.py::prepare_brats`](../src/token_mixer/data/prepare.py#L53-L122)
- [`src/token_mixer/pipelines/prepare_data.py::run_prepare`](../src/token_mixer/pipelines/prepare_data.py#L13-L50)
- [`src/token_mixer/data/cases.py::discover_cases`](../src/token_mixer/data/cases.py#L20-L54)
- [`src/token_mixer/data/labels.py::{detect_et_label,to_region_masks,regions_to_multiclass}`](../src/token_mixer/data/labels.py#L26-L59)
- [`src/token_mixer/data/splits.py::SplitManifest`](../src/token_mixer/data/splits.py#L70-L103)
- [`src/token_mixer/data/splits.py::create_split_manifest`](../src/token_mixer/data/splits.py#L106-L146)
- [`src/token_mixer/data/splits.py::save_split_manifest`](../src/token_mixer/data/splits.py#L149-L170)
- [`src/token_mixer/data/splits.py::load_split_manifest`](../src/token_mixer/data/splits.py#L173-L223)
- [`src/token_mixer/data/datasets.py::load_case_arrays`](../src/token_mixer/data/datasets.py#L15-L36)
- [`src/token_mixer/data/datasets.py::BratsPatchDataset`](../src/token_mixer/data/datasets.py#L47-L67)
- [`src/token_mixer/data/datasets.py::BratsVolumeDataset`](../src/token_mixer/data/datasets.py#L69-L92)
- [`src/token_mixer/data/datasets.py::BratsSliceDataset`](../src/token_mixer/data/datasets.py#L94-L144)
- [`src/token_mixer/data/transforms.py::preprocess_volume`](../src/token_mixer/data/transforms.py#L227-L269)
- [`src/token_mixer/data/transforms.py::crop_slice`](../src/token_mixer/data/transforms.py#L272-L292)
- [`src/token_mixer/data/transforms.py::build_monai_swinunetr_transform`](../src/token_mixer/data/transforms.py#L311-L345)
- [`src/token_mixer/pipelines/_baseline_common.py::build_volume_loaders`](../src/token_mixer/pipelines/_baseline_common.py#L410-L447)
- [`src/token_mixer/pipelines/_baseline_common.py::build_slice_loaders`](../src/token_mixer/pipelines/_baseline_common.py#L530-L575)
- [`src/token_mixer/pipelines/pretrain_cnn.py::build_dataloaders`](../src/token_mixer/pipelines/pretrain_cnn.py#L228-L324)

## End-To-End Flow

The supported BraTS path is:

```text
approved source layout
  -> prepare_brats
  -> canonical cases/<case_id> files
  -> discover_cases
  -> load and validate split manifest
  -> split case records
  -> patch/volume or slice datasets
  -> normalization, crop, and training transforms
  -> training and evaluation loaders
  -> pipeline handoff to training/evaluation and run artifacts
```

The CNN denoising path is separate:

```text
data/<runtime>/imagenet/<class>/*
  -> ImageFolder
  -> seeded train/validation ImageFolder subsets
  -> DenoisingDataset noisy/clean pairs
  -> 2-D CNN pretraining and encoder artifact
```

The ImageFolder path does not consume the persisted BraTS split manifest. The
BraTS loaders do not read `case_index.json`; that file is a compact preparation
descriptor for inspection and downstream tooling. Loaders rediscover canonical
cases and validate the configured manifest directly.

## Approved Source Layouts

`prepare_brats(source_root, destination_root, overwrite=False)` accepts one of
two source forms. It does not download or acquire data.

### Nested Case Directories

NIfTI files are grouped by their containing directory. The directory name is the
case ID. A source can therefore have a cohort directory above each case:

```text
<source-root>/
|-- <cohort>/
|   `-- <case-id>/
|       |-- <name containing t1n or legacy t1>.nii[.gz]
|       |-- <name containing t1c, t1ce, or t1gd>.nii[.gz]
|       |-- <name containing t2w or standalone t2>.nii[.gz]
|       |-- <name containing t2f or flair>.nii[.gz]
|       `-- <name containing seg or mask>.nii[.gz]
```

The matcher excludes `t1c`, `t1ce`, and `t1gd` from the legacy `t1` role. It
excludes `t2f` and `flair` from the standalone `t2` role. Multiple files matching
one role are ambiguous and fail preparation. A single file selected for more
than one role also fails instead of being guessed.

### nnU-Net-Style Directories

If either `imagesTr/` or `labelsTr/` exists, both must be directories:

```text
<source-root>/
|-- imagesTr/
|   |-- <case-id>_0000.nii.gz
|   |-- <case-id>_0001.nii.gz
|   |-- <case-id>_0002.nii.gz
|   `-- <case-id>_0003.nii.gz
`-- labelsTr/
    `-- <case-id>[._-]seg.nii.gz
```

Channels `0000` through `0003` map to `t1n`, `t1c`, `t2w`, and `t2f`. Labels
remove a trailing `.seg`, `_label`, `-mask`, or equivalent supported suffix
before matching the case ID. Duplicate normalized image or label IDs fail.
Files that do not match the supported NIfTI naming patterns are ignored; a case
is incomplete when any required image or label is absent.

## Preparation

`prepare_brats` performs these operations before returning prepared
`CaseRecord` values:

1. Require an existing source directory.
2. Discover complete nested or nnU-Net-style source cases.
3. Validate every source case ID before constructing destination paths.
4. Validate each NIfTI by loading it, saving the canonical destination file, and
   loading the saved file again.
5. Stage every case below a temporary preparation directory inside the
   destination root.
6. Preflight all existing canonical destinations before replacing anything.
7. Install staged cases atomically; overwrite only when the direct library call
   passes `overwrite=True`.
8. Return `discover_cases(destination_root)` after the commit.

Labels are copied without changing voxel values. Conversion from raw BraTS
values to region masks belongs to `token_mixer.data.labels`, not preparation.
`run_prepare` does not expose an overwrite flag: it reads
`cfg.paths.source_root` and `cfg.paths.data_root`, calls `prepare_brats` with
its default refusal to overwrite, and writes `case_index.json` under the
resolved data root. Each index path is relative to that data root, and the JSON
replacement uses a temporary file plus `os.replace`.

Preparation is a library/pipeline boundary, not a CLI experiment selector. The
CLI dispatch table contains segmentation and CNN experiments; it does not
contain `prepare`.

## Canonical Case Contract

Prepared BraTS data uses this repository-relative shape:

```text
data/<runtime>/brats/
`-- cases/
    `-- <case-id>/
        |-- t1n.nii.gz
        |-- t1c.nii.gz
        |-- t2w.nii.gz
        |-- t2f.nii.gz
        `-- segmentation.nii.gz
```

`discover_cases(root)` looks only below `root/cases`. It returns sorted complete
`CaseRecord` values when all five exact files are regular, non-symlink files
with non-zero size. Missing `root/cases`, a symlinked cases root, symlinked case
directories, symlinked required files, incomplete cases, and zero-byte files do
not produce a record. Discovery skips those cases rather than raising.

The discovery gate checks file presence and size. Array-level validation occurs
when `load_case_arrays` loads the case:

- Every modality and segmentation must be a 3-D NIfTI volume.
- All four modalities must share one spatial shape.
- Segmentation shape must equal the modality shape.
- Images are returned as float32 channel-first arrays with shape
  `[4, D, H, W]`.
- Segmentation remains a raw-label array with shape `[D, H, W]`; NIfTI-backed
  loads are float32, while an in-memory NumPy input retains its input dtype.

`load_nifti` converts NIfTI data to float32. Preparation's load/save check proves
that the file is readable and writable; it does not replace the explicit 3-D and
shape checks performed by `load_case_arrays`.

Modality order is fixed and must not be changed:

```text
[t1n, t1c, t2w, t2f]
```

This is the order used by `MODALITY_NAMES`, case loading, the 3-D tensor contract,
and the source-to-nnU-Net channel mapping.

## Raw Labels And Regions

Region order is fixed:

```text
[ET, TC, WT]
```

`detect_et_label` requires a 3-D segmentation and chooses label `4` when any
voxel has value `4`; otherwise it chooses `3` when any voxel has value `3`.
It raises when neither marker occurs. If both `3` and `4` occur, `4` wins.

`to_region_masks` converts raw labels to three float32 binary channels:

| Region | Active raw values |
| --- | --- |
| `ET` | `seg == et_label`, where `et_label` is detected or explicitly `3`/`4` |
| `TC` | `seg == 1` or the selected ET label |
| `WT` | `seg > 0` |

The shipped BraTS data group defaults `data.et_label: 4`. A dataset using ET
label `3` needs the explicit override `data.et_label=3`; otherwise ET voxels are
not selected when the configured label is used. An explicit setting also lets a
volume with no enhancing-tumor voxel use the intended convention without asking
the detector to infer it.

`regions_to_multiclass` converts `[ET, TC, WT]` masks to canonical class IDs
`0=background`, `1=ET`, `2=TC`, and `3=WT`. It writes broad regions first and
specific regions last, so ET has priority over TC and WT. The inverse
`multiclass_to_regions` is used at loader/evaluation seams when a four-class
slice target must become region channels.

Raw segmentation voxel values are never rewritten by preparation. Region
conversion happens in transforms or dataset adapters before model loss and
metric handling.

## Datasets And Transforms

### 3-D Case Datasets

| Dataset | One item | Training behavior | Evaluation behavior |
| --- | --- | --- | --- |
| `BratsPatchDataset` | One case as `(image, region_masks)` | Calls `preprocess_volume(..., training=True)` | Same class can disable training behavior |
| `BratsVolumeDataset` | One case as `(image, region_masks, case_id)` | Not a training dataset by default | Preserves full spatial extent when crop keys are removed |
| `BratsSliceDataset` | One flattened slice as `(image, canonical_label)` | Applies volume preprocessing and optional 2-D crop | Uses deterministic no-crop offsets, then optional center crop |

`BratsPatchDataset` and `BratsVolumeDataset` produce four image channels and
three region-mask channels. `BratsSliceDataset` converts region masks to
canonical four-class IDs before slicing, then returns a channel-first image and
`[H, W]` `uint8` labels. The shared slice loader wrapper converts those class IDs
back to three region channels before the training engine consumes them.

### Volume Preprocessing

`preprocess_volume` accepts channel-first `[4, D, H, W]` or channel-last
`[D, H, W, 4]` images. It rejects ambiguous layouts where both first and last
dimensions equal four. Labels may be raw `[D, H, W]` or binary region masks
with three channels; ambiguous mask layouts and non-binary pre-channelized masks
fail.

Processing order is:

1. Convert raw labels to `[ET, TC, WT]` masks.
2. Normalize each modality over non-zero voxels when `normalize` is true. Zero
   background remains zero. A constant non-zero channel becomes all zero after
   normalization rather than producing non-finite values.
3. Apply configured 3-D spatial padding.
4. Pad to the requested `patch_size`, `crop_size`, or `roi_size` when needed.
5. Crop a requested target. Training chooses seeded random starts; evaluation
   uses centered starts.
6. During training, apply configured flips and non-zero intensity scale, shift,
   jitter, and noise. Augmentations are driven by the configured NumPy generator
   or seed.
7. Return independent float32 image and mask arrays.

`crop_slice` applies the analogous pad/crop operation to `[C, H, W]` images and
`[H, W]` labels. The optional `build_monai_swinunetr_transform` adapter imports
MONAI only when requested, accepts a mapping containing `image` and `label` or
a `CaseRecord` under `case`, and delegates to the same canonical volume
preprocessing.

### Loader Builders

`build_volume_loaders` creates three DataLoaders from one manifest-backed case
universe:

- Training: `BratsPatchDataset`, shuffled, configured batch size and
  `drop_last`.
- Validation: `BratsVolumeDataset`, batch size `1`, no shuffle, no crop keys.
- Test: `BratsVolumeDataset`, batch size `1`, no shuffle, no crop keys.

`build_slice_loaders` uses `BratsSliceDataset` for all three splits and wraps
each DataLoader in `_CanonicalSliceLoader` so the engine receives three region
channels. Both builders pass a seeded DataLoader generator. When worker count is
non-zero, `seed_worker` is installed; pin memory, persistent workers, and
prefetch settings come from the composed configuration.

The MetaUNETR pipeline has a local `build_loaders` implementation with the same
manifest, patch-train, full-volume validation/test contract. ResUNet3D and
SwinUNETR call the shared volume builder. TransUNet calls the shared slice
builder.

### ImageFolder Datasets

`build_dataloaders` resolves the selected runtime root as
`data/<runtime>/imagenet`, requires a directory, and constructs torchvision
`ImageFolder` datasets. Class directories must contain image files. A seeded
permutation selects the configured subset, then a validation fraction is held
out with at least one validation image and at least one training image.

Training transforms convert to one or three channels, use random resized crop,
horizontal and vertical flips, rotation, and tensor conversion. Validation uses
resize, center crop, and tensor conversion. `DenoisingDataset` turns each image
into `(noisy, clean)` pairs using the configured non-negative `noise_std`.
The ImageFolder training loader honors `drop_last`; the shipped debug profile
uses two images, batch size one, and one epoch, while the shipped full profile
uses batch size two and 30 epochs.

## Split Manifest Contract

All BraTS segmentation pipelines are expected to consume the same persisted
manifest so comparisons use the same case IDs. The shipped profiles resolve the
manifest to:

```text
data/manifests/brats_seed42.json
```

The JSON written by `save_split_manifest` contains exactly these contract fields:

```json
{
  "seed": 42,
  "dataset_id": "brats",
  "val_fraction": 0.15,
  "test_fraction": 0.1,
  "train": ["..."],
  "val": ["..."],
  "test": ["..."]
}
```

`SplitManifest` validates strict seed and dataset ID types, list-of-string split
IDs, uniqueness within and across splits, and finite fractions in `[0, 1]` whose
sum is at most one. It sorts each split for stable serialization. Direct
construction can derive omitted fractions from observed split counts, but a
serialized manifest loaded from disk must contain both fraction fields.

`create_split_manifest` sorts case IDs before shuffling with
`numpy.random.default_rng(seed)`. It computes `int(case_count * fraction)` for
test and validation counts, so fractions are floored. Test cases are selected
first, validation cases next, and the remainder becomes train; each split is
sorted before return.

Before loader construction, the pipeline:

1. Loads and validates the manifest JSON.
2. Compares configured `dataset_id`, split seed, validation fraction, and test
   fraction with manifest metadata when those values are configured.
3. Rejects duplicate IDs in a split or across splits.
4. Discovers canonical cases below the selected data root.
5. Requires every manifest ID to exist in that root.
6. Applies `run.max_cases` only after manifest mapping and split selection.
7. Records manifest path, SHA-256 hash, split metadata, and effective counts in
   loader metadata for provenance.

`max_cases` is a debug limiter, not a manifest repair mechanism. A missing
manifest ID still fails before a case limit can make the run appear valid.

## Local And Cloud Roots

The active profiles use repository-relative runtime roots:

| Profile | BraTS root | ImageFolder root | Device/run default |
| --- | --- | --- | --- |
| `local` | `data/local/brats` | `data/local/imagenet` | `device: auto`, `run: debug` |
| `cloud` | `data/cloud/brats` | `data/cloud/imagenet` | `device: cuda`, `run: full` |

The same configured BraTS manifest path is used by both profiles. Therefore the
case IDs in that manifest must exist under whichever selected root is used. A
small debug fixture must not silently become the manifest for a full dataset;
use a separate manifest and profile override for a genuinely different case
universe.

Debug and full settings are resolved from the run group:

| Setting | `debug` | `full` |
| --- | ---: | ---: |
| `max_cases` | `2` | `null` |
| `batch_size` | `1` | `2` |
| `num_workers` / `workers` | `0` | `8` |
| `pin_memory` | `false` | `true` |
| `persistent_workers` | `false` | `true` |
| `phase1_epochs` | `1` | `20` |
| `phase2_epochs` | `1` | `80` |
| `epochs` for CNN pretraining | `1` | `30` |
| `validation_interval` | `1` | `5` |
| `use_amp` | `false` | `true` |
| `seed` | `42` | `42` |
| `deterministic` | `true` | `true` |

Experiment-level training values take precedence where an experiment YAML sets
them. For example, CNN pretraining sets `training.drop_last: true`, while the
segmentation experiment YAMLs set it to `false`; the run group's `drop_last`
value is not a blanket override over those explicit experiment values.

## Preflight And Failure Modes

Use config inspection before touching data. This composes settings and does not
dispatch a runner or train:

```bash
uv run python -m token_mixer --config-name local experiment=mod_a run=debug --cfg job
uv run python -m token_mixer --config-name cloud experiment=mod_b run=debug --cfg job
```

For a segmentation run, data preflight occurs when the selected pipeline builds
loaders, not during `--cfg job`. Confirm the manifest exists, then confirm its
case IDs and metadata match the selected runtime root before approving a run.
The clean checkout may contain only the manifest directory marker; a missing
manifest is an expected preflight failure until prepared metadata is supplied.

| Boundary | Failure or behavior | What to inspect |
| --- | --- | --- |
| Source root | `FileNotFoundError` when the source directory is absent | Approved source root and access permissions |
| Source completeness | Incomplete source IDs are warned and skipped; no complete case raises `ValueError` | Modality/label names and both nnU-Net directories |
| Source ambiguity | Duplicate roles, duplicate IDs, or unsafe IDs raise `ValueError` | Case names and one file per role |
| NIfTI validation | Read/save/reload failures raise `RuntimeError` with the source path | NIfTI integrity and imaging dependencies |
| Destination conflict | Existing canonical case destinations raise `FileExistsError` unless direct `overwrite=True` is used | Preserve existing cases; do not overwrite by accident |
| Staging commit | Preparation rolls back installed cases; if rollback itself fails, the backup location is retained in the raised error | Preserve the reported backup for recovery |
| Discovery | Missing roots, symlinks, empty files, or incomplete canonical cases are excluded | Exact five filenames below `cases/<case-id>` |
| Array load | Non-3-D volumes or mismatched spatial shapes raise `ValueError` | All five NIfTI array shapes |
| Labels | Missing ET markers, invalid ET override, non-3-D labels, ambiguous masks, or non-binary masks raise `ValueError` | `data.et_label`, raw values, and mask layout |
| Manifest | Missing fields, invalid types/fractions, duplicate IDs, metadata mismatch, or absent case IDs raise `ValueError` | `dataset_id`, seed, fractions, IDs, and selected root |
| Loader limits | `max_cases` is applied after manifest validation and ID mapping | Do not use it to hide missing data |
| MONAI adapter | Optional transform raises an import error when MONAI is unavailable | Install the imaging extra only when this adapter is requested |
| ImageFolder | Missing root or fewer than two usable images raises before loader creation | Class directories, image count, `drop_last`, and batch size |

Do not treat a debug composition or synthetic check as evidence of real BraTS
quality. Do not run full or cloud training as part of documentation validation.
