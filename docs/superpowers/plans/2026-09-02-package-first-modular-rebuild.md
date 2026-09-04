# Package-First Modular Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the scattered research scripts as one small package with shared data contracts, explicit model boundaries, local/cloud profiles, optional W&B tracking, and reproducible evaluation.

**Architecture:** Production code moves into `src/token_mixer/` and is split by responsibility: data, models, training, evaluation, and thin pipelines. Hydra composes `local.yaml` or `cloud.yaml` with data, model, experiment, and run settings; model code never owns paths, logging, checkpoint files, or metrics.

**Tech Stack:** Python 3.12, PyTorch, MONAI, Hydra, W&B, NumPy, nibabel, matplotlib, torchvision, optional timm and ml-collections for preserved model paths, pytest, Jupytext, uv.

## Global Constraints

- Preserve the paper's MetaUNETR bottleneck-Mamba baseline, Mod A encoder-Mamba
  ablation, Mod B decoder-Mamba ablation, and all four baseline tracks.
- Use descriptive experiment names: `metaunetr_mamba`, `mod_a`, `mod_b`,
  `swinunetr`, `resunet3d`, `transunet`, and `cnn_denoising_pretrain`.
- Move production `.py` files under `src/token_mixer/`; notebook `.py` files remain under `notebooks/`.
- Use `configs/local.yaml` and `configs/cloud.yaml` as the only runtime profile choices.
- Resolve paths from the repository root or explicit config values; remove `/home`, `/scratch`, `$USER`, and SLURM assumptions from production code.
- Use canonical segmentation order `[ET, TC, WT]` for all 3-D models.
- Use one persisted split manifest across all experiments.
- Apply sigmoid before binary thresholding; full-volume metrics are authoritative.
- Set Python, NumPy, PyTorch, CUDA, and DataLoader worker seeds; set `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False`.
- W&B is disabled by default and enabled only through config; no model or data module imports W&B.
- Checkpoint loading is pipeline-owned; weight-transfer helpers accept state dictionaries and return state dictionaries.
- Keep comments focused on non-obvious rationale, tensor contracts, numerical stability, or research decisions.
- Do not start full cloud training automatically. Full training requires explicit approval after debug evidence.
- Do not add new model architectures, generic plugin systems, model registries, or unnecessary framework abstractions.
- Do not commit full NIfTI datasets, checkpoints, or generated outputs.
- Cite each preserved architecture once in its module docstring using a verified paper or reference repository.

## File Map

Create or modify these production files:

```text
src/token_mixer/
├── __init__.py
├── __main__.py
├── cli.py
├── reproducibility.py
├── data/
│   ├── __init__.py
│   ├── cases.py
│   ├── datasets.py
│   ├── labels.py
│   ├── prepare.py
│   ├── splits.py
│   └── transforms.py
├── models/
│   ├── __init__.py
│   ├── cnn_pretrain.py
│   ├── resunet3d.py
│   ├── swinunetr.py
│   ├── transunet.py
│   ├── weight_transfer.py
│   ├── metaunetr/
│   │   ├── __init__.py
│   │   ├── decoder.py
│   │   ├── encoder.py
│   │   ├── mamba.py
│   │   ├── network.py
│   │   └── variants.py
├── training/
│   ├── __init__.py
│   ├── checkpoints.py
│   ├── engine.py
│   ├── phases.py
│   └── tracking.py
├── evaluation/
│   ├── __init__.py
│   ├── inference.py
│   ├── metrics.py
│   └── visualization.py
└── pipelines/
    ├── __init__.py
    ├── prepare_data.py
    ├── pretrain_cnn.py
    ├── train_metaunetr.py
    ├── train_resunet3d.py
    ├── train_swinunetr.py
    └── train_transunet.py
```

Create these configuration and support files:

```text
configs/
├── local.yaml
├── cloud.yaml
├── data/
│   ├── brats.yaml
│   └── imagenet.yaml
├── model/
│   ├── cnn_pretrain.yaml
│   ├── metaunetr.yaml
│   ├── resunet3d.yaml
│   ├── swinunetr.yaml
│   └── transunet.yaml
├── experiment/
│   ├── cnn_denoising_pretrain.yaml
│   ├── metaunetr_mamba.yaml
│   ├── mod_a.yaml
│   ├── mod_b.yaml
│   ├── resunet3d.yaml
│   ├── swinunetr.yaml
│   └── transunet.yaml
└── run/
    ├── debug.yaml
    └── full.yaml

data/
├── README.md
├── local/.gitkeep
├── cloud/.gitkeep
└── manifests/.gitkeep

notebooks/
├── 00_data_contract.py
├── 01_preprocessing_smoke.py
├── 02_model_shapes.py
└── 03_results_analysis.py
```

## Task 1: Package Scaffold And Dependencies

**Files:**
- Create: `src/token_mixer/__init__.py`
- Create: `src/token_mixer/data/__init__.py`
- Create: `src/token_mixer/models/__init__.py`
- Create: `src/token_mixer/training/__init__.py`
- Create: `src/token_mixer/evaluation/__init__.py`
- Create: `src/token_mixer/pipelines/__init__.py`
- Create: `tests/test_package_import.py`
- Modify: `pyproject.toml`
- Create: `data/README.md`
- Create: `data/local/.gitkeep`
- Create: `data/cloud/.gitkeep`
- Create: `data/manifests/.gitkeep`

**Interfaces:**
- `token_mixer.__version__ -> str`
- Package import must work after `uv sync --extra imaging --extra research --extra dev`.

- [ ] **Step 1: Write the failing package import test**

```python
from token_mixer import __version__


def test_package_exposes_version():
    assert __version__ == "0.1.0"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_package_import.py -q`

Expected: FAIL because `token_mixer` does not exist.

- [ ] **Step 3: Add package initialization**

```python
# src/token_mixer/__init__.py
__version__ = "0.1.0"
```

Create empty `__init__.py` files in each package directory.

- [ ] **Step 4: Configure src-layout packaging**

Replace the existing pytest block in `pyproject.toml` with this packaging and
test configuration:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/token_mixer"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
pythonpath = ["src"]
```

Rename the distribution name to `token-mixer-placement-matters`, set the
description to `Research pipeline for token mixer placement in medical image segmentation`, and add only preserved-model dependencies that are absent from the current manifest:

```toml
[project.optional-dependencies]
imaging = ["monai", "nibabel", "SimpleITK"]
research = ["timm", "ml-collections"]
```

Keep W&B and Hydra in the existing base dependencies. Do not add `cv2`,
`medpy`, or `tqdm`; the rebuild replaces those uses with existing PyTorch,
NumPy, and MONAI functionality.

- [ ] **Step 5: Add data directory documentation**

```markdown
# Data Directory

`local/` and `cloud/` are runtime data roots and are ignored by Git.

Use `local/` for a small debug sample. Cloud launchers must place or mount the
full dataset at `cloud/` before invoking the package. `manifests/` stores small,
tracked case split files; raw images, checkpoints, and generated outputs are
not committed.
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `uv sync --extra imaging --extra research --extra dev && uv run pytest tests/test_package_import.py -q`

Expected: PASS.

## Task 2: Canonical Labels And Case Records

**Files:**
- Create: `src/token_mixer/data/labels.py`
- Create: `src/token_mixer/data/cases.py`
- Create: `tests/data/test_labels.py`
- Create: `tests/data/test_cases.py`

**Interfaces:**
- `REGION_NAMES: tuple[str, str, str] = ("ET", "TC", "WT")`
- `detect_et_label(seg: np.ndarray) -> int`
- `to_region_masks(seg: np.ndarray, et_label: int | None = None) -> np.ndarray`
- `regions_to_multiclass(masks: np.ndarray) -> np.ndarray`
- `multiclass_to_regions(label: np.ndarray) -> np.ndarray`
- `CaseRecord(case_id: str, modalities: Mapping[str, Path], segmentation: Path)`
- `discover_cases(root: Path) -> list[CaseRecord]`
- `load_nifti(path: Path) -> np.ndarray`
- `normalize_nonzero(volume: np.ndarray) -> np.ndarray`

- [ ] **Step 1: Write failing label-contract tests**

```python
import numpy as np

from token_mixer.data.labels import (
    REGION_NAMES,
    multiclass_to_regions,
    regions_to_multiclass,
    to_region_masks,
)


def test_label_four_uses_et_tc_wt_order():
    seg = np.array([[[0, 1, 2, 4]]], dtype=np.uint8)
    masks = to_region_masks(seg)
    assert REGION_NAMES == ("ET", "TC", "WT")
    assert masks[:, 0, 0, :].tolist() == [
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0, 1.0],
    ]


def test_label_three_is_supported():
    seg = np.array([[[0, 1, 2, 3]]], dtype=np.uint8)
    assert to_region_masks(seg).shape == (3, 1, 1, 4)


def test_region_round_trip_uses_transunet_priority():
    masks = np.array([
        [[[0, 0, 0, 1]]],
        [[[0, 1, 0, 1]]],
        [[[0, 1, 1, 1]]],
    ], dtype=np.uint8)
    label = regions_to_multiclass(masks)
    assert label[0, 0].tolist() == [0, 2, 3, 1]
    np.testing.assert_array_equal(multiclass_to_regions(label), masks)
```

- [ ] **Step 2: Run label tests to verify they fail**

Run: `uv run pytest tests/data/test_labels.py -q`

Expected: FAIL because `labels.py` does not exist.

- [ ] **Step 3: Implement the label contract**

Implement `detect_et_label` to return `4` when `4` is present, otherwise `3`
when `3` is present, and raise `ValueError` for an ambiguous segmentation with
neither value. Implement region masks in `[ET, TC, WT]` order. Implement
multiclass conversion with assignment order WT, TC, ET so ET has priority.
Reject arrays with dimensions other than 3-D for raw labels or the documented
4-D shape for region masks.

- [ ] **Step 4: Run label tests to verify they pass**

Run: `uv run pytest tests/data/test_labels.py -q`

Expected: PASS.

- [ ] **Step 5: Write failing case-discovery tests**

```python
from pathlib import Path

from token_mixer.data.cases import discover_cases


def test_discover_cases_requires_all_modalities_and_label(tmp_path: Path):
    case_dir = tmp_path / "cases" / "CASE001"
    case_dir.mkdir(parents=True)
    for name in ("t1n.nii.gz", "t1c.nii.gz", "t2w.nii.gz", "t2f.nii.gz", "segmentation.nii.gz"):
        (case_dir / name).write_bytes(b"fixture")

    cases = discover_cases(tmp_path)

    assert len(cases) == 1
    assert cases[0].case_id == "CASE001"
    assert tuple(cases[0].modalities) == ("t1n", "t1c", "t2w", "t2f")


def test_incomplete_case_is_excluded(tmp_path: Path):
    case_dir = tmp_path / "cases" / "CASE001"
    case_dir.mkdir(parents=True)
    (case_dir / "t1n.nii.gz").write_bytes(b"fixture")
    assert discover_cases(tmp_path) == []
```

- [ ] **Step 6: Implement `CaseRecord` and case discovery**

Use the canonical human-readable layout:

```text
<root>/cases/<case_id>/
├── t1n.nii.gz
├── t1c.nii.gz
├── t2w.nii.gz
├── t2f.nii.gz
└── segmentation.nii.gz
```

`discover_cases` returns sorted complete cases and skips missing or zero-byte
files. `load_nifti` loads float32 data and raises a path-specific
`RuntimeError`. `normalize_nonzero` leaves zero background unchanged and
z-scores nonzero voxels.

- [ ] **Step 7: Run case tests**

Run: `uv run pytest tests/data/test_cases.py -q`

Expected: PASS.

## Task 3: Deterministic Splits And Preprocessing

**Files:**
- Create: `src/token_mixer/data/splits.py`
- Create: `src/token_mixer/data/transforms.py`
- Create: `src/token_mixer/data/datasets.py`
- Create: `tests/data/test_splits.py`
- Create: `tests/data/test_datasets.py`

**Interfaces:**
- `SplitManifest(seed: int, dataset_id: str, train: list[str], val: list[str], test: list[str])`
- `create_split_manifest(cases: Sequence[CaseRecord], seed: int, val_fraction: float, test_fraction: float, dataset_id: str) -> SplitManifest`
- `save_split_manifest(manifest: SplitManifest, path: Path) -> None`
- `load_split_manifest(path: Path) -> SplitManifest`
- `preprocess_volume(image: np.ndarray, label: np.ndarray, config: Mapping[str, Any], training: bool) -> tuple[np.ndarray, np.ndarray]`
- `BratsPatchDataset(cases, config, training: bool)`
- `BratsVolumeDataset(cases, config)`
- `BratsSliceDataset(cases, config, training: bool)`

- [ ] **Step 1: Write split reproducibility tests**

```python
from pathlib import Path

from token_mixer.data.cases import CaseRecord
from token_mixer.data.splits import create_split_manifest, save_split_manifest, load_split_manifest


def make_cases(count: int) -> list[CaseRecord]:
    return [CaseRecord(f"CASE{i:03d}", {}, Path("segmentation.nii.gz")) for i in range(count)]


def test_same_seed_produces_same_nonoverlapping_manifest(tmp_path: Path):
    cases = make_cases(20)
    first = create_split_manifest(cases, 42, 0.15, 0.10, "fixture")
    second = create_split_manifest(cases, 42, 0.15, 0.10, "fixture")
    assert first == second
    assert set(first.train).isdisjoint(first.val)
    assert set(first.train).isdisjoint(first.test)
    assert set(first.val).isdisjoint(first.test)

    path = tmp_path / "split.json"
    save_split_manifest(first, path)
    assert load_split_manifest(path) == first
```

- [ ] **Step 2: Implement split manifest creation**

Use a local `numpy.random.default_rng(seed)`, sort input cases by `case_id`
before shuffling, allocate test and validation cases once, and serialize the
dataset ID, seed, fractions, and sorted case IDs. Never recreate a test split
inside a model pipeline.

- [ ] **Step 3: Run split tests**

Run: `uv run pytest tests/data/test_splits.py -q`

Expected: PASS.

- [ ] **Step 4: Implement shared preprocessing**

Centralize nonzero normalization, padding, random crop, center crop, axis
flips, and intensity perturbation. Use the same label conversion and spatial
convention for all 3-D models. Keep patch size and slice size configurable;
do not hardcode `96` in dataset classes. Provide a small MONAI adapter for
SwinUNETR if MONAI requires dictionary transforms, but make it consume the
same `CaseRecord` and canonical label order. Raw BraTS labels use a
dataset-level `et_label` config value (`3` or `4`) when a volume contains no ET
voxels; standalone `detect_et_label` remains strict when neither label is
present, preventing silent convention guessing.

- [ ] **Step 5: Implement datasets**

`BratsPatchDataset` returns `(image, region_masks)` with shapes
`[4, D, H, W]` and `[3, D, H, W]`. `BratsVolumeDataset` returns the same
shapes plus `case_id`. `BratsSliceDataset` returns `[4, H, W]` and a
four-class label only through the TransUNet adapter.

- [ ] **Step 6: Add dataset shape tests**

Use NumPy fixtures without real NIfTI files to assert patch and volume output
shapes, canonical channel order, and deterministic center crops. Use
`pytest.importorskip("monai")` only for the MONAI-specific adapter test.

- [ ] **Step 7: Run dataset tests**

Run: `uv run pytest tests/data -q`

Expected: PASS.

## Task 4: Canonical Dataset Preparation

**Files:**
- Create: `src/token_mixer/data/prepare.py`
- Create: `src/token_mixer/pipelines/prepare_data.py`
- Create: `tests/data/test_prepare.py`

**Interfaces:**
- `prepare_brats(source_root: Path, destination_root: Path, overwrite: bool = False) -> list[CaseRecord]`
- `run_prepare(cfg: DictConfig) -> None`

- [ ] **Step 1: Write the preparation test**

Create five tiny NIfTI files under a nested subject directory, run
`prepare_brats`, and assert this output:

```text
destination_root/cases/CASE001/
├── t1n.nii.gz
├── t1c.nii.gz
├── t2w.nii.gz
├── t2f.nii.gz
└── segmentation.nii.gz
```

Assert label values `4` remain valid for the canonical loader and that a second
run without `overwrite=True` raises a clear `FileExistsError` rather than
deleting the destination.

- [ ] **Step 2: Implement preparation**

Move filename matching from `convert_to_nnunet.py:41-80` into one preparation
function. Accept both the current nested BraTS layout and current
`imagesTr/labelsTr` layout. Load and save NIfTI files to validate them. Do not
silently map raw label `4` to `3`; canonical label conversion happens in
`labels.py`, preserving source data.

- [ ] **Step 3: Add a thin preparation pipeline**

The pipeline reads `cfg.paths.source_root` and `cfg.paths.data_root`, calls
`prepare_brats`, writes a small case index, and prints counts. It contains no
model or training logic.

- [ ] **Step 4: Run preparation tests**

Run: `uv run pytest tests/data/test_prepare.py -q`

Expected: PASS.

## Task 5: Reproducibility, Phases, Checkpoints, And W&B

**Files:**
- Create: `src/token_mixer/reproducibility.py`
- Create: `src/token_mixer/training/phases.py`
- Create: `src/token_mixer/training/checkpoints.py`
- Create: `src/token_mixer/training/tracking.py`
- Create: `tests/test_reproducibility.py`
- Create: `tests/training/test_checkpoints.py`
- Create: `tests/training/test_tracking.py`

**Interfaces:**
- `seed_everything(seed: int, deterministic: bool = True) -> torch.Generator`
- `seed_worker(worker_id: int) -> None`
- `PhaseSpec(name: str, epochs: int, freeze_encoder: bool, encoder_lr: float, decoder_lr: float)`
- `apply_phase(model: nn.Module, phase: PhaseSpec) -> None`
- `CheckpointManager(root: Path)`
- `CheckpointManager.save(tag, model, optimizer, scheduler, scaler, state) -> Path`
- `CheckpointManager.load(path, model, optimizer=None, scheduler=None, scaler=None) -> dict[str, Any]`
- `Tracker.log(metrics: Mapping[str, float], step: int) -> None`
- `Tracker.log_summary(metrics: Mapping[str, float]) -> None`
- `Tracker.finish() -> None`
- `create_tracker(config: Mapping[str, Any], run_config: Mapping[str, Any]) -> Tracker`

- [ ] **Step 1: Write reproducibility tests**

```python
import random

import numpy as np
import torch

from token_mixer.reproducibility import seed_everything


def test_seed_everything_repeats_all_generators():
    seed_everything(42)
    first = (random.random(), np.random.rand(), torch.rand(1).item())
    seed_everything(42)
    second = (random.random(), np.random.rand(), torch.rand(1).item())
    assert first == second
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
```

- [ ] **Step 2: Implement reproducibility setup**

Seed Python, NumPy, PyTorch, and all CUDA devices when available. Return a
`torch.Generator` for DataLoader construction. Set the required cuDNN flags
and use `seed_worker` for worker-local NumPy and Python seeds.

- [ ] **Step 3: Write checkpoint round-trip test**

Train one `nn.Linear(2, 2)` step, save model/optimizer/scheduler/scaler and
state `{"epoch": 1, "global_step": 1}`, load into fresh objects, and assert
all model tensors and state values match.

- [ ] **Step 4: Implement checkpoint manager**

Write `best.pt`, `last.pt`, and phase-specific resume files below the pipeline
output directory. Include model state, optimizer state, scheduler state, AMP
scaler state, RNG state, composed config, manifest hash, code version, epoch,
global step, and monitored metric. File loading is explicit and pipeline-owned.

- [ ] **Step 5: Implement phase application**

Support encoder freeze for phase 1 and unfreeze for phase 2. Keep model-specific
parameter groups in the pipeline, but use one `PhaseSpec` shape and one
`apply_phase` implementation.

- [ ] **Step 6: Write tracking tests**

Assert `create_tracker({"enabled": False}, {})` returns a no-op tracker whose
`log`, `log_summary`, and `finish` methods do not import or initialize W&B.

- [ ] **Step 7: Implement optional W&B tracker**

Import W&B lazily only when enabled. Use `wandb.init(project=..., entity=...,
config=..., mode=..., dir=...)`, `run.log(...)`, and `run.finish()`. Support
`online`, `offline`, and `disabled`. Raise a clear pre-training error when
online mode is enabled but the package or `WANDB_API_KEY` is unavailable.
Log scalar train/validation metrics, phase, learning rate, parameter count,
manifest hash, code version, and composed config. Keep image and checkpoint
uploads disabled unless configured.

- [ ] **Step 8: Run training-support tests**

Run: `uv run pytest tests/test_reproducibility.py tests/training -q`

Expected: PASS.

## Task 6: Shared Training Engine And Evaluation

**Files:**
- Create: `src/token_mixer/training/engine.py`
- Create: `src/token_mixer/evaluation/metrics.py`
- Create: `src/token_mixer/evaluation/inference.py`
- Create: `src/token_mixer/evaluation/visualization.py`
- Create: `tests/training/test_engine.py`
- Create: `tests/evaluation/test_metrics.py`
- Create: `tests/evaluation/test_inference.py`

**Interfaces:**
- `FitResult(best_metric: float, best_epoch: int, history: list[dict[str, Any]])`
- `fit(model, train_loader, val_loader, loss_fn, evaluator, phases, config, tracker, checkpoints) -> FitResult`
- `logits_to_regions(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor`
- `dice_by_region(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]`
- `hd95_by_region(pred: torch.Tensor, target: torch.Tensor, spacing: tuple[float, float, float]) -> dict[str, float]`
- `evaluate_full_volumes(model, loader, roi_size, sw_batch_size, overlap, device) -> dict[str, float]`

- [ ] **Step 1: Write the metric threshold test**

```python
import torch

from token_mixer.evaluation.metrics import logits_to_regions


def test_threshold_uses_probability_half():
    logits = torch.tensor([[[[[-1.0, 0.0, 1.0, 0.6]]]]])
    result = logits_to_regions(logits, threshold=0.5)
    assert result.flatten().tolist() == [0, 0, 1, 1]
```

- [ ] **Step 2: Implement canonical metrics**

Apply sigmoid inside `logits_to_regions`, then threshold probabilities. Compute
Dice for `[ET, TC, WT]`. Use one explicit absent-mask policy: both empty gives
Dice `1.0` and HD95 `0.0`; one empty gives Dice `0.0` and HD95 `nan`. Aggregate
HD95 with `nanmean` and record excluded cases. Do not use `medpy`.

- [ ] **Step 3: Implement sliding-window inference**

Wrap MONAI `sliding_window_inference` in `evaluate_full_volumes`. The wrapper
returns raw logits to metrics, uses configured ROI and overlap, and records
case IDs. It must work under `torch.no_grad()` and restore model evaluation
state after completion.

- [ ] **Step 4: Implement the shared training engine**

The engine owns AMP context, gradient accumulation, gradient clipping, finite
loss checks, optimizer and scheduler steps, phase transitions, validation
callbacks, best-metric comparison, checkpoint saves, and tracker calls. The
pipeline supplies `loss_fn` and `evaluator`; the engine never branches on a
specific model name.

- [ ] **Step 5: Implement visualization**

Move overlay and metric plotting from the current scripts into functions that
consume arrays and output paths. Visualization functions do not load data,
create checkpoints, or make training decisions.

- [ ] **Step 6: Run shared training/evaluation tests**

Run: `uv run pytest tests/training tests/evaluation -q`

Expected: PASS.

## Task 7 (Superseded): Migrate `cnn_mamba_decoder`

> Do not execute this Mod B-only section. The active Task 7 scope is the
> paper-aligned amendment below.

**Files:**
- Create: `src/token_mixer/models/cnn_mamba_decoder/mamba.py`
- Create: `src/token_mixer/models/cnn_mamba_decoder/encoder.py`
- Create: `src/token_mixer/models/cnn_mamba_decoder/decoder.py`
- Create: `src/token_mixer/models/cnn_mamba_decoder/network.py`
- Create: `src/token_mixer/models/cnn_mamba_decoder/__init__.py`
- Create: `src/token_mixer/pipelines/train_cnn_mamba_decoder.py`
- Create: `tests/models/test_cnn_mamba_decoder.py`

**Interfaces:**
- `MambaBlock.forward(x: Tensor) -> Tensor`
- `Encoder3D.forward(x: Tensor) -> tuple[Tensor, list[Tensor]]`
- `MambaDecoder3D.forward(bottleneck: Tensor, skips: list[Tensor]) -> Tensor`
- `CnnMambaDecoder.forward(x: Tensor) -> Tensor`
- `build_model(cfg: DictConfig) -> CnnMambaDecoder`
- `run_cnn_mamba_decoder(cfg: DictConfig) -> FitResult`

- [ ] **Step 1: Write the model shape and placement tests**

```python
import torch

from token_mixer.models.cnn_mamba_decoder.network import CnnMambaDecoder


def test_cnn_mamba_decoder_output_shape_and_channel_contract():
    model = CnnMambaDecoder(
        in_channels=4,
        num_classes=3,
        feature_size=4,
        depths=(1, 1, 1, 1),
        window_size=2,
        num_heads=2,
        d_state=2,
        mamba_expand=1,
    ).eval()
    with torch.no_grad():
        logits = model(torch.randn(1, 4, 32, 32, 32))
    assert logits.shape == (1, 3, 32, 32, 32)


def test_full_resolution_decoder_has_no_mamba():
    model = CnnMambaDecoder(feature_size=4, num_heads=2, d_state=2, mamba_expand=1)
    assert not any("Mamba" in type(module).__name__ for module in model.decoder.final.modules())
```

- [ ] **Step 2: Run the model tests to verify they fail**

Run: `uv run pytest tests/models/test_cnn_mamba_decoder.py -q`

Expected: FAIL because the package model does not exist.

- [ ] **Step 3: Move the Mamba implementation**

Extract the optional CUDA import and pure PyTorch fallback from
`finetune_brats_mod_b.py:30-142` into `mamba.py`. Keep the fallback lazy and
free of global `sys.modules` mutation. The fallback must accept
`d_model`, `d_state`, `d_conv`, and `expand`, and preserve float32 SSM
accumulation before restoring the input dtype.

- [ ] **Step 4: Move the encoder**

Extract `MLP`, `DWConvBlock3D`, window partition/reversal,
`WindowedAttnBlock3D`, `Downsample3D`, and `Encoder3D` from
`finetune_brats_mod_b.py:510-727`. Preserve the active architecture and
channels-last internal representation. Return channels-first bottleneck and
skip tensors.

- [ ] **Step 5: Move the decoder and network**

Extract `TriCruciMamba3D`, `MambaDecoder3DBlock`, and `MambaDecoder3D` from
`finetune_brats_mod_b.py:734-944`. Name the final CNN-only stage `final` and
keep Mamba only in coarse decoder stages. Add `CnnMambaDecoder` to combine
encoder, decoder, and three-channel logits head.

- [ ] **Step 6: Add concise model documentation**

Each module gets one short docstring covering input/output tensors, Mamba axis
scans, and the verified architecture citation. Remove banner blocks and
historical fix logs.

- [ ] **Step 7: Move pure weight transfer**

Move `_inflate_conv_weight` and state-dict mapping from
`finetune_brats_mod_b.py:980-1147` into `models/weight_transfer.py`. The
pipeline loads `encoder_best.pth`, passes its state dictionary to a pure
function, checks critical keys and coverage, then loads the returned state.
Do not silently accept coverage below the configured minimum.

- [ ] **Step 8: Create the primary pipeline**

The pipeline builds canonical loaders, constructs the model, optionally loads
the pretraining artifact, runs sanity checks, creates phase specs, calls the
shared engine, reloads the best checkpoint, evaluates full volumes, writes
metrics and plots, and returns `FitResult`.

- [ ] **Step 9: Run primary model tests**

Run: `uv run pytest tests/models/test_cnn_mamba_decoder.py -q`

Expected: PASS on CPU using the fallback implementation.

## Task 7 (Active): Paper MetaUNETR Variants

This amendment supersedes the preceding Mod B-only Task 7 execution boundary.
The paper track is the only shipped model scope for this task and is the primary
experimental implementation for the manuscript table.

**Primary sources:**

- Oyetunji et al., *Token Mixer Placement Matters: A Systematic Encoder–Decoder
  Ablation Study of Mamba for Brain Tumour Segmentation on BraTS-Africa* (the
  manuscript supplied with this task).
- Lyu et al., *MetaUNETR: Rethinking Token Mixer Encoding for Efficient
  Multi-Organ Segmentation*, MICCAI 2024,
  https://papers.miccai.org/miccai-2024/paper/2749_paper.pdf.
- Official MetaUNETR implementation,
  https://github.com/lyupengju/MetaUNETR.
- Gu and Dao, *Mamba: Linear-Time Sequence Modeling with Selective State
  Spaces*, https://arxiv.org/abs/2312.00752, and the official implementation,
  https://github.com/state-spaces/mamba.

**Paper variants:**

- `metaunetr_mamba`: MetaUNETR baseline with Mamba only in bottleneck token
  mixing and the shared CNN decoder.
- `mod_a`: Mamba in every encoder token-mixer slot and the shared CNN decoder.
- `mod_b`: CNN encoder and Mamba decoder under the same paper feature widths,
  input/output contracts, and training configuration.

The paper track uses four MRI input channels, three raw-logit output channels,
base feature width 48, stage depths `(2, 2, 2, 2)`, stage widths
`(48, 96, 192, 384)`, and the paper's 96-cubed patch/inference settings when
composed by later configs. Tests use smaller explicit widths and volumes. The
paper manuscript's axis-fusion equation and the public MetaUNETR fragment differ
(`sum` versus `cat` followed by a projection); expose `axis_fusion` as a model
configuration, default the paper experiment to the manuscript equation, and
retain a reference-compatible `cat` mode. Record the selected mode in result
metadata rather than claiming exact reproduction without the experiment config.

**Files:**
- Create: `src/token_mixer/models/metaunetr/__init__.py`
- Create: `src/token_mixer/models/metaunetr/mamba.py`
- Create: `src/token_mixer/models/metaunetr/encoder.py`
- Create: `src/token_mixer/models/metaunetr/decoder.py`
- Create: `src/token_mixer/models/metaunetr/network.py`
- Create: `src/token_mixer/models/metaunetr/variants.py`
- Create: `src/token_mixer/pipelines/train_metaunetr.py`
- Create: `tests/models/test_metaunetr.py`

**Interfaces:**
- `MetaUNETR.forward(x: Tensor) -> Tensor`
- `build_metaunetr(cfg: DictConfig, variant: str) -> MetaUNETR`
- `run_metaunetr(cfg: DictConfig) -> FitResult`

- [x] **Step 1: Write paper-variant contract tests**

```python
import torch

from token_mixer.models.metaunetr.variants import build_metaunetr


def test_paper_variants_share_raw_logit_contract():
    for variant in ("metaunetr_mamba", "mod_a", "mod_b"):
        model = build_metaunetr(
            {
                "in_channels": 4,
                "num_classes": 3,
                "base_channels": 4,
                "depths": (1, 1, 1, 1),
                "window_size": 2,
                "num_heads": 2,
                "d_state": 2,
                "d_conv": 2,
                "mamba_expand": 1,
                "axis_fusion": "sum",
            },
            variant,
        ).eval()
        with torch.no_grad():
            logits = model(torch.randn(1, 4, 32, 32, 32))
        assert logits.shape == (1, 3, 32, 32, 32)
```

- [x] **Step 2: Run paper-variant tests to verify they fail**

Run: `uv run pytest tests/models/test_metaunetr.py -q`

Expected: FAIL because paper model modules do not exist.

- [x] **Step 3: Implement shared paper Mamba and scan primitives**

Implement lazy optional `mamba_ssm` selection and a pure PyTorch fallback. The
fallback accepts `d_model`, `d_state`, `d_conv`, and `expand`, accumulates SSM
state in float32, restores the input dtype, and never mutates `sys.modules` or
global CUDA/backend settings. Implement three-axis channels-last cross-scans
with configurable `sum` or `cat` fusion and preserve `[B, D, H, W, C]` shape.

- [x] **Step 4: Implement shared paper encoder and CNN decoder**

Use channels-first tensors at convolution boundaries and channels-last tensors
inside token-mixer blocks. Implement stem, four stages, downsampling, final
`16 * base_channels` bottleneck, skip outputs, and two residual sub-blocks per
stage. Token-mixer selection must support CNN, bottleneck-only Mamba, and
all-stage Mamba without model-name branches in the training engine. Use MONAI
`UnetrUpBlock` semantics for the paper CNN decoder, with lazy optional imports
and a clear error when imaging dependencies are unavailable.

- [x] **Step 5: Implement paper baseline, Mod A, and Mod B variants**

Keep one `MetaUNETR` network and select only encoder/decoder mixer placement:
`metaunetr_mamba` puts Mamba at bottleneck, `mod_a` puts Mamba in all encoder
token-mixer slots, and `mod_b` puts Mamba in decoder refinement blocks. All
variants retain identical channel widths, skip ordering, raw-logit head, and
configurable axis-fusion metadata. Do not add result claims or paper metrics to
the code.

- [x] **Step 6: Add paper citations and provenance documentation**

Document the supplied manuscript as the project-specific source, MetaUNETR as
the shared framework source, Mamba as the primitive source, and clearly label
the project's 3-D cross-scan and placement changes. State the public
MetaUNETR `sum`/`cat` discrepancy and selected configuration in module
docstrings/config metadata.

- [x] **Step 7: Create the paper pipeline boundary**

`train_metaunetr.py` selects one paper variant, builds canonical loaders, calls
the shared engine, and records variant, widths, depths, scan direction, and
axis-fusion mode. It keeps model construction and checkpoint file I/O outside
model modules; pretraining artifact conversion remains supplied by Task 8. The
legacy `finetune_brats_mod_b.py` script is reference-only and is not packaged.

- [x] **Step 8: Run model tests and CPU verification**

Run:

```bash
uv run pytest tests/models/test_metaunetr.py -q
uv run pytest -q
uv run python -m compileall -q src tests
```

Expected: all three paper variant shape/placement tests pass on CPU, the full
suite passes with only explicitly optional/external skips, and compilation exits
with no output.

Task 8 remains responsible for the 2-D denoising pretraining model and tested
2-D-to-3-D state-dict inflation. Task 9 remains responsible for nnU-Net-style,
SwinUNETR, and TransUNet baselines; it must not substitute those models for the
MetaUNETR rows in the paper table.

## Task 8: Migrate CNN Pretraining And Weight Transfer

**Files:**
- Create: `src/token_mixer/models/cnn_pretrain.py`
- Create: `src/token_mixer/pipelines/pretrain_cnn.py`
- Create: `src/token_mixer/models/weight_transfer.py`
- Create: `tests/models/test_weight_transfer.py`

**Interfaces:**
- `DenoisingAutoencoder.forward(x: Tensor) -> Tensor`
- `build_denoising_model(cfg: DictConfig) -> DenoisingAutoencoder`
- `inflate_encoder_state_dict(source: Mapping[str, Tensor], target: Mapping[str, Tensor]) -> tuple[dict[str, Tensor], dict[str, int]]`
- `run_cnn_denoising_pretrain(cfg: DictConfig) -> FitResult`

- [x] **Step 1: Write weight-transfer shape tests**

Test direct linear/normalization copies, depth inflation, MRI stem adaptation,
critical-key failure, and explicit skipped-key reporting. Use synthetic state
dictionaries with known shapes.

- [x] **Step 2: Move the denoising autoencoder**

Extract `DenoisingDataset`, `CNNBlock2D`, `PretrainCNNEncoder`, decoder stages,
loss, PSNR, and reconstruction plotting from `pretrain_cnn.py`. Remove hardcoded
paths and use canonical config paths. Keep 2-D ImageNet pretraining distinct
from 3-D segmentation training.

- [x] **Step 3: Implement and test state-dict inflation**

Preserve direct copies where shapes match. Implement explicit rules for the
stem, depthwise/standard convolution mismatch, and 2-D to 3-D depth kernels.
Return counts for direct, inflated, skipped, and total target keys. Raise when
critical encoder entry points are absent or coverage is below the experiment
threshold.

- [x] **Step 4: Create the pretraining pipeline**

Use the shared reproducibility, checkpoint, and optional tracking utilities.
Write `encoder_best.pth` under the configured experiment output and include
the source model config in checkpoint metadata.

- [x] **Step 5: Run pretraining and transfer tests**

Run: `uv run pytest tests/models/test_weight_transfer.py -q`

Expected: PASS.

## Task 9: Migrate Remaining Models And Adapters

**Files:**
- Create: `src/token_mixer/models/resunet3d.py`
- Create: `src/token_mixer/models/swinunetr.py`
- Create: `src/token_mixer/models/transunet.py`
- Create: `src/token_mixer/pipelines/train_resunet3d.py`
- Create: `src/token_mixer/pipelines/train_swinunetr.py`
- Create: `src/token_mixer/pipelines/train_transunet.py`
- Create: `tests/models/test_baseline_shapes.py`
- Create: `tests/models/test_transunet_config.py`

**Interfaces:**
- `build_resunet3d(cfg: DictConfig) -> nn.Module`
- `build_swinunetr(cfg: DictConfig) -> nn.Module`
- `build_transunet(cfg: DictConfig) -> nn.Module`
- `run_resunet3d(cfg: DictConfig) -> FitResult`
- `run_swinunetr(cfg: DictConfig) -> FitResult`
- `run_transunet(cfg: DictConfig) -> FitResult`

- [ ] **Step 1: Migrate the residual U-Net**

Extract `ResBlock3D`, `EncStage`, `DecStage`, and `ResUNet3D` from
`finetune_nnunet_brats.py:312-428`. Keep it explicitly named `resunet3d`.
Move ImageNet initialization into the tested weight-transfer path. Fix
validation to use canonical logits-to-regions conversion, reload best before
test evaluation, and make resume restore the model actually used by training.

- [ ] **Step 2: Migrate SwinUNETR**

Create a thin constructor around MONAI `SwinUNETR` using config values. Remove
global model creation, global output directory creation, and local split logic
from `train_swinunetr_new.py`. Use the shared canonical data adapter and
persisted split manifest.

- [ ] **Step 3: Migrate TransUNet adapter**

Move preprocessing and the external model wrapper from
`finetune_transunet.py`. Replace `cv2` resizing with PyTorch interpolation or
NumPy-compatible existing operations. Replace `medpy` metric calls with the
shared evaluation contract. Read `third_party.transunet_root` from config and
raise a clear error before training if the checkout or pretrained file is
missing. Keep its 2-D slice-based nature explicit in result metadata.

- [ ] **Step 4: Add baseline shape tests**

Run tiny CPU forward tests for ResUNet3D and SwinUNETR when imaging dependencies
are installed. Test TransUNet config validation without requiring the external
checkout; run the full adapter test only when its configured source exists.

- [ ] **Step 5: Run baseline tests**

Run: `uv run pytest tests/models -q`

Expected: PASS, with only the explicitly external TransUNet integration test
skipped when its configured checkout is absent.

## Task 10: Add Hydra Config Composition And CLI

**Files:**
- Create: `src/token_mixer/cli.py`
- Create: `src/token_mixer/__main__.py`
- Create: `configs/local.yaml`
- Create: `configs/cloud.yaml`
- Create: `configs/data/brats.yaml`
- Create: `configs/data/imagenet.yaml`
- Create: `configs/model/cnn_pretrain.yaml`
- Create: `configs/model/metaunetr.yaml`
- Create: `configs/model/resunet3d.yaml`
- Create: `configs/model/swinunetr.yaml`
- Create: `configs/model/transunet.yaml`
- Create: `configs/experiment/cnn_denoising_pretrain.yaml`
- Create: `configs/experiment/metaunetr_mamba.yaml`
- Create: `configs/experiment/mod_a.yaml`
- Create: `configs/experiment/mod_b.yaml`
- Create: `configs/experiment/resunet3d.yaml`
- Create: `configs/experiment/swinunetr.yaml`
- Create: `configs/experiment/transunet.yaml`
- Create: `configs/run/debug.yaml`
- Create: `configs/run/full.yaml`
- Create: `tests/test_cli_config.py`

**Interfaces:**
- `main() -> None`
- CLI dispatches by `cfg.experiment.name` to one pipeline.
- Native Hydra selection uses `--config-name local|cloud` and group overrides.

- [ ] **Step 1: Write config composition tests**

```python
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def test_local_profile_composes_debug_experiment():
    with initialize_config_dir(version_base=None, config_dir=str(Path("configs").resolve())):
        cfg = compose(config_name="local", overrides=["experiment=mod_a"])
    assert cfg.runtime == "local"
    assert cfg.run.name == "debug"
    assert cfg.experiment.name == "mod_a"
    assert cfg.paths.data_root.endswith("data/local/brats")
    assert OmegaConf.select(cfg, "tracking.enabled") is False


def test_cloud_profile_composes_full_experiment():
    with initialize_config_dir(version_base=None, config_dir=str(Path("configs").resolve())):
        cfg = compose(config_name="cloud", overrides=["experiment=mod_b"])
    assert cfg.runtime == "cloud"
    assert cfg.run.name == "full"
    assert cfg.device == "cuda"
    assert cfg.paths.data_root.endswith("data/cloud/brats")
```

- [ ] **Step 2: Create profile YAML files**

Use this shape for `configs/local.yaml`:

```yaml
defaults:
  - experiment: mod_a
  - run: debug
  - _self_

runtime: local
device: auto
paths:
  data_root: ${hydra:runtime.cwd}/data/local/brats
  output_root: ${hydra:runtime.cwd}/outputs
  manifest: ${hydra:runtime.cwd}/data/manifests/brats_seed42.json
tracking:
  enabled: false
  mode: disabled
  project: token-mixer-placement-matters
  entity: null
  run_name: null
  log_every_steps: 20
  log_images: false
  log_checkpoints: false
  directory: ${paths.output_root}/wandb
hydra:
  job:
    chdir: false
```

`configs/cloud.yaml` uses the same keys, with `runtime: cloud`, `device: cuda`,
`data/cloud/brats`, and `run: full`. Keep profiles provider-neutral.

- [ ] **Step 3: Create data, model, experiment, and run groups**

`data/brats.yaml` defines modality names, canonical region names, spacing,
patch size, and split fractions. `data/imagenet.yaml` defines ImageFolder
root and image size. Model files define architecture values only. Experiment
files define model/data selection, loss, optimizer, phase learning rates,
validation interval, and checkpoint metric. Run files define debug/full case
limits, workers, and epoch overrides.

- [ ] **Step 4: Implement CLI entrypoint**

```python
# src/token_mixer/__main__.py
from token_mixer.cli import main


if __name__ == "__main__":
    main()
```

`cli.py` uses `@hydra.main(version_base=None, config_path="../../configs",
config_name="local")`, dispatches to the named pipeline, and never constructs
models directly. Save the composed config under the run output directory.

- [ ] **Step 5: Run CLI/config tests**

Run: `uv run pytest tests/test_cli_config.py -q`

Expected: PASS.

- [ ] **Step 6: Verify user-facing commands**

Run:

```bash
uv run python -m token_mixer --config-name local experiment=mod_a run=debug --help
uv run python -m token_mixer --config-name cloud experiment=mod_b run=full --cfg job
```

Expected: help/config output contains no machine-specific absolute path and
does not start training.

## Task 11: Add Notebooks, README, And Data Instructions

**Files:**
- Create: `notebooks/00_data_contract.py`
- Create: `notebooks/01_preprocessing_smoke.py`
- Create: `notebooks/02_model_shapes.py`
- Create: `notebooks/03_results_analysis.py`
- Modify: `README.md`
- Modify: `data/README.md`
- Modify: `.gitignore`

- [ ] **Step 1: Create notebook source files**

Use Jupytext percent format. `00_data_contract.py` exercises labels and case
records. `01_preprocessing_smoke.py` loads the local fixture and prints tensor
shapes. `02_model_shapes.py` runs every model with debug config. `03_results_analysis.py`
reads saved metrics and creates comparison tables. Notebooks call package
functions; they do not become a second implementation.

- [ ] **Step 2: Sync Python notebooks to notebooks**

Run one-way synchronization only:

```bash
jupytext --sync notebooks/00_data_contract.py
jupytext --sync notebooks/01_preprocessing_smoke.py
jupytext --sync notebooks/02_model_shapes.py
jupytext --sync notebooks/03_results_analysis.py
```

Do not sync from `.ipynb` to `.py`.

- [ ] **Step 3: Document commands and profiles**

README must explain:

```bash
uv sync --extra imaging --extra research --extra dev
uv run python -m token_mixer --config-name local experiment=mod_a
uv run python -m token_mixer --config-name cloud experiment=mod_b
```

Document that local data belongs under `data/local`, cloud jobs must provide
full data under `data/cloud`, raw datasets and generated outputs are ignored,
and full training requires explicit approval.

- [ ] **Step 4: Update ignore rules**

Ignore contents of `data/local`, `data/cloud`, `outputs`, checkpoints, W&B
offline runs, caches, and Hydra runtime directories while retaining
`README.md`, `.gitkeep`, manifests, source, configs, and tests.

## Task 12: End-To-End Debug Verification And Cleanup

**Files:**
- Modify: all migrated package/config/test files as required by verification
- Delete after parity: `convert_to_nnunet.py`
- Delete after parity: `dataset.py`
- Delete after parity: `finetune_brats_mod_b.py`
- Delete after parity: `finetune_nnunet_brats.py`
- Delete after parity: `finetune_transunet.py`
- Delete after parity: `pretrain_cnn.py`
- Delete after parity: `train_swinunetr_new.py`
- Delete after replacement: `main.py`

- [ ] **Step 1: Run all unit tests**

Run: `uv run pytest -q`

Expected: all package, data, model, training, evaluation, and config tests
pass. External TransUNet tests may skip only when its configured checkout is
absent.

- [ ] **Step 2: Run static checks**

Run:

```bash
uv run python -m compileall -q src tests
uv run python -m token_mixer --config-name local experiment=mod_a run=debug --cfg job
```

Expected: zero syntax errors and a composed debug config with repository-rooted
paths.

- [ ] **Step 3: Run the local synthetic debug pipeline**

Use a generated tiny fixture with at least four complete cases. Run
`prepare`, one forward/loss/backward cycle for every model, one checkpoint
round trip, one full-volume metric pass, and disabled W&B tracking. Assert
outputs contain:

```text
outputs/<experiment>/<run_id>/
├── config.yaml
├── metrics.json
├── checkpoints/best.pt
└── provenance.json
```

No full dataset or long training is used in this step.

- [ ] **Step 4: Verify scientific contracts**

Check that every model uses the same case IDs from
`data/manifests/brats_seed42.json`, returns `[ET, TC, WT]` for 3-D models,
uses the shared sigmoid threshold, and reloads `best.pt` before evaluation.
Check that the three MetaUNETR variants share the paper channel/label/logit
contracts and differ only in the configured Mamba placement. Check that
weight-transfer coverage is recorded.

- [ ] **Step 5: Run cloud dry run**

Compose `cloud.yaml` with `run=debug` and verify device, data root,
output root, manifest path, and W&B settings. Do not invoke full training.

- [ ] **Step 6: Remove obsolete root scripts**

After tests and debug verification pass, remove the old scripts. Confirm no
package import references a deleted root module:

```bash
uv run python -c "import token_mixer; print(token_mixer.__version__)"
uv run pytest -q
git status --short
```

Expected: package imports, tests pass, and only intended new structure and
existing user files appear in Git status.

## Spec Coverage Review

- Runtime profiles and provider-neutral data paths: Tasks 1 and 10.
- Canonical labels and shared splits: Tasks 2 and 3.
- Safe data preparation: Task 4.
- Reproducibility and deterministic CUDA settings: Task 5.
- Optional W&B online/offline/disabled modes: Task 5.
- Shared training, checkpoint, metrics, and inference: Tasks 5 and 6.
- CNN plus Mamba decoder preservation: Task 7.
- CNN denoising pretraining and weight inflation: Task 8.
- ResUNet3D, SwinUNETR, and TransUNet preservation: Task 9.
- Hydra composition and one-command profile switching: Task 10.
- Notebook source-of-truth and documentation: Task 11.
- Debug evidence and no automatic full training: Task 12.
- Focused comments and citations: Tasks 7, 8, and 11.

## Execution Note

This plan deliberately contains no commit step because commit operations
require explicit user instruction in this workspace. Implementation should
still be reviewed after each task boundary.
