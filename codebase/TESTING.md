# Testing

This guide maps the active `pytest` suite to the contracts it protects. Tests
under [`tests/`](../tests/) are the executable contract surface for the package;
the prose here is a navigation aid, not a second implementation. The test
configuration is defined in [`pyproject.toml`](../pyproject.toml): pytest uses
`tests/` as its test path, `test_*.py` as its file pattern, and `src` as the
import path.

## Evidence Boundary

The suite is designed for local, repeatable contract checks. Most tests create
small tensors, arrays, temporary files, or fake external modules. They do not
require the repository's real BraTS or ImageNet data roots unless a test
explicitly opts into an external integration. Passing tests therefore establish
the tested interface and failure behavior, not model quality or experiment
results.

The active test surface is grouped into these layers:

| Layer | Active files | Contract evidence |
| --- | --- | --- |
| Package, CLI, and reproducibility | [`tests/test_package_import.py`](../tests/test_package_import.py), [`tests/test_cli_config.py`](../tests/test_cli_config.py), [`tests/test_reproducibility.py`](../tests/test_reproducibility.py) | Package version, Hydra composition, selector dispatch, config persistence, inspection-only CLI behavior, seeded Python/NumPy/PyTorch generators, and cuDNN flags |
| Data and manifests | [`tests/data/test_cases.py`](../tests/data/test_cases.py), [`tests/data/test_prepare.py`](../tests/data/test_prepare.py), [`tests/data/test_labels.py`](../tests/data/test_labels.py), [`tests/data/test_datasets.py`](../tests/data/test_datasets.py), [`tests/data/test_splits.py`](../tests/data/test_splits.py) | Case discovery, NIfTI loading, preparation, labels, transforms, dataset shapes, safe paths, split serialization, and manifest validation |
| Models and transfer | [`tests/models/`](../tests/models/) | Canonical tensor shapes, finite forwards/backwards, mixer placement, optional backends, external adapter boundaries, and ImageNet-to-3-D transfer rules |
| Evaluation | [`tests/evaluation/`](../tests/evaluation/) | Logit conversion, Dice, HD95 spacing and exclusion policy, volume inference state restoration, slice aggregation, and requested visualization outputs |
| Training state | [`tests/training/`](../tests/training/) | Phases, optimizer and scheduler behavior, finite-value guards, checkpoint payloads, tracking lifecycle, artifacts, exact resume, and warm start |
| Pipeline and integration | [`tests/pipelines/`](../tests/pipelines/), [`tests/integration/test_synthetic_debug.py`](../tests/integration/test_synthetic_debug.py) | Pipeline composition and handoff, loader metadata, best-checkpoint evaluation, transfer failure provenance, denoising export, and a small end-to-end synthetic run |

## Contract Map

### Model Contract

Model tests protect the canonical segmentation boundary rather than a claim of
architectural equivalence. The active checks cover:

- Four input channels and three raw output channels in `[ET, TC, WT]` order.
- Preservation of spatial shape and finite logits for tiny CPU forwards.
- Finite input and parameter gradients for selected models.
- MetaUNETR variant placement for `metaunetr_mamba`, `mod_a`, and `mod_b`,
  including invalid variants and invalid spatial sizes.
- The shared `encoder` seam and parameter grouping used by the fit engine.
- Lazy MONAI and Mamba boundaries, CPU fallback behavior, and failure without
  mutating module or optimizer state.
- TransUNet's external four-class to canonical three-region logit conversion,
  injected-backend path, checkout validation, and optional external integration.
- ResUNet3D transfer mapping, kernel inflation, MRI stem adaptation, coverage,
  and refusal to download weights implicitly.
- CNN denoising model shape, bounded output, dataset noise, MSE, and PSNR.

Relevant files include
[`tests/models/test_metaunetr.py`](../tests/models/test_metaunetr.py),
[`tests/models/test_resunet3d.py`](../tests/models/test_resunet3d.py),
[`tests/models/test_swinunetr.py`](../tests/models/test_swinunetr.py),
[`tests/models/test_transunet_config.py`](../tests/models/test_transunet_config.py),
[`tests/models/test_weight_transfer.py`](../tests/models/test_weight_transfer.py),
and [`tests/models/test_cnn_pretrain.py`](../tests/models/test_cnn_pretrain.py).

### Config Contract

[`tests/test_cli_config.py`](../tests/test_cli_config.py) composes the local and
cloud profiles, checks debug/full defaults, and verifies each active experiment
selects the expected model and data groups. It also checks that `_dispatch`
routes selectors without constructing models, `_run` saves the composed
configuration without changing the working directory, and successful dispatch
writes completion artifacts.

The CLI inspection cases run `--help` and `--cfg job`. They assert that the
command exits cleanly, does not emit a traceback or machine-specific project
path, and does not train. `--cfg job` is configuration evidence only; it does
not construct loaders, validate a manifest, or dispatch a runner.

### Data Contract

Data tests exercise the safe boundaries without reading the repository's large
or private data directories:

- `test_cases.py` checks complete canonical cases, sorted discovery, zero-byte
  exclusion, symlink exclusion, NIfTI float32 loading, error paths, and
  non-zero normalization.
- `test_prepare.py` checks nested and nnU-Net-style source mapping, normalized
  modality and label names, unsafe IDs, duplicate and incomplete sources,
  destination safety, atomic case-index writes, conflict preflight, staging
  rollback, and backup retention when rollback itself fails.
- `test_labels.py` checks ET label detection, fixed `[ET, TC, WT]` semantics,
  three-dimensional input requirements, binary region masks, and multiclass
  round trips.
- `test_datasets.py` checks preprocessing order, center crops, configured ET
  labels, deterministic training transforms, ambiguous layouts, dataset item
  shapes, and slice labels.

### Manifest Contract

[`tests/data/test_splits.py`](../tests/data/test_splits.py) protects strict
manifest types, finite fractions, non-overlapping IDs, sorted serialization,
round trips, required fraction metadata, and deterministic seeded splitting.
The loader tests in
[`tests/pipelines/test_baseline_pipelines.py`](../tests/pipelines/test_baseline_pipelines.py)
and [`tests/pipelines/test_train_metaunetr.py`](../tests/pipelines/test_train_metaunetr.py)
extend that contract to runtime use. They check manifest hashes, configured
dataset and split metadata, duplicate discovered IDs, missing IDs, split counts,
and the rule that `run.max_cases` is applied only after manifest validation and
case mapping.

### Resume And Checkpoint Contract

The primary reference is
[`tests/training/test_resume_contract.py`](../tests/training/test_resume_contract.py).
Read it before changing resume, warm-start, or checkpoint-root behavior. It
protects the contract across the shared 2-D and 3-D baseline wrappers, the CNN
pretraining pipeline, and the shared engine:

- Nested `run.resume`, `run.warm_start`, and `run.resume_mode` path resolution.
- Full checkpoint loading of model, optimizer, scheduler, scaler, Python,
  NumPy, PyTorch, and DataLoader generator state. CUDA RNG restoration is part
  of the implementation path but is outside this CPU-focused evidence.
- Metadata validation for architecture, model config, manifest hash, loss,
  phase plan, monitor, direction, optimizer, and scheduler.
- Exact resume preserving history, absolute epoch, global step, best metric, and
  best epoch.
- Exact resume copying the source `best.pt` into the new output before tracker
  creation or fitting, while leaving the source run untouched.
- Failure before tracker or fit when the source best checkpoint is absent.
- Warm start loading weights only, with fresh history, counters, optimizer
  state, and RNG state.
- Reproducibility fields in run artifacts.

The lower-level supporting tests are
[`tests/training/test_checkpoints.py`](../tests/training/test_checkpoints.py),
[`tests/training/test_engine.py`](../tests/training/test_engine.py),
[`tests/training/test_engine_resume.py`](../tests/training/test_engine_resume.py),
[`tests/training/test_artifacts.py`](../tests/training/test_artifacts.py), and
[`tests/training/test_tracking.py`](../tests/training/test_tracking.py).
Together they check atomic named checkpoint files, optional component rules,
phase freezing, gradient and metric guards, tracker cleanup, JSON-safe
artifacts, and the distinction between model-only load and full continuation.

## Synthetic Debug Harness

[`tests/integration/test_synthetic_debug.py`](../tests/integration/test_synthetic_debug.py)
is the active end-to-end debug harness. It uses pytest's temporary directory,
four small synthetic NIfTI cases, and a persisted synthetic split manifest. It
then exercises the following path without touching repository data:

1. Prepare nested cases through `run_prepare` and inspect the relative case index.
2. Build manifest-backed volume and slice loaders and assert four-channel image
   and three-region target shapes.
3. Run tiny MetaUNETR variants, ResUNet3D, and an injected TransUNet backend
   through forward, loss, and backward checks with finite gradients.
4. Run synthetic ResUNet3D transfer and inspect transfer metadata.
5. Run the optional tiny SwinUNETR path at a valid synthetic spatial size when
   its dependencies are available.
6. Build a four-image synthetic ImageFolder and exercise CNN denoising loss and
   gradients.
7. Save and reload a checkpoint, evaluate a region-identity model on the test
   loader, exercise the disabled tracker, and write metrics and provenance.

This harness is a wiring and contract check. It is not a full training run, a
real-data validation, or evidence that any model converges. Its outer test
requires MONAI; NIfTI and ImageFolder sections require their respective
optional file/image dependencies. The Swin section records an optional skip
when `einops` is unavailable.

## Optional Skips

Optional boundaries must be visible in test output. Use `-rs` when running
pytest so skip reasons are reported. Current skip mechanisms are:

| Boundary | Current behavior | Active tests |
| --- | --- | --- |
| `nibabel` | Tests that write or load real NIfTI fixtures call `pytest.importorskip`; placeholder discovery tests remain usable without it. | [`tests/data/test_cases.py`](../tests/data/test_cases.py), [`tests/data/test_prepare.py`](../tests/data/test_prepare.py), [`tests/integration/test_synthetic_debug.py`](../tests/integration/test_synthetic_debug.py) |
| MONAI | MONAI-only transforms and model/inference forwards use `pytest.importorskip`; the synthetic integration test is gated at its start. Tests that directly import a MONAI loss require the imaging dependency rather than silently passing. | [`tests/data/test_datasets.py`](../tests/data/test_datasets.py), [`tests/evaluation/test_inference.py`](../tests/evaluation/test_inference.py), [`tests/models/test_baseline_shapes.py`](../tests/models/test_baseline_shapes.py), [`tests/models/test_swinunetr.py`](../tests/models/test_swinunetr.py), [`tests/pipelines/test_baseline_pipelines.py`](../tests/pipelines/test_baseline_pipelines.py), [`tests/integration/test_synthetic_debug.py`](../tests/integration/test_synthetic_debug.py) |
| `einops` | MONAI SwinUNETR forward tests skip when the dependency is unavailable. Mamba CPU fallback tests do not require the external CUDA backend. | [`tests/models/test_baseline_shapes.py`](../tests/models/test_baseline_shapes.py), [`tests/models/test_swinunetr.py`](../tests/models/test_swinunetr.py), [`tests/integration/test_synthetic_debug.py`](../tests/integration/test_synthetic_debug.py) |
| Pillow | ImageFolder and synthetic image tests use `pytest.importorskip("PIL.Image")`. | [`tests/pipelines/test_pretrain_cnn.py`](../tests/pipelines/test_pretrain_cnn.py), [`tests/integration/test_synthetic_debug.py`](../tests/integration/test_synthetic_debug.py) |
| External TransUNet checkout and weights | The external integration skips unless `TRANSUNET_ROOT` and `TRANSUNET_PRETRAINED` are set and point to usable assets. Unit tests use injected backends or fake checkouts instead. | [`tests/models/test_transunet_config.py`](../tests/models/test_transunet_config.py) |
| Filesystem symlinks | Symlink safety tests skip when the host cannot create the requested symlink type. | [`tests/data/test_cases.py`](../tests/data/test_cases.py), [`tests/data/test_prepare.py`](../tests/data/test_prepare.py) |

A skip is not a passing check for the skipped boundary. Record the skip reason
with the test result and do not describe the optional integration as validated.

## Commands

Run commands from the repository root. `uv run` keeps execution inside the
project environment. Focused commands are the first evidence after a relevant
change:

```bash
# Package, configuration, and reproducibility
uv run pytest -q tests/test_package_import.py tests/test_cli_config.py tests/test_reproducibility.py

# Data and manifest contracts
uv run pytest -q tests/data

# Model contracts
uv run pytest -q tests/models

# Evaluation contracts
uv run pytest -q tests/evaluation

# Training and resume contracts
uv run pytest -q tests/training/test_resume_contract.py
uv run pytest -q tests/training

# Pipeline and synthetic integration contracts
uv run pytest -q tests/pipelines
uv run pytest -q -rs tests/integration/test_synthetic_debug.py
```

The full active suite is:

```bash
uv run pytest -q -rs
```

The repository validation ladder also includes syntax, lock, and diff checks:

```bash
uv run python -m compileall -q archive src tests
uv lock --check
git diff --check
```

After staging, repeat the whitespace check against the staged content:

```bash
git diff --cached --check
```

For a documentation-only change, these commands validate collection and source
syntax without starting training. Do not add a full training command to a
documentation check. Use [NOTEBOOKS.md](NOTEBOOKS.md) for notebook execution
and [MAINTENANCE.md](MAINTENANCE.md) for the complete change sequence.

## What Tests Do Not Prove

Even a clean full-suite result does not prove:

- Real BraTS or ImageNet files exist, are complete, or match the selected
  manifest and profile.
- NIfTI affine metadata, physical spacing, or external asset provenance is
  correct for an unseen dataset.
- Training convergence, held-out scientific quality, generalization, model
  ranking, or a meaningful comparison between 2-D slice and native 3-D
  protocols.
- GPU performance, CUDA-kernel determinism, multi-worker behavior on every
  platform, or behavior of an uninstalled optional backend.
- A real external TransUNet checkout or pretrained file works merely because
  its injected or fake boundary tests pass.
- A debug configuration, synthetic harness, checkpoint, W&B record, or output
  directory represents a completed full experiment.
- Notebook cells execute successfully or that generated `.ipynb` views are in
  sync. Notebook workflow is documented separately in [NOTEBOOKS.md](NOTEBOOKS.md).

Report these limits with experiment results. Keep runtime outputs,
checkpoints, W&B files, caches, bytecode, and paired notebook views out of the
source and documentation evidence set. Historical code under `archive/` is not
part of the active test contract.
