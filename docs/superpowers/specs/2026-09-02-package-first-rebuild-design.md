# Package-First Modular Rebuild

## Status

Approved design direction. Implementation has not started.

## Goal

Turn the current collection of research scripts into one small, importable
Python package with explicit data, model, training, evaluation, and runtime
boundaries.

Preserve the scientific purpose:

- Compare CNN plus Mamba decoder placement against established baselines.
- Keep the CNN encoder versus Mamba decoder ablation identifiable.
- Preserve SwinUNETR, residual U-Net, TransUNet, and CNN pretraining tracks.
- Make local and cloud execution differ only through configuration.
- Make every result reproducible and interpretable.

## Scope

Included:

- Move production Python code under `src/token_mixer/`.
- Create `configs/`, `data/`, `notebooks/`, and `tests/`.
- Compose runtime, data, model, experiment, and run settings with Hydra.
- Centralize BraTS case discovery, label conversion, normalization, and splits.
- Centralize full-volume inference, Dice, HD95, checkpointing, and provenance.
- Add optional W&B logging controlled by configuration.
- Remove hardcoded machine paths from production code.
- Remove stale comments and retain only short rationale comments.

Excluded:

- New model architectures.
- Automatic full-dataset or cloud training.
- Committing full NIfTI datasets, checkpoints, or generated outputs.
- Generic plugin systems, dependency injection frameworks, or model registries.
- Provider-specific cloud implementation before a provider is selected.

## Current Problems

The current repository is script-oriented rather than package-oriented:

- Data discovery and preprocessing are duplicated across three segmentation
  scripts.
- Label channels differ between scripts: `[ET, TC, WT]` versus `[TC, WT, ET]`.
- Each script creates its own split, so model comparisons do not necessarily
  use identical cases.
- Metrics and thresholding differ. One path thresholds raw logits without
  applying sigmoid first.
- Best checkpoints are not consistently reloaded before test evaluation.
- Resume handling is ineffective in at least one training path.
- Absolute `/home`, `/scratch`, `$USER`, and SLURM paths are embedded in code.
- `finetune_nnunet_brats.py` is a custom residual U-Net, not the nnUNet package.
- TransUNet depends on a checkout outside this repository.
- Model, data, training, plotting, checkpointing, and diagnostics are mixed in
  very large files.
- W&B and Hydra are declared or documented but not integrated into execution.
- No automated tests cover labels, splits, metrics, model shapes, or resume.

## Target Layout

```text
.
├── configs/
│   ├── local.yaml
│   ├── cloud.yaml
│   ├── data/
│   │   ├── brats.yaml
│   │   └── imagenet.yaml
│   ├── model/
│   │   ├── cnn_pretrain.yaml
│   │   ├── cnn_mamba_decoder.yaml
│   │   ├── resunet3d.yaml
│   │   ├── swinunetr.yaml
│   │   └── transunet.yaml
│   ├── experiment/
│   │   ├── cnn_denoising_pretrain.yaml
│   │   ├── cnn_mamba_decoder.yaml
│   │   ├── resunet3d.yaml
│   │   ├── swinunetr.yaml
│   │   └── transunet.yaml
│   └── run/
│       ├── debug.yaml
│       └── full.yaml
├── data/
│   ├── README.md
│   ├── local/
│   ├── cloud/
│   └── manifests/
├── notebooks/
│   ├── 00_data_contract.py
│   ├── 01_preprocessing_smoke.py
│   ├── 02_model_shapes.py
│   └── 03_results_analysis.py
├── src/
│   └── token_mixer/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py
│       ├── reproducibility.py
│       ├── data/
│       │   ├── cases.py
│       │   ├── datasets.py
│       │   ├── labels.py
│       │   ├── prepare.py
│       │   ├── splits.py
│       │   └── transforms.py
│       ├── models/
│       │   ├── cnn_pretrain.py
│       │   ├── resunet3d.py
│       │   ├── swinunetr.py
│       │   ├── transunet.py
│       │   ├── weight_transfer.py
│       │   └── cnn_mamba_decoder/
│       │       ├── __init__.py
│       │       ├── decoder.py
│       │       ├── encoder.py
│       │       ├── mamba.py
│       │       └── network.py
│       ├── training/
│       │   ├── checkpoints.py
│       │   ├── engine.py
│       │   ├── phases.py
│       │   └── tracking.py
│       ├── evaluation/
│       │   ├── inference.py
│       │   ├── metrics.py
│       │   └── visualization.py
│       └── pipelines/
│           ├── prepare_data.py
│           ├── pretrain_cnn.py
│           ├── train_cnn_mamba_decoder.py
│           ├── train_resunet3d.py
│           ├── train_swinunetr.py
│           └── train_transunet.py
└── tests/
    ├── data/
    │   ├── test_labels.py
    │   └── test_splits.py
    ├── evaluation/test_metrics.py
    ├── models/test_forward_shapes.py
    ├── training/test_checkpoints.py
    └── test_debug_pipeline.py
```

Production `.py` files move under `src/`. Notebook `.py` files remain under
`notebooks/` as the source for paired notebooks. Root-level experiment scripts
are removed after migration and verification.

## Runtime Profiles

`configs/local.yaml` and `configs/cloud.yaml` are the only user-facing
runtime choices.

`local.yaml` selects:

- `runtime: local`
- automatic CPU/GPU selection
- `data/local`
- debug-scale execution
- local `outputs`

`cloud.yaml` selects:

- `runtime: cloud`
- CUDA execution
- `data/cloud`
- full-scale execution
- the same repository-relative `outputs` contract

Cloud is provider-neutral. A hosted GPU and a serverless GPU both present their
dataset at `data/cloud` before invoking the package. Provider-specific data
mounting, downloading, or upload logic stays outside model and data modules.

Hydra composes the selected profile with one data config, one model config, one
experiment config, and one run-size config. It keeps model construction and
training control flow explicit in Python; no opaque `_target_` factories are
used.

Example commands:

```bash
python -m token_mixer --config-name local experiment=cnn_mamba_decoder
python -m token_mixer --config-name cloud experiment=cnn_mamba_decoder
```

Hydra must keep the process working directory unchanged. All paths resolve
from the repository root or explicit config values. The composed config is
saved with every run.

## Data Contract

`data.cases` owns one canonical `CaseRecord`:

```text
case_id
modalities.t1n
modalities.t1c
modalities.t2w
modalities.t2f
segmentation
```

`data.prepare` accepts the current nested BraTS layout and the current
standardized layout, validates NIfTI files, and writes or indexes the
repository-relative dataset location. It does not delete an existing dataset
without an explicit destructive flag.

`data.labels` accepts raw enhancing-tumor labels `3` and `4`, normalizes them to
one internal convention, and returns binary masks in one order:

```text
[ET, TC, WT]
```

All 3-D segmentation models use this order. The TransUNet adapter converts
between binary region masks and its four-class representation only at its
boundary:

```text
0 background, 1 ET, 2 TC, 3 WT
```

`data.splits` creates one deterministic manifest, for example
`data/manifests/brats_seed42.json`. Every experiment reads that manifest rather
than generating a private split. The manifest records seed, proportions,
case IDs, and source dataset identity.

Model-specific patch sizes, slice sizes, and augmentation choices remain in
model or experiment config. Shared label semantics, case identity, split
membership, and validation rules do not vary by model.

## Model Boundaries

Every model constructor receives configuration and returns a PyTorch module.
Every model receives tensors and returns raw logits. Models do not know about
filesystem paths, W&B, checkpoint files, datasets, or plots.

`cnn_mamba_decoder` preserves the current ablation:

- CNN encoder with depthwise convolution stages.
- Windowed attention at the bottleneck.
- Bidirectional axis-wise Mamba scans in coarse decoder stages.
- CNN-only full-resolution decoder refinement to avoid excessive memory use.
- Optional 2-D CNN checkpoint inflation into the 3-D encoder.

`swinunetr` remains a thin constructor around MONAI `SwinUNETR`.

`resunet3d` remains the custom 3-D residual U-Net baseline. It is not renamed
to imply use of the nnUNet package.

`transunet` becomes an adapter around the external TransUNet implementation.
Its checkout path is configuration-driven and its slice-based limitation is
recorded in experiment metadata.

`cnn_pretrain` contains the 2-D denoising autoencoder. `weight_transfer`
contains explicit, tested pure conversions from loaded state dictionaries to
3-D encoder state dictionaries. Pipelines perform checkpoint file I/O.

Each architecture module gets one short docstring containing purpose, tensor
contract, and verified paper or reference-repository citation. Historical
debug notes do not remain in model source.

## Training And Evaluation Boundaries

Pipelines assemble the following sequence:

```text
load config
-> initialize reproducibility
-> load or create split manifest
-> build datasets and loaders
-> build model
-> optionally load pretrained weights
-> run sanity checks
-> train phases
-> reload best checkpoint
-> run full-volume evaluation
-> save outputs and provenance
```

`training.engine` owns the shared loop, AMP, gradient accumulation, optimizer
steps, scheduler steps, and finite-value checks.

`training.phases` expresses encoder freeze/unfreeze and parameter groups without
duplicating the whole loop in each experiment.

`training.checkpoints` saves best, last, and resume state. Resume state includes
model, optimizer, scheduler, AMP scaler, RNG states, composed config, split
manifest hash, and code version.

`evaluation.inference` owns sliding-window inference. `evaluation.metrics`
owns one Dice and HD95 implementation. Sigmoid is applied before thresholding.
Full-volume metrics are authoritative; patch metrics are labeled diagnostic
metrics.

`evaluation.visualization` writes plots and overlays only after metrics are
computed. Plotting never controls training decisions.

## Optional W&B Tracking

`training.tracking` provides a small no-op interface and a W&B implementation.
The rest of the package calls only `log`, `log_summary`, and `finish`.

```yaml
tracking:
  enabled: false
  mode: disabled
  project: token-mixer-placement-matters
  entity: null
  run_name: null
  log_every_steps: 20
  log_images: false
  log_checkpoints: false
  directory: outputs/wandb
```

Disabled tracking performs no initialization and requires no credentials.
Online mode uses `WANDB_API_KEY` from the environment. Offline mode stores
local run files. Missing package or credentials fails before training starts,
not after an expensive run has begun.

W&B logs metrics, configuration, phase, learning rate, parameter counts,
manifest hash, and code version. Image and checkpoint upload remain opt-in.

## Reproducibility Requirements

`reproducibility.py` sets Python, NumPy, PyTorch, CUDA, and DataLoader worker
seeds. It sets:

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

The effective seed and deterministic settings are saved with each run.

## Verification Plan

The primary claim is that local/cloud selection changes execution environment,
not scientific meaning.

Evidence gates:

- Import and compile checks for the package.
- Label tests for raw labels `3` and `4`.
- Tests proving ET/TC/WT region construction and channel order.
- Split test proving identical manifests for identical seed and inputs.
- Tiny forward, loss, and backward tests for every model.
- Test proving no Mamba module exists in full-resolution decoder stage.
- Metric test proving sigmoid is applied before thresholding.
- Best-checkpoint reload and complete-resume tests.
- W&B-disabled test proving no initialization or credential requirement.
- Local debug run writing composed config, manifest hash, checkpoint, and
  metrics.
- Cloud profile dry run validating paths, device, and output locations.

Performance and segmentation quality are not claimed from smoke tests. Full
training requires explicit approval after debug evidence is reviewed.

## Migration Order

1. Create package, config, data, notebook, and test directories.
2. Add path-safe local/cloud profiles and debug/full run settings.
3. Implement and test canonical labels, case records, and split manifests.
4. Move data preparation and loader logic behind the canonical contract.
5. Split and migrate `cnn_mamba_decoder` first.
6. Migrate residual U-Net, SwinUNETR, and TransUNet adapters.
7. Extract shared training, checkpointing, evaluation, visualization, and W&B
   tracking.
8. Add Hydra CLI and pipeline commands.
9. Run unit tests and local debug workflow.
10. Compare migration outputs against available baseline fixtures.
11. Remove obsolete root scripts.
12. Run cloud dry run. Do not start full training automatically.

## Acceptance Criteria

The rebuild is ready for a full run when:

- No production module contains machine-specific absolute paths.
- One profile switch selects local or cloud execution.
- All segmentation models use the same case identity, split manifest, region
  order, and authoritative metric definitions.
- Each model can be understood from its module and experiment config without
  reading unrelated scripts.
- `cnn_mamba_decoder` still represents the CNN encoder plus Mamba decoder
  ablation.
- Best-checkpoint evaluation and resume behavior are verified.
- W&B can be enabled or disabled through config.
- Unit tests and local debug run pass.
- Generated configs and outputs identify model, data, split, seed, and code
  version.
- Full cloud training remains a deliberate user action.
