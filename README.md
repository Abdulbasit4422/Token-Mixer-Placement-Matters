# Token Mixer Placement Matters

Research pipeline for studying where token mixers belong in medical image
segmentation models. The package supports the MetaUNETR paper track and its
`metaunetr_mamba`, `mod_a`, and `mod_b` variants, together with the preserved
ResUNet3D, SwinUNETR, TransUNet, and 2-D CNN pretraining paths.

## Package Architecture

Production code lives under `src/token_mixer/`. Hydra configuration lives under
`configs/`; the command-line entry point is `token_mixer.cli`, exposed through
`src/token_mixer/__main__.py`.

| Package path | Responsibility |
| --- | --- |
| `token_mixer.data` | BraTS case discovery and preparation, preprocessing, labels, datasets, and deterministic split manifests |
| `token_mixer.models` | Tensor-only model constructors and adapters, including `models/metaunetr/` and optional external TransUNet integration |
| `token_mixer.training` | Shared phase-based training loop, checkpoint persistence, completion artifacts, reproducibility state, and optional W&B tracking |
| `token_mixer.evaluation` | Sliding-window inference, canonical Dice/HD95 metrics, and plots/overlays |
| `token_mixer.pipelines` | Thin orchestration boundaries that compose data, models, training, and evaluation |
| `configs/` | Local/cloud profiles plus data, model, experiment, and debug/full run groups |

Model modules operate on tensors. They do not own dataset paths, checkpoint
files, W&B initialization, or plotting. Pipelines own those boundaries, while
the shared training engine in `token_mixer.training.engine` remains independent
of any one model name.

Eight historical scripts are preserved under `archive/` and are not supported
entry points: `convert_to_nnunet.py`, `dataset.py`, `finetune_brats_mod_b.py`,
`finetune_nnunet_brats.py`, `finetune_transunet.py`, `main.py`, `pretrain_cnn.py`,
and `train_swinunetr_new.py`. The supported entry point is
`uv run python -m token_mixer`, which invokes the Hydra CLI.

## Install

The supported setup installs the base package plus imaging, research, and
development extras:

```bash
uv sync --extra imaging --extra research --extra dev
```

The `imaging` extra provides MONAI, `einops`, nibabel, and SimpleITK. SwinUNETR
requires MONAI and `einops`; MONAI inference/loss adapters and the MetaUNETR
3-D decoder also require MONAI, while nibabel is used for NIfTI loading and
preparation. The `research` extra provides `timm` and `ml-collections` for
optional preserved research paths. The `dev` extra provides the Jupyter and
test tooling used by the repository notebooks and tests.

TransUNet is a separate optional dependency boundary. Its adapter in
`src/token_mixer/models/transunet.py` expects an official
[Beckschen/TransUNet](https://github.com/Beckschen/TransUNet) checkout and a
compatible pretrained `.npz` file. Configure these through
`third_party.transunet_root` and `third_party.pretrained_path`; the checkout and
its own dependencies are not bundled in this package. The configured
`configs/model/transunet.yaml` path preserves the official `R50-ViT-B_16`
2-D slice model.

## Data And Profiles

Runtime data is deliberately outside version control:

- `configs/local.yaml` resolves BraTS data to `data/local/brats`, selects
  `runtime: local`, chooses automatic device selection, and defaults to
  `run: debug` with `max_cases: 2`.
- `configs/cloud.yaml` resolves BraTS data to `data/cloud/brats`, selects
  `runtime: cloud`, requests CUDA, and defaults to `run: full` with no case
  limit.
- Both profiles use `data/manifests/brats_seed42.json` and write into the same
  repository-relative `outputs/` tree. Cloud infrastructure must stage or mount
  data at `data/cloud` before starting the package; provider-specific transfer
  and mounting logic is outside this repository.

Prepare data into the canonical layout described in [`data/README.md`](data/README.md).
Preparation is library-only: no `prepare` experiment is dispatched by the CLI.
Call `token_mixer.data.prepare.prepare_brats` to validate supported source
layouts and copy complete cases below the selected runtime root, or call
`token_mixer.pipelines.prepare_data.run_prepare`, which invokes it and writes
portable `case_index.json` under that data root. After preparation, create the
manifest explicitly with `token_mixer.data.cases.discover_cases`,
`token_mixer.data.splits.create_split_manifest`, and
`token_mixer.data.splits.save_split_manifest`.

## Label And Logit Contract

All segmentation paths share one contract:

- BraTS raw segmentation accepts enhancing-tumor label `3` or `4`; conversion
  is centralized in `token_mixer.data.labels`.
- `configs/data/brats.yaml` defaults `data.et_label: 4`. A label-3 source
  dataset requires an explicit override `data.et_label=3`. Without that
  override, ET labels are not selected.
- The canonical region order is exactly `[ET, TC, WT]`, as exposed by
  `REGION_NAMES`.
- 3-D MRI inputs use four channel-first modalities in
  `[t1n, t1c, t2w, t2f]` order, normally shaped `[B, 4, D, H, W]`.
- 3-D segmentation models return three channel-first **raw logits**, normally
  shaped `[B, 3, D, H, W]`, in `[ET, TC, WT]` order. Models do not apply
  sigmoid or thresholding.
- `token_mixer.evaluation.metrics.logits_to_regions` applies sigmoid and then
  thresholds probabilities. This is the only shared logits-to-region boundary.
- TransUNet remains an explicit 2-D slice adapter. The external four-class
  representation is `0 background, 1 ET, 2 TC, 3 WT`; the adapter converts its
  output back to canonical raw region logits `[ET, TC, WT]` before shared
  metric handling.

This contract keeps model placement as the experimental variable. Channel
order, case identity, and split membership remain fixed between variants. The
shared metric implementation is consistent, but evaluation aggregation and
physical-spacing assumptions can differ by model family; see
[`Reproducibility`](#reproducibility).

## Manifests And Splits

Create one split manifest after the prepared cases are available. The split
implementation in `token_mixer.data.splits` sorts case IDs, shuffles with the
configured seed, assigns validation and test cases once, and saves a small JSON
record containing:

- `dataset_id`
- `seed`
- `val_fraction` and `test_fraction`
- sorted `train`, `val`, and `test` case IDs

The BraTS config uses seed `42`, validation fraction `0.15`, and test fraction
`0.10`; the expected shared file is
`data/manifests/brats_seed42.json`. Every BraTS segmentation experiment loads
this manifest through `paths.manifest` instead of creating a private split.
This persisted BraTS manifest does not apply to CNN denoising/ImageNet
pretraining. The `configs/experiment/cnn_denoising_pretrain.yaml` config uses a
seeded ImageFolder split under `data/<runtime>/imagenet`. The manifest is reviewable
metadata, not image data, and should remain tracked. If the prepared dataset or
dataset identity changes, create and review a new manifest rather than silently
reusing the old one.

## Supported Experiments And Dispatch

CLI supports exactly these experiment names:

| `experiment.name` | Dispatched runner |
| --- | --- |
| `cnn_denoising_pretrain` | `run_cnn_denoising_pretrain` |
| `metaunetr_mamba`, `mod_a`, `mod_b` | `run_metaunetr` |
| `resunet3d` | `run_resunet3d` |
| `swinunetr` | `run_swinunetr` |
| `transunet` | `run_transunet` |

An unknown experiment name raises `ValueError`; preparation is not one of the
dispatched experiment names.

## Commands

Local debug execution uses `configs/local.yaml` and `configs/experiment/mod_a.yaml`:

```bash
uv run python -m token_mixer --config-name local experiment=mod_a
```

Cloud/full execution uses `configs/cloud.yaml` and
`configs/experiment/mod_b.yaml`:

```bash
uv run python -m token_mixer --config-name cloud experiment=mod_b
```

The cloud command selects `run: full`; run it only after explicit approval for
the associated expensive compute and data use. To inspect the composed cloud
configuration without training, override the run group and request config
output:

```bash
uv run python -m token_mixer --config-name cloud experiment=mod_b run=debug --cfg job
```

Use only experiment names listed above. Imports, installation, config
composition, and notebook smoke checks do not start full training. No
full-dataset or cloud run is launched automatically.
Every full, long-running, paid, or otherwise expensive run requires explicit human approval before execution.

## Outputs And Checkpoints

Hydra resolves each run to the configured `paths.experiment_output` directory:

### Segmentation Runs

```text
outputs/
|-- <runtime>/<experiment>/<run>/<YYYY-MM-DD>/<HH-MM-SS>/
    |-- config.yaml
    |-- metrics.json
    |-- provenance.json
    `-- checkpoints/
        |-- best.pt
        |-- last.pt
        `-- <phase>_resume.pt
```

`src/token_mixer/cli.py` saves the composed Hydra config as `config.yaml`.
`token_mixer.training.checkpoints.CheckpointManager` writes `best.pt`,
`last.pt`, and phase-specific resume files under `paths.checkpoint_dir`.
Checkpoints include model and optimizer state plus scheduler, AMP scaler, RNG,
composed config, manifest hash, code version, epoch, and monitored metric
metadata. This is the current output contract for MetaUNETR and the other
segmentation pipelines. The CLI writes `metrics.json` and `provenance.json`
only after a pipeline returns a successful `FitResult`. `metrics.json` contains
best-checkpoint summary values, validation history, and held-out test metrics.
`provenance.json` contains code version, runtime, experiment, tracking mode, and
pipeline metadata including manifest hash, device, and canonical region
contract.

### Resume And Warm Start

`configs/run/debug.yaml` and `configs/run/full.yaml` default `run.resume` and
`run.warm_start` to `null`. Set either value to a checkpoint path, for example
`run.resume=outputs/cloud/mod_b/full/<date>/<time>/checkpoints/last.pt`.
`run.resume` performs an exact resume: it restores model, optimizer, scheduler,
AMP scaler, phase and training counters, best-metric tracking, history, and
random-number-generator/DataLoader state. It validates checkpoint schema and
compatibility metadata, including architecture/model configuration, manifest,
loss, optimizer, scheduler, monitor direction, and phase plan.

`run.warm_start` loads model weights only and starts fresh optimizer, scheduler,
AMP scaler, random-number-generator state, counters, and history. Setting
`run.resume_mode=warm_start` makes a configured `run.resume` path use these
warm-start semantics. Resolved `resume` and `warm_start` values are mutually
exclusive; configuring both is an error.

### CNN Pretraining

The `cnn_denoising_pretrain` experiment uses the seeded ImageFolder split and
the same run directory, but its current pipeline also exports an encoder
artifact:

```text
outputs/<runtime>/<experiment>/<run>/<YYYY-MM-DD>/<HH-MM-SS>/
|-- config.yaml
|-- metrics.json
|-- provenance.json
|-- encoder_best.pth
|-- checkpoints/
|   |-- best.pt
|   |-- last.pt
|   `-- <phase>_resume.pt
`-- cnn_reconstruction_grid.png  (optional)
```

`src/token_mixer/pipelines/pretrain_cnn.py` restores `checkpoints/best.pt`
before writing `encoder_best.pth`. The exported file contains CPU encoder
weights, source model configuration, best epoch, and validation loss. The
pipeline writes `cnn_reconstruction_grid.png` only when reconstruction-grid
visualization is enabled; it contains selected noisy, denoised, and clean
validation examples. Checkpointing must remain enabled for encoder export.

If a requested ImageNet weight transfer fails before training, the CLI writes a
failed-run `provenance.json` containing error and transfer status metadata, then
re-raises the error. It does not fabricate `metrics.json` or training metrics.
Hydra's `.hydra/` runtime metadata is ignored and is not part of the reviewed
run-artifact contract.

### Run Artifacts

Metrics, plots, and other generated analysis artifacts stay inside the run
directory; `notebooks/03_results_analysis.py` reads saved results from the
repository-relative `outputs/` tree when they are present. JSON artifacts use
atomic writes and represent non-finite metric values as JSON `null`.

W&B is disabled by default in both runtime profiles. If enabled, its configured
directory is `outputs/wandb`; offline run files and local W&B caches are ignored.
Online W&B mode requires credentials supplied through the environment, never
committed to this repository.

## Reproducibility

`configs/run/debug.yaml` and `configs/run/full.yaml` set seed `42` and
`deterministic: true`. `token_mixer.reproducibility.seed_everything` seeds
Python, NumPy, PyTorch, CUDA when available, and the DataLoader generator;
`seed_worker` handles worker-local Python and NumPy state. Deterministic mode
sets:

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

Keep the profile, experiment, run group, composed `config.yaml`, split manifest,
manifest hash, code version, and device with each result. Compare variants only
when they use the same prepared data, manifest, canonical label order, and
evaluation protocol. 3-D baselines use full-volume sliding-window evaluation
with configured physical spacing; TransUNet uses per-slice evaluation with unit
spacing. Their metric values, especially HD95 and aggregate scores, are not
automatically comparable across those protocols. Debug runs verify data flow
and tensor contracts; they do not establish segmentation quality or justify a
full training run.

Current verification snapshot: `uv run pytest -q` reports **397 passed, 6
skipped** (five symlink-capability skips and one external TransUNet integration
skip). No real-data or full-dataset training run has been performed, and the
external TransUNet checkout/pretrained asset validation remains unverified.

## Safe Data Handling

Raw medical images, generated case indexes, checkpoints, W&B files, Hydra
runtime metadata, and caches are ignored by `.gitignore`, but ignore rules are
not a security boundary. Keep raw data in approved access-controlled storage,
do not place PHI or credentials in Git, and verify `git status` before staging.
Share only reviewed manifests or derived results that are permitted for the
project. See [`data/README.md`](data/README.md) for the detailed data contract.
