# Token Mixer Placement Matters

Research pipeline for studying where token mixers belong in medical-image
segmentation models. Supported package includes paper-aligned MetaUNETR
experiments (`metaunetr_mamba`, `mod_a`, and `mod_b`), plus ResUNet3D,
SwinUNETR, TransUNet, and 2-D CNN denoising pretraining paths.

Run commands from repository root. Runtime data, checkpoints, W&B files, and
generated outputs stay outside reviewed source tree.

For maintainer/researcher architecture and implementation guides, see [codebase/README.md](codebase/README.md).

## Quickstart

### Prerequisites

Install Git, [`uv`](https://docs.astral.sh/uv/), and Python 3.12. Repository
pins Python `3.12` in `.python-version`.

```bash
git clone https://github.com/Abdulbasit4422/Token-Mixer-Placement-Matters.git token-mixer-placement-matters
cd token-mixer-placement-matters
uv python install 3.12
uv sync --extra imaging --extra research --extra dev
```

`imaging` installs MONAI, `einops`, nibabel, and SimpleITK. `research` installs
timm and `ml-collections` for optional preserved research paths. Jupytext and
ipykernel are base dependencies. `dev` adds JupyterLab,
collaboration/extension tooling, pytest, and pre-commit. Base package also
includes PyTorch, torchvision, Hydra, W&B, NumPy, and runtime dependencies.

Check installation before touching data:

```bash
uv run python --version
uv run python -c "import token_mixer; print(token_mixer.__version__)"
```

### Validation ladder

Keep checks separate. Config inspection is not a test; synthetic/debug
training is not a real-data experiment; neither is cloud/full training.

| Check | Command | What it does |
| --- | --- | --- |
| Config-only local composition | `uv run python -m token_mixer --config-name local experiment=mod_a run=debug --cfg job` | Composes and prints Hydra config. Does not train. |
| Config-only cloud dry run | `uv run python -m token_mixer --config-name cloud experiment=mod_b run=debug --cfg job` | Checks cloud paths, `device: cuda`, and debug overrides. Does not train. |
| Package tests | `uv run pytest -q` | Runs unit, model, pipeline, and integration tests. |
| Synthetic/debug harness | `uv run pytest tests/integration/test_synthetic_debug.py -q` | Creates temporary synthetic NIfTI/ImageFolder data and exercises forward, loss, backward, fit, checkpoints, metrics, and artifacts. No external dataset. |
| CPU local debug | `uv run python -m token_mixer --config-name local experiment=mod_a run=debug device=cpu` | Runs package debug training on prepared `data/local` cases. Requires valid BraTS manifest. |
| Syntax check | `uv run python -m compileall -q src tests` | Compiles package and test Python files. |

Local debug limits each segmentation split to two cases after manifest lookup,
uses zero DataLoader workers, and uses one epoch for each configured phase.
Default model widths can still be non-trivial on CPU. Synthetic harness is
first training-flow check when no real data exists.

Cloud/full training is intentionally separate and expensive. It requires
explicit approval for compute, data use, and expected runtime; the complete
command matrix appears below. Repository never starts these commands
automatically.

## Package architecture

Production code lives under `src/token_mixer/`. Hydra configuration lives under
`configs/`. Supported entry point is `token_mixer.cli`, exposed through
`src/token_mixer/__main__.py`:

```bash
uv run python -m token_mixer
```

| Package path | Responsibility |
| --- | --- |
| `token_mixer.data` | BraTS case discovery and preparation, preprocessing, labels, datasets, and deterministic split manifests |
| `token_mixer.models` | Tensor-only model constructors and adapters, including `models/metaunetr/` and optional external TransUNet integration |
| `token_mixer.training` | Shared phase-based training loop, checkpoint persistence, completion artifacts, reproducibility state, and optional W&B tracking |
| `token_mixer.evaluation` | Sliding-window inference, canonical Dice/HD95 metrics, and plots/overlays |
| `token_mixer.pipelines` | Thin orchestration boundaries composing data, models, training, and evaluation |
| `configs/` | Local/cloud profiles plus data, model, experiment, and debug/full run groups |

Model modules operate on tensors. They do not own dataset paths, checkpoint
files, W&B initialization, or plotting. Pipelines own those boundaries, while
shared engine in `token_mixer.training.engine` remains independent of any model
name.

Eight historical scripts are preserved under `archive/` and are not supported
entry points: `convert_to_nnunet.py`, `dataset.py`, `finetune_brats_mod_b.py`,
`finetune_nnunet_brats.py`, `finetune_transunet.py`, `main.py`,
`pretrain_cnn.py`, and `train_swinunetr_new.py`. Do not launch them for new
runs.

## Runtime environment variables

Package/runtime variables:

| Variable | Used when | Value |
| --- | --- | --- |
| `WANDB_API_KEY` | W&B `online` tracking | W&B API credential supplied by user or runtime secret store |
| `TRANSUNET_ROOT` | TransUNet CLI runs and notebook shape-check integration | Root of an official Beckschen/TransUNet checkout; required for actual TransUNet CLI runs |
| `TRANSUNET_PRETRAINED` | TransUNet CLI runs and notebook shape-check integration | Compatible official TransUNet pretrained `.npz` file; required for actual TransUNet CLI runs |

MCP-only variables are not needed for package training and are intentionally
omitted.

`TRANSUNET_ROOT` and `TRANSUNET_PRETRAINED` are required assets for every
non-injected TransUNet run, not optional runtime inputs. The package CLI does
not substitute these environment variables into Hydra configuration; export
them, then pass them as `third_party.transunet_root` and
`third_party.pretrained_path`.

Notebook-only variables are separate from package/runtime configuration and are
not read by the CLI:

| Variable | Notebook use | Default |
| --- | --- | --- |
| `TOKEN_MIXER_REPO_ROOT` | All notebooks: repository root override | Current working directory, with repository-root fallback |
| `TOKEN_MIXER_DATA_ROOT` | `notebooks/01_preprocessing_smoke.py`: local BraTS fixture root | `REPO_ROOT/data/local/brats` |
| `TOKEN_MIXER_RESULTS_ROOT` | `notebooks/03_results_analysis.py`: results scan root | `REPO_ROOT/outputs` |

Relative data/results overrides resolve from the notebook working directory.
These variables are intentionally not in `.env.example`.

### Safe `.env.example` usage

`.env.example` is a template, not a credential store. It contains empty values,
no machine paths, and no credentials. Copy it locally, edit it, and keep the
copy uncommitted:

```bash
cp .env.example .env
```

Package does **not** load `.env` automatically. Export values through your shell
or an approved dotenv tool before running a command. For Git Bash:

```bash
set -a
. ./.env
set +a
```

For PowerShell, set only values needed for current terminal, for example:

```powershell
$env:WANDB_API_KEY = "<your-wandb-key>"
$env:TRANSUNET_ROOT = "<transunet-checkout-root>"
$env:TRANSUNET_PRETRAINED = "<transunet-pretrained-npz>"
```

`.env` is ignored by Git. Check `git status` before staging anything.

The CLI does not substitute the two TransUNet values automatically. Pass them
into the exact Hydra config keys when dispatching TransUNet, for example:

```bash
uv run python -m token_mixer --config-name local experiment=transunet run=debug device=cpu third_party.transunet_root="$TRANSUNET_ROOT" third_party.pretrained_path="$TRANSUNET_PRETRAINED"
```

## Jupyter setup

JupyterLab is optional for package execution and is installed by the `dev`
extra. From Git Bash, use tracked `run-jup.sh.example` as local startup
template:

```bash
cp run-jup.sh.example run-jup.sh
# Edit run-jup.sh and replace token with a random local-only value, quoted:
# --IdentityProvider.token="<random-local-token>"
./run-jup.sh
```

Keep Jupyter terminal running and use separate terminal for package commands.
`run-jup.sh` is ignored and must never be committed.

The tracked script is Bash syntax and uses `.venv/Scripts/activate`; run it
from Git Bash or another compatible local Bash shell. PowerShell does not run
this script directly. After `uv sync --extra dev`, use `uv run` instead:

```powershell
uv run jupyter lab --port 8888 `
  --IdentityProvider.token="<random-local-token>" `
  --ip 127.0.0.1 `
  --ServerApp.allow_origin="*" `
  --ServerApp.disable_check_xsrf=True `
  --LabApp.collaborative=False
```

If using an activated PowerShell virtual environment, `jupyter lab` with the
same quoted options is equivalent. Replace the placeholder before starting.

Template binds Jupyter to `127.0.0.1`, but also allows all origins and disables
XSRF checks for local collaboration. Treat this as local-only development:
do not expose port, bind publicly, reuse token, or use it on untrusted network.

Notebook `.py` files are source of truth. Edit Python file first, then sync one
way into paired notebook:

```bash
uv run jupytext --sync notebooks/01_preprocessing_smoke.py
```

For multiple files, repeat same form:

```bash
uv run jupytext --sync notebooks/00_data_contract.py
uv run jupytext --sync notebooks/01_preprocessing_smoke.py
uv run jupytext --sync notebooks/02_model_shapes.py
uv run jupytext --sync notebooks/03_results_analysis.py
```

Never run `jupytext --sync` with an `.ipynb` path. Supported direction is
`.py` to `.ipynb` only.

## BraTS data boundary and preparation

Repository does not download or redistribute BraTS. Obtain data through an
approved source, follow access and licensing requirements, keep raw medical
images in access-controlled storage, and do not put PHI or raw data in Git.
Provider-specific transfer and cloud mounting are outside this repository.

### Accepted source layouts

`token_mixer.data.prepare.prepare_brats` accepts either source form below.

Nested case directories use recognizable modality and segmentation names:

```text
<source-root>/
|-- <case-id>/
    |-- <name containing t1n or legacy t1>.nii[.gz]
    |-- <name containing t1c, t1ce, or t1gd>.nii[.gz]
    |-- <name containing t2w or standalone t2>.nii[.gz]
    |-- <name containing t2f or flair>.nii[.gz]
    `-- <name containing seg or mask>.nii[.gz]
```

An nnU-Net-style source uses exact image channel suffixes and matching label:

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

Source detector requires both `imagesTr/` and `labelsTr/` when either is
present. Incomplete or ambiguous cases are rejected or warned about; NIfTI
files are loaded and saved during preparation for validation.

### Canonical runtime layout

Prepared data must use one of these profile roots:

- `data/local/brats` for local debug work
- `data/cloud/brats` for approved cloud/full work

Each root contains complete, non-empty 3-D NIfTI cases:

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

`token_mixer.data.cases.discover_cases(root)` expects `root/cases/` and returns
sorted complete cases in `[t1n, t1c, t2w, t2f]` order. Preparation preserves
raw segmentation voxel values and does not silently rewrite labels. Existing
canonical case destinations are refused unless library `overwrite=True` is
used explicitly.

Current `configs/local.yaml` and `configs/cloud.yaml` both point to the shared
manifest `data/manifests/brats_seed42.json`. Every case ID listed in that
manifest must exist below both `data/local/brats/cases/` and
`data/cloud/brats/cases/` when those profiles are used. `run.max_cases=2` is
applied only after manifest lookup and split selection; it does not make
missing manifest IDs valid. A tiny local fixture therefore must not generate a
manifest that is reused for a cloud/full run. A separate fixture requires a
separate manifest and profile override; no such profile is provided here.

### Prepare, discover, split, save

There is no `prepare` experiment dispatched by CLI. Preparation boundary is
library-only: `run_prepare` calls `prepare_brats` and writes compact
`case_index.json`; then persisted split is created explicitly. The following
Git Bash/POSIX-shell example takes user-supplied source root as final argument
and writes local canonical root. Use source data representing the reviewed case
universe for the shared manifest, not a two-case debug fixture:

`create_split_manifest` computes `int(case_count * fraction)`, so fractions are
floored. With the configured `val_fraction=0.15`, at least 7 cases are needed
for a non-empty validation split. With `test_fraction=0.10`, at least 10 cases
are needed for both non-empty validation and test splits. The example reports
split sizes and refuses to save the shared manifest when either required split
is empty.

```bash
uv run python - path/to/approved/brats-source <<'PY'
from pathlib import Path
import sys

from omegaconf import OmegaConf

from token_mixer.data.cases import discover_cases
from token_mixer.data.splits import create_split_manifest, save_split_manifest
from token_mixer.pipelines.prepare_data import run_prepare


source_root = Path(sys.argv[1]).expanduser()
data_root = Path("data/local/brats")
manifest_path = Path("data/manifests/brats_seed42.json")

run_prepare(
    OmegaConf.create(
        {
            "paths": {
                "source_root": str(source_root),
                "data_root": str(data_root),
            }
        }
    )
)
cases = discover_cases(data_root)
manifest = create_split_manifest(
    cases,
    seed=42,
    val_fraction=0.15,
    test_fraction=0.10,
    dataset_id="brats",
)
print(f"Prepared cases: {len(cases)}")
print(f"Split sizes: train={len(manifest.train)}, val={len(manifest.val)}, test={len(manifest.test)}")
if not manifest.val:
    raise ValueError("No validation cases; at least 7 prepared cases are needed")
if not manifest.test:
    raise ValueError(
        "No test cases; at least 10 prepared cases are needed for the configured fractions"
    )
save_split_manifest(manifest, manifest_path)
print(f"Saved manifest: {manifest_path}")
PY
```

Replace `path/to/approved/brats-source` with source root supplied by data
custodian. Command is intentionally not a downloader. To use another runtime
root, change `data_root` only when its case IDs match the reviewed manifest;
the same shared manifest must not describe a tiny fixture and then be reused
for full cloud data. Default profiles expect
`data/manifests/brats_seed42.json`.

Run clean-checkout preflight before segmentation training:

```bash
uv run python -c "import sys; from pathlib import Path; p=Path('data/manifests/brats_seed42.json'); sys.exit(f'Missing {p}; create it after preparing data') if not p.is_file() else print(f'Found {p}')"
```

Clean checkout normally has only `data/manifests/.gitkeep`, so missing manifest
is expected preflight failure until workflow completes. Every BraTS
segmentation experiment loads this one manifest instead of creating a private
split. The manifest is generated/reviewed data-specific metadata, not a
committed tiny fixture. Loader checks dataset ID, seed, validation/test
fractions, case IDs, and manifest hash, then requires every listed ID in the
selected runtime root.

### Labels and tensor contract

- Raw BraTS enhancing-tumor labels `3` and `4` are supported.
- `configs/data/brats.yaml` defaults to `data.et_label: 4`.
- Dataset whose enhancing-tumor voxels use label `3` requires explicit override
  `data.et_label=3`; otherwise ET labels are not selected.
- Canonical region order is exactly `[ET, TC, WT]`.
- 3-D inputs are `[B, 4, D, H, W]` in `[t1n, t1c, t2w, t2f]` order.
- 3-D models return three-channel raw logits `[B, 3, D, H, W]`; they do not
  apply sigmoid or thresholding.
- `token_mixer.evaluation.metrics.logits_to_regions` applies sigmoid and shared
  probability threshold.
- TransUNet is explicit 2-D slice adapter. Its external classes are
  `0 background, 1 ET, 2 TC, 3 WT`; it converts output to canonical raw region
  logits before shared metric handling.

`token_mixer.data.labels.detect_et_label` prefers label `4`, then label `3`,
and raises when neither is present. Dataset-level `data.et_label` remains
important for volumes with no enhancing-tumor voxels.

### Profiles

`configs/local.yaml` resolves repository-rooted paths to `data/local/brats`,
selects `runtime: local`, uses automatic device selection, and defaults to
`run: debug` with `max_cases: 2`. Force CPU with `device=cpu`.

`configs/cloud.yaml` resolves to `data/cloud/brats`, selects `runtime: cloud`,
requests `device: cuda`, and defaults to `run: full` with no case limit. Stage
or mount prepared data before launching it. Both profiles write to same
repository-relative `outputs/` tree and default W&B to disabled.

Debug run has `batch_size: 1`, zero workers, `phase1_epochs: 1`,
`phase2_epochs: 1`, and AMP disabled. Full run has `batch_size: 2`, eight
workers, `phase1_epochs: 20`, `phase2_epochs: 80`, and AMP enabled. CNN-only
`run.epochs` is `1` in debug and `30` in full.

## Run commands

Run commands from repository root. Local commands below use CPU debug settings;
they still require prepared inputs. Segmentation commands require the shared
manifest and all case IDs it names. CNN commands require ImageFolder data.
TransUNet commands additionally require exported `TRANSUNET_ROOT` and
`TRANSUNET_PRETRAINED` values as described above.

### Local debug commands (CPU)

```bash
uv run python -m token_mixer --config-name local experiment=cnn_denoising_pretrain run=debug device=cpu
uv run python -m token_mixer --config-name local experiment=metaunetr_mamba run=debug device=cpu
uv run python -m token_mixer --config-name local experiment=mod_a run=debug device=cpu
uv run python -m token_mixer --config-name local experiment=mod_b run=debug device=cpu
uv run python -m token_mixer --config-name local experiment=resunet3d run=debug device=cpu
uv run python -m token_mixer --config-name local experiment=swinunetr run=debug device=cpu
uv run python -m token_mixer --config-name local experiment=transunet run=debug device=cpu third_party.transunet_root="$TRANSUNET_ROOT" third_party.pretrained_path="$TRANSUNET_PRETRAINED"
```

### Cloud/full commands — EXPENSIVE; do not run without approval

These commands select `device: cuda`, unlimited cases, and full epoch plans.
Stage or mount complete cloud data before any approved launch.

```bash
uv run python -m token_mixer --config-name cloud experiment=cnn_denoising_pretrain run=full
uv run python -m token_mixer --config-name cloud experiment=metaunetr_mamba run=full
uv run python -m token_mixer --config-name cloud experiment=mod_a run=full
uv run python -m token_mixer --config-name cloud experiment=mod_b run=full
uv run python -m token_mixer --config-name cloud experiment=resunet3d run=full
uv run python -m token_mixer --config-name cloud experiment=swinunetr run=full
uv run python -m token_mixer --config-name cloud experiment=transunet run=full third_party.transunet_root="$TRANSUNET_ROOT" third_party.pretrained_path="$TRANSUNET_PRETRAINED"
```

## Supported experiments and dispatch

CLI dispatch supports exactly these seven `experiment.name` values:

| Experiment / exact CLI selector | Purpose | Dimensionality | Dependencies and required assets | Default monitor, loss, and phase notes | Dispatched runner |
| --- | --- | --- | --- | --- | --- |
| `experiment=cnn_denoising_pretrain` | 2-D denoising autoencoder pretraining for encoder artifact | 2-D | Base torchvision/PIL `ImageFolder`; `data/<runtime>/imagenet`; no BraTS or BraTS manifest | `monitor=mse`, minimize; `loss=mse`; one `pretrain` phase for `run.epochs` (`1` debug, `30` full), encoder trainable | `run_cnn_denoising_pretrain` |
| `experiment=metaunetr_mamba` | Paper baseline: Mamba at bottleneck, CNN decoder | 3-D | `imaging` extra, prepared BraTS, shared manifest; CPU fallback Mamba, optional CUDA `mamba_ssm` | `monitor=mean_dice`, maximize; `binary_cross_entropy_with_logits`; `encoder_frozen` for `run.phase1_epochs`, then `full_finetune` for `run.phase2_epochs` (`1+1` debug, `20+80` full) | `run_metaunetr` |
| `experiment=mod_a` | Encoder-Mamba ablation with shared CNN decoder | 3-D | Same MONAI/nibabel, BraTS, and manifest requirements as baseline | `monitor=mean_dice`, maximize; `binary_cross_entropy_with_logits`; `encoder_frozen` then `full_finetune` (`1+1` debug, `20+80` full) | `run_metaunetr` |
| `experiment=mod_b` | CNN encoder with Mamba in coarse decoder refinement blocks | 3-D | Same MONAI/nibabel, BraTS, and manifest requirements as baseline | `monitor=mean_dice`, maximize; `binary_cross_entropy_with_logits`; `encoder_frozen` then `full_finetune` (`1+1` debug, `20+80` full) | `run_metaunetr` |
| `experiment=resunet3d` | Custom five-stage residual U-Net baseline | 3-D | `imaging` extra, prepared BraTS, shared manifest; optional `research`/timm ResNet-18 transfer | `monitor=mean_dice`, maximize; `binary_cross_entropy_with_logits`; `encoder_frozen` then `full_finetune` (`1+1` debug, `20+80` full) | `run_resunet3d` |
| `experiment=swinunetr` | MONAI Swin UNETR baseline | 3-D | `imaging` extra, including MONAI and `einops`, prepared BraTS, shared manifest | `monitor=mean_dice`, maximize; `dice_ce`; one `train` phase for `run.phase2_epochs` (`1` debug, `80` full) | `run_swinunetr` |
| `experiment=transunet` | Official TransUNet R50-ViT-B/16 slice adapter | 2-D slices | `imaging` for BraTS NIfTI plus official external checkout, its own dependencies, and compatible `.npz`; `TRANSUNET_ROOT` and `TRANSUNET_PRETRAINED` are required and passed to `third_party.*` keys | `monitor=mean_dice`, maximize; `binary_cross_entropy_with_logits`; one `train` phase for `run.phase2_epochs` (`1` debug, `80` full) | `run_transunet` |

Model and data group selection is test-covered in `tests/test_cli_config.py`.
Unknown name raises `ValueError`; preparation is not dispatched.

### Model/reference notes

MetaUNETR configs use four MRI inputs, three raw-logit outputs, base width
`48`, depths `[2, 2, 2, 2]`, and `axis_fusion: sum`. `model.axis_fusion=cat`
selects reference-compatible concatenation/projection alternative. Manuscript
describes summed axis outputs while public MetaUNETR fragment uses
concatenation followed by projection; selected mode is recorded in metadata, so
no exact-reproduction claim is made without composed config.

Preserved architecture sources are documented in model modules:

- Project manuscript: Oyetunji et al., *Token Mixer Placement Matters: A
  Systematic Encoder–Decoder Ablation Study of Mamba for Brain Tumour
  Segmentation on BraTS-Africa*.
- MetaUNETR: Lyu et al., *MetaUNETR: Rethinking Token Mixer Encoding for
  Efficient Multi-Organ Segmentation*, [MICCAI 2024](https://papers.miccai.org/miccai-2024/paper/2749_paper.pdf),
  with the [official implementation](https://github.com/lyupengju/MetaUNETR).
- Mamba: Gu and Dao, [Mamba: Linear-Time Sequence Modeling with Selective
  State Spaces](https://arxiv.org/abs/2312.00752), with the [official
  implementation](https://github.com/state-spaces/mamba).
- SwinUNETR: [paper](https://arxiv.org/abs/2201.01266) and [MONAI
  implementation](https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/swin_unetr.py).
- TransUNet: Chen et al., [TransUNet: Transformers Make Strong Encoders for
  Medical Image Segmentation](https://arxiv.org/abs/2102.04306), with the
  [official implementation](https://github.com/Beckschen/TransUNet).
- ResUNet3D: preserved nnU-Net-style residual path, cross-checked against
  [nnU-Net](https://arxiv.org/abs/1809.10486) and its [reference
  implementation](https://github.com/MIC-DKFZ/nnUNet).

References ground preserved/adapted architectures; this README makes no claim
about paper metrics or exact weight reproduction.

## CNN/ImageFolder pretraining

CNN experiment is separate from BraTS segmentation.
`configs/data/imagenet.yaml` resolves ImageFolder root to
`data/<runtime>/imagenet`:

```text
data/local/imagenet/
|-- class-a/
|   |-- image-001.png
|   `-- image-003.png
`-- class-b/
    `-- image-002.png
```

Use class directories containing image files. CNN debug requires at least 2
images: local debug limits `run.max_cases` to 2, uses batch size 1, and the
training loader has `drop_last=true`. Default full config requires at least 3
images: it uses batch size 2, holds out 5% (at least one image), and also drops
an incomplete training batch. Model input is three channels by default, and
denoising adapter supports one or three channels. Pretraining uses seeded
ImageFolder split and does not consume `data/manifests/brats_seed42.json`.

Run local CPU debug after populating `data/local/imagenet`:

```bash
uv run python -m token_mixer --config-name local experiment=cnn_denoising_pretrain run=debug device=cpu
```

Enable optional noisy/denoised/clean validation visualization with the
explicit add-key override (the `visualization` section is absent by default):

```bash
uv run python -m token_mixer --config-name local experiment=cnn_denoising_pretrain run=debug device=cpu +visualization.reconstruction_grid=true
```

CNN `metrics.json` records `"test_metrics": null`: this pipeline has train and
validation loaders only. Segmentation pipelines separately evaluate their
held-out test split and write those test metrics.

Successful CNN runs export `encoder_best.pth` after restoring
`checkpoints/best.pt`. Packaged ResUNet3D transfer path currently loads timm
ResNet-18 through its own helper; it does not automatically read CNN
`encoder_best.pth` artifact.

### ResUNet3D ImageNet transfer flags

Transfer is disabled by default so local/debug runs remain offline. Exact model
config keys:

- `model.imagenet_transfer.enabled`
- `model.imagenet_transfer.download`
- `model.imagenet_transfer.cache_dir`
- `model.imagenet_transfer.source`

`source` records provenance metadata; it does not inject a source model.

For explicitly approved timm download, install research extra and use a
repository-relative cache path:

```bash
uv run python -m token_mixer --config-name local experiment=resunet3d run=debug device=cpu model.imagenet_transfer.enabled=true model.imagenet_transfer.download=true model.imagenet_transfer.cache_dir=data/local/timm-cache
```

`download=true` permits loading timm `resnet18.a1_in1k` pretrained asset. This
can use network access and is separate from full training approval. If requested
transfer fails before training, inspect failed-run provenance below.

## W&B tracking

Both runtime profiles default to `tracking.enabled=false` and
`tracking.mode=disabled`; no W&B run is initialized unless tracking is enabled.
Local metrics and provenance are still written after successful training. W&B
supports three modes.

Offline, no credential required:

```bash
uv run python -m token_mixer --config-name local experiment=mod_a run=debug device=cpu tracking.enabled=true tracking.mode=offline
```

Online, credential required:

```bash
export WANDB_API_KEY="<your-wandb-key>"
uv run python -m token_mixer --config-name local experiment=mod_a run=debug device=cpu tracking.enabled=true tracking.mode=online
```

PowerShell equivalent for credential:

```powershell
$env:WANDB_API_KEY = "<your-wandb-key>"
```

Tracker refuses online mode without `WANDB_API_KEY`. Offline files and local
W&B caches are ignored under `outputs/wandb` and `wandb/`. The profile fields
`tracking.log_every_steps`, `tracking.log_images`, and
`tracking.log_checkpoints` are currently unused by the tracker; changing them
does not enable step-frequency, image, or checkpoint logging. Do not commit
credentials or W&B runtime files.

## Outputs, checkpoints, and provenance

Hydra writes each run below configured experiment output:

```text
outputs/
`-- <runtime>/<experiment>/<run>/<YYYY-MM-DD>/<HH-MM-SS>/
    |-- config.yaml
    |-- metrics.json
    |-- provenance.json
    |-- checkpoints/
    |   |-- best.pt
    |   |-- last.pt
    |   `-- <phase>_resume.pt
    |-- encoder_best.pth                 (CNN pretraining only)
    `-- cnn_reconstruction_grid.png     (optional CNN visualization)
```

`config.yaml` is composed Hydra configuration saved before dispatch.
`metrics.json` is written after pipeline returns successful `FitResult` and
contains best-checkpoint values and validation history. Segmentation pipelines
also write held-out test metrics; CNN pretraining has no test loader and writes
`"test_metrics": null`.
`provenance.json` records code version, runtime, experiment, model metadata,
seed, device, tracking mode, manifest hash, monitor direction, and source
checkpoint where applicable. Non-finite JSON metric values become `null`.

### Checkpoint semantics

- `best.pt` is replaced only when monitored validation metric improves; Dice
  maximizes, while CNN MSE minimizes.
- `last.pt` records latest epoch state.
- `<phase>_resume.pt` records completed phase state.
- Checkpoints contain model, optimizer, scheduler, AMP scaler, Python/NumPy/
  PyTorch/CUDA RNG, DataLoader generator, composed config, manifest hash, code
  version, phase/counter state, history, and compatibility metadata.
- Segmentation pipelines reload `best.pt` before held-out test evaluation.
- CNN pretraining reloads `best.pt` before exporting CPU encoder weights to
  `encoder_best.pth`; checkpointing cannot be disabled for this experiment.
- `cnn_reconstruction_grid.png` is written only when reconstruction-grid
  visualization is explicitly enabled.

### Exact resume versus warm start

Both `run.resume` and `run.warm_start` default to `null` and are mutually
exclusive. Replace bracketed path segments below with existing
repository-relative run path.

Exact resume restores model, optimizer, scheduler, AMP scaler, phase and epoch
counters, best-metric tracking, history, RNG state, and DataLoader state. It
validates architecture, model config, manifest, loss, optimizer, scheduler,
monitor direction, and phase compatibility:

```bash
uv run python -m token_mixer --config-name local experiment=mod_a run=debug run.resume=outputs/local/mod_a/debug/<YYYY-MM-DD>/<HH-MM-SS>/checkpoints/last.pt
```

Warm start loads model weights only and starts fresh optimizer, scheduler, AMP
scaler, RNG state, counters, and history:

```bash
uv run python -m token_mixer --config-name local experiment=mod_a run=debug run.warm_start=outputs/local/mod_a/debug/<YYYY-MM-DD>/<HH-MM-SS>/checkpoints/best.pt
```

Equivalent warm-start mode using `run.resume` path. `resume_mode` is not a
field in the shipped run configs, so Hydra requires the explicit add-key
operator:

```bash
uv run python -m token_mixer --config-name local experiment=mod_a run=debug +run.resume_mode=warm_start run.resume=outputs/local/mod_a/debug/<YYYY-MM-DD>/<HH-MM-SS>/checkpoints/last.pt
```

Each resumed invocation gets a new timestamped output directory. For exact
resume, source run's `best.pt` seeds destination `best.pt` before fitting, even
when `run.resume` points to `last.pt` or a phase-resume checkpoint. If later
destination validation improves, destination `best.pt` is replaced and that
improvement is retained. Source checkpoint and source run are not mutated.
Warm start initializes model weights only and starts fresh best-metric tracking.

### Failed transfer provenance

When requested ResUNet3D ImageNet transfer fails before fit, CLI preserves
`config.yaml` and writes failed `provenance.json` containing `status: failed`,
error, transfer source/status/counts, model metadata, and tracking context. It
does not fabricate `metrics.json` or training metrics. This artifact is
specific to transfer failure path; missing TransUNet assets produce validation
error and do not constitute validated external integration.

## Local metrics and results analysis

Run headless notebook smoke checks without starting Jupyter:

```bash
uv run python notebooks/00_data_contract.py
uv run python notebooks/01_preprocessing_smoke.py
uv run python notebooks/02_model_shapes.py
```

`01_preprocessing_smoke.py` reports expected local layout and exits normally
when no fixture exists. `02_model_shapes.py` uses tiny CPU configs;
MONAI/SwinUNETR and external TransUNet checks are optional skips when assets are
unavailable.

Analyze saved local results with:

```bash
uv run python notebooks/03_results_analysis.py
```

Analysis notebook scans repository-relative `outputs/` for saved JSON/CSV files
whose names contain `metric`, normalizes common payload layouts, prints a
source-level table, and groups numeric values by recognized experiment name. It
does not train, query W&B, download data, or overwrite local metric artifacts.
With no completed runs it reports no saved metric files.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `Missing data/manifests/brats_seed42.json` | Acquire approved BraTS data, run preparation/split example, inspect case IDs, then rerun preflight. |
| Cases are not discovered | Check `data/<runtime>/brats/cases/<case-id>/`, all five filenames, non-zero sizes, and valid 3-D NIfTI content. |
| ET is empty for label-3 data | Add `data.et_label=3` to training command. Preparation preserves raw labels. |
| Manifest/data mismatch | Ensure selected local/cloud root contains every manifest case ID; regenerate and review metadata for dataset ID `brats`, seed `42`, validation fraction `0.15`, and test fraction `0.10`. Do not reuse a tiny fixture manifest for cloud/full data. |
| MONAI, nibabel, or `einops` import error | Run `uv sync --extra imaging --extra research --extra dev`; use imaging extra for 3-D and NIfTI paths. |
| ImageFolder root error | Create `data/<runtime>/imagenet/<class-name>/image-file` directories; provide at least 2 images for debug or 3 for default full because CNN training uses `drop_last=true`. |
| TransUNet validation error | Set required `TRANSUNET_ROOT` and `TRANSUNET_PRETRAINED`, pass them to `third_party.transunet_root` and `third_party.pretrained_path`, and verify checkout has `networks/vit_seg_modeling.py`, `vit_seg_modeling_resnet_skip.py`, and `vit_seg_configs.py`. |
| CUDA unavailable | Use local profile with `device=cpu` for development, or provision approved CUDA GPU before cloud profile. |
| ResUNet transfer fails | Install `--extra research`; set both `model.imagenet_transfer.enabled=true` and `model.imagenet_transfer.download=true` for CLI timm loading, or inspect failed `provenance.json`. |
| Resume compatibility error | Resume with same experiment, model, manifest, loss, optimizer, scheduler, monitor direction, and compatible phase plan. Use `run.warm_start` for intentional model-only initialization. |
| Online W&B error | Export `WANDB_API_KEY`, or use `tracking.mode=offline`/disabled. |
| Jupyter script does not start | Git Bash: copy example, replace quoted token, and run `./run-jup.sh`; PowerShell: use the documented `uv run jupyter lab` command instead. |
| `metrics.json` is absent | Pipeline did not return successful `FitResult`; inspect `config.yaml`, checkpoints, and any failed `provenance.json`. |

## Reproducibility and safe handling

Debug and full run configs use seed `42` and `deterministic: true`.
`token_mixer.reproducibility.seed_everything` seeds Python, NumPy, PyTorch,
CUDA when available, and DataLoader generator. `seed_worker` handles worker-
local Python and NumPy state. Deterministic mode sets:

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

Compare variants only when they use same prepared data, manifest, case
identity, canonical label order, and evaluation protocol. 3-D baselines use
full-volume sliding-window evaluation with configured physical spacing.
TransUNet uses per-slice evaluation with unit spacing; its metrics, especially
HD95 and aggregate scores, are not automatically comparable to 3-D protocols.
Debug runs verify data flow and tensor contracts; they do not establish
segmentation quality or justify a full training run.

Keep raw images, PHI, credentials, provider tokens, checkpoints, W&B files,
Hydra runtime metadata, and caches out of Git. Ignore rules reduce accidental
staging but are not security boundary. Inspect generated manifests before
sharing, use approved access controls, and verify `git status` before staging.
See [`data/README.md`](data/README.md) for detailed data contract.

## Current verification snapshot

Current repository verification reports:

```text
uv run pytest -q: 403 passed, 6 skipped
```

Six skips are environment-gated: five symlink-capability skips and one external
TransUNet integration skip. This snapshot does **not** claim a real BraTS-data
run, full-dataset/cloud training, or validation of an external TransUNet
checkout and pretrained asset. No such run is launched by README commands
unless user explicitly executes it after approval.
