# Configuration Flow

[Configuration-flow diagram](../assets/codebase/config-flow.svg)

This guide documents active Hydra composition and the configuration boundary
used by the Token Mixer package. The YAML files under `configs/` define defaults;
the CLI composes them and dispatches a selected experiment. Source code and
tests remain the authority when this prose disagrees with behavior.

## Source Map

- [`src/token_mixer/cli.py::main`](../src/token_mixer/cli.py#L103-L105)
- [`src/token_mixer/cli.py::_dispatch`](../src/token_mixer/cli.py#L39-L65)
- [`src/token_mixer/cli.py::_run`](../src/token_mixer/cli.py#L68-L101)
- [`src/token_mixer/cli.py::_save_composed_config`](../src/token_mixer/cli.py#L27-L30)
- [`src/token_mixer/reproducibility.py::seed_everything`](../src/token_mixer/reproducibility.py#L9-L20)
- [`src/token_mixer/pipelines/prepare_data.py::run_prepare`](../src/token_mixer/pipelines/prepare_data.py#L13-L50)
- [`src/token_mixer/pipelines/_baseline_common.py::_manifest_path`](../src/token_mixer/pipelines/_baseline_common.py#L145-L158)
- [`src/token_mixer/pipelines/_baseline_common.py::_validate_manifest_configuration`](../src/token_mixer/pipelines/_baseline_common.py#L238-L263)
- [`src/token_mixer/pipelines/_baseline_common.py::_limit_cases`](../src/token_mixer/pipelines/_baseline_common.py#L111-L135)
- [`src/token_mixer/pipelines/_baseline_common.py::_tracking_config`](../src/token_mixer/pipelines/_baseline_common.py#L1027-L1029)
- [`configs/local.yaml::defaults`](../configs/local.yaml#L1-L4)
- [`configs/cloud.yaml::defaults`](../configs/cloud.yaml#L1-L4)
- [`configs/data/brats.yaml::modalities`](../configs/data/brats.yaml#L1-L22)
- [`configs/data/imagenet.yaml::image_root`](../configs/data/imagenet.yaml#L1-L8)
- [`configs/run/debug.yaml::name`](../configs/run/debug.yaml#L1-L17)
- [`configs/run/full.yaml::name`](../configs/run/full.yaml#L1-L17)

## Composition

Hydra enters through `cli.py::main` with `configs/` as its configuration path
and `local` as its default config name. The root profile composes one experiment
group, one run group, and `_self_`:

```yaml
defaults:
  - experiment: mod_a
  - run: debug
  - _self_
```

`configs/local.yaml` selects `experiment: mod_a` and `run: debug`. The cloud
profile selects the same default experiment with `run: full`. Each experiment
file then composes one data group and one model group into the global `data` and
`model` keys before applying its own training settings:

```yaml
defaults:
  - /data@_global_.data: brats
  - /model@_global_.model: metaunetr
  - _self_
```

The `_self_` entries make the file's local values part of the composition. Run
values are referenced through OmegaConf interpolations such as
`${run.batch_size}`, `${run.phase1_epochs}`, and `${run.use_amp}`. This keeps
debug/full settings centralized while allowing an experiment to choose its loss,
phases, optimizer, scheduler, and checkpoint metric.

The CLI requires `experiment.name`. `_dispatch` maps that name to a runner; an
unknown name raises `ValueError` before training. Data preparation is not an
experiment selector. Its callable is
`src/token_mixer/pipelines/prepare_data.py::run_prepare`, which expects a caller
to provide `paths.source_root` and `paths.data_root`; the shipped local/cloud
profiles define `paths.data_root` but do not define `paths.source_root`.

## Selecting A Run

The package entrypoint is:

```bash
uv run python -m token_mixer [hydra overrides]
```

Useful composition examples:

```bash
uv run python -m token_mixer --config-name local experiment=mod_a run=debug
uv run python -m token_mixer --config-name cloud experiment=mod_b run=full
uv run python -m token_mixer --config-name local experiment=cnn_denoising_pretrain run=debug
```

Hydra overrides are written as `key=value` and apply after the selected group
values. For example, `data.et_label=3` selects the alternate BraTS ET marker,
and `run.max_cases=2` limits a debug run after manifest validation and split
mapping. Use `--cfg job` to print the composed job without entering `_run`:

```bash
uv run python -m token_mixer --config-name local experiment=mod_a run=debug --cfg job
```

This is a configuration preflight, not a data preflight. Loader construction
still checks the manifest and case root when a training runner executes.

## Profiles And Paths

The active top-level profiles are:

| Profile | Runtime | Device | Default run | BraTS root | ImageFolder root |
| --- | --- | --- | --- | --- | --- |
| `local` | `local` | `auto` | `debug` | `data/local/brats` | `data/local/imagenet` |
| `cloud` | `cloud` | `cuda` | `full` | `data/cloud/brats` | `data/cloud/imagenet` |

The roots are resolved below `${hydra:runtime.cwd}`. `paths.image_root` aliases
`${data.image_root}`. The BraTS data group leaves `data.image_root: null` because
segmentation loaders use `paths.data_root`; the ImageNet group supplies an
ImageFolder root under `data/${runtime}/imagenet`.

Both profiles currently point `paths.manifest` to the repository-relative
manifest name `data/manifests/brats_seed42.json`. The manifest is shared only
when its case IDs and metadata match the selected data root. A debug limit does
not repair a missing or mismatched manifest.

Run outputs resolve to:

```text
outputs/<runtime>/<experiment.name>/<run.name>/<date>/<time>/
`-- checkpoints/
```

The exact expression is
`${paths.output_root}/${runtime}/${experiment.name}/${run.name}/${now:%Y-%m-%d}/${now:%H-%M-%S}`.
Hydra sets this as its run directory and keeps `hydra.job.chdir: false`, so the
process working directory remains the repository working directory. `_run`
writes the unresolved composed configuration to `config.yaml` in that output
directory before dispatching. A successful `FitResult` then receives the normal
run-artifact writer; a runner exception is re-raised after the optional failed
transfer-provenance artifact path is attempted.

Tracking is disabled in both shipped profiles:

```yaml
tracking:
  enabled: false
  mode: disabled
```

The configured project name, optional entity/run name, logging interval, image
and checkpoint flags, and `${paths.output_root}/wandb` directory are inert until
tracking is explicitly enabled. Do not put credentials in YAML.

## Data Groups

### BraTS

`configs/data/brats.yaml` defines the 3-D segmentation contract:

| Key | Value |
| --- | --- |
| `name`, `dataset_id` | `brats`, `brats` |
| `modalities` | `[t1n, t1c, t2w, t2f]` |
| `region_names` | `[ET, TC, WT]` |
| `et_label` | `4` |
| `spacing` | `[1.0, 1.0, 1.0]` |
| `patch_size`, `roi_size` | `[96, 96, 96]` |
| `split_seed` | `42` |
| `val_fraction`, `test_fraction` | `0.15`, `0.10` |
| `normalize` | `true` |
| `flip_axes`, `flip_probability` | `[0, 1, 2]`, `0.5` |

The `et_label` value is an explicit convention. Override it to `3` for a
dataset whose enhancing-tumor marker is three. Label conversion and the fixed
modality/region orders are documented in [DATA.md](DATA.md).

### ImageNet

`configs/data/imagenet.yaml` defines the separate CNN pretraining input:

| Key | Value |
| --- | --- |
| `name`, `dataset_id` | `imagenet`, `imagenet` |
| `image_root` | `data/${runtime}/imagenet` |
| `root` | `${data.image_root}` |
| `image_size` | `96` |
| `in_channels` | `3` |
| `noise_std` | `0.15` |
| `val_fraction` | `0.05` |

This group selects an ImageFolder/denoising path; it does not reuse the BraTS
case manifest or four-channel volume contract.

## Experiment Selectors

The selector-to-group-to-runner contract is defined by these active experiment
groups: [`configs/experiment/cnn_denoising_pretrain.yaml`](../configs/experiment/cnn_denoising_pretrain.yaml#L1-L33),
[`configs/experiment/metaunetr_mamba.yaml`](../configs/experiment/metaunetr_mamba.yaml#L1-L38),
[`configs/experiment/mod_a.yaml`](../configs/experiment/mod_a.yaml#L1-L38),
[`configs/experiment/mod_b.yaml`](../configs/experiment/mod_b.yaml#L1-L38),
[`configs/experiment/resunet3d.yaml`](../configs/experiment/resunet3d.yaml#L1-L37),
[`configs/experiment/swinunetr.yaml`](../configs/experiment/swinunetr.yaml#L1-L32),
and [`configs/experiment/transunet.yaml`](../configs/experiment/transunet.yaml#L1-L32).
The resulting selector-to-group-to-runner contract is:

| Selector | Data group | Model group | Runner |
| --- | --- | --- | --- |
| `cnn_denoising_pretrain` | `imagenet` | `cnn_pretrain` | `pretrain_cnn::run_cnn_denoising_pretrain` |
| `metaunetr_mamba` | `brats` | `metaunetr` | `train_metaunetr::run_metaunetr` |
| `mod_a` | `brats` | `metaunetr` | `train_metaunetr::run_metaunetr` |
| `mod_b` | `brats` | `metaunetr` | `train_metaunetr::run_metaunetr` |
| `resunet3d` | `brats` | `resunet3d` | `train_resunet3d::run_resunet3d` |
| `swinunetr` | `brats` | `swinunetr` | `train_swinunetr::run_swinunetr` |
| `transunet` | `brats` | `transunet` | `train_transunet::run_transunet` |

The three MetaUNETR selectors share the model/data groups but differ through
their experiment `name` and `variant` values: `metaunetr_mamba`, `mod_a`, or
`mod_b`. The shared training phase shape is encoder-frozen first, then full
fine-tuning. SwinUNETR and TransUNet use one `train` phase. CNN pretraining uses
one `pretrain` phase and sets `training.drop_last: true`; segmentation experiment
files set it to `false`.

## Model Groups

Model groups define architecture-specific dimensions and options. The table is
grounded in these active YAML files: [`configs/model/cnn_pretrain.yaml`](../configs/model/cnn_pretrain.yaml#L1-L8),
[`configs/model/metaunetr.yaml`](../configs/model/metaunetr.yaml#L1-L16),
[`configs/model/resunet3d.yaml`](../configs/model/resunet3d.yaml#L1-L16),
[`configs/model/swinunetr.yaml`](../configs/model/swinunetr.yaml#L1-L15),
and [`configs/model/transunet.yaml`](../configs/model/transunet.yaml#L1-L12).

| Group | Contract highlights |
| --- | --- |
| `metaunetr` | `MetaUNETR`, `in_channels: 4`, `num_classes: 3`, base channels `48`, depths `[2,2,2,2]`, window `7`, heads `[3,6,12,24]`, Mamba state `16`, convolution `4`, expansion `2`, sum axis fusion |
| `resunet3d` | `ResUNet3D`, four input channels, three output channels, base features `32`, five depth entries, instance normalization; ImageNet transfer disabled by default |
| `swinunetr` | `SwinUNETR`, four input channels, three output channels, feature size `48`, `spatial_dims: 3`, patch size `2`, checkpointing enabled, v2 disabled |
| `transunet` | `TransUNet`, four input channels, image size `[224,224]`, `R50-ViT-B_16`, `n_skip: 3`, external class count `4`, canonical class count `3`, logits output |
| `cnn_pretrain` | Denoising autoencoder, three input channels, feature size `32`, depths `[1,1,1,1]`, image size `96` |

The TransUNet model requires an external checkout root at
`third_party.transunet_root` when the real external model is built. Its
pretrained file may be supplied through `third_party.pretrained_path` or the
model-level `pretrained_path`; both default to `null`. An injected external
model is handled by [`src/token_mixer/pipelines/train_transunet.py::_build_model`](../src/token_mixer/pipelines/train_transunet.py#L64-L68), which passes it to
[`src/token_mixer/models/transunet.py::build_transunet`](../src/token_mixer/models/transunet.py#L777-L844). That injected path bypasses external
checkout and file loading, so tests and controlled integration need not document
a machine path.

## Run Groups

`run/debug.yaml` and `run/full.yaml` centralize execution scale and
reproducibility settings:

| Setting | `debug` | `full` |
| --- | ---: | ---: |
| `max_cases` | `2` | `null` |
| `batch_size` | `1` | `2` |
| `num_workers`, `workers` | `0` | `8` |
| `pin_memory` | `false` | `true` |
| `persistent_workers` | `false` | `true` |
| `drop_last` | `false` | `false` |
| `epochs` | `1` | `30` |
| `phase1_epochs` | `1` | `20` |
| `phase2_epochs` | `1` | `80` |
| `validation_interval` | `1` | `5` |
| `use_amp` | `false` | `true` |
| `seed` | `42` | `42` |
| `deterministic` | `true` | `true` |
| `resume`, `warm_start` | `null` | `null` |

Training experiment files interpolate the run values into their `training`
sections. Their explicit `training.drop_last` value is consumed before the run
level fallback, so CNN pretraining remains `true` while segmentation remains
`false`. `seed_everything` receives the selected `run.seed` and deterministic
flag before a runner builds its model and loaders.

`run.max_cases` is applied after a pipeline loads and validates the manifest,
checks configured split metadata, maps IDs to discovered cases, and creates
split-specific case lists. It must not be used to conceal missing case IDs.

## Training Settings

Experiment groups own objective and phase settings; run groups own scale. The
active experiment values are:

| Experiment family | Loss | Optimizer | Scheduler | Phase behavior | Selection metric |
| --- | --- | --- | --- | --- | --- |
| MetaUNETR, ResUNet3D, TransUNet | `binary_cross_entropy_with_logits` | AdamW, weight decay `1e-4` | Cosine, epoch interval | Frozen encoder then full fine-tune, except TransUNet one train phase | Maximize `mean_dice` |
| SwinUNETR | `dice_ce` | AdamW, weight decay `1e-4` | Cosine, epoch interval | One train phase | Maximize `mean_dice` |
| CNN pretraining | `mse` | AdamW, weight decay `0.05` | Cosine, update interval, `eta_min: 1e-6` | One pretrain phase | Minimize `mse` |

The phase epoch values interpolate `run.phase1_epochs`, `run.phase2_epochs`, or
`run.epochs`. Encoder/decoder learning rates and freeze flags live in the
experiment YAML, not in the profile.

Resume and warm-start paths default to `null`. The shared baseline path helpers
reject both being configured at once. Exact resume and warm start have different
checkpoint semantics; configure one deliberately and document its source
checkpoint in the resulting run metadata.

## Preflight And Failures

Start with composition-only inspection:

```bash
uv run python -m token_mixer --config-name local experiment=mod_a run=debug --cfg job
uv run python -m token_mixer --config-name cloud experiment=cnn_denoising_pretrain run=debug --cfg job
```

Then inspect the resolved values that matter for the selected runner:

- `experiment.name`, `experiment.data_name`, and `experiment.model_name`.
- `paths.data_root`, `paths.manifest`, and `data.dataset_id`.
- `data.modalities`, `data.region_names`, `data.et_label`, and spatial sizes.
- `run.seed`, `run.deterministic`, `run.max_cases`, and loader settings.
- `third_party.transunet_root` and pretrained settings when selecting TransUNet
  or explicit ImageNet transfer.

Common configuration-boundary failures:

| Condition | Behavior |
| --- | --- |
| Missing or unknown `experiment.name` | `_experiment_name`/`_dispatch` raises `ValueError` |
| Missing `paths.data_root` or `paths.manifest` | BraTS loader construction raises `ValueError` |
| Missing manifest | Loader preflight raises before `max_cases` is applied |
| Manifest dataset/seed/fraction mismatch | Loader validation raises `ValueError` |
| Manifest ID absent under selected root | Loader validation raises `ValueError` |
| Both `resume` and `warm_start` configured | Shared pipeline raises `ValueError` |
| TransUNet root absent when external model is needed | External model validation raises before build |
| ResUNet3D ImageNet transfer disabled | No download or transfer occurs unless explicitly enabled |
| CNN data root missing or too few usable images | ImageFolder loader construction raises |
| `--cfg job` | Prints composed config and does not dispatch training |

Configuration inspection does not prove data integrity. Use the [DATA.md](DATA.md)
workflow for source preparation, canonical case discovery, NIfTI validation,
labels, and manifest contents. Do not run full or cloud training as a config
documentation check.
