# Reproducibility

This guide owns run identity, random-state handling, profile separation, and
the limits of what current artifacts prove. Training semantics belong to
[TRAINING.md](TRAINING.md). Manifest and loader construction belong to
[DATA.md](DATA.md). Hydra composition belongs to [CONFIG.md](CONFIG.md).

## Source Map

The reproducibility contract is grounded in these active symbols and configs:

- [`src/token_mixer/reproducibility.py::{seed_everything,seed_worker}`](../src/token_mixer/reproducibility.py#L9-L28)
- [`src/token_mixer/training/checkpoints.py::{CheckpointManager,_capture_rng_state,_restore_rng_state}`](../src/token_mixer/training/checkpoints.py#L21-L470)
- [`src/token_mixer/training/engine.py::{_checkpoint_metadata,fit}`](../src/token_mixer/training/engine.py#L98-L791)
- [`src/token_mixer/pipelines/_baseline_common.py::{_manifest_metadata,_loader_kwargs}`](../src/token_mixer/pipelines/_baseline_common.py#L276-L358), [`src/token_mixer/pipelines/_baseline_common.py::{run_3d_baseline,run_2d_baseline}`](../src/token_mixer/pipelines/_baseline_common.py#L1168-L1351)
- [`src/token_mixer/pipelines/train_metaunetr.py::build_loaders`](../src/token_mixer/pipelines/train_metaunetr.py#L484-L569), [`src/token_mixer/pipelines/train_metaunetr.py::run_metaunetr`](../src/token_mixer/pipelines/train_metaunetr.py#L860-L973)
- [`src/token_mixer/pipelines/pretrain_cnn.py::build_dataloaders`](../src/token_mixer/pipelines/pretrain_cnn.py#L228-L324), [`src/token_mixer/pipelines/pretrain_cnn.py::run_cnn_denoising_pretrain`](../src/token_mixer/pipelines/pretrain_cnn.py#L755-L859)
- [`src/token_mixer/cli.py::{_save_composed_config,_run}`](../src/token_mixer/cli.py#L27-L100)
- [`src/token_mixer/training/artifacts.py::{write_run_artifacts,write_failed_run_artifact}`](../src/token_mixer/training/artifacts.py#L101-L284)
- [`src/token_mixer/__init__.py::__version__`](../src/token_mixer/__init__.py#L1)
- [`src/token_mixer/training/tracking.py::create_tracker`](../src/token_mixer/training/tracking.py#L49-L81)
- [`configs/local.yaml::{runtime,paths,tracking}`](../configs/local.yaml#L6-L32), [`configs/cloud.yaml::{runtime,paths,tracking}`](../configs/cloud.yaml#L6-L32)
- [`configs/run/debug.yaml::{num_workers,seed,deterministic}`](../configs/run/debug.yaml#L1-L17), [`configs/run/full.yaml::{num_workers,seed,deterministic}`](../configs/run/full.yaml#L1-L17)
- [`configs/data/brats.yaml::{dataset_id,split_seed,val_fraction,test_fraction}`](../configs/data/brats.yaml#L1-L22)

The links use `path::symbol` citations. Runtime outputs, checkpoints, W&B
directories, caches, notebook views, archive files, and machine-local paths
are generated or deployment state, not source authority.

## Seed Setup

Every active training pipeline calls
`reproducibility.py::seed_everything` before model and loader construction. It:

1. Seeds Python's `random` module with the configured integer.
2. Seeds NumPy with `seed % 2**32`.
3. Seeds PyTorch's process-wide generator.
4. Seeds all CUDA devices when CUDA is available.
5. Sets `torch.backends.cudnn.deterministic` to the configured deterministic
   value and `torch.backends.cudnn.benchmark` to its inverse.
6. Returns a separately seeded `torch.Generator` for DataLoader and split
   operations.

The shipped `debug` and `full` run groups both set `seed: 42` and
`deterministic: true`. A different run must record its effective override in
the composed config and output provenance. These settings request deterministic
cuDNN behavior; they do not prove that every CUDA kernel or external operation
is deterministic.

`reproducibility.py::seed_worker` derives a worker seed from
`torch.initial_seed() % 2**32` and seeds Python and NumPy inside that worker.
The baseline and CNN loader builders install it when `num_workers` is non-zero.

## DataLoader State

The returned `torch.Generator` is passed into the train, validation, and test
DataLoaders. It controls shuffled ordering and is also used by the CNN
ImageFolder split permutation. Worker count, pinning, persistence, prefetching,
and batch settings come from the composed run configuration; see
[CONFIG.md](CONFIG.md) for the active debug/full values.

`CheckpointManager::save` serializes the generator's state as
`loader_generator_state` when a generator is supplied. It does not serialize a
live generator object. `CheckpointManager::load` restores that state into the
caller-provided generator. It also restores process RNG state for Python,
NumPy, PyTorch, and CUDA when present.

The active pipeline calls pass the generator to `fit`, so exact resume restores
the loader sequence and history along with optimizer and model state. A direct
caller that omits `loader_generator` is outside that full pipeline resume
contract.

## Manifest Identity

All BraTS segmentation pipelines validate one persisted split manifest before
building loaders. The active loader paths compare configured dataset ID, split
seed, validation fraction, and test fraction with the manifest; reject duplicate
IDs; discover the selected data root; require every manifest ID to exist; and
apply `max_cases` only after manifest mapping and split selection. Full details
are owned by [DATA.md](DATA.md).

The loader metadata assembled by `_baseline_common.py::_manifest_metadata`,
`::build_volume_loaders`, `::build_slice_loaders`, or the MetaUNETR loader
records:

| Field | Meaning |
| --- | --- |
| `manifest_path` | Configured manifest location as represented by the running process. |
| `manifest_hash` | SHA-256 digest of the manifest file bytes. |
| `dataset_id`, `split_seed`, `manifest_seed` | Split identity values. |
| `val_fraction`, `test_fraction` | Split proportions. |
| `split_counts` | Effective train, validation, and test counts after any debug limit. |
| `max_cases` | Applied limiter, or `null`. |

The hash is a manifest-file identity, not a hash of the NIfTI contents, source
directory, preprocessing code, or external assets. Identical manifest hashes
therefore do not by themselves prove identical underlying data.

## Config, Code, And Checkpoint Identity

Reproducing a run requires keeping these identities together:

| Identity | Current evidence |
| --- | --- |
| Manifest | Validated metadata and `manifest_hash` propagated from loader metadata. |
| Composed config | `cli.py::_save_composed_config` writes `config.yaml` before dispatch with `OmegaConf.save(..., resolve=False)`. The engine also stores a plain run config in training checkpoints. |
| Code version | Checkpoints and successful provenance record `code_version`; when not supplied, active code defaults to package `token_mixer.__version__`, currently defined in `src/token_mixer/__init__.py::__version__`. |
| Checkpoint metadata | `engine.py::_checkpoint_metadata` records phase plan, monitor, direction, model identity, manifest hash, model config, objective, optimizer, scheduler, and explicit metadata; `CheckpointManager::validate` checks it on load. |
| Checkpoint state | `best.pt`, `last.pt`, and phase-resume files carry model/training state and RNG state as described in [TRAINING.md](TRAINING.md). |

The current runtime records package code version, not an automatic Git commit
hash. A reproducibility record should therefore also retain the Git commit
used for the run outside the runtime-generated files.

Successful `provenance.json` contains selected identity fields, metadata, and
tracking configuration. It is complementary to the full composed `config.yaml`;
it is not a second complete configuration serialization.

## Profile Separation

The active profiles intentionally separate runtime, data root, device, run
scale, and output namespace:

| Profile | Runtime | BraTS root | ImageFolder root | Device | Default run |
| --- | --- | --- | --- | --- | --- |
| `local` | `local` | `data/local/brats` | `data/local/imagenet` | `auto` | `debug` |
| `cloud` | `cloud` | `data/cloud/brats` | `data/cloud/imagenet` | `cuda` | `full` |

Both profiles currently point at the same configured manifest name and set
tracking disabled. The manifest's case IDs must exist under whichever selected
root is used. A debug case limiter does not make a small data root equivalent
to a full root.

Run outputs use the repository-relative pattern:

```text
outputs/<runtime>/<experiment.name>/<run.name>/<date>/<time>/
  config.yaml
  checkpoints/
  metrics.json
  provenance.json
```

The runtime value is part of the output namespace, so local and cloud runs do
not share the same experiment directory when their configured output root is
the same. The `cloud` label is a configuration choice; it is not evidence that
cloud or GPU execution occurred.

## W&B Separation

W&B mode is orthogonal to profile identity. `tracking.py::create_tracker`
receives the tracking mapping plus the flattened run config. Disabled tracking
returns a no-op tracker without importing W&B. Offline and online modes
initialize W&B with the project, optional entity/name, directory, mode, and run
config; online mode requires `WANDB_API_KEY` to be present before initialization.

The shipped `local` and `cloud` profiles both use:

```yaml
tracking:
  enabled: false
  mode: disabled
```

Enabling W&B does not itself create a data split or checkpoint identity. Keep
the selected profile, experiment selector, run name, manifest hash, code
version, and checkpoint metadata in the run record. Never commit the API key or
any other secret.

## Run Record Checklist

For an interpretable experiment record, retain the following together:

1. Effective profile, experiment selector, run group, and all Hydra overrides.
2. Composed `config.yaml` from the run output.
3. Manifest path, SHA-256, dataset ID, split seed, fractions, and effective
   split counts.
4. Package code version and the Git commit recorded by the experiment owner.
5. Checkpoint filename, checkpoint type, phase state, and compatibility metadata.
6. Seed, deterministic setting, device, DataLoader worker settings, and whether
   exact resume or warm start was used.
7. W&B mode and run identity when tracking is enabled, or explicit disabled
   mode when it is not.
8. Data status: real dataset, synthetic fixture, or configuration-only
   preflight.

The last item prevents a successful config print or synthetic contract test
from being reported as a real-data result.

## Cloud/GPU Capacity Records

The following bounded capacity record was collected on the Shadeform A6000
host for commit `d03500e171982068eb5c3d0a9d6840c6fa22d188`. It used one
deterministic AdamW forward/backward/optimizer step per candidate, one process
at a time, with AMP enabled and an otherwise idle GPU. Segmentation probes used
synthetic tensors shaped `[B, 4, 96, 96, 96]`; the CNN probe used `[B, 3, 96,
96]`. These are memory-capacity measurements, not convergence or quality
results.

| Model | Largest passing batch tested | Peak allocated / reserved bytes | Step time | Next boundary |
| --- | ---: | ---: | ---: | --- |
| ResUNet3D | 8 | 12,520,522,240 / 16,020,144,128 | 1.24 s | No OOM through batch 8 |
| SwinUNETR | 8 | 17,758,090,752 / 24,972,886,016 | 2.26 s | No OOM through batch 8 |
| MetaUNETR-Mamba | 8 | 30,336,061,440 / 34,554,773,504 | 2.56 s | No OOM through batch 8 |
| Mod-B | 2 | 29,162,787,840 / 30,568,087,552 | 3.66 s | Batch 4 OOM; process reached about 47.09 GiB |
| Mod-A | 1 | 25,243,877,888 / 26,325,549,056 | 4.27 s | Batch 2 OOM; process reached about 47.34 GiB |

Operational capacity for this host is therefore **ResUNet3D, SwinUNETR, and
MetaUNETR at batch 8; Mod-B at batch 2; Mod-A at batch 1 only**. These are
largest passing candidates, not recommended production margins; retain the
planned 15--20% headroom and use gradient accumulation when a larger effective
batch is needed. Concurrent model processes were not measured and must not be
inferred from this table; run one model process per GPU until a separate
concurrency test is approved.

| Record field | Value |
| --- | --- |
| Host/GPU | Shadeform; 1 x NVIDIA RTX A6000, 49,140 MiB |
| Driver / CUDA / PyTorch | 595.84 / 13.0 / 2.13.0+cu130 |
| Python / seed / determinism | 3.12.14 / 42 / enabled |
| Manifest | `data/manifests/brats_seed42.json`; SHA-256 `5d58a3dd38ee82ac44e4f1625bd82a9900d987b2af8098c6299ecb26ce8335bb` |
| Probe artifact | `outputs/cloud/capacity/*.json` (ignored runtime evidence) |
| Data-loader workers | Not applicable to synthetic probe; real-data debug used 0 workers |

The CNN batch-1 probe also passed with peak allocated/reserved bytes
`67,010,048 / 75,497,472`. Raw probe outputs, checkpoints, and machine-local
data remain outside Git; preserve their hashes and paths alongside any future
run write-up.

## No-Real-Data Claims

This documentation does not claim model convergence or segmentation quality.
The capacity section is an explicit synthetic one-step GPU measurement, while
the real-data debug runs are bounded smoke checks only. The safe validation
boundaries remain composition inspection, contract tests, and clearly labelled
runtime evidence. The `--cfg job` option prints the composed configuration and
does not dispatch a runner; loader construction, manifest checks, and training
are separate steps.

The clean repository may not contain the configured manifest or data roots.
That absence is a data-preflight condition, not a reason to substitute a
synthetic manifest for a real experiment. Debug limits, synthetic arrays, unit
tests, and offline W&B initialization prove implementation seams only.

When a future result is written up, state whether it used real data and identify
the manifest, config, code version, checkpoint, and profile. Do not infer that
status from an output directory name, a checkpoint file, a W&B run, or a passing
unit test.
