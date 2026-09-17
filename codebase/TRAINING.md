# Training

This guide owns the active fit engine contract: pipeline handoff, phases,
selection metrics, checkpoints, resume modes, tracking, and run artifacts.
Input layouts, manifests, and loader construction belong to [DATA.md](DATA.md).
Hydra composition and profile values belong to [CONFIG.md](CONFIG.md).
Reproducibility identity and run-record rules belong to
[REPRODUCIBILITY.md](REPRODUCIBILITY.md). Efficiency telemetry, snapshot
gates, and benchmark separation belong to [BENCHMARKING.md](BENCHMARKING.md).

## Source Map

The execution claims here are grounded in active source and configuration:

- [`src/token_mixer/training/engine.py::{FitResult,fit}`](../src/token_mixer/training/engine.py)
- [`src/token_mixer/training/phases.py::{PhaseSpec,apply_phase}`](../src/token_mixer/training/phases.py)
- [`src/token_mixer/training/checkpoints.py::CheckpointManager`](../src/token_mixer/training/checkpoints.py)
- [`src/token_mixer/training/tracking.py::{Tracker,create_tracker}`](../src/token_mixer/training/tracking.py)
- [`src/token_mixer/training/artifacts.py::{write_run_artifacts,write_failed_run_artifact}`](../src/token_mixer/training/artifacts.py)
- [`src/token_mixer/evaluation/efficiency.py::{reset_peak_memory,peak_memory_gb,NvmlPowerSampler}`](../src/token_mixer/evaluation/efficiency.py)
- [`src/token_mixer/cli.py::{_run,_dispatch}`](../src/token_mixer/cli.py)
- [`src/token_mixer/pipelines/_baseline_common.py::{run_3d_baseline,run_2d_baseline}`](../src/token_mixer/pipelines/_baseline_common.py)
- [`src/token_mixer/pipelines/train_metaunetr.py::run_metaunetr`](../src/token_mixer/pipelines/train_metaunetr.py)
- [`src/token_mixer/pipelines/train_resunet3d.py::run_resunet3d`](../src/token_mixer/pipelines/train_resunet3d.py)
- [`src/token_mixer/pipelines/train_swinunetr.py::run_swinunetr`](../src/token_mixer/pipelines/train_swinunetr.py)
- [`src/token_mixer/pipelines/train_transunet.py::run_transunet`](../src/token_mixer/pipelines/train_transunet.py)
- [`src/token_mixer/pipelines/pretrain_cnn.py::run_cnn_denoising_pretrain`](../src/token_mixer/pipelines/pretrain_cnn.py)
- [`configs/experiment/metaunetr_mamba.yaml::training.phases`](../configs/experiment/metaunetr_mamba.yaml), [`configs/experiment/mod_a.yaml::training.phases`](../configs/experiment/mod_a.yaml), [`configs/experiment/mod_b.yaml::training.phases`](../configs/experiment/mod_b.yaml)
- [`configs/experiment/resunet3d.yaml::training.phases`](../configs/experiment/resunet3d.yaml), [`configs/experiment/swinunetr.yaml::training.phases`](../configs/experiment/swinunetr.yaml), [`configs/experiment/transunet.yaml::training.phases`](../configs/experiment/transunet.yaml)
- [`configs/experiment/cnn_denoising_pretrain.yaml::training.phases`](../configs/experiment/cnn_denoising_pretrain.yaml)
- [`configs/run/debug.yaml::{max_cases,epochs,phase1_epochs,phase2_epochs}`](../configs/run/debug.yaml), [`configs/run/full.yaml::{max_cases,epochs,phase1_epochs,phase2_epochs}`](../configs/run/full.yaml)

The links use `path::symbol` citations rather than generated API pages. Active
implementation and tests remain authoritative if this prose drifts.

## Pipeline To Engine

The supported orchestration path is:

```text
Hydra config
  -> cli.py::_run
  -> cli.py::_dispatch
  -> selected pipeline entrypoint
  -> model, loaders, evaluator, phases, checkpoints, tracker
  -> training.engine::fit
  -> best-checkpoint restore and test evaluation
  -> metrics.json and provenance.json
```

`cli.py::_run` resolves Hydra's output directory and writes `config.yaml`
before it dispatches. `cli.py::_dispatch` selects one runner from
`experiment.name`; an unknown selector fails before training. The runner builds
its model and loaders, then supplies the shared engine with a loss, evaluator,
phase list, tracker, and checkpoint manager.

The native baseline path is shared by the ResUNet3D and SwinUNETR runners, and
by the 2-D TransUNet runner through
`src/token_mixer/pipelines/_baseline_common.py::run_3d_baseline` and
`::run_2d_baseline`. Those helpers seed execution, build the loader metadata,
invoke `fit`, restore `best.pt`, evaluate the held-out loader, and return a
`FitResult` enriched with test metrics and metadata. MetaUNETR has its own
orchestration in `train_metaunetr.py::run_metaunetr`, but follows the same
checkpoint, tracker, and best-checkpoint boundary.

CNN denoising pretraining is a separate 2-D ImageFolder path. It uses
`pretrain_cnn.py::run_cnn_denoising_pretrain`, evaluates validation MSE through
the denoising evaluator, restores `best.pt`, and exports `encoder_best.pth`.
It has no BraTS test-volume result. The model-specific differences are listed
in [Model Exceptions](#model-exceptions); shared wording must not imply model
or protocol equivalence.

## FitResult

`engine.py::FitResult` is the model-independent completion value:

| Field | Meaning |
| --- | --- |
| `best_metric` | Best finite value of configured `monitor`; `NaN` if no best value was found. |
| `best_epoch` | Absolute epoch at which the best value was selected. |
| `history` | One mapping per completed epoch, including phase and optimizer counters. |
| `test_metrics` | Optional numeric mapping added by a pipeline after best-checkpoint test evaluation. |
| `metadata` | Optional pipeline provenance, such as architecture, variant, manifest hash, spacing, and source checkpoint. |

`fit` returns the first three fields after it logs the final summary. Direct
callers retain the default tracker finish behavior; pipeline helpers pass
`finish_tracker=False` so the tracker remains open through best-checkpoint
restore, held-out/final validation, local artifact writing, and W&B finalization.
The baseline helpers use
`_baseline_common.py::_result_with_test_metrics` to create a new result with
test metrics and loader/pipeline metadata. The CLI only writes completion
artifacts when the dispatched value is a `FitResult`.

## Phases And Transitions

`phases.py::PhaseSpec` contains five values: `name`, `epochs`,
`freeze_encoder`, `encoder_lr`, and `decoder_lr`. A model passed to `fit` must
expose an `nn.Module` named `encoder`. At each non-zero phase,
`phases.py::apply_phase` sets `requires_grad` on encoder parameters according
to `freeze_encoder`; it does not change non-encoder parameters.

The engine then creates fresh optimizer parameter groups for the current phase.
Only parameters with `requires_grad=True` enter a group. The encoder and all
other trainable parameters receive their phase-specific learning rates. While
the encoder is frozen, `engine.py::_train_epoch` also keeps it in evaluation
mode, preventing stateful layers such as BatchNorm from updating during the
frozen phase. A phase with zero epochs is skipped; negative phase epochs are
rejected.

Active phase plans are:

| Family and entrypoint | Active phase sequence | Profile interpolation |
| --- | --- | --- |
| `metaunetr_mamba`, `mod_a`, `mod_b` via `train_metaunetr.py::run_metaunetr` | `encoder_frozen`, then `full_finetune` | `run.phase1_epochs`, then `run.phase2_epochs` |
| `resunet3d` via `train_resunet3d.py::run_resunet3d` | `encoder_frozen`, then `full_finetune` | `run.phase1_epochs`, then `run.phase2_epochs` |
| `swinunetr` via `train_swinunetr.py::run_swinunetr` | One `train` phase | `run.phase2_epochs` |
| `transunet` via `train_transunet.py::run_transunet` | One `train` phase | `run.phase2_epochs` |
| `cnn_denoising_pretrain` via `pretrain_cnn.py::run_cnn_denoising_pretrain` | One `pretrain` phase | `run.epochs` |

The shipped debug values are one epoch for every selected phase. The shipped
full values are 20 plus 80 for the two-phase segmentation plans, 80 for the
single-phase SwinUNETR and TransUNet plans, and 30 for CNN pretraining. These
are configuration values, not evidence that a full run has been executed.

The engine rebuilds the optimizer and scheduler at each phase boundary. It
supports gradient accumulation, optional AMP, finite-loss and finite-gradient
checks, gradient clipping, and scheduler stepping at either epoch or update
interval. These controls affect execution but do not change phase ownership.

## Selection And History

`fit` reads `monitor` and `maximize` from the flattened pipeline config. A
candidate replaces the best value only when it is finite and better in the
configured direction. `validation_interval` controls validation by absolute
epoch; the final configured epoch is validated even when it is not divisible
by the interval. A missing monitored key, non-numeric value, or non-finite
value raises instead of silently selecting a checkpoint.

Each history row records at least `epoch`, `phase`, `phase_index`,
`phase_epoch`, `train_loss`, learning rate fields, and `global_step`. Validation
metrics are added to the same row. The segmentation experiment files monitor
`mean_dice` with `maximize: true`. CNN pretraining monitors `mse` with
`maximize: false`, so lower reconstruction error is better. Do not compare
these best values as if they were the same metric.

Completed epochs also append one scalar record with an absolute `train/epoch`
axis. Records retain train duration, optimizer-step delta, observed samples and
voxels, throughput, allocator peaks, `run/elapsed_seconds`, and `power/*`
fields. Validation records add `val/epoch` and `val/epoch_seconds` only when
validation runs; a new best adds `train/time_to_best_seconds` with its timing
scope. W&B definitions use `train/epoch` and `val/epoch` as axes while
`global_step` remains monotonic. `log_every_steps` and snapshot cadence do not
remove epoch records. `train/epoch_seconds` covers loader iteration, forward,
backward, and optimizer work; validation is timed separately. The process timer
starts after setup/checkpoint restoration, so `run/elapsed_seconds` is measured
process-segment elapsed time rather than fabricated wall time. When NVML is
enabled for CUDA, one sampler runs across the fit and each epoch records its
current board-power/energy snapshot; those values are cumulative sampler
snapshots, not allocator memory or a per-epoch reset. CPU allocator fields remain
`null` rather than implying a CUDA measurement.

When `early_stopping.enabled` is true, `monitor`, `mode`, `patience`,
`min_delta`, and `min_epochs` govern validation checks. Patience counts
consecutive non-improving validation evaluations, not raw epochs. Local
provenance and tracker summary retain stopped, stop-epoch, and best-epoch
state. Defaults remain disabled in shipped run groups. The last completed epoch
is logged and checkpointed before an early-stop break; a best checkpoint
selected on that validation is retained.

## Checkpoints And Metadata

Pipeline checkpoint roots come from `paths.checkpoint_dir` or the checkpoint
configuration. The active profile expression places them below the run output
directory. `CheckpointManager::save` writes named files atomically through a
temporary file and `os.replace`.

The engine writes these tags during a normal fit when a manager is present:

| File | When written | Load meaning |
| --- | --- | --- |
| `best.pt` | When a validation value becomes a new best | Model weights used for held-out evaluation and warm start by default |
| `last.pt` | At every completed epoch | Full continuation point for exact resume |
| `<phase>_resume.pt` | At the end of each completed phase | Phase-boundary continuation point |

The serialized payload includes model weights, optional optimizer, scheduler,
and AMP scaler states; epoch, phase, phase index, phase epoch, global step,
monitored metric, best metric and epoch, history, manifest hash, configuration,
checkpoint metadata, package code version, Python/NumPy/PyTorch/CUDA RNG
states, and the seeded DataLoader generator state when supplied. The active
pipeline paths always supply that generator.

Telemetry resume state includes measured cumulative train/validation/elapsed
seconds, `timing_scope`, time-to-best scope, and early-stopping counters. When a
snapshotter is active, its emitted-key ledger and bounded errors are stored too,
so resumed runs can deduplicate scheduled, best, and final previews. A resumed
process starts a new `perf_counter` segment and never treats the old timer as
continuous wall time.

`engine.py::_checkpoint_metadata` records compatibility information including
phase plan, monitor, direction, architecture, variant when present, manifest
hash, model configuration, loss, optimizer, scheduler, and explicit metadata.
`checkpoints.py::CheckpointManager::validate` compares the saved information
with the current expected metadata before a state load. This is a compatibility
gate, not a claim that a checkpoint contains the underlying dataset files.

`CheckpointManager::load` restores model and supplied optimizer, scheduler,
scaler, loader-generator, and RNG state. `CheckpointManager::load_model`
restores only model weights and deliberately leaves optimizer, loader, and RNG
state untouched. Pipelines use the latter for best-checkpoint evaluation and
warm start.

## Exact Resume

For the supported pipeline entrypoints, exact resume continues training state.
Configure `resume` or use a supported resume path resolution from
`_baseline_common.py::{resume_path,warm_start_path}`. `resume` and `warm_start`
are mutually exclusive, and either requires enabled checkpointing.

The pipeline sequence is:

1. Resolve the source `last.pt` path, or use the explicitly configured path.
2. Copy the source run's `best.pt` into the destination checkpoint root before
   creating the tracker or entering `fit`.
3. Let `fit` read and validate the source checkpoint metadata.
4. Load model, optimizer, scheduler, scaler, RNG, and DataLoader generator
   state, then continue at `phase_epoch + 1` with saved absolute epoch,
   `global_step`, best value, best epoch, and history.
5. Save continued `last.pt` and any improved `best.pt` in the destination root.

The pipeline contract requires the source best checkpoint even when the source
path is `last.pt`.
If it is absent, the pipeline fails before tracker creation or fitting; the
source run is not modified. In the normal pipeline call, the supplied loader
generator enables history restoration as well as shuffle-state restoration.
The checkpoint's phase, model, monitor, direction, manifest, and other expected
metadata must be compatible with the current run.

## Warm Start

Warm start transfers weights but starts a new optimization history. A boolean
`warm_start: true` resolves to the current checkpoint root's `best.pt`; a path
can be supplied directly. The pipeline records `resume_mode: warm_start` and
the source checkpoint in run metadata.

`fit` calls `CheckpointManager::load_model` for this mode. It does not restore
optimizer, scheduler, scaler, RNG, DataLoader generator, epoch counters, best
metric, or history. The new run begins at its configured first phase and writes
new checkpoints and artifacts. Use exact resume when the goal is continuation,
not transfer learning.

## Tracking

`tracking.py::Tracker` is a no-op interface with scalar, summary, table, image,
artifact, and lifecycle methods. `tracking.py::create_tracker` returns that
no-op tracker when tracking is disabled or `mode: disabled`; it does not import
W&B in those cases.

Supported W&B modes are:

| Configuration | Behavior |
| --- | --- |
| `enabled: false` | No-op tracker. |
| `enabled: true`, `mode: disabled` | No-op tracker despite enabled flag. |
| `enabled: true`, `mode: offline` | Initialize `wandb` in offline mode with the run config. |
| `enabled: true`, `mode: online` | Require `WANDB_API_KEY` or a matching standard `.netrc` entry, then initialize an online run. |

Initialization forwards project, entity, optional group/job type/tags, run
name, directory, mode, and the flattened run configuration. Epoch rows are
logged with their `global_step`; image logging is gated by `log_images`; tables,
summaries, and artifacts are forwarded through the tracker boundary; `finish`
is called after pipeline completion. Local defaults are disabled. Cloud
defaults use online project `token-mixer-placement-matters`, entity
`aniekanetimudo`, group `token-mixer-brats-seed42`, and `job_type: train`. No
credential value belongs in YAML, documentation, or an artifact.

The active W&B tracker forwards metrics, epoch axes, images, tables, summaries,
and artifact uploads; the disabled tracker remains a local no-op. Training
finalization uploads `config.yaml`,
`metrics.json`, `provenance.json`, and `best.pt` when present with `best` and
`latest` aliases. Benchmark uploads are separate and use `job_type: benchmark`;
see [BENCHMARKING.md](BENCHMARKING.md) for immutable source lineage and local
fallback. `artifact.wait()` completes the upload after `log_artifact`; the
returned immutable artifact reference is the downstream benchmark source.

With its default `finish_tracker=True`, `fit` finishes the tracker on setup,
training, evaluator, and checkpoint errors as well as on success. Pipeline
helpers pass `finish_tracker=False`, keep the tracker open through best restore,
held-out/final evaluation, local artifact writing, and tracker finalization, and
finish it in their outer `finally` block. If cleanup itself raises, the original
execution error is preserved. A tracker-construction error occurs before a
tracker exists and is reported directly by the pipeline.

Segmentation snapshots are an independent tracker-gated callback. Absolute
interval snapshots default to epochs `10`, `20`, `30`, and so on; a newly
selected best and a final snapshot may be added separately. Fixed train and
validation examples are used when enabled. See [BENCHMARKING.md](BENCHMARKING.md)
for image namespaces and disabled/local-only behavior.

## Run Artifacts And Failure Provenance

On entry, `cli.py::_run` writes the composed `config.yaml` with Hydra's
interpolations unresolved. After a successful `FitResult`,
`artifacts.py::write_run_artifacts` writes:

| Artifact | Contents |
| --- | --- |
| `metrics.json` | `best_epoch`, `best_metric`, complete epoch history, optional numeric test metrics, and timing/efficiency/power/early-stopping sections. |
| `provenance.json` | Schema version, code version, experiment, architecture, variant, model config, seed, device, manifest hash, monitor, direction, source checkpoint, runtime, protocol, nested timing/efficiency fields, early-stopping metadata, and tracking configuration. |

JSON conversion replaces non-finite floating-point values with `null`; this
keeps the artifact valid JSON without changing the in-memory metric behavior.

`artifacts.py::write_failed_run_artifact` writes only failure provenance and
does not fabricate metrics. The CLI attempts it for every dispatch exception;
requested ResUNet3D ImageNet transfer failures additionally carry a
`transfer_report`. The record includes `status: failed`, bounded error text, and
available architecture, model, seed, device, manifest, transfer, and tracking
metadata. Ordinary data, configuration, evaluator, or training exceptions are
still re-raised after the persistence attempt, and a failed provenance file is
not a completed-run result.

## Model Exceptions

| Pipeline | Exception to shared wording | Source boundary |
| --- | --- | --- |
| MetaUNETR variants | Uses custom orchestration and validates one variant, explicit spacing, and model-compatible spatial sizes before fitting. | [`src/token_mixer/pipelines/train_metaunetr.py::run_metaunetr`](../src/token_mixer/pipelines/train_metaunetr.py#L860-L973) |
| ResUNet3D | Uses native 3-D baseline flow. Optional ImageNet encoder transfer is disabled unless requested; requested transfer failures carry `transfer_report` for CLI failure provenance. | [`src/token_mixer/pipelines/train_resunet3d.py::{run_resunet3d,_build_model}`](../src/token_mixer/pipelines/train_resunet3d.py#L216-L322) |
| SwinUNETR | Uses native 3-D baseline flow and the optional MONAI `dice_ce` loss when configured. | [`src/token_mixer/pipelines/train_swinunetr.py::run_swinunetr`](../src/token_mixer/pipelines/train_swinunetr.py#L45-L59) |
| TransUNet | Uses 2-D slice loaders and slice evaluation, validates its external checkout and pretrained file before model construction, and returns canonical region logits through its adapter. | [`src/token_mixer/pipelines/train_transunet.py::run_transunet`](../src/token_mixer/pipelines/train_transunet.py#L71-L88) |
| CNN denoising pretraining | Uses ImageFolder denoising pairs, MSE minimization, one `pretrain` phase, encoder export, and optional reconstruction visualization. It is not a BraTS segmentation run. | [`src/token_mixer/pipelines/pretrain_cnn.py::run_cnn_denoising_pretrain`](../src/token_mixer/pipelines/pretrain_cnn.py#L755-L859) |

The shared engine requires the same named `encoder` seam across these models,
but that seam does not make their architecture, data, loss, or evaluation
protocols equivalent. Model guides should link here for shared fit behavior and
keep architecture-specific claims in their own guide.

## Safe Preflight

Composition inspection is safe and does not dispatch a runner:

```bash
uv run python -m token_mixer --config-name local experiment=mod_a run=debug --cfg job
```

Use [CONFIG.md](CONFIG.md) for the profile and override contract, then use
[DATA.md](DATA.md) to validate the selected manifest and data root. A config
preflight, synthetic test, or debug setting is not evidence of convergence,
real-data quality, or full-run completion.
