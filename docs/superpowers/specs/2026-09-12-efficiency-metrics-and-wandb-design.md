# Computational Efficiency Metrics and W&B Benchmarking

**Status:** Design revised per user clarification; implementation plan pending user review
**Date:** 2026-09-12

## Summary

Add reproducible computational-efficiency measurement to the existing
Token-Mixer-Placement-Matters workflow without making benchmark timing part of
normal training or introducing nnUZoo as a dependency.

The design has three coordinated measurement paths:

1. **Training telemetry:** low-overhead epoch/phase timing, throughput, and peak
   allocator memory plus NVML board-power/energy samples recorded during
   training, with configurable train/validation segmentation snapshots at
   selected absolute epochs.
2. **Explicit benchmark run:** a separate command that restores a selected
   checkpoint and measures static model cost, synchronized inference latency,
   throughput, peak memory, NVML board power/energy, and fixed-protocol
   case-level inference.
3. **Static operator accounting:** THOP measures MACs and fvcore measures FLOPs
   on the same fixed input, with tool versions, conventions, unsupported
   operators, and partial/unavailable status preserved in the result.

Training and benchmark W&B runs are linked through a stable study group, source
run ID, and checkpoint artifact. Local JSON remains the durable evidence path;
W&B provides searchable histories, summaries, tables, and artifacts when online
tracking is enabled.

Carbon/emissions estimates are excluded. Board-power/energy sampling is part of
the first implementation; NVML unavailability is an explicit measurement
status and does not invalidate non-power measurements.

## Decisions

- Use a **separate explicit benchmark command**. Training logs only cheap
  telemetry; benchmark timing never changes the training loop.
- Do not add nnUZoo. Adopt its controlled-comparison principles—shared data
  contract, preprocessing, folds, hardware, and reporting—but keep this
  repository's existing pipeline and model boundaries.
- Use the current hashed BraTS manifest for the first controlled study. Add
  repeated seeds/five-fold validation later for publication-grade uncertainty;
  do not delay the first pipeline validation for nnUZoo.
- Keep separate protocols for native 3-D full-volume segmentation and
  TransUNet 2-D slice inference. Do not compare one 2-D slice directly with one
  3-D patient volume.
- Keep CNN denoising pretraining, ordinary supervised segmentation,
  TransUNet slice inference, and MetaUNETR episodic training explicitly labeled
  as different task/protocol families.
- Do not create a single composite efficiency score. Report accuracy, latency,
  throughput, memory, and training time as separate axes and analyze Pareto
  trade-offs.
- Use native PyTorch/stdlib measurements for timing, parameter counts, and
  allocator memory. Use `ultralytics-thop==2.1.6` (distribution, imported as
  `thop`) for MACs and `fvcore==0.1.5.post20221221` for FLOPs on the same
  prepared input. Record each tool's version and convention separately; never
  silently treat unsupported custom Mamba operations as zero MACs/FLOPs.
- Add `ultralytics-thop==2.1.6`, `fvcore==0.1.5.post20221221`, and
  `nvidia-ml-py==13.610.43` to the project environment and lock them in
  `uv.lock`. `nvidia-ml-py` supplies the `pynvml` import; do not install the
  legacy `thop` distribution alongside `ultralytics-thop` or duplicate NVML
  bindings.
- Exclude CodeCarbon and carbon-equivalent estimates from the first study.
  NVML board-power/energy measurement is part of this implementation, but is
  reported as board-level measurement with explicit unavailable status when
  hardware, permissions, or the NVML library do not support it.
- Make cloud W&B tracking mandatory for training runs. Keep local tracking
  disabled by default, but make `configs/cloud.yaml` default to online tracking
  with explicit entity/project values. A cloud training run is not considered
  ready until its live epoch history is visible in W&B.
- Make segmentation snapshots explicit and tracker-controlled. Default cadence
  is `snapshot_interval_epochs=10`, meaning snapshots at absolute epochs `10`,
  `20`, `30`, and so on, plus a selected-best snapshot whenever a new best
  validation score is recorded and a final snapshot when available. Each
  snapshot covers both fixed training and validation examples and renders all
  canonical tumor regions `[ET, TC, WT]`.
  Snapshot generation is skipped when the snapshot feature is disabled,
  tracking is disabled, or `tracking.log_images` is false. Offline tracking may
  generate and store images in the local W&B run without network access; online
  tracking logs them to W&B. A separate explicit local-visualization override
  may request PNGs without W&B.
- Make W&B destination explicit for reproducibility:
  `entity=aniekanetimudo`, `project=token-mixer-placement-matters`, unless the
  user selects a team entity before implementation.
- Do not use unrelated historical project
  `unet_brain-tumor-segmentation_1`.

## Current system evidence

The repository currently follows:

```text
Hydra config
  -> src/token_mixer/cli.py dispatch
  -> model/data/pipeline construction
  -> training.engine.fit
  -> checkpoint restore
  -> held-out evaluation
  -> local metrics/provenance artifacts
```

Relevant active seams:

- `src/token_mixer/cli.py:39-100` dispatches Hydra jobs and writes
  `config.yaml`, `metrics.json`, and `provenance.json`.
- `configs/cloud.yaml:1-32` currently has W&B disabled:

  ```yaml
  tracking:
    enabled: false
    mode: disabled
    project: token-mixer-placement-matters
    entity: null
  ```

  This is the current pre-instrumentation state. The target cloud profile
  changes `enabled` to `true`, `mode` to `online`, and `entity` to
  `aniekanetimudo`; the local profile remains disabled.

- `src/token_mixer/training/tracking.py:49-81` supports disabled, offline, and
  online initialization.
- `src/token_mixer/training/engine.py:399-459` owns batch training;
  `633-705` owns epoch history and tracker logging; `775-791` writes final
  training summary.
- `src/token_mixer/training/artifacts.py:101-214` persists local history,
  metrics, and provenance.
- `src/token_mixer/evaluation/inference.py:188-288` owns MONAI sliding-window
  3-D inference.
- `src/token_mixer/pipelines/_baseline_common.py:717-786` owns direct 2-D
  slice inference and `1168-1351` owns shared 3-D evaluation boundaries.
- `src/token_mixer/pipelines/train_metaunetr.py:860-973` owns MetaUNETR's
  custom training/test orchestration.
- `src/token_mixer/pipelines/pretrain_cnn.py:755-859` owns CNN denoising
  training and validation; it has no segmentation test split.
- `src/token_mixer/evaluation/metrics.py:21-197` owns Dice, HD95, thresholding,
  and exclusion counts.

The cloud W&B credential was verified without creating a run: W&B SDK `0.29.0`
loaded credentials from `/home/shadeform/.netrc`, and a read-only API call
identified `aniekanetimudo`. This proves authentication, not write permission.
The current `tracking.py` online guard checks only `WANDB_API_KEY`, so it would
reject this valid `.netrc` credential. The implementation must use W&B's normal
credential discovery (environment or `.netrc`) without printing the secret. The
first online debug smoke must verify project creation/write access and live
history visibility.

The cloud GPU/data/weight preflight is complete. Historical synthetic capacity
records are evidence for batch selection, not training-quality or inference-
quality results.

## Design

### 1. Shared efficiency measurement boundary

Add one small shared efficiency module under `src/token_mixer/evaluation/`.
It owns pure measurement and normalization, not W&B calls or pipeline policy.

Responsibilities:

- count total and trainable parameters using `model.parameters()`;
- collect CUDA allocator peak allocated/reserved bytes;
- time synchronized CUDA work with `torch.cuda.Event`;
- use `time.perf_counter()` with explicit synchronization for CPU/fallback
  paths;
- summarize repeated samples as mean, median, p95, and standard deviation;
- record warmups, repetitions, batch size, input shape, precision, and protocol;
- run THOP MAC counting and fvcore FLOP counting on one fixed prepared input,
  retaining each tool/version, convention, unsupported-operator list, and
  partial result status;
- sample NVML power usage at a configured interval without synchronizing CUDA or
  blocking the measured model loop, then integrate samples into board-level
  average/max watts and joules for the named measurement boundary;
- normalize OOM, unsupported operations, NVML failures, and unavailable
  hardware fields into explicit status fields rather than silently dropping
  them.

The module must not allocate random inputs inside a timed model-only loop.
Inputs are prepared once, moved to the target device before timing, and reused.
End-to-end measurements may include preparation/transfer only when the boundary
is explicitly named as end-to-end.

NVML power is sampled from `nvidia-ml-py`/`pynvml` using a background sampler.
Each sample records monotonic time and milliwatts converted to watts. Energy
uses trapezoidal integration over samples within the named boundary. The result
is board-level energy, not isolated model energy; device index, sample interval,
sample count, NVML version, and status are recorded. No CUDA synchronization is
added for power sampling.

### 2. Training telemetry

Instrument `training.engine.fit` at epoch/phase boundaries. Do not synchronize
the GPU on every batch by default; that would alter the training performance
being measured.

Required training fields:

```text
train/epoch_seconds
train/optimizer_steps
train/samples_per_second
train/voxels_per_second
train/peak_memory_allocated_gb
train/peak_memory_reserved_gb
val/epoch_seconds
run/elapsed_seconds
train/time_to_best_seconds
```

Definitions:

- `train/epoch_seconds` includes loader iteration, forward, backward, and
  optimizer work for the training epoch; model setup and checkpoint download
  are excluded.
- `val/epoch_seconds` measures validation separately.
- `samples_per_second` and `voxels_per_second` use observed samples/voxels,
  not nominal batch size.
- peak memory is reset immediately before the first measured full optimizer
  step and reported as PyTorch allocator values; it is not added to NVML values.
- NVML power/energy fields describe board-level samples during the epoch
  boundary; they are not combined with allocator memory or presented as model
  attribution. If sampling is unavailable, fields are `null` and status names
  the reason.
- `time_to_best_seconds` uses the existing validation-selected best checkpoint.
  A time-to-target field is only added if a target is declared before training.
- MetaUNETR records distinct phase timing when its episodic/adaptation phases
  are observable; the aggregate must not hide those phases.

Every completed epoch appends scalar training metrics to existing history and
logs them to W&B against the monotonic `global_step`, with an explicit
`train/epoch` value for epoch-based charts. Validation metrics are logged at
the configured `validation_interval` and include their absolute epoch. The
segmentation snapshot interval is independent of scalar metric logging and
does not suppress, batch, or otherwise change per-epoch metric records.

#### Early stopping

Early stopping is configurable and evaluated only after validation metrics are
available. Its configuration includes `enabled`, `monitor` (defaulting to the
existing validation-selected metric), `mode`, `patience`, `min_delta`, and an
optional `min_epochs`. `patience` counts consecutive validation evaluations
without an improvement, not raw training epochs. When patience is exhausted,
the current epoch's metrics and any newly selected-best snapshot/checkpoint
are written before the loop exits. The actual stopping epoch and best epoch
are recorded in provenance and W&B config/summary. A scheduled snapshot at
the last interval and a best snapshot at a non-interval epoch are both kept.

Data-loader versus compute timing is optional diagnostic instrumentation. It is
not required on every run because fine-grained synchronization adds overhead.

Training telemetry is appended to existing epoch history and local artifacts.
It must not alter model, optimizer, RNG, or loader checkpoint semantics. If
resuming, distinguish per-process segment time from cumulative training time;
do not reconstruct elapsed time from a stale `perf_counter` value.

### 3. Training and validation segmentation snapshots

Segmentation snapshots are visualization evidence, not a replacement for
numeric metrics. They run only at configured absolute-epoch intervals and
after the model has completed the current training/validation work; they are
never part of the per-batch timing boundary.

The configuration is explicit:

```yaml
visualization:
  segmentation_snapshots:
    enabled: true
    snapshot_interval_epochs: 10
    include_best: true
    include_final: true
    splits: [train, val]
    sample_count: 1
    axis: 0
    image_channel: 3
    local_enabled: false
    output_dir: ${paths.experiment_output}/segmentation_snapshots
```

`tracking.log_images` is an independent W&B image gate. The effective
behavior is:

| Snapshot feature | Tracking mode | `tracking.log_images` | Behavior |
| --- | --- | --- | --- |
| disabled | any | any | No loader iteration, inference, matplotlib import, PNG construction, or W&B call |
| enabled | disabled | any | No W&B import or image construction; local PNGs only when `local_enabled=true` |
| enabled | offline | true | Construct snapshots, save local PNGs, and log to the local offline W&B run; no network |
| enabled | online | true | Construct snapshots, save local PNGs, and log images to the configured online W&B run |
| enabled | offline/online | false | Skip image construction and W&B image logging; local PNGs only when `local_enabled=true` |

The gate is evaluated before iterating a snapshot loader, running inference,
importing plotting code, or constructing an image object. `Tracker` exposes a
lazy `log_image`/`log_images` method. The base tracker is a no-op; the W&B
adapter constructs `wandb.Image` only after its mode and `log_images` gates
permit it. Pipelines never import W&B directly.

At every positive multiple of `snapshot_interval_epochs`, the engine invokes a
snapshot callback after the current epoch's work and any validation/best-
checkpoint decision using the absolute epoch number, not a phase-local epoch.
If that epoch is not a regular validation boundary, the callback runs only the
fixed-example visualization inference rather than a full validation pass. The
callback receives fixed, non-augmented train and validation examples
selected deterministically from the declared manifest. It records the absolute
epoch, global step, split, model/protocol, snapshot kind, and hashed case
identifier in metadata/captions. It does not reuse shuffled training batches,
so the same examples remain visually comparable across epochs. Resumed runs
deduplicate snapshots by `(split, epoch, kind, case_hash)`.

Native 3-D snapshots use the existing sliding-window evaluator. TransUNet
snapshots use its 2-D slice path, adapting shapes without changing label
semantics. Predictions pass through `logits_to_regions` and always contain all
canonical regions `[ET, TC, WT]`. `save_slice_visualization` renders a
region-aware multi-panel preview with MRI channel `3` (`t2f`), ground truth,
prediction, and overlay/error views; the existing WT→TC→ET color order and
yellow/red/cyan legend remain authoritative. If one slice cannot visibly
contain every region, the configured montage uses deterministic region-aware
slice selection rather than silently dropping a class.

Scheduled images use stable keys such as
`segmentation/train/epoch_0010` and `segmentation/val/epoch_0010`, with W&B
step set to the corresponding global step. A newly selected best at epoch 104
therefore produces `kind=best` images even when the interval is 10; the
scheduled image at epoch 100 is retained. Selected-best and final snapshots
are logged after best restoration/final validation when configured, and are
deduplicated if they coincide with a scheduled epoch. Local files live under
`<Hydra output>/segmentation_snapshots/{train,val}/`; W&B offline files remain
under the configured W&B directory. Snapshot rendering or logging failures
must preserve training/quality metrics, record a bounded snapshot failure in
local provenance, and never expose raw case IDs or credential values.

### 4. Static model-cost measurement

Run after model construction and before inference benchmarking:

```text
model/parameters
model/trainable_parameters
model/macs
model/flops
model/mac_tool
model/mac_tool_version
model/flop_tool
model/flop_tool_version
model/mac_convention
model/flop_convention
model/mac_status
model/flop_status
model/unsupported_ops
model/checkpoint_bytes
```

The exact composed input shape, channel count, ROI/patch shape, and precision
are part of the record. Parameters are authoritative. THOP MACs and fvcore
FLOPs are authoritative only for supported operators; an unsupported custom
Mamba/SSM path must produce a partial/unavailable status and explanatory list.
THOP's parent/child aggregation must be checked for custom Mamba/SSM hooks;
custom handlers may count only operations not already represented by child
modules, avoiding nested Linear/Norm double-counting. `model/macs` comes from
THOP's MAC convention and `model/flops` comes from fvcore's FLOP convention;
neither value is silently converted into the other.

Static values belong primarily in the benchmark summary because they do not
change each epoch. They may be copied into training summary for convenient
filtering.

### 5. Explicit inference benchmark

Add a benchmark runner/config through the existing Hydra/CLI dispatch. It
accepts a local best checkpoint or a pinned W&B artifact and emits a local
benchmark JSON plus a W&B benchmark run.

#### Model-level protocol

For fixed-shape tensors:

- `model.eval()` and `torch.inference_mode()`;
- 20 warmup iterations and a predeclared repeated sample count;
- CUDA synchronization before timing and after the final event;
- latency distribution: mean, median, p95, standard deviation;
- throughput sweep over declared physical batch sizes until OOM;
- preallocated device inputs for model-only throughput;
- reset peak allocator stats immediately before the measured section;
- stop or isolate the process after the first OOM boundary and record the
  boundary; do not swallow unrelated `RuntimeError`s.

Required fields:

```text
inference/latency_mean_ms
inference/latency_median_ms
inference/latency_p95_ms
inference/latency_std_ms
inference/throughput_samples_per_second
inference/throughput_voxels_per_second
inference/peak_memory_allocated_gb
inference/peak_memory_reserved_gb
power/average_watts
power/max_watts
power/energy_joules
power/sample_count
power/sample_interval_ms
power/status
inference/warmup_iterations
inference/repetitions
inference/batch_size
inference/protocol
```

#### Case-level protocol

Measure fixed held-out cases through the actual evaluator:

```text
inference/case_latency_median_ms
inference/case_latency_p95_ms
inference/cases_per_second
inference/sliding_window_count
inference/n_cases
inference/protocol
```

Use a declared case set and record its manifest/split hash. Full-volume cases
need not be repeated 100 times; a fixed held-out set is more meaningful than a
prohibitively expensive repetition count. Model-level repeated timing and
case-level end-to-end timing are separate measurements.

Boundaries are reported separately when collected:

```text
load_seconds
preprocess_seconds
model_compute_seconds
postprocess_seconds
metric_seconds
end_to_end_seconds
```

Model loading and checkpoint restoration are never silently included in
per-case latency.

Power fields are recorded separately for each model-level or case-level
measurement boundary when NVML is available. They describe board-level power
and energy during that boundary and never include checkpoint restoration.

#### Protocol families

- `native_3d_full_volume`: existing MONAI sliding-window path.
- `transunet_2d_slice`: direct slice path, aggregated over a declared case's
  slices when case-level latency is reported.
- `cnn_denoising_validation`: CNN pretraining task, not segmentation quality.

The benchmark records protocol family in every row and never merges these
families into one throughput chart.

### 6. Quality metrics and per-case evidence

Existing quality metrics remain unchanged and are paired with efficiency:

```text
test/dice_ET
test/dice_TC
test/dice_WT
test/avg_dice
test/hd95_*
test/hd95_excluded_cases
```

Non-finite HD95 values become `null` in serialized/W&B structured data, with
explicit exclusion counts. They must not become zero or disappear silently.

Case-level benchmark rows use hashed case identifiers and contain:

```text
case_id_hash
model
protocol
dice_by_region
hd95_by_region
latency_ms
voxel_count
spacing
sliding_window_count
exclusion_flags
```

No raw medical volumes, PHI, raw case IDs, or unapproved patient identifiers
are sent to W&B. Segmentation snapshot images are an explicit exception: they
are derived, de-identified previews produced only when the snapshot and image
gates permit them, and captions/metadata contain hashed case IDs only.

### 7. W&B lifecycle and schema

Extend `training/tracking.py` as the only W&B boundary. Pipelines must not call
the W&B SDK directly. The abstraction preserves disabled and offline behavior
while adding summary, table, artifact, and lazy image operations. `log_image`/
`log_images` must not construct `wandb.Image` until the adapter confirms
`tracking.enabled=true` and `tracking.log_images=true`; disabled mode must not
import W&B. Offline mode uses W&B's local offline run and never requires a
network connection.

The cloud profile is the online path used for the study. A completed epoch is
logged while the run remains open, so W&B charts update during training rather
than appearing only after process exit. Local and offline modes remain useful
for tests and development but are not valid substitutes for the cloud study's
online tracking gate.

#### Training run

```text
job_type: train
group: {study_id}
entity: aniekanetimudo
project: token-mixer-placement-matters
```

Record in config/provenance:

- schema version;
- model family, architecture, variant, dimensionality;
- dataset ID, manifest hash, split counts, fold, and seed;
- ROI/input shape, spacing, overlap, precision/AMP, workers, device;
- GPU model, driver, CUDA, PyTorch, Python, and W&B SDK versions;
- Git commit and dirty-tree state;
- optimizer, scheduler, effective batch, physical microbatch, and accumulation;
- source checkpoint/run ID for resume or warm start;
- protocol and intentional model-specific deviations;
- THOP distribution/import versions, MAC/FLOP conventions,
  unsupported-operator status, and NVML version/device/sampling configuration.

The training run remains open through:

```text
training
→ best-checkpoint restore
→ held-out quality evaluation
→ summary/provenance/checkpoint artifact logging
→ finish
```

During steps 7--12, the user can inspect the live run in W&B. The required
live charts are training/validation loss, regional/mean Dice, HD95, learning
rate, epoch duration, throughput, peak allocator memory, and the configured
train/validation segmentation snapshot series. System telemetry may appear as
additional W&B charts, but exact benchmark values come from the explicit
synchronized collectors.

This fixes the current ordering where held-out evaluation occurs after the
tracker finishes. If changing ownership of `finish()` is too invasive, create
an explicitly linked evaluation run; do not leave test metrics local-only.

#### Benchmark run

```text
job_type: benchmark
group: {study_id}
source_train_run_id: {train_run_id}
source_checkpoint_artifact: {entity/project/artifact:version}
protocol_id: {protocol_id}
```

Use a separate run so benchmark reruns do not collide with training's explicit
global step stream. Benchmark repetition rows use their own monotonic axis.

Use stable namespaces:

```text
train/*
val/*
test/*
model/*
inference/*
efficiency/*
segmentation/*
```

The existing training `global_step` remains monotonic. If custom axes are
needed, define them explicitly (`train/epoch`, `train/global_step`,
`benchmark/repetition`) rather than logging unrelated streams with repeated or
decreasing explicit steps.

#### Artifacts

Log only the selected best checkpoint by default, plus:

- `config.yaml`;
- `provenance.json`;
- `metrics.json`;
- benchmark JSON;
- optional per-case table.

Use aliases such as `best` and `latest`, and record the immutable artifact
version in every downstream benchmark. Call `artifact.wait()` after
`run.log_artifact(artifact)`; do not pass an unsupported `wait` argument to
`log_artifact`.

W&B Tables are sufficient for per-case rows. `wandb-workspaces` and programmatic
W&B Reports are not required for the first implementation; reports can use the
W&B UI after data is correctly logged.

### 8. Comparison protocol

The first study predeclares:

- the exact manifest and held-out split;
- preprocessing, normalization, augmentation, loss, optimizer, scheduler,
  epoch budget, and precision;
- GPU/software environment;
- physical and effective batch sizes;
- checkpoint selection rule;
- inference ROI/spacing/overlap and warmup/repetition policy;
- whether timing includes data transfer, preprocessing, and postprocessing;
- unsupported-FLOP handling;
- per-case statistical aggregation.

If physical batch differs because of capacity, either use a common effective
batch through gradient accumulation or label the physical-batch difference as
part of the comparison. Never present physical-batch throughput as an isolated
architecture property.

Use paired per-case comparisons on identical cases/folds. Wilcoxon signed-rank
is a reasonable non-parametric option for pairwise quality comparisons, with
multiple-comparison correction when several variants are compared. Report mean,
standard deviation, and case-level distributions rather than only one mean.

## Failure handling

- W&B disabled/offline: training and benchmark continue with local JSON
  artifacts; no metric silently disappears from the local evidence path.
- W&B online failure: preserve local artifacts, mark/log tracking failure, and
  fail the online smoke; do not discard a completed computation.
- FLOP counter unsupported: preserve parameters and timing, record unsupported
  operations, and set FLOP result unavailable/partial.
- CUDA unavailable: use CPU timing only when the protocol explicitly permits
  it; never combine CPU and CUDA results in one comparison table.
- OOM during a batch sweep: record the largest passing batch and first failing
  boundary; do not continue in a contaminated process without an explicit
  isolation policy.
- Non-finite quality metric: serialize as `null` and retain exclusion count.
- Benchmark failure: training run remains valid if its training/evaluation
  evidence completed; benchmark run is marked failed and linked to its source.
- THOP or fvcore failure/unsupported operator: preserve parameters, timing,
  memory, and whichever counter succeeded; set failed counter to `null`, retain
  tool/error/unsupported status, and never report zero as a measurement.
- NVML unavailable or permission denied: preserve all non-power measurements,
  set `power/status` to `unavailable` with a bounded reason, and do not fail
  training or benchmarking solely for missing board-power telemetry.
- Snapshot feature disabled, tracking disabled, or `tracking.log_images=false`:
  skip snapshot loader iteration, inference, plotting imports, image
  construction, and W&B calls. A separately enabled local-visualization path
  may still write PNGs without W&B.
- Offline image logging: write snapshots to the local W&B offline directory and
  local snapshot directory without network access. Online image logging:
  forward the same derived image through the W&B adapter. Snapshot rendering or
  logging errors preserve numeric training/quality evidence, record a bounded
  local failure, and never mask the original training exception.

## Testing and acceptance

### Unit and integration coverage

Extend existing seams rather than creating a second pipeline:

- `tests/training/test_engine.py`: epoch timing, throughput fields, peak-memory
  reset contract, NVML epoch power fields, snapshot callback cadence and
  absolute-epoch behavior, resume/cumulative timing behavior.
- `tests/training/test_tracking.py`: scalar forwarding, summaries, custom axes,
  disabled/offline behavior, image gating/lazy construction, and
  artifact/table no-op behavior.
- `tests/training/test_artifacts.py`: efficiency/provenance serialization and
  non-finite normalization.
- `tests/evaluation/test_inference.py`: synchronized timing seam, fixed input
  reuse, sliding-window protocol fields, snapshot prediction collection,
  case-row shape, and power-boundary fields.
- `tests/evaluation/test_visualization.py`: all-region ET/TC/WT overlays,
  deterministic slice/montage selection, channel `3`, and 2-D/3-D shape
  handling.
- `tests/evaluation/test_efficiency.py`: THOP MACs, fvcore FLOPs, unsupported
  operator reporting, NVML sampling/integration, unavailable hardware, and
  OOM normalization.
- baseline/MetaUNETR pipeline tests: best-checkpoint restore precedes test and
  benchmark measurement; snapshot callback wiring and best/final ordering.
- `tests/training/test_engine_resume.py`: resumed runs do not duplicate
  scheduled, best, or final snapshots.
- `tests/test_cli_config.py` and synthetic integration: snapshot defaults,
  disabled/offline gates, local output paths, image keys, and all-region schema.
- synthetic integration test: validates schema and wiring only; it must not be
  interpreted as a performance result.

### Pre-full-run gates

The first full run is blocked until all gates pass:

1. Existing full suite remains green.
2. Config/provenance contains model, data, hardware, software, Git, and
   protocol identity.
3. Training run logs train/validation history, configured segmentation snapshot
   images, and held-out quality before `finish()`; disabled/offline behavior is
   verified according to mode.
4. Separate benchmark command produces local JSON for one restored checkpoint.
5. Online W&B debug smoke succeeds against
   `aniekanetimudo/token-mixer-placement-matters` and creates no old-project
   linkage.
6. W&B run contains expected config, summary, history, checkpoint artifact, and
   train/validation segmentation image keys when cloud image logging is enabled.
7. Benchmark run links to the training run/artifact and contains static cost,
   latency, throughput, memory, and protocol fields.
8. Case-level table contains no sensitive identifiers.
9. Same-protocol results from two variants render as comparable rows.

## Implementation sequence

1. Add the efficiency data contract and pure measurement helpers.
2. Add and lock `ultralytics-thop==2.1.6`, `fvcore==0.1.5.post20221221`, and
   `nvidia-ml-py==13.610.43`; validate `import thop`, `import fvcore`, and
   `import pynvml`, plus fixed-input profiling conventions.
3. Make cloud tracking online by default and make authentication accept the
   verified `.netrc` credential source without exposing secrets.
4. Add low-overhead training telemetry, NVML sampling, and correct tracker
   lifecycle.
5. Add tracker-controlled train/validation segmentation snapshots, local
   visualization evidence, and absolute-epoch/best/final cadence tests.
6. Add benchmark CLI/config and model-level plus case-level protocols.
7. Extend local artifact and W&B tracking boundaries.
8. Add focused tests and synthetic integration assertions.
9. Run the existing suite and debug configuration.
10. Run the cheap online W&B smoke with explicit entity/project overrides and
   verify live epoch charts.
11. Benchmark one restored model and inspect local/W&B evidence, including MAC,
     FLOP, and NVML fields.
12. Obtain explicit approval before starting the full model matrix.

No full training or paid compute extension is part of this design approval.
Adding and locking `ultralytics-thop`, `fvcore`, and `nvidia-ml-py` is part of
the approved core implementation. Report tooling remains unnecessary for the
first study.
