# Computational Efficiency Metrics and W&B Benchmarking

**Status:** Design approved in conversation; implementation plan pending user review of this spec
**Date:** 2026-09-12

## Summary

Add reproducible computational-efficiency measurement to the existing
Token-Mixer-Placement-Matters workflow without making benchmark timing part of
normal training or introducing nnUZoo as a dependency.

The design has two complementary paths:

1. **Training telemetry:** low-overhead epoch/phase timing, throughput, and peak
   allocator memory recorded during training.
2. **Explicit benchmark run:** a separate command that restores a selected
   checkpoint and measures static model cost, synchronized inference latency,
   throughput, peak memory, and fixed-protocol case-level inference.

Training and benchmark W&B runs are linked through a stable study group, source
run ID, and checkpoint artifact. Local JSON remains the durable evidence path;
W&B provides searchable histories, summaries, tables, and artifacts when online
tracking is enabled.

Carbon/emissions estimates are excluded. Board-power/energy sampling and
profiler traces remain optional follow-up work and do not gate the first full
study.

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
  allocator memory. Add one optional FLOP counter only after dependency
  approval; prefer `fvcore` with explicit unsupported-operator reporting.
  Never combine multiple counters or silently treat unsupported custom Mamba
  operations as zero FLOPs.
- Exclude CodeCarbon and carbon-equivalent estimates from the first study.
  Optional NVML board-power/energy measurement may be added later as a clearly
  labeled measurement boundary.
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
The first online debug smoke must verify project creation/write access.

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
- run one optional FLOP/MAC counter and retain its tool/version and unsupported
  operator list;
- normalize OOM, unsupported operations, and unavailable hardware fields into
  explicit status fields rather than silently dropping them.

The module must not allocate random inputs inside a timed model-only loop.
Inputs are prepared once, moved to the target device before timing, and reused.
End-to-end measurements may include preparation/transfer only when the boundary
is explicitly named as end-to-end.

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
- `time_to_best_seconds` uses the existing validation-selected best checkpoint.
  A time-to-target field is only added if a target is declared before training.
- MetaUNETR records distinct phase timing when its episodic/adaptation phases
  are observable; the aggregate must not hide those phases.

Data-loader versus compute timing is optional diagnostic instrumentation. It is
not required on every run because fine-grained synchronization adds overhead.

Training telemetry is appended to existing epoch history and local artifacts.
It must not alter model, optimizer, RNG, or loader checkpoint semantics. If
resuming, distinguish per-process segment time from cumulative training time;
do not reconstruct elapsed time from a stale `perf_counter` value.

### 3. Static model-cost measurement

Run after model construction and before inference benchmarking:

```text
model/parameters
model/trainable_parameters
model/macs
model/flops
model/flop_tool
model/unsupported_ops
model/checkpoint_bytes
```

The exact composed input shape, channel count, ROI/patch shape, and precision
are part of the record. Parameters are authoritative. FLOPs/MACs are only
authoritative for supported operators; an unsupported custom Mamba/SSM path
must produce a partial/unavailable status and explanatory list.

Static values belong primarily in the benchmark summary because they do not
change each epoch. They may be copied into training summary for convenient
filtering.

### 4. Explicit inference benchmark

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

#### Protocol families

- `native_3d_full_volume`: existing MONAI sliding-window path.
- `transunet_2d_slice`: direct slice path, aggregated over a declared case's
  slices when case-level latency is reported.
- `cnn_denoising_validation`: CNN pretraining task, not segmentation quality.

The benchmark records protocol family in every row and never merges these
families into one throughput chart.

### 5. Quality metrics and per-case evidence

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

No raw medical data, PHI, images, or unapproved patient identifiers are sent
to W&B.

### 6. W&B lifecycle and schema

Extend `training/tracking.py` as the only W&B boundary. Pipelines must not call
the W&B SDK directly. The abstraction preserves disabled and offline no-op
behavior while adding optional summary, table, and artifact operations.

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
- protocol and intentional model-specific deviations.

The training run remains open through:

```text
training
→ best-checkpoint restore
→ held-out quality evaluation
→ summary/provenance/checkpoint artifact logging
→ finish
```

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

### 7. Comparison protocol

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

## Testing and acceptance

### Unit and integration coverage

Extend existing seams rather than creating a second pipeline:

- `tests/training/test_engine.py`: epoch timing, throughput fields, peak-memory
  reset contract, resume/cumulative timing behavior.
- `tests/training/test_tracking.py`: scalar forwarding, summaries, custom axes,
  disabled/offline behavior, and artifact/table no-op behavior.
- `tests/training/test_artifacts.py`: efficiency/provenance serialization and
  non-finite normalization.
- `tests/evaluation/test_inference.py`: synchronized timing seam, fixed input
  reuse, sliding-window protocol fields, and case-row shape.
- baseline/MetaUNETR pipeline tests: best-checkpoint restore precedes test and
  benchmark measurement.
- synthetic integration test: validates schema and wiring only; it must not be
  interpreted as a performance result.

### Pre-full-run gates

The first full run is blocked until all gates pass:

1. Existing full suite remains green.
2. Config/provenance contains model, data, hardware, software, Git, and
   protocol identity.
3. Training run logs train/validation history and held-out quality before
   `finish()`.
4. Separate benchmark command produces local JSON for one restored checkpoint.
5. Online W&B debug smoke succeeds against
   `aniekanetimudo/token-mixer-placement-matters` and creates no old-project
   linkage.
6. W&B run contains expected config, summary, history, and checkpoint artifact.
7. Benchmark run links to the training run/artifact and contains static cost,
   latency, throughput, memory, and protocol fields.
8. Case-level table contains no sensitive identifiers.
9. Same-protocol results from two variants render as comparable rows.

## Implementation sequence

1. Add the efficiency data contract and pure measurement helpers.
2. Add low-overhead training telemetry and correct tracker lifecycle.
3. Add benchmark CLI/config and model-level plus case-level protocols.
4. Extend local artifact and W&B tracking boundaries.
5. Add focused tests and synthetic integration assertions.
6. Run the existing suite and debug configuration.
7. Run the cheap online W&B smoke with explicit entity/project overrides.
8. Benchmark one restored model and inspect local/W&B evidence.
9. Obtain explicit approval before starting the full model matrix.

No full training, paid compute extension, or broad dependency installation is
part of this design approval. Adding `fvcore`, NVML bindings, or report tooling
requires separate dependency approval.
