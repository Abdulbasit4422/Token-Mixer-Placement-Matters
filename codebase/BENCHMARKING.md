# Benchmarking And Efficiency Evidence

This guide owns computational-efficiency evidence, benchmark protocol identity,
and its local/W&B lineage. Model construction belongs to the pipeline guides;
training semantics belong to [TRAINING.md](TRAINING.md); metric formulas and
inference boundaries belong to [EVALUATE.md](EVALUATE.md).

## Purpose

Benchmarking is an explicit, separate measurement run. It does not change the
training loop, split manifest, optimizer, or quality metrics. Report accuracy,
latency, throughput, memory, parameters, MACs, FLOPs, and board power as
separate axes; never replace them with a composite efficiency score.

Synthetic and fixture checks prove wiring and JSON schema only. They are not
hardware measurements, convergence evidence, model-quality evidence, or a
ranking of experiments.

## Protocol Families

Every model-level and case-level row carries one exact protocol name. These
families are separate comparison lanes:

| Protocol | Applies to | Measurement unit |
| --- | --- | --- |
| `cnn_denoising_validation` | CNN/ImageFolder denoising pretraining | 2-D denoising validation image |
| `transunet_2d_slice` | TransUNet | 2-D BraTS slice |
| `native_3d_full_volume` | ResUNet3D, SwinUNETR, and MetaUNETR variants (`metaunetr_mamba`, `mod_a`, `mod_b`) | Native 3-D full volume |

CNN output is denoising validation, not BraTS segmentation quality. TransUNet
uses its 2-D slice path. Native baselines and MetaUNETR use full-volume
sliding-window inference. A 2-D slice must not be compared with one 3-D
patient volume as if they were the same observation.

## Benchmark Source Contract

`configs/benchmark.yaml` intentionally leaves both source fields `null` for
safe composition inspection. `run_benchmark` requires exactly one source before
building a measurement result:

| Source | Contract | Persisted lineage |
| --- | --- | --- |
| `benchmark.source_checkpoint` | Local file, or directory resolved to `best.pt`; checkpoint metadata is validated with `CheckpointManager.load_model`. | Selected checkpoint path in benchmark provenance. |
| `benchmark.source_artifact` | Immutable W&B reference with a version such as `entity/project/name:v0`, or a supported digest such as `entity/project/name@sha256:<64-hex-digits>`. Mutable aliases (`:latest`, `:production`) are rejected. | Immutable artifact reference in `source_artifact` and `source_checkpoint_artifact`. |

`benchmark.source_train_run_id` is optional linkage metadata; it never selects
or replaces the source checkpoint. Remote artifact restoration uses the active
`Tracker` boundary, so disabled tracking can run local-source benchmarks but
cannot restore a remote artifact. A restored directory must expose `best.pt`.
The source is restored before model-level and case-level measurements, and
checkpoint restoration is excluded from every reported latency/power boundary.

## Model-Level Measurement

`token_mixer.evaluation.benchmark.run_model_protocol` prepares one fixed input
before measurement and passes it to static counters and repeated forward
measurement. Random input allocation is outside timed loops. Checkpoint
restoration and static file I/O are outside the model-inference timing scope.

Static fields are namespaced under `model/`:

- `model/parameters`, `model/trainable_parameters`, and
  `model/checkpoint_bytes`;
- `model/macs`, `model/mac_tool`, `model/mac_tool_version`,
  `model/mac_convention`, and `model/mac_status` from THOP;
- `model/flops`, `model/flop_tool`, `model/flop_tool_version`,
  `model/flop_convention`, and `model/flop_status` from fvcore;
- `model/unsupported_ops` and `model/uncalled_modules` for counter diagnostics.

THOP MACs and fvcore FLOPs use their own documented conventions. They are
reported independently on the same input; one count is never substituted for
the other. Repeated forward fields include latency mean, median, p95, standard
deviation, input shape, batch size, warmup count, repetition count, throughput,
and allocator peaks under `inference/`.

The project pins `ultralytics-thop` (import `thop`) and `fvcore`, but operator
support remains model-dependent. Each counter has independent `ok`, `partial`,
or `unavailable` status, bounded error text, tool/version, convention, and
unsupported-operator diagnostics. A partial numeric count is retained when
available; an unavailable MAC/FLOP value is `null`, never zero. Parameters are
always counted natively and remain authoritative. `nvidia-ml-py`/`pynvml` is
similarly best-effort: missing driver, permission, package, or sampling support
produces `power/status: unavailable` while preserving non-power measurements.

CPU timing uses `perf_counter`. CUDA timing synchronizes only at measurement
boundaries and uses CUDA events. Warmups are excluded from latency samples and
allocator peaks. NVML sampling covers the named forward/batch-sweep section,
not checkpoint restoration or static counting. `power/average_watts`,
`power/max_watts`, `power/energy_joules`, `power/sample_count`,
`power/sample_interval_ms`, and `power/status` remain separate from allocator
memory fields. CPU allocator peaks are `null`; they are not reported as zero.

The benchmark output combines these measurements without replacing one metric
with another. Model rows retain `largest_passing_batch`,
`first_failing_batch`, and `inference/sweep_status` for the declared physical
batch sweep. An OOM row is the first failing boundary; unrelated runtime errors
are not converted to benchmark status.

Training has a separate epoch-boundary telemetry contract. Each completed
epoch appends exactly one local/W&B scalar record with `train/epoch` as its
explicit absolute epoch axis. Records retain `train/epoch_seconds`, optimizer
steps, observed samples and voxels, samples/voxels per second, allocator
peaks, `run/elapsed_seconds`, and `power/*` fields. Validation adds
`val/epoch` and `val/epoch_seconds` only when validation runs. A new best adds
`train/time_to_best_seconds`; its scope is recorded. `log_every_steps` and
snapshot cadence do not suppress epoch records. Training's NVML sampler starts
after setup/checkpoint restoration and runs across the fit; epoch power fields
are snapshots of that sampler's cumulative samples, not isolated model-energy
attribution or per-epoch resets. See [TRAINING.md](TRAINING.md) for the engine
lifecycle.

## Case-Level Measurement

Case measurements reuse the declared protocol evaluator and keep model compute
separate from end-to-end work. Available boundaries are:

`load_seconds` → `preprocess_seconds` → `model_compute_seconds` →
`postprocess_seconds` → `metric_seconds` → `end_to_end_seconds`.

Checkpoint restoration is excluded from per-case latency. Native 3-D rows
record sliding-window count, voxel count, validated spacing, Dice/HD95 by
`ET`, `TC`, and `WT`, and exclusion flags. TransUNet rows use declared slice
aggregation; CNN rows use denoising validation fields and do not acquire
BraTS-region quality fields.

Serialized case rows contain `case_id_hash`, never raw case IDs. Captions,
tables, local JSON, and W&B payloads must not contain raw identifiers, PHI, or
medical volumes. `case_id_hash` is a fixed SHA-256-derived key used only for
de-identified grouping: current serializer uses the first 16 hexadecimal
characters of SHA-256 over the UTF-8 case-ID text. Generic case fields and
aliases (`case_id`, `case_ids`, `id`, and plural/camel-case forms) are redacted
recursively; existing hash fields are preserved. Non-finite HD95 values
serialize as `null` while exclusion counts remain visible. Raw IDs may exist in
the in-memory evaluator result before serialization, but never in the durable
benchmark artifact or tracker payload.

## W&B Linkage

W&B SDK calls stay inside `token_mixer.training.tracking`. Training cloud
defaults are:

```yaml
project: token-mixer-placement-matters
entity: aniekanetimudo
group: token-mixer-brats-seed42
job_type: train
mode: online
```

Benchmark runs use the same approved project/group but `job_type: benchmark`.
They carry `protocol_id`, `source_train_run_id` when supplied, and an
immutable `source_checkpoint_artifact` or local source-checkpoint identity.
Benchmark repetition steps are independent of training `global_step`.

Online initialization accepts `WANDB_API_KEY` or a matching credential entry
from the standard `.netrc` file for known W&B hosts. This is an availability
check only; the credential is never printed, put in composed config, or copied
into provenance. Offline mode initializes a local W&B run without a credential
or network connection. Disabled mode returns a no-op tracker without importing
W&B. `tracking.log_images` is an additional image gate, independent of scalar,
table, summary, and artifact operations.

Training history uses `train/*` and `val/*` namespaces. W&B metric definitions
plot those namespaces against `train/epoch` and `val/epoch`; the underlying
global step remains monotonic. Power uses `power/*`, snapshots use
`segmentation/train/*` and `segmentation/val/*`, and benchmark summaries use
`model/*`, `inference/*`, and `power/*`.

Successful training finalization keeps its active tracker open through
best-checkpoint restore, held-out/final validation, local artifact writing,
summary/table logging, and model-artifact upload. The model artifact contains
`config.yaml`, `metrics.json`, `provenance.json`, and `best.pt` when present,
with `best` and `latest` aliases; `artifact.wait()` completes the upload before
the pipeline finishes. The benchmark writes `benchmark.json` before
summary/table/artifact logging and finishes its separate run afterward. Local
disabled tracking still writes the JSON/checkpoint evidence but performs no
remote upload.

## Artifact And Provenance Contract

Training completion artifacts use schema version `2`:

| File | Durable contents |
| --- | --- |
| `config.yaml` | Composed Hydra configuration saved before dispatch, with secrets excluded by configuration practice. |
| `metrics.json` | Legacy `best_epoch`, `best_metric`, `history`, and `test_metrics` keys plus `timing`, `efficiency`, and `early_stopping` sections. Non-finite numbers become `null`. |
| `provenance.json` | `status`, code/experiment/model identity, seed/device, manifest hash, monitor/direction, protocol/evaluation split, source checkpoint, nested timing/efficiency, metadata, tracking configuration, and redacted case references. |

Current training provenance stores `timing.timing_scope` as
`process_segment`; early-stop state is retained in provenance metadata under
`early_stopping/stopped`, `early_stopping/stop_epoch`, and
`early_stopping/best_epoch`, and in checkpoint state with underscored names.
Checkpoint payloads also retain phase/global-step/history, compatibility
metadata, optimizer/scheduler/scaler state when applicable, RNG state, and
DataLoader generator state. Git commit and detailed host/software identity are
not inferred automatically by these writers; supply or retain them separately
when a study requires them.

Benchmark completion uses schema version `1` and one atomic `benchmark.json`:

```json
{
  "schema_version": 1,
  "summary": {"model/*": "...", "inference/*": "...", "power/*": "..."},
  "rows": [{"row_type": "model", "protocol": "..."}, {"row_type": "case", "case_id_hash": "..."}],
  "provenance": {"protocol": "...", "source_checkpoint": "..."}
}
```

The actual benchmark provenance additionally records input shape, warmups,
repetitions, physical batch sizes, manifest/split identity, source train run ID
when supplied, and whether checkpoint restoration was excluded from timing.
The benchmark artifact can be logged as a separate W&B artifact; its immutable
reference, not a mutable alias, is the source for downstream reruns.

Online authentication accepts `WANDB_API_KEY` or a matching standard `.netrc`
entry for a W&B host. Credential values are never printed or serialized.
Offline mode needs no credential and writes local W&B files without network
access. Disabled tracking creates a no-op tracker without importing W&B; it
also performs no W&B image construction. `tracking.log_images` independently
gates W&B images.

## Local Evidence

Tracking is disabled by default in the local profile, but local evidence still
contains successful `config.yaml`, `metrics.json`, `provenance.json`, and
checkpoint files under the configured run output. `metrics.json` retains the
legacy best/history/test keys plus schema-versioned timing, efficiency, power,
and early-stopping fields. `provenance.json` records protocol identity,
`timing.timing_scope: process_segment`, and early-stopping state in its metadata.
The metrics early-stopping object and checkpoint state retain the same stopped,
stop-epoch, and best-epoch values.

Segmentation snapshots use fixed deterministic train/validation examples when
enabled. `snapshot_interval_epochs: 10` is an absolute cadence (`10`, `20`,
`30`, ...), independent of scalar logging. A newly selected best snapshot is
also retained when its epoch is not an interval; a final snapshot uses restored
best weights when available. Snapshot keys are stable and include both train
and validation namespaces. Explicit `local_enabled: true` may save local PNGs
without W&B; if both W&B image logging and local output are disabled, no
snapshot loader iteration, inference, plotting import, or image construction
occurs.

`benchmark.json` is one atomic JSON file with `schema_version`, `summary`,
`rows`, and `provenance`. A local or offline run remains valid without network
tracking. If online logging fails after local serialization, local JSON and
provenance remain the primary evidence and the tracking failure is reported;
it must not silently redirect to another W&B project.

For a local-source smoke, disable tracking explicitly and provide one existing
checkpoint; for an artifact-source smoke, keep tracking active so the tracker
can restore the immutable reference. Neither smoke changes the source run or
constitutes hardware-performance evidence.

## Failure Semantics

- **THOP/fvcore:** `ok`, `partial`, or `unavailable` status is recorded per
  counter. Unsupported operators and errors are retained. A partial numeric
  count is kept when available; unavailable counts serialize as `null`, never
  zero.
- **NVML:** `power/status: unavailable` with a bounded reason represents a
  missing driver/package, permission failure, or sampling failure. CPU and
  synthetic fixture paths are explicitly non-hardware evidence. Missing board
  power never masks timing, profiling, or training results. Training may also
  record `power/status: disabled` when NVML is not configured.
- **OOM:** batch sweeps record the largest passing batch and first failing
  batch, mark sweep status `oom`, clean the CUDA cache when possible, and stop
  at the first OOM. An unrelated `RuntimeError` is re-raised.
- **Non-finite values:** JSON converts non-finite floats to `null`; it does not
  turn them into quality or efficiency zeros.
- **Snapshots/tracking:** disabled feature gates do no image work. Snapshot
  failures are bounded in local provenance and do not replace numeric training
  or evaluation errors. Tracker cleanup does not replace the original failure.

## Safe Smoke-Check Boundaries

Keep these checks distinct:

- `--cfg job` composes YAML only. It does not build a model, inspect data, or
  dispatch training/benchmarking.
- `tests/integration/test_synthetic_debug.py` uses temporary synthetic fixtures
  and schema/wiring assertions. It does not establish BraTS quality, convergence,
  hardware timing, NVML readings, or W&B write access.
- A local benchmark smoke measures one restored checkpoint and one declared
  protocol only. It is not a benchmark matrix and must not compare incompatible
  2-D/3-D or denoising/segmentation rows.
- An online cloud smoke, when separately approved, is debug-sized and verifies
  live epoch history, finalization order, and approved project linkage. It is
  not full training. Verify repository `PWD`, keep credentials outside config,
  and preserve local artifacts if online logging fails.

## Pre-full-run Gate

Run repository-relative checks before any cloud or full-data command:

```bash
uv lock --check
uv run pytest tests/integration/test_synthetic_debug.py -q
uv run python -m compileall -q src tests
uv run python -m token_mixer --config-name cloud experiment=mod_a run=debug --cfg job
git diff --check
```

Inspect composed cloud output for `device: cuda`, approved W&B destination,
`job_type: train`, snapshot gates, early-stopping settings, manifest inputs, and
absence of credential values. Epoch-axis behavior is a runtime tracker contract
and is checked by the synthetic/unit seams, not claimed from YAML composition
alone. After those checks, a separately approved cheap online debug smoke may
verify lifecycle, local fallback, and W&B linkage. A restored-model benchmark
must use one explicit local checkpoint or immutable artifact and a declared
protocol; it must remain separate from training.

Do not launch the full model/seed matrix from this guide. Full cloud training,
full benchmarking, paid compute, and any matrix expansion require explicit
approval specifying models, seeds, data, and expected compute first.
