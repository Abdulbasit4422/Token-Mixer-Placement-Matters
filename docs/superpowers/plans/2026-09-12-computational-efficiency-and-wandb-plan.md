# Computational Efficiency Metrics and W&B Benchmarking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reproducible training telemetry recorded against every epoch, an explicitly separate inference benchmark, durable local evidence, tracker-gated train/validation segmentation snapshots, configurable early stopping, and mandatory online W&B tracking for cloud training without changing model or split semantics.

**Architecture:** Keep `training.engine.fit` model-independent and add low-overhead epoch-boundary telemetry there. Put pure timing, parameter, memory, and normalization helpers under `evaluation/efficiency.py`; keep all W&B SDK calls inside `training/tracking.py`. Add a Hydra-selected benchmark pipeline that restores a local checkpoint or pinned W&B artifact, runs declared model-level and protocol-specific case-level measurements, writes one benchmark JSON, and logs a separate linked W&B benchmark run.

**Tech Stack:** Python 3.12, PyTorch, CUDA events/allocator APIs, Hydra/OmegaConf, MONAI sliding-window inference, `ultralytics-thop==2.1.6` (import `thop`) for MACs, `fvcore==0.1.5.post20221221` for FLOPs, `nvidia-ml-py==13.610.43`/`pynvml` for NVML power, pytest, W&B SDK `0.29.0`, JSON, existing `uv` environment.

## Global Constraints

- Do not start full training or paid compute as part of implementation; request explicit approval after all pre-full-run gates pass. The user-approved THOP, fvcore, and NVML dependency additions are part of this implementation and must be locked before source changes depend on them.
- Cloud training defaults to `entity=aniekanetimudo`, `project=token-mixer-placement-matters`, `mode=online`, and `job_type=train`; local tracking remains disabled by default.
- Scalar training metrics are recorded after every completed epoch and logged against an explicit epoch field for W&B charts; `log_every_steps` and snapshot cadence never suppress or batch those epoch records. Validation metrics retain existing `validation_interval` semantics and include their absolute epoch.
- Segmentation snapshots use `snapshot_interval_epochs: 10`, producing images at absolute epochs `10, 20, 30, 40, ...`, plus a best snapshot whenever validation selects a new best (for example, interval image at 100 and best image at 104) and a final snapshot when available. If snapshotting is disabled, no snapshot work occurs; if W&B image logging is disabled, no W&B import/image construction occurs, while explicit `local_enabled` may still produce local-only PNGs. Offline W&B uses local files without network; online W&B forwards images after all gates pass.
- Accept W&B credentials from `WANDB_API_KEY` or the standard `.netrc` credential source without printing or serializing secrets; never use unrelated project `unet_brain-tumor-segmentation_1`.
- All W&B SDK calls remain in `src/token_mixer/training/tracking.py`; pipelines consume `Tracker` methods only.
- Preserve existing model, optimizer, RNG, DataLoader, checkpoint, split, metric, resume, and warm-start semantics. Add telemetry fields; do not replace existing history fields.
- Do not synchronize CUDA on every training batch. Synchronize only at epoch/measurement boundaries and use `torch.cuda.Event` for CUDA timing.
- Prepare and move model-only benchmark inputs before the timed loop; never allocate random inputs inside a timed loop.
- Record timing boundary, protocol family, input shape, batch size, precision, warmups, repetitions, hardware/software, and OOM/unsupported statuses explicitly.
- Keep `native_3d_full_volume`, `transunet_2d_slice`, and `cnn_denoising_validation` in separate comparison rows and charts; never compare one 2-D slice with one 3-D patient volume.
- Report accuracy, training time, latency, throughput, memory, parameters, and FLOP status as separate axes; do not create a composite efficiency score.
- Do not add nnUZoo, CodeCarbon, or profiler tooling beyond the approved THOP/fvcore counters. Add exactly the `ultralytics-thop`, `fvcore`, and `nvidia-ml-py` distributions; import `thop`, `fvcore`, and `pynvml`; do not install legacy `thop` alongside `ultralytics-thop` or duplicate NVML bindings. Measure MACs, FLOPs, and NVML board power/energy in this implementation, with explicit partial/unavailable status when a tool, operator, GPU, or permission is unavailable.
- Non-finite HD95 values serialize as `null` and retain exclusion counts; never convert them to zero or silently omit them.
- Case-level serialized rows contain hashed case IDs only. Never send raw case IDs, raw medical volumes, PHI, or other sensitive identifiers to W&B. Derived de-identified ET/TC/WT snapshot previews may be logged only when snapshot and image gates permit them; captions contain hashes only.
- Benchmark OOM handling records largest passing and first failing batch, stops the sweep after the first OOM, and re-raises unrelated runtime errors.
- Edit `.py` sources, not paired notebooks. If a notebook pair is ever changed, sync only with `jupytext --sync <file>.py`.
- Before any cloud command, verify `PWD`; use repository-relative paths, stay inside the workspace, and do not access or create files outside it. Do not touch untracked `NUL`.
- Do not commit or push implementation changes unless user explicitly requests it. Inspect `git status`, `git diff`, and `git diff --check` before reporting work.
- Every task below has a focused test command. Final evidence must include `uv lock --check`, `uv run pytest -q -rs`, `uv run python -m compileall -q src tests`, `git diff --check`, the online W&B smoke result, and one inspected real-model benchmark result.

## Evidence Path and Ownership

| Claim | Primary owner | Minimum evidence |
| --- | --- | --- |
| Timing, MAC, FLOP, and power values are normalized correctly | `evaluation/efficiency.py` | CPU unit tests plus CUDA/profiler/NVML seam tests that verify synchronization/event paths, fixed input identity, operator status, and energy integration |
| Training telemetry does not change training semantics | `training/engine.py` | Existing engine/resume tests plus new field/step assertions and unchanged full-suite result |
| Held-out quality is visible before training W&B finish | pipeline lifecycle helper | Fake tracker event ordering: `fit → restore best → test → artifact/summary → finish` |
| Cloud profile reaches intended W&B destination | tracking/config smoke | Config composition, mocked auth-source tests, and one cheap online run checked in `aniekanetimudo/token-mixer-placement-matters` |
| Benchmark is separate and reproducible | benchmark pipeline | Local JSON schema test, restored-checkpoint integration test, separate `job_type=benchmark`, linked source run/artifact fields |
| Case rows are safe and comparable | inference/benchmark serializer | Hash/no-raw-ID test, protocol-family test, same-protocol synthetic comparison rows |

## File and Ownership Map

| Area | Files | Responsibility |
| --- | --- | --- |
| Pure measurements | Create `src/token_mixer/evaluation/efficiency.py`; test `tests/evaluation/test_efficiency.py` | Timing summaries, parameter counts, allocator peaks, THOP MACs, fvcore FLOPs, NVML power/energy, static-cost fields, model-only latency, throughput/OOM boundaries |
| W&B boundary | Modify `src/token_mixer/training/tracking.py`; test `tests/training/test_tracking.py` | Credential discovery, run metadata, scalar/summary/table/artifact forwarding, artifact restore, no-op behavior |
| Training telemetry | Modify `src/token_mixer/training/engine.py`; test `tests/training/test_engine.py` and `tests/training/test_engine_resume.py` | Epoch/phase timing, observed samples/voxels, peak allocator memory, elapsed/time-to-best fields, tracker finalization control |
| Training completion | Modify `src/token_mixer/pipelines/_baseline_common.py`, `src/token_mixer/pipelines/train_metaunetr.py`, `src/token_mixer/pipelines/pretrain_cnn.py`, `src/token_mixer/training/artifacts.py`; extend pipeline tests | Keep tracker open through best restore, held-out/validation quality, local JSON, W&B artifact/summary, and finish |
| Case inference | Modify `src/token_mixer/evaluation/inference.py`, `src/token_mixer/pipelines/_baseline_common.py`, and `src/token_mixer/data/datasets.py` only if case metadata is required; test evaluation/inference and pipeline suites | Optional case records, protocol boundaries, sliding-window counts, hashed identifiers, 2-D case/slice grouping |
| Benchmark orchestration | Create `src/token_mixer/evaluation/benchmark.py` and `src/token_mixer/pipelines/benchmark.py`; tests `tests/evaluation/test_benchmark.py` and `tests/pipelines/test_benchmark.py` | Benchmark result contract, model construction/restoration, protocol selection, local JSON, separate linked tracker run |
| CLI/config | Modify `src/token_mixer/cli.py`; create `configs/benchmark.yaml`; modify `configs/cloud.yaml`, `README.md`, `codebase/CONFIG.md`, `codebase/TRAINING.md`, `codebase/EVALUATE.md`, `codebase/REPRODUCIBILITY.md` | Hydra dispatch, online cloud defaults, benchmark invocation, operator/researcher documentation |

The efficiency and tracking contracts are prerequisites. Engine telemetry depends on efficiency helpers. Pipeline lifecycle depends on both. Case inference and benchmark runner can then proceed in parallel because they have disjoint implementation files; final integration waits for both.

---

### Task 1: Add measurement dependencies and efficiency contract

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `src/token_mixer/evaluation/efficiency.py`
- Create: `tests/evaluation/test_efficiency.py`
- Modify: `src/token_mixer/evaluation/__init__.py` only if the package currently exports evaluation symbols there

**Interfaces:**
- Adds the user-approved runtime distributions `ultralytics-thop==2.1.6`, `fvcore==0.1.5.post20221221`, and `nvidia-ml-py==13.610.43`; import them as `thop`, `fvcore`, and `pynvml`, and add no legacy `thop` or second NVML binding.
- Produces `TimingSummary`, `PowerStats`, `NvmlPowerSampler`, `summarize_timings`, `model_parameter_counts`, `checkpoint_size_bytes`, `reset_peak_memory`, `peak_memory_gb`, `measure_forward`, `measure_batch_sweep`, and `static_model_cost`.
- `measure_forward(model, inputs, *, warmup_iterations, repetitions, protocol) -> dict[str, Any]` returns status, latency summary fields, memory fields, input shape, batch size, warmup count, repetition count, and protocol.
- `measure_batch_sweep(model, input_factory, batch_sizes, *, warmup_iterations, repetitions, protocol) -> dict[str, Any]` returns per-batch rows plus `largest_passing_batch`, `first_failing_batch`, and sweep status.
- `static_model_cost(model, *, inputs, checkpoint_path=None) -> dict[str, Any]` runs THOP and fvcore against the same fixed input and returns parameter counts, checkpoint bytes, MAC/FLOP values, tool names/versions, conventions, unsupported operators, and independent status fields.
- `NvmlPowerSampler(device_index: int, interval_seconds: float)` exposes `start()`, `snapshot() -> PowerStats`, and `stop() -> PowerStats`; it is a best-effort board-level sampler that never fails unrelated measurements.

- [ ] **Step 1: Add the approved dependencies and lock them.** Add `ultralytics-thop==2.1.6`, `fvcore==0.1.5.post20221221`, and `nvidia-ml-py==13.610.43` to project runtime dependencies; record exact resolved versions in `uv.lock`, verify `import thop`, `import fvcore`, and `import pynvml`, and do not add legacy `thop` or `pynvml` as a second package.

Run: `uv add ultralytics-thop==2.1.6 fvcore==0.1.5.post20221221 nvidia-ml-py==13.610.43 && uv lock && uv run python -c "import thop, fvcore, pynvml; print('efficiency dependencies imported')"`

Expected: `pyproject.toml` and `uv.lock` contain all three packages, imports succeed, and no unrelated dependency group changes appear.

- [ ] **Step 2: Write failing measurement tests.** Cover `summarize_timings([1.0, 2.0, 3.0])`, empty/non-finite rejection, total/trainable parameter counts, checkpoint byte count, CPU forward timing, fixed input object reuse across warmups/repetitions, invalid warmup/repetition rejection, THOP MACs, fvcore FLOPs, tool/version/convention fields, an unsupported custom operator with partial status, NVML trapezoidal integration, NVML unavailable status, and a sweep that records a simulated CUDA OOM without swallowing `RuntimeError("unrelated")`.

```python
def test_measure_forward_reuses_prepared_input_and_returns_schema():
    seen = []

    class Model(torch.nn.Module):
        def forward(self, value):
            seen.append(id(value))
            return value

    inputs = torch.ones(2, 3, 4, 4)
    result = measure_forward(Model(), inputs, warmup_iterations=2, repetitions=3, protocol="fixture")

    assert len(seen) == 5
    assert set(seen) == {id(inputs)}
    assert result["status"] == "ok"
    assert result["latency_mean_ms"] >= 0.0
    assert result["batch_size"] == 2
    assert result["input_shape"] == [2, 3, 4, 4]
    assert result["protocol"] == "fixture"
```

- [ ] **Step 3: Run focused tests and verify failure.**

Run: `uv run pytest tests/evaluation/test_efficiency.py -q`

Expected: collection or assertion failure because `token_mixer.evaluation.efficiency` does not yet provide the measurement contract.

- [ ] **Step 4: Implement measurement helpers.** Use `time.perf_counter()` on CPU; on CUDA call `torch.cuda.synchronize()` before timing, use start/end `torch.cuda.Event(enable_timing=True)`, call `end_event.synchronize()`, and read `start_event.elapsed_time(end_event)`. Set and restore `model.training` state around `eval()`/`inference_mode()`. Reset allocator stats immediately before each measured section and return `None` memory values on CPU. Compute p95 with `numpy.percentile`; reject empty/non-finite samples. Run `thop.profile(model, inputs=(inputs,), verbose=False)` for MACs and `fvcore.nn.FlopCountAnalysis(model, (inputs,))` for FLOPs on the same prepared tensor. Record `thop.__version__`, `fvcore.__version__`, THOP MAC convention, fvcore FLOP convention, `unsupported_ops()`, `uncalled_modules()`, and independent `mac_status`/`flop_status`; retain a partial numeric count when the counter supports one and use `null` plus an error/status when it does not. Catch only `torch.cuda.OutOfMemoryError` or an error whose message explicitly contains `out of memory`, record the boundary, call `torch.cuda.empty_cache()` when available, and stop the sweep.

- [ ] **Step 5: Implement NVML sampling.** Lazily import `pynvml`, initialize the selected device handle, sample `nvmlDeviceGetPowerUsage` in milliwatts on a daemon thread at the configured interval, convert to watts, and integrate samples by monotonic timestamp with the trapezoid rule. `snapshot()` reports average watts, max watts, joules, sample count, interval, device index, NVML version, and status. `stop()` always joins/cleans up and calls `nvmlShutdown()` when initialization succeeded. Catch NVML import, initialization, permission, and sampling exceptions into `status="unavailable"` with bounded reason; never print a key or fail timing/profiling because board power is unavailable.

```python
@dataclass(frozen=True)
class TimingSummary:
    mean_ms: float
    median_ms: float
    p95_ms: float
    std_ms: float
    samples: int


def model_parameter_counts(model: nn.Module) -> dict[str, int]:
    parameters = list(model.parameters())
    return {
        "parameters": sum(parameter.numel() for parameter in parameters),
        "trainable_parameters": sum(
            parameter.numel() for parameter in parameters if parameter.requires_grad
        ),
    }
```

- [ ] **Step 6: Run focused tests and inspect values.**

Run: `uv run pytest tests/evaluation/test_efficiency.py -q`

Expected: all efficiency contract tests pass; no test allocates random inputs inside `measure_forward`.

- [ ] **Step 7: Run static checks for this task.**

Run: `uv lock --check && uv run python -m compileall -q src/token_mixer/evaluation tests/evaluation/test_efficiency.py && git diff --check`

Expected: exit code `0`.

---

### Task 2: Extend W&B boundary and make cloud destination explicit

**Files:**
- Modify: `src/token_mixer/training/tracking.py`
- Modify: `tests/training/test_tracking.py`
- Modify: `configs/cloud.yaml`
- Preserve: `configs/local.yaml` disabled tracking values

**Interfaces:**
- `Tracker.log` accepts `Mapping[str, Any]` so optional `None` status fields can be retained locally without changing no-op behavior.
- Add `Tracker.log_table(name: str, columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> None`.
- Add lazy `Tracker.log_images(images: Mapping[str, Path | Any], *, step: int, captions: Mapping[str, str] | None = None) -> None`; disabled mode must not import W&B or construct image objects, offline mode writes to the local W&B run without network, and online mode forwards images to the configured W&B run.
- Add `Tracker.define_metric(name: str, *, step_metric: str | None = None) -> None`; the W&B adapter defines `train/*` against `train/epoch` and `val/*` against `val/epoch`, while the underlying `global_step` remains monotonic.
- Add `Tracker.log_artifact(name: str, files: Mapping[str, Path], *, artifact_type: str = "model", aliases: Sequence[str] = ()) -> str | None`.
- Add `Tracker.restore_artifact(reference: str, destination: Path) -> Path`.
- Add read-only `Tracker.run_id -> str | None`.
- `_WandbTracker` forwards `group`, `job_type`, and optional tags from config; `log_artifact` calls `artifact.wait()` after `run.log_artifact(artifact, aliases=...)` and never passes unsupported `wait=` to `log_artifact`.
- Online credential validation accepts either `WANDB_API_KEY` or a matching standard `.netrc` entry and emits an error naming accepted sources, never the credential value.

- [ ] **Step 1: Add failing tracking tests.** Extend fake W&B tests for `.netrc` acceptance, missing both credential sources, group/job type forwarding, `define_metric` epoch-axis forwarding, table construction, artifact file aliases, artifact wait ordering, artifact restore, run ID exposure, and disabled/offline no-op methods. Keep existing `WANDB_API_KEY` and disabled-import tests.

```python
def test_online_tracker_accepts_netrc_without_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setattr(tracking_module, "_wandb_netrc_has_credentials", lambda: True)
    fake_run = SimpleNamespace(id="abc123", log=lambda *args, **kwargs: None, finish=lambda: None)
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=lambda **_: fake_run))

    tracker = create_tracker(
        {"enabled": True, "mode": "online", "directory": str(tmp_path)},
        {"seed": 42},
    )

    assert tracker.run_id == "abc123"
```

- [ ] **Step 2: Run tracking tests and verify failure.**

Run: `uv run pytest tests/training/test_tracking.py -q`

Expected: failures for the new methods and `.netrc` path.

- [ ] **Step 3: Implement the W&B-only adapter changes.** Use `netrc.netrc()` with known W&B hosts (`api.wandb.ai`, `api.wandb.com`, `wandb.ai`, `wandb.com`) only as a boolean availability check. Keep import lazy. Build `wandb.Table(columns=list(columns), data=[list(row) for row in rows])`. Build one `wandb.Artifact`, call `add_file(str(path), name=artifact_name)` for every mapping entry, log it, wait, and return its immutable `name:version` reference. For restore, call `run.use_artifact(reference).download(root=str(destination))` and return the downloaded `Path`, resolving `best.pt` when the download returns a directory. Base `Tracker` methods remain no-op except artifact restore, which raises a precise disabled-tracking error.

- [ ] **Step 4: Set cloud defaults without changing local defaults.** Change only `configs/cloud.yaml` tracking block to:

```yaml
tracking:
  enabled: true
  mode: online
  project: token-mixer-placement-matters
  entity: aniekanetimudo
  group: token-mixer-brats-seed42
  job_type: train
  run_name: null
  log_every_steps: 20
  log_images: true
  log_checkpoints: true
  directory: ${paths.output_root}/wandb
```

- [ ] **Step 5: Run focused tests and config composition.**

Run: `uv run pytest tests/training/test_tracking.py -q && uv run python -m token_mixer --config-name cloud experiment=mod_a run=debug --cfg job`

Expected: tracking tests pass; composed output contains `enabled: true`, `mode: online`, `log_images: true`, the approved entity/project, and no credential value. Epoch metric definitions remain independent of `log_every_steps` and snapshot interval.

---

### Task 3: Add epoch/phase telemetry without changing optimizer semantics

**Files:**
- Modify: `src/token_mixer/training/engine.py`
- Modify: `configs/run/full.yaml`, `configs/run/debug.yaml`
- Modify: `tests/training/test_engine.py`
- Modify: `tests/training/test_engine_resume.py`

**Interfaces:**
- Add immutable `EpochTrainResult(train_loss: float, global_step: int, samples: int, voxels: int)` for private `_train_epoch` return values.
- Extend `fit(..., *, resume=None, loader_generator=None, warm_start=None, finish_tracker: bool = True, snapshotter=None, early_stopping=None) -> FitResult`.
- `finish_tracker=False` leaves tracker open for pipeline completion; default remains `True` so existing direct callers retain current behavior.
- Preserve existing unnamespaced history keys and append namespaced fields after every completed epoch: `train/epoch`, `train/epoch_seconds`, `train/optimizer_steps`, `train/samples_per_second`, `train/voxels_per_second`, `train/peak_memory_allocated_gb`, `train/peak_memory_reserved_gb`, `power/average_watts`, `power/max_watts`, `power/energy_joules`, `power/sample_count`, `power/sample_interval_ms`, `power/status`, `val/epoch`, `val/epoch_seconds` when validation runs, `run/elapsed_seconds`, and `train/time_to_best_seconds` when a new best is selected. Scalar records are per epoch; image snapshots use separate cadence.
- Add configurable early stopping with `enabled`, `monitor`, `mode`, `patience`, `min_delta`, and optional `min_epochs`; patience counts consecutive validation evaluations without improvement, not raw epochs. Record `early_stopping/stopped`, `early_stopping/stop_epoch`, and `early_stopping/best_epoch` in local provenance and W&B summary.
- Add timing metadata with `scope="process_segment"`; resume state stores measured timing fields without fabricating elapsed wall time from an old `perf_counter` value.

- [ ] **Step 1: Add failing engine tests.** Verify every completed epoch emits scalar fields with `train/epoch` and monotonic global step, observed samples equal actual batch samples, voxels equal `batch * product(spatial dimensions)` excluding channels, optimizer steps are epoch deltas, validation time/`val/epoch` appear only when validation runs, time-to-best appears on best epoch, NVML snapshot fields are copied into the epoch record without changing optimizer work, configurable early stopping stops after the requested number of non-improving validation evaluations, tracker remains open with `finish_tracker=False`, and resume preserves global step/history while starting a new process-segment timer.

```python
def test_fit_records_observed_efficiency_fields_without_changing_steps():
    result = fit(
        _TinyEncoderDecoder(),
        _loader(),
        _loader(),
        nn.MSELoss(),
        lambda *_args: {"mean_dice": 0.5},
        [PhaseSpec("phase1", 1, False, 0.1, 0.1)],
        _config(),
        _RecordingTracker(),
        None,
    )

    record = result.history[0]
    assert record["global_step"] == 2
    assert record["train/optimizer_steps"] == 2
    assert record["train/samples_per_second"] > 0
    assert record["train/voxels_per_second"] > 0
    assert record["run/elapsed_seconds"] >= record["train/epoch_seconds"]
```

- [ ] **Step 2: Run engine tests and verify failure.**

Run: `uv run pytest tests/training/test_engine.py tests/training/test_engine_resume.py -q`

Expected: new telemetry assertions fail while existing semantic tests continue to identify regressions.

- [ ] **Step 3: Implement boundary timing, per-epoch scalar logging, and early stopping.** Start the process-segment timer only after model/device/optimizer/scheduler setup and checkpoint restoration. Start one `NvmlPowerSampler` for the fit when `efficiency.nvml.enabled` is true and the resolved device is CUDA; take a boundary snapshot after each epoch and stop it in `finally`. Before each measured epoch call `reset_peak_memory(device)` and the shared synchronization helper. Time `_train_epoch` as training time and evaluator execution as validation time. Do not add synchronization inside `_train_epoch`'s batch loop. Compute observed sample/voxel totals from each moved input tensor. Emit the scalar record after every completed epoch with `train/epoch` and, when applicable, `val/epoch`; use `tracker.log(record, step=global_step)` with unchanged monotonic global steps. Apply early-stopping patience only after validation evaluation and best-checkpoint selection, then persist stop/best epochs before breaking. Snapshot cadence is independent and handled by Task 3b. Wrap both normal and exceptional exits in the existing finish logic, conditioned on `finish_tracker`; NVML failure must never mask the training exception.

- [ ] **Step 4: Preserve resume semantics explicitly.** Store `timing_scope`, `segment_elapsed_seconds`, and measured cumulative sums in checkpoint state. On resume, read only those numeric measured values; restart `perf_counter` for the new process segment. Mark `train/time_to_best_seconds` as process-segment or cumulative-measured in metadata so a resumed run cannot be mistaken for one uninterrupted wall-clock run.

- [ ] **Step 5: Add explicit early-stopping config.** Add `early_stopping.enabled`, `monitor`, `mode`, `patience`, `min_delta`, and `min_epochs` to run configs. Keep debug runs disabled for short contract tests; make the full/cloud profile explicitly opt in with a reviewable patience value. Validate that patience counts validation checks and that stopping after epoch `N` retains the latest scalar metrics, selected best checkpoint, and best/final snapshot ordering.

```yaml
early_stopping:
  enabled: false
  monitor: mean_dice
  mode: max
  patience: 10
  min_delta: 0.0
  min_epochs: 0
```

Keep the default disabled so existing run budgets remain unchanged; an approved cloud run may override `run.early_stopping.enabled=true` without changing the implementation contract.

- [ ] **Step 6: Run focused and regression tests.**

Run: `uv run pytest tests/training/test_engine.py tests/training/test_engine_resume.py tests/training/test_tracking.py -q`

Expected: all tests pass and existing checkpoint/global-step assertions remain unchanged.

---

### Task 3b: Add tracker-gated train/validation segmentation snapshots

**Files:**
- Modify: `src/token_mixer/training/tracking.py`
- Modify: `src/token_mixer/training/engine.py`
- Modify: `src/token_mixer/evaluation/inference.py`
- Modify: `src/token_mixer/evaluation/visualization.py`
- Modify: `configs/cloud.yaml`, `configs/local.yaml`
- Extend: `tests/training/test_tracking.py`, `tests/training/test_engine.py`, `tests/training/test_engine_resume.py`, `tests/evaluation/test_inference.py`, `tests/evaluation/test_visualization.py`, `tests/test_cli_config.py`, `tests/integration/test_synthetic_debug.py`

**Interfaces:**
- Add `visualization.segmentation_snapshots` settings: `enabled`, `snapshot_interval_epochs: 10`, `include_best: true`, `include_final: true`, `splits: [train, val]`, `sample_count`, `axis`, `image_channel: 3`, `local_enabled`, and `output_dir`. The interval is absolute and emits at every multiple of 10, not from a list of selected epochs.
- Use Task 2's `Tracker.log_images(images, *, step, captions=None)` boundary. When snapshotting is disabled, do no image work; when W&B image logging is disabled, do not import W&B, construct image objects, iterate loaders, run inference, or import plotting code for W&B. Explicit `local_enabled` may still produce local-only PNGs. Offline mode stores enabled W&B images in the local offline run without network; online mode forwards them to W&B.
- Add a snapshot callback after each completed epoch at absolute interval epochs, not phase-local epochs, while also emitting a best snapshot whenever validation selects a new best. Use fixed non-augmented train/validation examples selected deterministically; resumed runs deduplicate `(split, epoch, kind, case_hash)`.
- Generate all-region `[ET, TC, WT]` previews through `logits_to_regions` and the existing `save_slice_visualization` contract. Native 3-D uses sliding-window inference; TransUNet uses its 2-D path. Use MRI channel `3` (`t2f`), retain WT→TC→ET colors, and use a region-aware montage when one slice cannot show every region.
- Save local images under `<Hydra output>/segmentation_snapshots/{train,val}/`; use keys such as `segmentation/train/epoch_0010` and `segmentation/val/epoch_0010`, with W&B step equal to the corresponding global step. Best snapshots use the current newly selected best weights; final snapshots use restored best weights after final validation, and all are deduplicated against scheduled snapshots.

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

Cloud keeps this block enabled with `tracking.log_images: true`; local defaults keep tracking and image logging disabled. Local-only evidence requires the explicit `local_enabled: true` override.

- [ ] **Step 1: Add failing snapshot tests.** Assert the mode matrix: disabled snapshotting performs no image work; `tracking.log_images=false` performs no W&B import/image construction or W&B snapshot inference while explicit `local_enabled=true` may produce local-only PNGs; offline mode forwards enabled W&B images to local files without network; online mode forwards W&B images. Assert cloud defaults enable image logging while local defaults remain disabled.

- [ ] **Step 2: Add failing cadence and rendering tests.** Assert interval callbacks occur exactly at absolute epochs `10`, `20`, `30`, `40`, ... through the configured budget, plus a best snapshot on a non-interval improvement (for example, scheduled epoch `100` and best epoch `104`) and final when configured; snapshots contain both train and validation splits, all ET/TC/WT regions, deterministic slice/sample selection, channel `3`, stable keys, and no raw case IDs. Assert resumed training does not duplicate prior snapshots and early stopping retains the last scalar record plus best/final snapshot ordering.

- [ ] **Step 3: Implement snapshot collection and tracker forwarding.** Evaluate the effective gate before loader iteration/inference/plotting: return immediately when the feature is disabled or both W&B image logging and local output are disabled; when only W&B image logging is disabled, run the local-only path only if `local_enabled=true`, without importing W&B or constructing `wandb.Image`. Reuse fixed non-augmented examples, canonical region conversion, existing 2-D/3-D evaluator boundaries, and local visualization helpers. Construct `wandb.Image` only inside the W&B adapter. Keep snapshot failures bounded in local provenance and never mask numeric training/quality exceptions.

- [ ] **Step 4: Run focused snapshot tests.**

Run: `uv run pytest tests/training/test_tracking.py tests/training/test_engine.py tests/training/test_engine_resume.py tests/evaluation/test_inference.py tests/evaluation/test_visualization.py tests/test_cli_config.py tests/integration/test_synthetic_debug.py -q`

Expected: disabled/offline/online/local-only gating, interval cadence, best/final ordering, all-region rendering, resume deduplication, local paths, and W&B image-key assertions pass.

---

### Task 4: Keep training W&B run open through quality and artifact completion

**Files:**
- Modify: `src/token_mixer/pipelines/_baseline_common.py`
- Modify: `src/token_mixer/pipelines/train_metaunetr.py`
- Modify: `src/token_mixer/pipelines/pretrain_cnn.py`
- Modify: `src/token_mixer/training/artifacts.py`
- Modify: `src/token_mixer/cli.py`
- Extend: `tests/pipelines/test_baseline_pipelines.py`, `tests/pipelines/test_train_metaunetr.py`, `tests/pipelines/test_pretrain_cnn.py`, `tests/test_cli_config.py`, `tests/training/test_artifacts.py`

**Interfaces:**
- Add shared `_output_dir(cfg) -> Path` resolution for Hydra `paths.experiment_output` and existing output aliases.
- Add shared `_namespace_metrics(prefix: str, metrics: Mapping[str, Any]) -> dict[str, Any]` that maps existing quality keys to `test/*` or `val/*` without removing raw local keys.
- Add shared `_finalize_training_run(cfg, tracker, result, checkpoints, *, protocol, extra_files=()) -> dict[str, Path]` that writes local artifacts, logs summaries/tables/artifacts through `Tracker`, and returns paths.
- All four training paths call `fit(..., finish_tracker=False)` when supported, restore `best.pt`, run held-out/test or final validation evaluation, call `_finalize_training_run`, then call `tracker.finish()` in `finally`.
- CLI still writes completion artifacts as a fallback for stubbed/custom runners, but skips rewriting existing `metrics.json` and `provenance.json` produced by pipeline finalization.
- `write_run_artifacts` schema version increments to `2` and serializes efficiency metadata/fields with `_json_safe`; `write_failed_run_artifact` is used for non-transfer failures too, while preserving the original exception.

- [ ] **Step 1: Add failing ordering tests.** Make fake trackers record events and assert `fit`, best restore, test evaluator, local artifact write, W&B summary/artifact calls, and `finish` occur in that order. Add assertions for `test/dice_ET`, `test/dice_TC`, `test/dice_WT`, `test/avg_dice`, `test/hd95_*`, and `test/hd95_excluded_cases`. Confirm a failed fit writes only failed provenance and still finishes the tracker.

```python
def test_baseline_finishes_tracker_after_held_out_evaluation(monkeypatch, tmp_path):
    events = []

    class Tracker:
        def log_summary(self, metrics):
            events.append("summary")

        def log_artifact(self, *args, **kwargs):
            events.append("artifact")
            return "entity/project/model:v0"

        def finish(self):
            events.append("finish")

    # Existing pipeline fixtures patch model/loaders/fit/restore/evaluator.
    # The assertions below are the required observable order for that fixture.
    assert events == ["fit", "restore_best", "test", "summary", "artifact", "finish"]
```

- [ ] **Step 2: Run pipeline/artifact tests and verify failure.**

Run: `uv run pytest tests/pipelines/test_baseline_pipelines.py tests/pipelines/test_train_metaunetr.py tests/pipelines/test_pretrain_cnn.py tests/training/test_artifacts.py tests/test_cli_config.py -q`

Expected: new lifecycle assertions fail because current `fit` finishes before held-out evaluation and CLI owns local artifact writing.

- [ ] **Step 3: Implement shared finalization.** Keep `fit` responsible for training history and its existing best summary. Pipeline finalization must:
  1. restore the selected best checkpoint;
  2. evaluate held-out data (or CNN final validation, explicitly labeled as denoising validation);
  3. merge test metrics into `FitResult`;
  4. write `metrics.json` and `provenance.json` locally;
  5. log namespaced summary values;
  6. log `config.yaml`, `metrics.json`, `provenance.json`, and selected `best.pt` when present as one model artifact with aliases `best` and `latest`; include final NVML summary/status fields without mixing board energy with allocator memory; and
  7. return before the `finally` block calls `finish()`.

- [ ] **Step 4: Preserve direct pipeline testability.** If a test fixture supplies a fit function without `finish_tracker`, `_invoke_fit` must inspect its signature and omit that keyword. Base `Tracker` supplies no-op finalization methods, so disabled tracking still writes local artifacts and never imports W&B. `write_run_artifacts` must retain old top-level `best_epoch`, `best_metric`, `history`, and `test_metrics` keys while adding schema/provenance/timing fields.

- [ ] **Step 5: Run focused regression tests.**

Run: `uv run pytest tests/pipelines tests/training/test_artifacts.py tests/test_cli_config.py -q`

Expected: all pipeline, artifact, CLI, and prior resume/transfer tests pass; fake tracker event order proves held-out quality precedes finish.

---

### Task 5: Add case-aware inference records and protocol boundaries

**Files:**
- Modify: `src/token_mixer/evaluation/inference.py`
- Modify: `src/token_mixer/pipelines/_baseline_common.py`
- Modify: `src/token_mixer/data/datasets.py` only if benchmark case grouping cannot use existing loader metadata
- Extend: `tests/evaluation/test_inference.py`, `tests/pipelines/test_baseline_pipelines.py`, `tests/data/test_datasets.py`

**Interfaces:**
- Preserve default `evaluate_full_volumes(...) -> dict[str, Any]` aggregate behavior.
- Add keyword-only `collect_case_records: bool = False` and return `case_records` only when requested. Each record contains in-memory `case_id`, `dice_by_region`, `hd95_by_region`, `latency_ms` when timing is enabled, `voxel_count`, validated `spacing`, `sliding_window_count`, and `exclusion_flags`.
- Add `sliding_window_count(spatial_shape: Sequence[int], roi_size: Sequence[int], overlap: float) -> int` using the same ROI/overlap convention as the MONAI call.
- Add `hash_case_id(case_id: str) -> str` in the benchmark serializer, not in generic quality metrics.
- Keep `evaluate_slices` aggregate output unchanged; benchmark case rows for TransUNet use declared slice aggregation and the same hashed identifier contract.

- [ ] **Step 1: Add failing case-record tests.** Verify full-volume records preserve spacing and exclusion flags, sliding-window count is positive and deterministic, default evaluation does not add records, and serialized case IDs are fixed-length SHA-256-derived values with no raw ID.

```python
def test_full_volume_case_records_are_opt_in_and_include_protocol_fields(monkeypatch):
    result = evaluate_full_volumes(
        model=_fixture_model(),
        loader=_fixture_volume_loader(),
        roi_size=(4, 4, 4),
        sw_batch_size=1,
        overlap=0.25,
        device="cpu",
        default_spacing=(1.0, 1.0, 2.0),
        collect_case_records=True,
    )

    row = result["case_records"][0]
    assert row["spacing"] == (1.0, 1.0, 2.0)
    assert row["voxel_count"] > 0
    assert row["sliding_window_count"] > 0
    assert set(row["exclusion_flags"]) == {"ET", "TC", "WT"}
```

- [ ] **Step 2: Run focused inference tests and verify failure.**

Run: `uv run pytest tests/evaluation/test_inference.py tests/data/test_datasets.py -q`

Expected: new opt-in record assertions fail; default aggregate tests continue to pass.

- [ ] **Step 3: Implement record collection without changing quality formulas.** Refactor only enough of the existing full-volume loop to count windows, measure named boundaries when requested, and append one record per declared case. Enforce batch size `1` for case-level timing or mark latency as batch latency rather than pretending it is per-case latency. Keep explicit spacing validation and HD95 exclusion behavior. For 2-D slices, retain current dataset/training tuple compatibility; if case identity must be exposed, append metadata in a backward-compatible third tuple element and update `_unpack_training_batch`/tests accordingly.

- [ ] **Step 4: Run focused tests and compile checks.**

Run: `uv run pytest tests/evaluation/test_inference.py tests/data/test_datasets.py tests/pipelines/test_baseline_pipelines.py -q && uv run python -m compileall -q src/token_mixer/evaluation src/token_mixer/data`

Expected: all existing aggregate metrics remain unchanged and new records pass schema checks.

---

### Task 6: Implement benchmark result contract and protocol runner

**Files:**
- Create: `src/token_mixer/evaluation/benchmark.py`
- Create: `src/token_mixer/pipelines/benchmark.py`
- Create: `tests/evaluation/test_benchmark.py`
- Create: `tests/pipelines/test_benchmark.py`

**Interfaces:**
- `BenchmarkResult` is an immutable dataclass with `summary: dict[str, Any]`, `rows: list[dict[str, Any]]`, and `provenance: dict[str, Any]`.
- `run_model_protocol(model, inputs, *, protocol, warmup_iterations, repetitions, batch_sizes) -> dict[str, Any]` combines `static_model_cost`, `measure_forward`, and `measure_batch_sweep` without calling W&B.
- `hash_case_id(case_id: str) -> str` returns the first 16 hexadecimal characters of SHA-256 over UTF-8 case ID text.
- `serialize_benchmark(output_dir: Path, result: BenchmarkResult) -> Path` writes one atomic JSON file with `schema_version`, `summary`, `rows`, and `provenance`, normalizing non-finite floats to `null`.
- `run_benchmark(cfg: Mapping[str, Any]) -> BenchmarkResult` selects protocol from experiment family, constructs the same model/data boundary as training, restores a local `best.pt` or uses `Tracker.restore_artifact`, measures model-level and case-level protocols, creates a separate tracker with `job_type=benchmark`, logs rows/table/artifact, and finishes it in `finally`.

- [ ] **Step 1: Add failing pure benchmark tests.** Cover protocol selection, static fields, THOP MACs, fvcore FLOPs, tool/version/convention fields, unsupported-operator partial status, NVML power/energy fields, model-level summary namespaces, fixed warmup/repetition values, non-finite JSON normalization, case-ID hashing, same-protocol row comparability, and OOM boundary reporting.

```python
def test_serialize_benchmark_contains_safe_case_rows(tmp_path):
    result = BenchmarkResult(
        summary={"inference/latency_mean_ms": 2.0},
        rows=[
            {
                "case_id_hash": hash_case_id("BraTS-001"),
                "protocol": "native_3d_full_volume",
                "latency_ms": float("nan"),
            }
        ],
        provenance={"manifest_hash": "sha256"},
    )

    path = serialize_benchmark(tmp_path, result)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["rows"][0]["case_id_hash"] == hash_case_id("BraTS-001")
    assert payload["rows"][0]["latency_ms"] is None
    assert "BraTS-001" not in path.read_text(encoding="utf-8")
```

- [ ] **Step 2: Run benchmark tests and verify failure.**

Run: `uv run pytest tests/evaluation/test_benchmark.py tests/pipelines/test_benchmark.py -q`

Expected: collection or assertion failure because benchmark modules do not yet exist.

- [ ] **Step 3: Implement protocol runner.** Use `model.eval()` and `torch.inference_mode()` for model-level measurements. Allocate fixed inputs once outside timing. Record:

```text
model/parameters
model/trainable_parameters
model/macs
model/flops
model/mac_tool
model/mac_tool_version
model/mac_convention
model/mac_status
model/flop_tool
model/flop_tool_version
model/flop_convention
model/flop_status
model/unsupported_ops
model/checkpoint_bytes
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

Use `model/macs` from THOP and `model/flops` from fvcore on the same fixed input. Record `model/mac_tool="thop"`, `model/flop_tool="fvcore"`, both package versions, both counting conventions, independent `mac_status`/`flop_status`, `model/unsupported_ops`, and `power/*` fields from NVML. Preserve partial counts and explicit unsupported/unavailable status when custom operators or NVML cannot be measured; never substitute zero. Preserve largest passing/first failing batch for throughput sweep. Power is board-level, sampled during the named benchmark section, and never includes checkpoint restoration.

- [ ] **Step 4: Implement model construction/restoration through existing boundaries.** Dispatch `metaunetr_mamba/mod_a/mod_b` through `build_metaunetr`, ResUNet3D through `build_resunet3d`, SwinUNETR through `build_swinunetr`, TransUNet through its existing validation/build functions, and CNN through `build_denoising_model`. Reuse existing loader builders and checkpoint metadata validation. Do not import W&B in this module; create a `Tracker` through `create_tracker` and use its artifact restore method only when `benchmark.source_artifact` is configured.

- [ ] **Step 5: Add fixed case-level protocol.** Run declared held-out cases through the actual evaluator with separate model compute and end-to-end boundaries. Emit rows with `case_id_hash`, `model`, `protocol`, `dice_by_region`, `hd95_by_region`, `latency_ms`, `voxel_count`, `spacing`, `sliding_window_count`, and `exclusion_flags`; emit aggregate `inference/case_latency_median_ms`, `inference/case_latency_p95_ms`, `inference/cases_per_second`, `inference/sliding_window_count`, and `inference/n_cases`. Record `load_seconds`, `preprocess_seconds`, `model_compute_seconds`, `postprocess_seconds`, `metric_seconds`, and `end_to_end_seconds` when boundaries are available. Exclude checkpoint restoration from per-case latency.

- [ ] **Step 6: Run benchmark-focused tests.**

Run: `uv run pytest tests/evaluation/test_benchmark.py tests/pipelines/test_benchmark.py tests/evaluation/test_inference.py -q`

Expected: local benchmark contract, protocol, restore, OOM, case safety, and inference tests pass.

---

### Task 7: Add Hydra benchmark command and complete W&B linkage

**Files:**
- Create: `configs/benchmark.yaml`
- Modify: `src/token_mixer/cli.py`
- Modify: `src/token_mixer/training/tracking.py` only for any missing source-run/artifact metadata forwarding discovered by Task 6
- Extend: `tests/test_cli_config.py`, `tests/pipelines/test_benchmark.py`

**Interfaces:**
- `configs/benchmark.yaml` defaults to `experiment: mod_a`, `run: full`, `command: benchmark`, `runtime: benchmark`, cloud data paths, and explicit benchmark settings.
- Benchmark config fields are `benchmark.source_checkpoint`, `benchmark.source_artifact`, `benchmark.source_train_run_id`, `benchmark.protocol`, `benchmark.input_shape`, `benchmark.warmup_iterations`, `benchmark.repetitions`, `benchmark.batch_sizes`, `benchmark.case_limit`, and `benchmark.output_name`.
- CLI dispatch checks `command == "benchmark"` before experiment name and calls `token_mixer.pipelines.benchmark.run_benchmark`; ordinary experiment dispatch remains unchanged.
- Benchmark tracker config uses `group: token-mixer-brats-seed42`, `job_type: benchmark`, `source_train_run_id`, `source_checkpoint_artifact`, and `protocol_id`.

- [ ] **Step 1: Add failing CLI/config tests.** Assert benchmark config composes without training, dispatches to `run_benchmark`, rejects missing checkpoint/artifact source with a clear error, and preserves ordinary experiment dispatch. Assert benchmark W&B metadata is separate from training global steps.

- [ ] **Step 2: Run CLI tests and verify failure.**

Run: `uv run pytest tests/test_cli_config.py tests/pipelines/test_benchmark.py -q`

Expected: benchmark config/dispatch tests fail before implementation.

- [ ] **Step 3: Add benchmark config.** Use this explicit baseline and require overrides for source and protocol-sensitive values:

```yaml
defaults:
  - experiment: mod_a
  - run: full
  - _self_

command: benchmark
runtime: benchmark
device: cuda
paths:
  data_root: ${hydra:runtime.cwd}/data/cloud/brats
  image_root: ${data.image_root}
  output_root: ${hydra:runtime.cwd}/outputs
  experiment_output: ${paths.output_root}/${runtime}/${experiment.name}/${run.name}/${now:%Y-%m-%d}/${now:%H-%M-%S}
  checkpoint_dir: ${paths.experiment_output}/checkpoints
  manifest: ${hydra:runtime.cwd}/data/manifests/brats_seed42.json

tracking:
  enabled: true
  mode: online
  project: token-mixer-placement-matters
  entity: aniekanetimudo
  group: token-mixer-brats-seed42
  job_type: benchmark
  run_name: null
  directory: ${paths.output_root}/wandb

efficiency:
  enabled: true
  nvml:
    enabled: true
    device_index: 0
    sample_interval_seconds: 0.25
  profiler:
    enabled: true
    mac_tool: thop
    flop_tool: fvcore

benchmark:
  source_checkpoint: null
  source_artifact: null
  source_train_run_id: null
  protocol: auto
  input_shape: null
  warmup_iterations: 20
  repetitions: 100
  batch_sizes: [1, 2, 4, 8]
  case_limit: null
  output_name: benchmark.json
```

- [ ] **Step 4: Implement dispatch and linkage.** Require exactly one source (`source_checkpoint` or `source_artifact`), record source train run ID and immutable artifact reference, keep benchmark repetition steps independent from training `global_step`, and write benchmark JSON before tracker finish. Local disabled mode remains valid for tests when explicitly overridden.

- [ ] **Step 5: Run config-only checks.**

Run: `uv run python -m token_mixer --config-name benchmark experiment=mod_a run=debug --cfg job && uv run python -m token_mixer --config-name local experiment=mod_a run=debug --cfg job`

Expected: both commands exit `0`, neither invokes model construction/training, and benchmark output contains explicit online destination without a secret.

---

### Task 8: Add synthetic integration/schema coverage and research documentation

**Files:**
- Modify: `tests/integration/test_synthetic_debug.py`
- Modify: `README.md`
- Modify: `codebase/CONFIG.md`
- Modify: `codebase/TRAINING.md`
- Modify: `codebase/EVALUATE.md`
- Modify: `codebase/REPRODUCIBILITY.md`
- Create: `codebase/BENCHMARKING.md`

**Interfaces:**
- Synthetic integration proves wiring and schema only; it must not claim performance or quality results. It must cover THOP/fvcore and NVML unavailable/fixture paths without treating those fixtures as hardware measurements.
- Documentation records exact protocol names, timing boundaries, per-epoch metric/epoch-axis behavior, snapshot interval/best semantics, early-stopping configuration, W&B namespaces, artifact lineage, `.netrc` behavior, local fallback, OOM/FLOP status semantics, and cloud gate commands.

- [ ] **Step 1: Add failing synthetic assertions.** Extend the existing synthetic harness to assert local `metrics.json` retains efficiency and power history plus one scalar record per completed epoch, `provenance.json` includes timing scope, protocol identity, and early-stopping state, benchmark JSON contains static `model/macs` and `model/flops` fields plus `power/*` status, case rows contain only `case_id_hash`, interval/best segmentation snapshots contain all ET/TC/WT regions, and disabled tracking produces no W&B import or image construction.

- [ ] **Step 2: Run synthetic test and verify failure.**

Run: `uv run pytest tests/integration/test_synthetic_debug.py -q`

Expected: new schema assertions fail before integration wiring is complete.

- [ ] **Step 3: Implement schema assertions and docs.** Add `BENCHMARKING.md` sections `Purpose`, `Protocol Families`, `Model-Level Measurement`, `Case-Level Measurement`, `W&B Linkage`, `Local Evidence`, `Failure Semantics`, and `Pre-full-run Gate`. Update existing guides rather than duplicating architecture facts. State that CNN output is denoising validation, TransUNet is 2-D slice protocol, and native baselines/MetaUNETR are 3-D full-volume protocol.

- [ ] **Step 4: Run synthetic and documentation checks.**

Run: `uv run pytest tests/integration/test_synthetic_debug.py -q && git diff --check`

Expected: synthetic schema/wiring passes; no raw case IDs, absolute machine paths, credentials, or unrelated W&B project names appear in reviewed docs.

---

### Task 9: Full repository verification and cheap online smoke

**Files:** No new source files. Inspect all changed files and generated local output only; do not stage runtime outputs, W&B files, checkpoints, raw data, or untracked `NUL`.

**Interfaces:** Uses completed Tasks 1–8. Produces evidence required before a real benchmark or full run.

- [ ] **Step 1: Run repository checks.**

Run: `uv lock --check && uv run pytest -q -rs && uv run python -m compileall -q src tests && git diff --check`

Expected: lock check, full suite, compileall, and whitespace checks all exit `0`; record exact test count and skip reason.

- [ ] **Step 2: Run configuration-only cloud inspection.**

From verified cloud workspace, run:

```bash
uv run python -m token_mixer --config-name cloud experiment=mod_a run=debug --cfg job
```

Expected: `device: cuda`, `tracking.enabled: true`, `tracking.mode: online`, `tracking.log_images: true`, approved entity/project/group/job type, `train/epoch` metric axis, `snapshot_interval_epochs: 10`, best/final snapshot gates, early-stopping settings, manifest path/hash inputs, and no credential value in output.

- [ ] **Step 3: Run cheap online W&B smoke, not full training.**

Use explicit cloud overrides for a debug-sized run, e.g. `experiment=mod_a run=debug`, and verify `PWD` before execution. The run must create/update only `aniekanetimudo/token-mixer-placement-matters`, remain open through held-out quality/artifact logging, and finish cleanly. Inspect live W&B history while the process is running.

Expected W&B evidence: `job_type=train`, group `token-mixer-brats-seed42`, config/provenance identity, live per-epoch `train/*` and `val/*` history plotted against `train/epoch`/`val/epoch`, regional/mean Dice, HD95, learning rate, epoch duration, throughput, allocator memory, `segmentation/train/*` and `segmentation/val/*` images at interval plus non-interval best cadence with ET/TC/WT overlays, held-out `test/*` summary, early-stopping state when enabled, and a `best`/`latest` checkpoint artifact. If online write fails, preserve local JSON and mark the smoke failed; do not retry against the old project.

- [ ] **Step 4: Run one restored-model benchmark only after smoke passes.**

Invoke benchmark with the smoke run's local best checkpoint or immutable W&B artifact, explicit `source_train_run_id`, and `run=debug`/small case limit. Inspect local `benchmark.json` and separate W&B `job_type=benchmark` run.

Expected benchmark evidence: static parameter, THOP MAC, fvcore FLOP, tool/version/convention, and NVML power fields; latency distribution; throughput/memory rows; protocol ID; checkpoint/source linkage; case-level hashed rows; exclusion counts; and no raw case IDs. Partial/unavailable status is acceptable only with an explicit reason and does not replace a successful counter with zero.

- [ ] **Step 5: Stop and request approval for full model matrix.**

Do not launch full cloud training automatically. Report pass/fail evidence, remaining uncertainty (especially unsupported FLOPs and physical-batch comparability), and ask for explicit approval specifying models, seeds, and expected compute before starting the expensive matrix.

## Spec Coverage Review

- Training fields and epoch-boundary semantics: Tasks 3 and 4.
- Per-epoch scalar records, W&B epoch axes, interval-only image cadence, best snapshots, and configurable early stopping: Tasks 2, 3, 3b, 4, 8, and 9.
- Static parameters, THOP MACs, fvcore FLOPs, NVML power/energy, memory, timing distributions, fixed inputs, OOM boundaries: Tasks 1 and 6.
- Separate model-level/case-level protocols and 2-D/3-D boundaries: Tasks 5 and 6.
- Quality pairing, HD95 null/exclusion semantics, safe case rows: Tasks 5, 6, and 8.
- W&B auth, online cloud defaults, lifecycle, tables, artifacts, aliases, wait ordering: Tasks 2, 4, 7, and 9.
- Snapshot cadence, all-region visualization, disabled/offline/online image gates, local PNG evidence, and safe captions: Task 3b, Tasks 4, 8, and 9.
- Provenance, Git/software/hardware/config identity: Tasks 2, 4, 7, and 8; existing metadata builders remain the source for model/data identity.
- Failure behavior for disabled/offline W&B, online failure, profiler/operator status, NVML unavailability, CUDA fallback, OOM, non-finite metrics, and benchmark failure: Tasks 1, 2, 4, 6, and 9.
- Full-suite and pre-full-run acceptance gates: Tasks 8 and 9.

## Plan Self-Review

- The three required measurement dependencies are explicit in Task 1 and must be recorded in both `pyproject.toml` and `uv.lock` before profiler/power code is used.
- Existing training history/global-step/checkpoint contracts remain additive and are covered by regression tests.
- W&B lifecycle has one owner: pipeline completion calls `Tracker` operations, and `Tracker.finish()` occurs after quality/artifact logging.
- Local JSON remains available when tracking is disabled/offline or online W&B fails.
- No implementation task launches full training; Task 9 explicitly stops at one smoke and one benchmark.
- FLOPs/MACs and NVML are core deliverables in Tasks 1, 3, 6, 7, and 9; no required measurement is deferred from this plan. Only future report tooling and carbon estimates remain excluded.
- Segmentation snapshots are tracker-controlled: disabled mode performs no image work, offline mode writes local W&B files without network, and online mode logs configured train/validation images after the same gates pass.
- Snapshot interval is independent of scalar metric cadence; best snapshots may occur on non-interval epochs, and early-stopping defaults remain explicit so existing budgets are not silently changed.
