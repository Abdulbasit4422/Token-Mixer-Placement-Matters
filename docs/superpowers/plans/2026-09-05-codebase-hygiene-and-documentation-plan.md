# Codebase Hygiene and Maintainer Documentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Track steps with checkbox syntax.

**Goal:** Remove only proven active-code clutter, replace workflow-history names, and add a curated SVG-backed maintainer/researcher map of the active Token Mixer codebase.

**Architecture:** Preserve `CLI → pipelines → data / models / evaluation / training`. Perform evidence-gated hygiene first, then add one-owner guides under `codebase/` with hand-authored SVGs under `assets/codebase/`. Archive and generated state remain outside scope.

**Tech stack:** Python 3.12, PyTorch, Hydra/OmegaConf, pytest, Jupytext, Markdown, standalone SVG, Git.

## Global constraints

- Use repository-relative paths in source, docs, prompts, and diagrams; never add machine-specific absolute paths.
- Exclude `archive/`, caches, runtime `outputs/`, checkpoints, W&B files, `.superpowers/`, and generated state from cleanup and documentation content.
- Remove only imports/helpers proven unused by active reference search and test collection.
- Preserve useful contract, optional-dependency, pipeline, and reproducibility tests.
- Rename active workflow-history names to behavior-based names; preserve archive names, Git history, planning records, and provenance references.
- Do not change model, data, configuration, dependency, checkpoint, training, evaluation, or runtime behavior.
- Preserve modular boundaries and private test seams. Defer wrapper/helper deduplication.
- `codebase/README.md` is canonical documentation index; root `README.md` remains onboarding and links to it.
- Use dedicated `TESTING.md` and `NOTEBOOKS.md` guides.
- Group `metaunetr_mamba`, `mod_a`, and `mod_b` in `METAUNETR.md`; use separate guides for ResUNet3D, SwinUNETR, TransUNet, and CNN pretraining.
- Store eight portable, hand-authored SVGs under `assets/codebase/`.
- Each guide and SVG cites stable `path::symbol` sources and passes a manual fact checklist.
- Model figures show stage-level internals, meaningful activations/normalizations, skip routes, mixer placement, and verified tensor shapes; do not invent unsupported operations.
- No documentation generator, diagram generator, link-check dependency, full BraTS run, external TransUNet run, or paid/cloud compute.
- Every task ends with focused validation and an atomic commit. Final validation includes `uv lock --check`, `uv run pytest -q -rs`, `uv run python -m compileall -q archive src tests`, and `git diff --check`.

The `.superpowers/` ignore rule is already committed as `d080865`; no `.superpowers/` files are tracked in Git.

## File and ownership map

| Area | Files | Responsibility |
| --- | --- | --- |
| Evidence | `docs/superpowers/audits/2026-09-05-codebase-hygiene-audit.md` | Candidate evidence, decisions, deferred work |
| Safe cleanup | `src/token_mixer/pipelines/pretrain_cnn.py`, `tests/evaluation/test_inference.py`, `tests/training/test_tracking.py` | Dead helper/import removal only |
| Active naming | `tests/training/test_task12_resume.py` → `tests/training/test_resume_contract.py` | Behavior-based test filename |
| Index | `codebase/README.md`, `README.md` | Maintainer navigation and root link |
| Cross-cutting guides | `codebase/ARCHITECTURE.md`, `DATA.md`, `CONFIG.md`, `TRAINING.md`, `EVALUATE.md`, `REPRODUCIBILITY.md`, `TESTING.md`, `NOTEBOOKS.md`, `MAINTENANCE.md` | One owner per subsystem |
| Model guides | `codebase/METAUNETR.md`, `RESUNET3D.md`, `SWINUNETR.md`, `TRANSUNET.md`, `CNN_PRETRAINING.md` | Model and experiment facts |
| Visual assets | `assets/codebase/*.svg` | Portable architecture, data, config, and model diagrams |

Model-guide tasks have disjoint files and may run in parallel after the skeleton exists. Cross-cutting guides establish shared vocabulary first. Final review waits for every writer.

---

### Task 1: Produce evidence-gated hygiene audit

**Files:** Create `docs/superpowers/audits/2026-09-05-codebase-hygiene-audit.md`; read active `src/`, `tests/`, `configs/`, `README.md`, and active docs; exclude archive/generated state.

**Interfaces:** Consumes current active tree and baseline test collection. Produces candidate table consumed by Tasks 2 and 3, with explicit `remove`, `retain`, or `defer` decisions.

- [ ] Establish baseline with `git status --short`, focused collection, and focused execution of `tests/evaluation/test_inference.py`, `tests/training/test_tracking.py`, and `tests/training/test_task12_resume.py`. Record actual counts.
- [ ] Prove `_torch_load` references with `git grep`; distinguish used `src/token_mixer/training/checkpoints.py::_torch_load` from suspected dead `pretrain_cnn.py::_torch_load`.
- [ ] Prove `Path` in `tests/evaluation/test_inference.py` and `os` in `tests/training/test_tracking.py` have no active uses. If evidence differs, retain them.
- [ ] Search `test_task12_resume`, `Task 12`, `task12`, and `task-12`; separate active filename references from historical records.
- [ ] Classify `train_swinunetr.py`, `train_resunet3d.py`, `train_transunet.py`, and `_baseline_common.py` as `defer`: config behavior differs and tests patch private seams.
- [ ] Write report sections `Scope`, `Baseline`, `Evidence`, `Decisions`, `Deferred Work`, and `Validation`; do not create an automatic deletion list.
- [ ] Run `git diff --check`, stage only audit file, inspect staged diff, and commit `docs: record code hygiene audit`.

---

### Task 2: Remove proven dead helper and imports

**Files:** Modify `src/token_mixer/pipelines/pretrain_cnn.py` around `_torch_load`; `tests/evaluation/test_inference.py` import block; `tests/training/test_tracking.py` import block.

**Interfaces:** Consumes Task 1 `remove` classifications. Produces identical runtime behavior without the unused CNN `_torch_load`, `Path`, or `os` imports.

- [ ] Run focused inference/tracking tests before editing.
- [ ] Remove only the local `_torch_load` block in `pretrain_cnn.py`; do not alter `src/token_mixer/training/checkpoints.py::_torch_load`.
- [ ] Remove `from pathlib import Path` and `import os` only when Task 1 evidence confirms no use.
- [ ] Confirm no references remain in those active files; run focused tests, `uv run python -m compileall -q src tests`, and `git diff --check`.
- [ ] Stage only the three listed files, inspect staged diff, and commit `refactor: remove dead code and imports`.

---

### Task 3: Rename resume test suite by behavior

**Files:** Rename `tests/training/test_task12_resume.py` to `tests/training/test_resume_contract.py`; preserve historical specs, `.superpowers/`, Git history, and provenance.

**Interfaces:** Consumes Task 1 naming classification and Task 2-clean tree. Produces same 15-test resume contract suite under behavior-based filename.

- [ ] Use `git mv` so rename history is preserved.
- [ ] Update active references only. Do not rewrite historical `docs/superpowers/` or ignored `.superpowers/` records.
- [ ] Run collection and execution for `tests/training/test_resume_contract.py`; expected baseline is 15 collected/passed tests.
- [ ] Confirm no active `test_task12_resume` or task-number name remains in `src`, `tests`, root README, or `codebase`.
- [ ] Inspect staged rename and commit `test: rename resume contract suite`.

---

### Task 4: Create documentation skeleton and ownership index

**Files:** Create `codebase/README.md`, `codebase/ARCHITECTURE.md`, `codebase/MAINTENANCE.md`; modify root `README.md`.

**Interfaces:** Consumes cleaned active names and package boundary `CLI → pipelines → data / models / evaluation / training`. Produces canonical navigation, boundary guide, maintenance rules, and root link.

- [ ] Make `codebase/README.md` contain reading order, system map, cross-cutting guides, model guides, diagram assets, source-of-truth rule, and archive boundary. Link every planned guide and SVG.
- [ ] Make `ARCHITECTURE.md` explain ownership and dependency direction. Cite `src/token_mixer/cli.py::_dispatch` and package entrypoints with `path::symbol` references.
- [ ] Make `MAINTENANCE.md` define naming, removal evidence, boundary rules, test preservation, documentation updates, and SVG fact checklist.
- [ ] Add to root README: `For maintainer/researcher architecture and implementation guides, see [codebase/README.md](codebase/README.md).`
- [ ] Check `git diff --check`, repository-relative links, and absence of machine-specific paths/secrets; commit `docs: add codebase guide skeleton`.

---

### Task 5: Document data and configuration flows

**Files:** Create `codebase/DATA.md`, `codebase/CONFIG.md`, `assets/codebase/data-flow.svg`, `assets/codebase/config-flow.svg`; modify `codebase/README.md`.

**Interfaces:** Consumes `src/token_mixer/data/`, `configs/`, `prepare_data.py`, and existing data contracts. Produces two guides and two linked standalone SVGs.

- [ ] `DATA.md` covers source layouts, `prepare_brats`, `run_prepare`, `discover_cases`, labels, transforms, datasets, `SplitManifest`, local/cloud roots, manifest contract, preflight, and failure modes.
- [ ] Cite `prepare.py::prepare_brats`, `cases.py::discover_cases`, `labels.py::{detect_et_label,to_region_masks,regions_to_multiclass}`, split-manifest symbols, and dataset/transform builders.
- [ ] Preserve modality order `[t1n,t1c,t2w,t2f]`, region order `[ET,TC,WT]`, raw-label facts, 3-D NIfTI facts, and debug/full profile distinction.
- [ ] `CONFIG.md` covers Hydra entrypoint, config groups, local/cloud profiles, debug/full settings, output resolution, experiment selectors, model/data/run groups, and CLI overrides; cite `src/token_mixer/cli.py::main` and YAML paths.
- [ ] `data-flow.svg` shows approved source → preparation → canonical cases → discovery → manifest → datasets/transforms → training/evaluation/artifacts, plus separate CNN/ImageFolder branch.
- [ ] `config-flow.svg` shows profile/experiment/model/data/run composition resolving into runtime settings and CLI dispatch; do not imply composition itself trains.
- [ ] Parse both SVGs as XML, manually inspect arrows/labels/grayscale/facts, run `git diff --check`, and commit `docs: document data and config flows`.

---

### Task 6: Document training, evaluation, and reproducibility

**Files:** Create `codebase/TRAINING.md`, `codebase/EVALUATE.md`, `codebase/REPRODUCIBILITY.md`; modify `codebase/README.md`.

**Interfaces:** Consumes training/evaluation/reproducibility modules and Task 5 vocabulary. Produces cross-cutting guides used by model guides; no source changes.

- [ ] `TRAINING.md` covers pipeline-to-engine flow, `FitResult`, `PhaseSpec`, transitions, monitor direction, checkpoints/metadata, exact resume, warm start, W&B modes, tracker lifecycle, artifacts, failure provenance, and model exceptions.
- [ ] Cite `engine.py::{FitResult,fit}`, `phases.py::{PhaseSpec,apply_phase}`, `checkpoints.py::CheckpointManager`, `tracking.py::{Tracker,create_tracker}`, and pipeline entrypoints.
- [ ] `EVALUATE.md` covers logits-to-regions, Dice, HD95, sliding-window full-volume inference, spacing, slice-based TransUNet evaluation, overlays, histories, and reconstruction grids; cite metric/inference/visualization symbols.
- [ ] State 3-D versus 2-D protocol differences explicitly.
- [ ] `REPRODUCIBILITY.md` covers seeds, deterministic CUDA flags, DataLoader state, manifest identity/hash, composed config, code version, checkpoint metadata, W&B mode, profile separation, and no-real-data claims; cite `src/token_mixer/reproducibility.py`.
- [ ] Verify every guide has `path::symbol` sources and no ownership duplication; commit `docs: document training and evaluation`.

---

### Task 7: Document testing, notebooks, and maintenance workflow

**Files:** Create `codebase/TESTING.md`, `codebase/NOTEBOOKS.md`; modify `codebase/MAINTENANCE.md` and `codebase/README.md`.

**Interfaces:** Consumes test tree, notebook `.py` companions, cleanup rules, and AGENTS instructions. Produces dedicated active-test and notebook workflows.

- [ ] `TESTING.md` documents test layers, synthetic/debug harness, model/config contracts, data/manifest tests, resume/checkpoint contracts, optional skips, focused commands, full command, and what tests do not prove. Reference `tests/training/test_resume_contract.py`.
- [ ] `NOTEBOOKS.md` documents `notebooks/00_data_contract.py`, `01_preprocessing_smoke.py`, `02_model_shapes.py`, and `03_results_analysis.py`; state `.py` source-of-truth and one-way Jupytext sync.
- [ ] Add maintenance sequence: identify source of truth; search references; preserve seams; update owning guide/SVG; run focused/full/compile/lock/diff checks; exclude archive/generated state.
- [ ] Check links and forbidden absolute paths/secrets, inspect staged diff, and commit `docs: document testing and notebooks`.

---

### Task 8: Document MetaUNETR variants and mixer placement

**Files:** Create `codebase/METAUNETR.md`, `assets/codebase/model-placement-overview.svg`, `assets/codebase/metaunetr-variants.svg`; modify `codebase/README.md`.

**Interfaces:** Consumes `src/token_mixer/models/metaunetr/{network,encoder,decoder,mamba,variants}.py`, `train_metaunetr.py`, three experiment YAMLs, and tests. Produces shared guide and placement figures.

- [ ] Document `MetaUNETR`, `Encoder3D`, `CnnDecoder`, `MambaDecoder3D`, `Mamba`, `CrossScan3D`, `TriCruciMamba3D`, and `build_metaunetr`: stages, activations/normalizations/projections, skips, mixer placement, fallback/external Mamba, settings, and outputs.
- [ ] Compare exact rows `metaunetr_mamba | bottleneck mixer`, `mod_a | encoder-stage mixer`, and `mod_b | coarse decoder refinement mixer`.
- [ ] Include verified `[B,4,D,H,W]`, token-scan, and `[B,3,D,H,W]` shapes plus phase/loss/monitor settings; do not imply official Mamba-3 MIMO behavior.
- [ ] Draw aligned panels, grouping borders, orthogonal connectors, input/output arrows, short labels, and legend; manually fact-check against source.
- [ ] Parse SVGs, run `git diff --check`, inspect staged diff, and commit `docs: document MetaUNETR variants`.

---

### Task 9: Document ResUNet3D

**Files:** Create `codebase/RESUNET3D.md`, `assets/codebase/resunet3d.svg`; modify `codebase/README.md`.

**Interfaces:** Consumes `src/token_mixer/models/resunet3d.py`, `train_resunet3d.py`, weight-transfer module, configs, and tests. Produces five-stage residual U-Net guide and SVG.

- [ ] Cite `ResBlock3D`, `EncStage`, `DecStage`, `ResUNetEncoder`, `ResUNetDecoder`, `ResUNet3D`, and `build_resunet3d`.
- [ ] Cover stages, residual blocks, downsampling, skip alignment, normalization, widths/depths, freeze/unfreeze phases, logits/loss/monitor, transfer flags, and provenance.
- [ ] Distinguish timm ResNet-18 transfer from CNN `encoder_best.pth`.
- [ ] Draw verified channel/spatial boundaries and optional transfer marker; parse SVG, inspect facts, run `git diff --check`, and commit `docs: document ResUNet3D`.

---

### Task 10: Document SwinUNETR

**Files:** Create `codebase/SWINUNETR.md`, `assets/codebase/swinunetr.svg`; modify `codebase/README.md`.

**Interfaces:** Consumes `src/token_mixer/models/swinunetr.py`, MONAI transform builder, pipeline/config/tests, and MONAI boundary. Produces adapter/transform guide and SVG.

- [ ] Cite `SwinUNETRAdapter` and `build_swinunetr`; cover lazy MONAI import, config aliases, spatial/window constraints, encoder exposure, forward shape, transforms, loss/phase/monitor, and environment-gated behavior.
- [ ] Draw input, hierarchical Swin stages, decoder/skips, adapter boundary, and logits with verified annotations.
- [ ] Parse SVG, manually inspect, run `git diff --check`, and commit `docs: document SwinUNETR`.

---

### Task 11: Document TransUNet boundary

**Files:** Create `codebase/TRANSUNET.md`, `assets/codebase/transunet.svg`; modify `codebase/README.md`.

**Interfaces:** Consumes `src/token_mixer/models/transunet.py`, `train_transunet.py`, configs, tests, and README environment contract. Produces explicit 2-D slice-adapter guide and SVG.

- [ ] Cite validation, metadata, output adaptation, resize, adapter, loader, and builder symbols.
- [ ] Cover checkout/pretrained `.npz` requirements, lazy import/validation, input-stem adaptation, slice resize, output kinds, four-class-to-three-region conversion, unit spacing, CLI overrides, failures, and no external-asset validation claim without assets.
- [ ] Draw volume-to-slice adapter, external R50-ViT boundary, output adaptation, reconstruction/metric boundary, and canonical logits; label 2-D/3-D protocol differences.
- [ ] Parse SVG, manually inspect facts, run `git diff --check`, and commit `docs: document TransUNet`.

---

### Task 12: Document CNN denoising pretraining

**Files:** Create `codebase/CNN_PRETRAINING.md`, `assets/codebase/cnn-pretraining.svg`; modify `codebase/README.md`.

**Interfaces:** Consumes `src/token_mixer/models/cnn_pretrain.py`, `pretrain_cnn.py`, ImageFolder config, and tests. Produces separate ImageFolder/pretraining guide and SVG.

- [ ] Cite `DenoisingDataset`, `PretrainCNNEncoder`, `DecoderStage2D`, `DenoisingAutoencoder`, `build_denoising_model`, `mse_loss`, `compute_psNR`, and `evaluate_denoising`.
- [ ] Cover ImageFolder layout, seeded split, channel contract, encoder/decoder stages, MSE/PSNR, `drop_last` image-count constraints, reconstruction-grid override, `encoder_best.pth`, and `metrics.json` with `test_metrics: null`.
- [ ] Draw input images → denoising encoder → decoder → reconstruction/validation metric → exported encoder artifact; keep CNN separate from BraTS flow.
- [ ] Parse SVG, manually inspect facts, run `git diff --check`, and commit `docs: document CNN pretraining`.

---

### Task 13: Complete final index, fact review, and validation

**Files:** Review all active guides/assets; modify root `README.md`, `codebase/README.md`, or guides only for links/factual corrections. Never stage archive/generated state.

**Interfaces:** Consumes every prior guide/asset and source reference. Produces reviewed, cross-linked documentation with no stale active names or unsupported claims.

- [ ] Verify `codebase/README.md` links all fourteen guides and eight SVGs: `data-flow.svg`, `config-flow.svg`, `model-placement-overview.svg`, `metaunetr-variants.svg`, `resunet3d.svg`, `swinunetr.svg`, `transunet.svg`, and `cnn-pretraining.svg`.
- [ ] Search active `README.md`, `codebase`, `src`, and `tests` for machine-specific paths, MCP-only variables, stale `test_task12_resume`, archive internals, and generated-output instructions; expected no matches.
- [ ] Parse every SVG as XML and manually inspect title/description, legibility, grayscale clarity, arrow direction, source-fact alignment, and absence of invented layers/shapes.
- [ ] Run `uv lock --check`, `uv run pytest -q -rs`, `uv run python -m compileall -q archive src tests`, `git diff --check`, and `git status --short`.
- [ ] Expected baseline remains 403 passed and 6 skipped unless collection output changed for a documented reason; no new failures.
- [ ] Stage only intended README/docs/assets corrections, inspect staged diff, and commit `docs: complete codebase guide set`.

## Review and execution gates

- Task 1 audit must be reviewed before deletion or rename.
- Tasks 2 and 3 must preserve focused-test behavior before documentation work begins.
- Cross-cutting guides establish vocabulary before model-guide writers start.
- Model-guide tasks may run concurrently only with disjoint file scopes; each writer stops after listed files and returns validation evidence.
- Task 13 waits for every prior task to have a committed result or an explicitly user-approved exception.
- No step authorizes full training, cloud/GPU work, external data acquisition, or architecture redesign.
