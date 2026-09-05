# Codebase Hygiene and Maintainer Documentation

**Status:** Design approved in conversation; implementation plan pending user review of this spec
**Date:** 2026-09-05

## Summary

Improve active-project clarity without removing useful research infrastructure, then
document the cleaned system for future maintainers and researchers. Work proceeds in
two phases:

1. Evidence-gated code/test hygiene and behavior-based naming.
2. Curated `codebase/` documentation with portable, hand-authored SVG diagrams.

`archive/` remains historical and is excluded from cleanup and detailed documentation.
No model, data, configuration, dependency, or training behavior changes are planned.

## Decisions

The following decisions were approved during brainstorming:

- Use one umbrella specification with sequential phases.
- Remove only proven dead code and unused imports; do not force consolidation.
- Rename active workflow-history names; preserve archive names and Git history.
- Optimize guides for future maintainers and research collaborators.
- Make SVG the default diagram format. Use Mermaid only when SVG would be materially
  worse for a specific diagram.
- Store reusable diagrams under top-level `assets/`, grouped in `assets/codebase/`.
- Hand-author SVGs and verify them with source references and a fact checklist; do not
  add a diagram generator or documentation-test framework.
- Target stage-level model detail plus meaningful activations, normalizations, skip
  routes, mixer placement, and verified tensor shapes.
- Treat tests and notebooks as first-class active surfaces with separate guides.
- Group `metaunetr_mamba`, `mod_a`, and `mod_b` in one `METAUNETR.md`; give each other
  active model family its own guide.
- Use atomic commits for dead-code cleanup, test/name cleanup, and documentation/assets.
- Prefer curated execution paths with stable `path::symbol` references over an
  exhaustive generated API catalog.

## Current system evidence

The repository currently follows this dependency direction:

```text
CLI → pipelines → data / models / evaluation / training
```

Important contracts to preserve and document:

- Configuration lives under `configs/`, not `config/`.
- BraTS inputs use four modalities in `[t1n, t1c, t2w, t2f]` order.
- Models emit three raw region-logit channels in `[ET, TC, WT]` order.
- Raw enhancing-tumor labels `3` and `4` are supported; `configs/data/brats.yaml`
  defaults to `data.et_label: 4`.
- Local profile uses debug settings and `data/local/brats`; cloud profile uses full
  settings and `data/cloud/brats`.
- Optional MONAI, Mamba, timm, W&B, and external TransUNet boundaries are intentionally
  lazy or explicitly configured.
- `tests/training/test_task12_resume.py` protects generic resume contracts and is useful
  coverage, but its filename leaks an implementation-history label.

The initial audit found no justification for broad wrapper/helper deduplication. The
pipeline wrappers have different configuration behavior, and tests patch private seams.
That work is deferred until a separate evidence-backed design exists.

## Phase 1: code and test hygiene

### Audit protocol

Before editing any candidate:

1. Search imports, calls, exports, fixtures, test collection, and documentation links.
2. Confirm whether candidate is genuinely unused, workflow-history-only, or still a
   useful compatibility/provenance seam.
3. Record classification as `remove`, `rename`, `retain`, or `defer`.
4. Make only approved low-risk changes.
5. Run focused checks, then the full suite.

No candidate is removed because it merely looks repetitive. Overlap is acceptable when
it protects a distinct public, pipeline, optional-dependency, or reproducibility contract.

### Initial cleanup candidates

These candidates have direct audit evidence and are the starting point, not an automatic
deletion list:

- Remove unused private `_torch_load` in `src/token_mixer/pipelines/pretrain_cnn.py`.
- Remove unused imports in `tests/evaluation/test_inference.py` and
  `tests/training/test_tracking.py`.
- Rename `tests/training/test_task12_resume.py` to behavior-based
  `tests/training/test_resume_contract.py`; update references and test collection.
- Search active source/tests/project-facing docs for task-number or workflow-history
  names. Rename only names that obscure behavior. Preserve historical references in
  `archive/`, commit history, and planning/provenance records.

Explicitly defer:

- merging `train_swinunetr.py`, `train_resunet3d.py`, and `train_transunet.py` wrappers;
- merging `_baseline_common.py` helpers solely to reduce line count;
- deleting overlapping resume tests that protect pipeline-level behavior;
- changing model, data, configuration, dependency, checkpoint, or training semantics.

### Phase 1 acceptance

- Removed symbols and imports have no remaining active references.
- Renamed test is collected under behavior-based name and retains all assertions.
- Focused tests pass for affected evaluation, tracking, and resume behavior.
- Full `uv run pytest -q -rs` passes with no new failures or unexplained skips.
- `uv run python -m compileall -q archive src tests` passes.
- `uv lock --check` and `git diff --check` pass.
- Cleanup commits contain no archive, generated output, data, or OpenCode changes.

## Phase 2: `codebase/` documentation

### Directory contract

Create the following curated documentation set:

```text
codebase/
├── README.md
├── ARCHITECTURE.md
├── DATA.md
├── CONFIG.md
├── TRAINING.md
├── EVALUATE.md
├── REPRODUCIBILITY.md
├── TESTING.md
├── NOTEBOOKS.md
├── MAINTENANCE.md
├── METAUNETR.md
├── RESUNET3D.md
├── SWINUNETR.md
├── TRANSUNET.md
└── CNN_PRETRAINING.md

assets/
└── codebase/
    ├── data-flow.svg
    ├── config-flow.svg
    ├── model-placement-overview.svg
    ├── metaunetr-variants.svg
    ├── resunet3d.svg
    ├── swinunetr.svg
    ├── transunet.svg
    └── cnn-pretraining.svg
```

`codebase/README.md` is the maintainer/researcher entrypoint. Root `README.md` keeps
clone/setup/first-run onboarding and links to the codebase index.

### Guide ownership

Each guide owns one subject and links to adjacent guides instead of duplicating them:

| Guide | Owns | Does not own |
| --- | --- | --- |
| `ARCHITECTURE.md` | Package layers, dependency direction, orchestration boundaries | Layer-specific implementation detail |
| `DATA.md` | Source layouts, preparation, case discovery, manifests, transforms, labels | Optimizer/checkpoint behavior |
| `CONFIG.md` | Hydra defaults, profile/experiment/run composition, output resolution | Model internals |
| `TRAINING.md` | Engine, phases, loaders, checkpoints, resume, warm start, W&B, artifacts | Metric definitions and notebook workflow |
| `EVALUATE.md` | Inference, spacing, metrics, label conversion, visualizations | Training orchestration |
| `REPRODUCIBILITY.md` | Seeds, deterministic settings, manifests, provenance, environment | General package setup |
| `TESTING.md` | Test taxonomy, fixtures, optional skips, validation commands | Production implementation |
| `NOTEBOOKS.md` | Notebook purposes/order and `.py` source-of-truth workflow | Package API reference |
| `MAINTENANCE.md` | Naming, boundaries, safe refactors, doc/SVG update checklist | Historical archive contents |
| Model guides | Layer sequence, operations, tensor boundaries, settings, decisions, source refs | Shared data/training explanations |

### Model guide content

Each model guide documents:

1. Purpose and supported CLI selector(s).
2. Source-of-truth modules and important `path::symbol` references.
3. Input/output tensor contracts and dimensionality.
4. Layer-by-layer encoder, mixer, decoder, activation, normalization, projection, and
   skip-connection sequence at stage granularity.
5. Config keys, defaults, phase settings, losses, and monitored metrics.
6. Variant/reference differences and decisions that prevent an exact paper-reproduction
   claim.
7. Optional dependencies and failure behavior.
8. Checkpoint/artifact outputs and relevant tests.
9. Diagram legend and fact-check status.

`METAUNETR.md` covers the shared network once, then explicitly compares:

- `metaunetr_mamba`: mixer at the bottleneck;
- `mod_a`: encoder-stage mixer placement;
- `mod_b`: coarse decoder refinement mixer placement;
- shared CNN decoder and final logits contract.

The other guides cover ResUNet3D, SwinUNETR, TransUNet, and CNN denoising pretraining.
TransUNet must clearly document its 2-D slice adapter, external checkout/pretrained
asset boundary, and mapping between its four-class output and the project’s three-region
contract. CNN documentation must remain separate from BraTS segmentation.

## Diagram design

### SVG requirements

SVG is the default because it is portable, embeddable, inspectable in Git, and suitable
for both codebase guides and future paper figures. Every SVG must:

- use a stable `viewBox` and standalone dimensions;
- embed styles and avoid external fonts, scripts, and remote assets;
- include accessible `<title>` and `<desc>` content;
- use explicit input/output labels and arrowheads on meaningful edges;
- show shapes only where verified by source code or tests;
- use consistent color-plus-shape semantics, with grayscale-readable borders;
- avoid decorative gradients, shadows, and unverified visual detail;
- be linked or embedded from the owning guide.

SVGs are hand-authored. No generator, schema, or new diagram dependency is part of this
work. Each guide maintains the evidence needed to update its SVG safely.

### Planned diagrams

- `data-flow.svg`: approved raw source → preparation → case discovery → split manifest →
  datasets/transforms → training → artifacts/evaluation.
- `config-flow.svg`: profile + experiment + model/data/run groups → resolved config →
  runtime/output/checkpoint paths.
- `model-placement-overview.svg`: shared encoder/decoder shape with Mamba placement for
  baseline, Mod A, and Mod B.
- Model SVGs: stage-level layer flow, tensor boundaries, skip routes, and model-specific
  operations.

### Mamba-inspired visual grammar

The official Mamba comparison asset uses aligned block panels, thick outer grouping,
orthogonal connectors, flat fills, shape-plus-color encoding, short labels, symbolic
annotations, and a single legend. These conventions are useful, but the project diagrams
must not imply official Mamba-3 internals or MIMO behavior when the implementation does
not use them.

Reference sources:

- [Mamba diagram asset](https://github.com/state-spaces/mamba/blob/main/assets/mamba3.png)
- [Mamba-2 implementation](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba2.py)
- [Mamba-3 implementation](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba3.py)
- [Mamba-3 paper, section 3.4](https://arxiv.org/html/2603.15569#S3.SS4)

Project diagrams should add verified local contracts at meaningful boundaries:

- model input: `[B, 4, D, H, W]`;
- token-mixer input: `[B, D, H, W, C]`;
- depth scan: `[B·H·W, D, C]`;
- height scan: `[B·D·W, H, C]`;
- width scan: `[B·D·H, W, C]`;
- logits: `[B, 3, D, H, W]`.

Avoid annotating every edge when it harms readability. Every annotation that remains must
be tied to a verified code contract.

Mermaid is allowed only for a relationship or flow where automatic text layout is clearly
more maintainable than a hand-authored SVG. Any exception must be stated in the owning
guide and use the same labels and contract vocabulary.

## Documentation fact-check protocol

Before accepting each guide or SVG:

1. Trace each execution claim to active source, config, or test.
2. Confirm every function name and path exists after Phase 1 renames.
3. Confirm tensor shapes, stage ordering, labels, spacing, and optional dependencies.
4. Confirm model variants differ only where source/config behavior differs.
5. Confirm outputs, failures, checkpoint semantics, and W&B modes.
6. Inspect SVG visually for alignment, legibility, arrow direction, and grayscale clarity.
7. Record the reviewed source paths/symbols and date in the guide.
8. Check cross-links and ensure no guide pulls in archive internals or generated output.

This is a manual review protocol, not a new runtime test suite. Existing tests remain the
behavior authority; source/configuration remain the implementation authority; guides and
SVGs explain those authorities.

## Phase 2 acceptance

- `codebase/README.md` links every guide and asset and is linked from root `README.md`.
- Every active model family and cross-cutting subsystem has one clear owner guide.
- Guides explain important paths/functions and decisions without duplicating obvious code.
- All planned SVGs render standalone and pass the SVG fact checklist.
- No diagram claims unsupported operations, shapes, or paper equivalence.
- Active guides contain no task-number naming except explicit historical context.
- Archive remains untouched and excluded from detailed documentation.
- Documentation links, paths, source symbols, and asset references are manually checked.
- Existing full test, compile, lock, and diff checks pass after documentation changes.

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| “Redundant” code still protects a contract | Require usage/evidence classification before removal; retain contract tests |
| Rename breaks hidden references | Search active tree, test collection, docs, and tooling before `git mv` |
| SVG drifts from implementation | Require source/symbol references, tensor checklist, and visual review |
| Guides become bloated | Curated execution paths, ownership boundaries, and explicit non-goals |
| Docs duplicate each other | One owner per subject; cross-link instead of copying |
| Mermaid/SVG style becomes inconsistent | Shared legend, palette, shapes, arrow rules, and asset review checklist |
| External/reference architecture is overstated | Label adapted/fallback paths and cite local implementation differences |

## Non-goals

- Rewriting model architecture or training behavior.
- Removing useful tests because they overlap.
- Consolidating all pipeline wrappers into one abstraction.
- Editing or reorganizing `archive/`.
- Adding a documentation generator, diagram generator, link-check dependency, or API-doc
  build system.
- Running full BraTS training, external TransUNet integration, or paid/cloud compute.
- Claiming paper metrics or exact reproduction from documentation alone.

## Transition to implementation planning

After the user reviews and approves this written spec, create an implementation plan with
bounded tasks and review gates in this order:

1. Candidate evidence report.
2. Proven dead-code/import cleanup.
3. Behavior-based test rename.
4. Documentation skeleton and index.
5. Cross-cutting guides.
6. Model guides and SVG assets.
7. Final link/fact/visual review and repository validation.

No implementation work starts before the written spec and subsequent implementation plan
are approved.
