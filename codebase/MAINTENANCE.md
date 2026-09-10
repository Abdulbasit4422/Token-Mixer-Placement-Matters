# Maintenance

This guide defines safe change practice for the active package and its curated
maintainer documentation. It favors evidence over visual cleanup: similar code
is not proof of dead code, and a shorter file is not proof of a better boundary.

## Source And Ownership Rules

Use the source-of-truth order from the [codebase index](README.md): active
`src/`, `configs/`, and `tests/` establish behavior and contracts; notebook
`.py` files establish notebook source; guides and SVGs explain those surfaces.
Do not make documentation, generated output, or archive code a substitute for
active implementation evidence.

Keep changes inside the owning boundary:

- CLI dispatch and top-level artifact handoff belong to `src/token_mixer/cli.py`.
- Experiment orchestration belongs to `src/token_mixer/pipelines/`.
- Case preparation, labels, manifests, datasets, and transforms belong to `src/token_mixer/data/`.
- Tensor-facing architecture and adapters belong to `src/token_mixer/models/`.
- Inference, metric conversion, metrics, and plots belong to `src/token_mixer/evaluation/`.
- Fit state, phases, checkpoints, tracking, and artifact payloads belong to `src/token_mixer/training/`.
- Hydra defaults and run choices belong to `configs/`, not hidden Python branches.

Preserve the package boundary `CLI -> pipelines -> data / models / evaluation /
training`. Shared helpers may expose intentional seams; do not merge wrappers
or move code only to reduce line count.

## Naming

Name active files and symbols after behavior, ownership, or domain concepts.
Avoid task numbers, temporary workflow labels, and implementation-history names
in active paths. The resume suite is named
`tests/training/test_resume_contract.py` because it protects resume behavior,
not a planning task.

Keep historical names in `archive/`, Git history, and planning/provenance
records when they are part of the record. Do not rewrite history to make active
naming look uniform.

Use exact package and configuration names already exposed by the active CLI:
`metaunetr_mamba`, `mod_a`, `mod_b`, `resunet3d`, `swinunetr`,
`transunet`, and `cnn_denoising_pretrain`. A rename that changes a selector,
checkpoint metadata field, or persisted artifact name is a behavior change and
requires separate review.

## Removal Evidence

Before removing a helper, import, test, guide, or asset:

1. Search active source, tests, configs, and project-facing docs for definitions, imports, calls, exports, fixtures, and links.
2. Check test collection and focused execution for the affected contract.
3. Classify the candidate as `remove`, `rename`, `retain`, or `defer` with the evidence recorded in an audit or task report.
4. Confirm it is not an optional-dependency seam, pipeline-private patch seam, compatibility path, provenance field, or reproducibility contract.
5. Make the smallest approved edit and rerun the focused checks before broad validation.

The earlier hygiene audit is the evidence record for this repository:
[`docs/superpowers/audits/2026-09-05-codebase-hygiene-audit.md`](../docs/superpowers/audits/2026-09-05-codebase-hygiene-audit.md).
It distinguishes the unused CNN pipeline `_torch_load` from the used
`src/token_mixer/training/checkpoints.py::_torch_load`, and retains wrapper and
test seams whose behavior differs. Matching names alone are not removal proof.

Never remove a test because another test appears to cover the same happy path.
Keep tests that protect checkpoints, resume/warm-start semantics, provenance,
optional dependencies, tensor contracts, failure behavior, or private seams
patched by active pipeline tests.

## Archive And Generated-State Boundary

Do not edit or reorganize `archive/` as part of active maintenance. It is
historical and unsupported. Do not use archive internals as evidence for current
architecture or add archive details to new guides.

Keep generated and runtime state outside reviewed source changes:

- `outputs/`, checkpoints, W&B files, caches, and bytecode are not documentation inputs.
- Paired `.ipynb` files are generated views of notebook `.py` sources; edit `.py` first and sync one way with Jupytext.
- Machine-specific paths, local data contents, PHI, credentials, tokens, and external checkouts do not belong in committed docs, guides, or SVGs.
- Do not add a generator, link-check dependency, diagram generator, or generated API catalog for this documentation set.

## Test Preservation

For source changes, preserve the nearest contract tests and private seams first.
Run focused tests before and after the edit, then run the repository validation
ladder appropriate to the change. For documentation-only changes, do not alter
or regenerate tests; validate Markdown, links, whitespace, and scoped Git state.

The standard evidence sequence for behavior changes is:

```text
focused collection -> focused execution -> compile check -> full suite -> diff check
```

The final repository plan also requires `uv lock --check`,
`uv run pytest -q -rs`, and `uv run python -m compileall -q archive src tests`.
Do not start full training, external TransUNet execution, cloud/GPU work, or
real-data acquisition as a documentation or hygiene check.

## Documentation Update Rules

When active behavior changes:

1. Identify the source owner and confirm its public/fixture/test contract.
2. Search active references before editing names or paths.
3. Update the owning cross-cutting or model guide, not every guide that mentions the concept.
4. Update `codebase/README.md` only when navigation, ownership, or planned assets change.
5. Update the root `README.md` only for onboarding or the maintainer-guide link.
6. Add stable `path::symbol` references for new execution claims.
7. Update the owning SVG when a shown boundary, stage, tensor shape, or route changes.
8. Check that no link points to archive internals, runtime outputs, machine paths, or secrets.

One guide owns one subject. Link adjacent guides instead of duplicating their
contracts. A model guide may summarize the shared input/output contract, but
[DATA.md](DATA.md), [TRAINING.md](TRAINING.md), and [EVALUATE.md](EVALUATE.md)
remain owners of their respective shared behavior.

## SVG Fact Checklist

Before accepting a new or changed asset under `assets/codebase/`, verify every
item below against active source, config, or tests:

- [ ] SVG parses as XML and has standalone dimensions plus a stable `viewBox`.
- [ ] SVG contains accessible `<title>` and `<desc>` elements.
- [ ] Every meaningful edge has an understandable direction and arrowhead where needed.
- [ ] Every block, stage, operation, activation, normalization, projection, skip route, and optional boundary shown is present in source or explicitly labeled as an external boundary.
- [ ] Every tensor shape, channel count, spatial boundary, label order, dimensionality, and spacing annotation is verified; omit uncertain annotations.
- [ ] Model variant or mixer-placement differences match config and active implementation, without implying unsupported paper internals or exact reproduction.
- [ ] Color is paired with shape, border, or text so the figure remains readable in grayscale.
- [ ] No external fonts, scripts, remote assets, gradients, shadows, or generated layout dependency is required.
- [ ] Owning guide links the SVG and records the source `path::symbol` references used for its facts.
- [ ] Arrow direction, label legibility, grouping, and grayscale clarity receive a manual inspection after XML parsing.

## Maintenance Workflow

Use this sequence for each scoped maintenance change:

1. Read [ARCHITECTURE.md](ARCHITECTURE.md) and identify the owning boundary.
2. Locate active source, config, tests, and existing guide/SVG references; exclude archive and generated state.
3. Search definitions, imports, callers, selectors, fixtures, links, and persisted names before changing anything.
4. Record removal or rename evidence and preserve useful contract tests and private seams.
5. Edit the smallest owning source, test, guide, or asset set. Do not fold later documentation tasks into the current change.
6. Run focused checks, Markdown/link/whitespace checks, and source-specific validation; use the full plan validation before the final documentation release.
7. Review the diff for machine paths, secrets, unsupported claims, stale symbols, broken ownership, and accidental generated/archive files.
8. Stage only intended files, inspect the staged diff, and create one atomic commit with a behavior-based message.

For notebook changes, the editable source is the `.py` companion. For SVG
changes, complete the fact checklist above. For a selector or package-boundary
change, update the architecture map and affected model/config guide together.

## Task 4 Source References

- [`src/token_mixer/cli.py`](../src/token_mixer/cli.py)::`_dispatch`, `::_run`
- [`src/token_mixer/pipelines/_baseline_common.py`](../src/token_mixer/pipelines/_baseline_common.py)::`run_2d_baseline`, `::run_3d_baseline`
- [`src/token_mixer/training/engine.py`](../src/token_mixer/training/engine.py)::`fit`
- [`tests/training/test_resume_contract.py`](../tests/training/test_resume_contract.py)
- [`docs/superpowers/audits/2026-09-05-codebase-hygiene-audit.md`](../docs/superpowers/audits/2026-09-05-codebase-hygiene-audit.md)

This guide is intentionally procedural. It does not define runtime behavior or
authorize cleanup outside the active package.
