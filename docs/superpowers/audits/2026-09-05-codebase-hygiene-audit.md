# Codebase Hygiene Audit

**Planned audit date:** 2026-09-05
**Evidence collected:** 2026-09-10
**Status:** Complete; evidence supports bounded follow-up cleanup and rename work. No source or test changes are part of this audit.

## Scope

This audit evaluates the current tracked active tree before any hygiene edit.
Included surfaces:

- `src/` production package code.
- `tests/` active tests and their test-collection names.
- `configs/` Hydra profiles, experiment groups, and model groups.
- Root `README.md` and `data/README.md`.
- Tracked active repository guidance in `MODAL.md`, read as part of active-doc
  scope review; no Modal code was written or run.
- Tracked project planning/specification documents under `docs/superpowers/`, searched for historical references but not treated as cleanup targets.

Excluded surfaces:

- `archive/`, which remains historical and is not a supported entry-point tree.
- Runtime outputs, checkpoints, W&B files, caches, bytecode, notebooks' generated views, and other generated state.
- `.superpowers/` working records, except for the required task handoff report.

The audit is evidence collection only. It does not remove code, remove imports,
rename tests, consolidate wrappers, or change model, data, configuration,
dependency, checkpoint, training, evaluation, or runtime behavior.

## Baseline

### Worktree

Command:

```text
git status --short
```

Result: no output. The tracked worktree was clean before audit changes.

### Focused collection

Command:

```text
uv run pytest --collect-only -q tests/evaluation/test_inference.py tests/training/test_tracking.py tests/training/test_task12_resume.py
```

Combined result: **36 tests collected**.

| Active test file | Collected |
| --- | ---: |
| `tests/evaluation/test_inference.py` | 15 |
| `tests/training/test_tracking.py` | 6 |
| `tests/training/test_task12_resume.py` | 15 |
| **Total** | **36** |

The per-file counts were confirmed with the same `--collect-only -q` command
run once for each file.

### Focused execution

Command:

```text
uv run pytest -q tests/evaluation/test_inference.py tests/training/test_tracking.py tests/training/test_task12_resume.py
```

Result: **36 passed in 110.42s**.

An initial invocation with a 120-second tool timeout exceeded that tool limit;
the same focused command was rerun with a 300-second timeout and completed with
no test failures.

## Evidence

### `_torch_load` references

Commands:

```text
git grep -n -e '_torch_load' -- src/token_mixer tests configs README.md data docs
git grep -n -e '_torch_load' -- archive
```

Active code results:

- `src/token_mixer/training/checkpoints.py:52` defines `_torch_load`, and
  `src/token_mixer/training/checkpoints.py:332` calls it from
  `CheckpointManager.read`. This is used and must remain.
- `src/token_mixer/pipelines/pretrain_cnn.py:583` defines a second private
  `_torch_load`, but the active file has no call to it and no active source,
  test, config, README, or data reference imports or calls it. The CNN pipeline
  restores its best checkpoint through the direct
  `CheckpointManager.load_model` call in
  `src/token_mixer/pipelines/pretrain_cnn.py:601`; the implementation begins
  at `src/token_mixer/training/checkpoints.py:382`. The pipeline invokes its
  local restore helper at `pretrain_cnn.py:827`, which reaches that direct call.
- The remaining active-tree matches are planning/specification text describing
  this audit and follow-up task. They are not runtime references.
- The archive search returned no `_torch_load` matches.

This distinguishes the used checkpoint loader from the suspected dead CNN
helper without treating matching names as interchangeable.

### Unused test imports

Commands:

```text
git grep -n -E '(^|[^[:alnum:]_])Path([^[:alnum:]_]|$)' -- tests/evaluation/test_inference.py
git grep -n -E '(^|[^[:alnum:]_])os([^[:alnum:]_]|$)' -- tests/training/test_tracking.py
```

Results:

- `tests/evaluation/test_inference.py:1:from pathlib import Path` was the
  only `Path` match. No test, fixture, annotation, or other active use exists
  in that file.
- `tests/training/test_tracking.py:2:import os` was the only `os` match. No
  test, fixture, or other active use exists in that file.

Both imports are candidates for removal after the focused tests are rerun by
the cleanup task.

### Resume naming and task-history references

Commands:

```text
git ls-files -- '*task12*' '*task-12*'
git grep -n -E 'test_task12_resume|Task 12|task12|task-12' -- src tests configs README.md data
git grep -n -E 'test_task12_resume|Task 12|task12|task-12' -- docs
git grep -n -E 'test_task12_resume|Task 12|task12|task-12' -- archive
```

Results:

- The active filename search returned exactly
  `tests/training/test_task12_resume.py`.
- The active content search over `src/`, `tests/`, `configs/`, README, and
  data returned no matches. The filename is therefore the active history leak;
  the test contents themselves describe generic resume, warm-start,
  checkpoint, and artifact contracts.
- Historical planning/specification matches are limited to the tracked
  `docs/superpowers/` records, including the earlier package-rebuild Task 12
  section and the approved hygiene plan/spec references to this audit and its
  future rename. These records are provenance and planning history, not active
  workflow names, so they remain unchanged.
- The archive search returned no matches.

The active suite can be renamed to `tests/training/test_resume_contract.py`
while preserving all 15 tests and their assertions. The rename must update
active references only; historical planning/provenance records remain intact.

### Wrapper and helper overlap

The following commands and source/config/test reads were used to assess
deduplication risk:

```text
git grep -n -E 'train_swinunetr|train_resunet3d|train_transunet|_baseline_common' -- src tests configs
```

Evidence against a line-count-only merge:

- `src/token_mixer/cli.py:51-62` dispatches `resunet3d`, `swinunetr`, and
  `transunet` through separate modules and runners.
- `configs/experiment/swinunetr.yaml:9-32` selects a single `dice_ce` train
  phase, while `configs/experiment/resunet3d.yaml:9-37` selects BCE loss and
  separate `encoder_frozen` and `full_finetune` phases. The ResUNet model
  config also owns explicit ImageNet transfer settings at
  `configs/model/resunet3d.yaml:10-16`.
- `configs/experiment/transunet.yaml:9-32` is a separate single-phase path;
  `configs/model/transunet.yaml:3-12` records its 2-D input size, external
  class count, canonical class count, and pretrained boundary.
- `src/token_mixer/pipelines/train_resunet3d.py:31-248` contains transfer
  option parsing, source-model injection, transfer counts, and failure
  provenance that do not belong to the SwinUNETR wrapper.
- `src/token_mixer/pipelines/train_transunet.py:51-88` validates an external
  checkout before dispatch unless a model is injected, and uses the 2-D
  baseline path. This is different from both native 3-D wrappers.
- `src/token_mixer/pipelines/_baseline_common.py` owns shared canonical loader,
  evaluation, checkpoint, phase, and 2-D/3-D orchestration adapters. It is
  imported by all three wrappers and directly by active tests.
- `tests/pipelines/test_baseline_pipelines.py:253-358` parameterizes all three
  wrapper entry points but patches each module's builder seams independently;
  it also asserts 3-D spacing metadata versus TransUNet 2-D metadata and
  checks the TransUNet pre-dispatch validation boundary.
- `tests/pipelines/test_train_resunet3d.py:78-195` patches the ResUNet transfer
  seam and verifies loaded counts, failure provenance, and the absence of fit
  on transfer failure.
- `tests/training/test_task12_resume.py:13-23` imports private
  `_baseline_common` resume helpers, and its later tests patch pipeline-private
  seams. These are intentional test contracts, not proof of duplicate dead
  code.

The wrappers and `_baseline_common.py` therefore have distinct configuration,
dimensionality, optional-dependency, provenance, and test-seam behavior.

## Decisions

The table records evidence-backed classifications for follow-up tasks. It is a
decision record, not an executable deletion list.

| Candidate | Evidence-backed decision | Follow-up |
| --- | --- | --- |
| `src/token_mixer/pipelines/pretrain_cnn.py::_torch_load` | **remove**; definition-only in active code, with checkpoint loading already owned by `CheckpointManager` | Task 2; remove only this local helper and rerun focused checks |
| `src/token_mixer/training/checkpoints.py::_torch_load` | **retain**; called by `CheckpointManager.read` | No change |
| `from pathlib import Path` in `tests/evaluation/test_inference.py` | **remove**; import-only match with no active use | Task 2; rerun inference tests |
| `import os` in `tests/training/test_tracking.py` | **remove**; import-only match with no active use | Task 2; rerun tracking tests |
| `tests/training/test_task12_resume.py` filename | **rename, retaining behavior**; active filename leaks implementation history while 15 generic resume-contract tests remain useful | Task 3; use `git mv` to `tests/training/test_resume_contract.py`, update active references, preserve all assertions |
| `train_swinunetr.py`, `train_resunet3d.py`, `train_transunet.py` | **defer**; wrappers differ in loss/phases, dimensionality, external assets, transfer behavior, and private seams | Separate evidence-backed design only; no merge in hygiene cleanup |
| `_baseline_common.py` helpers | **defer**; shared helpers are active imports and define canonical 2-D/3-D behavior used by tests and pipelines | Separate design and contract tests required before any consolidation |
| Overlapping resume/pipeline contract tests | **retain**; overlap protects checkpoint, resume, warm-start, provenance, and pipeline-boundary behavior | No deletion based on repetition |
| Archive and historical planning/provenance names | **retain/exclude**; they are explicitly outside active cleanup scope | No edits |

## Deferred Work

Wrapper/helper deduplication is deferred. Future work would need to show that
configuration behavior, dimensionality boundaries, optional dependency timing,
failure provenance, checkpoint semantics, and patched private seams can be
preserved without adding a more complex abstraction. Similar-looking wrapper
length is not sufficient evidence.

Task 2 should independently re-run reference searches after removing only the
three proven dead imports/helpers. Task 3 should use a Git rename, collect the
renamed suite, and search active paths again. Neither task is authorized to
rewrite historical `docs/superpowers/` records, archive files, generated state,
or runtime behavior.

## Validation

- Required focused collection completed: 36 tests collected, split 15/6/15
  across inference/tracking/resume files.
- Required focused execution completed: 36 passed, no failures or skips.
- Initial 120-second execution timeout was superseded by a successful
  300-second invocation; the timeout did not produce a test failure.
- No full suite, compile pass, real-data run, external TransUNet run, full
  training run, cloud/GPU work, or external-data operation was started.
- `git diff --check` completed with no output after the audit fix.
- `git diff --cached --check` completed with no output after staging; staged
  file inspection showed only this audit path, and the staged diff contained
  only the four review corrections.
- The fix commit and final handoff status are recorded in
  `.superpowers/sdd/task-1-report.md`.
