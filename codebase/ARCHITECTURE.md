# Architecture

This guide records the active package boundary and dependency direction. It is
not an exhaustive API catalog. Runtime behavior remains owned by `src/`,
configuration by `configs/`, and contract expectations by `tests/`.

## Entry Point And Dispatch

The supported command is `python -m token_mixer` from repository root.

1. [`src/token_mixer/__main__.py`](../src/token_mixer/__main__.py)::`main` imports the Hydra entrypoint and calls it when the module is executed.
2. [`src/token_mixer/cli.py`](../src/token_mixer/cli.py)::`main` composes the default `local` configuration from `configs/` and enters `_run`.
3. `src/token_mixer/cli.py::_run` saves the composed configuration, calls `_dispatch`, preserves selected transfer-failure provenance, and writes run artifacts for a returned `FitResult`.
4. `src/token_mixer/cli.py::_dispatch` reads `experiment.name` and selects the runner without importing every optional model integration eagerly.

The dispatch table is the active experiment contract:

| `experiment.name` | Runner | Boundary |
| --- | --- | --- |
| `cnn_denoising_pretrain` | [`src/token_mixer/pipelines/pretrain_cnn.py`](../src/token_mixer/pipelines/pretrain_cnn.py)::`run_cnn_denoising_pretrain` | 2-D ImageFolder denoising pretraining |
| `metaunetr_mamba` | [`src/token_mixer/pipelines/train_metaunetr.py`](../src/token_mixer/pipelines/train_metaunetr.py)::`run_metaunetr` | 3-D MetaUNETR baseline |
| `mod_a` | `src/token_mixer/pipelines/train_metaunetr.py::run_metaunetr` | 3-D encoder-stage mixer variant |
| `mod_b` | `src/token_mixer/pipelines/train_metaunetr.py::run_metaunetr` | 3-D coarse-decoder mixer variant |
| `resunet3d` | [`src/token_mixer/pipelines/train_resunet3d.py`](../src/token_mixer/pipelines/train_resunet3d.py)::`run_resunet3d` | 3-D residual U-Net baseline and optional transfer |
| `swinunetr` | [`src/token_mixer/pipelines/train_swinunetr.py`](../src/token_mixer/pipelines/train_swinunetr.py)::`run_swinunetr` | MONAI 3-D baseline |
| `transunet` | [`src/token_mixer/pipelines/train_transunet.py`](../src/token_mixer/pipelines/train_transunet.py)::`run_transunet` | External 2-D slice adapter |

Unknown selectors raise from `_dispatch`. Data preparation is deliberately not
one of these selectors: [`src/token_mixer/pipelines/prepare_data.py`](../src/token_mixer/pipelines/prepare_data.py)::`run_prepare`
is a library/pipeline entrypoint for preparation work.

## Ownership Map

The primary runtime direction is:

```text
CLI -> pipelines -> data / models / evaluation / training
```

| Area | Owns | Does not own |
| --- | --- | --- |
| `src/token_mixer/cli.py` | Hydra entrypoint, selector validation, runner dispatch, composed-config persistence, and top-level artifact handoff | Model internals, dataset discovery, optimization logic, or metric definitions |
| `src/token_mixer/pipelines/` | Experiment orchestration: model construction, loader/evaluator selection, phases, checkpoint/tracker setup, fit invocation, and test evaluation | Reusable layer implementations or archive execution |
| `src/token_mixer/data/` | Case discovery/preparation, raw-label conversion, split manifests, datasets, and transforms | Optimizer state, W&B lifecycle, or model architecture |
| `src/token_mixer/models/` | Tensor-facing model constructors and adapters, including MetaUNETR, ResUNet3D, SwinUNETR, TransUNet, and CNN pretraining | Dataset paths, checkpoint persistence, W&B initialization, or plotting |
| `src/token_mixer/evaluation/` | Logit conversion, Dice/HD95 metrics, full-volume or slice evaluation, and visualizations | Training phases, checkpoint selection, or data preparation |
| `src/token_mixer/training/` | Shared fit engine, phase transitions, checkpoint state, tracking, and run-artifact payloads | Choosing which experiment runs or implementing model layers |
| `src/token_mixer/reproducibility.py` | Seed setup, deterministic flags, and worker seeding used by pipelines | Experiment selection or model-specific behavior |
| `configs/` | Hydra profiles, data/model/experiment/run groups, defaults, and runtime values | Python orchestration and generated run output |
| `tests/` | Executable contracts for package behavior, seams, optional boundaries, and failure modes | Production runtime decisions |

The table is an ownership map, not permission to move code across boundaries.
For example, baseline wrappers use shared adapters, while their loss, phase,
dimensionality, external-asset, and transfer behavior can remain different.

## Pipeline Flow

`_dispatch` delegates to a pipeline. The pipeline then composes the following
responsibilities:

1. Resolve the selected model builder and any optional dependency or asset boundary.
2. Build manifest-backed loaders through a model-specific or shared adapter.
3. Build the loss, evaluator, phases, checkpoint manager, tracker, and reproducibility generator.
4. Call the shared training engine [`src/token_mixer/training/engine.py`](../src/token_mixer/training/engine.py)::`fit`.
5. Restore the best checkpoint and evaluate the held-out split where that pipeline provides one.
6. Return `FitResult` to the CLI, which persists config, metrics, provenance, and checkpoint-related artifacts.

The shared baseline seam is [`src/token_mixer/pipelines/_baseline_common.py`](../src/token_mixer/pipelines/_baseline_common.py)::`run_3d_baseline` and `::run_2d_baseline`.
The 3-D path uses volume loaders and explicit physical spacing for evaluation.
The 2-D path uses slice loaders and the canonical slice evaluator; it is used by
the TransUNet adapter. The distinction is a runtime contract, not a diagrammatic
detail to collapse.

## Component Boundaries

### Data To Model

Data builders produce batches and canonical targets. Model builders consume
tensors and return model outputs. The canonical segmentation contract is four
MRI input channels for 3-D BraTS paths and three raw region-logit channels in
`[ET, TC, WT]` order. Data and evaluation share the region vocabulary from
`src/token_mixer/data/labels.py`; this shared vocabulary is intentional.

### Model To Training

Pipelines provide a model, loaders, loss, evaluator, phases, and runtime config
to `fit`. The training engine owns optimizer/scheduler state, phase transitions,
checkpoint state, and tracker lifecycle. It should not need to know the model
name beyond metadata supplied by the pipeline.

### Training To Evaluation

Validation metrics drive monitored checkpoint selection through the evaluator.
After fitting, baseline pipelines restore `best.pt` before held-out evaluation.
Evaluation owns metric conversion and spacing rules; training owns when the
evaluator runs and which checkpoint is authoritative.

### External And Optional Dependencies

Optional MONAI, Mamba, timm, W&B, and external TransUNet assets are boundaries
of the relevant model, pipeline, or tracking code. Keep imports and validation
at the existing boundary. Do not make a documentation change imply that an
external checkout, pretrained asset, cloud dataset, or GPU run was validated.

## Change Ownership

Use the narrowest owner when changing behavior:

| Change | First source to inspect | Guide to update |
| --- | --- | --- |
| New or changed experiment selector | `src/token_mixer/cli.py::_dispatch`, `configs/experiment/` | [CONFIG.md](CONFIG.md), affected model guide, and this guide if boundary changes |
| Loader, manifest, label, or transform behavior | `src/token_mixer/data/`, `src/token_mixer/pipelines/_baseline_common.py` | [DATA.md](DATA.md), [EVALUATE.md](EVALUATE.md) when metric inputs change |
| Model layers or tensor shapes | `src/token_mixer/models/`, model config, model tests | Model guide and owned SVG |
| Phase, resume, checkpoint, or tracking behavior | `src/token_mixer/training/`, pipeline adapters | [TRAINING.md](TRAINING.md), [REPRODUCIBILITY.md](REPRODUCIBILITY.md), [TESTING.md](TESTING.md) |
| Metric, spacing, inference, or visualization behavior | `src/token_mixer/evaluation/` and baseline evaluator adapters | [EVALUATE.md](EVALUATE.md) and owned SVG if a boundary is shown |
| Notebook workflow | `notebooks/*.py` and paired `.ipynb` | [NOTEBOOKS.md](NOTEBOOKS.md) |

## Explicit Boundaries

Do not treat these as active architecture surfaces:

- `archive/` is historical and unsupported.
- `outputs/`, checkpoints, W&B files, caches, bytecode, and notebook-generated views are runtime or generated state.
- Historical planning/specification records preserve provenance and are not active symbol references to rewrite during cleanup.
- A documentation guide or SVG is explanatory and cannot become a second implementation source of truth.

The architecture map should stay small enough to review against active source.
Wrapper consolidation, broad helper deduplication, archive cleanup, generated
documentation, and runtime behavior changes require separate evidence and scope.

## Source References

Primary references for this guide:

- [`src/token_mixer/__main__.py`](../src/token_mixer/__main__.py)::`main`
- [`src/token_mixer/cli.py`](../src/token_mixer/cli.py)::`main`, `::_run`, `::_dispatch`
- [`src/token_mixer/pipelines/_baseline_common.py`](../src/token_mixer/pipelines/_baseline_common.py)::`run_2d_baseline`, `::run_3d_baseline`
- [`src/token_mixer/training/engine.py`](../src/token_mixer/training/engine.py)::`fit`, `::FitResult`
- [`src/token_mixer/data/cases.py`](../src/token_mixer/data/cases.py)::`discover_cases`
- [`src/token_mixer/evaluation/inference.py`](../src/token_mixer/evaluation/inference.py)::`evaluate_full_volumes`
- [`src/token_mixer/evaluation/metrics.py`](../src/token_mixer/evaluation/metrics.py)::`logits_to_regions`, `::dice_by_region`, `::hd95_by_region`
- [`src/token_mixer/training/artifacts.py`](../src/token_mixer/training/artifacts.py)::`write_run_artifacts`, `::write_failed_run_artifact`
- [`src/token_mixer/models/metaunetr/variants.py`](../src/token_mixer/models/metaunetr/variants.py)::`build_metaunetr`
- [`src/token_mixer/models/resunet3d.py`](../src/token_mixer/models/resunet3d.py)::`build_resunet3d`
- [`src/token_mixer/models/swinunetr.py`](../src/token_mixer/models/swinunetr.py)::`build_swinunetr`
- [`src/token_mixer/models/transunet.py`](../src/token_mixer/models/transunet.py)::`build_transunet`
- [`src/token_mixer/models/cnn_pretrain.py`](../src/token_mixer/models/cnn_pretrain.py)::`build_denoising_model`

These references are intentionally stable file-and-symbol references rather
than generated API pages.
