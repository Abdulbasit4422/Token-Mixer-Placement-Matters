# Codebase Guides

Maintainer and researcher map for the active Token Mixer package. This index
describes where behavior lives, which guide owns each subject, and which
surfaces are intentionally outside the reviewed source tree.

## Reading Order

Read guides in this order when changing an existing experiment or onboarding a
new maintainer:

1. [Architecture](ARCHITECTURE.md) defines the runtime boundary and dependency direction.
2. [Data](DATA.md) defines input preparation, labels, manifests, loaders, and transforms.
3. [Configuration](CONFIG.md) explains Hydra composition, profiles, selectors, and overrides.
4. [Training](TRAINING.md) explains phases, checkpoints, resume, tracking, and artifacts.
5. [Evaluation](EVALUATE.md) explains inference, metrics, spacing, and visualizations.
6. [Reproducibility](REPRODUCIBILITY.md) explains seeds, deterministic execution, and provenance.
7. [Testing](TESTING.md) maps contract tests to the behaviors they protect.
8. [Notebooks](NOTEBOOKS.md) explains exploratory checks and the Jupytext source-of-truth rule.
9. [Model guides](#model-guides) explain architecture-specific tensor paths and experiment choices.
10. [Maintenance](MAINTENANCE.md) is the change checklist for source, tests, guides, and SVGs.

All fourteen guide files listed below are active reviewed guides. Links stay
here as the canonical navigation surface; each guide owns its implementation
details.

## System Map

The supported runtime path is:

```text
python -m token_mixer
  -> src/token_mixer/__main__.py::main
  -> src/token_mixer/cli.py::main
  -> src/token_mixer/cli.py::_run
  -> src/token_mixer/cli.py::_dispatch
  -> selected pipeline runner
  -> data / models / evaluation / training
  -> FitResult and run artifacts
```

`_dispatch` is the active selector boundary. It maps
`cnn_denoising_pretrain` to the CNN pretraining runner,
`metaunetr_mamba`, `mod_a`, and `mod_b` to the MetaUNETR runner, and
`resunet3d`, `swinunetr`, and `transunet` to their respective runners. Data
preparation is not a CLI experiment selector; its package entrypoint is
`src/token_mixer/pipelines/prepare_data.py::run_prepare`.

The package boundary is intentionally orchestration-first:

```text
CLI -> pipelines -> data / models / evaluation / training
```

The arrows identify ownership of orchestration, not a claim that every Python
import is a strict one-way layer. Canonical label names are shared by data,
model, and evaluation code, and the baseline adapter centralizes common loader,
metric, checkpoint, and training-engine seams.

## Cross-Cutting Guides

Each guide owns one subject. Cross-link to its owner instead of copying its
implementation details into another guide.

| Guide | Owns |
| --- | --- |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Package ownership, dispatch, dependency direction, and boundaries |
| [DATA.md](DATA.md) | Source layouts, preparation, discovery, labels, manifests, datasets, and transforms |
| [CONFIG.md](CONFIG.md) | Hydra groups, profiles, experiment selectors, run settings, and output resolution |
| [TRAINING.md](TRAINING.md) | Engine, phases, pipeline handoff, checkpoints, resume, warm start, tracking, and artifacts |
| [EVALUATE.md](EVALUATE.md) | Inference, spacing, Dice/HD95 metrics, logit conversion, and visualizations |
| [REPRODUCIBILITY.md](REPRODUCIBILITY.md) | Seeds, deterministic settings, manifest identity, provenance, and run separation |
| [TESTING.md](TESTING.md) | Test taxonomy, contract seams, optional skips, and validation commands |
| [NOTEBOOKS.md](NOTEBOOKS.md) | Notebook purposes, execution order, and one-way Jupytext synchronization |
| [MAINTENANCE.md](MAINTENANCE.md) | Naming, removal evidence, ownership changes, test preservation, and doc/SVG review |

## Model Guides

This index contains five active model guides. Each guide documents an active
package path and links to its owned architecture asset where one exists.

| Guide | Scope |
| --- | --- |
| [METAUNETR.md](METAUNETR.md) | Active; `metaunetr_mamba`, `mod_a`, and `mod_b` placement variants |
| [RESUNET3D.md](RESUNET3D.md) | Active; native 3-D residual U-Net baseline and ImageNet transfer boundary |
| [SWINUNETR.md](SWINUNETR.md) | Active; MONAI SwinUNETR adapter and native 3-D baseline path |
| [TRANSUNET.md](TRANSUNET.md) | Active; external 2-D slice adapter, checkout boundary, and canonical output conversion |
| [CNN_PRETRAINING.md](CNN_PRETRAINING.md) | Active; separate ImageFolder denoising pretraining and encoder export |

Model guides document architecture-specific behavior. They do not redefine the
shared data, training, evaluation, or reproducibility contracts.

## Diagram Assets

Portable, hand-authored SVGs belong under `assets/codebase/`. Each asset is
owned by the guide that explains it, uses repository-relative source references,
and must pass the fact checklist in [MAINTENANCE.md](MAINTENANCE.md).

| Asset | Status and owner |
| --- | --- |
| [data-flow.svg](../assets/codebase/data-flow.svg) | Active; owned by [DATA.md](DATA.md) |
| [config-flow.svg](../assets/codebase/config-flow.svg) | Active; owned by [CONFIG.md](CONFIG.md) |
| [model-placement-overview.svg](../assets/codebase/model-placement-overview.svg) | Active; owned by [METAUNETR.md](METAUNETR.md) |
| [metaunetr-variants.svg](../assets/codebase/metaunetr-variants.svg) | Active; owned by [METAUNETR.md](METAUNETR.md) |
| [resunet3d.svg](../assets/codebase/resunet3d.svg) | Active; owned by [RESUNET3D.md](RESUNET3D.md) |
| [swinunetr.svg](../assets/codebase/swinunetr.svg) | Active; owned by [SWINUNETR.md](SWINUNETR.md) |
| [transunet.svg](../assets/codebase/transunet.svg) | Active; owned by [TRANSUNET.md](TRANSUNET.md) |
| [cnn-pretraining.svg](../assets/codebase/cnn-pretraining.svg) | Active; owned by [CNN_PRETRAINING.md](CNN_PRETRAINING.md) |

All eight SVG assets are active hand-authored diagrams owned by the guides shown
above; their SVG files are maintained separately from this index. No generated
diagram or documentation output is a source of truth.

## Source Of Truth

Use this precedence when sources disagree:

1. Active implementation under `src/` defines runtime behavior.
2. Active Hydra files under `configs/` define composition and defaults.
3. Active tests under `tests/` define protected contracts, seams, and expected failures.
4. Notebook `.py` files under `notebooks/` are the editable source for paired notebooks; sync `.py` to `.ipynb` only.
5. Hand-authored guides and SVGs explain the first four surfaces and must cite stable `path::symbol` references.
6. Runtime outputs, checkpoints, W&B files, caches, bytecode, and paired notebook views are generated state, not implementation authority.

When documentation disagrees with source, fix the owning guide rather than
changing behavior to match prose. When a source symbol moves, update its guide,
index links, and any owned SVG in the same maintenance change.

## Archive Boundary

`archive/` is historical code, not a supported entry-point tree. It is excluded
from active ownership maps, removal decisions, detailed architecture claims,
and diagram source facts. Preserve archive filenames and historical planning or
provenance records unless a separate task explicitly changes history.

The same boundary applies to generated runtime state such as `outputs/`,
checkpoints, W&B files, caches, bytecode, and generated notebook views. Do not
document a machine-specific path, credential, token, or local runtime artifact.

## Index Maintenance

Add a link here when a new owned guide or diagram is reviewed. Keep one clear
owner per subject, retain repository-relative links, and leave implementation
details in the owning guide. The root [README.md](../README.md) remains the
onboarding and first-run document; this index is the maintainer/researcher
navigation layer.
