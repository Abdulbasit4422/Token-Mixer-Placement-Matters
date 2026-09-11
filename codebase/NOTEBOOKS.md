# Notebooks

The `notebooks/` directory contains exploratory contract checks and results
tables, paired through Jupytext. The four active `.py` companions are the
editable source surface. Their paired `.ipynb` files are generated notebook
views for interactive use, not independent implementations.

## Execution Order

Run the companions in this order when investigating a new data or model change:

1. [`00_data_contract.py`](../notebooks/00_data_contract.py) checks labels and
   case metadata without requiring downloaded data.
2. [`01_preprocessing_smoke.py`](../notebooks/01_preprocessing_smoke.py) loads
   one available local case and checks preprocessing, or reports the expected
   layout when no fixture is present.
3. [`02_model_shapes.py`](../notebooks/02_model_shapes.py) runs tiny CPU shape
   checks across packaged model tracks and records optional skips.
4. [`03_results_analysis.py`](../notebooks/03_results_analysis.py) reads saved
   metric artifacts and builds comparison tables after an approved run.

The sequence is a debugging and interpretation workflow, not a replacement for
the active pytest suite. `03_results_analysis.py` depends on saved artifacts but
does not create them or run training.

## Notebook Map

### `00_data_contract.py`

Purpose: exercise package-wide segmentation labels and case-record contracts
without requiring a downloaded dataset.

It verifies that:

- `REGION_NAMES` is exactly `(ET, TC, WT)`.
- Raw labels using ET marker `4` and ET marker `3` are detected and converted
  to the same canonical region order.
- Region masks round-trip through canonical multiclass labels with
  `0=background`, `1=ET`, `2=TC`, and `3=WT`.
- A `CaseRecord` keeps the four canonical modality names and explicit paths.
- `discover_cases` can be called against the repository-relative local data
  convention and report how many complete cases are present.

It constructs metadata paths only; it does not load NIfTI files, download data,
or train a model. `TOKEN_MIXER_REPO_ROOT` may select a local repository root;
otherwise the notebook uses its current working directory and falls back when
the expected `src/token_mixer` package is not found.

### `01_preprocessing_smoke.py`

Purpose: load one complete case from the configured local data root and exercise
the package's inference-style preprocessing path.

The notebook:

- Seeds execution with `42`.
- Uses `TOKEN_MIXER_DATA_ROOT` when set, otherwise
  `data/local/brats` relative to the repository root.
- Discovers canonical cases and selects the first one when available.
- Builds a non-training `BratsPatchDataset` with a `32 x 32 x 32` patch,
  normalization enabled, and flips disabled.
- Checks four image channels and three target region channels, then prints
  shapes, dtypes, and region order.
- Reports the expected five-file case layout and exits normally when no complete
  local fixture is present.

The notebook never downloads data or trains. If a local case exists but
`nibabel` is unavailable, it reports that optional dependency boundary instead
of claiming preprocessing succeeded. It uses only a local fixture supplied by
the user; no path value belongs in committed documentation.

### `02_model_shapes.py`

Purpose: run tiny, deterministic CPU forwards against each packaged model track
without touching a dataset or starting training.

It loads `configs/run/debug.yaml` for the debug seed and embeds explicit small
model configurations. Every executed check seeds with the debug seed, creates a
CPU model, runs a `torch.no_grad()` forward, verifies the expected output shape,
and rejects non-finite outputs. Results are printed as `PASS` or `SKIP` and also
collected into a pandas shape table.

The checks cover:

- `metaunetr_mamba`, `mod_a`, and `mod_b` with a shared tiny 3-D contract.
- `resunet3d` with a tiny 3-D input and canonical three-region output.
- `cnn_denoising_pretrain` with a tiny 2-D input and shape-preserving output.
- `swinunetr` when MONAI and `einops` are available.
- `transunet` only when `TRANSUNET_ROOT` and `TRANSUNET_PRETRAINED` identify
  usable external assets; the values are read from the environment and never
  embedded in the source.

Missing optional dependencies or external assets produce explicit `SKIP` rows.
The notebook's tiny forward checks establish tensor and finite-output seams
only. They do not establish pretrained-weight correctness, GPU behavior,
convergence, or model quality.

### `03_results_analysis.py`

Purpose: normalize saved metric artifacts into source-level and model-level
comparison tables for interpretation after an approved run.

It uses `TOKEN_MIXER_RESULTS_ROOT` when set, otherwise the repository-relative
`outputs` directory. It recursively reads `.json` and `.csv` files whose stem
contains `metric`, handles common `metrics`, `test_metrics`, and `results`
payload shapes, records parse errors, infers model and split labels from paths,
and groups numeric values by model. It keeps canonical `(ET, TC, WT)` names
visible in the output.

When no results root or metric files are present, it prints an explanatory
message and returns empty tables. It does not run training, evaluate a
checkpoint, validate a manifest, or prove that files represent real data. The
model-level table is a convenience aggregation, not a statistical test or an
automatic scientific ranking.

## Source Of Truth

Edit `.py` companions first, normally in Neovim. The percent-format source is
where notebook code, markdown cells, imports, and execution logic are reviewed.
The paired `.ipynb` is a generated or interactive view and must not become a
second source of truth.

When a `.py` companion changes outside a live Jupyter/Jupyter MCP save event,
the paired notebook can be stale. Sync explicitly from the `.py` file:

```bash
uv run jupytext --sync notebooks/00_data_contract.py
uv run jupytext --sync notebooks/01_preprocessing_smoke.py
uv run jupytext --sync notebooks/02_model_shapes.py
uv run jupytext --sync notebooks/03_results_analysis.py
```

These are one-way sync commands: `.py` to `.ipynb`. Never run
`jupytext --sync` with an `.ipynb` path. Reverse sync can copy generated cell
outputs into the `.py` source and corrupt the editable file. This documentation
change does not require notebook synchronization because no notebook companion
was edited.

The pairing format is declared in [`pyproject.toml`](../pyproject.toml) as
`ipynb,py:percent`. Keep that setting and the four companion names stable unless
a separate notebook workflow change is reviewed.

## Local Inputs And Boundaries

The companions use environment variables only for local, optional inputs:

| Variable | Used by | Meaning |
| --- | --- | --- |
| `TOKEN_MIXER_REPO_ROOT` | `00`, `01`, `02`, `03` | Repository root override; defaults to the current working directory when it contains the package |
| `TOKEN_MIXER_DATA_ROOT` | `01` | Local prepared BraTS root override |
| `TOKEN_MIXER_RESULTS_ROOT` | `03` | Saved metrics root override |
| `TRANSUNET_ROOT` | `02` | External TransUNet checkout for opt-in shape integration |
| `TRANSUNET_PRETRAINED` | `02` | External TransUNet pretrained file for opt-in shape integration |

Do not commit values for these variables, local data, credentials, tokens, or
machine-specific paths. The notebooks may inspect `data/` and `outputs/` when
they exist, but those runtime or data trees are not documentation sources.

## Evidence Limits

Notebook output is exploratory evidence. A passing cell can show that a local
fixture, tiny tensor, or saved artifact reached the intended seam at that time.
It cannot establish real-data quality, convergence, reproducibility across
hardware, or scientific model ranking. Use the active tests in
[`TESTING.md`](TESTING.md) for executable contract evidence and retain the
manifest, composed config, code version, checkpoint, profile, and data status
when interpreting a future experiment.

Keep paired `.ipynb` files, output images, checkpoints, W&B state, caches, and
bytecode outside source review. Do not edit or reorganize [`archive/`](../archive/)
for notebook work; it is historical and unsupported.
