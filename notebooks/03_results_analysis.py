# %% [markdown]
# # Results Analysis
#
# Read saved metric artifacts from a configurable results root and build
# source-level and model-level comparison tables. The default root is the
# repository-relative `outputs` directory. This notebook does not assume a
# particular machine path and does not run training.

# %%
from collections.abc import Mapping
import json
from pathlib import Path
import os
import sys

import pandas as pd


configured_root = os.environ.get("TOKEN_MIXER_REPO_ROOT")
REPO_ROOT = Path(configured_root).expanduser() if configured_root else Path.cwd()
if not (REPO_ROOT / "src" / "token_mixer").is_dir():
    REPO_ROOT = Path.cwd()
SOURCE_ROOT = REPO_ROOT / "src"
if SOURCE_ROOT.is_dir() and str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from token_mixer.data.labels import REGION_NAMES

configured_results_root = os.environ.get("TOKEN_MIXER_RESULTS_ROOT")
RESULTS_ROOT = (
    Path(configured_results_root).expanduser()
    if configured_results_root
    else REPO_ROOT / "outputs"
)
if not RESULTS_ROOT.is_absolute():
    RESULTS_ROOT = (Path.cwd() / RESULTS_ROOT).resolve()


def payload_rows(payload: object) -> list[dict[str, object]]:
    """Normalize common JSON metric layouts into row dictionaries."""
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, Mapping)]
    if not isinstance(payload, Mapping):
        return []

    for nested_key in ("metrics", "test_metrics", "results"):
        nested = payload.get(nested_key)
        if isinstance(nested, (Mapping, list)):
            rows = payload_rows(nested)
            if rows:
                return rows

    if payload and all(isinstance(value, Mapping) for value in payload.values()):
        flattened: dict[str, object] = {}
        for group, values in payload.items():
            for key, value in values.items():
                flattened[f"{group}_{key}"] = value
        return [flattened]
    return [dict(payload)]


def model_from_path(path: Path) -> str:
    known_models = (
        "metaunetr_mamba",
        "mod_a",
        "mod_b",
        "resunet3d",
        "swinunetr",
        "transunet",
        "cnn_denoising_pretrain",
    )
    path_text = "/".join(part.lower() for part in path.parts)
    for model in known_models:
        if model in path_text:
            return model
    return path.parent.name or "unknown"


def split_from_path(path: Path) -> str:
    stem = path.stem.lower()
    for split in ("train", "val", "validation", "test"):
        if split in stem:
            return split
    return "unknown"


def metric_paths(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".json", ".csv"}
        and "metric" in path.stem.lower()
    )


def read_metric_file(path: Path) -> list[dict[str, object]]:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path).to_dict(orient="records")
    with path.open("r", encoding="utf-8") as handle:
        return payload_rows(json.load(handle))


records: list[dict[str, object]] = []
read_errors: list[str] = []
for metric_path in metric_paths(RESULTS_ROOT):
    try:
        rows = read_metric_file(metric_path)
    except (OSError, ValueError, json.JSONDecodeError, pd.errors.ParserError) as exc:
        read_errors.append(f"{metric_path}: {exc}")
        continue
    try:
        source = metric_path.relative_to(RESULTS_ROOT).as_posix()
    except ValueError:
        source = str(metric_path)
    for row in rows:
        records.append(
            {
                "source": source,
                "model": model_from_path(metric_path),
                "split": split_from_path(metric_path),
                **{str(key): value for key, value in row.items()},
            }
        )

metrics_table = pd.DataFrame(records)
if metrics_table.empty:
    comparison_table = pd.DataFrame(columns=["model", "records"])
else:
    numeric_columns = metrics_table.select_dtypes(include="number").columns.tolist()
    grouped = (
        metrics_table.groupby("model", dropna=False)[numeric_columns]
        .mean(numeric_only=True)
        .reset_index()
        if numeric_columns
        else metrics_table[["model"]].drop_duplicates()
    )
    record_counts = (
        metrics_table.groupby("model", dropna=False)
        .size()
        .rename("records")
        .reset_index()
    )
    comparison_table = record_counts.merge(grouped, on="model", how="left")

print(f"Results root: {RESULTS_ROOT}")
print(f"Metric files read: {len(metric_paths(RESULTS_ROOT))}")
print(f"Metric rows: {len(metrics_table)}")
if read_errors:
    print("Files skipped because they could not be parsed:")
    for error in read_errors:
        print(f"- {error}")
if metrics_table.empty:
    print(
        "No saved metric files found. Run an approved experiment first, then "
        "rerun this notebook."
    )
else:
    print("\nSource metrics:")
    print(metrics_table.to_string(index=False))
    print("\nModel comparison (mean numeric metrics):")
    print(comparison_table.to_string(index=False))

# Keep canonical region names visible for downstream table interpretation.
print(f"Canonical segmentation regions: {REGION_NAMES}")
metrics_table
comparison_table
