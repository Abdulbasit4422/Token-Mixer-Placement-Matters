from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .cases import CaseRecord


def _validate_seed(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("seed must be an integer")
    return value


def _validate_dataset_id(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("dataset_id must be a string")
    return value


def _validate_fraction(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite number between 0 and 1")
    value = float(value)
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be finite and between 0 and 1")
    return value


def _validate_fraction_pair(
    val_fraction: Any,
    test_fraction: Any,
    *,
    allow_missing: bool,
) -> tuple[float | None, float | None]:
    if val_fraction is None or test_fraction is None:
        if allow_missing and val_fraction is None and test_fraction is None:
            return None, None
        raise ValueError("fraction metadata must include val_fraction and test_fraction")

    val_fraction = _validate_fraction("val_fraction", val_fraction)
    test_fraction = _validate_fraction("test_fraction", test_fraction)
    if val_fraction + test_fraction > 1.0:
        raise ValueError("val_fraction and test_fraction must sum to at most 1")
    return val_fraction, test_fraction


def _validate_split_list(name: str, values: Any) -> None:
    if not isinstance(values, list) or any(not isinstance(case_id, str) for case_id in values):
        raise ValueError(f"{name} split IDs must be a list of strings")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} split IDs must be unique")


def _validate_split_lists(train: Any, val: Any, test: Any) -> None:
    split_values = {"train": train, "val": val, "test": test}
    for name, values in split_values.items():
        _validate_split_list(name, values)
    all_ids = train + val + test
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("split IDs must be unique across train, val, and test")


@dataclass(frozen=True)
class SplitManifest:
    """Persisted case IDs for one deterministic dataset split."""

    seed: int
    dataset_id: str
    train: list[str]
    val: list[str]
    test: list[str]
    # Defaults preserve five-argument construction used by existing callers.
    val_fraction: float | None = field(default=None, repr=False)
    test_fraction: float | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        _validate_seed(self.seed)
        _validate_dataset_id(self.dataset_id)
        _validate_split_lists(self.train, self.val, self.test)
        object.__setattr__(self, "train", sorted(self.train))
        object.__setattr__(self, "val", sorted(self.val))
        object.__setattr__(self, "test", sorted(self.test))
        val_fraction, test_fraction = _validate_fraction_pair(
            self.val_fraction,
            self.test_fraction,
            allow_missing=True,
        )
        if val_fraction is None:
            total = len(self.train) + len(self.val) + len(self.test)
            if total == 0:
                val_fraction = test_fraction = 0.0
            else:
                val_fraction = len(self.val) / total
                test_fraction = len(self.test) / total
        object.__setattr__(self, "val_fraction", val_fraction)
        object.__setattr__(self, "test_fraction", test_fraction)


def create_split_manifest(
    cases: Sequence[CaseRecord],
    seed: int,
    val_fraction: float,
    test_fraction: float,
    dataset_id: str,
) -> SplitManifest:
    """Create one reproducible, non-overlapping split of case IDs."""
    _validate_seed(seed)
    _validate_dataset_id(dataset_id)
    val_fraction = _validate_fraction("val_fraction", val_fraction)
    test_fraction = _validate_fraction("test_fraction", test_fraction)
    if val_fraction + test_fraction > 1.0:
        raise ValueError("val_fraction and test_fraction must sum to at most 1")

    case_ids = [case.case_id for case in cases]
    if any(not isinstance(case_id, str) for case_id in case_ids):
        raise ValueError("case IDs must be strings")
    case_ids.sort()
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("case IDs must be unique")

    shuffled = np.asarray(case_ids, dtype=object)
    np.random.default_rng(seed).shuffle(shuffled)
    test_count = int(len(case_ids) * test_fraction)
    val_count = int(len(case_ids) * val_fraction)

    test = sorted(str(case_id) for case_id in shuffled[:test_count])
    val_start = test_count
    val = sorted(str(case_id) for case_id in shuffled[val_start : val_start + val_count])
    train = sorted(str(case_id) for case_id in shuffled[val_start + val_count :])

    return SplitManifest(
        seed=seed,
        dataset_id=dataset_id,
        train=train,
        val=val,
        test=test,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )


def save_split_manifest(manifest: SplitManifest, path: Path) -> None:
    """Write a human-readable, canonical JSON split manifest."""
    _validate_fraction_pair(
        manifest.val_fraction,
        manifest.test_fraction,
        allow_missing=False,
    )
    _validate_seed(manifest.seed)
    _validate_dataset_id(manifest.dataset_id)
    _validate_split_lists(manifest.train, manifest.val, manifest.test)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": manifest.seed,
        "dataset_id": manifest.dataset_id,
        "val_fraction": manifest.val_fraction,
        "test_fraction": manifest.test_fraction,
        "train": sorted(manifest.train),
        "val": sorted(manifest.val),
        "test": sorted(manifest.test),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_split_manifest(path: Path) -> SplitManifest:
    """Load and validate a split manifest written by ``save_split_manifest``."""
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read split manifest '{path}': {exc}") from exc

    required = {
        "seed",
        "dataset_id",
        "val_fraction",
        "test_fraction",
        "train",
        "val",
        "test",
    }
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError(
            "Invalid split manifest: missing required fields"
            + (
                f": {sorted(required - set(payload))}"
                if isinstance(payload, dict)
                else ""
            )
        )

    split_values = {name: payload[name] for name in ("train", "val", "test")}
    try:
        _validate_seed(payload["seed"])
        _validate_dataset_id(payload["dataset_id"])
        _validate_split_lists(
            split_values["train"], split_values["val"], split_values["test"]
        )
        val_fraction, test_fraction = _validate_fraction_pair(
            payload["val_fraction"],
            payload["test_fraction"],
            allow_missing=False,
        )
    except ValueError as exc:
        raise ValueError(f"Invalid split manifest: {exc}") from exc

    return SplitManifest(
        seed=payload["seed"],
        dataset_id=payload["dataset_id"],
        train=sorted(split_values["train"]),
        val=sorted(split_values["val"]),
        test=sorted(split_values["test"]),
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )
