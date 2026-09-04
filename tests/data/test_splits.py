import json
from pathlib import Path

import pytest

from token_mixer.data.cases import CaseRecord
from token_mixer.data.splits import (
    SplitManifest,
    create_split_manifest,
    load_split_manifest,
    save_split_manifest,
)


def make_cases(count: int) -> list[CaseRecord]:
    return [CaseRecord(f"CASE{i:03d}", {}, Path("segmentation.nii.gz")) for i in range(count)]


def test_same_seed_produces_same_nonoverlapping_manifest(tmp_path: Path):
    cases = make_cases(20)
    first = create_split_manifest(cases, 42, 0.15, 0.10, "fixture")
    second = create_split_manifest(cases, 42, 0.15, 0.10, "fixture")
    assert first == second
    assert set(first.train).isdisjoint(first.val)
    assert set(first.train).isdisjoint(first.test)
    assert set(first.val).isdisjoint(first.test)

    path = tmp_path / "split.json"
    save_split_manifest(first, path)
    assert load_split_manifest(path) == first


def test_split_manifest_serializes_fractions_and_sorted_case_ids(tmp_path: Path):
    cases = [
        CaseRecord("CASE003", {}, Path("segmentation.nii.gz")),
        CaseRecord("CASE001", {}, Path("segmentation.nii.gz")),
        CaseRecord("CASE002", {}, Path("segmentation.nii.gz")),
        CaseRecord("CASE004", {}, Path("segmentation.nii.gz")),
        CaseRecord("CASE005", {}, Path("segmentation.nii.gz")),
    ]
    manifest = create_split_manifest(cases, 7, 0.2, 0.2, "fixture-v1")
    path = tmp_path / "split.json"

    save_split_manifest(manifest, path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["dataset_id"] == "fixture-v1"
    assert payload["seed"] == 7
    assert payload["val_fraction"] == 0.2
    assert payload["test_fraction"] == 0.2
    assert payload["train"] == sorted(payload["train"])
    assert payload["val"] == sorted(payload["val"])
    assert payload["test"] == sorted(payload["test"])
    assert sorted(payload["train"] + payload["val"] + payload["test"]) == [
        "CASE001",
        "CASE002",
        "CASE003",
        "CASE004",
        "CASE005",
    ]


@pytest.mark.parametrize(
    ("val_fraction", "test_fraction"),
    [(-0.1, 0.1), (0.1, -0.1), (0.61, 0.4), (0.5, 0.51)],
)
def test_split_manifest_rejects_invalid_fractions(
    val_fraction: float, test_fraction: float
):
    with pytest.raises(ValueError):
        create_split_manifest(make_cases(10), 42, val_fraction, test_fraction, "fixture")


def test_split_manifest_load_rejects_missing_fields(tmp_path: Path):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps({"seed": 1}), encoding="utf-8")

    with pytest.raises(ValueError, match="split manifest"):
        load_split_manifest(path)


def test_split_manifest_load_requires_fraction_metadata(tmp_path: Path):
    path = tmp_path / "missing-fractions.json"
    path.write_text(
        json.dumps(
            {
                "seed": 1,
                "dataset_id": "fixture",
                "train": ["CASE001"],
                "val": [],
                "test": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="fraction"):
        load_split_manifest(path)


def test_split_manifest_five_argument_construction_persists_observed_fractions(
    tmp_path: Path,
):
    manifest = SplitManifest(
        seed=42,
        dataset_id="fixture",
        train=["CASE001", "CASE002"],
        val=["CASE003"],
        test=["CASE004"],
    )
    path = tmp_path / "derived-fractions.json"

    assert manifest.val_fraction == 0.25
    assert manifest.test_fraction == 0.25

    save_split_manifest(manifest, path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["val_fraction"] == 0.25
    assert payload["test_fraction"] == 0.25
    assert load_split_manifest(path) == manifest


def test_split_manifest_save_uses_zero_fractions_for_empty_manifest(tmp_path: Path):
    manifest = SplitManifest(42, "fixture", [], [], [])
    path = tmp_path / "empty.json"

    assert manifest.val_fraction == 0.0
    assert manifest.test_fraction == 0.0

    save_split_manifest(manifest, path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["val_fraction"] == 0.0
    assert payload["test_fraction"] == 0.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seed", 42.0),
        ("seed", True),
        ("seed", "42"),
        ("dataset_id", 42),
        ("dataset_id", None),
    ],
)
def test_split_manifest_rejects_non_strict_core_types(field: str, value):
    kwargs = {
        "seed": 42,
        "dataset_id": "fixture",
        "train": ["CASE001"],
        "val": [],
        "test": [],
        "val_fraction": 0.1,
        "test_fraction": 0.2,
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=field):
        SplitManifest(**kwargs)


@pytest.mark.parametrize(
    "overrides",
    [
        {"train": ("CASE001",)},
        {"val": ["CASE001", 1]},
        {"test": ["CASE001", "CASE001"]},
        {"train": ["CASE001"], "val": ["CASE001"]},
    ],
)
def test_split_manifest_rejects_invalid_split_lists(overrides: dict):
    kwargs = {
        "seed": 42,
        "dataset_id": "fixture",
        "train": ["CASE001"],
        "val": [],
        "test": [],
        "val_fraction": 0.1,
        "test_fraction": 0.2,
    }
    kwargs.update(overrides)

    with pytest.raises(ValueError, match="split"):
        SplitManifest(**kwargs)


@pytest.mark.parametrize(
    ("val_fraction", "test_fraction"),
    [
        (-0.1, 0.1),
        (0.1, -0.1),
        (1.1, 0.0),
        (0.0, 1.1),
        (float("nan"), 0.0),
        (float("inf"), 0.0),
        (0.6, 0.5),
    ],
)
def test_split_manifest_rejects_invalid_fraction_metadata(
    val_fraction: float, test_fraction: float
):
    with pytest.raises(ValueError, match="fraction"):
        SplitManifest(
            seed=42,
            dataset_id="fixture",
            train=["CASE001"],
            val=[],
            test=[],
            val_fraction=val_fraction,
            test_fraction=test_fraction,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [("val_fraction", "0.1"), ("test_fraction", True)],
)
def test_split_manifest_load_rejects_invalid_serialized_fraction_types(
    tmp_path: Path, field: str, value
):
    payload = {
        "seed": 42,
        "dataset_id": "fixture",
        "train": ["CASE001"],
        "val": [],
        "test": [],
        "val_fraction": 0.1,
        "test_fraction": 0.2,
    }
    if field == "test_fraction":
        payload["val_fraction"] = 0.0
    payload[field] = value
    path = tmp_path / "invalid-fraction-type.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="fraction"):
        load_split_manifest(path)


def test_split_manifest_round_trip_accepts_explicit_dataclass():
    manifest = SplitManifest(
        seed=42,
        dataset_id="fixture",
        train=["CASE001"],
        val=[],
        test=[],
    )

    assert manifest.train == ["CASE001"]


def test_split_manifest_sorts_direct_lists_for_round_trip(tmp_path: Path):
    manifest = SplitManifest(
        seed=42,
        dataset_id="fixture",
        train=["CASE002", "CASE001"],
        val=["CASE004", "CASE003"],
        test=["CASE005"],
        val_fraction=0.4,
        test_fraction=0.2,
    )
    path = tmp_path / "unsorted.json"

    assert manifest.train == ["CASE001", "CASE002"]
    assert manifest.val == ["CASE003", "CASE004"]
    save_split_manifest(manifest, path)

    assert load_split_manifest(path) == manifest


def test_split_manifest_equality_includes_fraction_metadata():
    first = SplitManifest(
        seed=42,
        dataset_id="fixture",
        train=["CASE001"],
        val=[],
        test=[],
        val_fraction=0.1,
        test_fraction=0.2,
    )
    second = SplitManifest(
        seed=42,
        dataset_id="fixture",
        train=["CASE001"],
        val=[],
        test=[],
        val_fraction=0.2,
        test_fraction=0.2,
    )

    assert first != second
