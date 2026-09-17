from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import token_mixer.pipelines.benchmark as pipeline
from token_mixer.pipelines import _baseline_common as common
from token_mixer.evaluation.benchmark import hash_case_id
from token_mixer.training.engine import FitResult


class _Tiny3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv3d(4, 2, kernel_size=1)
        self.decoder = nn.Conv3d(2, 3, kernel_size=1)

    def forward(self, image):
        return self.decoder(self.encoder(image))


def test_protocol_selection_keeps_families_separate():
    assert pipeline.select_protocol({"experiment": "mod_a"}) == "native_3d_full_volume"
    assert pipeline.select_protocol({"experiment": "resunet3d"}) == "native_3d_full_volume"
    assert pipeline.select_protocol({"experiment": "swinunetr"}) == "native_3d_full_volume"
    assert pipeline.select_protocol({"experiment": "transunet"}) == "transunet_2d_slice"
    assert pipeline.select_protocol({"experiment": "cnn"}) == "cnn_denoising_validation"
    assert pipeline.select_protocol({"experiment": "cnn_denoising_pretrain"}) == "cnn_denoising_validation"
    assert pipeline.select_protocol({"experiment": "cnn_pretrain"}) == "cnn_denoising_validation"

    with pytest.raises(ValueError, match="protocol|experiment"):
        pipeline.select_protocol({"experiment": "unknown"})


def test_auto_protocol_uses_experiment_family_default():
    assert pipeline.select_protocol(
        {"experiment": "transunet", "benchmark": {"protocol": "auto"}}
    ) == "transunet_2d_slice"


def test_benchmark_forwards_nested_efficiency_settings(monkeypatch: pytest.MonkeyPatch):
    calls = {}

    def fake_protocol(_model, _inputs, **kwargs):
        calls.update(kwargs)
        return {}

    monkeypatch.setattr(pipeline, "run_model_protocol", fake_protocol)

    pipeline._invoke_model_protocol(
        nn.Identity(),
        torch.zeros(1, 1),
        protocol="fixture",
        warmup_iterations=0,
        repetitions=1,
        batch_sizes=[1],
        checkpoint=Path("checkpoint.pt"),
        benchmark_cfg={},
        efficiency_cfg={
            "enabled": True,
            "nvml": {
                "enabled": True,
                "device_index": 3,
                "sample_interval_seconds": 0.25,
            },
            "profiler": {
                "enabled": True,
                "mac_tool": "thop",
                "flop_tool": "fvcore",
            },
        },
    )

    assert calls["power_enabled"] is True
    assert calls["power_device_index"] == 3
    assert calls["power_interval_seconds"] == pytest.approx(0.25)
    assert calls["profiler_enabled"] is True
    assert calls["mac_tool"] == "thop"
    assert calls["flop_tool"] == "fvcore"


def test_native_fixed_input_adds_missing_batch_dimension():
    value = pipeline._as_model_input(torch.zeros(4, 3, 5, 6), "resunet3d")

    assert tuple(value.shape) == (1, 4, 3, 5, 6)


def test_prepare_inputs_requires_explicit_synthetic_opt_in_for_loader_failure():
    malformed_loader = [object()]

    with pytest.raises(ValueError, match="prepare benchmark inputs|evaluation loader"):
        pipeline._prepare_inputs(
            {}, "cnn", malformed_loader, torch.device("cpu")
        )

    synthetic = pipeline._prepare_inputs(
        {"benchmark": {"allow_synthetic_inputs": True}},
        "cnn",
        malformed_loader,
        torch.device("cpu"),
    )
    assert tuple(synthetic.shape) == (1, 3, 32, 32)


def test_case_rows_are_typed_and_tracking_table_redacts_raw_case_ids():
    rows = pipeline._case_records_from_evaluator(
        {"case_records": [{"case_id": "SECRET-CASE"}]},
        "ResUNet3D",
        "native_3d_full_volume",
    )

    assert rows[0]["row_type"] == "case"
    assert rows[0]["case_id_hash"] == hash_case_id("SECRET-CASE")
    assert "case_id" not in rows[0]

    columns, table_rows = pipeline._table_payload(
        [{"row_type": "case", "case_id": "SECRET-CASE"}]
    )
    assert "case_id" not in columns
    assert hash_case_id("SECRET-CASE") in table_rows[0]
    assert "SECRET-CASE" not in repr(table_rows)


def test_case_records_canonicalize_raw_aliases_over_supplied_hashes():
    supplied_hash = hash_case_id("wrong-case")
    rows = pipeline._case_records_from_evaluator(
        {
            "case_records": [
                {"caseId": "RAW-CASE", "case_id_hash": supplied_hash},
                {"id": "RAW-ID", "case_id_hash": "invalid-opaque"},
                {"case_id_hash": "invalid-opaque"},
            ]
        },
        "ResUNet3D",
        "native_3d_full_volume",
    )

    assert rows[0]["case_id_hash"] == hash_case_id("RAW-CASE")
    assert rows[1]["case_id_hash"] == hash_case_id("RAW-ID")
    assert len(rows[2]["case_id_hash"]) == 16
    assert all(character in "0123456789abcdef" for character in rows[2]["case_id_hash"])


def test_baseline_training_history_table_redacts_case_aliases(tmp_path: Path):
    raw_ids = ["CASE-HISTORY-001", "BraTS-HISTORY-002"]
    captured = {}

    class Tracker:
        def log_summary(self, summary):
            captured["summary"] = dict(summary)

        def log_table(self, name, columns, rows):
            captured["table"] = (name, list(columns), [list(row) for row in rows])

        def log_artifact(self, *_args, **_kwargs):
            return None

    result = FitResult(
        0.5,
        1,
        [
            {
                "epoch": 1,
                "caseIds": raw_ids,
                "nested": {"id": raw_ids[0]},
                "loss": 0.25,
            }
        ],
        {"mean_dice": 0.5},
    )

    common._finalize_training_run(
        {"paths": {"experiment_output": str(tmp_path)}},
        Tracker(),
        result,
        None,
        protocol="fixture",
    )

    name, columns, rows = captured["table"]
    assert name == "training/history"
    assert columns == ["epoch", "case_id_hashes", "nested", "loss"]
    assert rows == [
        [
            1,
            [hash_case_id(raw_id) for raw_id in raw_ids],
            {"case_id_hash": hash_case_id(raw_ids[0])},
            0.25,
        ]
    ]
    assert all(raw_id not in repr(captured) for raw_id in raw_ids)


def test_restore_checkpoint_rejects_incompatible_model_config_with_matching_shapes(
    tmp_path: Path,
):
    model = _Tiny3D()
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint_path = checkpoint_dir / "best.pt"
    torch.save(
        {
            "schema_version": 2,
            "model_state_dict": model.state_dict(),
            "checkpoint_metadata": {
                "architecture": "ResUNet3D",
                "model_config": {"base_features": 2},
            },
        },
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="model_config"):
        pipeline._restore_checkpoint(
            {
                "model": {"base_features": 4},
                "paths": {"checkpoint_dir": str(checkpoint_dir)},
            },
            object(),
            model,
            tmp_path / "output",
            "resunet3d",
            {},
        )


def test_restore_checkpoint_matches_metaunetr_training_model_config_canonicalization(
    tmp_path: Path,
):
    model = _Tiny3D()
    checkpoint_dir = tmp_path / "meta-checkpoints"
    checkpoint_dir.mkdir()
    torch.save(
        {
            "schema_version": 2,
            "model_state_dict": model.state_dict(),
            "checkpoint_metadata": {
                "architecture": "MetaUNETR",
                "variant": "mod_a",
                "model_config": {"base_channels": 2},
            },
        },
        checkpoint_dir / "best.pt",
    )

    pipeline._restore_checkpoint(
        {
            "experiment": "mod_a",
            "model": {"base_channels": 2, "scan_direction": "axial"},
            "paths": {"checkpoint_dir": str(checkpoint_dir)},
        },
        object(),
        model,
        tmp_path / "meta-output",
        "metaunetr",
        {},
    )


def test_restore_checkpoint_matches_cnn_training_default_model_config(
    tmp_path: Path,
):
    model = nn.Identity()
    checkpoint_dir = tmp_path / "cnn-checkpoints"
    checkpoint_dir.mkdir()
    torch.save(
        {
            "schema_version": 2,
            "model_state_dict": model.state_dict(),
            "checkpoint_metadata": {
                "architecture": "denoising_autoencoder",
                "model_config": {
                    "in_channels": 3,
                    "feature_size": 32,
                    "depths": (1, 1, 1, 1),
                    "image_size": 96,
                    "mlp_ratio": 4.0,
                },
            },
        },
        checkpoint_dir / "best.pt",
    )

    pipeline._restore_checkpoint(
        {
            "experiment": "cnn_pretrain",
            "model": {"in_channels": 3},
            "paths": {"checkpoint_dir": str(checkpoint_dir)},
        },
        object(),
        model,
        tmp_path / "cnn-output",
        "cnn",
        {},
    )


@pytest.mark.parametrize(
    "sources",
    [
        {},
        {
            "source_checkpoint": "checkpoints/best.pt",
            "source_artifact": "entity/project/model:v1",
        },
    ],
)
def test_benchmark_requires_exactly_one_checkpoint_source(sources):
    with pytest.raises(ValueError, match="exactly one.*source"):
        pipeline._validate_benchmark_sources({"benchmark": sources})


def test_benchmark_accepts_one_checkpoint_source():
    pipeline._validate_benchmark_sources(
        {"benchmark": {"source_checkpoint": "checkpoints/best.pt"}}
    )
    pipeline._validate_benchmark_sources(
        {"benchmark": {"source_artifact": "entity/project/model:v1"}}
    )
    pipeline._validate_benchmark_sources(
        {
            "benchmark": {
                "source_artifact": "entity/project/model@sha256:"
                + "a" * 64
            }
        }
    )


@pytest.mark.parametrize(
    "reference",
    [
        "entity/project/model:latest@sha256:" + "a" * 64,
        "entity/project/model:v1@sha256:" + "a" * 64,
    ],
)
def test_benchmark_rejects_digest_reference_with_qualified_collection(reference):
    with pytest.raises(ValueError, match="immutable"):
        pipeline._validate_benchmark_sources(
            {"benchmark": {"source_artifact": reference}}
        )


@pytest.mark.parametrize(
    "reference",
    [
        "entity/project/model:latest",
        "entity/project/model:production",
        "entity/project/model:best",
        "entity/project/model",
    ],
)
def test_benchmark_rejects_mutable_or_unqualified_artifact_sources(reference):
    with pytest.raises(ValueError, match="immutable"):
        pipeline._validate_benchmark_sources(
            {"benchmark": {"source_artifact": reference}}
        )


def test_benchmark_tracker_metadata_links_source_without_training_steps():
    cfg = {
        "tracking": {"group": "token-mixer-brats-seed42"},
        "benchmark": {
            "source_artifact": "entity/project/model:v1",
            "source_train_run_id": "train-run-42",
        },
    }

    tracker_config = pipeline._tracker_config(
        cfg,
        "native_3d_full_volume",
        "entity/project/model:v1",
    )
    run_config = pipeline._run_config(
        cfg,
        "metaunetr",
        "MetaUNETR",
        "native_3d_full_volume",
        (1, 4, 32, 32, 32),
        torch.device("cpu"),
        {},
        20,
        100,
        (1, 2, 4, 8),
    )

    assert tracker_config["job_type"] == "benchmark"
    assert tracker_config["group"] == "token-mixer-brats-seed42"
    assert tracker_config["source_train_run_id"] == "train-run-42"
    assert tracker_config["source_checkpoint_artifact"] == "entity/project/model:v1"
    assert tracker_config["protocol_id"] == "native_3d_full_volume"
    assert run_config["source_train_run_id"] == "train-run-42"
    assert run_config["source_checkpoint_artifact"] == "entity/project/model:v1"
    assert run_config["protocol_id"] == "native_3d_full_volume"
    assert "global_step" not in run_config


def test_cnn_case_rows_include_available_timing_boundaries():
    class IdentityDenoiser(nn.Module):
        def forward(self, image):
            return image

    rows, summary = pipeline._cnn_case_protocol(
        IdentityDenoiser(),
        [
            {
                "noisy": torch.ones(1, 1, 3, 4),
                "clean": torch.zeros(1, 1, 3, 4),
                "case_id": ["CASE-CNN"],
            }
        ],
        torch.device("cpu"),
        "cnn_denoising_validation",
        "CNN",
    )

    row = rows[0]
    assert row["case_id_hash"] == hash_case_id("CASE-CNN")
    assert row["latency_ms"] >= 0.0
    for key in (
        "load_seconds",
        "preprocess_seconds",
        "model_compute_seconds",
        "postprocess_seconds",
        "metric_seconds",
        "end_to_end_seconds",
    ):
        assert row[key] >= 0.0
    assert summary["inference/n_cases"] == 1


def test_batch_two_case_timing_is_rejected_for_slice_and_cnn():
    class IdentityDenoiser(nn.Module):
        def forward(self, image):
            return image

    with pytest.raises(ValueError, match="batch size 1"):
        pipeline._cnn_case_protocol(
            IdentityDenoiser(),
            [
                {
                    "noisy": torch.ones(2, 1, 3, 4),
                    "clean": torch.zeros(2, 1, 3, 4),
                    "case_id": ["CASE-CNN-1", "CASE-CNN-2"],
                }
            ],
            torch.device("cpu"),
            "cnn_denoising_validation",
            "CNN",
        )

    class SliceModel(nn.Module):
        def forward(self, image):
            return torch.ones(image.shape[0], 3, *image.shape[-2:])

    with pytest.raises(ValueError, match="batch size 1"):
        pipeline._slice_case_protocol(
            {},
            SliceModel(),
            [
                {
                    "image": torch.ones(2, 1, 2, 2),
                    "label": torch.zeros(2, 3, 2, 2),
                    "case_id": ["CASE-SLICE-1", "CASE-SLICE-2"],
                }
            ],
            torch.device("cpu"),
            "transunet_2d_slice",
            "TransUNet",
        )


def test_case_latency_rejects_multi_case_batches_for_slice_and_cnn_protocols():
    class IdentityDenoiser(nn.Module):
        def forward(self, image):
            return image

    with pytest.raises(ValueError, match="batch size 1"):
        pipeline._cnn_case_protocol(
            IdentityDenoiser(),
            [
                {
                    "noisy": torch.ones(2, 1, 2, 2),
                    "clean": torch.zeros(2, 1, 2, 2),
                    "case_id": ["CASE-CNN-1", "CASE-CNN-2"],
                }
            ],
            torch.device("cpu"),
            "cnn_denoising_validation",
            "CNN",
        )

    class SliceModel(nn.Module):
        def forward(self, image):
            return torch.ones(image.shape[0], 3, *image.shape[-2:])

    with pytest.raises(ValueError, match="batch size 1"):
        pipeline._slice_case_protocol(
            {},
            SliceModel(),
            [
                {
                    "image": torch.ones(2, 1, 2, 2),
                    "label": torch.zeros(2, 3, 2, 2),
                    "case_id": ["CASE-SLICE-1", "CASE-SLICE-2"],
                }
            ],
            torch.device("cpu"),
            "transunet_2d_slice",
            "TransUNet",
        )

def test_benchmark_rows_expose_metadata_power_fields_and_hd95_counts():
    model_row = pipeline._model_row(
        "CNN",
        "cnn_denoising_validation",
        {"model/macs": None, "model/flops": None},
    )
    for key in ("precision", "timing_boundary", "hardware", "software"):
        assert key in model_row
        assert model_row[key] is None

    rows = pipeline._case_records_from_evaluator(
        {"case_records": [{"case_id": "CASE-SCHEMA"}]},
        "CNN",
        "cnn_denoising_validation",
    )
    row = rows[0]
    assert row["hd95_excluded_by_region"] == {}
    assert row["power/average_watts"] is None
    assert row["power/max_watts"] is None
    assert row["power/energy_joules"] is None
    assert row["power/sample_count"] == 0
    assert row["power/status"] == "unsupported"


def test_case_power_is_explicitly_unsupported_outside_model_protocol_scope():
    rows = pipeline._case_records_from_evaluator(
        {"case_records": [{"case_id": "CASE-POWER"}]},
        "CNN",
        "cnn_denoising_validation",
    )

    row = rows[0]
    assert row["power/status"] == "unsupported"
    assert row["power/measurement_scope"] == "model_protocol_only"
    assert "case" in row["power/reason"].lower()
    assert "model" in row["power/reason"].lower()


def test_case_limit_stops_native_evaluator_after_declared_cases():
    evaluated_batches: list[object] = []

    def evaluator(_model, loader):
        batches = list(loader)
        evaluated_batches.extend(batches)
        return {
            "case_records": [
                {"case_id": batch["case_id"][0]} for batch in batches
            ]
        }

    loader = [
        {"image": torch.zeros(1, 4, 2, 2, 2), "case_id": ["CASE-1"]},
        {"image": torch.zeros(1, 4, 2, 2, 2), "case_id": ["CASE-2"]},
    ]

    rows, summary = pipeline._native_case_protocol(
        {
            "benchmark": {"case_limit": 1, "case_evaluator": evaluator}
        },
        nn.Identity(),
        loader,
        torch.device("cpu"),
        "native_3d_full_volume",
        "ResUNet3D",
    )

    assert len(evaluated_batches) == 1
    assert len(rows) == 1
    assert summary["inference/n_cases"] == 1


def test_case_limit_stops_transunet_after_declared_cases(monkeypatch):
    calls = 0
    monkeypatch.setattr(
        pipeline,
        "dice_by_region",
        lambda *_args: {region: 1.0 for region in pipeline.REGION_NAMES},
    )
    monkeypatch.setattr(
        pipeline,
        "hd95_by_region",
        lambda *_args, **_kwargs: {region: 0.0 for region in pipeline.REGION_NAMES},
    )
    monkeypatch.setattr(
        pipeline,
        "hd95_excluded_by_region",
        lambda *_args: {region: 0 for region in pipeline.REGION_NAMES},
    )

    class SliceModel(nn.Module):
        def forward(self, image):
            nonlocal calls
            calls += int(image.shape[0])
            return torch.ones(image.shape[0], 3, *image.shape[-2:])

    loader = [
        {
            "image": torch.ones(1, 1, 2, 2),
            "label": torch.zeros(1, 3, 2, 2),
            "case_id": ["CASE-1"],
        },
        {
            "image": torch.ones(1, 1, 2, 2),
            "label": torch.zeros(1, 3, 2, 2),
            "case_id": ["CASE-2"],
        },
    ]

    rows, summary = pipeline._slice_case_protocol(
        {"benchmark": {"case_limit": 1}},
        SliceModel(),
        loader,
        torch.device("cpu"),
        "transunet_2d_slice",
        "TransUNet",
    )

    assert calls == 1
    assert len(rows) == 1
    assert summary["inference/n_cases"] == 1


def test_case_limit_stops_cnn_after_declared_cases():
    calls = 0

    class IdentityDenoiser(nn.Module):
        def forward(self, image):
            nonlocal calls
            calls += int(image.shape[0])
            return image

    loader = [
        {
            "noisy": torch.ones(1, 1, 3, 4),
            "clean": torch.zeros(1, 1, 3, 4),
            "case_id": ["CASE-1"],
        },
        {
            "noisy": torch.ones(1, 1, 3, 4),
            "clean": torch.zeros(1, 1, 3, 4),
            "case_id": ["CASE-2"],
        },
    ]

    rows, summary = pipeline._cnn_case_protocol(
        IdentityDenoiser(),
        loader,
        torch.device("cpu"),
        "cnn_denoising_validation",
        "CNN",
        case_limit=1,
    )

    assert calls == 1
    assert len(rows) == 1
    assert summary["inference/n_cases"] == 1


def test_case_aggregation_preserves_hd95_exclusion_totals_and_unique_cases():
    rows = [
        {
            "case_id_hash": "a" * 16,
            "latency_ms": 1.0,
            "hd95_excluded_by_region": {"ET": 2, "TC": 0, "WT": 1},
            "exclusion_flags": ["ET", "WT"],
        },
        {
            "case_id_hash": "b" * 16,
            "latency_ms": 2.0,
            "hd95_excluded_by_region": {"ET": 0, "TC": 3, "WT": 0},
            "exclusion_flags": ["TC"],
        },
        {
            "case_id_hash": "a" * 16,
            "latency_ms": 3.0,
            "hd95_excluded_by_region": {"ET": 1, "TC": 0, "WT": 0},
            "exclusion_flags": ["ET"],
        },
    ]

    summary = pipeline._aggregate_case_rows(rows, "native_3d_full_volume")

    assert summary["inference/hd95_excluded_by_region"] == {
        "ET": 3,
        "TC": 3,
        "WT": 1,
    }
    assert summary["inference/hd95_excluded_ET"] == 3
    assert summary["inference/hd95_excluded_TC"] == 3
    assert summary["inference/hd95_excluded_WT"] == 1
    assert summary["inference/hd95_excluded_cases"] == 2


@pytest.mark.parametrize("case_limit", [-1, True, 1.5, "1"])
def test_case_limit_requires_nonnegative_integer_or_null(case_limit):
    with pytest.raises(ValueError, match="case_limit"):
        pipeline._run_case_protocol(
            {"benchmark": {"case_limit": case_limit}},
            "cnn",
            "cnn_denoising_validation",
            "CNN",
            nn.Identity(),
            [],
            torch.device("cpu"),
        )


def test_transunet_case_metrics_receive_batched_slice_regions(
    monkeypatch: pytest.MonkeyPatch,
):
    observed_shapes: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []

    def fake_dice(prediction, target):
        observed_shapes.append(("dice", tuple(prediction.shape), tuple(target.shape)))
        assert prediction.ndim == target.ndim == 5
        assert prediction.shape[:2] == target.shape[:2] == (1, 3)
        return {region: 1.0 for region in pipeline.REGION_NAMES}

    def fake_hd95(prediction, target, *, spacing):
        del spacing
        observed_shapes.append(("hd95", tuple(prediction.shape), tuple(target.shape)))
        assert prediction.ndim == target.ndim == 5
        assert prediction.shape[:2] == target.shape[:2] == (1, 3)
        return {region: 0.0 for region in pipeline.REGION_NAMES}

    def fake_excluded(prediction, target):
        observed_shapes.append(("excluded", tuple(prediction.shape), tuple(target.shape)))
        assert prediction.ndim == target.ndim == 5
        assert prediction.shape[:2] == target.shape[:2] == (1, 3)
        return {region: 0 for region in pipeline.REGION_NAMES}

    monkeypatch.setattr(pipeline, "dice_by_region", fake_dice)
    monkeypatch.setattr(pipeline, "hd95_by_region", fake_hd95)
    monkeypatch.setattr(pipeline, "hd95_excluded_by_region", fake_excluded)

    class SliceModel(nn.Module):
        def forward(self, image):
            return torch.ones(image.shape[0], 3, *image.shape[-2:])

    rows, summary = pipeline._slice_case_protocol(
        {},
        SliceModel(),
        [
            {
                "image": torch.ones(1, 1, 2, 3),
                "label": torch.zeros(1, 3, 2, 3),
                "case_id": ["CASE-2D"],
            }
        ],
        torch.device("cpu"),
        "transunet_2d_slice",
        "TransUNet",
    )

    assert rows[0]["case_id_hash"] == hash_case_id("CASE-2D")
    assert summary["inference/n_cases"] == 1
    assert {name for name, _prediction, _target in observed_shapes} == {
        "dice",
        "hd95",
        "excluded",
    }


def test_transunet_cuda_case_timing_synchronizes_around_forward(
    monkeypatch: pytest.MonkeyPatch,
):
    device = torch.device("cuda:2")
    synchronize_calls: list[torch.device] = []

    monkeypatch.setattr(torch.Tensor, "to", lambda tensor, *_args, **_kwargs: tensor)
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda selected_device=None: synchronize_calls.append(selected_device),
    )
    monkeypatch.setattr(
        pipeline,
        "dice_by_region",
        lambda *_args: {region: 1.0 for region in pipeline.REGION_NAMES},
    )
    monkeypatch.setattr(
        pipeline,
        "hd95_by_region",
        lambda *_args, **_kwargs: {region: 0.0 for region in pipeline.REGION_NAMES},
    )
    monkeypatch.setattr(
        pipeline,
        "hd95_excluded_by_region",
        lambda *_args: {region: 0 for region in pipeline.REGION_NAMES},
    )

    class SliceModel(nn.Module):
        def forward(self, image):
            return torch.ones(image.shape[0], 3, *image.shape[-2:])

    pipeline._slice_case_protocol(
        {},
        SliceModel(),
        [
            {
                "image": torch.ones(1, 1, 2, 2),
                "label": torch.zeros(1, 3, 2, 2),
                "case_id": ["CASE-CUDA"],
            }
        ],
        device,
        "transunet_2d_slice",
        "TransUNet",
    )

    assert synchronize_calls == [device, device]


def test_transunet_loader_iteration_failure_restores_training_states():
    class FailingLoader:
        def __iter__(self):
            raise RuntimeError("loader iteration failed")

    model = nn.Sequential(nn.Linear(1, 1), nn.ReLU())
    model.train()
    model[1].eval()
    original_states = [module.training for module in model.modules()]

    with pytest.raises(RuntimeError, match="loader iteration failed"):
        pipeline._slice_case_protocol(
            {},
            model,
            FailingLoader(),
            torch.device("cpu"),
            "transunet_2d_slice",
            "TransUNet",
        )

    assert [module.training for module in model.modules()] == original_states


def test_run_benchmark_restores_local_checkpoint_and_logs_separate_tracker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    model = _Tiny3D()
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint_path = checkpoint_dir / "best.pt"
    torch.save({"model_state_dict": model.state_dict(), "schema_version": 2}, checkpoint_path)

    events: list[object] = []

    class Tracker:
        def log_summary(self, summary):
            events.append(("summary", dict(summary)))

        def log_table(self, name, columns, rows):
            events.append(("table", name, list(columns), [list(row) for row in rows]))

        def log_artifact(self, name, files, **kwargs):
            events.append(("artifact", name, dict(files), kwargs))
            return "benchmark:v0"

        def finish(self):
            events.append("finish")

    tracker = Tracker()
    cfg = {
        "experiment": "mod_a",
        "variant": "mod_a",
        "device": "cpu",
        "paths": {
            "experiment_output": str(tmp_path / "output"),
            "checkpoint_dir": str(checkpoint_dir),
        },
        "benchmark": {
            "source_checkpoint": str(checkpoint_path),
            "input_shape": [1, 4, 4, 4, 4],
            "warmup_iterations": 0,
            "repetitions": 1,
            "batch_sizes": [1],
            "output_name": "custom-benchmark.json",
        },
        "tracking": {"enabled": False},
    }

    monkeypatch.setattr(pipeline, "build_metaunetr", lambda *_args: model)
    monkeypatch.setattr(
        pipeline,
        "build_metaunetr_loaders",
        lambda *_args: ("train", "val", {"manifest_hash": "fixture"}),
    )
    monkeypatch.setattr(pipeline, "create_tracker", lambda *_args: tracker)
    monkeypatch.setattr(
        pipeline,
        "run_model_protocol",
        lambda *_args, **kwargs: {
            "model/parameters": 1,
            "model/trainable_parameters": 1,
            "model/macs": 1,
            "model/flops": 1,
            "model/mac_status": "ok",
            "model/flop_status": "ok",
            "model/unsupported_ops": {},
            "inference/latency_mean_ms": 1.0,
            "inference/latency_median_ms": 1.0,
            "inference/latency_p95_ms": 1.0,
            "inference/latency_std_ms": 0.0,
            "inference/throughput_samples_per_second": 1000.0,
            "inference/throughput_voxels_per_second": 64000.0,
            "inference/peak_memory_allocated_gb": None,
            "inference/peak_memory_reserved_gb": None,
            "power/status": "unavailable",
            "inference/warmup_iterations": kwargs["warmup_iterations"],
            "inference/repetitions": kwargs["repetitions"],
            "inference/batch_size": 1,
            "inference/protocol": kwargs["protocol"],
            "inference/sweep_status": "ok",
            "inference/sweep_rows": [],
            "inference/largest_passing_batch": 1,
            "inference/first_failing_batch": None,
        },
    )

    result = pipeline.run_benchmark(cfg)

    assert result.provenance["protocol"] == "native_3d_full_volume"
    assert result.provenance["source_checkpoint"] == str(checkpoint_path)
    assert result.rows[0]["protocol"] == "native_3d_full_volume"
    assert result.rows[0]["model"] == "mod_a"
    assert (tmp_path / "output" / "custom-benchmark.json").is_file()
    assert not (tmp_path / "output" / "benchmark.json").exists()
    assert [event for event in events if event == "finish"] == ["finish"]
    assert any(event[0] == "table" and event[1] == "benchmark/rows" for event in events)
    assert any(event[0] == "artifact" for event in events)
    artifact_event = next(event for event in events if event[0] == "artifact")
    assert set(artifact_event[2]) == {"custom-benchmark.json"}


def test_run_benchmark_restores_artifact_and_hashes_case_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    model = _Tiny3D()
    events: list[object] = []
    artifact_checkpoint = tmp_path / "restored" / "best.pt"
    artifact_checkpoint.parent.mkdir()
    torch.save({"model_state_dict": model.state_dict(), "schema_version": 2}, artifact_checkpoint)

    class Tracker:
        def restore_artifact(self, reference, destination):
            events.append(("restore", reference, destination))
            return artifact_checkpoint

        def log_summary(self, _summary):
            pass

        def log_table(self, *_args):
            pass

        def log_artifact(self, *_args, **_kwargs):
            pass

        def finish(self):
            events.append("finish")

    cfg = {
        "experiment": "resunet3d",
        "device": "cpu",
        "paths": {"experiment_output": str(tmp_path / "output")},
        "benchmark": {
            "source_artifact": "entity/project/model:v1",
            "input_shape": [1, 4, 4, 4, 4],
            "warmup_iterations": 0,
            "repetitions": 1,
            "batch_sizes": [1],
        },
        "tracking": {"enabled": False},
    }
    monkeypatch.setattr(pipeline, "build_resunet3d", lambda _cfg: model)
    monkeypatch.setattr(pipeline, "build_volume_loaders", lambda *_args: ("train", "val", "test", {}))
    monkeypatch.setattr(pipeline, "create_tracker", lambda *_args: Tracker())
    monkeypatch.setattr(
        pipeline,
        "run_model_protocol",
        lambda *_args, **kwargs: {
            "model/parameters": 1,
            "model/trainable_parameters": 1,
            "model/macs": 1,
            "model/flops": 1,
            "model/mac_status": "ok",
            "model/flop_status": "ok",
            "model/unsupported_ops": {},
            "inference/latency_mean_ms": 1.0,
            "inference/latency_median_ms": 1.0,
            "inference/latency_p95_ms": 1.0,
            "inference/latency_std_ms": 0.0,
            "inference/throughput_samples_per_second": 1000.0,
            "inference/throughput_voxels_per_second": 1000.0,
            "inference/peak_memory_allocated_gb": None,
            "inference/peak_memory_reserved_gb": None,
            "power/status": "unavailable",
            "inference/warmup_iterations": kwargs["warmup_iterations"],
            "inference/repetitions": kwargs["repetitions"],
            "inference/batch_size": 1,
            "inference/protocol": kwargs["protocol"],
            "inference/sweep_status": "ok",
            "inference/sweep_rows": [],
            "inference/largest_passing_batch": 1,
            "inference/first_failing_batch": None,
        },
    )
    monkeypatch.setattr(
        pipeline,
        "_run_case_protocol",
        lambda *_args, **_kwargs: (
            [{
                "case_id_hash": hash_case_id("SECRET-CASE"),
                "protocol": "native_3d_full_volume",
                "latency_ms": 2.0,
            }],
            {
                "inference/n_cases": 1,
                "inference/hd95_excluded_by_region": {
                    "ET": 1,
                    "TC": 0,
                    "WT": 2,
                },
                "inference/hd95_excluded_ET": 1,
                "inference/hd95_excluded_TC": 0,
                "inference/hd95_excluded_WT": 2,
                "inference/hd95_excluded_cases": 1,
            },
        ),
    )

    result = pipeline.run_benchmark(cfg)

    assert any(event[0] == "restore" for event in events)
    assert result.provenance["source_checkpoint_artifact"] == "entity/project/model:v1"
    assert result.provenance["hd95_excluded_by_region"] == {
        "ET": 1,
        "TC": 0,
        "WT": 2,
    }
    assert result.provenance["hd95_excluded_cases"] == 1
    assert result.rows[-1]["case_id_hash"] == hash_case_id("SECRET-CASE")
    assert "SECRET-CASE" not in (tmp_path / "output" / "benchmark.json").read_text()
    assert events[-1] == "finish"


def test_tracker_finishes_when_benchmark_measurement_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    class Tracker:
        finished = False

        def finish(self):
            self.finished = True

    tracker = Tracker()
    model = _Tiny3D()
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    torch.save({"model_state_dict": model.state_dict(), "schema_version": 2}, checkpoint_dir / "best.pt")
    cfg = {
        "experiment": "resunet3d",
        "device": "cpu",
        "paths": {
            "experiment_output": str(tmp_path / "output"),
            "checkpoint_dir": str(checkpoint_dir),
        },
        "benchmark": {
            "source_checkpoint": str(checkpoint_dir / "best.pt"),
            "input_shape": [1, 4, 4, 4, 4],
        },
        "tracking": {"enabled": False},
    }
    monkeypatch.setattr(pipeline, "build_resunet3d", lambda _cfg: model)
    monkeypatch.setattr(pipeline, "build_volume_loaders", lambda *_args: ("train", "val", "test", {}))
    monkeypatch.setattr(pipeline, "create_tracker", lambda *_args: tracker)
    monkeypatch.setattr(pipeline, "run_model_protocol", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    with pytest.raises(RuntimeError, match="boom"):
        pipeline.run_benchmark(cfg)

    assert tracker.finished is True
