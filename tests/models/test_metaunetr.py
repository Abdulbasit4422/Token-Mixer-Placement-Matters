from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from token_mixer.models.metaunetr.mamba import CrossScan3D
from token_mixer.models.metaunetr.mamba import TriCruciMamba3D
from token_mixer.models.metaunetr.encoder import Encoder3D
from token_mixer.models.metaunetr.network import MetaUNETR
from token_mixer.models.metaunetr.variants import build_metaunetr


def tiny_config() -> dict[str, object]:
    return {
        "in_channels": 4,
        "num_classes": 3,
        "base_channels": 4,
        "depths": (1, 1, 1, 1),
        "window_size": 2,
        "num_heads": 2,
        "d_state": 2,
        "d_conv": 2,
        "mamba_expand": 1,
        "axis_fusion": "sum",
    }


def _mamba_module_names(model: torch.nn.Module) -> list[str]:
    return [
        name
        for name, module in model.named_modules()
        if "mamba" in type(module).__name__.lower()
    ]


def test_paper_variants_share_raw_logit_contract():
    for variant in ("metaunetr_mamba", "mod_a", "mod_b"):
        model = build_metaunetr(tiny_config(), variant).eval()
        with torch.no_grad():
            logits = model(torch.randn(1, 4, 32, 32, 32))
        assert logits.shape == (1, 3, 32, 32, 32)
        assert model.output_regions == ("ET", "TC", "WT")


def test_baseline_has_mamba_only_in_bottleneck_path():
    model = build_metaunetr(tiny_config(), "metaunetr_mamba")
    names = _mamba_module_names(model)

    assert names
    assert all("encoder.bottleneck" in name for name in names)


def test_baseline_has_one_bottleneck_mixer_and_official_residual_adapter():
    config = tiny_config()
    config["depths"] = (2, 2, 2, 2)
    model = build_metaunetr(config, "metaunetr_mamba")

    bottleneck_mixers = [
        module
        for module in model.encoder.bottleneck.modules()
        if isinstance(module, TriCruciMamba3D)
    ]

    assert len(bottleneck_mixers) == 1
    assert not isinstance(model.encoder.bottleneck, nn.Sequential)
    assert type(model.encoder.encoder10).__name__ == "UnetrBasicBlock"


def test_mod_a_has_mamba_encoder_blocks_and_no_mamba_decoder_blocks():
    model = build_metaunetr(tiny_config(), "mod_a")
    names = _mamba_module_names(model)

    assert all(any(name.startswith(f"encoder.stages.{stage}") for name in names)
               for stage in range(4))
    assert not any(name.startswith("decoder") for name in names)
    assert not any(isinstance(module, TriCruciMamba3D)
                   for module in model.encoder.bottleneck.modules())


def test_mod_b_has_coarse_decoder_mamba_but_cnn_final_stage():
    model = build_metaunetr(tiny_config(), "mod_b")
    names = _mamba_module_names(model)

    assert any(name.startswith("decoder.coarse") for name in names)
    assert not any(name.startswith("encoder") for name in names)
    assert not any(name.startswith("decoder.final") for name in names)


def test_invalid_variant_and_spatial_size_fail_clearly():
    with pytest.raises(ValueError, match="variant"):
        build_metaunetr(tiny_config(), "unknown")

    model = build_metaunetr(tiny_config(), "metaunetr_mamba")
    with pytest.raises(ValueError, match="divisible"):
        model(torch.randn(1, 4, 31, 32, 32))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("in_channels", 3, "exactly 4 input channels"),
        ("num_classes", 2, "exactly 3 output classes"),
    ],
)
def test_public_metaunetr_builder_enforces_canonical_channel_contract(
    field: str, value: int, message: str
):
    config = tiny_config()
    config[field] = value

    with pytest.raises(ValueError, match=message):
        build_metaunetr(config, "metaunetr_mamba")


def test_public_metaunetr_builder_checks_out_channels_alias_too():
    config = tiny_config()
    config.pop("num_classes")
    config["out_channels"] = 2

    with pytest.raises(ValueError, match="exactly 3 output classes"):
        build_metaunetr(config, "metaunetr_mamba")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"in_channels": 3}, "exactly 4 input channels"),
        ({"num_classes": 2}, "exactly 3 output classes"),
    ],
)
def test_direct_metaunetr_construction_enforces_canonical_channel_contract(
    kwargs: dict[str, int], message: str
):
    with pytest.raises(ValueError, match=message):
        MetaUNETR(base_channels=4, depths=(1, 1, 1, 1), **kwargs)


def test_cpu_fallback_import_does_not_mutate_sys_modules():
    module = importlib.import_module("token_mixer.models.metaunetr.mamba")
    before = dict(sys.modules)
    importlib.reload(module)
    assert sys.modules == before


def test_cpu_forward_never_loads_or_uses_optional_cuda_backend(monkeypatch):
    module = importlib.import_module("token_mixer.models.metaunetr.mamba")
    imports: list[str] = []
    external_calls: list[bool] = []
    real_import = module.importlib.import_module

    class FakeExternal(nn.Module):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            external_calls.append(True)
            return x

    def import_spy(name: str, *args: object, **kwargs: object) -> object:
        if name == "mamba_ssm":
            imports.append(name)
            return SimpleNamespace(Mamba=FakeExternal)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(module.importlib, "import_module", import_spy)
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    before = dict(sys.modules)

    mixer = module.Mamba(d_model=4, d_state=2, d_conv=2, expand=1).eval()
    with torch.no_grad():
        output = mixer(torch.randn(1, 3, 4))

    assert output.shape == (1, 3, 4)
    assert imports == []
    assert external_calls == []
    assert sys.modules == before
    assert isinstance(mixer.impl, module.FallbackMamba)


def test_unloadable_cuda_extension_selects_fallback(monkeypatch):
    module = importlib.import_module("token_mixer.models.metaunetr.mamba")

    def broken_import(*args: object, **kwargs: object) -> object:
        raise OSError("mamba_ssm CUDA extension failed to load")

    monkeypatch.setattr(module.importlib, "import_module", broken_import)
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)

    assert module._external_mamba_class(torch.device("cuda")) is None


def test_forward_does_not_materialize_external_backend_or_change_schema(monkeypatch):
    module = importlib.import_module("token_mixer.models.metaunetr.mamba")
    cpu_input = torch.randn(1, 3, 4)
    external_calls: list[bool] = []

    class FakeCudaInput:
        dtype = cpu_input.dtype
        device = torch.device("cuda")

        def float(self) -> torch.Tensor:
            return cpu_input

        def to(self, *args: object, **kwargs: object) -> torch.Tensor:
            return cpu_input.to(*args, **kwargs)

    class FakeExternal(nn.Module):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(4))

        def to(self, *args: object, **kwargs: object) -> "FakeExternal":
            return self

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            external_calls.append(True)
            return x

    monkeypatch.setattr(module, "_external_mamba_class", lambda device: FakeExternal)

    mixer = module.Mamba(d_model=4, d_state=2, d_conv=2, expand=1).eval()
    monkeypatch.setattr(mixer, "_fallback_forward", lambda _x: cpu_input)
    before_parameter_names = tuple(name for name, _ in mixer.named_parameters())
    before_state_keys = tuple(mixer.state_dict())
    optimizer = torch.optim.Adam(mixer.parameters())
    before_optimizer_parameters = tuple(
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    )

    with torch.no_grad():
        output = mixer(FakeCudaInput())

    assert output.shape == cpu_input.shape
    assert external_calls == []
    assert mixer._external_impl is None
    assert tuple(name for name, _ in mixer.named_parameters()) == before_parameter_names
    assert tuple(mixer.state_dict()) == before_state_keys
    assert tuple(
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ) == before_optimizer_parameters


def test_explicit_cuda_backend_materializes_before_optimizer_and_stays_stable(monkeypatch):
    module = importlib.import_module("token_mixer.models.metaunetr.mamba")
    external_calls: list[bool] = []

    class FakeExternal(nn.Module):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(4))

        def to(self, *args: object, **kwargs: object) -> "FakeExternal":
            return self

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            external_calls.append(True)
            return x

    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(module, "_external_mamba_class", lambda device: FakeExternal)

    mixer = module.Mamba(
        d_model=4,
        d_state=2,
        d_conv=2,
        expand=1,
        execution_device=torch.device("cuda"),
    ).eval()

    assert mixer._external_impl is not None
    before_parameter_names = tuple(name for name, _ in mixer.named_parameters())
    before_state_keys = tuple(mixer.state_dict())
    optimizer = torch.optim.Adam(mixer.parameters())
    before_optimizer_parameters = tuple(
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    )

    mixer.to("cpu")
    with torch.no_grad():
        output = mixer(torch.randn(1, 3, 4))

    assert output.shape == (1, 3, 4)
    assert external_calls == []
    assert tuple(name for name, _ in mixer.named_parameters()) == before_parameter_names
    assert tuple(mixer.state_dict()) == before_state_keys
    assert tuple(
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ) == before_optimizer_parameters


def test_explicit_execution_device_propagates_to_mamba_placements(monkeypatch):
    module = importlib.import_module("token_mixer.models.metaunetr.mamba")

    class FakeExternal(nn.Module):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__()

        def to(self, *args: object, **kwargs: object) -> "FakeExternal":
            return self

    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(module, "_external_mamba_class", lambda device: FakeExternal)

    config = tiny_config()
    config["execution_device"] = "cuda"
    model = build_metaunetr(config, "mod_b")
    wrappers = [item for item in model.modules() if isinstance(item, module.Mamba)]

    assert wrappers
    assert all(
        getattr(item, "execution_device", None) == torch.device("cuda")
        for item in wrappers
    )
    assert all(item._external_impl is not None for item in wrappers)


def test_cpu_forward_preserves_optimizer_and_state_schema(monkeypatch):
    module = importlib.import_module("token_mixer.models.metaunetr.mamba")
    backend_constructions: list[torch.device] = []

    class FakeExternal(nn.Module):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__()
            backend_constructions.append(torch.device("cuda"))

    monkeypatch.setattr(module, "_external_mamba_class", lambda device: FakeExternal)
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)

    mixer = module.Mamba(d_model=4, d_state=2, d_conv=2, expand=1).eval()
    optimizer = torch.optim.Adam(mixer.parameters())
    before_parameter_names = tuple(name for name, _ in mixer.named_parameters())
    before_state_keys = tuple(mixer.state_dict())
    before_optimizer_parameters = tuple(
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    )

    with torch.no_grad():
        output = mixer(torch.randn(1, 3, 4))

    assert output.shape == (1, 3, 4)
    assert backend_constructions == []
    assert isinstance(mixer.impl, module.FallbackMamba)
    assert mixer._external_impl is None
    assert tuple(name for name, _ in mixer.named_parameters()) == before_parameter_names
    assert tuple(mixer.state_dict()) == before_state_keys
    assert tuple(
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ) == before_optimizer_parameters


def test_external_execution_failure_falls_back_without_schema_mutation(monkeypatch):
    module = importlib.import_module("token_mixer.models.metaunetr.mamba")
    cpu_input = torch.randn(1, 3, 4)
    external_calls: list[bool] = []

    class BrokenExternal(nn.Module):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(4))

        def to(self, *args: object, **kwargs: object) -> "BrokenExternal":
            return self

        def forward(self, x: torch.Tensor) -> None:
            external_calls.append(True)
            return None

    class FakeCudaInput:
        dtype = cpu_input.dtype
        device = torch.device("cuda")

        def float(self) -> torch.Tensor:
            return cpu_input

    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(module, "_external_mamba_class", lambda device: BrokenExternal)

    mixer = module.Mamba(
        d_model=4,
        d_state=2,
        d_conv=2,
        expand=1,
        execution_device=torch.device("cuda"),
    ).eval()
    monkeypatch.setattr(mixer, "_fallback_forward", lambda _x: cpu_input)
    optimizer = torch.optim.Adam(mixer.parameters())
    before_parameter_names = tuple(name for name, _ in mixer.named_parameters())
    before_state_keys = tuple(mixer.state_dict())
    before_optimizer_parameters = tuple(
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    )

    with torch.no_grad():
        output = mixer(FakeCudaInput())

    assert output.shape == cpu_input.shape
    assert external_calls == [True]
    assert mixer._external_disabled is True
    assert mixer.using_external is False
    assert tuple(name for name, _ in mixer.named_parameters()) == before_parameter_names
    assert tuple(mixer.state_dict()) == before_state_keys
    assert tuple(
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ) == before_optimizer_parameters


def test_forward_backward_produces_finite_logits_and_gradients():
    model = build_metaunetr(tiny_config(), "metaunetr_mamba").train()
    image = torch.randn(1, 4, 32, 32, 32, requires_grad=True)

    logits = model(image)
    logits.square().mean().backward()

    assert torch.isfinite(logits).all()
    assert image.grad is not None
    assert torch.isfinite(image.grad).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_cat_axis_fusion_preserves_channel_width():
    mixer = CrossScan3D(
        dim=4,
        axis_fusion="cat",
        d_state=2,
        d_conv=2,
        expand=1,
    )
    output = mixer(torch.randn(1, 4, 4, 4, 4))

    assert output.shape == (1, 4, 4, 4, 4)


def test_sum_axis_fusion_projects_and_records_channel_width():
    mixer = CrossScan3D(
        dim=4,
        axis_fusion="sum",
        d_state=2,
        d_conv=2,
        expand=1,
    ).eval()

    assert mixer.axis_fusion == "sum"
    assert isinstance(mixer.projection, nn.Linear)
    assert mixer.projection.in_features == 4
    assert mixer.projection.out_features == 4
    with torch.no_grad():
        output = mixer(torch.randn(1, 2, 2, 2, 4))
    assert output.shape == (1, 2, 2, 2, 4)


def test_half_input_fallback_block_is_finite_on_cpu():
    mixer = TriCruciMamba3D(
        dim=4,
        mlp_ratio=2.0,
        d_state=2,
        d_conv=2,
        expand=1,
    ).eval()

    with torch.no_grad():
        output = mixer(torch.randn(1, 2, 2, 2, 4).half())

    assert output.dtype == torch.float16
    assert torch.isfinite(output).all()


def test_encoder_uses_raw_stage_inputs_and_normalized_hidden_states():
    encoder = Encoder3D(
        in_channels=4,
        base_channels=4,
        depths=(1, 1, 1, 1),
        stage_mixer="cnn",
        bottleneck_mixer="cnn",
        norm_name=("group", {"num_groups": 1}),
    ).eval()
    adapter_names = ("encoder1", "encoder2", "encoder3", "encoder4", "encoder5", "encoder10")
    assert all(type(getattr(encoder, name)).__name__ == "UnetrBasicBlock"
               for name in adapter_names)

    captured: dict[str, torch.Tensor] = {}
    events: list[str] = []
    handles = []

    def capture_input(name: str):
        def hook(_module: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
            captured[name] = args[0].detach()
            events.append(name)

        return hook

    def capture_output(name: str):
        def hook(_module: nn.Module, _args: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            captured[name] = output.detach()
            events.append(name)

        return hook

    assert not hasattr(encoder, "stem_norm")
    handles.append(encoder.stem.register_forward_hook(capture_output("stem_raw")))
    for index, stage in enumerate(encoder.stages):
        handles.append(stage.register_forward_pre_hook(capture_input(f"stage{index}_input")))
        handles.append(stage.register_forward_hook(capture_output(f"stage{index}_output")))
    for index, adapter in enumerate(
        (encoder.encoder2, encoder.encoder3, encoder.encoder4, encoder.encoder5, encoder.encoder10)
    ):
        handles.append(adapter.register_forward_pre_hook(capture_input(f"adapter{index}")))
    for index, downsample in enumerate(encoder.downsamples):
        handles.append(downsample.register_forward_hook(capture_output(f"down{index}_output")))

    image = torch.randn(1, 4, 32, 32, 32)
    with torch.no_grad():
        bottleneck, skips = encoder(image)
    for handle in handles:
        handle.remove()

    expected_skip_shapes = (
        (1, 4, 32, 32, 32),
        (1, 4, 16, 16, 16),
        (1, 8, 8, 8, 8),
        (1, 16, 4, 4, 4),
        (1, 32, 2, 2, 2),
    )
    assert [tuple(skip.shape) for skip in skips] == list(expected_skip_shapes)
    assert tuple(bottleneck.shape) == (1, 64, 1, 1, 1)

    stem_channels_last = captured["stem_raw"].permute(0, 2, 3, 4, 1)
    assert torch.allclose(captured["stage0_input"], stem_channels_last)
    assert events.index("stem_raw") < events.index("adapter0") < events.index("stage0_input")

    adapter_sources = [
        stem_channels_last,
        captured["down0_output"],
        captured["down1_output"],
        captured["down2_output"],
        captured["down3_output"],
    ]
    for index, source in enumerate(adapter_sources):
        expected = F.layer_norm(source, [source.shape[-1]]).permute(0, 4, 1, 2, 3)
        assert torch.allclose(captured[f"adapter{index}"], expected)

    for index in range(4):
        assert events.index(f"adapter{index}") < events.index(f"stage{index}_input")

    for index in range(4):
        if index < 3:
            assert torch.allclose(
                captured[f"stage{index + 1}_input"],
                captured[f"down{index}_output"],
            )
            assert events.index(f"stage{index}_output") < events.index(f"down{index}_output")
            assert events.index(f"down{index}_output") < events.index(f"adapter{index + 1}")
