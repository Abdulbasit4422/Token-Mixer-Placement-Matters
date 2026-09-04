"""Mamba token mixing and three-axis cross-scans for MetaUNETR.

The axis decomposition follows Lyu et al., *MetaUNETR: Rethinking Token
Mixer Encoding for Efficient Multi-Organ Segmentation* and its official
implementation. The manuscript describes summing axis outputs, whereas the
public ``Mamba3Dcross`` implementation concatenates them before projection;
both choices are exposed here. The optional Mamba dependency is deliberately
loaded only for CUDA execution, and CPU inputs always use the local fallback.
"""

from __future__ import annotations

import importlib
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _drop_path(x: Tensor, drop_prob: float, training: bool) -> Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = x.new_empty(shape).bernoulli_(keep_prob)
    return x * mask.div(keep_prob)


class FallbackMamba(nn.Module):
    """Small pure-PyTorch selective-state-space sequence mixer.

    Input and output have shape ``[B, L, d_model]``. The recurrence state is
    always accumulated in float32, then the output is restored to input dtype.
    This is a numerically stable CPU fallback, not a claim of bit identity
    with the CUDA implementation from https://github.com/state-spaces/mamba.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        **_: object,
    ) -> None:
        super().__init__()
        if d_model < 1 or d_state < 1 or d_conv < 1 or expand < 1:
            raise ValueError("d_model, d_state, d_conv, and expand must be positive")

        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_conv = int(d_conv)
        self.expand = int(expand)
        self.d_inner = self.d_model * self.expand
        self.dt_rank = max(1, math.ceil(self.d_model / 16))

        self.in_proj = nn.Linear(self.d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=self.d_conv,
            padding=self.d_conv - 1,
            groups=self.d_inner,
        )
        self.x_proj = nn.Linear(
            self.d_inner,
            self.dt_rank + 2 * self.d_state,
            bias=False,
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, self.d_state + 1, dtype=torch.float32))
            .unsqueeze(0)
            .expand(self.d_inner, -1)
            .clone()
        )
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

        nn.init.normal_(self.dt_proj.weight, std=0.02)
        nn.init.constant_(self.dt_proj.bias, -2.25)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"Mamba expects [B, L, C], got {tuple(x.shape)}")
        if not x.is_floating_point():
            raise TypeError("Mamba input must have a floating-point dtype")

        input_dtype = x.dtype
        work_dtype = self.in_proj.weight.dtype
        x_work = x.to(work_dtype)
        batch, length, _ = x_work.shape

        xz = self.in_proj(x_work)
        x_branch, z = xz.chunk(2, dim=-1)
        x_branch = self.conv1d(x_branch.transpose(1, 2))[..., :length]
        x_branch = F.silu(x_branch.transpose(1, 2))

        x_delta = self.x_proj(x_branch)
        dt_raw, b_param, c_param = torch.split(
            x_delta,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1,
        )
        dt = F.softplus(self.dt_proj(dt_raw)).clamp(min=1e-4, max=1.0)
        state = x_work.new_zeros(
            batch,
            self.d_inner,
            self.d_state,
            dtype=torch.float32,
        )
        A = -torch.exp(self.A_log.float())
        outputs: list[Tensor] = []

        for step in range(length):
            dt_step = dt[:, step].float()
            d_a = torch.exp(dt_step.unsqueeze(-1) * A.unsqueeze(0))
            d_b = dt_step.unsqueeze(-1) * b_param[:, step].float().unsqueeze(1)
            state = state * d_a + x_branch[:, step].float().unsqueeze(-1) * d_b
            y = (state * c_param[:, step].float().unsqueeze(1)).sum(dim=-1)
            y = y + self.D.float().view(1, -1) * x_branch[:, step].float()
            outputs.append(y * F.silu(z[:, step].float()))

        y = torch.stack(outputs, dim=1)
        y = self.norm(y)
        output = self.out_proj(y.to(work_dtype))
        return output.to(input_dtype)


def _external_mamba_class(device: torch.device) -> type[nn.Module] | None:
    """Return CUDA Mamba implementation when available for ``device``."""

    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    try:
        module = importlib.import_module("mamba_ssm")
    except Exception:
        return None
    candidate = getattr(module, "Mamba", None)
    return candidate if isinstance(candidate, type) else None


class Mamba(nn.Module):
    """Wrapper selecting an explicitly placed CUDA Mamba or local fallback.

    The optional implementation is materialized only during construction when
    ``execution_device`` explicitly names a CUDA device. This keeps optimizer
    and checkpoint schemas stable after construction.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        execution_device: torch.device | str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_conv = int(d_conv)
        self.expand = int(expand)
        self.execution_device = (
            torch.device(execution_device) if execution_device is not None else None
        )
        self._kwargs = dict(kwargs)
        self.impl = FallbackMamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            **kwargs,
        )
        self._external_impl: nn.Module | None = None
        self._external_disabled = False
        self.using_external = False

        if self.execution_device is not None and self.execution_device.type == "cuda":
            self._materialize_external(self.execution_device)

    def _materialize_external(self, device: torch.device) -> None:
        if device.type != "cuda" or self._external_disabled:
            return
        if self._external_impl is not None:
            return

        external = _external_mamba_class(device)
        if external is None:
            self._external_disabled = True
            return
        try:
            implementation = external(
                d_model=self.d_model,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                **self._kwargs,
            )
            if not isinstance(implementation, nn.Module):
                raise TypeError("mamba_ssm.Mamba must return an nn.Module")
            implementation.to(device=device)
        except Exception:
            self._external_disabled = True
            return

        self._external_impl = implementation

    def _fallback_forward(self, x: Tensor) -> Tensor:
        parameter = next(self.impl.parameters(), None)
        if parameter is not None and parameter.device != x.device:
            self.impl.to(device=x.device)
        return self.impl(x)

    def forward(self, x: Tensor) -> Tensor:
        input_dtype = x.dtype
        external = self._external_impl
        if (
            self.execution_device is None
            or self.execution_device.type != "cuda"
            or x.device.type != "cuda"
            or self._external_disabled
            or external is None
        ):
            self.using_external = False
            output = self._fallback_forward(x)
        else:
            try:
                output = external(x.float())
                if not isinstance(output, Tensor):
                    raise TypeError("external Mamba must return a tensor")
                output = output.to(input_dtype)
                self.using_external = True
                return output
            except Exception:
                self._external_disabled = True
                self.using_external = False
                output = self._fallback_forward(x)
        return output.to(input_dtype)


class CrossScan3D(nn.Module):
    """Apply one sequence mixer along depth, height, and width axes.

    A ``[B, D, H, W, C]`` tensor is folded into effective batches of
    ``[B*H*W, D, C]``, ``[B*D*W, H, C]``, and ``[B*D*H, W, C]``. Axis outputs
    are summed and pointwise projected for the manuscript equation, or
    concatenated and projected for the public-reference construction. Output
    shape always matches input.
    """

    def __init__(
        self,
        dim: int,
        axis_fusion: str = "sum",
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        execution_device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if axis_fusion not in {"sum", "cat"}:
            raise ValueError("axis_fusion must be 'sum' or 'cat'")
        self.dim = int(dim)
        self.axis_fusion = axis_fusion
        self.depth_mamba = Mamba(
            d_model=self.dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            execution_device=execution_device,
        )
        self.height_mamba = Mamba(
            d_model=self.dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            execution_device=execution_device,
        )
        self.width_mamba = Mamba(
            d_model=self.dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            execution_device=execution_device,
        )
        self.projection = nn.Linear(
            3 * self.dim if axis_fusion == "cat" else self.dim,
            self.dim,
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 5 or x.shape[-1] != self.dim:
            raise ValueError(
                f"CrossScan3D expects [B, D, H, W, {self.dim}], got {tuple(x.shape)}"
            )
        batch, depth, height, width, channels = x.shape

        depth_seq = x.permute(0, 2, 3, 1, 4).contiguous().view(
            batch * height * width, depth, channels
        )
        depth_out = self.depth_mamba(depth_seq).view(
            batch, height, width, depth, channels
        ).permute(0, 3, 1, 2, 4).contiguous()

        height_seq = x.permute(0, 1, 3, 2, 4).contiguous().view(
            batch * depth * width, height, channels
        )
        height_out = self.height_mamba(height_seq).view(
            batch, depth, width, height, channels
        ).permute(0, 1, 3, 2, 4).contiguous()

        width_seq = x.contiguous().view(batch * depth * height, width, channels)
        width_out = self.width_mamba(width_seq).view(batch, depth, height, width, channels)

        if self.axis_fusion == "sum":
            fused = depth_out + height_out + width_out
        else:
            fused = torch.cat((depth_out, height_out, width_out), dim=-1)
        return self.projection(fused.to(self.projection.weight.dtype)).to(x.dtype)


class TriCruciMamba3D(nn.Module):
    """Residual TriCruci Mamba block operating on channels-last volumes."""

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        axis_fusion: str = "sum",
        execution_device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        hidden_dim = max(1, int(dim * mlp_ratio))
        self.norm1 = nn.LayerNorm(dim)
        self.cross_scan = CrossScan3D(
            dim=dim,
            axis_fusion=axis_fusion,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            execution_device=execution_device,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.drop_path = float(drop_path)

    def forward(self, x: Tensor) -> Tensor:
        x = x + _drop_path(self.cross_scan(self.norm1(x)), self.drop_path, self.training)
        mlp_input = self.norm2(x).to(next(self.mlp.parameters()).dtype)
        mlp_output = self.mlp(mlp_input).to(x.dtype)
        x = x + _drop_path(mlp_output, self.drop_path, self.training)
        return x


MambaBlock = TriCruciMamba3D


__all__ = [
    "CrossScan3D",
    "FallbackMamba",
    "Mamba",
    "MambaBlock",
    "TriCruciMamba3D",
]
