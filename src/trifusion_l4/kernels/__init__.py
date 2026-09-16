"""Inference-only SM89 kernels with an independent PyTorch reference."""

import torch
import torch.nn.functional as F
from torch import nn
import triton as tr
from . import primitives as p


def rms_reference(x, weight, eps):
    xf = x.double() if x.dtype == torch.float64 else x.float()
    return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps) * weight).to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, backend="reference"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self.backend = backend
        self.native_kernels = False

    def forward(self, x):
        if self.backend == "reference":
            return rms_reference(x, self.weight, self.eps)
        from .native import rms

        return rms(x, self.weight, self.eps)


def swiglu(x, backend="l4"):
    if backend == "reference":
        gate, up = x.float().chunk(2, -1)
        return (F.silu(gate) * up).to(x.dtype)
    x = x.contiguous()
    d = x.shape[-1] // 2
    y = x.new_empty((*x.shape[:-1], d))
    p._swiglu[(tr.cdiv(y.numel(), 512),)](
        x, y, y, y, y.numel(), d, False, 512, enable_fp_fusion=False
    )
    return y


def reverse_valid(x, lengths, backend="l4"):
    if backend == "reference":
        b, t, d = x.shape
        pos = torch.arange(t, device=x.device)[None, :]
        idx = (lengths[:, None] - 1 - pos).clamp_min(0)
        return x.gather(1, idx[..., None].expand(b, t, d)).masked_fill(
            (pos >= lengths[:, None])[..., None], 0
        )
    x = x.contiguous()
    y = torch.empty_like(x)
    p._reverse[(tr.cdiv(x.numel(), 512),)](x, lengths, y, x.shape[1], x.shape[2], x.numel(), 512)
    return y


def gated_concat(x, gate, backend="l4"):
    if backend == "reference":
        return (
            (x.float().reshape(*x.shape[:-1], 3, -1) * (2 * gate.float().sigmoid())[..., None])
            .reshape_as(x)
            .to(x.dtype)
        )
    x = x.contiguous()
    gate = gate.contiguous()
    y = torch.empty_like(x)
    d = x.shape[-1] // 3
    p._gate[(gate.numel(),)](
        x,
        gate,
        y,
        y,
        y,
        gate,
        gate.numel(),
        d,
        tr.next_power_of_2(d),
        False,
        enable_fp_fusion=False,
    )
    return y
