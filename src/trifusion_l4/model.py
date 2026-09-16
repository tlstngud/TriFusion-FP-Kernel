"""Executable TriFusion-FP v3 specification. No teacher inference in forward."""

from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F
from .front import Frontend
from .kernels import RMSNorm, reverse_valid, swiglu, gated_concat
from .masking import Grids, masked
from .attention import masked_attention


@dataclass
class ModelConfig:
    dim: int = 1024
    depth: int = 16
    ffn_dim: int = 2816
    heads: int = 16
    state_dim: int = 128
    evidence_rank: int = 64
    dropout: float = 0.1
    backend: str = "l4"
    # Legacy training flags are accepted only for checkpoint-config compatibility.
    checkpoint_blocks: bool = False
    checkpoint_frontend: bool = False
    channels_last: bool = False
    buffer_fusion: bool = True
    native_kernels: bool = True
    mamba_packed: bool = True
    paired_mamba: bool = False
    mamba_parallel_bwd: bool = True
    mamba_fast_launch: bool = True
    paired_mamba_grid_limit: int = 64


def mamba_factory(config):
    from .mamba import Mamba3Inference

    return Mamba3Inference(config)


def rope(x, cos=None, sin=None):
    d = x.shape[-1]
    if cos is None:
        pos = torch.arange(x.shape[-2], device=x.device).float()
        freq = 10000.0 ** (-torch.arange(0, d, 2, device=x.device).float() / d)
        phase = pos[:, None] * freq[None, :]
        cos, sin = phase.cos(), phase.sin()
    a, b = x.float().reshape(*x.shape[:-1], d // 2, 2).unbind(-1)
    return torch.stack((a * cos - b * sin, a * sin + b * cos), -1).flatten(-2).to(x.dtype)


class LocalCNN(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.backend = c.backend
        self.buffer_fusion = c.buffer_fusion
        self.native_kernels = c.native_kernels
        self.up = nn.Linear(c.dim, 2 * c.dim, bias=False)
        self.dw = nn.Conv1d(c.dim, c.dim, 7, padding=3, groups=c.dim)
        self.norm = RMSNorm(c.dim, backend=c.backend)
        self.out = nn.Linear(c.dim, c.dim, bias=False)

    def forward(self, x, mask, lengths):
        h = F.glu(self.up(x), dim=-1)
        if self.backend == "reference":
            h = masked(h, mask)
        if self.backend == "l4" and self.buffer_fusion:
            from .kernels.buffers import depthwise_masked

            h = depthwise_masked(h, self.dw, lengths)
        else:
            h = masked(self.dw(h.transpose(1, 2)).transpose(1, 2), mask)
        y = self.out(F.silu(self.norm(h)))
        return masked(y, mask) if self.backend == "reference" else y


class GlobalAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads, self.head_dim = c.heads, c.dim // c.heads
        self.backend, self.buffer_fusion = c.backend, c.buffer_fusion
        self.native_kernels = c.native_kernels
        self.qkv = nn.Linear(c.dim, 3 * c.dim, bias=False)
        self.out = nn.Linear(c.dim, c.dim, bias=False)

    def forward(self, x, grids):
        p = grids.pool(x)
        if self.backend == "l4" and self.buffer_fusion:
            from .kernels.buffers import RotaryQKV

            q, k, v = RotaryQKV.apply(
                self.qkv(p), grids.rope_cos, grids.rope_sin, self.heads
            ).unbind(0)
        else:
            qkv = (
                self.qkv(p).view(*p.shape[:2], 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv.unbind(0)
            q, k = rope(q, grids.rope_cos, grids.rope_sin), rope(k, grids.rope_cos, grids.rope_sin)
        y = masked_attention(q, k, v, grids.low_mask[:, None, None, :])
        y = self.out(y.transpose(1, 2).reshape_as(p))
        return grids.interpolate(y)


class TriFusionBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        d, r = c.dim, c.evidence_rank
        self.backend = c.backend
        self.buffer_fusion = c.buffer_fusion
        self.native_kernels = c.native_kernels
        self.paired_mamba = c.paired_mamba
        self.paired_mamba_grid_limit = c.paired_mamba_grid_limit
        self.pre = RMSNorm(d, backend=c.backend)
        self.local, self.attention = LocalCNN(c), GlobalAttention(c)
        self.cn, self.an, self.mn = [RMSNorm(d, backend=c.backend) for _ in range(3)]
        self.e_down = nn.Linear(2 * d, r)
        self.e_value, self.e_gate = nn.Linear(r, d), nn.Linear(r, d)
        self.u_norm = RMSNorm(d, backend=c.backend)
        self.forward_mamba, self.backward_mamba = mamba_factory(c), mamba_factory(c)
        self.gate = nn.Linear(3 * d, 3)
        self.merge_dw = nn.Conv1d(3 * d, 3 * d, 3, padding=1, groups=3 * d)
        self.merge = nn.Linear(3 * d, d, bias=False)
        self.ff_norm = RMSNorm(d, backend=c.backend)
        self.gate_up = nn.Linear(d, 2 * c.ffn_dim, bias=False)
        self.down = nn.Linear(c.ffn_dim, d, bias=False)
        self.dropout = nn.Dropout(c.dropout)
        nn.init.zeros_(self.e_gate.weight)
        nn.init.zeros_(self.e_gate.bias)
        nn.init.normal_(self.e_value.weight, std=1e-3)
        nn.init.zeros_(self.e_value.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        with torch.no_grad():
            self.merge.weight.mul_(1 / math.sqrt(2 * c.depth))
            self.down.weight.mul_(1 / math.sqrt(2 * c.depth))

    def forward(self, x, grids):
        mask = grids.mask
        # One boundary mask gives zero values AND zero padded input gradients.
        # Bias-free projections, RMSNorm and SiLU then preserve these zeros.
        optimized = self.backend == "l4"
        fused = optimized and self.buffer_fusion
        if fused:
            from .kernels.buffers import mask_buffer, depthwise_masked

            apply_mask = mask_buffer
        else:
            apply_mask = masked
        if optimized:
            x = apply_mask(x, mask)

        def zero_preserved(value):
            return value if optimized else masked(value, mask)

        h = zero_preserved(self.pre(x))
        cn = zero_preserved(self.cn(self.local(h, mask, grids.lengths)))
        an = zero_preserved(self.an(self.attention(h, grids)))
        z = F.silu(self.e_down(torch.cat((cn, an), -1)))
        use_native = optimized and self.native_kernels and x.is_cuda and x.shape[-1] in (128, 1024)
        if use_native:
            from .kernels import native

            u = native.evidence_norm(
                x, self.e_value(z), self.e_gate(z), self.u_norm.weight, mask, self.u_norm.eps
            )
        else:
            evidence = self.e_value(z).float() * (2 * torch.sigmoid(self.e_gate(z).float()))
            u = apply_mask(self.u_norm(x + evidence), mask)
        # Each dense row has an independent zero initial state. Causal forward
        # cannot read tail padding; backward reverses only the valid prefix.
        mf = apply_mask(self.forward_mamba(u), mask)
        mb = reverse_valid(
            self.backward_mamba(reverse_valid(u, grids.lengths, self.backend)),
            grids.lengths,
            self.backend,
        )
        mn = (
            native.mean_norm(mf, mb, self.mn.weight, self.mn.eps)
            if use_native
            else zero_preserved(self.mn((mf.float() + mb.float()) * 0.5))
        )
        s = torch.cat((cn, mn, an), -1)
        v = zero_preserved(gated_concat(s, self.gate(s), self.backend))
        conv = (
            depthwise_masked(v, self.merge_dw, grids.lengths)
            if fused
            else masked(self.merge_dw(v.transpose(1, 2)).transpose(1, 2), mask)
        )
        bf16 = (
            torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.bfloat16
        )
        merge_input = native.add(v, conv, bf16=bf16) if use_native else v + conv
        r = zero_preserved(x + self.dropout(self.merge(merge_input)))
        f = swiglu(self.gate_up(self.ff_norm(r)), self.backend)
        down = self.dropout(self.down(f))
        output = native.add(r, down, mask) if use_native else apply_mask(r + down, mask)
        return output, mf, mb


class TriFusion(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        if c.dim % c.heads or c.dim % 32:
            raise ValueError("dim must divide attention heads and Mamba headdim")
        self.front = Frontend(c.backend, c.channels_last, c.checkpoint_frontend)
        self.front_projection = (
            nn.Identity() if c.dim == 1024 else nn.Linear(1024, c.dim, bias=False)
        )
        self.blocks = nn.ModuleList([TriFusionBlock(c) for _ in range(c.depth)])
        from .decoder import Decoder

        self.decoder = Decoder(c.dim)
        self.memory_head = nn.Linear(c.dim, 2)
        self.set_buffer_fusion(c.buffer_fusion)
        self.set_native_kernels(c.native_kernels)

    def set_backend(self, backend):
        if backend not in ("reference", "l4"):
            raise ValueError(backend)
        # Change the backend between inference runs; parameter layout stays fixed.
        for module in self.modules():
            if hasattr(module, "backend"):
                module.backend = backend
        self.config.backend = backend

    def set_buffer_fusion(self, enabled):
        """Switch fused inference buffers; the parameter layout stays fixed."""
        for module in self.modules():
            if hasattr(module, "buffer_fusion"):
                module.buffer_fusion = bool(enabled)
        self.config.buffer_fusion = bool(enabled)

    def set_native_kernels(self, enabled):
        """Switch the fused Triton inference implementation."""
        for module in self.modules():
            if hasattr(module, "native_kernels"):
                module.native_kernels = bool(enabled)
        self.config.native_kernels = bool(enabled)

    def encode(self, audio, lengths, channels):
        x, fine, main_lengths, fine_lengths = self.front(audio, lengths, channels)
        x = self.front_projection(x).float()
        grids = Grids.make(
            main_lengths,
            x.shape[1],
            self.config.dim // self.config.heads,
            self.config.buffer_fusion and self.config.backend == "l4",
            self.config.native_kernels and self.config.backend == "l4",
        )
        for block in self.blocks:
            x, mf, mb = block(x, grids)
        return x, fine, grids, fine_lengths, mf, mb

    def forward(self, audio, lengths, channels):
        return self.decoder(*self.encode(audio, lengths, channels))
