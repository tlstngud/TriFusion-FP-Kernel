"""Fused inference kernels compiled to SM89 by Triton, with FP32 statistics."""

import torch
import triton as tr
import triton.language as tl


@tr.jit
def _norm(
    X,
    A,
    B,
    W,
    MASK,
    Y,
    M,
    D: tl.constexpr,
    EPS: tl.constexpr,
    MODE: tl.constexpr,
    BD: tl.constexpr,
    BR: tl.constexpr,
):
    row = tl.program_id(0) * BR + tl.arange(0, BR)
    col = tl.arange(0, BD)
    valid = row < M
    if MODE == 1:
        valid = valid & tl.load(MASK + row, row < M, 0)
    off = row[:, None] * D + col[None, :]
    mask = valid[:, None] & (col[None, :] < D)
    x = tl.load(X + off, mask, 0).to(tl.float32)
    if MODE == 1:
        a = tl.load(A + off, mask, 0).to(tl.float32)
        b = tl.load(B + off, mask, 0).to(tl.float32)
        x = x + a * (2 * tl.sigmoid(b))
    elif MODE == 2:
        x = (x + tl.load(A + off, mask, 0).to(tl.float32)) * 0.5
    r = tl.rsqrt(tl.sum(x * x, 1) / D + EPS)
    w = tl.load(W + col, col < D, 0)
    tl.store(Y + off, x * r[:, None] * w[None, :], (row[:, None] < M) & (col[None, :] < D))


def _apply_norm(x, a, b, w, mask, eps, mode):
    x = x.contiguous()
    a = a.contiguous()
    b = b.contiguous()
    dtype = torch.float32 if mode else x.dtype
    y = torch.empty(x.shape, device=x.device, dtype=dtype)
    d = x.shape[-1]
    m = x.numel() // d
    _norm[(tr.cdiv(m, 4),)](
        x,
        a,
        b,
        w,
        mask,
        y,
        m,
        d,
        eps,
        mode,
        tr.next_power_of_2(d),
        4,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return y


def rms(x, weight, eps=1e-5):
    return _apply_norm(x, x, x, weight, x, eps, 0)


def evidence_norm(x, value, gate, weight, mask, eps=1e-5):
    return _apply_norm(x, value, gate, weight, mask, eps, 1)


def mean_norm(a, b, weight, eps=1e-5):
    return _apply_norm(a, b, b, weight, a, eps, 2)


@tr.jit
def _add(A, B, MASK, Y, N, D: tl.constexpr, MASKED: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < N
    if MASKED:
        valid = valid & tl.load(MASK + i // D, i < N, 0)
    a = tl.load(A + i, valid, 0).to(tl.float32)
    b = tl.load(B + i, valid, 0).to(tl.float32)
    tl.store(Y + i, a + b, i < N)


def add(a, b, mask=None, bf16=False):
    a = a.contiguous()
    b = b.contiguous()
    dtype = torch.bfloat16 if bf16 else torch.promote_types(a.dtype, b.dtype)
    y = torch.empty(a.shape, device=a.device, dtype=dtype)
    _add[(tr.cdiv(a.numel(), 512),)](
        a, b, a if mask is None else mask, y, a.numel(), a.shape[-1], mask is not None, 512
    )
    return y


@tr.jit
def _interpolate(X, LEFT, RIGHT, ALPHA, LENGTHS, Y, T, LT, N, D: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    b = i // (T * D)
    t = i // D % T
    c = i % D
    valid = (i < N) & (t < tl.load(LENGTHS + b, i < N, 0))
    row = b * T + t
    left = tl.load(LEFT + row, i < N, 0)
    right = tl.load(RIGHT + row, i < N, 0)
    alpha = tl.load(ALPHA + row, i < N, 0)
    a = tl.load(X + (b * LT + left) * D + c, valid, 0).to(tl.float32)
    z = tl.load(X + (b * LT + right) * D + c, valid, 0).to(tl.float32)
    tl.store(Y + i, a + (z - a) * alpha, i < N)


def interpolate(low, grids):
    low = low.contiguous()
    b, t = grids.left.shape
    d = low.shape[-1]
    y = low.new_empty((b, t, d))
    _interpolate[(tr.cdiv(y.numel(), 512),)](
        low,
        grids.left,
        grids.right,
        grids.alpha,
        grids.lengths,
        y,
        t,
        low.shape[1],
        y.numel(),
        d,
        512,
        enable_fp_fusion=False,
    )
    return y


@tr.jit
def _bands(X, Y, T, F: tl.constexpr, C: tl.constexpr, N, SB, SC, ST, SF, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c = i % C
    band = i // C % 8
    t = i // (C * 8) % T
    b = i // (C * 8 * T)
    start = band * F // 8
    end = ((band + 1) * F + 7) // 8
    acc = tl.full((BLOCK,), 0, tl.float32)
    for j in tl.static_range(tr.cdiv(F, 8) + 1):
        f = start + j
        acc += tl.load(X + b * SB + c * SC + t * ST + f * SF, (i < N) & (f < end), 0).to(tl.float32)
    tl.store(Y + i, acc / (end - start), i < N)


def bands(x):
    b, c, t, f = x.shape
    y = x.new_empty((b, t, 8, c))
    _bands[(tr.cdiv(y.numel(), 256),)](
        x, y, t, f, c, y.numel(), *x.stride(), 256, enable_fp_fusion=False
    )
    return y
