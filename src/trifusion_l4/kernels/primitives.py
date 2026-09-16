"""Forward primitives copied from validated formulas; no autograd wrapper."""

import triton as tr

import triton.language as tl


@tr.jit
def _rms_fwd(X, W, Y, R, M, N: tl.constexpr, EPS: tl.constexpr, BN: tl.constexpr, BR: tl.constexpr):
    row = tl.program_id(0) * BR + tl.arange(0, BR)
    col = tl.arange(0, BN)
    x = tl.load(X + row[:, None] * N + col[None, :], (row[:, None] < M) & (col[None, :] < N), 0).to(
        tl.float32
    )
    w = tl.load(W + col, col < N, 0).to(tl.float32)
    r = tl.rsqrt(tl.sum(x * x, 1) / N + EPS)
    tl.store(
        Y + row[:, None] * N + col[None, :],
        x * r[:, None] * w[None, :],
        (row[:, None] < M) & (col[None, :] < N),
    )
    tl.store(R + row, r, row < M)


@tr.jit
def _swiglu(X, Y, DY, DX, TOTAL, D: tl.constexpr, BACK: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    off = (i // D) * (2 * D) + i % D
    g = tl.load(X + off, i < TOTAL, 0).to(tl.float32)
    u = tl.load(X + off + D, i < TOTAL, 0).to(tl.float32)
    s = tl.sigmoid(g)
    if BACK:
        dy = tl.load(DY + i, i < TOTAL, 0).to(tl.float32)
        tl.store(DX + off, dy * u * s * (1 + g * (1 - s)), i < TOTAL)
        tl.store(DX + off + D, dy * g * s, i < TOTAL)
    else:
        tl.store(Y + i, g * s * u, i < TOTAL)


@tr.jit
def _reverse(X, L, Y, T, D: tl.constexpr, TOTAL, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    b, t, c = i // (T * D), (i // D) % T, i % D
    length = tl.load(L + b, i < TOTAL, 0)
    src = (b * T + length - 1 - t) * D + c
    x = tl.load(X + src, (i < TOTAL) & (t < length), 0)
    tl.store(Y + i, x, i < TOTAL)


@tr.jit
def _gate(X, G, Y, DY, DX, DG, M, D: tl.constexpr, BD: tl.constexpr, BACK: tl.constexpr):
    # One CTA for each token/branch, reduce scalar gate gradients without atomics.
    row = tl.program_id(0)
    col = tl.arange(0, BD)
    x = tl.load(X + row * D + col, col < D, 0).to(tl.float32)
    s = tl.sigmoid(tl.load(G + row).to(tl.float32))
    if BACK:
        dy = tl.load(DY + row * D + col, col < D, 0).to(tl.float32)
        tl.store(DX + row * D + col, dy * (2 * s), col < D)
        tl.store(DG + row, tl.sum(dy * x, 0) * (2 * s * (1 - s)))
    else:
        tl.store(Y + row * D + col, x * (2 * s), col < D)


@tr.jit
def _mask(X, MASK, Y, TOTAL, D: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = tl.load(MASK + i // D, i < TOTAL, 0)
    v = tl.load(X + i, (i < TOTAL) & valid, 0)
    tl.store(Y + i, v, i < TOTAL)


@tr.jit
def _dw_forward(X, W, BIAS, L, Y, T, TOTAL, D: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    b, t, c = i // (T * D), i // D % T, i % D
    length = tl.load(L + b, i < TOTAL, 0)
    acc = tl.full((BLOCK,), 0, tl.float32)
    for k in tl.static_range(K):
        source = t + k - K // 2
        x = tl.load(X + (b * T + source) * D + c, (i < TOTAL) & (source >= 0) & (source < T), 0).to(
            tl.float32
        )
        w = tl.load(W + c * K + k, i < TOTAL, 0).to(tl.float32)
        acc += x * w
    acc += tl.load(BIAS + c, i < TOTAL, 0).to(tl.float32)
    tl.store(Y + i, tl.where(t < length, acc, 0), i < TOTAL)


@tr.jit
def _pool4(X, L, Y, T, LT, TOTAL, D: tl.constexpr, BACK: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if BACK:
        b, t, c = i // (T * D), i // D % T, i % D
        length = tl.load(L + b, i < TOTAL, 0)
        group = t // 4
        count = tl.maximum(tl.minimum(length - group * 4, 4), 1)
        v = (
            tl.load(X + (b * LT + group) * D + c, (i < TOTAL) & (t < length), 0).to(tl.float32)
            / count
        )
    else:
        b, t, c = i // (LT * D), i // D % LT, i % D
        length = tl.load(L + b, i < TOTAL, 0)
        acc = tl.full((BLOCK,), 0, tl.float32)
        for k in tl.static_range(4):
            source = t * 4 + k
            acc += tl.load(X + (b * T + source) * D + c, (i < TOTAL) & (source < length), 0).to(
                tl.float32
            )
        count = tl.maximum(tl.minimum(length - t * 4, 4), 1)
        v = acc / count
    tl.store(Y + i, v, i < TOTAL)


@tr.jit
def _rotary_qkv(
    X,
    COS,
    SIN,
    Y,
    T,
    B,
    H: tl.constexpr,
    D: tl.constexpr,
    TOTAL,
    BACK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    # One lane handles an adjacent rotary pair. Packed output [3,B,H,T,D]
    c = i % (D // 2)
    t = i // (D // 2) % T
    h = i // (D // 2 * T) % H
    b = i // (D // 2 * T * H) % B
    qkv = i // (D // 2 * T * H * B)
    raw = ((b * T + t) * (3 * H) + qkv * H + h) * D + 2 * c
    packed = 2 * i
    source = packed if BACK else raw
    a = tl.load(X + source, i < TOTAL, 0).to(tl.float32)
    z = tl.load(X + source + 1, i < TOTAL, 0).to(tl.float32)
    co = tl.load(COS + t * (D // 2) + c, i < TOTAL, 0)
    si = tl.load(SIN + t * (D // 2) + c, i < TOTAL, 0)
    if BACK:
        u, v = a * co + z * si, z * co - a * si
    else:
        u, v = a * co - z * si, a * si + z * co
    dest = raw if BACK else packed
    tl.store(Y + dest, tl.where(qkv < 2, u, a), i < TOTAL)
    tl.store(Y + dest + 1, tl.where(qkv < 2, v, z), i < TOTAL)


@tr.jit
def _front_norm(
    X,
    W,
    L,
    Y,
    R,
    T,
    F: tl.constexpr,
    M,
    C: tl.constexpr,
    SB,
    SC,
    ST: tl.constexpr,
    SF: tl.constexpr,
    EPS: tl.constexpr,
    SILU: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
):
    row = tl.program_id(0) * BR + tl.arange(0, BR)
    c = tl.arange(0, BC)
    b, t, f = row // (T * F), row // F % T, row % F
    length = tl.load(L + b, row < M, 0)
    valid = (row[:, None] < M) & (t[:, None] < length[:, None]) & (c[None, :] < C)
    off = b[:, None] * SB + c[None, :] * SC + t[:, None] * ST + f[:, None] * SF
    x = tl.load(X + off, valid, 0).to(tl.float32)
    w = tl.load(W + c, c < C, 0).to(tl.float32)
    r = tl.rsqrt(tl.sum(x * x, 1) / C + EPS)
    # Preserve the BF16 rounding boundary before PyTorch SiLU.
    y = (x * r[:, None] * w[None, :]).to(Y.dtype.element_ty).to(tl.float32)
    if SILU:
        y = y * tl.sigmoid(y)
    tl.store(Y + row[:, None] * C + c[None, :], y, (row[:, None] < M) & (c[None, :] < C))
    tl.store(R + row, r, row < M)


@tr.jit
def _controls(
    A,
    DT,
    BIAS,
    OUT_A,
    OUT_DT,
    T,
    M,
    H: tl.constexpr,
    SB,
    ST,
    FLOOR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, h = i // H, i % H
    b, t = row // T, row % T
    a = tl.load(A + b * SB + t * ST + h, row < M, 0).to(tl.float32)
    dt = tl.load(DT + b * SB + t * ST + h, row < M, 0).to(tl.float32)
    bias = tl.load(BIAS + h, row < M, 0).to(tl.float32)
    control = tl.minimum(-(tl.maximum(a, 0) + 1 / (1 - tl.minimum(a, 0))), -FLOOR)
    v = dt + bias
    step = tl.where(v > 20, v, tl.extra.cuda.libdevice.log1p(tl.exp(v)))
    out = (b * H + h) * T + t
    tl.store(OUT_DT + out, step, row < M)
    tl.store(OUT_A + out, control * step, row < M)
