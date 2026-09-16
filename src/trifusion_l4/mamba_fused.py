# SPDX-License-Identifier: Apache-2.0
# Modified/derived from Mamba-3; see THIRD_PARTY_NOTICES.md.
"""SM89 one-pass Mamba-3 SISO inference, without rotated-Q/K or angle banks.

The 64-frame recurrence and intermediate BF16 rounding boundaries match the
portable upstream forward. Angle scan/modulo, bias, rotary, decay, gating and
state updates execute in one kernel. No backward/final-state outputs exist.
Derived from the pinned Mamba source; its license is in vendor/LICENSE.
"""

import torch
import triton as tr
import triton.language as tl
from .vendor.utils import cos_approx, sin_approx, tanh_approx


@tr.jit
def _fused(
    Q,
    K,
    V,
    ADT,
    DT,
    TRAP,
    QB,
    KB,
    ANGLE,
    D,
    Z,
    OUT,
    SQB,
    SQT,
    SQH,
    SQC,
    SKB,
    SKT,
    SKH,
    SKC,
    SVB,
    SVT,
    SVH,
    SVC,
    SAB,
    SAH,
    SAT,
    SDB,
    SDH,
    SDT,
    STB,
    STH,
    STT,
    SGB,
    SGT,
    SGH,
    SGC,
    SZB,
    SZT,
    SZH,
    SZC,
    OB,
    OT,
    OH,
    OC,
    T,
    H: tl.constexpr,
    HQ: tl.constexpr,
    C: tl.constexpr = 64,
    N: tl.constexpr = 128,
    P: tl.constexpr = 64,
    R: tl.constexpr = 32,
):
    head = tl.program_id(0)
    batch = tl.program_id(1)
    qh = head // (H // HQ)
    qbase = Q + batch * SQB + qh * SQH
    kbase = K + batch * SKB + qh * SKH
    vbase = V + batch * SVB + head * SVH
    zbase = Z + batch * SZB + head * SZH
    dtbase = DT + batch * SDB + head * SDH
    abase = ADT + batch * SAB + head * SAH
    trapbase = TRAP + batch * STB + head * STH
    gbase = ANGLE + batch * SGB + head * SGH
    outbase = OUT + batch * OB + head * OH
    ti = tl.arange(0, C)
    ni = tl.arange(0, N)
    pi = tl.arange(0, P)
    ri = tl.arange(0, N // 2)
    qb = tl.load(QB + head * N + ni).to(tl.float32)
    kb = tl.load(KB + head * N + ni).to(tl.float32)
    skip = tl.load(D + head).to(tl.float32)
    state = tl.zeros((P, N), tl.float32)
    angle_state = tl.zeros((N // 2,), tl.float32)
    for chunk in range(tl.cdiv(T, C)):
        pos = chunk * C + ti
        valid = pos < T
        dt = tl.load(dtbase + pos * SDT, valid, 0).to(tl.float32)
        dn = tl.load(dtbase + (pos + 1) * SDT, pos + 1 < T, 0).to(tl.float32)
        trap = tl.sigmoid(tl.load(trapbase + pos * STT, valid, 0).to(tl.float32))
        tn = tl.sigmoid(tl.load(trapbase + (pos + 1) * STT, pos + 1 < T, 0).to(tl.float32))
        gamma = dt * trap
        scale = dn * (1 - tn) + gamma
        raw = tl.load(
            gbase + pos[:, None] * SGT + ri[None, :] * SGC, valid[:, None] & (ri[None, :] < R), 0
        ).to(tl.float32)
        vals = (tanh_approx(raw) * 3.141592653589793) * dt[:, None]
        phase = tl.cumsum(vals, 0) + angle_state[None, :]
        phase = phase - 6.283185307179586 * tl.floor(phase / 6.283185307179586)
        angle_state = angle_state + tl.sum(vals, 0)
        angle_state = angle_state - 6.283185307179586 * tl.floor(angle_state / 6.283185307179586)
        co = cos_approx(phase)
        si = sin_approx(phase)
        q = (
            tl.load(qbase + pos[:, None] * SQT + ni[None, :] * SQC, valid[:, None], 0).to(
                tl.float32
            )
            + qb[None, :]
        )
        k = (
            tl.load(kbase + pos[:, None] * SKT + ni[None, :] * SKC, valid[:, None], 0).to(
                tl.float32
            )
            + kb[None, :]
        )
        qk = tl.sum(q * k, 1) * gamma
        q0, q1 = tl.split(tl.reshape(q, (C, N // 2, 2)))
        k0, k1 = tl.split(tl.reshape(k, (C, N // 2, 2)))
        qr = tl.reshape(tl.join(q0 * co - q1 * si, q0 * si + q1 * co), (C, N)).to(
            Q.dtype.element_ty
        )
        kr = (
            tl.reshape(tl.join(k0 * co - k1 * si, k0 * si + k1 * co), (C, N)) * scale[:, None]
        ).to(K.dtype.element_ty)
        # The two-pass baseline zero-fills tail loads from its rotated banks.
        qr = tl.where(valid[:, None], qr, 0)
        kr = tl.where(valid[:, None], kr, 0)
        v = tl.load(vbase + pos[:, None] * SVT + pi[None, :] * SVC, valid[:, None], 0)
        z = tl.load(zbase + pos[:, None] * SZT + pi[None, :] * SZC, valid[:, None], 0).to(
            tl.float32
        )
        da = tl.load(abase + pos * SAT, valid, 0) * 1.44269504089
        cs = tl.cumsum(da)
        total = tl.sum(da)
        acc = tl.dot(qr, tl.trans(state).to(qr.dtype)) * tl.exp2(cs)[:, None]
        score = tl.dot(qr, tl.trans(kr)) * tl.exp2(tl.minimum(cs[:, None] - cs[None, :], 0.0))
        score = tl.where(ti[:, None] > ti[None, :], score, 0.0)
        acc += tl.dot(score.to(v.dtype), v)
        acc += (skip + qk)[:, None] * v
        acc = acc * (z * tl.sigmoid(z))
        tl.store(outbase + pos[:, None] * OT + pi[None, :] * OC, acc, valid[:, None])
        # Scale V in FP32, then round to K's dtype at the same boundary.
        scaled = v.to(tl.float32) * tl.exp2(total - cs)[:, None]
        state = state * tl.exp2(total) + tl.dot(tl.trans(scaled).to(kr.dtype), kr)


def fused_mamba(q, k, v, adt, dt, trap, qb, kb, angles, d, z, warps=8):
    b, t, h, p = v.shape
    n = q.shape[-1]
    r = angles.shape[-1]
    if n != 128 or p != 64 or r != 32:
        raise ValueError("L4 recurrence requires state=128, head=64, rotary=32")
    out = torch.empty_like(v, memory_format=torch.contiguous_format)
    _fused[(h, b)](
        q,
        k,
        v,
        adt,
        dt,
        trap,
        qb,
        kb,
        angles,
        d,
        z,
        out,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *adt.stride(),
        *dt.stride(),
        *trap.stride(),
        *angles.stride(),
        *z.stride(),
        *out.stride(),
        t,
        h,
        q.shape[2],
        num_warps=warps,
        num_stages=1,
        enable_fp_fusion=False,
    )
    return out
