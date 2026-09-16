import torch
import triton as tr
from . import primitives as p


def mask_buffer(x, mask):
    x = x.contiguous()
    y = torch.empty_like(x)
    p._mask[(tr.cdiv(x.numel(), 512),)](x, mask, y, x.numel(), x.shape[-1], 512)
    return y


def depthwise_masked(x, layer, lengths):
    dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else x.dtype
    x = x.to(dtype).contiguous()
    weight = layer.weight.to(dtype)
    bias = layer.bias.to(dtype)
    y = torch.empty_like(x)
    p._dw_forward[(tr.cdiv(x.numel(), 256),)](
        x,
        weight,
        bias,
        lengths,
        y,
        x.shape[1],
        x.numel(),
        x.shape[-1],
        weight.shape[-1],
        256,
        enable_fp_fusion=False,
    )
    return y


class Pool4:
    @staticmethod
    def apply(x, lengths):
        b, t, d = x.shape
        y = x.new_empty((b, tr.cdiv(t, 4), d))
        p._pool4[(tr.cdiv(y.numel(), 512),)](
            x, lengths, y, t, y.shape[1], y.numel(), d, False, 512, enable_fp_fusion=False
        )
        return y


class RotaryQKV:
    @staticmethod
    def apply(x, cos, sin, heads):
        b, t, width = x.shape
        d = width // (3 * heads)
        y = x.new_empty((3, b, heads, t, d))
        p._rotary_qkv[(tr.cdiv(x.numel() // 2, 512),)](
            x, cos, sin, y, t, b, heads, d, x.numel() // 2, False, 512, enable_fp_fusion=False
        )
        return y


def front_norm(x, lengths, norm, silu=False):
    b, c, t, f = x.shape
    y = x.new_empty((b, t, f, c))
    r = torch.empty(b * t * f, device=x.device)
    p._front_norm[(tr.cdiv(b * t * f, 4),)](
        x,
        norm.weight,
        lengths,
        y,
        r,
        t,
        f,
        b * t * f,
        c,
        *x.stride(),
        norm.eps,
        silu,
        4,
        tr.next_power_of_2(c),
        enable_fp_fusion=False,
    )
    return y.permute(0, 3, 1, 2)


def mamba_controls(a, dt, bias, floor):
    b, t, h = a.shape
    oa = torch.empty((b, h, t), device=a.device)
    od = torch.empty_like(oa)
    p._controls[(tr.cdiv(a.numel(), 256),)](
        a,
        dt,
        bias,
        oa,
        od,
        t,
        b * t,
        h,
        a.stride(0),
        a.stride(1),
        floor,
        256,
        enable_fp_fusion=False,
    )
    return oa, od
