"""FP32 masked attention for the validated inference path."""

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def masked_attention(q, k, v, mask):
    # Match the FP32 attention path used for numerical validation. Projection
    # GEMMs remain BF16; attention keeps FP32 for large-offset/cancellation cases.
    dtype = q.dtype
    with (
        torch.autocast(q.device.type, enabled=False),
        sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]),
    ):
        if dtype in (torch.float16, torch.bfloat16):
            q, k, v = q.float(), k.float(), v.float()
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0).to(dtype)
