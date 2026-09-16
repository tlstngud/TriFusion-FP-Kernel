from dataclasses import dataclass
import torch
import torch.nn.functional as F


def time_mask(lengths, time):
    return torch.arange(time, device=lengths.device)[None, :] < lengths[:, None]


def masked(x, mask):
    return x.masked_fill(~mask[..., None], 0)


@dataclass
class Grids:
    lengths: torch.Tensor
    mask: torch.Tensor
    low_lengths: torch.Tensor
    low_mask: torch.Tensor
    left: torch.Tensor
    right: torch.Tensor
    alpha: torch.Tensor
    rope_cos: torch.Tensor
    rope_sin: torch.Tensor
    buffer_fusion: bool = False
    native_kernels: bool = False

    @classmethod
    def make(cls, lengths, time, head_dim=64, buffer_fusion=False, native_kernels=False):
        low_lengths = (lengths + 3) // 4
        low_t = (time + 3) // 4
        # align_corners=False, separately for each valid source/target grid.
        pos = torch.arange(time, device=lengths.device)[None, :].float() + 0.5
        pos = (pos * low_lengths[:, None] / lengths[:, None] - 0.5).clamp_min(0)
        pos = torch.minimum(pos, (low_lengths - 1)[:, None])
        left = pos.long()
        right = torch.minimum(left + 1, (low_lengths - 1)[:, None])
        # Q/K in all 16 blocks share the same deterministic rotary grid.
        phase = (
            torch.arange(low_t, device=lengths.device).float()[:, None]
            * (
                10000.0 ** (-torch.arange(0, head_dim, 2, device=lengths.device).float() / head_dim)
            )[None, :]
        )
        return cls(
            lengths,
            time_mask(lengths, time),
            low_lengths,
            time_mask(low_lengths, low_t),
            left,
            right,
            pos - left,
            phase.cos(),
            phase.sin(),
            buffer_fusion,
            native_kernels,
        )

    def pool(self, x):
        if self.buffer_fusion:
            from .kernels.buffers import Pool4

            return Pool4.apply(x.contiguous(), self.lengths)
        b, t, d = x.shape
        xp = F.pad(masked(x, self.mask), (0, 0, 0, (-t) % 4))
        mp = F.pad(self.mask, (0, (-t) % 4)).view(b, -1, 4)
        # FP32 reductions, last partial group divided only by valid count.
        y = xp.float().view(b, -1, 4, d).sum(2) / mp.sum(2).clamp_min(1)[..., None]
        return y.to(x.dtype)

    def interpolate(self, low):
        if self.native_kernels:
            from .kernels.native import interpolate

            return interpolate(low, self)
        shape = (*self.left.shape, low.shape[-1])
        a = low.gather(1, self.left[..., None].expand(shape))
        b = low.gather(1, self.right[..., None].expand(shape))
        y = a.float() + (b.float() - a.float()) * self.alpha[..., None]
        return masked(y.to(low.dtype), self.mask)
