import math
import torch
from torch import nn
import torch.nn.functional as F
from .masking import time_mask
from .attention import masked_attention


def sin_position(time, dim, device):
    phase = (
        torch.arange(time, device=device).float()[:, None]
        * (10000.0 ** (-torch.arange(0, dim, 2, device=device).float() / dim))[None, :]
    )
    return torch.stack((phase.sin(), phase.cos()), -1).flatten(-2)


class QueryRead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.kv = nn.Linear(input_dim, 512, bias=False)
        self.q = nn.Linear(256, 256, bias=False)
        self.out = nn.Linear(256, 256, bias=False)

    def forward(self, query, x, mask):
        b, t, _ = x.shape
        kv = self.kv(x).view(b, t, 2, 8, 32).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        q = self.q(query).view(10, 8, 32).transpose(0, 1).unsqueeze(0).expand(b, -1, -1, -1)
        h = masked_attention(q, k, v, mask[:, None, None, :])
        h = self.out(h.transpose(1, 2).reshape(b, 10, 256))
        return h[:, :4].mean(1), h[:, 4:8].mean(1), h[:, 8:].mean(1)


class Decoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(10, 256) * 0.02)
        self.band = nn.Parameter(torch.randn(8, 64) * 0.02)
        self.deep_read, self.fine_read = QueryRead(dim), QueryRead(64)
        self.global_proj, self.memory_proj = nn.Linear(dim, 256), nn.Linear(2 * dim, 256)
        self.local_logits = nn.Linear(512, 2)
        self.local_features = nn.Linear(512, 256)
        self.voice_lat = nn.Sequential(nn.Linear(1280, 256), nn.SiLU())
        self.music_lat = nn.Sequential(nn.Linear(1280, 256), nn.SiLU())
        self.voice_head, self.music_head = nn.Linear(256, 2), nn.Linear(256, 2)
        self.file_head = nn.Sequential(nn.Linear(1536, 256), nn.SiLU(), nn.Linear(256, 1))

    def forward(self, x, fine, grids, fine_lengths, mf, mb):
        b, tf, _, _ = fine.shape
        fm = time_mask(fine_lengths, tf)
        deep_q = self.deep_read(self.query, x, grids.mask)
        fp = (
            fine
            + sin_position(tf, 64, fine.device)[None, :, None, :].to(fine.dtype)
            + self.band.to(fine.dtype)[None, None, :, :]
        )
        fine_q = self.fine_read(
            self.query, fp.flatten(1, 2), fm[:, :, None].expand(b, tf, 8).reshape(b, -1)
        )
        global_feature = self.global_proj(
            (x.float() * grids.mask[..., None]).sum(1) / grids.lengths[:, None]
        )
        end = mf[torch.arange(b, device=x.device), grids.lengths - 1]
        memory = self.memory_proj(torch.cat((end, mb[:, 0]), -1))
        flat = fine.flatten(2)
        local = self.local_logits(flat)
        features = self.local_features(flat)
        k = min(25, tf)
        scores = local.float().masked_fill(~fm[..., None], -torch.inf)
        idx = scores.transpose(1, 2).topk(k, -1).indices
        selected = (
            features[:, None, :, :]
            .expand(-1, 2, -1, -1)
            .gather(2, idx[..., None].expand(-1, -1, -1, 256))
        )
        valid_k = torch.arange(k, device=x.device)[None, :] < fine_lengths.clamp_max(k)[:, None]
        top = (selected.float() * valid_k[:, None, :, None]).sum(2) / valid_k.sum(1)[:, None, None]
        voice = self.voice_lat(
            torch.cat((deep_q[0], fine_q[0], global_feature, top[:, 0], memory), -1)
        )
        music = self.music_lat(
            torch.cat((deep_q[1], fine_q[1], global_feature, top[:, 1], memory), -1)
        )
        v, m = self.voice_head(voice), self.music_head(music)
        f = self.file_head(
            torch.cat((deep_q[2], fine_q[2], global_feature, memory, voice, music), -1)
        )
        logits = torch.cat((f, v[:, :1], m[:, :1], v[:, 1:], m[:, 1:]), -1).float()
        return {
            "logits": logits,
            "local_logits": local.float(),
            "local_mask": fm,
            "features": x,
            "feature_mask": grids.mask,
            "embedding": global_feature.float(),
        }
