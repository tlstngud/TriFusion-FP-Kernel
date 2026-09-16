# SPDX-License-Identifier: Apache-2.0
# Modified/derived from Mamba-3; see THIRD_PARTY_NOTICES.md.
"""Frozen Mamba-3 SISO forward with the original checkpoint parameter layout."""

import torch
from torch import nn
import torch.nn.functional as F
from .kernels import RMSNorm
from .kernels.buffers import mamba_controls
from .angle import angle_dt_fwd
from .mamba_portable import mamba3_siso_fwd


class Mamba3Inference(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_inner = config.dim * 2
        self.nheads = self.d_inner // 64
        self.headdim = 64
        self.d_state = config.state_dim
        self.num_bc_heads = 1
        self.num_rope_angles = self.d_state // 4
        self.A_floor = 1e-4
        self.chunk_size = 64
        self.backend = config.backend
        self.in_proj = nn.Linear(
            config.dim,
            2 * self.d_inner + 2 * self.d_state + 3 * self.nheads + self.num_rope_angles,
            bias=False,
        )
        self.out_proj = nn.Linear(self.d_inner, config.dim, bias=False)
        self.dt_bias = nn.Parameter(torch.full((self.nheads,), -4.0))
        self.B_bias = nn.Parameter(torch.ones(self.nheads, 1, self.d_state))
        self.C_bias = nn.Parameter(torch.ones_like(self.B_bias))
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.B_norm = RMSNorm(self.d_state, backend=config.backend)
        self.C_norm = RMSNorm(self.d_state, backend=config.backend)
        self.fused_recurrence = False

    def forward(self, u):
        b, t, _ = u.shape
        p = self.in_proj(u)
        z, v, k, q, ddt, a, trap, angles = p.split(
            [
                self.d_inner,
                self.d_inner,
                self.d_state,
                self.d_state,
                self.nheads,
                self.nheads,
                self.nheads,
                self.num_rope_angles,
            ],
            -1,
        )
        q = self.C_norm(q.reshape(b, t, 1, self.d_state))
        k = self.B_norm(k.reshape(b, t, 1, self.d_state))
        if self.backend == "reference":
            af = a.float()
            control = -(af.clamp_min(0) + 1 / (1 - af.clamp_max(0))).clamp_min(self.A_floor)
            dt = F.softplus(ddt.float() + self.dt_bias)
            adt = (control * dt).transpose(1, 2).contiguous()
            dt = dt.transpose(1, 2).contiguous()
        else:
            adt, dt = mamba_controls(a, ddt, self.dt_bias, self.A_floor)
        angles = angles.unsqueeze(2).expand(b, t, self.nheads, self.num_rope_angles)
        if self.fused_recurrence:
            from .mamba_fused import fused_mamba

            y = fused_mamba(
                q,
                k,
                v.reshape(b, t, self.nheads, 64),
                adt,
                dt,
                trap.transpose(1, 2),
                self.C_bias[:, 0],
                self.B_bias[:, 0],
                angles,
                self.D,
                z.reshape(b, t, self.nheads, 64),
            )
            return self.out_proj(y.reshape(b, t, self.d_inner))
        angles = angles.float()
        phase = angle_dt_fwd(angles, dt, chunk_size=self.chunk_size)
        y = mamba3_siso_fwd(
            Q=q,
            K=k,
            V=v.reshape(b, t, self.nheads, 64),
            ADT=adt,
            DT=dt,
            Trap=trap.transpose(1, 2),
            Q_bias=self.C_bias[:, 0],
            K_bias=self.B_bias[:, 0],
            Angles=phase,
            D=self.D,
            Z=z.reshape(b, t, self.nheads, 64),
            chunk_size=self.chunk_size,
            store_states_adt_outv=False,
            return_final_states=False,
        )[0]
        return self.out_proj(y.reshape(b, t, self.d_inner))
