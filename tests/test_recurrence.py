import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
    pytest.skip(
        "Recurrence tests require the validated L4/SM89 CUDA environment", allow_module_level=True
    )
pytestmark = pytest.mark.gpu
from trifusion_l4.angle import angle_dt_fwd
from trifusion_l4.mamba_portable import mamba3_siso_fwd
from trifusion_l4.mamba_fused import fused_mamba


def inputs(batch, time, heads=4):
    torch.manual_seed(121 + time)
    q, k = [
        torch.randn(batch, time, 1, 128, device="cuda", dtype=torch.bfloat16) * 0.2
        for _ in range(2)
    ]
    # Strided projected V/Z/angles exercise the eliminated contiguous copies.
    raw = torch.randn(batch, time, heads * 64 * 2 + 32, device="cuda", dtype=torch.bfloat16) * 0.2
    v = raw[:, :, : heads * 64].reshape(batch, time, heads, 64)
    z = raw[:, :, heads * 64 : heads * 128].reshape_as(v)
    angles = raw[:, :, -32:].unsqueeze(2).expand(batch, time, heads, 32)
    dt = torch.rand(batch, heads, time, device="cuda") * 0.1 + 0.01
    adt = -dt * (torch.rand_like(dt) * 2 + 0.2)
    trap = torch.randn_like(dt)
    qb, kb = [torch.randn(heads, 128, device="cuda") * 0.1 + 1 for _ in range(2)]
    d = torch.ones(heads, device="cuda")
    return q, k, v, adt, dt, trap, qb, kb, angles, d, z


@pytest.mark.parametrize(
    "batch,time,heads",
    [(1, 1, 4), (2, 26, 4), (1, 63, 32), (1, 64, 32), (2, 65, 4), (1, 255, 32), (1, 1501, 32)],
)
def test_fused_recurrence_matches_portable(batch, time, heads):
    args = inputs(batch, time, heads)
    q, k, v, adt, dt, trap, qb, kb, angles, d, z = args
    phase = angle_dt_fwd(angles.float(), dt, chunk_size=64)
    ref = mamba3_siso_fwd(q, k, v, adt, dt, trap, qb, kb, phase, d, z, chunk_size=64)[0]
    actual = fused_mamba(*args)
    assert actual.isfinite().all()
    error = (actual.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-20)
    assert error < 0.006, float(error)
    torch.testing.assert_close(actual, ref, atol=0.04, rtol=0.02)


def test_short_recurrence_against_independent_float64_formula():
    args = inputs(1, 17, 4)
    q, k, v, adt, dt, trap, qb, kb, angles, d, z = args
    # Independent unchunked mathematical recurrence, without Triton helpers.
    phase = (angles.double().tanh() * torch.pi * dt.double().transpose(1, 2)[..., None]).cumsum(1)
    phase = torch.nn.functional.pad(phase, (0, 32))

    def rotate(x, bias):
        x = x.double().expand(-1, -1, 4, -1) + bias.double()[None, None]
        a, b = x.reshape(1, 17, 4, 64, 2).unbind(-1)
        return torch.stack(
            (a * phase.cos() - b * phase.sin(), a * phase.sin() + b * phase.cos()), -1
        ).flatten(-2)

    qr, kr = rotate(q, qb), rotate(k, kb)
    gamma = dt.double() * trap.double().sigmoid()
    shifted = torch.nn.functional.pad(
        (dt.double() * (1 - trap.double().sigmoid()))[:, :, 1:], (0, 1)
    )
    scale = gamma + shifted
    state = torch.zeros(1, 4, 64, 128, device="cuda", dtype=torch.float64)
    outputs = []
    for i in range(17):
        decay = adt[:, :, i].double().exp()
        vi = v[:, i].double()
        previous = torch.einsum("bhn,bhpn->bhp", qr[:, i], state) * decay[..., None]
        direct = ((q[:, i].double() + qb) * (k[:, i].double() + kb)).sum(-1) * gamma[:, :, i]
        outputs.append(
            (previous + (d + direct)[..., None] * vi) * torch.nn.functional.silu(z[:, i].double())
        )
        state = (
            state * decay[..., None, None]
            + vi[..., None] * kr[:, i, :, None, :] * scale[:, :, i, None, None]
        )
    ref = torch.stack(outputs, 1)
    actual = fused_mamba(*args).double()
    assert ((actual - ref).norm() / ref.norm()) < 0.012
