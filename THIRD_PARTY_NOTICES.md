# Third-party notices

## Mamba / Mamba-3

- Project: https://github.com/state-spaces/mamba
- Revision: `e9594ce1c732d97440f0332fdc43170a2294dbfa`
- License: Apache-2.0; preserved in `src/trifusion_l4/vendor/LICENSE`.
- Copyright notices on the Mamba-3 forward and utility sources identify
  **Dao AI Lab and Goombalab, 2025**. Original notices are preserved.

Unmodified upstream files are kept in `src/trifusion_l4/vendor/`:
`mamba3_siso_fwd.py`, `angle_dt.py`, and `utils.py`. Their original repository
paths and SHA-256 hashes are recorded in `vendor/UPSTREAM.json`.

Modified and derived files:

- `mamba_portable.py`: replaces Hopper TMA descriptors with block-pointer
  loads/stores; fixes the launch configuration and Triton 3.3 compatibility.
- `angle.py`: keeps the forward angle scan with local utility imports and a
  fixed launch configuration.
- `mamba_fused.py`: combines the angle scan, rotary transforms, recurrence,
  skip connection, and gate into an inference-only SM89 kernel. It preserves
  the original 64-frame chunk equations and BF16 rounding boundaries.
- `mamba.py`: provides the compatible parameter layout and inference wrapper.

PyTorch, Triton, NumPy, SciPy, SoundFile, and einops are separately installed
runtime dependencies and retain their respective upstream licenses.
