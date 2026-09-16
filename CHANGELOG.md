# Changelog

## 0.1.0

- Initial public, inference-only L4 package.
- Fused Mamba-3 recurrence and buffer/activation kernels with a portable reference.
- Frozen BF16 weights, meta-device loading, and bounded CUDA Graph replay.
- Lazy Python API, `trifusion-l4` CLI, and offline CSV interface.
- Packaging/CPU CI, synthetic GPU recurrence tests, benchmark tool, and provenance.
