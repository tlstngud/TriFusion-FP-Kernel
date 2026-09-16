# TriFusion-FP-Kernel

**NVIDIA L4 inference kernels and runtime for TriFusion-FP audio models.**

[한국어](README.ko.md) · [API](docs/api.md) · [Benchmarks](docs/benchmarks.md) · [Contributing](CONTRIBUTING.md)

This library combines a fused Mamba-3 recurrence, fused activation/buffer kernels,
frozen BF16 weights, and bounded CUDA Graph replay. It provides a Python tensor
API and an offline audio-to-CSV command. The current release is **inference-only**.

## What executes on the GPU?

| Operation | Implementation |
|---|---|
| Projection GEMMs / convolutions | PyTorch dispatch to cuBLAS / cuDNN |
| Mamba-3, normalization, gates, masking, pooling, buffer fusion | Triton GPU kernels |
| Fast trigonometric helpers | Upstream inline PTX inside Triton |
| Repeated model dispatch | CUDA Graph replay |
| Audio decoding and scheduling | Python, NumPy, SciPy, SoundFile |

Triton is a GPU kernel language/compiler. This is not an all-C++ rewrite or a
C++ ABI library. Moving a wrapper to C++ does not, by itself, remove the cost of
GPU math or memory transfers. No custom `nvcc` build is needed for this release.

## Install

Validated GPU stack: **Linux, NVIDIA L4 (SM89), Python 3.11.15,
PyTorch 2.7.1+cu128, Triton 3.3.1**. Other GPUs and newer framework versions have
not been validated. CPU-only installation supports metadata and CSV tooling.

```bash
git clone https://github.com/tlstngud/TriFusion-FP-Kernel.git
cd TriFusion-FP-Kernel

# Install the tested CUDA build in a dedicated environment.
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install '.[l4]'
trifusion-l4 info
```

If the correct CUDA PyTorch build is already installed, start with the second
installation command. The `l4` extra pins the tested framework versions. The
base package (`pip install .`) does not install or replace PyTorch.

Pretrained checkpoints and audio datasets are not included. Supply your own
compatible `model` + `model_config` checkpoint as described in [the API guide](docs/api.md).
This project is installable from GitHub; it has not been published to PyPI.

## Tensor API

```python
import torch
from trifusion_l4 import configure_runtime, load_student, InferenceEngine

configure_runtime()
model = load_student("/path/to/your/model.pt")
engine = InferenceEngine(model, graphs=True, max_graphs=1)

# audio: float32 [batch, channels, samples], PCM16-equivalent, 16 kHz.
# lengths/channels: int64 [batch] on the same CUDA device.
logits = engine(audio, lengths, channels)
probabilities = logits.float().sigmoid()
engine.clear()
```

The five output tasks, in order, are `file_fake`, `voice_fake`, `music_fake`,
`voice_present`, and `music_present`. There is no gradient/backward API for the
fused inference runtime. FP32 attention and control/normalization calculations
are retained; this release does not use FP8 or INT8 quantization.

A tensor-level `fused_mamba` entry point is also available; see [its contract](docs/api.md#fused-recurrence).

## Offline file inference

```text
work/
  data/
    sample_submission.csv
    test/
      TEST_0000.wav
      TEST_0001.mp3
      TEST_0002.flac
```

```csv
ID,FILE_FAKE_PROB,VOICE_FAKE_PROB,MUSIC_FAKE_PROB,VOICE_PRESENT_PROB,MUSIC_PRESENT_PROB
```

```bash
trifusion-l4 predict --weights /path/to/your/model.pt --root /path/to/work
```

The command writes `work/output/submission.csv` and a runtime report. It reads
IDs and column order from the template. WAV, FLAC, and MP3 decoding was exercised
in the original L4 validation. Audio is resampled to 16 kHz and quantized to the
model's PCM16 convention. Mono/stereo and lengths up to 72 seconds are supported;
longer files are rejected rather than silently truncated.

## Measured performance

Pre-publication measurements on one L4, six CPU cores, the same private checkpoint
and the same real audio inputs. Eleven repetitions; median model-forward latency:

| Input | Portable reference | Fused + CUDA Graph | Speedup |
|---|---:|---:|---:|
| 4 s mono × 16 | 152.71 ms | 79.70 ms | 1.92× |
| 4 s mono × 4 | 117.55 ms | 24.39 ms | 4.82× |
| 32.81 s mono × 1 | 118.33 ms | 42.44 ms | 2.79× |

These are same-GPU inference comparisons, not H100 training comparisons. The
portable reference preserves the model equations while replacing Hopper-only
TMA loads. Timings exclude weight loading, initial compilation, and audio decoding.
The private checkpoint and recordings are not distributed, so these exact
full-model timings cannot be reproduced solely from this repository.

The original implementation also completed a 1,200-file fixture in 40.28 s,
including process start, a fresh Triton cache, decoding and CSV output, with
1.51 GiB peak reserved GPU memory. That fixture repeated 128 validation recordings
and was not an unseen competition test set. Packaging/API changes have CPU checks;
full-model L4 timings have not been rerun for this public package.

[Methodology, numerical limits, and reproduction commands](docs/benchmarks.md).

## Development and license

```bash
python -m pip install '.[dev]'
python -m pytest -q                    # GPU/dependency tests skip when unavailable
python -m build
python -m twine check dist/*
# On the validated L4 environment with the l4 extra installed:
python -m pytest -q tests/test_recurrence.py
```

Apache-2.0. Mamba-3 derivatives and original upstream files are identified in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). This is an early library release;
its API is not yet stable.
