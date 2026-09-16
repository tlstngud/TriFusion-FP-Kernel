# API and compatibility

## Checkpoint format

`load_student(path)` reads a locally produced, trusted, model-only PyTorch
checkpoint with `weights_only=True` and memory mapping:

```python
{
    "model_config": {"dim": 1024, "depth": 16, ...},
    "model": model_state_dict,
}
```

The state dictionary must match `trifusion_l4.model.TriFusion`. Keys prefixed
`kd_features.` and `memory_head.` are stripped; other missing or unexpected
parameters fail strict loading. Optimizer/RNG resume snapshots need a separate
model-only export. No Hugging Face download is performed.

The model is created on the meta device, checkpoint tensors are attached with
`assign=True`, and the non-persistent STFT window is materialized explicitly.
Linear/Conv weights are converted once to BF16. Norm/control parameters remain
FP32. All parameters are frozen and the model is put into evaluation mode.

## Runtime

- `configure_runtime(threads=2, enforce_l4=True)`: requires CUDA and, by default,
  SM89. Sets CPU threads and disables TF32 and cuDNN autotuning. SM89 checking
  does not certify every SM89 GPU; measurements were made specifically on L4.
- `load_student(path, optimized=True, device="cuda")`: returns the frozen model.
  `optimized=False` selects the portable reference. `device="cpu"` is useful
  for checkpoint handling, not CPU inference through the CUDA runtime.
- `InferenceEngine(model, graphs=False, max_graphs=2, max_reserved_gib=19.5)`:
  `model` must already be in evaluation mode. Use a positive graph-cache size.
- `engine(audio, lengths, channels)`: returns **logits** `[B, 5]`; call
  `.float().sigmoid()` to obtain probabilities.
- `engine.clear()`: releases cached graphs and calls the CUDA allocator's
  empty-cache operation. Coordinate it with other CUDA users in the process.

Inputs: contiguous CUDA `float32` audio `[B, C, N]`, `C` equal to 1 or 2, and
CUDA `int64` lengths/channels `[B]`. Audio is 16 kHz, PCM16-equivalent in
`[-1, 1)`. Every length is positive and at most `N`. Each channels value is
positive and at most `C`. Keep audio and metadata on the same CUDA device.

Graph keys include audio shape and dtype. Audio, lengths, and channels are copied
before replay; recurrence states start independently for each file. Returned
logits own their storage. One engine is intended for sequential calls from one
inference loop; concurrent calls require external synchronization.

## Fused recurrence

```python
from trifusion_l4 import fused_mamba

y = fused_mamba(q, k, v, adt, dt, trap, qb, kb, angles, d, z)
```

This is a low-level forward-only entry point. It is not a drop-in general Mamba
training implementation. Call it under `torch.inference_mode()`.

| Tensor | Shape / dtype |
|---|---|
| `q`, `k` | `[B, T, Hq, 128]`, BF16; `H` divisible by `Hq` |
| `v`, `z` | `[B, T, H, 64]`, BF16 |
| `adt`, `dt`, `trap` | `[B, H, T]`, FP32 control values |
| `qb`, `kb` | `[H, 128]`, contiguous FP32 |
| `angles` | `[B, T, H, 32]`, raw BF16 angle controls |
| `d` | `[H]`, contiguous FP32 |
| output | `[B, T, H, 64]`, BF16 |

All tensors must be CUDA tensors on the same device. Time/batch strides from
projected tensors are supported. The recurrence uses 64-frame chunks and eight
warps per program by default. `dt` is already softplus-transformed; `adt` is
already the signed decay multiplied by `dt`. `trap` remains pre-sigmoid.

## CSV tools

`trifusion_l4.submission_format` uses only the Python standard library and can be
imported without GPU dependencies. Official columns use the `_PROB` suffix;
unsuffixed names are accepted for older local fixtures. Template row and column
order are preserved. IDs are matched to filenames, stems, or relative paths.

The file runner uses length/channel buckets, four decoder workers and at most
two pending CPU batches. The default padded-audio budget is 64 seconds per batch
with at most 16 files. Inputs over 72 seconds are rejected. The extension lookup
also recognizes OGG/M4A, but decoding those containers depends on the available
libsndfile build; the original end-to-end validation covered WAV/FLAC/MP3.
