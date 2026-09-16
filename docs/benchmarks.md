# Benchmarks and validation limits

## Original L4 measurements

The aggregate report in `benchmarks/results/l4_20260916.json` was measured on
NVIDIA L4, PyTorch 2.7.1+cu128, Triton 3.3.1, Python 3.11.15, and six CPU cores.
The host was Ubuntu 24.04.1. BF16 projection/convolution weights and FP32
attention/control calculations were used.

The comparison uses the same private checkpoint and inputs. The portable
reference preserves the upstream model equations while replacing Hopper TMA
accesses with SM89-compatible block loads. Model-forward timing includes host
launch and synchronization and excludes loading, initial compilation, decoding,
and CSV writing. Eleven repetitions are reported without selecting the fastest
single run. Larger batches reduce the relative benefit from launch elimination.

A separate full-program fixture repeated 128 validation recordings to make
1,200 IDs, with 12,205.4 seconds of audio in total. It contained WAV/FLAC/MP3 and
some resampled stereo inputs. Process start, loading, a fresh Triton cache,
inference, and CSV writing took 40.28 seconds. Peak reserved GPU memory was
1.506 GiB; process peak RSS was 1.750 GiB. Fixture generation was outside the
timer. This was not an unseen benchmark or the official competition test set.

These measurements predate public-package metadata, CLI, and code formatting.
The numerical kernel ASTs are preserved. The public package has installation
and CPU interface checks; fresh full-model L4 performance results are not claimed.
Neither the private checkpoint nor recordings are distributed. Performance
changes with GPU, driver, thermal state, batch shape, and checkpoint dimensions.

## Numerical validation performed on L4

- Eight recurrence tests: boundary/tail sizes, strided inputs, the portable
  reference, and an independent FP64 unchunked recurrence.
- Whole model: 128 private validation inputs, finite outputs, maximum probability
  difference `0.00158268`, mean difference `0.0000166251` against the reference.
- Maximum raw-logit difference was `4.0` on saturated outputs. This is not a
  bitwise-equivalent model optimization.
- Single-file vs batch probability difference reached `0.003332`; padding
  difference `0.0001035`; tested file permutation and graph/eager comparisons
  had zero difference. Batch-independent state does not imply bitwise-identical
  BF16 arithmetic across different GEMM shapes.
- Two expanded 60/72-second stereo signals passed whole-model comparison.

These checks constrain numerical drift. They do not establish accuracy on every
class or a public leaderboard score. CPU CI does not execute GPU kernels.

## Reproduce with your own checkpoint

Install the tested L4 stack and use your own compatible model-only checkpoint:

```bash
python benchmarks/benchmark_inference.py --weights /path/to/model.pt \
  --seconds 4 --batch 16 --channels 1 --repeats 11 --output benchmark.json
python benchmarks/benchmark_inference.py --weights /path/to/model.pt \
  --seconds 60 --batch 1 --channels 2 --repeats 11 --output benchmark-long.json
python -m pytest -q tests/test_recurrence.py
```

The benchmark script generates deterministic random audio and reports its
synthetic origin. Use the same checkpoint and shapes when comparing variants.
It checks finite outputs, but synthetic timing input does not measure detector
accuracy. Core recurrence tests require no checkpoint or dataset.
