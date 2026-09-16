# Contributing

Use a virtual environment. Install `.[dev]` for packaging/CPU tests and `.[l4]`
with the documented CUDA PyTorch wheel when working on GPU code.

```bash
ruff check .
ruff format --check .
python -m pytest -q
python -m build
python -m twine check dist/*
```

For numerical kernel changes, run the recurrence tests on L4 with the pinned
framework stack. Report shapes, warmup, repetitions, synchronization, peak memory,
and both latency and output differences. Clearly separate model-forward timings
from decoding/loading/compilation. Keep the portable reference independent of
the optimized implementation.

Preserve upstream license notices and update `THIRD_PARTY_NOTICES.md` when
changing derived code. Keep model weights, audio, machine credentials, local
paths, and per-record inference outputs out of commits. GPU CI is not currently
available; CPU checks alone are not GPU validation.
