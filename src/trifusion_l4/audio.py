"""Bounded file decoding, exact 16-kHz preprocessing, and length buckets."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch

SAMPLE_RATE = 16000
MAX_SAMPLES = 72 * SAMPLE_RATE
BUCKET_SECONDS = (4, 8, 12, 16, 24, 32, 40, 48, 60, 72)


@dataclass(frozen=True)
class AudioFile:
    index: int
    path: Path
    samples: int
    channels: int


def inspect_audio(index, path):
    path = Path(path)
    info = sf.info(path)
    samples = math.ceil(info.frames * SAMPLE_RATE / info.samplerate)
    if info.channels not in (1, 2) or not 0 < samples <= MAX_SAMPLES:
        raise ValueError(f"Unsupported audio dimensions: {path}")
    return AudioFile(index, path, samples, info.channels)


def decode_audio(record):
    audio, sr = sf.read(record.path, dtype="float32", always_2d=True)
    if sr != SAMPLE_RATE:
        divisor = math.gcd(sr, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // divisor, sr // divisor, axis=0).astype(
            np.float32
        )
    if (
        len(audio) != record.samples
        or audio.shape[1] != record.channels
        or not np.isfinite(audio).all()
    ):
        raise ValueError(f"Audio metadata or finite-value check failed: {record.path}")
    # Match the canonical PCM16 quantization used by this run's train/dev bank.
    pcm = np.clip(np.rint(audio * 32768), -32768, 32767).astype(np.int16)
    return torch.from_numpy(np.ascontiguousarray((pcm.astype(np.float32) / 32768).T))


def plan_batches(records, padded_seconds=120, max_batch=16, round_buckets=True):
    groups = {}
    for r in records:
        limit = (
            next(n for n in BUCKET_SECONDS if r.samples <= n * SAMPLE_RATE) * SAMPLE_RATE
            if round_buckets
            else r.samples
        )
        groups.setdefault((r.channels, limit), []).append(r)
    result = []
    for (channels, limit), rows in sorted(groups.items()):
        count = max(1, min(max_batch, int(padded_seconds * SAMPLE_RATE) // limit))
        for start in range(0, len(rows), count):
            result.append((limit, rows[start : start + count]))
    return result


def decoded_batches(plan, workers=4):
    """At most two audio batches are pending; no full-test waveform cache."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        iterator = iter(plan)

        def submit():
            try:
                limit, rows = next(iterator)
            except StopIteration:
                return None
            return limit, rows, [pool.submit(decode_audio, r) for r in rows]

        pending = submit()
        while pending is not None:
            limit, rows, futures = pending
            pending = submit()
            audio = torch.zeros(
                len(rows), rows[0].channels, limit, pin_memory=torch.cuda.is_available()
            )
            for i, (record, future) in enumerate(zip(rows, futures)):
                audio[i, :, : record.samples] = future.result()
            yield dict(
                audio=audio,
                lengths=torch.tensor([r.samples for r in rows]),
                channels=torch.tensor([r.channels for r in rows]),
                indices=[r.index for r in rows],
            )
