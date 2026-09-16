"""Compare full-model forward paths with a user checkpoint and synthetic audio."""

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import torch
from trifusion_l4 import configure_runtime, load_student, InferenceEngine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--seconds", type=float, default=4)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--channels", type=int, choices=[1, 2], default=1)
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.seconds <= 72 or args.batch < 1 or args.repeats < 1:
        parser.error("Use a positive batch/repeat count and 0 < seconds <= 72")
    configure_runtime()
    torch.manual_seed(551)
    samples = max(1, round(args.seconds * 16000))
    audio = torch.randn(args.batch, args.channels, samples, device="cuda") * 0.1
    lengths = torch.full((args.batch,), samples, device="cuda", dtype=torch.int64)
    channels = torch.full((args.batch,), args.channels, device="cuda", dtype=torch.int64)
    report = {
        "input": "synthetic random audio",
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "shape": list(audio.shape),
        "variants": [],
    }
    reference = None
    for name in ["reference", "fused", "graph"]:
        model = load_student(args.weights, optimized=name != "reference")
        engine = InferenceEngine(model, graphs=name == "graph", max_graphs=1)
        start = time.perf_counter()
        output = engine(audio, lengths, channels)
        torch.cuda.synchronize()
        cold = time.perf_counter() - start
        if not bool(output.isfinite().all()):
            raise RuntimeError("Nonfinite benchmark output")
        probabilities = output.float().sigmoid().cpu()
        if reference is None:
            reference = probabilities
        difference = float((probabilities - reference).abs().max())
        for _ in range(3):
            engine(audio, lengths, channels)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            engine(audio, lengths, channels)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        row = {
            "variant": name,
            "first_forward_seconds": cold,
            "median_seconds": statistics.median(times),
            "times": times,
            "probability_max_abs": difference,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        }
        report["variants"].append(row)
        print(json.dumps(row), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        engine.clear()
        del output, engine, model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
