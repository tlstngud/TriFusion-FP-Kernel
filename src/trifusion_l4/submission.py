"""Read the actual evaluation template and preserve its IDs and column order."""

from pathlib import Path
import json
import os
import resource
import time
import numpy as np
import torch
from .audio import inspect_audio, plan_batches, decoded_batches
from .runtime import configure_runtime, load_student, InferenceEngine

from .submission_format import OUTPUTS, discover_inputs, write_submission


def run(root, weights, graphs=True, padded_seconds=64, max_batch=16, workers=4):
    start = time.perf_counter()
    allowed = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, set(allowed[:6]))
    configure_runtime(threads=2)
    columns, rows, aliases, paths = discover_inputs(root)
    records = [inspect_audio(i, path) for i, path in enumerate(paths)]
    plan = plan_batches(records, padded_seconds, max_batch, round_buckets=graphs)
    metadata_seconds = time.perf_counter() - start
    torch.cuda.reset_peak_memory_stats()
    model = load_student(weights, optimized=True)
    engine = InferenceEngine(model, graphs=graphs, max_graphs=1)
    model_load_seconds = time.perf_counter() - start - metadata_seconds
    result = np.full((len(rows), 5), np.nan, dtype=np.float32)
    seen = np.zeros(len(rows), dtype=bool)
    with torch.inference_mode():
        for batch in decoded_batches(plan, workers):
            indices = batch["indices"]
            if seen[indices].any():
                raise ValueError("An input was scheduled twice")
            inputs = [
                batch[name].cuda(non_blocking=True) for name in ("audio", "lengths", "channels")
            ]
            try:
                logits = engine(*inputs)
            except torch.cuda.OutOfMemoryError:
                logits = None
            if logits is None:
                # Leave the exception handler so the failed graph's activations
                # are released before retrying every file independently.
                engine.clear()
                parts = []
                for i in range(len(indices)):
                    parts.append(engine.eager(*(x[i : i + 1] for x in inputs)))
                logits = torch.cat(parts)
            result[indices] = logits.float().sigmoid().cpu().numpy()
            seen[indices] = True
    if not seen.all():
        raise ValueError("Missing predictions")
    output = Path(root) / "output"
    write_submission(output / "submission.csv", columns, rows, aliases, result)
    report = dict(
        files=len(rows),
        batches=len(plan),
        wall_seconds=time.perf_counter() - start,
        metadata_seconds=metadata_seconds,
        model_load_seconds=model_load_seconds,
        peak_process_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
        graph_build_seconds=engine.graph_build_seconds,
        graph_hits=engine.graph_hits,
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(),
    )
    (output / "runtime_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return report
