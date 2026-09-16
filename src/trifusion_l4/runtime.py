"""Bounded-memory inference and optional CUDA graph replay for frozen weights."""

from collections import OrderedDict
from dataclasses import fields
import gc
import time
import torch
from .model import TriFusion, ModelConfig


def load_student(path, optimized=True, device="cuda"):
    # Trusted locally produced weights, never downloaded by the inference code.
    saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    config = dict(saved["model_config"])
    known = {f.name for f in fields(ModelConfig)}
    if set(config) - known:
        raise ValueError("Unsupported model configuration fields")
    config.update(
        backend="l4" if optimized else "reference",
        checkpoint_blocks=False,
        checkpoint_frontend=False,
        buffer_fusion=optimized,
        native_kernels=optimized,
        paired_mamba=False,
    )
    # Allocate shapes only; avoid initializing 528M parameters that the checkpoint
    # immediately overwrites. assign=True attaches the mmap-backed CPU tensors.
    with torch.device("meta"):
        model = TriFusion(ModelConfig(**config))
    # Both training checkpoints and stripped deployment weights are accepted.
    del model.memory_head
    state = {
        k: v
        for k, v in saved["model"].items()
        if not k.startswith(("kd_features.", "memory_head."))
    }
    model.load_state_dict(state, strict=True, assign=True)
    # The STFT window is deliberately non-persistent, so materialize it explicitly.
    model.front.window = torch.hann_window(400, periodic=True)
    model.eval().requires_grad_(False)
    # No training-only head in the deployed network; decoder endpoint reads stay.
    for block in model.blocks:
        block.forward_mamba.fused_recurrence = optimized
        block.backward_mamba.fused_recurrence = optimized
    if optimized:
        prepare_static_weights(model)
    return model.to(device)


def prepare_static_weights(model):
    """Cast frozen GEMM/conv parameters once; normalization and controls stay FP32."""
    for module in model.modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)):
            module.to(dtype=torch.bfloat16)


def configure_runtime(threads=2, enforce_l4=True):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    if enforce_l4 and torch.cuda.get_device_capability() != (8, 9):
        raise RuntimeError("This runtime is validated for L4/SM89")
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


class InferenceEngine:
    def __init__(self, model, graphs=False, max_graphs=2, max_reserved_gib=19.5):
        self.model = model
        self.graphs = graphs
        self.max_graphs = max_graphs
        self.limit = int(max_reserved_gib * 2**30)
        self.cache = OrderedDict()
        self.disabled_shapes = set()
        self.graph_build_seconds = 0.0
        self.graph_hits = 0

    @torch.inference_mode()
    def eager(self, audio, lengths, channels):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.model(audio, lengths, channels)["logits"]

    def clear(self):
        self.cache.clear()
        gc.collect()
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def __call__(self, audio, lengths, channels):
        key = (tuple(audio.shape), audio.dtype)
        if not self.graphs or key in self.disabled_shapes:
            return self.eager(audio, lengths, channels)
        if key not in self.cache:
            while len(self.cache) >= self.max_graphs:
                self.cache.popitem(last=False)
            gc.collect()
            torch.cuda.empty_cache()
            start = time.perf_counter()
            x = audio.clone()
            l = lengths.clone()
            c = channels.clone()
            # Compile all kernels and initialize cuFFT/cuBLAS before capture.
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self.eager(x, l, c)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            failed = False
            try:
                with torch.cuda.graph(graph):
                    out = self.eager(x, l, c)
            except torch.cuda.OutOfMemoryError:
                failed = True
            if failed:
                del graph, x, l, c
                self.clear()
                self.disabled_shapes.add(key)
                return self.eager(audio, lengths, channels)
            self.graph_build_seconds += time.perf_counter() - start
            if torch.cuda.memory_reserved() > self.limit:
                del graph, out, x, l, c
                self.clear()
                self.disabled_shapes.add(key)
                return self.eager(audio, lengths, channels)
            self.cache[key] = (graph, x, l, c, out)
        graph, x, l, c, out = self.cache[key]
        self.cache.move_to_end(key)
        x.copy_(audio)
        l.copy_(lengths)
        c.copy_(channels)
        graph.replay()
        self.graph_hits += 1
        # Return an owned result; a later replay may overwrite graph outputs.
        return out.clone()
