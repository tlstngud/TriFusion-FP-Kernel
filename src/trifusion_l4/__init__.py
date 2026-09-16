"""TriFusion-FP inference library for NVIDIA L4.

CUDA dependencies are loaded only when a GPU API is accessed. Metadata and
CSV tools can be used without installing the optional ``l4`` dependencies.
"""

from importlib import import_module

__version__ = "0.1.0"
__all__ = ["__version__", "configure_runtime", "load_student", "InferenceEngine", "fused_mamba"]

_EXPORTS = {
    "configure_runtime": (".runtime", "configure_runtime"),
    "load_student": (".runtime", "load_student"),
    "InferenceEngine": (".runtime", "InferenceEngine"),
    "fused_mamba": (".mamba_fused", "fused_mamba"),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    try:
        value = getattr(import_module(module_name, __name__), attribute)
    except ModuleNotFoundError as error:
        if error.name in {"torch", "triton", "einops", "numpy", "scipy", "soundfile"}:
            raise ModuleNotFoundError(
                "The L4 runtime requires the optional GPU dependencies. Install the "
                "documented PyTorch CUDA 12.8 wheel, then trifusion-fp-kernel[l4].",
                name=error.name,
            ) from error
        raise
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
