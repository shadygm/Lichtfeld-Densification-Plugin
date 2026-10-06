"""Dense pipeline configuration, result and execution API."""
from .config import DensePipelineConfig


def __getattr__(name):
    if name == "run_dense_pipeline":
        from .runner import run_dense_pipeline

        return run_dense_pipeline
    if name in {"PipelineCancelled", "PipelineResult"}:
        from . import types

        return getattr(types, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["DensePipelineConfig", "PipelineCancelled", "PipelineResult", "run_dense_pipeline"]
