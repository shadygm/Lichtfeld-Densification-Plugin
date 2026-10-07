"""Scope matcher CUDA backend tuning to one pipeline run."""
from contextlib import contextmanager

import torch


@contextmanager
def matcher_backend_settings(setting: str):
    if not torch.cuda.is_available():
        yield
        return
    matmul = torch.backends.cuda.matmul
    cudnn = torch.backends.cudnn
    previous = matmul.allow_tf32, cudnn.allow_tf32, cudnn.benchmark
    try:
        matmul.allow_tf32 = True
        cudnn.allow_tf32 = True
        # Kernel search costs more than it saves for fast-mode inference.
        cudnn.benchmark = setting != "fast"
        yield
    finally:
        matmul.allow_tf32, cudnn.allow_tf32, cudnn.benchmark = previous
