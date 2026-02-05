"""Sampling strategies for certainty maps."""
from __future__ import annotations

import numpy as np
import torch


def select_samples_with_coverage(cert_map: torch.Tensor, M: int, cap: float = 0.9,
                                 border: int = 2, tiles: int = 24,
                                 no_filter: bool = False) -> np.ndarray:
    """Select sample indices from a certainty map using coverage-aware sampling."""
    cert = cert_map.clone().to("cpu")
    cert = torch.clamp(cert, max=cap)
    H, W = cert.shape
    if no_filter:
        flat = cert.reshape(-1).numpy()
        if flat.size == 0:
            return np.zeros((0,), dtype=np.int64)
        order = np.argsort(-flat)
        sel = order[:min(M, flat.size)]
        return sel

    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    inside = (xx >= border) & (xx <= W - 1 - border) & (yy >= border) & (yy <= H - 1 - border)
    weights = (cert * inside.float()).reshape(-1)
    s = weights.sum()
    if s <= 0:
        return np.zeros((0,), dtype=np.int64)
    weights = (weights / s).numpy()

    m_main = int(M * 0.85)
    idx_main = np.random.choice(weights.size, size=min(m_main, weights.size), replace=False, p=weights)

    tile = max(1, W // tiles)
    gx = (xx // tile).reshape(-1).numpy()
    gy = (yy // tile).reshape(-1).numpy()
    bins = gx * 100000 + gy
    order = np.argsort(-weights)
    seen = set()
    idx_cov = []
    for i in order:
        if weights[i] <= 0:
            break
        b = int(bins[i])
        if b in seen:
            continue
        seen.add(b)
        idx_cov.append(i)
        if len(idx_cov) >= M - len(idx_main):
            break

    sel_idx = np.unique(np.concatenate([idx_main, np.asarray(idx_cov, dtype=np.int64)]))
    return sel_idx
