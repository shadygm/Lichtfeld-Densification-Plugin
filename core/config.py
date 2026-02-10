"""Shared configuration dataclasses."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DensePipelineConfig:
    output_path: str
    roma_setting: str = "fast"
    num_refs: float = 0.8
    nns_per_ref: int = 3
    matches_per_ref: int = 10000
    certainty_thresh: float = 0.20
    reproj_thresh: float = 1.0
    sampson_thresh: float = 5.0
    min_parallax_deg: float = 0.5
    max_points: int = 0
    no_filter: bool = False
    seed: int = 0
    viz_interval: int = 3
