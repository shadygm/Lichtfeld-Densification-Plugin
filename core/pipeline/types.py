"""Shared data exchanged between pipeline stages."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import torch
from ..cameras.models import CameraRecord
from .config import DensePipelineConfig
from ..images.io import apply_mask_to_rgb, load_mask_resized_np, load_rgb_resized

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    xyz: np.ndarray
    rgb: np.ndarray
    err: np.ndarray
    tracks: List[List[Tuple[int, float, float]]]
    elapsed_seconds: float
    pairs_processed: int


class PipelineCancelled(RuntimeError):
    """Raised when a running dense pipeline is cancelled."""


@dataclass
class _PackedReferenceBatch:
    """Single reference package produced by threaded pack workers."""

    ref_id: int
    ref_path: str
    imA_np: np.ndarray
    maskA_np: Optional[np.ndarray]
    wA_cam: int
    hA_cam: int
    nn_ids: List[int]
    nn_masks: List[Optional[np.ndarray]]
    nn_arrays: List[np.ndarray]


@dataclass(frozen=True)
class _CameraLookup:
    img_ids: List[int]
    by_id: Dict[int, CameraRecord]
    distorted_ids: set[int]


@dataclass(frozen=True)
class _PackContext:
    cameras: _CameraLookup
    nn_table: np.ndarray
    nns_per_ref: int
    w_match: int
    h_match: int
    load_image: Callable = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        cameras, size = self.cameras.by_id, (self.w_match, self.h_match)

        @lru_cache(maxsize=256)
        def load_image(uid):
            camera = cameras[uid]
            image = load_rgb_resized(camera.image_path, size)
            mask = None
            if camera.mask_path:
                try:
                    mask = load_mask_resized_np(camera.mask_path, size)
                    image = apply_mask_to_rgb(image, mask)
                except Exception as exc:
                    logger.warning(f"Failed to load/apply mask for image {uid}: {exc}")
                    mask = None
            return np.asarray(image, dtype=np.uint8), mask

        object.__setattr__(self, "load_image", load_image)


@dataclass
class _MatchedReference:
    packed: _PackedReferenceBatch
    warp_list_cpu: List[torch.Tensor]
    cert_list_cpu: List[torch.Tensor]
    pair_index_by_nbr: Dict[int, int]
    image_by_nbr: Dict[int, np.ndarray]


@dataclass(frozen=True)
class _TriangulationContext:
    cameras: _CameraLookup
    config: DensePipelineConfig
    matcher_sample_cap: float
    w_match: int
    h_match: int


@dataclass
class _TriangulatedReference:
    xyz: np.ndarray
    rgb: np.ndarray
    err: np.ndarray
    tracks: List[List[Tuple[int, float, float]]]
    debug_matches_by_nbr: Dict[int, np.ndarray]
    debug_cert_by_nbr: Dict[int, np.ndarray]


@dataclass
class _PipelineAccumulator:
    xyz_parts: List[np.ndarray] = field(default_factory=list)
    rgb_parts: List[np.ndarray] = field(default_factory=list)
    err_parts: List[np.ndarray] = field(default_factory=list)
    track_parts: List[List[List[Tuple[int, float, float]]]] = field(default_factory=list)
    pairs_processed: int = 0
    pair_counter: int = 0

    def append(self, tri_ref: _TriangulatedReference) -> None:
        self.xyz_parts.append(tri_ref.xyz)
        self.rgb_parts.append(tri_ref.rgb)
        self.err_parts.append(tri_ref.err)
        self.track_parts.append(tri_ref.tracks)
        self.pairs_processed += 1
