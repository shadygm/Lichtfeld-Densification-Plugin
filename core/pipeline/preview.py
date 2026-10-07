"""Build debug matches and publish intermediate point clouds."""
from __future__ import annotations

import logging
import os
from typing import Callable, Optional
import numpy as np
from ..images.io import to_uint8_rgb
from ..previews.matches import MatchPreview, MatchDebugState
from ..reconstruction.writers import write_ply
from ..reconstruction.cloud import DenseCloud
from .types import (
    _CameraLookup,
    _MatchedReference,
    _PipelineAccumulator,
    _TriangulatedReference,
)
from .control import _raise_if_cancelled

logger = logging.getLogger(__name__)

_DEBUG_PREVIEW_INTERVAL = 3
_PREVIEW_MAX_MATCHES = 10000


def _emit_debug_previews(
    matched_ref: _MatchedReference,
    tri_ref: _TriangulatedReference,
    debug_state: Optional[MatchDebugState],
    cameras: _CameraLookup,
    total_pairs_est: int,
    pair_counter: int,
    cancel_requested: Optional[Callable[[], bool]],
) -> None:
    if debug_state is None:
        return

    packed = matched_ref.packed
    total_pairs_val = total_pairs_est if total_pairs_est > 0 else max(pair_counter, 1)
    for nbr_id, matches in tri_ref.debug_matches_by_nbr.items():
        _raise_if_cancelled(cancel_requested)
        imB_np = matched_ref.image_by_nbr.get(nbr_id)
        pair_idx = matched_ref.pair_index_by_nbr.get(nbr_id)
        cert_norm = tri_ref.debug_cert_by_nbr.get(nbr_id)
        if imB_np is None or pair_idx is None or cert_norm is None:
            continue
        should_preview = (
            not debug_state.is_auto_step()
            or _DEBUG_PREVIEW_INTERVAL <= 0
            or pair_idx % _DEBUG_PREVIEW_INTERVAL == 1
        )
        if not should_preview:
            continue
        try:
            preview = _build_filtered_match_preview(
                packed.imA_np,
                imB_np,
                matches,
                cert_norm,
                packed.ref_id,
                nbr_id,
                os.path.basename(packed.ref_path),
                os.path.basename(cameras.by_id[nbr_id].image_path),
                pair_idx,
                total_pairs_val,
                match_count=int(matches.shape[0]),
            )
            if preview:
                debug_state.submit_preview(preview)
        except Exception as exc:
            logger.warning(f"Debug preview failed: {exc}")


def _emit_intermediate_preview(
    on_sequential_viz: Optional[Callable[[str], None]],
    intermediate_ply_base: Optional[str],
    viz_interval: int,
    points: _PipelineAccumulator,
    cancel_requested: Optional[Callable[[], bool]],
    on_cloud_preview: Optional[Callable[[DenseCloud], None]] = None,
) -> None:
    if (
        not (on_sequential_viz or on_cloud_preview)
        or viz_interval <= 0
        or (on_cloud_preview is None and not intermediate_ply_base)
        or points.pairs_processed % viz_interval != 0
    ):
        return

    _raise_if_cancelled(cancel_requested)
    try:
        xyz_so_far = np.concatenate(points.xyz_parts, axis=0)
        rgb_so_far = np.concatenate(points.rgb_parts, axis=0)
        colors = to_uint8_rgb(rgb_so_far)
        if on_cloud_preview is not None:
            on_cloud_preview(DenseCloud(xyz_so_far.astype(np.float32, copy=False), colors))
        else:
            intermediate_ply_path = f"{intermediate_ply_base}_{points.pairs_processed}.ply"
            write_ply(intermediate_ply_path, xyz_so_far, colors)
            on_sequential_viz(intermediate_ply_path)
        logger.debug(f"Live update: {xyz_so_far.shape[0]:,} points after {points.pairs_processed} refs")
    except Exception as exc:
        logger.warning(f"Failed to emit intermediate cloud: {exc}")


def _build_filtered_match_preview(
    imA_np: np.ndarray,
    imB_np: np.ndarray,
    matches: np.ndarray,
    cert_norm: np.ndarray,
    ref_id: int,
    nbr_id: int,
    ref_label: str,
    nbr_label: str,
    pair_index: int,
    total_pairs: int,
    match_count: int,
    max_matches: int = _PREVIEW_MAX_MATCHES,
) -> Optional[MatchPreview]:
    """Build a debug preview from correspondences that survived filtering.

    The ``matches`` input must be in pixel coordinates of the resized match
    images and must represent correspondences that are used for triangulation.
    """

    if matches is None or cert_norm is None:
        return None
    if matches.size == 0 or cert_norm.size == 0:
        return None

    matches = np.asarray(matches, dtype=np.float32)
    cert_norm = np.asarray(cert_norm, dtype=np.float32)
    total = int(match_count if match_count > 0 else matches.shape[0])

    if matches.shape[0] > max_matches > 0:
        seed = ((int(ref_id) & 0xFFFF_FFFF) * 73856093) ^ ((int(nbr_id) & 0xFFFF_FFFF) * 19349663)
        rng = np.random.default_rng(seed & 0xFFFF_FFFF)
        sel_idx = rng.choice(matches.shape[0], size=max_matches, replace=False)
        matches = matches[sel_idx]
        cert_norm = cert_norm[sel_idx]

    return MatchPreview(
        ref_id=ref_id,
        nbr_id=nbr_id,
        ref_label=ref_label,
        nbr_label=nbr_label,
        left_image=imA_np,
        right_image=imB_np,
        matches=matches,
        cert_norm=cert_norm.astype(np.float32, copy=False),
        match_count=total,
        pair_index=int(pair_index),
        total_pairs=int(total_pairs),
    )
