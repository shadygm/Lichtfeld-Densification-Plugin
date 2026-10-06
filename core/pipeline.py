"""Orchestrate dense matching and overlapping triangulation."""
from __future__ import annotations

import logging
import time
import gc
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional
import numpy as np
import torch
from .camera_models import CameraRecord
from .config import DensePipelineConfig
from .geometry import camera_model_name
from .debug_viz import MatchDebugState
from .threaded_dataloader import ThreadedReferenceLoader
from .stages.types import (
    PipelineCancelled,
    PipelineResult,
    _PackContext,
    _PipelineAccumulator,
    _TriangulationContext,
)
from .stages.preparation import (
    _build_camera_lookup,
    _build_pack_loader,
)
from .stages.control import (
    _cleanup_pipeline_runtime,
    _estimate_total_pairs,
    _raise_if_cancelled,
    _report_matching_progress,
    _report_model_setup_status,
)
from .stages.matching import _collect_reference_matches
from .stages.preview import (
    _emit_debug_previews,
    _emit_intermediate_preview,
    _prepare_intermediate_ply_base,
)
from .stages.triangulation import _triangulate_ref

logger = logging.getLogger(__name__)


def run_dense_pipeline(
    camera_records: List[CameraRecord],
    refs_local: List[int],
    nn_table: np.ndarray,
    config: DensePipelineConfig,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    on_sequential_viz: Optional[Callable[[str], None]] = None,
    debug_state: Optional[MatchDebugState] = None,
    cancel_requested: Optional[Callable[[], bool]] = None,
) -> PipelineResult:
    np.random.seed(config.seed)

    cameras = _build_camera_lookup(camera_records)
    distortion_models = sorted(
        {
            camera_model_name(cameras.by_id[uid].colmap_camera)
            for uid in cameras.distorted_ids
        }
    )
    if distortion_models:
        msg = "Distortion-aware COLMAP projection enabled: " + ", ".join(distortion_models)
        logger.info(msg)
        if progress_callback is not None:
            progress_callback(3.0, msg)
        if config.sampson_thresh > 0:
            logger.info(
                "Sampson pre-filter is skipped for distortion-aware camera pairs; "
                "reprojection, cheirality, and parallax filters remain active."
            )
            if progress_callback is not None:
                progress_callback(4.0, "Sampson filter skipped for distorted/fisheye camera pairs")
    total_pairs_est = _estimate_total_pairs(refs_local, nn_table, cameras.img_ids, config.nns_per_ref)
    if debug_state:
        debug_state.set_total_pairs(total_pairs_est)

    viz_interval = config.viz_interval
    intermediate_ply_base = _prepare_intermediate_ply_base(config.output_path, viz_interval, on_sequential_viz)

    points = _PipelineAccumulator()
    t0 = time.time()
    matcher: Optional[RomaMatcher] = None
    pack_loader: Optional[ThreadedReferenceLoader[Optional[_PackedReferenceBatch]]] = None
    triangulator = None
    pack_ctx = None

    try:
        from .matcher import RomaMatcher, has_cached_romav2_weights

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_cached = has_cached_romav2_weights()
        _report_model_setup_status(progress_callback, model_cached)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # Fast runs spend more time searching kernels than the search saves.
            torch.backends.cudnn.benchmark = config.roma_setting != "fast"
        matcher = RomaMatcher(device=device, mode="outdoor", setting=config.roma_setting)
        if not model_cached and progress_callback is not None:
            progress_callback(10.0, "RoMa v2 model installation complete. Starting matching...")
        _raise_if_cancelled(cancel_requested)

        total_refs = len(refs_local)
        w_match, h_match = matcher.w_resized, matcher.h_resized
        pack_ctx = _PackContext(
            cameras=cameras,
            nn_table=nn_table,
            nns_per_ref=config.nns_per_ref,
            w_match=w_match,
            h_match=h_match,
        )
        tri_ctx = _TriangulationContext(
            cameras=cameras,
            config=config,
            matcher_sample_cap=matcher.sample_thresh,
            w_match=w_match,
            h_match=h_match,
        )

        pack_loader = _build_pack_loader(refs_local, pack_ctx, config, cancel_requested)
        pack_loader_iter = iter(pack_loader)

        # One CPU task overlaps the next GPU match; consume results in order.
        triangulator = ThreadPoolExecutor(max_workers=1, thread_name_prefix="triangulation")
        pending = None

        def finish_pending():
            nonlocal pending
            if pending is None:
                return
            future, matched, collect_debug, pair_counter = pending
            pending = None
            try:
                tri_ref = future.result()
            except Exception as ex:
                logger.error(f"Triangulation error for ref {matched.packed.ref_id}: {ex}")
                return
            _raise_if_cancelled(cancel_requested)
            if tri_ref is None:
                return
            points.append(tri_ref)
            if collect_debug:
                _emit_debug_previews(
                    matched_ref=matched, tri_ref=tri_ref, debug_state=debug_state,
                    cameras=cameras, total_pairs_est=total_pairs_est,
                    pair_counter=pair_counter, cancel_requested=cancel_requested,
                )
            _emit_intermediate_preview(
                on_sequential_viz=on_sequential_viz,
                intermediate_ply_base=intermediate_ply_base,
                viz_interval=viz_interval, points=points,
                cancel_requested=cancel_requested,
            )

        matching_started = time.perf_counter()
        refs_consumed = 0
        _report_matching_progress(progress_callback, 0, total_refs, matching_started)
        while True:
            _raise_if_cancelled(cancel_requested)
            try:
                packed = next(pack_loader_iter)
            except StopIteration:
                break

            refs_consumed += 1

            if packed is None:
                continue
            _raise_if_cancelled(cancel_requested)

            matched_ref, points.pair_counter = _collect_reference_matches(
                packed=packed,
                matcher=matcher,
                config=config,
                pair_counter=points.pair_counter,
                cancel_requested=cancel_requested,
            )
            _report_matching_progress(progress_callback, refs_consumed, total_refs, matching_started)
            if matched_ref is None:
                continue

            finish_pending()
            collect_debug_matches = debug_state is not None and debug_state.is_enabled()
            pending = (
                triangulator.submit(
                    _triangulate_ref, matched_ref, tri_ctx,
                    collect_debug_matches=collect_debug_matches,
                ),
                matched_ref, collect_debug_matches, points.pair_counter,
            )

        finish_pending()

    finally:
        if triangulator is not None:
            triangulator.shutdown(wait=True, cancel_futures=True)
        _cleanup_pipeline_runtime(pack_loader, matcher, debug_state)
        if pack_ctx is not None:
            pack_ctx.load_image.cache_clear()
        pack_loader = None
        matcher = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    _raise_if_cancelled(cancel_requested)

    if progress_callback:
        progress_callback(90.0, "Finalizing triangulation...")

    if not points.xyz_parts:
        raise RuntimeError("No points triangulated. Try adjusting parameters.")

    xyz = np.concatenate(points.xyz_parts, axis=0)
    rgb = np.concatenate(points.rgb_parts, axis=0)
    err = np.concatenate(points.err_parts, axis=0)
    tracks = [track for track_part in points.track_parts for track in track_part]
    elapsed = time.time() - t0

    return PipelineResult(
        xyz=xyz,
        rgb=rgb,
        err=err,
        tracks=tracks,
        elapsed_seconds=elapsed,
        pairs_processed=points.pairs_processed,
    )
