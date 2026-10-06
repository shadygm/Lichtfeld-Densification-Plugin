"""Cancellation, progress and cleanup for dense pipeline jobs."""
from __future__ import annotations

import logging
import time
from typing import Callable, List, Optional
import numpy as np
from ..debug_viz import MatchDebugState
from ..threaded_dataloader import ThreadedReferenceLoader
from .types import PipelineCancelled, _PackedReferenceBatch

logger = logging.getLogger(__name__)


def _is_cancelled(cancel_requested: Optional[Callable[[], bool]]) -> bool:
    if cancel_requested is None:
        return False
    try:
        return bool(cancel_requested())
    except Exception as exc:
        logger.warning(f"Cancellation callback failed: {exc}")
        return False


def _raise_if_cancelled(cancel_requested: Optional[Callable[[], bool]]) -> None:
    if _is_cancelled(cancel_requested):
        raise PipelineCancelled("Cancelled")


def _estimate_total_pairs(
    refs_local: List[int],
    nn_table: np.ndarray,
    img_ids: List[int],
    nns_per_ref: int,
) -> int:
    return sum(
        sum(1 for n in nn_table[ref_idx][:nns_per_ref] if img_ids[n] != img_ids[ref_idx])
        for ref_idx in refs_local
    )


def _report_matching_progress(
    progress_callback: Optional[Callable[[float, str], None]],
    refs_consumed: int,
    total_refs: int,
    start_time: float,
) -> None:
    if progress_callback is None:
        return
    pct = 10.0 + (float(refs_consumed) / max(1, total_refs)) * 80.0
    message = f"Matching {refs_consumed}/{total_refs}"
    if refs_consumed:
        elapsed = max(0.001, time.perf_counter() - start_time)
        message += f" | {refs_consumed / elapsed:.1f} it/s"
    progress_callback(pct, message)


def _report_model_setup_status(
    progress_callback: Optional[Callable[[float, str], None]],
    model_cached: bool,
) -> None:
    if model_cached:
        msg = "Initializing RoMa v2 model..."
    else:
        msg = "Installing model weights..."
    if progress_callback is not None:
        progress_callback(10.0, msg)
    logger.info(msg)
    if not model_cached:
        from ..matcher import romav2_cached_weights_paths

        cache_hints = ", ".join(romav2_cached_weights_paths())
        logger.info(f"RoMaV2 weights not found in cache; expected cache paths: {cache_hints}")


def _cleanup_pipeline_runtime(
    pack_loader: Optional[ThreadedReferenceLoader[Optional[_PackedReferenceBatch]]],
    matcher: Optional[RomaMatcher],
    debug_state: Optional[MatchDebugState],
) -> None:
    if pack_loader is not None:
        pack_loader.close(wait=True)
    if matcher is not None:
        try:
            matcher.close()
        except Exception as exc:
            logger.warning(f"Matcher cleanup failed: {exc}")
    if debug_state:
        debug_state.release_waiters()
