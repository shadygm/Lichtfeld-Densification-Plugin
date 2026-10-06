"""Prepare and prefetch images for selected reference cameras."""
from __future__ import annotations

import logging
from typing import Callable, List, Optional
import numpy as np
from ..camera_models import CameraRecord
from ..config import DensePipelineConfig
from ..geometry import uses_distortion_aware_projection
from ..threaded_dataloader import ThreadedReferenceLoader
from .types import _CameraLookup, _PackContext, _PackedReferenceBatch
from .control import _is_cancelled

logger = logging.getLogger(__name__)


def _pack_reference_batch(
    ref_local: int,
    pack_ctx: _PackContext,
    cancel_requested: Optional[Callable[[], bool]] = None,
) -> Optional[_PackedReferenceBatch]:
    """Load and preprocess a single reference package for compute."""

    if _is_cancelled(cancel_requested):
        return None

    img_ids = pack_ctx.cameras.img_ids
    nn_table = pack_ctx.nn_table
    nns_per_ref = pack_ctx.nns_per_ref
    ref_id = img_ids[ref_local]
    ref_camera = pack_ctx.cameras.by_id[ref_id]
    ref_path = ref_camera.image_path
    try:
        imA_np, maskA_np = pack_ctx.load_image(ref_id)
    except Exception as exc:
        logger.warning(f"Failed to load reference {ref_path}: {exc}")
        return None
    if _is_cancelled(cancel_requested):
        return None

    local_nns = nn_table[ref_local][:nns_per_ref]
    if len(local_nns) == 0:
        return None

    nn_ids: List[int] = []
    nn_masks: List[Optional[np.ndarray]] = []
    nn_arrays: List[np.ndarray] = []

    for nn_local in local_nns:
        if _is_cancelled(cancel_requested):
            return None

        nbr_id = img_ids[nn_local]
        if nbr_id == ref_id:
            continue
        try:
            imB_np, maskB_np = pack_ctx.load_image(nbr_id)
            if _is_cancelled(cancel_requested):
                return None

            nn_ids.append(nbr_id)
            nn_masks.append(maskB_np)
            nn_arrays.append(imB_np)
        except Exception as exc:
            logger.warning(f"Failed to load neighbor {nbr_id}: {exc}")

    if not nn_arrays:
        return None

    wA_cam, hA_cam = ref_camera.width, ref_camera.height
    return _PackedReferenceBatch(
        ref_id=ref_id,
        ref_path=ref_path,
        imA_np=imA_np,
        maskA_np=maskA_np,
        wA_cam=wA_cam,
        hA_cam=hA_cam,
        nn_ids=nn_ids,
        nn_masks=nn_masks,
        nn_arrays=nn_arrays,
    )


class _PackedReferenceDataset:
    """Indexable dataset that packs reference batches for compute."""

    def __init__(
        self,
        refs_local: List[int],
        pack_ctx: _PackContext,
        cancel_requested: Optional[Callable[[], bool]] = None,
    ) -> None:
        self._refs_local = refs_local
        self._pack_ctx = pack_ctx
        self._cancel_requested = cancel_requested

    def __len__(self) -> int:
        return len(self._refs_local)

    def __getitem__(self, index: int) -> Optional[_PackedReferenceBatch]:
        ref_local = self._refs_local[index]
        return _pack_reference_batch(
            ref_local,
            self._pack_ctx,
            self._cancel_requested,
        )


def _build_camera_lookup(camera_records: List[CameraRecord]) -> _CameraLookup:
    return _CameraLookup(
        img_ids=[cam.uid for cam in camera_records],
        by_id={cam.uid: cam for cam in camera_records},
        distorted_ids={cam.uid for cam in camera_records if uses_distortion_aware_projection(cam.colmap_camera)},
    )


def _build_pack_loader(
    refs_local: List[int],
    pack_ctx: _PackContext,
    config: DensePipelineConfig,
    cancel_requested: Optional[Callable[[], bool]],
) -> ThreadedReferenceLoader[Optional[_PackedReferenceBatch]]:
    prefetch_packages = max(1, int(config.prefetch_packages))
    pack_workers = max(1, int(config.pack_workers))
    dataset = _PackedReferenceDataset(
        refs_local=refs_local,
        pack_ctx=pack_ctx,
        cancel_requested=cancel_requested,
    )
    return ThreadedReferenceLoader(
        dataset=dataset,
        num_workers=pack_workers,
        prefetch_size=prefetch_packages,
        cancel_requested=cancel_requested,
    )
