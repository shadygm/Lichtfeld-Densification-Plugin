"""Collect dense matcher outputs and apply image masks."""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from .config import DensePipelineConfig
from .types import _MatchedReference, _PackedReferenceBatch
from .control import _raise_if_cancelled


def _mask_tensor_for_hw(
    mask_np: np.ndarray,
    target_hw: Tuple[int, int],
    cache: Optional[Dict[Tuple[int, int], torch.Tensor]] = None,
) -> torch.Tensor:
    """Return a float mask tensor resized to the requested HxW."""
    if cache is not None:
        cached = cache.get(target_hw)
        if cached is not None:
            return cached

    mask_t = torch.from_numpy(mask_np.astype(np.float32, copy=False))
    if mask_t.shape != target_hw:
        mask_t = F.interpolate(
            mask_t.view(1, 1, mask_t.shape[0], mask_t.shape[1]),
            size=target_hw,
            mode="nearest",
        ).squeeze(0).squeeze(0)

    if cache is not None:
        cache[target_hw] = mask_t
    return mask_t


def _collect_reference_matches(
    packed: _PackedReferenceBatch,
    matcher: RomaMatcher,
    config: DensePipelineConfig,
    pair_counter: int,
    cancel_requested: Optional[Callable[[], bool]],
) -> Tuple[Optional[_MatchedReference], int]:
    imA = Image.fromarray(np.ascontiguousarray(packed.imA_np))
    nn_images = [Image.fromarray(np.ascontiguousarray(arr)) for arr in packed.nn_arrays]

    warp_list_cpu: List[torch.Tensor] = []
    cert_list_cpu: List[torch.Tensor] = []
    pair_index_by_nbr: Dict[int, int] = {}
    image_by_nbr: Dict[int, np.ndarray] = {}

    batch_results = matcher.match_grids_batch(
        imA, nn_images, reference_key=packed.ref_id, neighbor_keys=packed.nn_ids
    )
    _raise_if_cancelled(cancel_requested)

    maskA_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    for (warp_hw, cert_hw), maskB_np, nbr_id, imB_np in zip(batch_results, packed.nn_masks, packed.nn_ids, packed.nn_arrays):
        _raise_if_cancelled(cancel_requested)
        if not config.no_filter:
            cert_hw = torch.where(cert_hw >= config.certainty_thresh, cert_hw, 0.0)

        output_hw = (int(warp_hw.shape[0]), int(warp_hw.shape[1]))
        cert_hw_shape = (int(cert_hw.shape[0]), int(cert_hw.shape[1]))
        if cert_hw_shape != output_hw:
            # Keep robust if certainty and warp shapes ever diverge.
            output_hw = cert_hw_shape

        if packed.maskA_np is not None:
            maskA_t = _mask_tensor_for_hw(packed.maskA_np, output_hw, cache=maskA_cache)
            cert_hw = cert_hw * maskA_t.to(device=cert_hw.device, dtype=cert_hw.dtype)

        if maskB_np is not None:
            maskB_t = _mask_tensor_for_hw(maskB_np, output_hw)
            maskB_t = maskB_t.to(device=cert_hw.device, dtype=cert_hw.dtype).view(1, 1, output_hw[0], output_hw[1])
            gridB = warp_hw[..., 2:4].unsqueeze(0)
            maskB_warp = F.grid_sample(
                maskB_t,
                gridB,
                mode="nearest",
                padding_mode="zeros",
                align_corners=False,
            )
            cert_hw = cert_hw * maskB_warp.squeeze(0).squeeze(0)

        warp_cpu = warp_hw.detach().to("cpu", non_blocking=True)
        cert_cpu = cert_hw.detach().to("cpu", non_blocking=True)
        pair_counter += 1
        pair_index_by_nbr[nbr_id] = pair_counter
        image_by_nbr[nbr_id] = imB_np

        warp_list_cpu.append(warp_cpu)
        cert_list_cpu.append(cert_cpu)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    del batch_results

    if not cert_list_cpu:
        return None, pair_counter

    return (
        _MatchedReference(
            packed=packed,
            warp_list_cpu=warp_list_cpu,
            cert_list_cpu=cert_list_cpu,
            pair_index_by_nbr=pair_index_by_nbr,
            image_by_nbr=image_by_nbr,
        ),
        pair_counter,
    )
