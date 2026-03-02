"""Shared dense pipeline orchestration."""
from __future__ import annotations

import os
import time
import gc
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import lichtfeld as lf
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .camera_models import CameraRecord
from .config import DensePipelineConfig
from .geometry import (
    cheirality_mask,
    dlt_triangulate_batch,
    fundamental_from_world2cam,
    parallax_mask,
    reprojection_errors,
    sampson_error,
)
from .image_utils import apply_mask_to_rgb, load_mask_resized_np, load_rgb_resized, to_uint8_rgb
from .debug_viz import MatchPreview, MatchDebugState
from .matcher import RomaMatcher, has_cached_romav2_weights, romav2_cached_weights_paths
from .sampling import select_samples_with_coverage
from .threaded_dataloader import ThreadedReferenceLoader
from .writers import write_ply

# How often to generate a debug preview (every N pairs). 0 = every pair.
_DEBUG_PREVIEW_INTERVAL = 3


@dataclass
class PipelineResult:
    xyz: np.ndarray
    rgb: np.ndarray
    err: np.ndarray
    elapsed_seconds: float
    pairs_processed: int


class PipelineCancelled(RuntimeError):
    """Raised when a running dense pipeline is cancelled."""


_PREVIEW_MAX_MATCHES = 10000


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
    path_by: Dict[int, str]
    mask_by: Dict[int, Optional[str]]
    size_by: Dict[int, Tuple[int, int]]
    K_by: Dict[int, np.ndarray]
    R_by: Dict[int, np.ndarray]
    t_by: Dict[int, np.ndarray]
    P_by: Dict[int, np.ndarray]
    C_by: Dict[int, np.ndarray]


@dataclass(frozen=True)
class _PackContext:
    cameras: _CameraLookup
    nn_table: np.ndarray
    nns_per_ref: int
    w_match: int
    h_match: int


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
    debug_matches_by_nbr: Dict[int, np.ndarray]
    debug_cert_by_nbr: Dict[int, np.ndarray]


@dataclass
class _PipelineAccumulator:
    xyz_parts: List[np.ndarray] = field(default_factory=list)
    rgb_parts: List[np.ndarray] = field(default_factory=list)
    err_parts: List[np.ndarray] = field(default_factory=list)
    pairs_processed: int = 0
    pair_counter: int = 0

    def append(self, tri_ref: _TriangulatedReference) -> None:
        self.xyz_parts.append(tri_ref.xyz)
        self.rgb_parts.append(tri_ref.rgb)
        self.err_parts.append(tri_ref.err)
        self.pairs_processed += 1


def _pack_reference_batch(
    ref_local: int,
    pack_ctx: _PackContext,
    cancel_requested: Optional[Callable[[], bool]] = None,
) -> Optional[_PackedReferenceBatch]:
    """Load and preprocess a single reference package for compute."""

    if _is_cancelled(cancel_requested):
        return None

    img_ids = pack_ctx.cameras.img_ids
    path_by = pack_ctx.cameras.path_by
    mask_by = pack_ctx.cameras.mask_by
    size_by = pack_ctx.cameras.size_by
    nn_table = pack_ctx.nn_table
    nns_per_ref = pack_ctx.nns_per_ref
    w_match = pack_ctx.w_match
    h_match = pack_ctx.h_match

    ref_id = img_ids[ref_local]
    ref_path = path_by[ref_id]
    try:
        imA = load_rgb_resized(ref_path, (w_match, h_match))
    except Exception as exc:
        lf.log.warn(f"Failed to load reference {ref_path}: {exc}")
        return None
    if _is_cancelled(cancel_requested):
        return None

    maskA_np = None
    ref_mask_path = mask_by.get(ref_id)
    if ref_mask_path:
        if _is_cancelled(cancel_requested):
            return None
        try:
            maskA_np = load_mask_resized_np(ref_mask_path, (w_match, h_match))
            imA = apply_mask_to_rgb(imA, maskA_np)
        except Exception as exc:
            lf.log.warn(f"Failed to load/apply mask for reference {ref_id}: {exc}")
            maskA_np = None

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
            img_path = path_by[nbr_id]
            imB = load_rgb_resized(img_path, (w_match, h_match))
            if _is_cancelled(cancel_requested):
                return None

            maskB_np = None
            nbr_mask_path = mask_by.get(nbr_id)
            if nbr_mask_path:
                if _is_cancelled(cancel_requested):
                    return None
                try:
                    maskB_np = load_mask_resized_np(nbr_mask_path, (w_match, h_match))
                    imB = apply_mask_to_rgb(imB, maskB_np)
                except Exception as exc:
                    lf.log.warn(f"Failed to load/apply mask for neighbor {nbr_id}: {exc}")
                    maskB_np = None

            nn_ids.append(nbr_id)
            nn_masks.append(maskB_np)
            nn_arrays.append(np.asarray(imB, dtype=np.uint8))
        except Exception as exc:
            lf.log.warn(f"Failed to load neighbor {nbr_id}: {exc}")

    if not nn_arrays:
        return None

    imA_np = np.asarray(imA, dtype=np.uint8)
    wA_cam, hA_cam = size_by[ref_id]
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


def _is_cancelled(cancel_requested: Optional[Callable[[], bool]]) -> bool:
    if cancel_requested is None:
        return False
    try:
        return bool(cancel_requested())
    except Exception as exc:
        lf.log.warn(f"Cancellation callback failed: {exc}")
        return False


def _raise_if_cancelled(cancel_requested: Optional[Callable[[], bool]]) -> None:
    if _is_cancelled(cancel_requested):
        raise PipelineCancelled("Cancelled")


def _build_camera_lookup(camera_records: List[CameraRecord]) -> _CameraLookup:
    return _CameraLookup(
        img_ids=[cam.uid for cam in camera_records],
        path_by={cam.uid: cam.image_path for cam in camera_records},
        mask_by={cam.uid: getattr(cam, "mask_path", None) for cam in camera_records},
        size_by={cam.uid: (cam.width, cam.height) for cam in camera_records},
        K_by={cam.uid: cam.K for cam in camera_records},
        R_by={cam.uid: cam.R for cam in camera_records},
        t_by={cam.uid: cam.t for cam in camera_records},
        P_by={cam.uid: cam.P for cam in camera_records},
        C_by={cam.uid: cam.C for cam in camera_records},
    )


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


def _prepare_intermediate_ply_base(
    output_path: str,
    viz_interval: int,
    on_sequential_viz: Optional[Callable[[str], None]],
) -> Optional[str]:
    if not on_sequential_viz or viz_interval <= 0:
        return None
    output_dir = os.path.dirname(output_path)
    base_no_ext = os.path.splitext(os.path.basename(output_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f"{base_no_ext}_intermediate")


def _report_matching_progress(
    progress_callback: Optional[Callable[[float, str], None]],
    refs_consumed: int,
    total_refs: int,
    start_time: float,
) -> None:
    if progress_callback is None:
        return
    pct = 10.0 + (float(refs_consumed - 1) / max(1, total_refs)) * 80.0
    progress_callback(
        pct,
        f"matching {refs_consumed}/{total_refs} | {refs_consumed / max(0.001, time.time() - start_time):.1f} it/s",
    )


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
    lf.log.info(msg)
    if not model_cached:
        cache_hints = ", ".join(romav2_cached_weights_paths())
        lf.log.info(f"RoMaV2 weights not found in cache; expected cache paths: {cache_hints}")


def _build_pack_loader(
    refs_local: List[int],
    pack_ctx: _PackContext,
    config: DensePipelineConfig,
    cancel_requested: Optional[Callable[[], bool]],
) -> ThreadedReferenceLoader[Optional[_PackedReferenceBatch]]:
    prefetch_packages = max(1, int(getattr(config, "prefetch_packages", 8)))
    pack_workers = max(1, int(getattr(config, "pack_workers", 4)))
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

    batch_results = matcher.match_grids_batch(imA, nn_images)
    _raise_if_cancelled(cancel_requested)

    maskA_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    for (warp_hw, cert_hw), maskB_np, nbr_id, imB_np in zip(batch_results, packed.nn_masks, packed.nn_ids, packed.nn_arrays):
        _raise_if_cancelled(cancel_requested)
        cert_hw = torch.clamp(cert_hw, min=config.certainty_thresh)

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
                os.path.basename(cameras.path_by[nbr_id]),
                pair_idx,
                total_pairs_val,
                match_count=int(matches.shape[0]),
            )
            if preview:
                debug_state.submit_preview(preview)
        except Exception as exc:
            lf.log.warn(f"Debug preview failed: {exc}")


def _emit_intermediate_preview(
    on_sequential_viz: Optional[Callable[[str], None]],
    intermediate_ply_base: Optional[str],
    viz_interval: int,
    points: _PipelineAccumulator,
    cancel_requested: Optional[Callable[[], bool]],
) -> None:
    if (
        not on_sequential_viz
        or viz_interval <= 0
        or not intermediate_ply_base
        or points.pairs_processed % viz_interval != 0
    ):
        return

    _raise_if_cancelled(cancel_requested)
    try:
        xyz_so_far = np.concatenate(points.xyz_parts, axis=0)
        rgb_so_far = np.concatenate(points.rgb_parts, axis=0)
        intermediate_ply_path = f"{intermediate_ply_base}_{points.pairs_processed}.ply"
        write_ply(intermediate_ply_path, xyz_so_far, to_uint8_rgb(rgb_so_far))
        lf.log.debug(f"Live update: {xyz_so_far.shape[0]:,} points after {points.pairs_processed} refs")
        on_sequential_viz(intermediate_ply_path)
    except Exception as exc:
        lf.log.warn(f"Failed to emit intermediate PLY: {exc}")


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
            lf.log.warn(f"Matcher cleanup failed: {exc}")
    if debug_state:
        debug_state.release_waiters()


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


def _triangulate_ref(
    matched_ref: _MatchedReference,
    tri_ctx: _TriangulationContext,
    collect_debug_matches: bool = False,
) -> Optional[_TriangulatedReference]:
    """Triangulate matches for a single reference view."""

    packed = matched_ref.packed
    ref_id = packed.ref_id
    nn_ids = packed.nn_ids
    imA_np = packed.imA_np
    wA_cam = packed.wA_cam
    hA_cam = packed.hA_cam

    w_match = tri_ctx.w_match
    h_match = tri_ctx.h_match
    config = tri_ctx.config
    matcher_sample_cap = tri_ctx.matcher_sample_cap

    cameras = tri_ctx.cameras
    size_by = cameras.size_by
    K_by = cameras.K_by
    R_by = cameras.R_by
    t_by = cameras.t_by
    P_by = cameras.P_by
    C_by = cameras.C_by

    warp_list = matched_ref.warp_list_cpu
    cert_list = matched_ref.cert_list_cpu

    H, W = cert_list[0].shape
    device_cpu = torch.device("cpu")
    cert_stack = torch.stack(cert_list, dim=0).to(device_cpu)
    best_cert, best_k = torch.max(cert_stack, dim=0)
    warp_stack = torch.stack(warp_list, dim=0).to(device_cpu)

    ys = torch.arange(H, device=device_cpu).unsqueeze(1).expand(H, W)
    xs = torch.arange(W, device=device_cpu).unsqueeze(0).expand(H, W)
    agg = warp_stack[best_k, ys, xs].reshape(-1, 4).numpy()

    sel_idx = select_samples_with_coverage(
        best_cert,
        config.matches_per_ref,
        cap=matcher_sample_cap,
        border=2,
        tiles=24,
        no_filter=config.no_filter,
    )
    if sel_idx.size == 0:
        return None

    nn_idx_flat = best_k.reshape(-1).numpy()[sel_idx]
    sel = agg[sel_idx]
    xA = (sel[:, 0] + 1.0) * 0.5 * (w_match - 1)
    yA = (sel[:, 1] + 1.0) * 0.5 * (h_match - 1)
    xB_norm = sel[:, 2]
    yB_norm = sel[:, 3]
    cert_sel = best_cert.reshape(-1).numpy()[sel_idx]

    hA_img, wA_img = imA_np.shape[0], imA_np.shape[1]
    sxA_img = wA_img / float(w_match)
    syA_img = hA_img / float(h_match)
    xA_img = xA * sxA_img
    yA_img = yA * syA_img

    xa0 = np.clip(np.floor(xA_img).astype(np.int32), 0, wA_img - 1)
    ya0 = np.clip(np.floor(yA_img).astype(np.int32), 0, hA_img - 1)
    xa1 = np.clip(xa0 + 1, 0, wA_img - 1)
    ya1 = np.clip(ya0 + 1, 0, hA_img - 1)
    wa = (xa1 - xA_img) * (ya1 - yA_img)
    wb = (xA_img - xa0) * (ya1 - yA_img)
    wc = (xa1 - xA_img) * (yA_img - ya0)
    wd = (xA_img - xa0) * (yA_img - ya0)
    Ia = imA_np[ya0, xa0].astype(np.float32)
    Ib = imA_np[ya0, xa1].astype(np.float32)
    Ic = imA_np[ya1, xa0].astype(np.float32)
    Id = imA_np[ya1, xa1].astype(np.float32)
    rgb_ref = (Ia * wa[:, None] + Ib * wb[:, None] + Ic * wc[:, None] + Id * wd[:, None]) / 255.0

    sxA = wA_cam / float(w_match)
    syA = hA_cam / float(h_match)
    uvA_full = np.stack([xA * sxA, yA * syA], axis=1)

    groups: Dict[int, List[int]] = {}
    for i, kidx in enumerate(nn_idx_flat):
        nbr_id = nn_ids[int(kidx)]
        groups.setdefault(nbr_id, []).append(i)

    job_xyz, job_rgb, job_err = [], [], []
    debug_matches_by_nbr: Dict[int, np.ndarray] = {}
    debug_cert_by_nbr: Dict[int, np.ndarray] = {}
    cert_denom = float(matcher_sample_cap) if float(matcher_sample_cap) > 1e-6 else 1.0

    for nbr_id, idxs in groups.items():
        idxs = np.asarray(idxs, dtype=np.int64)
        wB_cam, hB_cam = size_by[nbr_id]
        sxB = wB_cam / float(w_match)
        syB = hB_cam / float(h_match)

        xB = (xB_norm[idxs] + 1.0) * 0.5 * (w_match - 1)
        yB = (yB_norm[idxs] + 1.0) * 0.5 * (h_match - 1)
        uvB = np.stack([xB * sxB, yB * syB], axis=1)
        xA_pair = xA[idxs]
        yA_pair = yA[idxs]
        cert_pair = cert_sel[idxs]

        if (not config.no_filter) and config.sampson_thresh > 0:
            F = fundamental_from_world2cam(
                K_by[ref_id],
                R_by[ref_id],
                t_by[ref_id],
                K_by[nbr_id],
                R_by[nbr_id],
                t_by[nbr_id],
            )
            se = sampson_error(F, uvA_full[idxs], uvB)
            good = se < float(config.sampson_thresh)
            if not np.any(good):
                continue
            idxs = idxs[good]
            xB = xB[good]
            yB = yB[good]
            uvB = uvB[good]
            xA_pair = xA_pair[good]
            yA_pair = yA_pair[good]
            cert_pair = cert_pair[good]
        if idxs.size == 0:
            continue

        P1, P2 = P_by[ref_id], P_by[nbr_id]
        uvA = uvA_full[idxs]
        Xi = dlt_triangulate_batch(P1, P2, uvA, uvB)

        err1 = reprojection_errors(P1, Xi, uvA)
        err2 = reprojection_errors(P2, Xi, uvB)
        err = np.maximum(err1, err2)

        if config.no_filter:
            finite_mask = np.isfinite(Xi).all(axis=1) & np.isfinite(err)
            if not np.any(finite_mask):
                continue
            keep = finite_mask
        else:
            keep = err <= float(config.reproj_thresh)
            keep &= cheirality_mask(P1, Xi)
            keep &= cheirality_mask(P2, Xi)
            if config.min_parallax_deg > 0:
                keep &= parallax_mask(C_by[ref_id], C_by[nbr_id], Xi, min_deg=config.min_parallax_deg)
            if not np.any(keep):
                continue

        Xw = Xi[keep][:, :3].astype(np.float32)
        col = rgb_ref[idxs][keep].astype(np.float32)
        e = err[keep].astype(np.float32)

        job_xyz.append(Xw)
        job_rgb.append(col)
        job_err.append(e)

        if collect_debug_matches:
            xA_keep = np.clip(xA_pair[keep], 0.0, float(w_match - 1))
            yA_keep = np.clip(yA_pair[keep], 0.0, float(h_match - 1))
            xB_keep = np.clip(xB[keep], 0.0, float(w_match - 1))
            yB_keep = np.clip(yB[keep], 0.0, float(h_match - 1))
            matches_keep = np.stack([xA_keep, yA_keep, xB_keep, yB_keep], axis=1).astype(np.float32, copy=False)
            cert_keep = np.clip(cert_pair[keep] / cert_denom, 0.0, 1.0).astype(np.float32, copy=False)
            debug_matches_by_nbr[nbr_id] = matches_keep
            debug_cert_by_nbr[nbr_id] = cert_keep

    if not job_xyz:
        return None

    return _TriangulatedReference(
        xyz=np.concatenate(job_xyz, axis=0),
        rgb=np.concatenate(job_rgb, axis=0),
        err=np.concatenate(job_err, axis=0),
        debug_matches_by_nbr=debug_matches_by_nbr,
        debug_cert_by_nbr=debug_cert_by_nbr,
    )


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
    total_pairs_est = _estimate_total_pairs(refs_local, nn_table, cameras.img_ids, config.nns_per_ref)
    if debug_state:
        debug_state.set_total_pairs(total_pairs_est)

    viz_interval = config.viz_interval
    intermediate_ply_base = _prepare_intermediate_ply_base(config.output_path, viz_interval, on_sequential_viz)

    points = _PipelineAccumulator()
    t0 = time.time()
    matcher: Optional[RomaMatcher] = None
    pack_loader: Optional[ThreadedReferenceLoader[Optional[_PackedReferenceBatch]]] = None

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_cached = has_cached_romav2_weights()
        _report_model_setup_status(progress_callback, model_cached)
        if not model_cached and progress_callback is not None:
            progress_callback(10.0, "RoMa v2 model installation complete. Starting matching...")
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        matcher = RomaMatcher(device=device, mode="outdoor", setting=config.roma_setting)
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

        refs_consumed = 0
        while True:
            _raise_if_cancelled(cancel_requested)
            try:
                packed = next(pack_loader_iter)
            except StopIteration:
                break

            refs_consumed += 1
            _report_matching_progress(progress_callback, refs_consumed, total_refs, t0)

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
            if matched_ref is None:
                continue

            collect_debug_matches = debug_state is not None and debug_state.is_enabled()

            try:
                tri_ref = _triangulate_ref(
                    matched_ref,
                    tri_ctx,
                    collect_debug_matches=collect_debug_matches,
                )
            except Exception as ex:
                lf.log.error(f"Triangulation error for ref {packed.ref_id}: {ex}")
                tri_ref = None

            if tri_ref is None:
                continue

            points.append(tri_ref)
            if collect_debug_matches:
                _emit_debug_previews(
                    matched_ref=matched_ref,
                    tri_ref=tri_ref,
                    debug_state=debug_state,
                    cameras=cameras,
                    total_pairs_est=total_pairs_est,
                    pair_counter=points.pair_counter,
                    cancel_requested=cancel_requested,
                )
            _emit_intermediate_preview(
                on_sequential_viz=on_sequential_viz,
                intermediate_ply_base=intermediate_ply_base,
                viz_interval=viz_interval,
                points=points,
                cancel_requested=cancel_requested,
            )

    finally:
        _cleanup_pipeline_runtime(pack_loader, matcher, debug_state)
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
    elapsed = time.time() - t0

    return PipelineResult(
        xyz=xyz,
        rgb=rgb,
        err=err,
        elapsed_seconds=elapsed,
        pairs_processed=points.pairs_processed,
    )
