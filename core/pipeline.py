"""Shared dense pipeline orchestration."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
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
from .matcher import RomaMatcher
from .sampling import select_samples_with_coverage
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

_PREVIEW_MAX_MATCHES = 10000


def _build_match_preview(
    imA_np: np.ndarray,
    imB_np: np.ndarray,
    warp_cpu: torch.Tensor,
    cert_cpu: torch.Tensor,
    sample_cap: float,
    ref_id: int,
    nbr_id: int,
    ref_label: str,
    nbr_label: str,
    pair_index: int,
    total_pairs: int,
    max_matches: int = _PREVIEW_MAX_MATCHES,
) -> Optional[MatchPreview]:
    """Cheap visualization: sample top-certainty correspondences.

    No geometric filtering is performed — this is purely for visual debugging.
    Returns raw match endpoints so the UI can draw overlays (e.g. via
    ``layout.draw_line``).
    """

    if warp_cpu.numel() == 0 or cert_cpu.numel() == 0:
        return None

    H, W = cert_cpu.shape
    sel_idx = select_samples_with_coverage(
        cert_cpu,
        max_matches,
        cap=sample_cap,
        border=2,
        tiles=12,
        no_filter=True,
    )
    if sel_idx.size == 0:
        return None

    agg = warp_cpu.reshape(-1, 4)[sel_idx].numpy()
    # grid coords are in [-1, 1] with align_corners=False
    xA = (agg[:, 0] + 1.0) * 0.5 * W - 0.5
    yA = (agg[:, 1] + 1.0) * 0.5 * H - 0.5
    xB = (agg[:, 2] + 1.0) * 0.5 * W - 0.5
    yB = (agg[:, 3] + 1.0) * 0.5 * H - 0.5

    xA = np.clip(xA, 0.0, W - 1.0)
    yA = np.clip(yA, 0.0, H - 1.0)
    xB = np.clip(xB, 0.0, W - 1.0)
    yB = np.clip(yB, 0.0, H - 1.0)

    # Color by certainty: weak -> red, strong -> green.
    # Note: select_samples_with_coverage clamps cert to <= sample_cap.
    cert_sel = cert_cpu.reshape(-1)[sel_idx].detach().to("cpu").numpy()
    denom = float(sample_cap) if float(sample_cap) > 1e-6 else 1.0
    cert_norm = np.clip(cert_sel / denom, 0.0, 1.0)

    matches = np.stack([xA, yA, xB, yB], axis=1).astype(np.float32, copy=False)
    return MatchPreview(
        ref_id=ref_id,
        nbr_id=nbr_id,
        ref_label=ref_label,
        nbr_label=nbr_label,
        left_image=imA_np,
        right_image=imB_np,
        matches=matches,
        cert_norm=cert_norm.astype(np.float32, copy=False),
        match_count=int(len(sel_idx)),
        pair_index=int(pair_index),
        total_pairs=int(total_pairs),
    )


def _triangulate_ref(
    ref_id: int,
    nn_ids: List[int],
    warp_list: List[torch.Tensor],
    cert_list: List[torch.Tensor],
    imA_np: np.ndarray,
    w_match: int,
    h_match: int,
    wA_cam: int,
    hA_cam: int,
    K_by: Dict[int, np.ndarray],
    R_by: Dict[int, np.ndarray],
    t_by: Dict[int, np.ndarray],
    P_by: Dict[int, np.ndarray],
    C_by: Dict[int, np.ndarray],
    size_by: Dict[int, Tuple[int, int]],
    config: DensePipelineConfig,
    matcher_sample_cap: float,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Triangulate matches for a single reference view. Returns (xyz, rgb, err) or None."""

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

    for nbr_id, idxs in groups.items():
        idxs = np.asarray(idxs, dtype=np.int64)
        wB_cam, hB_cam = size_by[nbr_id]
        sxB = wB_cam / float(w_match)
        syB = hB_cam / float(h_match)

        xB = (xB_norm[idxs] + 1.0) * 0.5 * (w_match - 1)
        yB = (yB_norm[idxs] + 1.0) * 0.5 * (h_match - 1)
        uvB = np.stack([xB * sxB, yB * syB], axis=1)

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

        Xw = Xi[keep][:, :3].astype(np.float64)
        col = rgb_ref[idxs][keep].astype(np.float32)
        e = err[keep].astype(np.float64)

        job_xyz.append(Xw)
        job_rgb.append(col)
        job_err.append(e)

    if not job_xyz:
        return None

    xyz = np.concatenate(job_xyz, axis=0)
    rgb = np.concatenate(job_rgb, axis=0)
    err = np.concatenate(job_err, axis=0)
    return xyz, rgb, err


def run_dense_pipeline(
    camera_records: List[CameraRecord],
    refs_local: List[int],
    nn_table: np.ndarray,
    config: DensePipelineConfig,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    on_sequential_viz: Optional[Callable[[str], None]] = None,
    debug_state: Optional[MatchDebugState] = None,
) -> PipelineResult:
    np.random.seed(config.seed)

    img_ids = [cam.uid for cam in camera_records]
    path_by = {cam.uid: cam.image_path for cam in camera_records}
    mask_by = {cam.uid: getattr(cam, "mask_path", None) for cam in camera_records}
    size_by = {cam.uid: (cam.width, cam.height) for cam in camera_records}
    K_by = {cam.uid: cam.K for cam in camera_records}
    R_by = {cam.uid: cam.R for cam in camera_records}
    t_by = {cam.uid: cam.t for cam in camera_records}
    P_by = {cam.uid: cam.P for cam in camera_records}
    C_by = {cam.uid: cam.C for cam in camera_records}

    total_pairs_est = sum(
        sum(1 for n in nn_table[ref_idx][: config.nns_per_ref] if img_ids[n] != img_ids[ref_idx])
        for ref_idx in refs_local
    )
    if debug_state:
        debug_state.set_total_pairs(total_pairs_est)

    viz_interval = config.viz_interval
    intermediate_ply_base = None
    if on_sequential_viz and viz_interval > 0:
        output_dir = os.path.dirname(config.output_path)
        base_no_ext = os.path.splitext(os.path.basename(config.output_path))[0]
        os.makedirs(output_dir, exist_ok=True)
        intermediate_ply_base = os.path.join(output_dir, f"{base_no_ext}_intermediate")

    all_xyz: List[np.ndarray] = []
    all_rgb: List[np.ndarray] = []
    all_err: List[np.ndarray] = []
    pairs_processed = 0
    pair_counter = 0
    t0 = time.time()
    matcher = None

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        matcher = RomaMatcher(device=device, mode="outdoor", setting=config.roma_setting)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        total_refs = len(refs_local)
        w_match, h_match = matcher.w_resized, matcher.h_resized

        for idx, ref_local in enumerate(refs_local):
            if progress_callback:
                pct = 10.0 + (float(idx) / max(1, total_refs)) * 80.0
                progress_callback(pct, f"Matching view {idx + 1}/{total_refs}")

            ref_id = img_ids[ref_local]
            ref_path = path_by[ref_id]
            try:
                imA = load_rgb_resized(ref_path, (w_match, h_match))
            except Exception as exc:
                lf.log.warn(f"Failed to load reference {ref_path}: {exc}")
                continue

            # If a mask is available, restrict matching to masked-in pixels.
            maskA_np = None
            ref_mask_path = mask_by.get(ref_id)
            if ref_mask_path:
                try:
                    maskA_np = load_mask_resized_np(ref_mask_path, (w_match, h_match))
                    imA = apply_mask_to_rgb(imA, maskA_np)
                except Exception as exc:
                    lf.log.warn(f"Failed to load/apply mask for reference {ref_id}: {exc}")
                    maskA_np = None

            imA_np = np.asarray(imA, dtype=np.uint8)
            wA_cam, hA_cam = size_by[ref_id]
            local_nns = nn_table[ref_local][: config.nns_per_ref]
            if len(local_nns) == 0:
                continue

            warp_list_cpu: List[torch.Tensor] = []
            cert_list_cpu: List[torch.Tensor] = []
            nn_ids: List[int] = []
            nn_masks: List[Optional[np.ndarray]] = []
            nn_images: List[Image.Image] = []

            # Load neighbors sequentially
            for nn_local in local_nns:
                nbr_id = img_ids[nn_local]
                if nbr_id == ref_id:
                    continue
                try:
                    img_path = path_by[nbr_id]
                    imB = load_rgb_resized(img_path, (w_match, h_match))

                    maskB_np = None
                    nbr_mask_path = mask_by.get(nbr_id)
                    if nbr_mask_path:
                        try:
                            maskB_np = load_mask_resized_np(nbr_mask_path, (w_match, h_match))
                            imB = apply_mask_to_rgb(imB, maskB_np)
                        except Exception as exc:
                            lf.log.warn(f"Failed to load/apply mask for neighbor {nbr_id}: {exc}")
                            maskB_np = None

                    nn_images.append(imB)
                    nn_ids.append(nbr_id)
                    nn_masks.append(maskB_np)
                except Exception as exc:
                    lf.log.warn(f"Failed to load neighbor: {exc}")

            if not nn_images:
                continue

            nn_arrays = [np.asarray(img, dtype=np.uint8) for img in nn_images]
            batch_results = matcher.match_grids_batch(imA, nn_images)
            # Precompute reference mask tensor on the match device (if present).
            maskA_t = None
            if maskA_np is not None:
                maskA_t = torch.from_numpy(maskA_np.astype(np.float32))

            for (warp_hw, cert_hw), maskB_np, nbr_id, imB_np in zip(batch_results, nn_masks, nn_ids, nn_arrays):
                cert_hw = torch.clamp(cert_hw, min=config.certainty_thresh)

                # Apply reference mask (A): cert becomes 0 outside mask.
                if maskA_t is not None:
                    cert_hw = cert_hw * maskA_t.to(device=cert_hw.device, dtype=cert_hw.dtype)

                # Apply neighbor mask (B): cert becomes 0 if warp lands outside B's mask.
                if maskB_np is not None:
                    maskB_t = (
                        torch.from_numpy(maskB_np.astype(np.float32))
                        .to(device=cert_hw.device, dtype=cert_hw.dtype)
                        .view(1, 1, cert_hw.shape[0], cert_hw.shape[1])
                    )
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

                if debug_state is not None and debug_state.is_enabled():
                    # Throttle: only build a preview every N pairs (or always in manual-step mode)
                    should_preview = (
                        not debug_state.is_auto_step()
                        or _DEBUG_PREVIEW_INTERVAL <= 0
                        or pair_counter % _DEBUG_PREVIEW_INTERVAL == 1
                    )
                    if should_preview:
                        try:
                            preview = _build_match_preview(
                                imA_np,
                                imB_np,
                                warp_cpu,
                                cert_cpu,
                                matcher.sample_thresh,
                                ref_id,
                                nbr_id,
                                os.path.basename(ref_path),
                                os.path.basename(path_by[nbr_id]),
                                pair_counter,
                                total_pairs_est if total_pairs_est > 0 else pair_counter,
                            )
                            if preview:
                                debug_state.submit_preview(preview)
                        except Exception as exc:
                            lf.log.warn(f"Debug preview failed: {exc}")

                warp_list_cpu.append(warp_cpu)
                cert_list_cpu.append(cert_cpu)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            del batch_results

            if not cert_list_cpu:
                continue

            # Triangulate this reference view's matches inline
            try:
                result = _triangulate_ref(
                    ref_id,
                    nn_ids,
                    warp_list_cpu,
                    cert_list_cpu,
                    imA_np,
                    w_match,
                    h_match,
                    wA_cam,
                    hA_cam,
                    K_by,
                    R_by,
                    t_by,
                    P_by,
                    C_by,
                    size_by,
                    config,
                    matcher.sample_thresh,
                )
            except Exception as ex:
                lf.log.error(f"Triangulation error for ref {ref_id}: {ex}")
                result = None

            if result is not None:
                xyz, rgb, err = result
                all_xyz.append(xyz)
                all_rgb.append(rgb)
                all_err.append(err)
                pairs_processed += 1

                # Intermediate visualization
                if (
                    on_sequential_viz
                    and viz_interval > 0
                    and pairs_processed % viz_interval == 0
                    and intermediate_ply_base
                ):
                    try:
                        xyz_so_far = np.concatenate(all_xyz, axis=0)
                        rgb_so_far = np.concatenate(all_rgb, axis=0)
                        intermediate_ply_path = f"{intermediate_ply_base}_{pairs_processed}.ply"
                        write_ply(intermediate_ply_path, xyz_so_far, to_uint8_rgb(rgb_so_far))
                        lf.log.debug(
                            f"Live update: {xyz_so_far.shape[0]:,} points after {pairs_processed} refs"
                        )
                        on_sequential_viz(intermediate_ply_path)
                    except Exception as exc:
                        lf.log.warn(f"Failed to emit intermediate PLY: {exc}")

    finally:
        if matcher is not None:
            del matcher
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    if progress_callback:
        progress_callback(90.0, "Finalizing triangulation...")

    if debug_state:
        debug_state.release_waiters()

    if not all_xyz:
        raise RuntimeError("No points triangulated. Try adjusting parameters.")

    xyz = np.concatenate(all_xyz, axis=0)
    rgb = np.concatenate(all_rgb, axis=0)
    err = np.concatenate(all_err, axis=0)
    elapsed = time.time() - t0

    return PipelineResult(xyz=xyz, rgb=rgb, err=err, elapsed_seconds=elapsed, pairs_processed=pairs_processed)
