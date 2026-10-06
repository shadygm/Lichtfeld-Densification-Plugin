"""Triangulate sampled matches and validate observation support."""
from __future__ import annotations

import logging
from typing import Optional
import numpy as np
import torch
from ..cameras.geometry import (
    Rt_from_Rt,
    cheirality_mask,
    cheirality_mask_Rt,
    dlt_triangulate_batch,
    fundamental_from_world2cam,
    parallax_mask,
    reprojection_errors_camera,
    reprojection_errors,
    sampson_error,
    unproject_pixels,
)
from ..matching.sampling import select_samples_with_coverage
from .types import _MatchedReference, _TriangulatedReference, _TriangulationContext
from ..reconstruction.tracks import ObservationTrackBuilder

logger = logging.getLogger(__name__)


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
    ref_camera = cameras.by_id[ref_id]

    warp_list = matched_ref.warp_list_cpu
    cert_list = matched_ref.cert_list_cpu

    best_cert = torch.stack(cert_list, dim=0).amax(dim=0)

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

    # All pairs share the matcher-generated reference grid. Gather just the
    # sampled locations instead of stacking and selecting full dense warps.
    selected_warps = np.stack([warp.reshape(-1, 4)[sel_idx].numpy() for warp in warp_list])
    selected_certs = np.stack([cert.reshape(-1)[sel_idx].numpy() for cert in cert_list])
    sel = selected_warps[0]
    xA = (sel[:, 0] + 1.0) * 0.5 * (w_match - 1)
    yA = (sel[:, 1] + 1.0) * 0.5 * (h_match - 1)

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

    sample_count = len(sel_idx)
    candidate_xyz = np.zeros((len(nn_ids), sample_count, 3), dtype=np.float32)
    candidate_err = np.zeros((len(nn_ids), sample_count), dtype=np.float32)
    candidate_uv = np.zeros((len(nn_ids), sample_count, 2), dtype=np.float32)
    candidate_valid = np.zeros((len(nn_ids), sample_count), dtype=bool)

    for kidx, nbr_id in enumerate(nn_ids):
        idxs = np.arange(sel_idx.shape[0], dtype=np.int64)
        if not config.no_filter:
            idxs = idxs[selected_certs[kidx] > 0]
        neighbor = cameras.by_id[nbr_id]
        wB_cam, hB_cam = neighbor.width, neighbor.height
        sxB = wB_cam / float(w_match)
        syB = hB_cam / float(h_match)

        xB_norm_k = selected_warps[kidx, idxs, 2]
        yB_norm_k = selected_warps[kidx, idxs, 3]
        xB = (xB_norm_k + 1.0) * 0.5 * (w_match - 1)
        yB = (yB_norm_k + 1.0) * 0.5 * (h_match - 1)
        uvB = np.stack([xB * sxB, yB * syB], axis=1)

        pair_uses_distortion_aware = bool(
            ref_id in cameras.distorted_ids
            or nbr_id in cameras.distorted_ids
        )

        if (not pair_uses_distortion_aware) and (not config.no_filter) and config.sampson_thresh > 0:
            F = fundamental_from_world2cam(
                ref_camera.K,
                ref_camera.R,
                ref_camera.t,
                neighbor.K,
                neighbor.R,
                neighbor.t,
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

        uvA = uvA_full[idxs]
        P1, P2 = ref_camera.P, neighbor.P

        if pair_uses_distortion_aware:
            cam1 = ref_camera.colmap_camera
            cam2 = neighbor.colmap_camera
            if cam1 is None or cam2 is None:
                logger.warning(
                    "Distortion-aware triangulation requested but COLMAP camera "
                    f"metadata is missing for pair {ref_id}->{nbr_id}; skipping pair."
                )
                continue

            uvA_cam, validA = unproject_pixels(cam1, uvA)
            uvB_cam, validB = unproject_pixels(cam2, uvB)
            valid_uv = validA & validB
            if not np.any(valid_uv):
                continue

            idxs = idxs[valid_uv]
            uvA = uvA[valid_uv]
            uvB = uvB[valid_uv]
            uvA_used = uvA
            uvB_used = uvB
            uvA_cam = uvA_cam[valid_uv]
            uvB_cam = uvB_cam[valid_uv]
            xB = xB[valid_uv]
            yB = yB[valid_uv]

            Rt1 = Rt_from_Rt(ref_camera.R, ref_camera.t)
            Rt2 = Rt_from_Rt(neighbor.R, neighbor.t)
            Xi = dlt_triangulate_batch(Rt1, Rt2, uvA_cam, uvB_cam)

            err1, valid_proj1 = reprojection_errors_camera(
                cam1, ref_camera.R, ref_camera.t, Xi, uvA_used
            )
            err2, valid_proj2 = reprojection_errors_camera(
                cam2, neighbor.R, neighbor.t, Xi, uvB_used
            )
            err = np.maximum(err1, err2)
            projection_valid = valid_proj1 & valid_proj2
            cheirality_valid = cheirality_mask_Rt(ref_camera.R, ref_camera.t, Xi)
            cheirality_valid &= cheirality_mask_Rt(neighbor.R, neighbor.t, Xi)
        else:
            Xi = dlt_triangulate_batch(P1, P2, uvA, uvB)

            err1 = reprojection_errors(P1, Xi, uvA)
            err2 = reprojection_errors(P2, Xi, uvB)
            err = np.maximum(err1, err2)
            projection_valid = np.isfinite(err)
            cheirality_valid = cheirality_mask(P1, Xi)
            cheirality_valid &= cheirality_mask(P2, Xi)

        if config.no_filter:
            finite_mask = np.isfinite(Xi).all(axis=1) & np.isfinite(err) & projection_valid
            if not np.any(finite_mask):
                continue
            keep = finite_mask
        else:
            keep = np.isfinite(Xi).all(axis=1) & projection_valid
            keep &= err <= float(config.reproj_thresh)
            keep &= cheirality_valid
            if config.min_parallax_deg > 0:
                keep &= parallax_mask(ref_camera.C, neighbor.C, Xi, min_deg=config.min_parallax_deg)
            if not np.any(keep):
                continue

        kept_idxs = idxs[keep]
        Xw = Xi[keep][:, :3].astype(np.float32)
        e = err[keep].astype(np.float32)
        uvB_keep = uvB[keep]
        candidate_xyz[kidx, kept_idxs] = Xw
        candidate_err[kidx, kept_idxs] = e
        candidate_uv[kidx, kept_idxs] = uvB_keep
        candidate_valid[kidx, kept_idxs] = True

    # Fuse the same inverse-error-weighted pair estimates, then validate support
    # in camera-sized batches instead of making tiny NumPy calls for every point.
    weights = np.where(candidate_valid, 1.0 / np.maximum(candidate_err, 1.0e-4), 0.0)
    xyz = (candidate_xyz * weights[..., None]).sum(axis=0) / np.maximum(weights.sum(axis=0), 1.0e-8)[:, None]
    homogeneous = np.column_stack((xyz, np.ones(sample_count, dtype=np.float32)))

    def support_errors(camera_id, points, observations):
        camera = cameras.by_id[camera_id]
        if camera_id in cameras.distorted_ids:
            return reprojection_errors_camera(camera.colmap_camera, camera.R, camera.t, points, observations)[0]
        errors = reprojection_errors(camera.P, points, observations)
        if not config.no_filter:
            # GEMM and per-point GEMV round differently. Preserve the original
            # acceptance decision for observations near the filtering boundary.
            boundary = np.flatnonzero(np.abs(errors - config.reproj_thresh) < 0.002)
            for index in boundary:
                errors[index] = reprojection_errors(
                    camera.P, points[index:index + 1], observations[index:index + 1],
                )[0]
        return errors

    ref_errors = support_errors(ref_id, homogeneous, uvA_full)
    valid = candidate_valid.any(axis=0)
    if not config.no_filter:
        valid &= np.isfinite(ref_errors) & (ref_errors <= config.reproj_thresh)
    active = np.flatnonzero(valid)
    if not len(active):
        return None
    tracks = ObservationTrackBuilder(len(active), len(nn_ids) + 1, tri_ctx.retain_observations)
    tracks.append(np.arange(len(active)), ref_id, uvA_full[active])
    errors = np.where(np.isfinite(ref_errors), ref_errors, 0.0)
    has_error = np.isfinite(ref_errors)
    debug_matches_by_nbr = {}
    debug_cert_by_nbr = {}
    accepted_by_id = {ref_id: np.ones(sample_count, dtype=bool)}
    cert_denom = float(matcher_sample_cap) if matcher_sample_cap > 1.0e-6 else 1.0

    for kidx, nbr_id in enumerate(nn_ids):
        indices = np.flatnonzero(candidate_valid[kidx] & valid)
        if not len(indices):
            continue
        support = support_errors(nbr_id, homogeneous[indices], candidate_uv[kidx, indices])
        accepted = np.ones(len(indices), dtype=bool)
        if not config.no_filter:
            accepted &= np.isfinite(support) & (support <= config.reproj_thresh)
        seen = accepted_by_id.setdefault(nbr_id, np.zeros(sample_count, dtype=bool))
        accepted &= ~seen[indices]
        indices, support = indices[accepted], support[accepted]
        seen[indices] = True
        finite = np.isfinite(support)
        supported = indices[finite]
        errors[supported] = np.maximum(errors[supported], np.maximum(support[finite], candidate_err[kidx, supported]))
        has_error[supported] = True
        positions = np.searchsorted(active, indices)
        tracks.append(positions, nbr_id, candidate_uv[kidx, indices])
        if collect_debug_matches and len(indices):
            xB = (selected_warps[kidx, indices, 2] + 1.0) * 0.5 * (w_match - 1)
            yB = (selected_warps[kidx, indices, 3] + 1.0) * 0.5 * (h_match - 1)
            matches = np.column_stack((
                np.clip(xA[indices], 0, w_match - 1), np.clip(yA[indices], 0, h_match - 1),
                np.clip(xB, 0, w_match - 1), np.clip(yB, 0, h_match - 1),
            )).astype(np.float32)
            certainty = np.clip(selected_certs[kidx, indices] / cert_denom, 0, 1).astype(np.float32)
            if nbr_id in debug_matches_by_nbr:
                debug_matches_by_nbr[nbr_id] = np.concatenate((debug_matches_by_nbr[nbr_id], matches))
                debug_cert_by_nbr[nbr_id] = np.concatenate((debug_cert_by_nbr[nbr_id], certainty))
            else:
                debug_matches_by_nbr[nbr_id] = matches
                debug_cert_by_nbr[nbr_id] = certainty

    errors = np.where(has_error, errors, candidate_err.max(axis=0))
    return _TriangulatedReference(
        xyz=xyz[active], rgb=rgb_ref[active].astype(np.float32), err=errors[active].astype(np.float32),
        tracks=tracks.finish(), debug_matches_by_nbr=debug_matches_by_nbr,
        debug_cert_by_nbr=debug_cert_by_nbr,
    )
