"""Shared dense pipeline orchestration."""
from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from queue import Queue
from typing import Callable, Dict, List, Optional, Tuple

import lichtfeld as lf
import numpy as np
import torch
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
from .image_utils import load_rgb_resized, to_uint8_rgb
from .matcher import RomaMatcher
from .sampling import select_samples_with_coverage
from .writers import write_ply


@dataclass
class PipelineResult:
    xyz: np.ndarray
    rgb: np.ndarray
    err: np.ndarray
    elapsed_seconds: float
    pairs_processed: int


def make_cpu_worker(
    job_q: Queue,
    res_q: Queue,
    K_by: Dict[int, np.ndarray],
    R_by: Dict[int, np.ndarray],
    t_by: Dict[int, np.ndarray],
    P_by: Dict[int, np.ndarray],
    C_by: Dict[int, np.ndarray],
    size_by: Dict[int, Tuple[int, int]],
    config: DensePipelineConfig,
    matcher_sample_cap: float,
):
    def worker():
        while True:
            job = job_q.get()
            if job is None:
                job_q.task_done()
                break
            try:
                ref_id = job["ref_id"]
                nn_ids: List[int] = job["nn_ids"]
                warp_list: List[torch.Tensor] = job["warp_list"]
                cert_list: List[torch.Tensor] = job["cert_list"]
                imA_np: np.ndarray = job["imA_np"]
                w_match, h_match = job["w_match"], job["h_match"]
                wA_cam, hA_cam = job["wA_cam"], job["hA_cam"]

                device_cpu = torch.device("cpu")
                H, W = cert_list[0].shape
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
                    res_q.put((None, None, None))
                    job_q.task_done()
                    continue

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

                if job_xyz:
                    xyz = np.concatenate(job_xyz, axis=0)
                    rgb = np.concatenate(job_rgb, axis=0)
                    err = np.concatenate(job_err, axis=0)
                    res_q.put((xyz, rgb, err))
                else:
                    res_q.put((None, None, None))

            except Exception as ex:
                lf.log.error(f"CPU worker error: {ex}")
                res_q.put((None, None, None))
            finally:
                job_q.task_done()

    return worker


def _start_result_collector(
    res_q: Queue,
    collector_state: Dict[str, object],
    on_sequential_viz: Optional[Callable[[str], None]],
    viz_interval: int,
    intermediate_ply_base: Optional[str],
):
    def result_collector():
        state = collector_state
        while True:
            try:
                xyz, rgb, err = res_q.get(timeout=0.1)
            except Exception:
                with state["lock"]:
                    if state["stop"]:
                        break
                continue

            if xyz is None:
                continue

            with state["lock"]:
                state["all_xyz"].append(xyz)
                state["all_rgb"].append(rgb)
                state["all_err"].append(err)
                state["pairs_processed"] += 1
                pairs_count = state["pairs_processed"]
                current_xyz = list(state["all_xyz"])
                current_rgb = list(state["all_rgb"])

            if (
                on_sequential_viz
                and viz_interval > 0
                and pairs_count % viz_interval == 0
                and current_xyz
                and intermediate_ply_base
            ):
                try:
                    xyz_so_far = np.concatenate(current_xyz, axis=0)
                    rgb_so_far = np.concatenate(current_rgb, axis=0)
                    intermediate_ply_path = f"{intermediate_ply_base}_{pairs_count}.ply"
                    write_ply(intermediate_ply_path, xyz_so_far, to_uint8_rgb(rgb_so_far))
                    lf.log.debug(
                        f"Live update: {xyz_so_far.shape[0]:,} points after {pairs_count} refs"
                    )
                    on_sequential_viz(intermediate_ply_path)
                except Exception as exc:
                    lf.log.warn(f"Failed to emit intermediate PLY: {exc}")

    thread = threading.Thread(target=result_collector, daemon=True)
    thread.start()
    return thread


def run_dense_pipeline(
    camera_records: List[CameraRecord],
    refs_local: List[int],
    nn_table: np.ndarray,
    config: DensePipelineConfig,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    on_sequential_viz: Optional[Callable[[str], None]] = None,
) -> PipelineResult:
    np.random.seed(config.seed)

    img_ids = [cam.uid for cam in camera_records]
    path_by = {cam.uid: cam.image_path for cam in camera_records}
    size_by = {cam.uid: (cam.width, cam.height) for cam in camera_records}
    K_by = {cam.uid: cam.K for cam in camera_records}
    R_by = {cam.uid: cam.R for cam in camera_records}
    t_by = {cam.uid: cam.t for cam in camera_records}
    P_by = {cam.uid: cam.P for cam in camera_records}
    C_by = {cam.uid: cam.C for cam in camera_records}

    job_q = Queue(maxsize=8)
    res_q = Queue()
    collector_state = {
        "all_xyz": [],
        "all_rgb": [],
        "all_err": [],
        "pairs_processed": 0,
        "stop": False,
        "lock": threading.Lock(),
    }

    viz_interval = config.viz_interval
    intermediate_ply_base = None
    if on_sequential_viz and viz_interval > 0:
        output_dir = os.path.dirname(config.output_path)
        base_no_ext = os.path.splitext(os.path.basename(config.output_path))[0]
        os.makedirs(output_dir, exist_ok=True)
        intermediate_ply_base = os.path.join(output_dir, f"{base_no_ext}_intermediate")

    matcher = None
    worker_thread = None
    collector_thread = None
    jobs_enqueued = 0
    t0 = time.time()

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        matcher = RomaMatcher(device=device, mode="outdoor", setting=config.roma_setting)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        worker_fn = make_cpu_worker(
            job_q,
            res_q,
            K_by,
            R_by,
            t_by,
            P_by,
            C_by,
            size_by,
            config,
            matcher.sample_thresh,
        )
        worker_thread = threading.Thread(target=worker_fn, daemon=True)
        worker_thread.start()

        collector_thread = _start_result_collector(
            res_q,
            collector_state,
            on_sequential_viz,
            viz_interval,
            intermediate_ply_base,
        )

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

            imA_np = np.asarray(imA, dtype=np.uint8)
            wA_cam, hA_cam = size_by[ref_id]
            local_nns = nn_table[ref_local][: config.nns_per_ref]
            if len(local_nns) == 0:
                continue

            warp_list_cpu: List[torch.Tensor] = []
            cert_list_cpu: List[torch.Tensor] = []
            nn_ids: List[int] = []

            def load_neighbor(nn_local: int) -> Tuple[Optional[Image.Image], Optional[int], Optional[int]]:
                nbr_id = img_ids[nn_local]
                if nbr_id == ref_id:
                    return None, None, None
                try:
                    img_path = path_by[nbr_id]
                    imB = load_rgb_resized(img_path, (w_match, h_match))
                    return imB, nn_local, nbr_id
                except Exception as exc:
                    lf.log.warn(f"Failed to load neighbor: {exc}")
                    return None, None, None

            with ThreadPoolExecutor(max_workers=4) as executor:
                load_results = list(executor.map(load_neighbor, local_nns))

            nn_images = []
            for imB, nn_local, nbr_id in load_results:
                if imB is not None and nn_local is not None and nbr_id is not None:
                    nn_images.append(imB)
                    nn_ids.append(nbr_id)

            if not nn_images:
                continue

            batch_results = matcher.match_grids_batch(imA, nn_images)
            for warp_hw, cert_hw in batch_results:
                cert_hw = torch.clamp(cert_hw, min=config.certainty_thresh)
                warp_cpu = warp_hw.detach().to("cpu", non_blocking=True)
                cert_cpu = cert_hw.detach().to("cpu", non_blocking=True)
                warp_list_cpu.append(warp_cpu)
                cert_list_cpu.append(cert_cpu)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            del batch_results

            if not cert_list_cpu:
                continue

            job = dict(
                ref_id=ref_id,
                nn_ids=nn_ids,
                warp_list=warp_list_cpu,
                cert_list=cert_list_cpu,
                imA_np=imA_np,
                w_match=w_match,
                h_match=h_match,
                wA_cam=wA_cam,
                hA_cam=hA_cam,
            )
            job_q.put(job)
            jobs_enqueued += 1

    finally:
        if matcher is not None:
            del matcher
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    if progress_callback:
        progress_callback(90.0, "Finalizing triangulation...")

    job_q.put(None)
    job_q.join()
    if worker_thread:
        worker_thread.join(timeout=1.0)

    with collector_state["lock"]:
        collector_state["stop"] = True
    if collector_thread:
        collector_thread.join(timeout=1.0)

    while True:
        try:
            xyz, rgb, err = res_q.get_nowait()
            if xyz is not None:
                with collector_state["lock"]:
                    collector_state["all_xyz"].append(xyz)
                    collector_state["all_rgb"].append(rgb)
                    collector_state["all_err"].append(err)
        except Exception:
            break

    with collector_state["lock"]:
        all_xyz = collector_state["all_xyz"]
        all_rgb = collector_state["all_rgb"]
        all_err = collector_state["all_err"]
        pairs_processed = collector_state["pairs_processed"]

    if not all_xyz:
        raise RuntimeError("No points triangulated. Try adjusting parameters.")

    xyz = np.concatenate(all_xyz, axis=0)
    rgb = np.concatenate(all_rgb, axis=0)
    err = np.concatenate(all_err, axis=0)
    elapsed = time.time() - t0

    return PipelineResult(xyz=xyz, rgb=rgb, err=err, elapsed_seconds=elapsed, pairs_processed=pairs_processed)
