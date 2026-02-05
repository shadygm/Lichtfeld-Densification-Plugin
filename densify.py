#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""High-level orchestration for the LichtFeld densification pipeline."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import lichtfeld as lf
import numpy as np
import pycolmap

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from core.camera_models import CameraRecord
from core.config import DensePipelineConfig
from core.geometry import K_from_camera, P_from_KRt, cam_center_world, pose_world2cam
from core.image_utils import find_image, image_dir, to_uint8_rgb
from core.pipeline import run_dense_pipeline
from core.selection import nearest_neighbors, select_cameras_by_visibility, select_cameras_kcenters
from core.writers import write_ply, write_points3D_bin


def load_reconstruction(sparse_dir: str):
    rec = pycolmap.Reconstruction(sparse_dir)
    return rec, rec.cameras, rec.images


def _build_camera_records_from_colmap(
    cams: Dict[int, pycolmap.Camera],
    imgs: Dict[int, pycolmap.Image],
    images_dir: str,
) -> Tuple[List[CameraRecord], List[int]]:
    records: List[CameraRecord] = []
    img_ids = sorted(list(imgs.keys()))
    for iid in img_ids:
        im = imgs[iid]
        cam = cams[im.camera_id]
        img_path = find_image(images_dir, im.name)
        K = K_from_camera(cam)
        R, t = pose_world2cam(im)
        P = P_from_KRt(K, R, t)
        C = cam_center_world(R, t)
        records.append(
            CameraRecord(
                uid=iid,
                image_path=img_path,
                width=cam.width,
                height=cam.height,
                K=K,
                R=R,
                t=t,
                P=P,
                C=C,
            )
        )
    return records, img_ids


def _flat_pose_stack(records: List[CameraRecord]) -> np.ndarray:
    return np.stack([cam.flat_pose() for cam in records], axis=0)


def _select_reference_indices(
    rec: pycolmap.Reconstruction,
    flat_poses: np.ndarray,
    img_ids: List[int],
    num_refs: int,
) -> List[int]:
    idx_map = {iid: i for i, iid in enumerate(img_ids)}
    try:
        refs = select_cameras_by_visibility(rec, num_refs)
        return [idx_map[r] for r in refs if r in idx_map]
    except Exception as exc:
        lf.log.warn(f"Visibility-based selection failed: {exc}")
        return select_cameras_kcenters(flat_poses, num_refs)


def _apply_point_cap(
    xyz: np.ndarray,
    rgb: np.ndarray,
    err: np.ndarray,
    max_points: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if max_points > 0 and xyz.shape[0] > max_points:
        sel = np.random.default_rng(seed).choice(xyz.shape[0], size=max_points, replace=False)
        return xyz[sel], rgb[sel], err[sel]
    return xyz, rgb, err


def _write_output(path: str, xyz: np.ndarray, rgb: np.ndarray, err: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rgb_uint8 = to_uint8_rgb(rgb)
    if path.lower().endswith(".ply"):
        write_ply(path, xyz, rgb_uint8)
    else:
        write_points3D_bin(path, xyz, rgb_uint8, err)


def dense_init(args, progress_callback: Optional[Callable[[float, str], None]] = None) -> int:
    np.random.seed(args.seed)
    scene_root = os.path.abspath(args.scene_root)
    sparse_dir = os.path.join(scene_root, "sparse", "0")
    images_dir = image_dir(scene_root, args.images_subdir)

    rec, cams, imgs = load_reconstruction(sparse_dir)
    records, img_ids = _build_camera_records_from_colmap(cams, imgs, images_dir)
    flat_poses = _flat_pose_stack(records)

    num_refs = int(round(args.num_refs * len(img_ids))) if args.num_refs <= 1.0 else int(args.num_refs)
    refs_local = _select_reference_indices(rec, flat_poses, img_ids, max(1, num_refs))
    nn_table = nearest_neighbors(flat_poses, max(1, args.nns_per_ref))

    config = DensePipelineConfig(
        output_path=os.path.join(sparse_dir, args.out_name),
        roma_setting=args.roma_setting,
        num_refs=args.num_refs,
        nns_per_ref=args.nns_per_ref,
        matches_per_ref=args.matches_per_ref,
        certainty_thresh=args.certainty_thresh,
        reproj_thresh=args.reproj_thresh,
        sampson_thresh=args.sampson_thresh,
        min_parallax_deg=args.min_parallax_deg,
        max_points=args.max_points,
        no_filter=args.no_filter,
        seed=args.seed,
        viz_interval=0,
    )

    result = run_dense_pipeline(
        records,
        refs_local,
        nn_table,
        config,
        progress_callback=progress_callback,
        on_sequential_viz=None,
    )

    xyz, rgb, err = _apply_point_cap(result.xyz, result.rgb, result.err, args.max_points, args.seed)
    if progress_callback:
        progress_callback(95.0, "Writing output...")
    _write_output(config.output_path, xyz, rgb, err)
    lf.log.info(f"Dense reconstruction finished: {xyz.shape[0]:,} points -> {config.output_path}")
    if progress_callback:
        progress_callback(100.0, f"Done! {xyz.shape[0]:,} points")
    return 0


LFSDenseConfig = DensePipelineConfig


def extract_cameras_from_lfs(camera_nodes) -> List[CameraRecord]:
    records: List[CameraRecord] = []
    for node in camera_nodes:
        if not getattr(node, "has_camera", False):
            continue
        width = node.camera_width
        height = node.camera_height
        fx = node.camera_focal_x
        fy = node.camera_focal_y
        cx = width / 2.0
        cy = height / 2.0
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        R = np.asarray(node.camera_R, dtype=np.float64)
        t = np.asarray(node.camera_T, dtype=np.float64).reshape(3, 1)
        P = K @ np.concatenate([R, t], axis=1)
        C = (-R.T @ t).reshape(3)
        records.append(
            CameraRecord(
                uid=node.camera_uid,
                image_path=node.image_path,
                width=width,
                height=height,
                K=K,
                R=R,
                t=t,
                P=P,
                C=C,
            )
        )
    return records


def dense_init_from_lfs(
    camera_nodes,
    config: DensePipelineConfig,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    on_sequential_viz: Optional[Callable[[str], None]] = None,
) -> Tuple[int, Optional[str]]:
    np.random.seed(config.seed)
    if progress_callback:
        progress_callback(2.0, "Extracting camera data from scene...")

    records = extract_cameras_from_lfs(camera_nodes)
    if len(records) < 2:
        return 1, "Need at least 2 cameras for dense initialization"

    flat_poses = _flat_pose_stack(records)
    num_refs = int(round(config.num_refs * len(records))) if config.num_refs <= 1.0 else int(config.num_refs)
    refs_local = select_cameras_kcenters(flat_poses, max(1, num_refs))
    nn_table = nearest_neighbors(flat_poses, max(1, config.nns_per_ref))

    lf.log.info(f"Prepared {len(records)} cameras (refs={len(refs_local)})")

    try:
        result = run_dense_pipeline(
            records,
            refs_local,
            nn_table,
            config,
            progress_callback=progress_callback,
            on_sequential_viz=on_sequential_viz,
        )
    except RuntimeError as exc:
        return 1, str(exc)

    xyz, rgb, err = _apply_point_cap(result.xyz, result.rgb, result.err, config.max_points, config.seed)
    if progress_callback:
        progress_callback(95.0, "Writing output PLY...")
    write_ply(config.output_path, xyz, to_uint8_rgb(rgb))
    lf.log.info(f"Dense point cloud saved to {config.output_path} ({xyz.shape[0]:,} points)")
    if progress_callback:
        progress_callback(100.0, f"Done! {xyz.shape[0]:,} points")
    return 0, config.output_path


def build_argparser():
    ap = argparse.ArgumentParser(
        "Dense COLMAP initializer (EDGS-style + RoMa v2) with GPU↔CPU pipelining"
    )
    ap.add_argument("--scene_root", type=str, required=True, help="Path containing images*/ and sparse/0/")
    ap.add_argument(
        "--images_subdir",
        type=str,
        default="images_2",
        help="Which images dir to read under scene_root",
    )
    ap.add_argument(
        "--out_name",
        type=str,
        default="points3D_dense.ply",
        help="Output filename under sparse/0/",
    )
    ap.add_argument(
        "--roma_setting",
        type=str,
        default="fast",
        choices=["precise", "high", "base", "fast", "turbo"],
        help="RoMaV2 quality/speed setting",
    )
    ap.add_argument(
        "--roma_model",
        type=str,
        default="outdoor",
        choices=["outdoor", "indoor"],
        help="Legacy flag for compatibility (RoMaV2 is unified)",
    )
    ap.add_argument(
        "--num_refs",
        type=float,
        default=0.75,
        help="Fraction (<=1) or count (>1) of frames to use as references",
    )
    ap.add_argument(
        "--nns_per_ref",
        type=int,
        default=4,
        help="Nearest neighbors per reference (3-5 is robust)",
    )
    ap.add_argument(
        "--matches_per_ref",
        type=int,
        default=12000,
        help="Samples per ref after aggregation",
    )
    ap.add_argument(
        "--certainty_thresh",
        type=float,
        default=0.20,
        help="Min certainty floor before selection",
    )
    ap.add_argument(
        "--reproj_thresh",
        type=float,
        default=1.5,
        help="Max reprojection error (px)",
    )
    ap.add_argument(
        "--sampson_thresh",
        type=float,
        default=5.0,
        help="Max Sampson error (px^2) pre-triangulation (<=0 disables)",
    )
    ap.add_argument(
        "--min_parallax_deg",
        type=float,
        default=0.5,
        help="Min parallax angle in degrees",
    )
    ap.add_argument(
        "--no_filter",
        action="store_true",
        help="Disable geometric filtering (debug only)",
    )
    ap.add_argument(
        "--max_points",
        type=int,
        default=0,
        help="Optional cap on total points (0 = unlimited)",
    )
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    return ap


if __name__ == "__main__":
    cli_args = build_argparser().parse_args()
    raise SystemExit(dense_init(cli_args))
