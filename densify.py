#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""High-level orchestration for the LichtFeld densification pipeline."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

import lichtfeld as lf
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from .core.camera_models import CameraRecord
from .core.config import DensePipelineConfig
from .core.geometry import K_from_camera, P_from_KRt, cam_center_world, pose_world2cam
from .core.image_utils import find_image, image_dir, to_uint8_rgb
from .core.selection import nearest_neighbors, select_cameras_by_visibility, select_cameras_kcenters
from .core.writers import write_ply, write_points3D_bin, write_sparse_model_bin

if TYPE_CHECKING:
    import pycolmap


def _voxel_downsample(
    xyz: np.ndarray, rgb: np.ndarray, voxel_size: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Downsample points so they are roughly *voxel_size* apart.

    Uses Open3D voxel grid downsampling which averages points
    falling in the same voxel, producing a uniform distribution.
    """
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    if rgb.max() > 1.0:
        pcd.colors = o3d.utility.Vector3dVector(rgb[:, :3].astype(np.float64) / 255.0)
    else:
        pcd.colors = o3d.utility.Vector3dVector(rgb[:, :3].astype(np.float64))

    down = pcd.voxel_down_sample(voxel_size=float(voxel_size))

    out_xyz = np.asarray(down.points, dtype=np.float32)
    out_rgb = np.asarray(down.colors, dtype=np.float32)  # [0, 1]
    return out_xyz, out_rgb


def _voxel_select_track_preserving(
    xyz: np.ndarray,
    rgb: np.ndarray,
    err: np.ndarray,
    tracks: Sequence[Sequence[Tuple[int, float, float]]],
    voxel_size: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[List[Tuple[int, float, float]]]]:
    if voxel_size <= 0.0 or xyz.shape[0] == 0:
        return xyz, rgb, err, [list(t) for t in tracks]

    voxels = np.floor(xyz / float(voxel_size)).astype(np.int64)
    chosen: Dict[Tuple[int, int, int], int] = {}
    for idx, voxel in enumerate(voxels):
        key = (int(voxel[0]), int(voxel[1]), int(voxel[2]))
        prev = chosen.get(key)
        if prev is None:
            chosen[key] = idx
            continue
        track_len = len(tracks[idx])
        prev_track_len = len(tracks[prev])
        if track_len > prev_track_len or (track_len == prev_track_len and float(err[idx]) < float(err[prev])):
            chosen[key] = idx

    sel = np.asarray(sorted(chosen.values()), dtype=np.int64)
    return xyz[sel], rgb[sel], err[sel], [list(tracks[i]) for i in sel]



def load_reconstruction(sparse_dir: str):
    import pycolmap

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
                mask_path=None,
                width=cam.width,
                height=cam.height,
                K=K,
                R=R,
                t=t,
                P=P,
                C=C,
                colmap_camera=cam,
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
    if num_refs >= len(img_ids):
        return list(range(len(img_ids)))
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
    tracks: Optional[Sequence[Sequence[Tuple[int, float, float]]]],
    max_points: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[List[List[Tuple[int, float, float]]]]]:
    if max_points > 0 and xyz.shape[0] > max_points:
        sel = np.random.default_rng(seed).choice(xyz.shape[0], size=max_points, replace=False)
        capped_tracks = None if tracks is None else [list(tracks[i]) for i in sel]
        return xyz[sel], rgb[sel], err[sel], capped_tracks
    return xyz, rgb, err, None if tracks is None else [list(t) for t in tracks]


def _apply_track_filter(
    xyz: np.ndarray,
    rgb: np.ndarray,
    err: np.ndarray,
    tracks: Sequence[Sequence[Tuple[int, float, float]]],
    min_track_length: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[List[Tuple[int, float, float]]]]:
    min_track = max(0, int(min_track_length))
    if min_track <= 0:
        return xyz, rgb, err, [list(t) for t in tracks]
    keep = np.asarray([len(track) >= min_track for track in tracks], dtype=bool)
    return xyz[keep], rgb[keep], err[keep], [list(track) for track, ok in zip(tracks, keep) if ok]


def _log_track_filter_stats(
    label: str,
    tracks_before: Sequence[Sequence[Tuple[int, float, float]]],
    tracks_after: Sequence[Sequence[Tuple[int, float, float]]],
    min_track_length: int,
) -> None:
    before = len(tracks_before)
    after = len(tracks_after)
    kept_pct = 100.0 if before == 0 else (float(after) / float(before)) * 100.0
    before_lengths = np.asarray([len(track) for track in tracks_before], dtype=np.int32)
    after_lengths = np.asarray([len(track) for track in tracks_after], dtype=np.int32)

    def fmt(lengths: np.ndarray) -> str:
        if lengths.size == 0:
            return "empty"
        return (
            f"min={int(lengths.min())}, "
            f"mean={float(lengths.mean()):.2f}, "
            f"max={int(lengths.max())}"
        )

    if int(min_track_length) > 0:
        lf.log.info(
            f"{label}: min_track_length={int(min_track_length)} kept "
            f"{after:,}/{before:,} points ({kept_pct:.1f}%). "
            f"Before [{fmt(before_lengths)}], after [{fmt(after_lengths)}]"
        )
    else:
        lf.log.info(
            f"{label}: track filtering disabled (min_track_length=0). "
            f"{before:,} points, track lengths [{fmt(before_lengths)}]"
        )


def _effective_neighbor_count(requested: int, camera_count: int) -> int:
    if camera_count <= 1:
        return 0
    return max(1, min(int(requested), camera_count - 1))


def _write_output(path: str, xyz: np.ndarray, rgb: np.ndarray, err: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rgb_uint8 = to_uint8_rgb(rgb)
    if path.lower().endswith(".ply"):
        write_ply(path, xyz, rgb_uint8)
    else:
        write_points3D_bin(path, xyz, rgb_uint8, err)


def _cancel_requested(cancel_requested: Optional[Callable[[], bool]]) -> bool:
    if cancel_requested is None:
        return False
    try:
        return bool(cancel_requested())
    except Exception as exc:
        lf.log.warn(f"Cancellation callback failed: {exc}")
        return False


def _split_refs_local(refs_local: Sequence[int], chunk_count: int) -> List[List[int]]:
    chunks = np.array_split(np.asarray(list(refs_local), dtype=np.int64), max(1, int(chunk_count)))
    return [chunk.astype(np.int64).tolist() for chunk in chunks if chunk.size > 0]


def _chunk_output_dir(output_path: str) -> str:
    output = Path(output_path)
    return str(output.with_suffix("")) + "_chunks"


def _jsonable_array(value: np.ndarray) -> List[Any]:
    return np.asarray(value).tolist()


def _camera_record_fingerprint(record: CameraRecord) -> Dict[str, Any]:
    camera = getattr(record, "colmap_camera", None)
    camera_model = ""
    camera_params: List[float] = []
    if camera is not None:
        camera_model = str(getattr(camera, "model", ""))
        if hasattr(getattr(camera, "model", None), "name"):
            camera_model = str(camera.model.name)
        if hasattr(camera, "params"):
            camera_params = [float(v) for v in np.asarray(camera.params, dtype=np.float64).reshape(-1)]

    return {
        "uid": int(record.uid),
        "image_path": str(record.image_path),
        "mask_path": str(record.mask_path) if record.mask_path is not None else None,
        "width": int(record.width),
        "height": int(record.height),
        "K": _jsonable_array(np.asarray(record.K, dtype=np.float64)),
        "R": _jsonable_array(np.asarray(record.R, dtype=np.float64)),
        "t": _jsonable_array(np.asarray(record.t, dtype=np.float64)),
        "C": _jsonable_array(np.asarray(record.C, dtype=np.float64)),
        "camera_model": camera_model,
        "camera_params": camera_params,
    }


def _chunk_run_manifest(
    records: Sequence[CameraRecord],
    refs_local: Sequence[int],
    nn_table: np.ndarray,
    config: DensePipelineConfig,
    chunk_count: int,
    max_points_per_batch: int,
    chunks: Sequence[Sequence[int]],
) -> Dict[str, Any]:
    refs = [int(v) for v in refs_local]
    nn_table_arr = np.asarray(nn_table)
    ref_neighbor_rows = {
        str(ref): [int(v) for v in nn_table_arr[int(ref)].reshape(-1).tolist()]
        for ref in refs
    }
    return {
        "schema": "densification.chunked_resume.v1",
        "records": [_camera_record_fingerprint(record) for record in records],
        "refs_local": refs,
        "ref_neighbor_rows": ref_neighbor_rows,
        "chunks": [[int(v) for v in chunk] for chunk in chunks],
        "options": {
            "roma_setting": str(config.roma_setting),
            "num_refs": float(config.num_refs),
            "nns_per_ref": int(config.nns_per_ref),
            "matches_per_ref": int(config.matches_per_ref),
            "certainty_thresh": float(config.certainty_thresh),
            "reproj_thresh": float(config.reproj_thresh),
            "sampson_thresh": float(config.sampson_thresh),
            "min_parallax_deg": float(config.min_parallax_deg),
            "max_points": int(config.max_points),
            "min_track_length": int(config.min_track_length),
            "no_filter": bool(config.no_filter),
            "use_masks": bool(config.use_masks),
            "voxel_size": float(config.voxel_size),
            "seed": int(config.seed),
            "prefetch_packages": int(config.prefetch_packages),
            "pack_workers": int(config.pack_workers),
            "chunk_count": int(chunk_count),
            "max_points_per_batch": int(max_points_per_batch),
        },
    }


def _canonical_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _chunk_run_fingerprint(manifest: Dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(manifest).encode("utf-8")).hexdigest()


def _chunk_metadata(
    manifest: Dict[str, Any],
    fingerprint: str,
    chunk_idx: int,
    chunk_refs: Sequence[int],
) -> Dict[str, Any]:
    return {
        "schema": "densification.chunk.v1",
        "run_fingerprint": fingerprint,
        "run_manifest": manifest,
        "chunk_index": int(chunk_idx),
        "chunk_refs": [int(v) for v in chunk_refs],
    }


def _save_npz_chunk(
    path_out: str,
    xyz: np.ndarray,
    rgb: np.ndarray,
    metadata: Dict[str, Any],
) -> int:
    os.makedirs(os.path.dirname(path_out), exist_ok=True)
    rgb_uint8 = to_uint8_rgb(rgb)
    np.savez(
        path_out,
        xyz=xyz.astype(np.float32, copy=False),
        rgb=rgb_uint8,
        metadata_json=np.asarray(_canonical_json(metadata)),
    )
    return int(xyz.shape[0])


def _npz_chunk_count(path_in: str) -> int:
    with np.load(path_in) as data:
        return int(data["xyz"].shape[0])


def _load_existing_chunk(path_in: str, expected_metadata: Dict[str, Any]) -> Optional[int]:
    if not os.path.isfile(path_in):
        return None
    try:
        with np.load(path_in) as data:
            xyz = data["xyz"]
            rgb = data["rgb"]
            if xyz.ndim != 2 or xyz.shape[1] != 3:
                return None
            if rgb.ndim != 2 or rgb.shape[1] != 3:
                return None
            if rgb.shape[0] != xyz.shape[0]:
                return None
            if "metadata_json" not in data:
                lf.log.warn(f"Could not reuse existing chunk {path_in}: missing chunk metadata")
                return None
            try:
                metadata = json.loads(str(data["metadata_json"].item()))
            except Exception:
                lf.log.warn(f"Could not reuse existing chunk {path_in}: invalid chunk metadata")
                return None
            if metadata != expected_metadata:
                lf.log.warn(f"Could not reuse existing chunk {path_in}: fingerprint mismatch")
                return None
            count = int(xyz.shape[0])
            return count
    except Exception as exc:
        lf.log.warn(f"Could not reuse existing chunk {path_in}: {exc}")
        return None


def _write_ply_vertices(file_obj, xyz: np.ndarray, rgb_uint8: np.ndarray) -> None:
    if xyz.shape[0] == 0:
        return
    if rgb_uint8.dtype != np.uint8:
        rgb_uint8 = to_uint8_rgb(rgb_uint8)
    packed = np.empty(
        xyz.shape[0],
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    xyz32 = xyz.astype(np.float32, copy=False)
    packed["x"] = xyz32[:, 0]
    packed["y"] = xyz32[:, 1]
    packed["z"] = xyz32[:, 2]
    packed["red"] = rgb_uint8[:, 0]
    packed["green"] = rgb_uint8[:, 1]
    packed["blue"] = rgb_uint8[:, 2]
    packed.tofile(file_obj)


def _write_ply_from_npz_chunks(
    output_path: str,
    chunk_paths: Sequence[str],
    max_points: int,
    seed: int,
) -> Tuple[int, int]:
    counts = [_npz_chunk_count(path) for path in chunk_paths]
    total = int(sum(counts))
    if total <= 0:
        raise RuntimeError("No points remain after chunked densification.")

    output_count = total
    selected_global: Optional[np.ndarray] = None
    if max_points > 0 and total > int(max_points):
        output_count = int(max_points)
        selected_global = np.sort(
            np.random.default_rng(seed).choice(total, size=output_count, replace=False)
        )

    header = f"""ply
format binary_little_endian 1.0
element vertex {output_count}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(header.encode("ascii"))
        global_offset = 0
        selected_pos = 0
        for path, count in zip(chunk_paths, counts):
            if count <= 0:
                continue
            local_idx = None
            if selected_global is not None:
                start = selected_pos
                while selected_pos < selected_global.size and selected_global[selected_pos] < global_offset + count:
                    selected_pos += 1
                if selected_pos == start:
                    global_offset += count
                    continue
                local_idx = selected_global[start:selected_pos] - global_offset

            with np.load(path) as data:
                xyz = data["xyz"]
                rgb = data["rgb"]
                if local_idx is not None:
                    xyz = xyz[local_idx]
                    rgb = rgb[local_idx]
                _write_ply_vertices(f, xyz, rgb)
            global_offset += count

    return output_count, total


def _run_dense_pipeline_chunked(
    records: List[CameraRecord],
    refs_local: Sequence[int],
    nn_table: np.ndarray,
    config: DensePipelineConfig,
    chunk_count: int,
    max_points_per_batch: int,
    resume_chunks: bool,
    progress_callback: Optional[Callable[[float, str], None]],
    debug_state=None,
    cancel_requested: Optional[Callable[[], bool]] = None,
) -> int:
    if not config.output_path.lower().endswith(".ply"):
        raise RuntimeError("Chunked CLI densification currently supports PLY output only.")

    from .core.pipeline import PipelineCancelled, run_dense_pipeline

    chunks = _split_refs_local(refs_local, chunk_count)
    chunk_dir = _chunk_output_dir(config.output_path)
    os.makedirs(chunk_dir, exist_ok=True)
    run_manifest = _chunk_run_manifest(
        records,
        refs_local,
        nn_table,
        config,
        chunk_count,
        max_points_per_batch,
        chunks,
    )
    run_fingerprint = _chunk_run_fingerprint(run_manifest)
    lf.log.info(
        f"Chunked densification enabled: {len(chunks)} chunks, "
        f"global refs={len(refs_local)}, neighbors remain global, "
        f"fingerprint={run_fingerprint[:12]}"
    )
    if progress_callback:
        progress_callback(5.0, f"Chunked densification: {len(chunks)} chunks")

    chunk_paths: List[str] = []
    chunk_counts: List[int] = []
    for chunk_idx, chunk_refs in enumerate(chunks, start=1):
        chunk_path = os.path.join(chunk_dir, f"chunk_{chunk_idx:04d}.npz")
        chunk_metadata = _chunk_metadata(run_manifest, run_fingerprint, chunk_idx, chunk_refs)
        if resume_chunks:
            existing_count = _load_existing_chunk(chunk_path, chunk_metadata)
            if existing_count is not None:
                chunk_paths.append(chunk_path)
                chunk_counts.append(existing_count)
                lf.log.info(
                    f"Chunk {chunk_idx}/{len(chunks)} reused "
                    f"{existing_count:,} points -> {chunk_path}"
                )
                if progress_callback:
                    progress_callback(
                        5.0 + (chunk_idx / max(1, len(chunks))) * 88.0,
                        f"Reused chunk {chunk_idx}/{len(chunks)} ({existing_count:,} points)",
                    )
                continue

        if _cancel_requested(cancel_requested):
            if progress_callback:
                progress_callback(0.0, "Cancelled")
            return 2

        lf.log.info(
            f"Chunk {chunk_idx}/{len(chunks)}: processing {len(chunk_refs)} reference views"
        )

        def chunk_progress(pct: float, msg: str, _idx=chunk_idx, _total=len(chunks)) -> None:
            if progress_callback is None:
                return
            chunk_base = 5.0 + ((_idx - 1) / max(1, _total)) * 88.0
            chunk_span = 88.0 / max(1, _total)
            mapped_pct = chunk_base + (float(pct) / 100.0) * chunk_span
            progress_callback(mapped_pct, f"Matching chunk {_idx}/{_total}: {msg}")

        try:
            result = run_dense_pipeline(
                records,
                chunk_refs,
                nn_table,
                config,
                progress_callback=chunk_progress,
                on_sequential_viz=None,
                debug_state=debug_state,
                cancel_requested=cancel_requested,
            )
        except PipelineCancelled:
            if progress_callback:
                progress_callback(0.0, "Cancelled")
            return 2
        except RuntimeError as exc:
            gc.collect()
            raise RuntimeError(
                f"Chunk {chunk_idx}/{len(chunks)} failed; final PLY was not written. "
                "Completed chunks remain available for --resume_chunks."
            ) from exc

        tracks_before = [list(track) for track in result.tracks]
        xyz, rgb, err, tracks = _apply_track_filter(
            result.xyz,
            result.rgb,
            result.err,
            result.tracks,
            config.min_track_length,
        )
        _log_track_filter_stats(
            f"Chunk {chunk_idx}/{len(chunks)} track filter",
            tracks_before,
            tracks,
            config.min_track_length,
        )
        if xyz.shape[0] == 0:
            lf.log.warn(f"Chunk {chunk_idx}/{len(chunks)} produced no points after filtering.")
            empty_rgb = np.empty((0, 3), dtype=np.uint8)
            count = _save_npz_chunk(chunk_path, xyz, empty_rgb, chunk_metadata)
            chunk_paths.append(chunk_path)
            chunk_counts.append(count)
            lf.log.info(f"Chunk {chunk_idx}/{len(chunks)} saved empty completion marker -> {chunk_path}")
            del result, tracks_before, xyz, rgb, err, tracks
            gc.collect()
            continue

        if max_points_per_batch > 0:
            xyz, rgb, err, tracks = _apply_point_cap(
                xyz,
                rgb,
                err,
                tracks,
                int(max_points_per_batch),
                config.seed + chunk_idx,
            )

        count = _save_npz_chunk(chunk_path, xyz, rgb, chunk_metadata)
        chunk_paths.append(chunk_path)
        chunk_counts.append(count)
        lf.log.info(f"Chunk {chunk_idx}/{len(chunks)} saved {count:,} points -> {chunk_path}")

        del result, tracks_before, xyz, rgb, err, tracks
        gc.collect()

    if not chunk_paths:
        raise RuntimeError("No points remain after chunked densification.")

    if progress_callback:
        progress_callback(94.0, "Merging chunked point output...")
    output_path = config.output_path
    tmp_output_path = output_path + ".tmp"
    output_count, total_count = _write_ply_from_npz_chunks(
        tmp_output_path,
        chunk_paths,
        config.max_points,
        config.seed,
    )
    os.replace(tmp_output_path, output_path)
    lf.log.info(
        f"Chunked dense reconstruction finished: wrote {output_count:,}/{total_count:,} "
        f"points from {len(chunk_paths)} chunks -> {output_path}"
    )
    lf.log.info(
        "Chunk point counts: "
        + ", ".join(f"{idx + 1}:{count:,}" for idx, count in enumerate(chunk_counts))
    )
    if progress_callback:
        progress_callback(100.0, f"Done! {output_count:,} points")
    return 0


def dense_init(
    args,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    debug_state=None,
    cancel_requested: Optional[Callable[[], bool]] = None,
) -> int:
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
        min_track_length=args.min_track_length,
        no_filter=args.no_filter,
        seed=args.seed,
        viz_interval=0,
        prefetch_packages=args.prefetch_packages,
        pack_workers=args.pack_workers,
    )

    if int(getattr(args, "chunked_batches", 1)) > 1:
        return _run_dense_pipeline_chunked(
            records,
            refs_local,
            nn_table,
            config,
            int(args.chunked_batches),
            int(getattr(args, "max_points_per_batch", 0)),
            bool(getattr(args, "resume_chunks", False)),
            progress_callback=progress_callback,
            debug_state=debug_state,
            cancel_requested=cancel_requested,
        )

    from .core.pipeline import PipelineCancelled, run_dense_pipeline

    try:
        result = run_dense_pipeline(
            records,
            refs_local,
            nn_table,
            config,
            progress_callback=progress_callback,
            on_sequential_viz=None,
            debug_state=debug_state,
            cancel_requested=cancel_requested,
        )
    except PipelineCancelled:
        if progress_callback:
            progress_callback(0.0, "Cancelled")
        return 2
    if _cancel_requested(cancel_requested):
        if progress_callback:
            progress_callback(0.0, "Cancelled")
        return 2

    tracks = getattr(result, "tracks", None)
    if tracks is not None:
        tracks_before = [list(track) for track in tracks]
        xyz, rgb, err, tracks = _apply_track_filter(
            result.xyz,
            result.rgb,
            result.err,
            tracks,
            args.min_track_length,
        )
        _log_track_filter_stats("COLMAP track filter", tracks_before, tracks, args.min_track_length)
    else:
        xyz, rgb, err = result.xyz, result.rgb, result.err
    if xyz.shape[0] == 0:
        raise RuntimeError("No points remain after track-length filtering.")
    xyz, rgb, err, tracks = _apply_point_cap(xyz, rgb, err, tracks, args.max_points, args.seed)
    if progress_callback:
        progress_callback(95.0, "Writing output...")
    _write_output(config.output_path, xyz, rgb, err)
    lf.log.info(f"Dense reconstruction finished: {xyz.shape[0]:,} points -> {config.output_path}")
    if progress_callback:
        progress_callback(100.0, f"Done! {xyz.shape[0]:,} points")
    return 0


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
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        R = np.asarray(node.camera_R, dtype=np.float32)
        t = np.asarray(node.camera_T, dtype=np.float32).reshape(3, 1)
        P = K @ np.concatenate([R, t], axis=1)
        C = (-R.T @ t).reshape(3)
        records.append(
            CameraRecord(
                uid=node.camera_uid,
                image_path=node.image_path,
                mask_path=(node.mask_path if getattr(node, "has_mask", False) else None),
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
    debug_state=None,
    cancel_requested: Optional[Callable[[], bool]] = None,
) -> Tuple[int, Optional[str]]:
    np.random.seed(config.seed)
    if progress_callback:
        progress_callback(2.0, "Extracting camera data from scene...")

    records = extract_cameras_from_lfs(camera_nodes)
    if not config.use_masks:
        for r in records:
            r.mask_path = None
    if len(records) < 2:
        return 1, "Need at least 2 cameras for dense initialization"

    flat_poses = _flat_pose_stack(records)
    num_refs = int(round(config.num_refs * len(records))) if config.num_refs <= 1.0 else int(config.num_refs)
    refs_local = select_cameras_kcenters(flat_poses, max(1, num_refs))
    effective_nns = _effective_neighbor_count(config.nns_per_ref, len(records))
    if effective_nns < 1:
        return 1, "Need at least 2 cameras for dense initialization"
    if effective_nns != int(config.nns_per_ref):
        lf.log.info(
            "Clamping neighbors per reference from "
            f"{config.nns_per_ref} to {effective_nns} for {len(records)} ROI cameras"
        )
    nn_table = nearest_neighbors(flat_poses, effective_nns)

    lf.log.info(f"Prepared {len(records)} cameras (refs={len(refs_local)})")

    from .core.pipeline import PipelineCancelled, run_dense_pipeline

    try:
        result = run_dense_pipeline(
            records,
            refs_local,
            nn_table,
            config,
            progress_callback=progress_callback,
            on_sequential_viz=on_sequential_viz,
            debug_state=debug_state,
            cancel_requested=cancel_requested,
        )
    except PipelineCancelled:
        return 2, "Cancelled"
    except RuntimeError as exc:
        return 1, str(exc)
    if _cancel_requested(cancel_requested):
        return 2, "Cancelled"

    tracks_before = [list(track) for track in result.tracks]
    xyz, rgb, err, tracks = _apply_track_filter(
        result.xyz,
        result.rgb,
        result.err,
        result.tracks,
        config.min_track_length,
    )
    _log_track_filter_stats("COLMAP track filter", tracks_before, tracks, config.min_track_length)
    if xyz.shape[0] == 0:
        return 1, "No points remain after track-length filtering."

    xyz, rgb, err, tracks = _apply_point_cap(xyz, rgb, err, tracks, config.max_points, config.seed)

    # Voxel-based distance filtering for uniform point distribution
    if config.voxel_size > 0.0:
        if progress_callback:
            progress_callback(93.0, "Applying distance filter...")
        xyz, rgb, err, tracks = _voxel_select_track_preserving(xyz, rgb, err, tracks, config.voxel_size)
        lf.log.info(f"Distance filter ({config.voxel_size:.4f}): {xyz.shape[0]:,} points remaining")

    if progress_callback:
        progress_callback(95.0, "Writing COLMAP sparse output...")
    write_sparse_model_bin(config.output_path, records, xyz, to_uint8_rgb(rgb), err, tracks)
    lf.log.info(f"Dense sparse model saved to {config.output_path} ({xyz.shape[0]:,} points)")
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
    ap.add_argument(
        "--chunked_batches",
        type=int,
        default=1,
        help=(
            "Split selected reference views into this many RAM-safer chunks. "
            "Neighbors remain global; <=1 keeps the legacy single-pass mode."
        ),
    )
    ap.add_argument(
        "--max_points_per_batch",
        type=int,
        default=0,
        help=(
            "Optional per-chunk point cap before chunk files are merged "
            "(0 = no per-chunk cap; final --max_points still applies globally)."
        ),
    )
    ap.add_argument(
        "--resume_chunks",
        action="store_true",
        help=(
            "Reuse existing chunk_XXXX.npz files only when their saved run "
            "fingerprint matches the current reconstruction, chunk assignment, "
            "and point-generating options."
        ),
    )
    ap.add_argument(
        "--min_track_length",
        type=int,
        default=1,
        help="Minimum generated track length to keep (0 = disabled)",
    )
    ap.add_argument(
        "--prefetch_packages",
        type=int,
        default=8,
        help="Approximate total reference packages prefetched by threaded pack workers",
    )
    ap.add_argument(
        "--pack_workers",
        type=int,
        default=4,
        help="Number of threads used to pack reference packages",
    )
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    return ap


def _cli_progress_callback() -> Callable[[float, str], None]:
    last = {"pct": None, "msg": None, "matching_bucket": -1, "matching_time": 0.0}

    def _report(pct: float, msg: str) -> None:
        pct_f = float(pct)
        msg_s = str(msg)
        is_matching = msg_s.startswith("Matching ")
        if is_matching:
            now = time.time()
            bucket = int(pct_f)
            should_print = (
                bucket > int(last["matching_bucket"])
                or now - float(last["matching_time"]) >= 10.0
                or pct_f >= 89.9
            )
            if not should_print:
                return
            last["matching_bucket"] = bucket
            last["matching_time"] = now
        if last["pct"] == pct_f and last["msg"] == msg_s:
            return
        last["pct"] = pct_f
        last["msg"] = msg_s
        print(f"[{pct_f:6.2f}%] {msg_s}", flush=True)

    return _report


if __name__ == "__main__":
    cli_args = build_argparser().parse_args()
    raise SystemExit(dense_init(cli_args, progress_callback=_cli_progress_callback()))
