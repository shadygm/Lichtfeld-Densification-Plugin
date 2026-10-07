"""Point cloud writers."""
from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..cameras.models import CameraRecord
from .tracks import ObservationTracks, observation_tracks as numeric_tracks


def ensure_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


_IMAGE_POINT_DTYPE = np.dtype([("x", "<f8"), ("y", "<f8"), ("point_id", "<i8")])
_TRACK_INDEX_DTYPE = np.dtype([("image_id", "<i4"), ("point2d_index", "<i4")])


@dataclass
class _IndexedTracks:
    lengths: np.ndarray
    offsets: np.ndarray
    entries: np.ndarray


def _rotation_matrix_to_qvec(R: np.ndarray) -> np.ndarray:
    """Return COLMAP qvec [qw, qx, qy, qz] from a world-to-camera R."""

    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    K = np.array(
        [
            [R[0, 0] - R[1, 1] - R[2, 2], 0.0, 0.0, 0.0],
            [R[1, 0] + R[0, 1], R[1, 1] - R[0, 0] - R[2, 2], 0.0, 0.0],
            [R[2, 0] + R[0, 2], R[2, 1] + R[1, 2], R[2, 2] - R[0, 0] - R[1, 1], 0.0],
            [R[1, 2] - R[2, 1], R[2, 0] - R[0, 2], R[0, 1] - R[1, 0], R[0, 0] + R[1, 1] + R[2, 2]],
        ],
        dtype=np.float64,
    )
    K /= 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec / max(np.linalg.norm(qvec), 1.0e-12)


def _camera_params_from_record(record: CameraRecord) -> Tuple[int, int, int, List[float]]:
    # COLMAP camera model id 1 is PINHOLE with fx, fy, cx, cy.
    K = np.asarray(record.K, dtype=np.float64).reshape(3, 3)
    return (
        int(record.uid),
        1,
        int(record.width),
        int(record.height),
        [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])],
    )


def write_cameras_bin(path_out: str, camera_records: Sequence[CameraRecord]) -> None:
    ensure_dir(path_out)
    with open(path_out, "wb") as f:
        f.write(struct.pack("<Q", len(camera_records)))
        for record in camera_records:
            camera_id, model_id, width, height, params = _camera_params_from_record(record)
            f.write(struct.pack("<iiQQ", camera_id, model_id, width, height))
            f.write(struct.pack("<" + "d" * len(params), *params))


def write_images_bin(
    path_out: str,
    camera_records: Sequence[CameraRecord],
    image_points2d: Dict[int, Sequence[Tuple[float, float, int]]],
) -> None:
    ensure_dir(path_out)
    with open(path_out, "wb") as f:
        f.write(struct.pack("<Q", len(camera_records)))
        for record in camera_records:
            image_id = int(record.uid)
            qvec = _rotation_matrix_to_qvec(record.R)
            tvec = np.asarray(record.t, dtype=np.float64).reshape(3)
            camera_id = image_id
            f.write(struct.pack("<i", image_id))
            f.write(struct.pack("<dddd", *[float(v) for v in qvec]))
            f.write(struct.pack("<ddd", *[float(v) for v in tvec]))
            f.write(struct.pack("<i", camera_id))
            name = os.path.basename(record.image_path or f"{image_id}.png")
            f.write(name.encode("utf-8") + b"\x00")
            points2d = image_points2d.get(image_id, ())
            f.write(struct.pack("<Q", len(points2d)))
            if isinstance(points2d, np.ndarray) and points2d.dtype == _IMAGE_POINT_DTYPE:
                points2d.tofile(f)
            else:
                for x, y, point3d_id in points2d:
                    f.write(struct.pack("<ddq", float(x), float(y), int(point3d_id)))


def write_points3D_bin(
    path_out: str,
    xyz: np.ndarray,
    rgb_uint8: np.ndarray,
    errors: Optional[np.ndarray] = None,
    tracks: Optional[Sequence[Sequence[Tuple[int, int]]] | _IndexedTracks] = None,
) -> None:
    N = xyz.shape[0]
    if errors is None:
        errors = np.zeros((N,), dtype=np.float32)
    ensure_dir(path_out)
    with open(path_out, "wb") as f:
        f.write(struct.pack("<Q", N))
        for i in range(N):
            f.write(struct.pack("<Q", i + 1))
            f.write(struct.pack("<ddd", float(xyz[i, 0]), float(xyz[i, 1]), float(xyz[i, 2])))
            f.write(struct.pack("<BBB", int(rgb_uint8[i, 0]), int(rgb_uint8[i, 1]), int(rgb_uint8[i, 2])))
            f.write(struct.pack("<d", float(errors[i])))
            if isinstance(tracks, _IndexedTracks):
                f.write(struct.pack("<Q", int(tracks.lengths[i])))
                f.write(tracks.entries[tracks.offsets[i]:tracks.offsets[i + 1]].tobytes())
            else:
                track = tracks[i] if tracks is not None else ()
                f.write(struct.pack("<Q", len(track)))
                for image_id, point2d_idx in track:
                    f.write(struct.pack("<ii", int(image_id), int(point2d_idx)))


def _index_observations(tracks: ObservationTracks, camera_records):
    """Build COLMAP's per-image rows and per-point indices in numeric buffers."""
    tracks.require_observations()
    point_indices = tracks.point_indices()
    known = np.isin(tracks.camera_ids, [record.uid for record in camera_records])
    selected = np.flatnonzero(known)
    points, images = point_indices[selected], tracks.camera_ids[selected]
    if len(selected):
        # Keep the first observation of each image in each point's track.
        order = np.lexsort((images, points))
        first = np.r_[True, (points[order][1:] != points[order][:-1])
                      | (images[order][1:] != images[order][:-1])]
        selected = selected[np.sort(order[first])]
    points = point_indices[selected]
    indexed = ObservationTracks(np.bincount(points, minlength=len(tracks)).astype(np.uint32),
                                 tracks.camera_ids[selected], tracks.pixels[selected])
    pairs = np.empty(len(selected), dtype=_TRACK_INDEX_DTYPE)
    image_points2d = {}
    for camera_id, observations in indexed.camera_groups():
        rows = np.empty(len(observations), dtype=_IMAGE_POINT_DTYPE)
        rows["x"], rows["y"] = indexed.pixels[observations].T
        rows["point_id"] = points[observations] + 1
        image_points2d[camera_id] = rows
        pairs["image_id"][observations] = camera_id
        pairs["point2d_index"][observations] = np.arange(len(observations))
    return image_points2d, _IndexedTracks(indexed.lengths, indexed.offsets, pairs)


def write_sparse_model_bin(
    sparse_dir: str,
    camera_records: Sequence[CameraRecord],
    xyz: np.ndarray,
    rgb_uint8: np.ndarray,
    errors: Optional[np.ndarray],
    observation_tracks: ObservationTracks,
) -> None:
    Path(sparse_dir).mkdir(parents=True, exist_ok=True)
    # Accept legacy rows at this boundary without rebuilding them internally.
    tracks = numeric_tracks(observation_tracks)
    if len(tracks) != len(xyz):
        raise ValueError("Each exported point must have one observation track")
    image_points2d, indexed_tracks = _index_observations(tracks, camera_records)

    write_cameras_bin(os.path.join(sparse_dir, "cameras.bin"), camera_records)
    write_images_bin(os.path.join(sparse_dir, "images.bin"), camera_records, image_points2d)
    write_points3D_bin(
        os.path.join(sparse_dir, "points3D.bin"),
        xyz,
        rgb_uint8,
        errors,
        indexed_tracks,
    )


def read_points3D_bin_point_cloud(
    path_in: str,
    min_track_length: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xyz_parts: List[Tuple[float, float, float]] = []
    rgb_parts: List[Tuple[int, int, int]] = []
    track_lengths: List[int] = []
    min_track = max(0, int(min_track_length))
    with open(path_in, "rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_points):
            f.read(8)  # point3D id
            x, y, z = struct.unpack("<ddd", f.read(24))
            r, g, b = struct.unpack("<BBB", f.read(3))
            f.read(8)  # error
            track_len = struct.unpack("<Q", f.read(8))[0]
            f.read(8 * track_len)
            if min_track == 0 or track_len >= min_track:
                xyz_parts.append((x, y, z))
                rgb_parts.append((r, g, b))
                track_lengths.append(int(track_len))
    xyz = np.asarray(xyz_parts, dtype=np.float32).reshape((-1, 3))
    rgb = np.asarray(rgb_parts, dtype=np.uint8).reshape((-1, 3))
    tracks = np.asarray(track_lengths, dtype=np.int32)
    return xyz, rgb, tracks


def write_ply_vertices(file_obj, xyz: np.ndarray, rgb_uint8: np.ndarray) -> None:
    """Write packed binary vertices in bulk, including intermediate previews."""
    packed = np.empty(len(xyz), dtype=[
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    for column, name in enumerate(("x", "y", "z")):
        packed[name] = xyz[:, column]
    for column, name in enumerate(("red", "green", "blue")):
        packed[name] = rgb_uint8[:, column]
    packed.tofile(file_obj)


def write_ply(path_out: str, xyz: np.ndarray, rgb_uint8: np.ndarray) -> None:
    ensure_dir(path_out)
    N = xyz.shape[0]
    header = f"""ply
format binary_little_endian 1.0
element vertex {N}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    with open(path_out, "wb") as f:
        f.write(header.encode("ascii"))
        write_ply_vertices(f, xyz, rgb_uint8)
