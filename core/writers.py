"""Point cloud writers."""
from __future__ import annotations

import os
import struct
from typing import Optional

import numpy as np


def ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def write_points3D_bin(path_out: str, xyz: np.ndarray, rgb_uint8: np.ndarray,
                        errors: Optional[np.ndarray] = None) -> None:
    N = xyz.shape[0]
    if errors is None:
        errors = np.zeros((N,), dtype=np.float64)
    with open(path_out, "wb") as f:
        f.write(struct.pack("<Q", N))
        for i in range(N):
            f.write(struct.pack("<Q", i + 1))
            f.write(struct.pack("<ddd", float(xyz[i, 0]), float(xyz[i, 1]), float(xyz[i, 2])))
            f.write(struct.pack("<BBB", int(rgb_uint8[i, 0]), int(rgb_uint8[i, 1]), int(rgb_uint8[i, 2])))
            f.write(struct.pack("<d", float(errors[i])))


def write_ply(path_out: str, xyz: np.ndarray, rgb_uint8: np.ndarray) -> None:
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
        for i in range(N):
            f.write(struct.pack("<fff", float(xyz[i, 0]), float(xyz[i, 1]), float(xyz[i, 2])))
            f.write(struct.pack("BBB", int(rgb_uint8[i, 0]), int(rgb_uint8[i, 1]), int(rgb_uint8[i, 2])))
