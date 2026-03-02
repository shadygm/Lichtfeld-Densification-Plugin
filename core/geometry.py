"""Geometry helpers for densification pipeline."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pycolmap


def K_from_camera(cam: pycolmap.Camera) -> np.ndarray:
    K = np.eye(3, dtype=np.float32)
    model = str(cam.model.name).upper()
    p = np.asarray(cam.params, dtype=np.float32)
    w, h = cam.width, cam.height
    if "PINHOLE" in model and "SIMPLE" not in model:
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    elif "SIMPLE_PINHOLE" in model:
        fx = fy = p[0]
        cx, cy = p[1], p[2]
    elif "SIMPLE_RADIAL" in model or model == "RADIAL":
        fx = fy = p[0]
        cx, cy = p[1], p[2]
    elif "OPENCV" in model or "FISHEYE" in model:
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    else:
        fx = fy = p[0]
        cx = p[1] if len(p) > 1 else w / 2
        cy = p[2] if len(p) > 2 else h / 2
    K[0, 0], K[1, 1], K[0, 2], K[1, 2] = fx, fy, cx, cy
    return K


def pose_world2cam(im: pycolmap.Image) -> Tuple[np.ndarray, np.ndarray]:
    if hasattr(im, "cam_from_world"):
        cfw = im.cam_from_world
        cfw = cfw() if callable(cfw) else cfw
        R = np.asarray(cfw.rotation.matrix(), dtype=np.float32)
        t = np.asarray(cfw.translation, dtype=np.float32).reshape(3, 1)
    else:
        R = im.qvec.to_rotation_matrix()
        t = np.asarray(im.tvec, dtype=np.float32).reshape(3, 1)
    return R, t


def P_from_KRt(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return K @ np.concatenate([R, t], axis=1)


def cam_center_world(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (-R.T @ t).reshape(3)


def skew(v: np.ndarray) -> np.ndarray:
    vx, vy, vz = v.flatten()
    return np.array([[0, -vz, vy], [vz, 0, -vx], [-vy, vx, 0]], dtype=np.float32)


def dlt_triangulate_batch(P1, P2, uv1, uv2):
    N = uv1.shape[0]
    if N == 0:
        return np.zeros((0,4), dtype=np.float32)

    p10, p11, p12 = P1[0], P1[1], P1[2]
    p20, p21, p22 = P2[0], P2[1], P2[2]

    u1 = uv1[:, 0:1]
    v1 = uv1[:, 1:2]
    u2 = uv2[:, 0:1]
    v2 = uv2[:, 1:2]

    A = np.empty((N,4,4), dtype=np.float32)
    A[:,0,:] = u1 * p12 - p10
    A[:,1,:] = v1 * p12 - p11
    A[:,2,:] = u2 * p22 - p20
    A[:,3,:] = v2 * p22 - p21

    if N == 1:
        # Avoid batch SVD collapse
        _, _, Vt = np.linalg.svd(A[0])
        Xh = Vt[-1]
        w = Xh[3] if abs(Xh[3]) > 1e-12 else 1e-12
        return (Xh / w)[None, :]

    _, _, Vt = np.linalg.svd(A)
    Xh = Vt[:, -1, :]
    w = np.where(np.abs(Xh[:,3:4]) < 1e-12, 1e-12, Xh[:,3:4])
    return Xh / w
    


def reprojection_errors(P: np.ndarray, X: np.ndarray, uv: np.ndarray) -> np.ndarray:
    # Project: (N,4) @ (4,3) -> (N,3)
    proj = X @ P.T

    # Normalize by depth
    z = np.maximum(proj[:, 2], 1e-12)
    u_pred = proj[:, 0] / z
    v_pred = proj[:, 1] / z

    # Reprojection residuals
    du = u_pred - uv[:, 0]
    dv = v_pred - uv[:, 1]

    return np.sqrt(du * du + dv * dv)


def cheirality_mask(P: np.ndarray, X: np.ndarray) -> np.ndarray:
    Xh = X.T
    z = (P @ Xh)[2, :]
    return z > 0.0


def parallax_mask(C1: np.ndarray, C2: np.ndarray, X: np.ndarray, min_deg: float = 0.5) -> np.ndarray:
    v1 = X[:, :3] - C1.reshape(1, 3)
    v2 = X[:, :3] - C2.reshape(1, 3)
    v1 /= np.linalg.norm(v1, axis=1, keepdims=True) + 1e-12
    v2 /= np.linalg.norm(v2, axis=1, keepdims=True) + 1e-12
    ang = np.degrees(np.arccos(np.clip(np.sum(v1 * v2, axis=1), -1.0, 1.0)))
    return ang >= float(min_deg)


def fundamental_from_world2cam(K1: np.ndarray, R1: np.ndarray, t1: np.ndarray,
                               K2: np.ndarray, R2: np.ndarray, t2: np.ndarray) -> np.ndarray:
    R = R2 @ R1.T
    t = (t2 - R @ t1).reshape(3)
    E = skew(t) @ R
    K1i = np.linalg.inv(K1)
    K2i = np.linalg.inv(K2)
    F = K2i.T @ E @ K1i
    return F


def sampson_error(F: np.ndarray, uv1: np.ndarray, uv2: np.ndarray) -> np.ndarray:
    N = uv1.shape[0]
    x1 = np.concatenate([uv1, np.ones((N, 1))], axis=1)
    x2 = np.concatenate([uv2, np.ones((N, 1))], axis=1)
    Fx1 = (F @ x1.T).T
    Ftx2 = (F.T @ x2.T).T
    x2Fx1 = np.sum(x2 * Fx1, axis=1)
    den = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2 + 1e-12
    return (x2Fx1 ** 2) / den
