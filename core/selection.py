"""Camera selection utilities."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pycolmap


def select_cameras_by_visibility(rec: pycolmap.Reconstruction, k: int) -> List[int]:
    """Select cameras maximizing unique 3D point coverage."""
    if not rec.points3D:
        raise ValueError("Visibility-based selection requires a sparse point cloud.")
    img_to_pts: Dict[int, List[int]] = {
        img.image_id: [p.point3D_id for p in img.points2D if p.has_point3D() and p.point3D_id != -1]
        for img in rec.images.values()
    }
    k = min(k, len(img_to_pts))
    selected_cams: List[int] = []
    covered_pts = set()
    scores = {iid: len(pids) for iid, pids in img_to_pts.items()}
    for _ in range(k):
        if not scores:
            break
        best_cam = max(scores, key=scores.get)
        selected_cams.append(best_cam)
        newly_covered = set(img_to_pts[best_cam]) - covered_pts
        covered_pts.update(newly_covered)
        del scores[best_cam]
        for cam_id, cam_pts in img_to_pts.items():
            if cam_id in scores:
                scores[cam_id] = len(set(cam_pts) - covered_pts)
    return sorted(selected_cams)


def select_cameras_kcenters(flat_poses: np.ndarray, k: int) -> List[int]:
    """k-centers selection using normalized camera poses."""
    X = np.asarray(flat_poses, dtype=np.float32)
    n = X.shape[0]
    k = max(1, min(int(k), n))
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True) + 1e-8
    Xn = (X - mu) / sigma
    first = int(np.argmax(np.einsum("nd,nd->n", Xn, Xn)))
    centers = [first]
    dist = np.linalg.norm(Xn - Xn[first], axis=1)
    dist[first] = -np.inf
    for _ in range(1, k):
        nxt = int(np.argmax(dist))
        centers.append(nxt)
        d = np.linalg.norm(Xn - Xn[nxt], axis=1)
        dist = np.minimum(dist, d)
        dist[nxt] = -np.inf
    return sorted(centers)


def nearest_neighbors(flat_poses: np.ndarray, k: int) -> np.ndarray:
    """Compute Euclidean nearest neighbors across camera poses."""
    import torch

    matrix = torch.from_numpy(flat_poses.astype(np.float32))
    with torch.no_grad():
        dist = torch.cdist(matrix, matrix, p=2)
        dist.fill_diagonal_(float("inf"))
        _, idx = torch.topk(dist, k, largest=False, dim=1)
    return idx.cpu().numpy()
