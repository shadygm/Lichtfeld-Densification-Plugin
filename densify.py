#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dense, accurate COLMAP pointcloud initializer (EDGS-inspired, RoMa v2-driven)
with pipelined GPU (matching) <> CPU (filtering/triangulation) overlap.

Pipeline
--------
- Main thread (producer): for each reference view, runs RoMa v2 on GPU against its K nearest
  neighbors, collects all (warp, certainty) maps, converts them to CPU tensors, and pushes
  a 'job' (one per reference) onto a queue.
- CPU worker thread (consumer): pulls jobs and performs:
    * argmax aggregation across neighbors
    * EDGS-style sampling (cap + multinomial + coverage)
    * per-neighbor grouping
    * optional Sampson gating
    * DLT triangulation, reprojection/cheirality/parallax filters
    * bilinear color reading from the reference image
  It returns (xyz, rgb, err) arrays per job on a results queue.

Effect
------
While the GPU is busy computing RoMa v2 for reference r+1, the CPU processes and triangulates
the previously enqueued reference r, keeping both devices utilized.

RoMa v2 Settings
----------------
- "precise": H_lr=800, W_lr=800, H_hr=1280, W_hr=1280, bidirectional=True (highest quality)
- "high": H_lr=640, W_lr=640, H_hr=960, W_hr=960, bidirectional=True (high quality, less VRAM)
- "base": H_lr=640, W_lr=640, no high-res refinement, bidirectional=False
- "fast": H_lr=512, W_lr=512, no high-res refinement, bidirectional=False (default)
- "turbo": H_lr=320, W_lr=320, no high-res refinement, bidirectional=False (fastest)

Dependencies
------------
pip install pycolmap romav2 Pillow numpy scipy tqdm open3d  # (open3d optional for --viz)
"""

from __future__ import annotations

import os, sys, argparse, struct, time, threading
from queue import Queue
from typing import Dict, Tuple, List, Optional, Callable
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add local RoMaV2 to path
import pathlib as _pathlib
_THIS_DIR = _pathlib.Path(__file__).parent.resolve()
_ROMAV2_SRC = _THIS_DIR / "RoMaV2" / "src"
if str(_ROMAV2_SRC) not in sys.path:
    sys.path.insert(0, str(_ROMAV2_SRC))

# --------- IO / deps ----------
import pycolmap
from scipy.spatial.distance import cdist

# RoMa v2 (romav2)
import torch
import torch.nn.functional as F
from romav2 import RoMaV2

# LichtFeld logging
import lichtfeld as lf


# ==========================
# Small utilities
# =========================="

def image_dir(scene_root: str, preferred: str) -> str:
    cand = os.path.join(scene_root, preferred)
    if os.path.isdir(cand): return cand
    for alt in ["images_4", "images_2", "images_8", "images"]:
        p = os.path.join(scene_root, alt)
        if os.path.isdir(p): return p
    raise FileNotFoundError("Could not locate an images directory under scene_root.")

def to_uint8_rgb(arr_float01: np.ndarray) -> np.ndarray:
    return np.clip(np.round(arr_float01 * 255.0), 0, 255).astype(np.uint8)

def find_image(root: str, name: str) -> str:
    p = os.path.join(root, name)
    if os.path.isfile(p): return p
    q = os.path.join(root, os.path.basename(name))
    if os.path.isfile(q): return q
    raise FileNotFoundError(f"Image '{name}' not found under {root}")

@lru_cache(maxsize=4096)
def load_rgb_resized(path: str, size: tuple[int,int]) -> Image.Image:
    im = Image.open(path).convert("RGB")
    if im.size != size:
        im = im.resize(size, Image.BILINEAR)
    return im


# ==========================
# COLMAP helpers
# ==========================

def load_reconstruction(sparse_dir: str):
    rec = pycolmap.Reconstruction(sparse_dir)
    return rec, rec.cameras, rec.images

def K_from_camera(cam: pycolmap.Camera) -> np.ndarray:
    K = np.eye(3, dtype=np.float64)
    model = str(cam.model.name).upper()
    p = np.asarray(cam.params, dtype=np.float64)
    w, h = cam.width, cam.height
    if "PINHOLE" in model and "SIMPLE" not in model:
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    elif "SIMPLE_PINHOLE" in model:
        fx = fy = p[0]; cx, cy = p[1], p[2]
    elif "SIMPLE_RADIAL" in model or model == "RADIAL":
        fx = fy = p[0]; cx, cy = p[1], p[2]
    elif "OPENCV" in model or "FISHEYE" in model:
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    else:
        fx = fy = p[0]
        cx = p[1] if len(p) > 1 else w/2
        cy = p[2] if len(p) > 2 else h/2
    K[0,0], K[1,1], K[0,2], K[1,2] = fx, fy, cx, cy
    return K

def pose_world2cam(im: pycolmap.Image) -> Tuple[np.ndarray, np.ndarray]:
    if hasattr(im, "cam_from_world"):
        cfw = im.cam_from_world
        cfw = cfw() if callable(cfw) else cfw
        R = np.asarray(cfw.rotation.matrix(), dtype=np.float64)
        t = np.asarray(cfw.translation, dtype=np.float64).reshape(3,1)
    else:
        R = im.qvec.to_rotation_matrix()
        t = np.asarray(im.tvec, dtype=np.float64).reshape(3,1)
    return R, t

def P_from_KRt(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return K @ np.concatenate([R, t], axis=1)  # 3x4

def cam_center_world(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (-R.T @ t).reshape(3)

def skew(v: np.ndarray) -> np.ndarray:
    vx, vy, vz = v.flatten()
    return np.array([[0, -vz, vy], [vz, 0, -vx], [-vy, vx, 0]], dtype=np.float64)


# ==========================
# Neighbor selection
# ==========================

def select_cameras_by_visibility(rec: pycolmap.Reconstruction, K: int) -> List[int]:
    if not rec.points3D:
        raise ValueError("Visibility-based selection requires a sparse point cloud.")
    img_to_pts = {
        img.image_id: [p.point3D_id for p in img.points2D if p.has_point3D() and p.point3D_id != -1]
        for img in rec.images.values()
    }
    K = min(K, len(img_to_pts))
    selected_cams = []
    covered_pts = set()
    scores = {iid: len(pids) for iid, pids in img_to_pts.items()}
    for _ in range(K):
        if not scores: break
        best_cam = max(scores, key=scores.get)
        selected_cams.append(best_cam)
        newly_covered = set(img_to_pts[best_cam]) - covered_pts
        covered_pts.update(newly_covered)
        del scores[best_cam]
        for cam_id, cam_pts in img_to_pts.items():
            if cam_id in scores:
                scores[cam_id] = len(set(cam_pts) - covered_pts)
    return sorted(selected_cams)

def select_cameras_kcenters(flat_poses: np.ndarray, K: int) -> List[int]:
    X = np.asarray(flat_poses, dtype=np.float32)
    N = X.shape[0]; K = max(1, min(int(K), N))
    mu = X.mean(axis=0, keepdims=True); sigma = X.std(axis=0, keepdims=True) + 1e-8
    Xn = (X - mu) / sigma
    first = int(np.argmax(np.einsum("nd,nd->n", Xn, Xn)))
    centers = [first]
    dist = np.linalg.norm(Xn - Xn[first], axis=1); dist[first] = -np.inf
    for _ in range(1, K):
        nxt = int(np.argmax(dist)); centers.append(nxt)
        d = np.linalg.norm(Xn - Xn[nxt], axis=1)
        dist = np.minimum(dist, d); dist[nxt] = -np.inf
    return sorted(centers)

def nearest_neighbors(flat_poses: np.ndarray, k: int) -> np.ndarray:
    import torch as _torch
    M = _torch.from_numpy(flat_poses.astype(np.float32))
    with _torch.no_grad():
        D = _torch.cdist(M, M, p=2)
        D.fill_diagonal_(float("inf"))
        _, idx = _torch.topk(D, k, largest=False, dim=1)
    return idx.numpy()  # [N,k]


# ==========================
# RoMa v2 wrapper (GPU)
# ==========================

class RomaMatcher:
    """Wrapper around RoMaV2 for dense matching.
    
    RoMaV2 API:
    - model.match(img_A, img_B) -> dict with warp_AB, overlap_AB, precision_AB, etc.
    - model.sample(preds, num_corresp) -> (matches, overlaps, precision_AB, precision_BA)
    - model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B) -> (kptsA, kptsB)
    
    Settings available via model.apply_setting(setting):
    - "precise": H_lr=800, W_lr=800, H_hr=1280, W_hr=1280, bidirectional=True
    - "high": H_lr=640, W_lr=640, H_hr=960, W_hr=960, bidirectional=True (custom setting)
    - "base": H_lr=640, W_lr=640, H_hr=None, W_hr=None, bidirectional=False
    - "fast": H_lr=512, W_lr=512, H_hr=None, W_lr=None, bidirectional=False
    - "turbo": H_lr=320, W_lr=320, H_hr=None, W_hr=None, bidirectional=False
    """
    def __init__(self, device="cuda", mode="outdoor", setting="fast"):
        self.device = torch.device(device)
        # RoMaV2 doesn't have indoor/outdoor distinction - it's a unified model
        # mode is kept for API compatibility but ignored
        torch.set_float32_matmul_precision('highest')  # Required by RoMaV2
        # Disable torch.compile to avoid functorch dependency issues
        self.model = RoMaV2(RoMaV2.Cfg(compile=False))
        
        # Apply setting - "high" is a custom setting not in RoMaV2's built-in options
        if setting == "high":
            # Custom high-quality setting: between "precise" and "base"
            # Uses bidirectional matching with moderate resolution
            self.model.H_lr = 640
            self.model.W_lr = 640
            self.model.H_hr = 960
            self.model.W_hr = 960
            self.model.bidirectional = True
        else:
            self.model.apply_setting(setting)
        
        self.model.to(self.device)
        self.model.eval()
        # For compatibility with original code expectations
        self.sample_thresh = 0.9  # cap for certainty in sampling
        # Store resolution for reference
        self.w_resized = self.model.W_lr
        self.h_resized = self.model.H_lr
        lf.log.info(f"RoMaV2 initialized with setting='{setting}' (H_lr={self.model.H_lr}, W_lr={self.model.W_lr}) on {device}")

    def _preprocess_image_batch(self, images: List[Image.Image]) -> torch.Tensor:
        """Convert PIL images to normalized tensor batch on GPU.
        
        Args:
            images: List of PIL Images
            
        Returns:
            Tensor [B, 3, H, W] normalized to [0, 1] on GPU
        """
        import torchvision.transforms.functional as TF
        tensors = [TF.to_tensor(img).to(self.device) for img in images]
        return torch.stack(tensors, dim=0)

    @torch.inference_mode()
    def match_grids(self, imA: Image.Image, imB: Image.Image):
        """Match two images and return warp and certainty maps.
        
        Args:
            imA: Reference image (PIL Image)
            imB: Neighbor image (PIL Image)
            
        Returns:
            warp: [H, W, 4] tensor with (xA, yA, xB, yB) in [-1, 1] normalized coords
            cert: [H, W] tensor with overlap/certainty values in [0, 1]
        """
        # Ensure matmul precision is set (RoMa v2 requirement)
        torch.set_float32_matmul_precision('highest')
        # RoMaV2.match() accepts ImageLike (paths, PIL Images, tensors, numpy arrays)
        preds = self.model.match(imA, imB)
        
        # Extract warp A->B and overlap/certainty
        # warp_AB is [B, H, W, 2] - coordinates in B for each pixel in A
        # overlap_AB is [B, H, W, 1] - overlap/certainty map (note trailing dim)
        warp_AB = preds["warp_AB"][0]  # [H, W, 2] - coords in B
        overlap_AB = preds["overlap_AB"][0].squeeze(-1)  # [H, W] - squeeze the trailing dim
        
        H, W = overlap_AB.shape
        
        # Create grid for A coordinates (normalized [-1, 1])
        yy = torch.linspace(-1 + 1/H, 1 - 1/H, H, device=self.device)
        xx = torch.linspace(-1 + 1/W, 1 - 1/W, W, device=self.device)
        yy, xx = torch.meshgrid(yy, xx, indexing="ij")
        gridA = torch.stack([xx, yy], dim=-1)  # [H, W, 2]
        
        # Combine to create full warp: (xA, yA, xB, yB)
        warp = torch.cat([gridA, warp_AB], dim=-1)  # [H, W, 4]
        
        return warp.contiguous(), overlap_AB.contiguous()  # [H,W,4], [H,W]

    @torch.inference_mode()
    def match_grids_batch(self, imA: Image.Image, imB_list: List[Image.Image]):
        """Batch match reference image against multiple neighbor images.
        
        Args:
            imA: Reference image (PIL Image)
            imB_list: List of neighbor images (PIL Images)
            
        Returns:
            List of (warp, cert) tuples for each neighbor
        """
        if not imB_list:
            return []
        
        results = []
        # Process in smaller batches to avoid OOM
        batch_size = 4 if self.device.type == "cuda" else 2
        
        for i in range(0, len(imB_list), batch_size):
            batch = imB_list[i:i+batch_size]
            for imB in batch:
                warp, cert = self.match_grids(imA, imB)
                results.append((warp, cert))
        
        return results


# ==========================
# Geometry & filters
# ==========================

def dlt_triangulate_batch(P1: np.ndarray, P2: np.ndarray, uv1: np.ndarray, uv2: np.ndarray) -> np.ndarray:
    N = uv1.shape[0]
    X = np.zeros((N,4), dtype=np.float64)
    p10,p11,p12 = P1[0],P1[1],P1[2]
    p20,p21,p22 = P2[0],P2[1],P2[2]
    for i in range(N):
        u1,v1 = uv1[i]; u2,v2 = uv2[i]
        A = np.stack([
            u1*p12 - p10,
            v1*p12 - p11,
            u2*p22 - p20,
            v2*p22 - p21
        ], axis=0)
        _,_,Vt = np.linalg.svd(A)
        Xh = Vt[-1]
        if abs(Xh[3]) < 1e-12: Xh[3] = 1e-12
        X[i] = Xh / Xh[3]
    return X

def reprojection_errors(P: np.ndarray, X: np.ndarray, uv: np.ndarray) -> np.ndarray:
    Xh = X.T
    proj = (P @ Xh)
    proj = proj / np.maximum(1e-12, proj[2:3,:])
    pred = proj[:2,:].T
    return np.linalg.norm(pred - uv, axis=1)

def cheirality_mask(P: np.ndarray, X: np.ndarray) -> np.ndarray:
    Xh = X.T
    z = (P @ Xh)[2,:]
    return (z > 0.0)

def parallax_mask(C1: np.ndarray, C2: np.ndarray, X: np.ndarray, min_deg=0.5) -> np.ndarray:
    v1 = X[:,:3] - C1.reshape(1,3)
    v2 = X[:,:3] - C2.reshape(1,3)
    v1 /= np.linalg.norm(v1, axis=1, keepdims=True) + 1e-12
    v2 /= np.linalg.norm(v2, axis=1, keepdims=True) + 1e-12
    ang = np.degrees(np.arccos(np.clip(np.sum(v1*v2,axis=1), -1.0, 1.0)))
    return ang >= float(min_deg)

def fundamental_from_world2cam(K1,R1,t1,K2,R2,t2) -> np.ndarray:
    R = R2 @ R1.T
    t = (t2 - R @ t1).reshape(3)
    E = skew(t) @ R
    K1i = np.linalg.inv(K1); K2i = np.linalg.inv(K2)
    F = K2i.T @ E @ K1i
    return F

def sampson_error(F: np.ndarray, uv1: np.ndarray, uv2: np.ndarray) -> np.ndarray:
    N = uv1.shape[0]
    x1 = np.concatenate([uv1, np.ones((N,1))], axis=1)
    x2 = np.concatenate([uv2, np.ones((N,1))], axis=1)
    Fx1 = (F @ x1.T).T
    Ftx2 = (F.T @ x2.T).T
    x2Fx1 = np.sum(x2 * Fx1, axis=1)
    den = Fx1[:,0]**2 + Fx1[:,1]**2 + Ftx2[:,0]**2 + Ftx2[:,1]**2 + 1e-12
    return (x2Fx1**2) / den


# ==========================
# EDGS-style sampling (cap + multinomial + coverage fill)
# ==========================

def select_samples_with_coverage(cert_map: torch.Tensor, M: int, cap: float = 0.9,
                                 border: int = 2, tiles: int = 24, no_filter: bool = False) -> np.ndarray:
    """Select sample indices from a certainty map.

    If `no_filter` is True, return the top-`M` pixels by certainty (fast, raw),
    otherwise use EDGS-style mixed sampling with coverage filling.
    """
    cert = cert_map.clone().to("cpu")
    cert = torch.clamp(cert, max=cap)
    H, W = cert.shape
    # Fast path: no_filter returns the top-M by certainty (no coverage constraints)
    if no_filter:
        flat = cert.reshape(-1).numpy()
        if flat.size == 0:
            return np.zeros((0,), dtype=np.int64)
        order = np.argsort(-flat)
        sel = order[:min(M, flat.size)]
        return sel

    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    inside = (xx >= border) & (xx <= W - 1 - border) & (yy >= border) & (yy <= H - 1 - border)
    weights = (cert * inside.float()).reshape(-1)
    s = weights.sum()
    if s <= 0:
        return np.zeros((0,), dtype=np.int64)
    weights = (weights / s).numpy()

    # A slightly more permissive main-sampling fraction (more main samples)
    m_main = int(M * 0.85)
    idx_main = np.random.choice(weights.size, size=min(m_main, weights.size), replace=False, p=weights)

    tile = max(1, W // tiles)
    gx = (xx // tile).reshape(-1).numpy()
    gy = (yy // tile).reshape(-1).numpy()
    bins = gx * 100000 + gy
    order = np.argsort(-weights)  # descending by weight
    seen = set(); idx_cov = []
    for i in order:
        if weights[i] <= 0: break
        b = int(bins[i])
        if b in seen: continue
        seen.add(b); idx_cov.append(i)
        if len(idx_cov) >= M - len(idx_main): break

    sel_idx = np.unique(np.concatenate([idx_main, np.asarray(idx_cov, dtype=np.int64)]))
    return sel_idx


# ==========================
# Writers
# ==========================

def write_points3D_bin(path_out: str, xyz: np.ndarray, rgb_uint8: np.ndarray, errors: Optional[np.ndarray]=None):
    N = xyz.shape[0]
    if errors is None:
        errors = np.zeros((N,), dtype=np.float64)
    with open(path_out, "wb") as f:
        f.write(struct.pack("<Q", N))
        for i in range(N):
            f.write(struct.pack("<Q", i+1))
            f.write(struct.pack("<ddd", float(xyz[i,0]), float(xyz[i,1]), float(xyz[i,2])))


def write_ply(path_out: str, xyz: np.ndarray, rgb_uint8: np.ndarray):
    """Write a PLY point cloud file.
    
    Args:
        path_out: Output path for PLY file
        xyz: [N, 3] float array of 3D positions
        rgb_uint8: [N, 3] uint8 array of RGB colors
    """
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
            f.write(struct.pack("<fff", float(xyz[i,0]), float(xyz[i,1]), float(xyz[i,2])))
            f.write(struct.pack("BBB", int(rgb_uint8[i,0]), int(rgb_uint8[i,1]), int(rgb_uint8[i,2])))


# ==========================
# CPU worker (consumer)
# ==========================

def make_cpu_worker(job_q: Queue, res_q: Queue,
                    K_by, R_by, t_by, P_by, C_by, name_by, size_by,
                    args, matcher_sample_cap: float, no_filter: bool = False):
    """
    job: dict with keys:
        ref_id, nn_ids (list[int]), warp_list (list[torch.Tensor CPU [H,W,4]]),
        cert_list (list[torch.Tensor CPU [H,W]]), imA_np (Himg,Wimg,3 uint8),
        w_match, h_match, wA_cam, hA_cam
    Puts (xyz, rgb, err) per job onto res_q.
    """
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

                # --- Argmax aggregation across neighbors (CPU tensors) ---
                H, W = cert_list[0].shape
                cert_stack = torch.stack(cert_list, dim=0).to(device_cpu)            # [K,H,W]
                best_cert, best_k = torch.max(cert_stack, dim=0)                     # [H,W], [H,W]
                warp_stack = torch.stack(warp_list, dim=0).to(device_cpu)            # [K,H,W,4]

                ys = torch.arange(H, device=device_cpu).unsqueeze(1).expand(H, W)
                xs = torch.arange(W, device=device_cpu).unsqueeze(0).expand(H, W)
                agg = warp_stack[best_k, ys, xs]                                     # [H,W,4]
                agg = agg.reshape(-1, 4).numpy()                                     # (H*W,4)

                # --- EDGS sampling (or raw top-certainty sampling if no_filter) ---
                sel_idx = select_samples_with_coverage(best_cert, args.matches_per_ref,
                                                       cap=matcher_sample_cap, border=2, tiles=24,
                                                       no_filter=no_filter)
                if sel_idx.size == 0:
                    res_q.put((None, None, None))
                    job_q.task_done()
                    continue

                # Selected warps / winner-NN for each sample
                nn_idx_flat = best_k.reshape(-1).numpy()[sel_idx]                    # [S]
                sel = agg[sel_idx]                                                   # [S,4]

                xA = (sel[:,0] + 1.0)*0.5*(w_match-1)
                yA = (sel[:,1] + 1.0)*0.5*(h_match-1)
                xB_norm = sel[:,2]; yB_norm = sel[:,3]  # in [-1,1]

                # --- Colors from reference (bilinear) ---
                hA_img, wA_img = imA_np.shape[0], imA_np.shape[1]
                sxA_img = wA_img / float(w_match)
                syA_img = hA_img / float(h_match)
                xA_img = xA * sxA_img
                yA_img = yA * syA_img
                xa0 = np.clip(np.floor(xA_img).astype(np.int32), 0, wA_img-1)
                ya0 = np.clip(np.floor(yA_img).astype(np.int32), 0, hA_img-1)
                xa1 = np.clip(xa0+1, 0, wA_img-1)
                ya1 = np.clip(ya0+1, 0, hA_img-1)
                wa = (xa1 - xA_img)*(ya1 - yA_img)
                wb = (xA_img - xa0)*(ya1 - yA_img)
                wc = (xa1 - xA_img)*(yA_img - ya0)
                wd = (xA_img - xa0)*(yA_img - ya0)
                Ia = imA_np[ya0, xa0].astype(np.float32)
                Ib = imA_np[ya0, xa1].astype(np.float32)
                Ic = imA_np[ya1, xa0].astype(np.float32)
                Id = imA_np[ya1, xa1].astype(np.float32)
                rgb_ref = (Ia*wa[:,None] + Ib*wb[:,None] + Ic*wc[:,None] + Id*wd[:,None]) / 255.0  # [S,3]

                # --- Scale A to camera intrinsics resolution for triangulation ---
                sxA = wA_cam / float(w_match)
                syA = hA_cam / float(h_match)
                uvA_full = np.stack([xA * sxA, yA * syA], axis=1)  # [S,2]

                # --- Group by neighbor for correct B sizing ---
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

                    xB = (xB_norm[idxs] + 1.0)*0.5*(w_match-1)
                    yB = (yB_norm[idxs] + 1.0)*0.5*(h_match-1)
                    uvB = np.stack([xB * sxB, yB * syB], axis=1)

                    # Optional Sampson pre-triangulation (skipped in no_filter)
                    if (not no_filter) and args.sampson_thresh > 0:
                        F = fundamental_from_world2cam(K_by[ref_id], R_by[ref_id], t_by[ref_id],
                                                       K_by[nbr_id], R_by[nbr_id], t_by[nbr_id])
                        se = sampson_error(F, uvA_full[idxs], uvB)
                        good = (se < float(args.sampson_thresh))
                        if not np.any(good):
                            continue
                        idxs = idxs[good]
                        xB = xB[good]; yB = yB[good]
                        uvB = uvB[good]
                    if idxs.size == 0:
                        continue

                    P1, P2 = P_by[ref_id], P_by[nbr_id]
                    uvA = uvA_full[idxs]
                    Xi = dlt_triangulate_batch(P1, P2, uvA, uvB)     # [Mi,4]

                    err1 = reprojection_errors(P1, Xi, uvA)
                    err2 = reprojection_errors(P2, Xi, uvB)
                    err  = np.maximum(err1, err2)

                    # If no_filter is enabled, accept all triangulated points (except NaNs/infs)
                    if no_filter:
                        # sanity: remove NaN/inf and extremely large reprojection errors
                        finite_mask = np.isfinite(Xi).all(axis=1) & np.isfinite(err)
                        if not np.any(finite_mask):
                            continue
                        keep = finite_mask
                    else:
                        keep = (err <= float(args.reproj_thresh))
                        keep &= cheirality_mask(P1, Xi)
                        keep &= cheirality_mask(P2, Xi)
                        if args.min_parallax_deg > 0:
                            keep &= parallax_mask(C_by[ref_id], C_by[nbr_id], Xi, min_deg=args.min_parallax_deg)

                        if not np.any(keep):
                            continue

                    Xw = Xi[keep][:,:3].astype(np.float64)
                    col = rgb_ref[idxs][keep].astype(np.float32)
                    e   = err[keep].astype(np.float64)

                    job_xyz.append(Xw); job_rgb.append(col); job_err.append(e)

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


# ==========================
# Main pipeline (producer)
# ==========================

def dense_init(args, progress_callback: Optional[Callable[[float, str], None]] = None):
    np.random.seed(args.seed)
    scene_root = os.path.abspath(args.scene_root)
    sparse_dir = os.path.join(scene_root, "sparse", "0")
    images_dir = image_dir(scene_root, args.images_subdir)

    lf.log.debug(f"Scene   : {scene_root}")
    lf.log.debug(f"Sparse  : {sparse_dir}")
    lf.log.debug(f"Images  : {images_dir}")

    rec, cams, imgs = load_reconstruction(sparse_dir)
    img_ids = sorted(list(imgs.keys()))
    lf.log.info(f"Loaded {len(cams)} cameras, {len(imgs)} images.")

    # Camera / pose dicts
    K_by: Dict[int,np.ndarray] = {}
    R_by: Dict[int,np.ndarray] = {}
    t_by: Dict[int,np.ndarray] = {}
    P_by: Dict[int,np.ndarray] = {}
    C_by: Dict[int,np.ndarray] = {}
    name_by: Dict[int,str] = {}
    size_by: Dict[int,Tuple[int,int]] = {}

    flat_poses = []
    for iid in img_ids:
        im = imgs[iid]
        cam = cams[im.camera_id]
        K = K_from_camera(cam)
        R, t = pose_world2cam(im)
        P = P_from_KRt(K,R,t)
        C = cam_center_world(R,t)
        K_by[iid], R_by[iid], t_by[iid], P_by[iid], C_by[iid] = K, R, t, P, C
        name_by[iid] = im.name
        size_by[iid] = (cam.width, cam.height)
        T = np.eye(4); T[:3,:3] = R; T[:3,3] = t.reshape(3)
        flat_poses.append(T.reshape(-1))
    flat_poses = np.stack(flat_poses, axis=0)

    # Reference selection + neighbors
    num_refs = int(round(args.num_refs * len(img_ids))) if args.num_refs <= 1.0 else int(args.num_refs)
    try:
        lf.log.debug("Selecting reference views by sparse visibility...")
        refs = select_cameras_by_visibility(rec, num_refs)
        img_id_to_local_idx = {iid: i for i, iid in enumerate(img_ids)}
        refs_local = [img_id_to_local_idx[i] for i in refs]
    except (ValueError, KeyError) as e:
        lf.log.warn(f"Visibility selection failed ({e}), using k-centers on poses.")
        refs_local = select_cameras_kcenters(flat_poses, num_refs)
        refs = [img_ids[i] for i in refs_local]

    nn_table = nearest_neighbors(flat_poses, max(1, args.nns_per_ref))  # local indices

    matcher = None
    worker_thread = None
    job_q = Queue(maxsize=8)   # small buffer to bound memory
    res_q = Queue()

    try:
        # RoMa v2 matcher (GPU if available)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        matcher = RomaMatcher(device=device, mode=args.roma_model, setting=args.roma_setting)
        lf.log.info(f"RoMaV2: setting={args.roma_setting} on {device}")
        
        # Enable CUDA optimizations
        if torch.cuda.is_available():
            # Use TF32 for better performance on Ampere+ GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # Enable cuDNN autotuner for optimal convolution algorithms
            torch.backends.cudnn.benchmark = True

        # Queues & worker
        worker_fn = make_cpu_worker(job_q, res_q,
                                    K_by, R_by, t_by, P_by, C_by, name_by, size_by,
                                    args, matcher.sample_thresh, no_filter=args.no_filter)
        worker_thread = threading.Thread(target=worker_fn, daemon=True)
        worker_thread.start()

        # Producer loop (GPU stage) — enqueue one job per reference
        jobs_enqueued = 0
        t0 = time.time()
        
        total_refs = len(refs_local)

        # Only iterate selected refs (not all frames)
        for i, ref_local in enumerate(tqdm(refs_local, desc="GPU matching / enqueue")):
            # Report progress (10% to 90%)
            if progress_callback:
                # Weighted progress: matching is the heavy part
                pct = 10.0 + (float(i) / max(1, total_refs)) * 80.0
                progress_callback(pct, f"Matching view {i+1}/{total_refs}")

            ref_id = img_ids[ref_local]
            ref_name = name_by[ref_id]
            ref_path = find_image(images_dir, ref_name)
            w_match, h_match = matcher.w_resized, matcher.h_resized
            imA = load_rgb_resized(ref_path, (w_match, h_match))
            imA_np = np.asarray(imA, dtype=np.uint8)          # for CPU bilinear/color
            wA_cam, hA_cam = size_by[ref_id]
            local_nns = nn_table[ref_local][:args.nns_per_ref]
            if len(local_nns) == 0:
                continue

            warp_list_cpu: List[torch.Tensor] = []
            cert_list_cpu: List[torch.Tensor] = []
            nn_ids: List[int] = []

            # Parallel image loading to prevent I/O bottleneck
            def load_neighbor(nn_local):
                nbr_id = img_ids[nn_local]
                if nbr_id == ref_id:
                    return None, None, None
                try:
                    img_path = find_image(images_dir, name_by[nbr_id])
                    imB = load_rgb_resized(img_path, (w_match, h_match))
                    return imB, nn_local, nbr_id
                except Exception as e:
                    lf.log.warn(f"Failed to load neighbor {name_by.get(nbr_id, 'unknown')}: {e}")
                    return None, None, None
            
            nn_images = []
            valid_nn_locals = []
            
            # Use thread pool for parallel I/O (4 threads is good for disk/network I/O)
            with ThreadPoolExecutor(max_workers=4) as executor:
                load_results = list(executor.map(load_neighbor, local_nns))
            
            for imB, nn_local, nbr_id in load_results:
                if imB is not None:
                    nn_images.append(imB)
                    valid_nn_locals.append(nn_local)
                    nn_ids.append(nbr_id)

            if not nn_images:
                continue

            # Batch process all neighbors on GPU for better utilization
            batch_results = matcher.match_grids_batch(imA, nn_images)
            
            for warp_hw, cert_hw in batch_results:
                cert_hw = torch.clamp(cert_hw, min=args.certainty_thresh)  # mild floor (on GPU)
                # Asynchronously move to CPU for the worker and free GPU ASAP
                # (non_blocking=True allows overlap when conditions for async transfers are met)
                warp_cpu = warp_hw.detach().to("cpu", non_blocking=True)
                cert_cpu = cert_hw.detach().to("cpu", non_blocking=True)
                warp_list_cpu.append(warp_cpu)
                cert_list_cpu.append(cert_cpu)
            
            # Ensure all async transfers complete before enqueueing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Free GPU memory of the batch results immediately by dropping references
            del batch_results

            if not cert_list_cpu:
                continue

            # Enqueue one job per reference (CPU will aggregate+filter+triangulate)
            job = dict(
                ref_id=ref_id,
                nn_ids=nn_ids,
                warp_list=warp_list_cpu,
                cert_list=cert_list_cpu,
                imA_np=imA_np,
                w_match=w_match, h_match=h_match,
                wA_cam=wA_cam, hA_cam=hA_cam,
            )
            job_q.put(job)
            jobs_enqueued += 1

    finally:
        # Cleanup RoMa model immediately after GPU loop
        if matcher is not None:
            del matcher
            matcher = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        lf.log.debug("RoMaV2 model unloaded and GPU memory cleared.")

    # Signal worker to finish and wait
    if progress_callback:
        progress_callback(90.0, "Finalizing triangulation...")
    
    job_q.put(None)
    job_q.join()
    if worker_thread:
        worker_thread.join(timeout=1.0)

    # Collect results
    all_xyz, all_rgb, all_err = [], [], []
    results_collected = 0
    while results_collected < jobs_enqueued:
        try:
            xyz, rgb, err = res_q.get_nowait()
        except Exception:
            break
        results_collected += 1
        if xyz is None:
            continue
        all_xyz.append(xyz); all_rgb.append(rgb); all_err.append(err)

    if not all_xyz:
        raise RuntimeError("No points triangulated. Try increasing --num_refs/--nns_per_ref/--matches_per_ref or lowering thresholds.")

    xyz = np.concatenate(all_xyz, axis=0)
    rgb = np.concatenate(all_rgb, axis=0)
    err = np.concatenate(all_err, axis=0)
    lf.log.debug(f"Triangulated points: {xyz.shape[0]} in {time.time()-t0:.1f}s (pipelined GPU↔CPU).")

    # Optional cap
    if args.max_points > 0 and xyz.shape[0] > args.max_points:
        sel = np.random.default_rng(args.seed).choice(xyz.shape[0], size=args.max_points, replace=False)
        xyz, rgb, err = xyz[sel], rgb[sel], err[sel]
        lf.log.debug(f"Capped to {xyz.shape[0]} points (--max_points).")

    if progress_callback:
        progress_callback(95.0, "Writing output...")

    return 0


# ==========================
# LFS-based dense initialization
# ==========================

from dataclasses import dataclass as _dataclass

@_dataclass
class LFSCameraData:
    """Camera data extracted from LichtFeld Studio."""
    uid: int
    image_path: str
    width: int
    height: int
    K: np.ndarray          # [3, 3] intrinsics
    R: np.ndarray          # [3, 3] rotation (world-to-camera)
    t: np.ndarray          # [3, 1] translation (world-to-camera)
    P: np.ndarray          # [3, 4] projection matrix
    C: np.ndarray          # [3] camera center in world coords


@_dataclass
class LFSDenseConfig:
    """Configuration for LFS-based dense initialization."""
    output_path: str                # Path for output PLY file
    roma_setting: str = "fast"
    num_refs: float = 0.75
    nns_per_ref: int = 4
    matches_per_ref: int = 12000
    certainty_thresh: float = 0.20
    reproj_thresh: float = 1.5
    sampson_thresh: float = 5.0
    min_parallax_deg: float = 0.5
    max_points: int = 0
    no_filter: bool = False
    seed: int = 0
    viz_interval: int = 10  # Visualize every N pairs (0 = only at end)


def extract_cameras_from_lfs(camera_nodes) -> List[LFSCameraData]:
    """Extract camera data from LichtFeld Studio scene graph camera nodes.
    
    Args:
        camera_nodes: List of scene nodes with has_camera=True
        
    Returns:
        List of LFSCameraData with all camera information
    """
    camera_list = []
    
    for node in camera_nodes:
        if not node.has_camera:
            continue
            
        # Get image dimensions
        width = node.camera_width
        height = node.camera_height
        
        # Build intrinsic matrix K from focal lengths and principal point (center)
        fx = node.camera_focal_x
        fy = node.camera_focal_y
        cx = width / 2.0   # Principal point at image center
        cy = height / 2.0
        
        K = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        
        # Get rotation and translation (world-to-camera)
        # camera_R and camera_T are Tensors - convert to numpy arrays
        R = np.asarray(node.camera_R).astype(np.float64)  # [3, 3]
        T = np.asarray(node.camera_T).astype(np.float64).reshape(3, 1)  # [3, 1]
        
        # Compute projection matrix P = K @ [R | t]
        P = K @ np.concatenate([R, T], axis=1)  # [3, 4]
        
        # Camera center in world coordinates: C = -R^T @ t
        C = (-R.T @ T).reshape(3)
        
        camera_data = LFSCameraData(
            uid=node.camera_uid,
            image_path=node.image_path,
            width=width,
            height=height,
            K=K,
            R=R,
            t=T,
            P=P,
            C=C
        )
        camera_list.append(camera_data)
    
    return camera_list


def dense_init_from_lfs(
    camera_nodes,  # List of scene nodes with has_camera=True
    config: LFSDenseConfig,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    on_sequential_viz: Optional[Callable[[str], None]] = None,
) -> Tuple[int, Optional[str]]:
    """Run dense initialization using camera nodes from LichtFeld Studio scene graph.
    
    Args:
        camera_nodes: List of scene nodes with has_camera=True
        config: LFSDenseConfig with parameters
        progress_callback: Optional callback for progress updates
        on_sequential_viz: Optional callback for sequential PLY visualization every 10 image pairs
        
    Returns:
        Tuple of (return_code, output_path or error_message)
    """
    np.random.seed(config.seed)
    
    if progress_callback:
        progress_callback(2.0, "Extracting camera data from scene...")
    
    # Extract camera data from LFS scene graph nodes
    camera_list = extract_cameras_from_lfs(camera_nodes)
    if len(camera_list) < 2:
        return 1, "Need at least 2 cameras for dense initialization"
    
    lf.log.debug(f"Loaded {len(camera_list)} cameras from LichtFeld Studio.")
    
    # Build lookup dictionaries by UID
    K_by: Dict[int, np.ndarray] = {}
    R_by: Dict[int, np.ndarray] = {}
    t_by: Dict[int, np.ndarray] = {}
    P_by: Dict[int, np.ndarray] = {}
    C_by: Dict[int, np.ndarray] = {}
    path_by: Dict[int, str] = {}
    size_by: Dict[int, Tuple[int, int]] = {}
    
    img_ids = []
    flat_poses = []
    
    for cam in camera_list:
        uid = cam.uid
        img_ids.append(uid)
        K_by[uid] = cam.K
        R_by[uid] = cam.R
        t_by[uid] = cam.t
        P_by[uid] = cam.P
        C_by[uid] = cam.C
        path_by[uid] = cam.image_path
        size_by[uid] = (cam.width, cam.height)
        
        # Build flat pose for neighbor selection
        T = np.eye(4)
        T[:3, :3] = cam.R
        T[:3, 3] = cam.t.reshape(3)
        flat_poses.append(T.reshape(-1))
    
    flat_poses = np.stack(flat_poses, axis=0)
    
    if progress_callback:
        progress_callback(5.0, "Selecting reference views...")
    
    # Reference selection using k-centers on poses
    num_refs = int(round(config.num_refs * len(img_ids))) if config.num_refs <= 1.0 else int(config.num_refs)
    refs_local = select_cameras_kcenters(flat_poses, num_refs)
    refs = [img_ids[i] for i in refs_local]
    
    lf.log.debug(f"Selected {len(refs)} reference views.")
    
    # Compute nearest neighbors
    nn_table = nearest_neighbors(flat_poses, max(1, config.nns_per_ref))
    
    # Create args namespace for CPU worker compatibility
    from argparse import Namespace
    args = Namespace(
        matches_per_ref=config.matches_per_ref,
        sampson_thresh=config.sampson_thresh,
        reproj_thresh=config.reproj_thresh,
        min_parallax_deg=config.min_parallax_deg,
        no_filter=config.no_filter,
    )
    
    matcher = None
    worker_thread = None
    collector_thread = None
    job_q = Queue(maxsize=8)
    res_q = Queue()
    
    # Shared state for result collector thread
    collector_state = {
        'all_xyz': [],
        'all_rgb': [],
        'all_err': [],
        'pairs_processed': 0,
        'stop': False,
        'lock': threading.Lock(),
    }
    
    # Prepare intermediate PLY path if sequential viz is enabled
    viz_interval = config.viz_interval
    intermediate_ply_base = None
    if on_sequential_viz and viz_interval > 0:
        output_dir = os.path.dirname(config.output_path)
        output_base = os.path.basename(config.output_path)
        base_no_ext = os.path.splitext(output_base)[0]
        os.makedirs(output_dir, exist_ok=True)
        intermediate_ply_base = os.path.join(output_dir, f"{base_no_ext}_intermediate")
    
    def result_collector():
        """Collect results from CPU worker and trigger visualization concurrently."""
        state = collector_state
        while True:
            try:
                # Use blocking get with timeout so we can check stop flag
                xyz, rgb, err = res_q.get(timeout=0.1)
            except Exception:
                # Check if we should stop
                with state['lock']:
                    if state['stop']:
                        break
                continue
            
            if xyz is None:
                continue
            
            with state['lock']:
                state['all_xyz'].append(xyz)
                state['all_rgb'].append(rgb)
                state['all_err'].append(err)
                state['pairs_processed'] += 1
                pairs_count = state['pairs_processed']
                current_xyz = list(state['all_xyz'])
                current_rgb = list(state['all_rgb'])
            
            # Trigger visualization every viz_interval pairs
            if on_sequential_viz and viz_interval > 0 and pairs_count % viz_interval == 0 and current_xyz:
                try:
                    # Concatenate results so far
                    xyz_so_far = np.concatenate(current_xyz, axis=0)
                    rgb_so_far = np.concatenate(current_rgb, axis=0)
                    
                    # Write intermediate PLY
                    intermediate_ply_path = f"{intermediate_ply_base}_{pairs_count}.ply"
                    write_ply(intermediate_ply_path, xyz_so_far, to_uint8_rgb(rgb_so_far))
                    lf.log.debug(f"Live update: {xyz_so_far.shape[0]:,} points ({pairs_count} refs processed)")
                    
                    # Trigger visualization callback
                    on_sequential_viz(intermediate_ply_path)
                except Exception as e:
                    lf.log.warn(f"Failed to write intermediate PLY: {e}")
    
    try:
        # Initialize RoMa v2 matcher
        device = "cuda" if torch.cuda.is_available() else "cpu"
        matcher = RomaMatcher(device=device, mode="outdoor", setting=config.roma_setting)
        lf.log.info(f"RoMaV2: setting={config.roma_setting} on {device}")
        
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        
        # Start CPU worker
        worker_fn = make_cpu_worker(
            job_q, res_q,
            K_by, R_by, t_by, P_by, C_by, path_by, size_by,
            args, matcher.sample_thresh, no_filter=config.no_filter
        )
        worker_thread = threading.Thread(target=worker_fn, daemon=True)
        worker_thread.start()
        
        # Start result collector thread for concurrent visualization
        collector_thread = threading.Thread(target=result_collector, daemon=True)
        collector_thread.start()
        
        # Producer loop
        jobs_enqueued = 0
        t0 = time.time()
        total_refs = len(refs_local)
        w_match, h_match = matcher.w_resized, matcher.h_resized
        
        for i, ref_local in enumerate(tqdm(refs_local, desc="GPU matching / enqueue")):
            if progress_callback:
                pct = 10.0 + (float(i) / max(1, total_refs)) * 80.0
                progress_callback(pct, f"Matching view {i+1}/{total_refs}")
            
            ref_id = img_ids[ref_local]
            ref_path = path_by[ref_id]
            
            # Load and resize reference image
            try:
                imA = load_rgb_resized(ref_path, (w_match, h_match))
            except Exception as e:
                lf.log.warn(f"Failed to load reference {ref_path}: {e}")
                continue
            
            imA_np = np.asarray(imA, dtype=np.uint8)
            wA_cam, hA_cam = size_by[ref_id]
            
            local_nns = nn_table[ref_local][:config.nns_per_ref]
            if len(local_nns) == 0:
                continue
            
            warp_list_cpu: List[torch.Tensor] = []
            cert_list_cpu: List[torch.Tensor] = []
            nn_ids: List[int] = []
            
            # Load neighbor images
            def load_neighbor(nn_local):
                nbr_id = img_ids[nn_local]
                if nbr_id == ref_id:
                    return None, None, None
                try:
                    img_path = path_by[nbr_id]
                    imB = load_rgb_resized(img_path, (w_match, h_match))
                    return imB, nn_local, nbr_id
                except Exception as e:
                    lf.log.warn(f"Failed to load neighbor: {e}")
                    return None, None, None
            
            nn_images = []
            valid_nn_locals = []
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                load_results = list(executor.map(load_neighbor, local_nns))
            
            for imB, nn_local, nbr_id in load_results:
                if imB is not None:
                    nn_images.append(imB)
                    valid_nn_locals.append(nn_local)
                    nn_ids.append(nbr_id)
            
            if not nn_images:
                continue
            
            # Batch match on GPU
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
                w_match=w_match, h_match=h_match,
                wA_cam=wA_cam, hA_cam=hA_cam,
            )
            job_q.put(job)
            jobs_enqueued += 1
    
    finally:
        if matcher is not None:
            del matcher
            matcher = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    
    # Signal worker to finish
    if progress_callback:
        progress_callback(90.0, "Finalizing triangulation...")
    
    job_q.put(None)
    job_q.join()
    if worker_thread:
        worker_thread.join(timeout=1.0)
    
    # Signal collector to stop and wait for it
    with collector_state['lock']:
        collector_state['stop'] = True
    if collector_thread:
        collector_thread.join(timeout=1.0)
    
    # Drain any remaining results from the queue
    while True:
        try:
            xyz, rgb, err = res_q.get_nowait()
            if xyz is not None:
                with collector_state['lock']:
                    collector_state['all_xyz'].append(xyz)
                    collector_state['all_rgb'].append(rgb)
                    collector_state['all_err'].append(err)
        except Exception:
            break
    
    # Get final results from collector state
    with collector_state['lock']:
        all_xyz = collector_state['all_xyz']
        all_rgb = collector_state['all_rgb']
        all_err = collector_state['all_err']
    
    if not all_xyz:
        return 1, "No points triangulated. Try adjusting parameters."
    
    xyz = np.concatenate(all_xyz, axis=0)
    rgb = np.concatenate(all_rgb, axis=0)
    err = np.concatenate(all_err, axis=0)
    lf.log.info(f"Triangulated points: {xyz.shape[0]} in {time.time()-t0:.1f}s")
    
    # Optional cap
    if config.max_points > 0 and xyz.shape[0] > config.max_points:
        sel = np.random.default_rng(config.seed).choice(xyz.shape[0], size=config.max_points, replace=False)
        xyz, rgb, err = xyz[sel], rgb[sel], err[sel]
        lf.log.info(f"Capped to {xyz.shape[0]} points (max_points).")
    
    if progress_callback:
        progress_callback(95.0, "Writing output PLY...")
    
    # Write PLY file
    output_path = config.output_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    write_ply(output_path, xyz, to_uint8_rgb(rgb))
    lf.log.info(f"Wrote dense point cloud to {output_path}")
    
    if progress_callback:
        progress_callback(100.0, f"Done! {xyz.shape[0]:,} points")
    
    return 0, output_path


# ==========================
# CLI
# ==========================

def build_argparser():
    ap = argparse.ArgumentParser("Dense COLMAP initializer (EDGS-style + RoMa v2) with GPU↔CPU pipelining")
    ap.add_argument("--scene_root", type=str, required=True, help="Path containing images*/ and sparse/0/")
    ap.add_argument("--images_subdir", type=str, default="images_2", help="Which images dir to read under scene_root")
    ap.add_argument("--out_name", type=str, default="points3D_dense.bin", help="Output filename under sparse/0/")
    ap.add_argument("--roma_model", type=str, default="outdoor", choices=["outdoor","indoor"], help="RoMa model variant (kept for compatibility, RoMaV2 is unified)")
    ap.add_argument("--roma_setting", type=str, default="fast", choices=["precise","base","fast","turbo"], help="RoMaV2 quality/speed setting")

    # Selection
    ap.add_argument("--num_refs", type=float, default=0.75, help="Fraction (<=1) or count (>1) of frames to use as references")
    ap.add_argument("--nns_per_ref", type=int, default=4, help="Nearest neighbors per reference (3-5 is robust)")

    # Sampling / gating
    ap.add_argument("--matches_per_ref", type=int, default=12000, help="Samples per ref after aggregation")
    ap.add_argument("--certainty_thresh", type=float, default=0.05, help="Min certainty floor before selection (more permissive)")
    ap.add_argument("--reproj_thresh", type=float, default=4.0, help="Max reprojection error (px) — relaxed by default")
    ap.add_argument("--sampson_thresh", type=float, default=50.0, help="Max Sampson error (px^2) pre-triangulation (<=0 to disable) — relaxed by default")
    ap.add_argument("--min_parallax_deg", type=float, default=0.1, help="Min parallax (deg); set 0 to disable (relaxed)")
    ap.add_argument("--no_filter", action="store_true", help="Disable all geometric filtering (reprojection/sampson/cheirality/parallax). Useful for debugging/raw output.")

    # Output shaping
    ap.add_argument("--max_points", type=int, default=0, help="Optional cap on total points (0 = unlimited)")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    return ap

if __name__ == "__main__":
    args = build_argparser().parse_args()
    raise SystemExit(dense_init(args))
