from typing import Literal
import torch
import torch.nn.functional as F
from romav2.device import device
from einops import einsum
from romav2.types import ConfidenceMode
import numpy as np


def to_homogeneous(x: torch.Tensor) -> torch.Tensor:
    return torch.cat((x, torch.ones_like(x[..., :1])), dim=-1)


def from_homogeneous(x: torch.Tensor) -> torch.Tensor:
    return x[..., :-1] / x[..., -1:]


def get_normalized_grid(
    B: int,
    H: int,
    W: int,
    overload_device: torch.device | None = None,
) -> torch.Tensor:
    x1_n = torch.meshgrid(
        *[
            torch.linspace(-1 + 1 / n, 1 - 1 / n, n, device=overload_device or device)
            for n in (B, H, W)
        ],
        indexing="ij",
    )
    x1_n = torch.stack((x1_n[2], x1_n[1]), dim=-1).reshape(B, H, W, 2)
    return x1_n


def get_pixel_grid(
    B: int,
    *,
    H: int,
    W: int,
    overload_device: torch.device | None = None,
) -> torch.Tensor:
    x1_n = torch.meshgrid(
        *[torch.arange(n, device=overload_device or device) + 0.5 for n in (B, H, W)],
        indexing="ij",
    )
    x1_n = torch.stack((x1_n[2], x1_n[1]), dim=-1).reshape(B, H, W, 2)
    return x1_n


def to_normalized(x: torch.Tensor, *, H: int, W: int) -> torch.Tensor:
    return torch.stack((2 * x[..., 0] / W, 2 * x[..., 1] / H), dim=-1) - 1


def to_pixel(x: torch.Tensor, *, H: int, W: int) -> torch.Tensor:
    return torch.stack(((x[..., 0] + 1) / 2 * W, (x[..., 1] + 1) / 2 * H), dim=-1)


def warp_and_overlap_from_depth(
    *,
    depth_A: torch.Tensor,
    depth_B: torch.Tensor,
    K_A: torch.Tensor,
    K_B: torch.Tensor,
    T_AB: torch.Tensor,
    rel_depth_error_threshold: float | None,
    mode: ConfidenceMode,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = depth_A.device
    if mode == "covis" and rel_depth_error_threshold is None:
        raise ValueError("rel_depth_error_threshold must be provided for covis mode")
    B, H_A, W_A, one = depth_A.shape
    B, H_B, W_B, one = depth_B.shape
    assert one == 1, "depth_A and depth_B must have shape (B, H, W, 1)"
    grid = to_homogeneous(get_pixel_grid(B, H=H_A, W=W_A, overload_device=device))
    x = einsum(
        grid, K_A.inverse(), "batch H W pixel, batch calib pixel -> batch H W calib"
    )
    x = x * depth_A
    x = to_homogeneous(x)
    x = einsum(x, T_AB, "batch H W calib, batch world calib -> batch H W world")
    x_AB = from_homogeneous(x)
    x = einsum(x_AB, K_B, "batch H W world, batch pixel world -> batch H W pixel")
    x = from_homogeneous(x)
    warp = to_normalized(x, H=H_B, W=W_B)
    z_AB = x_AB[..., -1:]
    pos_and_finite_depth = (
        (z_AB > 0.0).logical_and(z_AB.isfinite()).logical_and(~depth_B.isnan())
    )
    z_B = bhwc_grid_sample(depth_B, warp, mode="bilinear", align_corners=False)
    rel_depth_error = (z_B - z_AB) / z_B
    pos_depth_and_within_frame = (pos_and_finite_depth).logical_and(
        warp.abs().amax(dim=-1, keepdim=True) < 1.0
    )
    if mode == "covis" or mode == "covis_and_correct":
        assert rel_depth_error_threshold is not None
        consistent_depth = (
            rel_depth_error.abs() < rel_depth_error_threshold
        ).logical_and(pos_and_finite_depth)
        conf = consistent_depth
    elif mode == "frame":
        conf = pos_depth_and_within_frame
    elif mode == "positive":
        conf = pos_and_finite_depth
    else:
        raise ValueError(f"Unknown confidence mode: {mode}")
    return warp, conf.float()


def bhwc_interpolate(
    x: torch.Tensor,
    size: tuple[int, int],
    mode: str = "bilinear",
    align_corners: bool | None = None,
) -> torch.Tensor:
    return F.interpolate(
        x.permute(0, 3, 1, 2), size=size, mode=mode, align_corners=align_corners
    ).permute(0, 2, 3, 1)


def bhwc_grid_sample(
    x: torch.Tensor,
    grid: torch.Tensor,
    mode: str = "bilinear",
    align_corners: bool | None = None,
) -> torch.Tensor:
    return F.grid_sample(
        x.permute(0, 3, 1, 2), grid, mode=mode, align_corners=align_corners
    ).permute(0, 2, 3, 1)


def angle_error_mat(R1, R2):
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos = np.clip(cos, -1.0, 1.0)  # numercial errors can make it out of bounds
    return np.rad2deg(np.abs(np.arccos(cos)))


def angle_error_vec(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))


def compute_pose_error(R_gt, t_gt, R, t):
    error_t = angle_error_vec(t.squeeze(), t_gt)
    error_t = np.minimum(error_t, 180 - error_t)  # ambiguity of E estimation
    error_R = angle_error_mat(R, R_gt)
    return error_t, error_R


def pose_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0.0, errors]
    recall = np.r_[0.0, recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index - 1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e) / t)  # type: ignore
    return aucs


def cov_mat_from_cov_params(c: torch.Tensor) -> torch.Tensor:
    return prec_mat_from_prec_params(c)


def prec_mat_from_prec_params(p: torch.Tensor) -> torch.Tensor:
    P = p.new_zeros(p.shape[0], p.shape[1], p.shape[2], 2, 2)
    P[..., 0, 0] = p[..., 0]
    P[..., 1, 0] = p[..., 1]
    P[..., 0, 1] = p[..., 1]
    P[..., 1, 1] = p[..., 2]
    return P


def to_double_angle_rep(v: torch.Tensor) -> torch.Tensor:
    angle = torch.atan2(v[..., 1], v[..., 0])
    double_angle_rep = torch.stack((torch.cos(2 * angle), torch.sin(2 * angle)), dim=-1)
    return double_angle_rep


def prec_mat_to_flow(
    P: torch.Tensor, vis_max: float, mode: Literal["smallest", "largest"] = "largest"
) -> torch.Tensor:
    vals, vecs = torch.linalg.eigh(P)
    if mode == "smallest":
        vis_vec = vecs[..., 0]
    elif mode == "largest":
        vis_vec = vecs[..., -1]
    else:
        raise ValueError(f"Invalid mode: {mode}")
    # select upper half-plane
    vis_vec = torch.where(vis_vec[..., 1:2] >= 0, vis_vec, -vis_vec)
    double_angle_rep = to_double_angle_rep(vis_vec)
    scale = (vals[..., 0] * vals[..., 1]).pow(0.25).clamp(0, vis_max)[..., None]
    flow = scale * double_angle_rep
    return flow


def prec_params_to_flow(
    p: torch.Tensor, vis_max: float, mode: Literal["smallest", "largest"] = "largest"
) -> torch.Tensor:
    P = prec_mat_from_prec_params(p)
    return prec_mat_to_flow(P, vis_max, mode)


def overlap_from_warps(
    warp_A_to_B: torch.Tensor, warp_B_to_A: torch.Tensor, error_threshold: float = 5e-3
) -> torch.Tensor:
    B, H, W, _ = warp_A_to_B.shape
    grid = get_normalized_grid(B, H, W, overload_device=warp_A_to_B.device)
    in_frame = warp_A_to_B.abs().le(1.0).all(dim=-1, keepdim=True)
    fwd_bwd = bhwc_grid_sample(
        warp_B_to_A, warp_A_to_B, mode="bilinear", align_corners=False
    )
    fwd_bwd_error = (fwd_bwd - grid).norm(dim=-1, keepdim=True)
    overlap = (fwd_bwd_error < error_threshold).logical_and(in_frame)
    return overlap.float()


# Code taken from https://github.com/PruneTruong/DenseMatching/blob/40c29a6b5c35e86b9509e65ab0cd12553d998e5f/validation/utils_pose_estimation.py
# --- GEOMETRY ---
def estimate_pose_cv2_ransac(
    kpts0: np.ndarray,
    kpts1: np.ndarray,
    K0: np.ndarray,
    K1: np.ndarray,
    norm_thresh: float,
    conf: float = 0.99999,
):
    import cv2

    if len(kpts0) < 5:
        return None
    K0inv = np.linalg.inv(K0[:2, :2])
    K1inv = np.linalg.inv(K1[:2, :2])

    kpts0 = (K0inv @ (kpts0 - K0[None, :2, 2]).T).T
    kpts1 = (K1inv @ (kpts1 - K1[None, :2, 2]).T).T
    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=norm_thresh, prob=conf
    )

    ret = None
    if E is not None:
        best_num_inliers = 0

        for _E in np.split(E, len(E) // 3):
            n, R, t, _ = cv2.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)  # type: ignore
            if n > best_num_inliers:
                best_num_inliers = n
                ret = (R, t, mask.ravel() > 0)
    assert ret is not None
    return ret


def compute_relative_pose(R1, t1, R2, t2):
    rots = R2 @ (R1.mT)
    trans = -rots @ t1 + t2
    return rots, trans
