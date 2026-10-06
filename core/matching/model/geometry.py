import torch
import torch.nn.functional as F
from .device import device


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


def prec_mat_from_prec_params(p: torch.Tensor) -> torch.Tensor:
    P = p.new_zeros(p.shape[0], p.shape[1], p.shape[2], 2, 2)
    P[..., 0, 0] = p[..., 0]
    P[..., 1, 0] = p[..., 1]
    P[..., 0, 1] = p[..., 1]
    P[..., 1, 1] = p[..., 2]
    return P


# Code taken from https://github.com/PruneTruong/DenseMatching/blob/40c29a6b5c35e86b9509e65ab0cd12553d998e5f/validation/utils_pose_estimation.py
# --- GEOMETRY ---
