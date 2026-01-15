import torch
import torch.nn.functional as F
import numpy as np
from romav2.device import device


def vis(img1, img2, warp_AB, warp_BA, certainty_AB, certainty_BA):
    H, W = warp_AB.shape[1:3]
    x1 = F.interpolate(
        (torch.tensor(np.array(img1)) / 255).to(device).permute(2, 0, 1)[None],
        size=(H, W),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    x2 = F.interpolate(
        (torch.tensor(np.array(img2)) / 255).to(device).permute(2, 0, 1)[None],
        size=(H, W),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    im2_transfer_rgb = F.grid_sample(
        x2, warp_AB, mode="bilinear", align_corners=False
    )[0]
    im1_transfer_rgb = F.grid_sample(
        x1, warp_BA, mode="bilinear", align_corners=False
    )[0]
    warp_im = torch.cat((im2_transfer_rgb, im1_transfer_rgb), dim=2)  # .permute(1,2,0)
    white_im = torch.ones((H, 2 * W), device=device)
    certainty = torch.cat((certainty_AB, certainty_BA), dim=2)

    vis_im = certainty[0] * warp_im + (1 - certainty[0]) * white_im
    x = torch.cat((x1[0], x2[0]), dim=2)
    vis_im = torch.cat((x, vis_im), dim=1)
    return vis_im
