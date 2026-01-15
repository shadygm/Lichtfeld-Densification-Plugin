from pathlib import Path

import torch
from PIL import Image
from romav2.io import tensor_to_pil
from romav2.device import device
from romav2 import RoMaV2

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--im_A_path", default="assets/toronto_A.jpg", type=str)
    parser.add_argument("--im_B_path", default="assets/toronto_B.jpg", type=str)
    parser.add_argument("--save_path", default="demo/roma_v2_std_toronto.jpg", type=str)
    parser.add_argument("--std_max", default=8.0, type=float)

    args, _ = parser.parse_known_args()
    im1_path = args.im_A_path
    im2_path = args.im_B_path
    save_path = args.save_path

    # Create model

    model = RoMaV2()
    model.apply_setting("precise")

    H, W = model.H_hr, model.W_hr

    im1 = Image.open(im1_path).resize((W, H))
    im2 = Image.open(im2_path).resize((W, H))

    # Match
    preds = model.match(im1_path, im2_path)
    warp_AB, overlap_AB = preds["warp_AB"][0], preds["overlap_AB"][0]
    warp_BA, overlap_BA = preds["warp_BA"][0], preds["overlap_BA"][0]
    precision_AB, precision_BA = (
        preds["precision_AB"][0],
        preds["precision_BA"][0],
    )

    std_AB = torch.linalg.det(precision_AB) ** (-1 / 4)
    std_BA = torch.linalg.det(precision_BA) ** (-1 / 4)

    std_im = torch.cat((std_AB, std_BA), dim=1)
    overlap = torch.cat((overlap_AB, overlap_BA), dim=1)[..., 0]
    white_im = torch.ones((H, 2 * W), device=device)
    std_im = (std_im / args.std_max).clamp(0, 1)
    vis_im = overlap * std_im + (1 - overlap) * white_im
    if not Path(save_path).exists():
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(vis_im).save(save_path)
