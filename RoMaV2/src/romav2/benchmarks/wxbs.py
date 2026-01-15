from dataclasses import dataclass

import torch

from tqdm import tqdm
from wxbs_benchmark.dataset import WxBSDataset
from wxbs_benchmark.evaluation import evaluate_corrs

from romav2.romav2 import RoMaV2
from romav2.io import tensor_to_pil
from romav2.device import device
from pathlib import Path
from romav2.geometry import to_normalized, to_pixel, bhwc_grid_sample
from romav2.vis import vis


class WxBSBenchmark:
    @dataclass(frozen=True)
    class Cfg:
        subset: str = "test"
        dataset_path: str = "data/WxBS"
        download: bool = True

    def __init__(self, cfg: Cfg):
        self.subset = cfg.subset
        WxBSDataset.urls["v1.1"][0] = (
            "https://github.com/Parskatt/storage/releases/download/wxbs/WxBS_v1.1.zip"
        )

        def wrap(f):
            def __getitem__(self, idx):
                out = f(self, idx)
                return {
                    **out,
                    "imgfname1": self.pairs[idx][0],
                    "imgfname2": self.pairs[idx][1],
                }

            return __getitem__

        WxBSDataset.__getitem__ = wrap(WxBSDataset.__getitem__)
        self.dataset = WxBSDataset(
            cfg.dataset_path, subset=self.subset, download=cfg.download
        )

    def __call__(self, model: RoMaV2, step: int):
        estimated_right = []
        estimated_left = []
        for pair_dict in tqdm(self.dataset):
            H_A, W_A = pair_dict["img1"].shape[:2]
            H_B, W_B = pair_dict["img2"].shape[:2]
            points_left = torch.from_numpy(pair_dict["pts"][:, :2]).to(device).float()
            points_right = torch.from_numpy(pair_dict["pts"][:, 2:4]).to(device).float()
            n_points_left = to_normalized(points_left, H=H_A, W=W_A)
            n_points_right = to_normalized(points_right, H=H_B, W=W_B)
            if model.name.lower() == "roma":
                preds: tuple[torch.Tensor, torch.Tensor] = model.match(
                    pair_dict["imgfname1"], pair_dict["imgfname2"]
                )  # type: ignore
                warp_bidirectional = preds[0]
                W_pred = warp_bidirectional.shape[2]
                warp_AB = warp_bidirectional[:, :, : W_pred // 2, 2:]
                warp_BA = warp_bidirectional[:, :, W_pred // 2 :, :2]
                overlap_AB = preds[1][:, :, : W_pred // 2]
                overlap_BA = preds[1][:, :, W_pred // 2 :]
            else:
                preds = model.match(pair_dict["img1"], pair_dict["img2"])
                warp_AB = preds["warp_AB"]
                warp_BA = preds["warp_BA"]
                overlap_AB = preds["overlap_AB"]
                overlap_BA = preds["overlap_BA"]
            n_est_points_right = bhwc_grid_sample(
                warp_AB,
                n_points_left[None, None],
                mode="bilinear",
                align_corners=False,
            )
            n_est_points_left = bhwc_grid_sample(
                warp_BA,
                n_points_right[None, None],
                mode="bilinear",
                align_corners=False,
            )
            est_points_right = to_pixel(n_est_points_right, H=H_B, W=W_B)
            est_points_left = to_pixel(n_est_points_left, H=H_A, W=W_A)
            # model.to_pixel_coordinates(preds["warp_AB"], H_A, W_A, H_B, W_B)
            vis_im_path = Path(f"vis/wxbs_{model.name}_{pair_dict['name']}.jpg")
            if not vis_im_path.exists():
                vis_im = vis(
                    pair_dict["img1"],
                    pair_dict["img2"],
                    warp_AB,
                    warp_BA,
                    overlap_AB,
                    overlap_BA,
                )
                vis_im_path.parent.mkdir(parents=True, exist_ok=True)
                tensor_to_pil(vis_im).save(vis_im_path)
            estimated_right.append(est_points_right.cpu().numpy()[0, 0])
            estimated_left.append(est_points_left.cpu().numpy()[0, 0])
        result_dict, thresholds = evaluate_corrs(
            estimated_right, estimated_left, self.subset
        )
        return result_dict, thresholds
