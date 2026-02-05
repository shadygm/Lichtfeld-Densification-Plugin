"""RoMaV2 matcher wrapper."""
from __future__ import annotations

from typing import List, Tuple

import lichtfeld as lf
import torch
from PIL import Image

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_ROMA_SRC = _ROOT / "RoMaV2" / "src"
if str(_ROMA_SRC) not in sys.path:
    sys.path.insert(0, str(_ROMA_SRC))

from romav2 import RoMaV2


class RomaMatcher:
    """Wrapper around RoMaV2 for dense matching."""

    def __init__(self, device: str = "cuda", mode: str = "outdoor", setting: str = "fast"):
        self.device = torch.device(device)
        torch.set_float32_matmul_precision("highest")
        self.model = RoMaV2(RoMaV2.Cfg(compile=False))
        if setting == "high":
            self.model.H_lr = 640
            self.model.W_lr = 640
            self.model.H_hr = 960
            self.model.W_hr = 960
            self.model.bidirectional = True
        else:
            self.model.apply_setting(setting)
        self.model.to(self.device)
        self.model.eval()
        self.sample_thresh = 0.9
        self.w_resized = self.model.W_lr
        self.h_resized = self.model.H_lr
        lf.log.info(
            f"RoMaV2 initialized (setting={setting}, H_lr={self.model.H_lr}, W_lr={self.model.W_lr}, device={device})"
        )

    @torch.inference_mode()
    def match_grids(self, imA: Image.Image, imB: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        torch.set_float32_matmul_precision("highest")
        preds = self.model.match(imA, imB)
        warp_AB = preds["warp_AB"][0]
        overlap_AB = preds["overlap_AB"][0].squeeze(-1)
        H, W = overlap_AB.shape
        yy = torch.linspace(-1 + 1 / H, 1 - 1 / H, H, device=self.device)
        xx = torch.linspace(-1 + 1 / W, 1 - 1 / W, W, device=self.device)
        yy, xx = torch.meshgrid(yy, xx, indexing="ij")
        gridA = torch.stack([xx, yy], dim=-1)
        warp = torch.cat([gridA, warp_AB], dim=-1)
        return warp.contiguous(), overlap_AB.contiguous()

    @torch.inference_mode()
    def match_grids_batch(self, imA: Image.Image, imB_list: List[Image.Image]):
        if not imB_list:
            return []
        results = []
        batch_size = 4 if self.device.type == "cuda" else 2
        for i in range(0, len(imB_list), batch_size):
            batch = imB_list[i : i + batch_size]
            for imB in batch:
                warp, cert = self.match_grids(imA, imB)
                results.append((warp, cert))
        return results
