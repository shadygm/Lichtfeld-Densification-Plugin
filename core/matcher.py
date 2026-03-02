"""RoMaV2 matcher wrapper."""
from __future__ import annotations

import gc
from typing import List, Tuple

import lichtfeld as lf
import numpy as np
import torch
import torch.nn.functional as torch_F
from PIL import Image

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_ROMA_SRC = _ROOT / "RoMaV2" / "src"
if str(_ROMA_SRC) not in sys.path:
    sys.path.insert(0, str(_ROMA_SRC))

from romav2 import RoMaV2
from romav2.romav2 import _interpolate_warp_and_confidence


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

    def close(self) -> None:
        """Release model weights and CUDA allocations for this matcher."""
        model = getattr(self, "model", None)
        if model is None:
            return
        try:
            model.to("cpu")
        except Exception:
            pass
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @torch.inference_mode()
    def match_grids(self, imA: Image.Image, imB: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.model is None:
            raise RuntimeError("RoMaV2 model has been released; create a new matcher before matching.")
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

    def _prepare_image(self, img: Image.Image) -> torch.Tensor:
        """Convert a PIL image to a float32 BCHW tensor on device."""
        t = torch.from_numpy(np.array(img)).permute(2, 0, 1).to(self.device)
        if t.dtype == torch.uint8:
            t = t.float().div_(255.0)
        return t.unsqueeze(0)

    @torch.inference_mode()
    def match_grids_batch(self, imA: Image.Image, imB_list: List[Image.Image]):
        """Match a reference image against a list of neighbors.

        Backbone features and refiner features for *imA* are extracted once
        and reused for every neighbor, avoiding redundant GPU work.
        """
        if not imB_list:
            return []
        if self.model is None:
            raise RuntimeError("RoMaV2 model has been released; create a new matcher.")
        torch.set_float32_matmul_precision("highest")
        model = self.model

        # --- reference image: preprocess once ---------------------------------
        img_A = self._prepare_image(imA)
        img_A_lr = torch_F.interpolate(
            img_A, size=(model.H_lr, model.W_lr),
            mode="bicubic", align_corners=False, antialias=True,
        )
        has_hr = model.H_hr is not None and model.W_hr is not None
        img_A_hr = (
            torch_F.interpolate(
                img_A, size=(model.H_hr, model.W_hr),
                mode="bicubic", align_corners=False, antialias=True,
            )
            if has_hr else None
        )

        # Backbone features for A (the expensive ViT pass) — computed once.
        f_A = model.f(img_A_lr)

        # Refiner (VGG) features for A at each resolution stage — computed once.
        refiner_feats_A = [None, None]  # [lr_stage, hr_stage]
        stage_imgs_A = [img_A_lr, img_A_hr]
        for idx, sA in enumerate(stage_imgs_A):
            if sA is not None:
                refiner_feats_A[idx] = model.refiner_features(sA)

        # --- per-neighbor matching --------------------------------------------
        results: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for imB in imB_list:
            img_B = self._prepare_image(imB)
            img_B_lr = torch_F.interpolate(
                img_B, size=(model.H_lr, model.W_lr),
                mode="bicubic", align_corners=False, antialias=True,
            )
            img_B_hr = (
                torch_F.interpolate(
                    img_B, size=(model.H_hr, model.W_hr),
                    mode="bicubic", align_corners=False, antialias=True,
                )
                if has_hr else None
            )

            # Backbone features for B (unique per neighbor).
            f_B = model.f(img_B_lr)

            # Shallow-copy f_A: the matcher's head mutates f_list_A[-1].
            f_A_copy = list(f_A)

            # Cross-attention matching.
            matcher_out = model.matcher(
                f_A_copy, f_B,
                img_A=img_A_lr, img_B=img_B_lr,
                bidirectional=model.bidirectional,
            )
            warp_AB = matcher_out["warp_AB"]
            confidence_AB = matcher_out["confidence_AB"]
            warp_BA = matcher_out.get("warp_BA")
            confidence_BA = matcher_out.get("confidence_BA")

            # Refinement stages (lr, then optionally hr).
            stage_imgs_B = [img_B_lr, img_B_hr]
            for stage, (sA, sB) in enumerate(zip(stage_imgs_A, stage_imgs_B)):
                if sA is None or sB is None:
                    continue
                _, _, Hs, Ws = sA.shape
                scale_factor = torch.tensor(
                    (Ws / model.anchor_width, Hs / model.anchor_height),
                    device=self.device,
                )
                cached_A = refiner_feats_A[stage]
                ref_feats_B = model.refiner_features(sB)

                for ps_str, refiner in model.refiners.items():
                    ps = int(ps_str)
                    zero_out = has_hr and ps == 4 and stage == 1
                    warp_AB, confidence_AB = _interpolate_warp_and_confidence(
                        warp=warp_AB, confidence=confidence_AB,
                        H=Hs, W=Ws, patch_size=ps,
                        zero_out_precision=zero_out,
                    )
                    if model.bidirectional and warp_BA is not None:
                        warp_BA, confidence_BA = _interpolate_warp_and_confidence(
                            warp=warp_BA, confidence=confidence_BA,
                            H=Hs, W=Ws, patch_size=ps,
                            zero_out_precision=zero_out,
                        )

                    fp_A = cached_A[ps]
                    fp_B = ref_feats_B[ps]
                    out_AB = refiner(
                        f_A=fp_A, f_B=fp_B,
                        prev_warp=warp_AB,
                        prev_confidence=confidence_AB,
                        scale_factor=scale_factor,
                    )
                    warp_AB = out_AB["warp"]
                    confidence_AB = out_AB["confidence"]
                    if model.bidirectional and warp_BA is not None:
                        out_BA = refiner(
                            f_A=fp_B, f_B=fp_A,
                            prev_warp=warp_BA,
                            prev_confidence=confidence_BA,
                            scale_factor=scale_factor,
                        )
                        warp_BA = out_BA["warp"]
                        confidence_BA = out_BA["confidence"]

            # Map raw confidence → overlap (sigmoid), matching model.match().
            overlap = confidence_AB[..., :1].sigmoid()
            if model.threshold is not None:
                overlap[overlap > model.threshold] = 1.0
            overlap = overlap[0].squeeze(-1)  # (H_out, W_out)
            warp_AB_final = warp_AB[0]  # (H_out, W_out, 2)

            # Build the reference-side grid at the *output* resolution (which
            # equals lr when there is no hr stage, or hr/patch_size otherwise).
            H_out, W_out = overlap.shape
            yy = torch.linspace(-1 + 1 / H_out, 1 - 1 / H_out, H_out, device=self.device)
            xx = torch.linspace(-1 + 1 / W_out, 1 - 1 / W_out, W_out, device=self.device)
            yy, xx = torch.meshgrid(yy, xx, indexing="ij")
            gridA = torch.stack([xx, yy], dim=-1)

            warp = torch.cat([gridA, warp_AB_final], dim=-1)
            results.append((warp.contiguous(), overlap.contiguous()))

        return results
