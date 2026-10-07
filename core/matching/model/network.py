from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import OrderedDict


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import logging
from .device import device
from .features import Descriptor, FineFeatures
from .geometry import bhwc_interpolate, prec_mat_from_prec_params
from .io import check_not_i16
from .matcher import Matcher
from .refiner import Refiners
from .types import Setting, ImageLike

logger = logging.getLogger(__name__)


def _interpolate_warp_and_confidence(
    *,
    warp: torch.Tensor,
    confidence: torch.Tensor,
    H: int,
    W: int,
    patch_size: int,
    zero_out_precision: bool,
):
    warp = bhwc_interpolate(
        warp.detach(),
        size=(H // patch_size, W // patch_size),
        mode="bilinear",
        align_corners=False,
    )
    if zero_out_precision:
        # delta at 4 is absolute, and if we
        # for the second pass we therefore can't use first pred.
        # overlap is fine since it's relative to matcher pred.
        confidence[..., 1:] = 0.0

    confidence = bhwc_interpolate(
        confidence.detach(),
        size=(H // patch_size, W // patch_size),
        mode="bilinear",
        align_corners=False,
    )
    return warp, confidence


def _map_confidence(*, confidence: torch.Tensor, threshold: float | None):
    overlap = confidence[..., :1].sigmoid()
    if threshold is not None:
        overlap[overlap > threshold] = 1.0
    precision = prec_mat_from_prec_params(confidence[..., 1:4])
    return overlap, precision


class RoMaV2(nn.Module):
    @dataclass(frozen=True)
    class Cfg:
        descriptor: Descriptor.Cfg = Descriptor.Cfg()
        matcher: Matcher.Cfg = Matcher.Cfg()
        refiners: Refiners.Cfg = Refiners.Cfg()
        refiner_features: FineFeatures.Cfg = FineFeatures.Cfg()
        anchor_width: int = 512
        anchor_height: int = 512
        setting: Setting = "precise"
        compile: bool = True
        name: str = "RoMa v2"

    # settings
    H_lr: int
    W_lr: int
    H_hr: int | None
    W_hr: int | None
    bidirectional: bool
    threshold: float | None
    balanced_sampling: bool

    def __init__(self, cfg: Cfg | None = None):
        super().__init__()
        if cfg is None:
            # default
            cfg = RoMaV2.Cfg()
            
        weights = torch.hub.load_state_dict_from_url(
            "https://github.com/Parskatt/RoMaV2/releases/download/weights/romav2.pt"
        )
        self.f = Descriptor(cfg.descriptor)
        self.matcher = Matcher(cfg.matcher)
        self.cfg = cfg
        self.anchor_width = cfg.anchor_width
        self.anchor_height = cfg.anchor_height
        self.refiners = Refiners(cfg.refiners)
        self.refiner_features = FineFeatures(cfg.refiner_features)
        self.to(device)
        self.eval()
        self.apply_setting(cfg.setting)
        self.name = cfg.name
        self.load_state_dict(weights)
        if cfg.compile:
            logger.info(f"Compiling {self.name}...")
            self.compile()
        # logger.info(f"{self.name} initialized.")

    def apply_setting(self, setting: Setting):
        if setting in ["mega1500", "scannet1500", "wxbs", "satast"]:
            self.H_lr = 800
            self.W_lr = 800
            self.H_hr = 1024
            self.W_hr = 1024
            self.bidirectional = True
            self.threshold = 0.05
            self.balanced_sampling = True
        elif setting == "turbo":
            self.H_lr = 320
            self.W_lr = 320
            self.H_hr = None
            self.W_hr = None
            self.bidirectional = False
            self.threshold = None
            self.balanced_sampling = True
        elif setting == "fast":
            self.H_lr = 512
            self.W_lr = 512
            self.H_hr = None
            self.W_hr = None
            self.bidirectional = False
            self.threshold = None
            self.balanced_sampling = True
        elif setting == "base":
            self.H_lr = 640
            self.W_lr = 640
            self.H_hr = None
            self.W_hr = None
            self.bidirectional = False
            self.threshold = None
            self.balanced_sampling = True
        elif setting == "precise":
            self.H_lr = 800
            self.W_lr = 800
            self.H_hr = 1280
            self.W_hr = 1280
            self.bidirectional = True
            self.threshold = None
            self.balanced_sampling = True
        else:
            raise TypeError(f"Invalid setting: {setting}")

    @torch.inference_mode()
    def _forward_from_features(
        self,
        f_list_A: list[torch.Tensor],
        img_A_lr: torch.Tensor,
        img_B_lr: torch.Tensor,
        img_A_hr: torch.Tensor | None = None,
        img_B_hr: torch.Tensor | None = None,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor] | torch.Tensor]:
        if torch.get_float32_matmul_precision() != "highest":
            raise RuntimeError("Float32 matmul precision must be set to highest")
        assert not self.training, "Currently only inference mode released"
        # assumes images between [0, 1]
        # init preds
        predictions = OrderedDict()
        f_B = self.f(img_B_lr)
        # match feats
        matcher_output = self.matcher(
            # The prediction head replaces the final feature in each list.
            # Keep both cached descriptor lists intact across matching pairs.
            list(f_list_A), list(f_B), img_A=img_A_lr, img_B=img_B_lr, bidirectional=self.bidirectional
        )
        # return matcher_output
        predictions["matcher"] = matcher_output
        warp_AB, confidence_AB = (
            matcher_output["warp_AB"],
            matcher_output["confidence_AB"],
        )
        if self.bidirectional:
            warp_BA, confidence_BA = (
                matcher_output["warp_BA"],
                matcher_output["confidence_BA"],
            )
        else:
            warp_BA = None
            confidence_BA = None
        # refine warp, maybe twice (if hr is available)
        for stage, (img_A, img_B) in enumerate(
            zip([img_A_lr, img_A_hr], [img_B_lr, img_B_hr])
        ):
            if img_A is None or img_B is None:
                continue
            B, C, H, W = img_A.shape
            scale_factor = torch.tensor(
                (W / self.anchor_width, H / self.anchor_height), device=img_A.device
            )
            refiner_features_A = self.refiner_features(img_A)
            refiner_features_B = self.refiner_features(img_B)
            for patch_size_str, refiner in self.refiners.items():
                patch_size = int(patch_size_str)
                zero_out_precision = (
                    img_A_hr is not None and patch_size == 4 and stage == 1
                )
                warp_AB, confidence_AB = _interpolate_warp_and_confidence(
                    warp=warp_AB,
                    confidence=confidence_AB,
                    H=H,
                    W=W,
                    patch_size=patch_size,
                    zero_out_precision=zero_out_precision,
                )
                if self.bidirectional:
                    warp_BA, confidence_BA = _interpolate_warp_and_confidence(
                        warp=warp_BA,
                        confidence=confidence_BA,
                        H=H,
                        W=W,
                        patch_size=patch_size,
                        zero_out_precision=zero_out_precision,
                    )

                f_patch_A = refiner_features_A[patch_size]
                f_patch_B = refiner_features_B[patch_size]
                refiner_output_AB = refiner(
                    f_A=f_patch_A,
                    f_B=f_patch_B,
                    prev_warp=warp_AB,
                    prev_confidence=confidence_AB,
                    scale_factor=scale_factor,
                )
                if self.bidirectional:
                    refiner_output_BA = refiner(
                        f_A=f_patch_B,
                        f_B=f_patch_A,
                        prev_warp=warp_BA,
                        prev_confidence=confidence_BA,
                        scale_factor=scale_factor,
                    )
                else:
                    refiner_output_BA = None
                predictions[f"refiner_{patch_size}_AB"] = refiner_output_AB
                predictions[f"refiner_{patch_size}_BA"] = refiner_output_BA
                warp_AB, confidence_AB = (
                    refiner_output_AB["warp"],
                    refiner_output_AB["confidence"],
                )
                if self.bidirectional:
                    warp_BA, confidence_BA = (
                        refiner_output_BA["warp"],
                        refiner_output_BA["confidence"],
                    )
            predictions["warp_AB"] = warp_AB
            predictions["confidence_AB"] = confidence_AB
            if self.bidirectional:
                predictions["warp_BA"] = warp_BA
                predictions["confidence_BA"] = confidence_BA
            else:
                predictions["warp_BA"] = None
                predictions["confidence_BA"] = None
        return predictions

    @torch.inference_mode()
    def forward(
        self,
        img_A_lr: torch.Tensor,
        img_B_lr: torch.Tensor,
        img_A_hr: torch.Tensor | None = None,
        img_B_hr: torch.Tensor | None = None,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor] | torch.Tensor]:
        f_list_A = self.f(img_A_lr)
        return self._forward_from_features(
            f_list_A=f_list_A,
            img_A_lr=img_A_lr,
            img_B_lr=img_B_lr,
            img_A_hr=img_A_hr,
            img_B_hr=img_B_hr,
        )

    def _load_image(self, img_like: ImageLike) -> torch.Tensor:
        if isinstance(img_like, str) or isinstance(img_like, Path):
            img_pil = Image.open(img_like)
            check_not_i16(img_pil)
            img_pil = img_pil.convert("RGB")
            img = torch.from_numpy(np.array(img_pil)).permute(2, 0, 1).to(device)
        elif isinstance(img_like, Image.Image):
            img = torch.from_numpy(np.array(img_like)).permute(2, 0, 1).to(device)
        elif isinstance(img_like, np.ndarray):
            assert img_like.shape[-1] == 3, (
                f"Image must have 3 channels, but got shape {img_like.shape=}"
            )
            img = torch.from_numpy(img_like).permute(2, 0, 1).to(device)
        elif isinstance(img_like, torch.Tensor):
            assert img_like.shape[1] == 3, (
                f"Image must have 3 channels, but got shape {img_like.shape=}"
            )
            img = img_like
        else:
            raise ValueError(f"Unsupported image type: {type(img_like)}")

        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        if len(img.shape) == 3:
            img = img[None]
        return img

    def _resize_match_image(
        self,
        img: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        img_lr = F.interpolate(
            img,
            size=(self.H_lr, self.W_lr),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        if self.H_hr is not None and self.W_hr is not None:
            img_hr = F.interpolate(
                img,
                size=(self.H_hr, self.W_hr),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
        else:
            img_hr = None
        return img_lr, img_hr

    @torch.inference_mode()
    def _match_core(
        self,
        f_list_A: list[torch.Tensor],
        img_A_lr: torch.Tensor,
        img_B_lr: torch.Tensor,
        img_A_hr: torch.Tensor | None = None,
        img_B_hr: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        preds = self._forward_from_features(
            f_list_A=f_list_A,
            img_A_lr=img_A_lr,
            img_B_lr=img_B_lr,
            img_A_hr=img_A_hr,
            img_B_hr=img_B_hr,
        )

        warp_AB = preds["warp_AB"]
        confidence_AB = preds["confidence_AB"]
        warp_BA = preds["warp_BA"]
        confidence_BA = preds["confidence_BA"]
        overlap_AB, precision_AB = _map_confidence(
            confidence=confidence_AB, threshold=self.threshold
        )
        if self.bidirectional:
            overlap_BA, precision_BA = _map_confidence(
                confidence=confidence_BA, threshold=self.threshold
            )
        else:
            overlap_BA = None
            precision_BA = None

        return {
            "warp_AB": warp_AB,
            "confidence_AB": confidence_AB,
            "overlap_AB": overlap_AB,
            "precision_AB": precision_AB,
            "warp_BA": warp_BA,
            "confidence_BA": confidence_BA,
            "overlap_BA": overlap_BA,
            "precision_BA": precision_BA,
        }
