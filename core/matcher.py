"""RoMaV2 matcher wrapper."""
from __future__ import annotations

from collections import OrderedDict

import gc
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import logging
import torch
from PIL import Image

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
_ROMA_SRC = _ROOT / "RoMaV2" / "src"
if str(_ROMA_SRC) not in sys.path:
    sys.path.insert(0, str(_ROMA_SRC))


_ROMA_WEIGHTS_FILE = "romav2.pt"


def _dedupe_paths(paths: List[str]) -> List[str]:
    unique: List[str] = []
    seen = set()
    for path in paths:
        if not path:
            continue
        norm = os.path.normcase(os.path.normpath(path))
        if norm in seen:
            continue
        seen.add(norm)
        unique.append(path)
    return unique


def _roma_checkpoint_dirs() -> List[str]:
    candidates: List[str] = []
    try:
        candidates.append(os.path.join(torch.hub.get_dir(), "checkpoints"))
    except Exception:
        pass

    torch_home = os.getenv("TORCH_HOME")
    if torch_home:
        candidates.append(os.path.join(os.path.expanduser(torch_home), "hub", "checkpoints"))

    xdg_cache_home = os.getenv("XDG_CACHE_HOME", "~/.cache")
    candidates.append(os.path.join(os.path.expanduser(xdg_cache_home), "torch", "hub", "checkpoints"))

    if os.name == "nt":
        local_app_data = os.getenv("LOCALAPPDATA")
        if local_app_data:
            candidates.append(os.path.join(local_app_data, "torch", "hub", "checkpoints"))
        user_profile = os.getenv("USERPROFILE")
        if user_profile:
            candidates.append(os.path.join(user_profile, "AppData", "Local", "torch", "hub", "checkpoints"))

    # Preserve order while removing duplicates across path style differences.
    return _dedupe_paths(candidates)


def romav2_cached_weights_paths() -> List[str]:
    return [os.path.join(root, _ROMA_WEIGHTS_FILE) for root in _roma_checkpoint_dirs()]


def has_cached_romav2_weights() -> bool:
    return any(os.path.isfile(path) for path in romav2_cached_weights_paths())


class _CachedFeatures(torch.nn.Module):
    """Bounded inference cache for features of retained, immutable image tensors."""

    def __init__(self, module, capacity=24):
        super().__init__()
        self.module = module
        self.capacity = capacity
        self.cache = OrderedDict()

    def forward(self, image):
        key = id(image)
        cached = self.cache.get(key)
        if cached is not None and cached[0] is image:
            self.cache.move_to_end(key)
            return cached[1]
        features = self.module(image)
        self.cache[key] = (image, features)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return features


class RomaMatcher:
    """Wrapper around RoMaV2 for dense matching."""

    def __init__(self, device: str = "cuda", mode: str = "outdoor", setting: str = "fast"):
        from romav2 import RoMaV2

        del mode  # Legacy arg kept for API compatibility.
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
        descriptor_config = self.model.cfg.descriptor
        if descriptor_config.name == "dinov3_vitl16" and all(index > 0 for index in descriptor_config.layer_idx):
            # Intermediate features precede the unused tail of the transformer.
            self.model.f.blocks = self.model.f.blocks[:max(descriptor_config.layer_idx) + 1]
        self.model.to(self.device)
        self.model.eval()
        lr_pixels = self.model.H_lr * self.model.W_lr
        hr_pixels = (self.model.H_hr or 0) * (self.model.W_hr or 0)
        # Scale entry limits with resolution so high-detail modes retain less.
        self._image_capacity = max(2, 256 * 512**2 // (lr_pixels + hr_pixels))
        self.model.f = _CachedFeatures(self.model.f, max(2, 256 * 512**2 // lr_pixels))
        self.model.refiner_features = _CachedFeatures(
            self.model.refiner_features, max(2, 24 * 512**2 // max(lr_pixels, hr_pixels))
        )
        self._image_cache = OrderedDict()
        self.sample_thresh = 0.9
        self.w_resized = self.model.W_lr
        self.h_resized = self.model.H_lr
        self._grid_cache: Dict[Tuple[int, int], torch.Tensor] = {}
        logger.info(
            f"RoMaV2 initialized (setting={setting}, H_lr={self.model.H_lr}, W_lr={self.model.W_lr}, device={device})"
        )

    def close(self) -> None:
        """Release model weights and CUDA allocations for this matcher."""
        model = getattr(self, "model", None)
        if model is None:
            return
        self._image_cache.clear()
        model.f.cache.clear()
        model.refiner_features.cache.clear()
        try:
            model.to("cpu")
        except Exception:
            pass
        self.model = None
        self._grid_cache.clear()
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

    def _get_reference_grid(self, H: int, W: int) -> torch.Tensor:
        key = (H, W)
        grid = self._grid_cache.get(key)
        if grid is not None and grid.device == self.device:
            return grid

        yy = torch.linspace(-1 + 1 / H, 1 - 1 / H, H, device=self.device)
        xx = torch.linspace(-1 + 1 / W, 1 - 1 / W, W, device=self.device)
        yy, xx = torch.meshgrid(yy, xx, indexing="ij")
        grid = torch.stack([xx, yy], dim=-1).contiguous()
        self._grid_cache[key] = grid
        return grid

    def _prepare_image(self, image, key):
        if key is not None and key in self._image_cache:
            self._image_cache.move_to_end(key)
            return self._image_cache[key]
        prepared = self.model._resize_match_image(self.model._load_image(image))
        if key is not None:
            self._image_cache[key] = prepared
            if len(self._image_cache) > self._image_capacity:
                self._image_cache.popitem(last=False)
        return prepared

    @torch.inference_mode()
    def _match_grids_batch_cached_reference(
        self,
        imA: Image.Image,
        imB_list: List[Image.Image],
        reference_key=None,
        neighbor_keys=None,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        model = self.model
        if model is None:
            raise RuntimeError("RoMaV2 model has been released; create a new matcher before matching.")
        img_A_lr, img_A_hr = self._prepare_image(imA, reference_key)
        f_list_A = model.f(img_A_lr)
        results: List[Tuple[torch.Tensor, torch.Tensor]] = []

        keys = neighbor_keys if neighbor_keys is not None else [None] * len(imB_list)
        for imB, key in zip(imB_list, keys):
            img_B_lr, img_B_hr = self._prepare_image(imB, key)
            preds = model._match_core(
                f_list_A=f_list_A,
                img_A_lr=img_A_lr,
                img_B_lr=img_B_lr,
                img_B_hr=img_B_hr,
                img_A_hr=img_A_hr,
            )
            warp_AB_hw = preds["warp_AB"][0]
            overlap_AB_hw = preds["overlap_AB"][0].squeeze(-1)
            H, W = overlap_AB_hw.shape
            gridA = self._get_reference_grid(H, W)
            warp = torch.cat([gridA, warp_AB_hw], dim=-1)
            results.append((warp.contiguous(), overlap_AB_hw.contiguous()))
            del preds, warp_AB_hw, overlap_AB_hw, warp

        del f_list_A
        del img_A_lr
        if img_A_hr is not None:
            del img_A_hr

        return results

    @torch.inference_mode()
    def match_grids(self, imA: Image.Image, imB: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_results = self.match_grids_batch(imA, [imB])
        if not batch_results:
            raise RuntimeError("RoMaV2 returned no matches for the requested pair.")
        return batch_results[0]

    @torch.inference_mode()
    def match_grids_batch(
        self, imA: Image.Image, imB_list: List[Image.Image], *,
        reference_key=None, neighbor_keys=None,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Match a batch; optional keys identify immutable images within this job."""
        if self.model is None:
            raise RuntimeError("RoMaV2 model has been released; create a new matcher before matching.")
        if neighbor_keys is not None and len(neighbor_keys) != len(imB_list):
            raise ValueError("Each neighbor image must have one cache key.")
        if not imB_list:
            return []
        torch.set_float32_matmul_precision("highest")
        return self._match_grids_batch_cached_reference(imA, imB_list, reference_key, neighbor_keys)
