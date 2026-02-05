"""Image and filesystem helpers for densification."""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Tuple

import numpy as np
from PIL import Image


def image_dir(scene_root: str, preferred: str) -> str:
    """Detect the most appropriate images directory under a scene root."""
    cand = os.path.join(scene_root, preferred)
    if os.path.isdir(cand):
        return cand
    for alt in ["images_4", "images_2", "images_8", "images"]:
        alt_path = os.path.join(scene_root, alt)
        if os.path.isdir(alt_path):
            return alt_path
    raise FileNotFoundError("Could not locate an images directory under scene_root.")


def to_uint8_rgb(arr_float01: np.ndarray) -> np.ndarray:
    """Convert RGB float array in [0, 1] to uint8."""
    return np.clip(np.round(arr_float01 * 255.0), 0, 255).astype(np.uint8)


def find_image(root: str, name: str) -> str:
    """Find an image on disk using absolute or basename lookup."""
    candidate = os.path.join(root, name)
    if os.path.isfile(candidate):
        return candidate
    fallback = os.path.join(root, os.path.basename(name))
    if os.path.isfile(fallback):
        return fallback
    raise FileNotFoundError(f"Image '{name}' not found under {root}")


@lru_cache(maxsize=4096)
def load_rgb_resized(path: str, size: Tuple[int, int]) -> Image.Image:
    """Load an RGB image and resize it to the requested size."""
    im = Image.open(path).convert("RGB")
    if im.size != size:
        im = im.resize(size, Image.BILINEAR)
    return im
