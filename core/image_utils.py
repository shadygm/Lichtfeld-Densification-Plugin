"""Image and filesystem helpers for densification."""
from __future__ import annotations

import os
from typing import Optional, Tuple

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


def _load_exr(path: str) -> Image.Image:
    """Load an OpenEXR file (any compression) as an 8-bit RGB PIL image.

    Pillow has no EXR decoder. Linear values are clipped to [0, 1] without a
    transfer curve, matching how LichtFeld feeds EXR pixels to training.
    """
    import OpenEXR

    with OpenEXR.File(path, separate_channels=True) as exr:
        channels = {name: ch.pixels for name, ch in exr.channels().items()}

    def pick(suffix: str) -> Optional[np.ndarray]:
        if suffix in channels:
            return channels[suffix]
        for name in sorted(channels):
            if name.rsplit(".", 1)[-1] == suffix:
                return channels[name]
        return None

    if all(name in channels for name in ("R", "G", "B")):
        r, g, b = (channels[name] for name in ("R", "G", "B"))
    else:
        prefixes = sorted({name.rsplit(".", 1)[0] for name in channels if name.endswith(".R")})
        prefix = next((p for p in prefixes if all(f"{p}.{suffix}" in channels for suffix in ("R", "G", "B"))), None)
        if prefix is not None:
            r, g, b = (channels[f"{prefix}.{suffix}"] for suffix in ("R", "G", "B"))
        else:
            y = pick("Y")
            if y is None:
                raise ValueError(f"EXR has no RGB or Y channels: {sorted(channels)}")
            r = g = b = y
    rgb = np.stack([r, g, b], axis=-1).astype(np.float32)
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
    return Image.fromarray(to_uint8_rgb(rgb), mode="RGB")


def _open_image(path: str) -> Image.Image:
    """Open an image with Pillow, routing EXR files through OpenEXR."""
    if path.lower().endswith(".exr"):
        return _load_exr(path)
    return Image.open(path)


def find_image(root: str, name: str) -> str:
    """Find an image on disk using absolute or basename lookup."""
    candidate = os.path.join(root, name)
    if os.path.isfile(candidate):
        return candidate
    fallback = os.path.join(root, os.path.basename(name))
    if os.path.isfile(fallback):
        return fallback
    raise FileNotFoundError(f"Image '{name}' not found under {root}")


def load_mask_resized_np(
    path: str,
    size: Tuple[int, int],
    *,
    invert: bool = False,
    threshold: float = 0.5,
) -> np.ndarray:
    """Load a binary mask (uint8 {0,1}) and resize to the requested size.

    Mask convention: 1 = "interesting / keep", 0 = ignore.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    im = _open_image(path)
    # Convert to single channel; handles RGB/RGBA masks gracefully.
    im = im.convert("L")
    if im.size != size:
        # Nearest preserves hard mask edges.
        im = im.resize(size, Image.NEAREST)
    arr = np.asarray(im, dtype=np.uint8)
    # Normalize into boolean using threshold in [0,1].
    keep = (arr.astype(np.float32) / 255.0) > float(threshold)
    if invert:
        keep = ~keep
    return keep.astype(np.uint8)


def apply_mask_to_rgb(im: Image.Image, mask01: np.ndarray) -> Image.Image:
    """Apply a {0,1} mask to an RGB PIL image (masked pixels become black)."""
    if im.mode != "RGB":
        im = im.convert("RGB")
    rgb = np.asarray(im, dtype=np.uint8)
    if mask01.ndim != 2:
        raise ValueError(f"mask01 must be HxW, got shape={mask01.shape}")
    if rgb.shape[0] != mask01.shape[0] or rgb.shape[1] != mask01.shape[1]:
        raise ValueError(
            f"mask01 shape {mask01.shape} must match image shape {(rgb.shape[0], rgb.shape[1])}"
        )
    rgb = rgb.copy()
    rgb[mask01 == 0] = 0
    return Image.fromarray(rgb, mode="RGB")


def load_rgb_resized(path: str, size: Tuple[int, int]) -> Image.Image:
    """Load an RGB image and resize it to the requested size."""
    im = _open_image(path).convert("RGB")
    if im.size != size:
        im = im.resize(size, Image.BILINEAR)
    return im
