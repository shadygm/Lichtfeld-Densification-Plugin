from __future__ import annotations
from typing import Callable
from pathlib import Path
import torch
from typing import Literal
import numpy as np
from PIL import Image


HeadType = Literal["dpt-no-pos"]
NormType = Literal["batch"]
RefinersType = Literal["roma-4-pow2"]
MatcherStyle = Literal["romav2"]
DescriptorName = Literal["dinov3_vitl16", "dinov2_vitl14"]
Normalizer = Callable[[torch.Tensor], torch.Tensor]
ImageLike = torch.Tensor | np.ndarray | str | Path | Image.Image
Setting = Literal[
    "mega1500", "scannet1500", "wxbs", "satast", "base", "precise", "turbo", "fast"
]
