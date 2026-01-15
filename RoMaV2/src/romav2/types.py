from __future__ import annotations
from typing import Callable
from dataclasses import dataclass
from pathlib import Path
import torch
from typing import Literal
import numpy as np
from PIL import Image


@dataclass
class Batch:
    img_A: torch.Tensor
    img_B: torch.Tensor
    depth_A: torch.Tensor
    depth_B: torch.Tensor
    K_A: torch.Tensor
    K_B: torch.Tensor
    pose_A: torch.Tensor
    pose_B: torch.Tensor
    T_AB: torch.Tensor
    img_A_path: Path
    img_B_path: Path
    source: GTSource | list[GTSource]
    warp_A_to_B: torch.Tensor
    warp_B_to_A: torch.Tensor
    mask_A_to_B: torch.Tensor
    mask_B_to_A: torch.Tensor

    def to(self, device: torch.device) -> Batch:
        return Batch(
            img_A=self.img_A.to(device),
            img_B=self.img_B.to(device),
            K_A=self.K_A.to(device),
            K_B=self.K_B.to(device),
            pose_A=self.pose_A.to(device),
            pose_B=self.pose_B.to(device),
            depth_A=self.depth_A.to(device),
            depth_B=self.depth_B.to(device),
            warp_A_to_B=self.warp_A_to_B.to(device),
            warp_B_to_A=self.warp_B_to_A.to(device),
            T_AB=self.T_AB.to(device),
            img_A_path=self.img_A_path,
            img_B_path=self.img_B_path,
            source=self.source,
            mask_A_to_B=self.mask_A_to_B.to(device),
            mask_B_to_A=self.mask_B_to_A.to(device),
        )

    @classmethod
    def collate(cls, samples: list[Batch]) -> Batch:
        keys = samples[0].__dict__.keys()
        batch = {}
        for k in keys:
            if isinstance(samples[0].__dict__[k], torch.Tensor):
                batch[k] = torch.stack([s.__dict__[k] for s in samples])
            else:
                batch[k] = [s.__dict__[k] for s in samples]
        return Batch(**batch)  # type: ignore[missing-arguments]


GTSource = Literal["depth", "warp"]
HeadType = Literal["dpt-no-pos"]
SampleMode = Literal["frame_distance", "overlap"]
ConfidenceMode = Literal["covis", "frame", "positive"]
NormType = Literal["batch"]
RefinersType = Literal["roma-4-pow2"]
MatcherStyle = Literal["romav2"]
DescriptorName = Literal["dinov3_vitl16", "dinov2_vitl14"]
Normalizer = Callable[[torch.Tensor], torch.Tensor]
OptimizerName = Literal["adamw"]
ImageLike = torch.Tensor | np.ndarray | str | Path | Image.Image
Setting = Literal[
    "mega1500", "scannet1500", "wxbs", "satast", "base", "precise", "turbo", "fast"
]
