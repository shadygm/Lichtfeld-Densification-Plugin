"""Camera data structures."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class CameraRecord:
    uid: int
    image_path: str
    width: int
    height: int
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    P: np.ndarray
    C: np.ndarray

    def flat_pose(self) -> np.ndarray:
        """Return flattened 4x4 pose for clustering/nearest neighbors."""
        T = np.eye(4)
        T[:3, :3] = self.R
        T[:3, 3] = self.t.reshape(3)
        return T.reshape(-1)
