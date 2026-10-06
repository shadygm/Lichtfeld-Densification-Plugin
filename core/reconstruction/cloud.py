"""Final cloud handed from a background job to the scene thread."""
from dataclasses import dataclass

import numpy as np


@dataclass
class DenseCloud:
    points: np.ndarray
    colors: np.ndarray
    output_path: str | None = None
