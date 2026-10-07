"""Choose the point-cloud API supported by the running LichtFeld build."""
from dataclasses import dataclass
import re
from typing import Protocol

import lichtfeld as lf
import numpy as np


class CloudUpdateTicket(Protocol):
    state: str
    error: str

    def cancel(self) -> bool: ...


@dataclass
class SynchronousUpdateTicket:
    # Also protects snapshots during synchronous scene-mutation callbacks.
    state: str = "uploading"
    error: str = ""

    def cancel(self) -> bool:
        return False


def supports_async_publication(point_cloud) -> bool:
    versions = (getattr(lf, "__version__", ""),
                getattr(getattr(lf, "build_info", None), "version", ""))
    for version in versions:
        match = re.match(r"^v?(\d+)\.(\d+)\.(\d+)(?:$|[-+])", str(version))
        if match:
            if tuple(map(int, match.groups())) <= (0, 5, 4):
                return False
            break
    return callable(getattr(point_cloud, "set_data_async", None))


def set_point_cloud_data(point_cloud, points, colors) -> CloudUpdateTicket:
    if not isinstance(points, lf.Tensor):
        points = np.ascontiguousarray(points, dtype=np.float32)
    if not isinstance(colors, lf.Tensor):
        colors = np.asarray(colors)
        colors = np.ascontiguousarray(colors, dtype=np.uint8 if colors.dtype == np.uint8 else np.float32)
    if supports_async_publication(point_cloud):
        # Native async publication retains these immutable input owners.
        return point_cloud.set_data_async(points, colors, queue_policy="latest")

    if not isinstance(points, lf.Tensor):
        points = lf.Tensor.from_numpy(points)
    if not isinstance(colors, lf.Tensor):
        colors = lf.Tensor.from_numpy(colors)
    point_cloud.set_data(points, colors)
    return SynchronousUpdateTicket(state="published")
