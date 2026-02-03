# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Dense Point Cloud Initialization Plugin for LichtFeld Studio.

Uses RoMa v2 to densify sparse COLMAP reconstructions with accurate,
high-quality point clouds.
"""

import lichtfeld as lf

from .densify import dense_init
from .panels.densification import (
    DensificationPanel,
    DensifyConfig,
    DensifyResult,
    DensifyJob,
    DensifyStage,
)

_classes = [DensificationPanel]


def on_load():
    """Called when plugin loads."""
    for cls in _classes:
        lf.register_class(cls)
    lf.log.info("Dense Initialization plugin loaded")


def on_unload():
    """Called when plugin unloads."""
    for cls in reversed(_classes):
        lf.unregister_class(cls)
    lf.log.info("Dense Initialization plugin unloaded")


__all__ = [
    "dense_init",
    "DensificationPanel",
    "DensifyConfig",
    "DensifyResult",
    "DensifyJob",
    "DensifyStage",
]
