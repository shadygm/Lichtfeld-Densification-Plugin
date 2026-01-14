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

_panel_class = None


def on_load():
    """Called when plugin loads."""
    global _panel_class

    _panel_class = DensificationPanel
    lf.ui.register_panel(DensificationPanel)
    lf.log.info("Dense Initialization plugin loaded")


def on_unload():
    """Called when plugin unloads."""
    global _panel_class

    if _panel_class:
        lf.ui.unregister_panel(_panel_class)
        _panel_class = None
    lf.log.info("Dense Initialization plugin unloaded")


__all__ = [
    "dense_init",
    "DensificationPanel",
    "DensifyConfig",
    "DensifyResult",
    "DensifyJob",
    "DensifyStage",
]
