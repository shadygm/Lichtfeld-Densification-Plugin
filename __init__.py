# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Dense Point Cloud Initialization Plugin for LichtFeld Studio.

Uses RoMa v2 to densify sparse COLMAP reconstructions with accurate,
high-quality point clouds.
"""

# ── Sanitize sys.argv before importing any C-extensions that link gflags/glog.
# pycolmap embeds gflags which calls ParseCommandLineFlags(sys.argv) during
# module init.  On Windows (where multiprocessing uses 'spawn') every DataLoader
# worker re-imports all modules and re-triggers gflags parsing.  If the host app
# set sys.argv to e.g. ['LichtFeld.exe', '-s', 'scene.lfs'], gflags fails with
# "Parse error: Flag could not be matched: 's'".  Stripping everything after the
# program name prevents this while keeping the program name intact.
import sys as _sys
_original_argv = _sys.argv
_sys.argv = _sys.argv[:1]

import lichtfeld as lf

from .densify import dense_init
from .core.config import DensePipelineConfig
from .panels.densification import (
    DensificationPanel,
    DensifyResult,
    DensifyJob,
    DensifyStage,
)

# Restore original argv now that gflags-bearing modules have been imported.
_sys.argv = _original_argv
del _original_argv

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
    "DensePipelineConfig",
    "DensificationPanel",
    "DensifyResult",
    "DensifyJob",
    "DensifyStage",
]
