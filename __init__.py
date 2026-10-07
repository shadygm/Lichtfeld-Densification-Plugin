# SPDX-FileCopyrightText: 2025 Shady Gmira
# SPDX-License-Identifier: GPL-3.0-or-later
"""Dense Point Cloud Initialization Plugin for LichtFeld Studio.

Uses RoMa v2 to densify sparse COLMAP reconstructions with accurate,
high-quality point clouds.
"""

import lichtfeld as lf

_classes = []
_densify_job_cls = None


def _load_panel_classes():
    """Import panel classes only when LFS asks this plugin to register UI."""
    global _classes, _densify_job_cls
    if not _classes:
        from .panels.densification import DensificationPanel, DensifyJob

        _classes = [DensificationPanel]
        _densify_job_cls = DensifyJob
    return _classes


def on_load():
    """Called when plugin loads."""
    for cls in _load_panel_classes():
        lf.register_class(cls)
    lf.log.info("Dense Initialization plugin loaded")


def on_unload():
    """Called when plugin unloads."""
    global _classes, _densify_job_cls
    if _densify_job_cls is not None:
        try:
            _densify_job_cls.cancel_all(timeout=5.0)
        except Exception as exc:
            lf.log.warn(f"Failed to cancel active densification jobs on unload: {exc}")

    for cls in reversed(_classes):
        lf.unregister_class(cls)
    _classes = []
    _densify_job_cls = None
    lf.log.info("Dense Initialization plugin unloaded")


def __getattr__(name):
    if name == "dense_init":
        from .densify import dense_init

        return dense_init
    if name == "DensePipelineConfig":
        from .core.pipeline.config import DensePipelineConfig

        return DensePipelineConfig
    if name in {"DensificationPanel", "DensifyResult", "DensifyJob", "DensifyStage"}:
        from .panels import densification

        return getattr(densification, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "dense_init",
    "DensePipelineConfig",
    "DensificationPanel",
    "DensifyResult",
    "DensifyJob",
    "DensifyStage",
]
