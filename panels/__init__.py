# SPDX-FileCopyrightText: 2025 Shady Gmira
# SPDX-License-Identifier: GPL-3.0-or-later
"""Dense Initialization plugin panels."""

import importlib


def __getattr__(name):
    if name == "DensePipelineConfig":
        from ..core.pipeline.config import DensePipelineConfig

        return DensePipelineConfig
    if name in {"DensificationPanel", "DensifyResult", "DensifyJob", "DensifyStage"}:
        module = importlib.import_module(f"{__name__}.densification")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "DensificationPanel",
    "DensePipelineConfig",
    "DensifyResult",
    "DensifyJob",
    "DensifyStage",
]
