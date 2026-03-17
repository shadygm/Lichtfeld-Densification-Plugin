# SPDX-FileCopyrightText: 2025 Shady Gmira
# SPDX-License-Identifier: GPL-3.0-or-later
"""Floating debug panel for match visualization during densification."""

from pathlib import Path
from typing import Optional

import lichtfeld as lf

from ..core.debug_viz import MatchDebugState, MatchPreview


class DebugMatchesPanel(lf.ui.Panel):
    id = "densify.debug_matches"
    label = "Debug Matches"
    space = lf.ui.PanelSpace.FLOATING
    order = 99
    size = (340.0, 220.0)
    template = str(Path(__file__).resolve().with_name("debug_matches.rml"))
    update_interval_ms = 100

    # Shared debug state – set by DensificationPanel before enabling
    _debug_state: Optional[MatchDebugState] = None

    def __init__(self) -> None:
        self._handle = None
        self._doc = None
        self._last_pair_index: int = -1

    # ── Class-level helpers shared with DensificationPanel ──

    @classmethod
    def set_debug_state(cls, state: MatchDebugState) -> None:
        cls._debug_state = state

    # ── Retained lifecycle ──────────────────────────────

    def on_mount(self, doc):
        self._doc = doc

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("debug_matches")
        if model is None:
            return

        model.bind_func("has_preview", self._has_preview)
        model.bind_func("pair_index", self._pair_index)
        model.bind_func("total_pairs", self._total_pairs)
        model.bind_func("ref_label", self._ref_label)
        model.bind_func("ref_id", self._ref_id)
        model.bind_func("nbr_label", self._nbr_label)
        model.bind_func("nbr_id", self._nbr_id)
        model.bind_func("match_count", self._match_count)

        model.bind(
            "auto_step",
            lambda: self._debug_state.is_auto_step() if self._debug_state else True,
            self._set_auto_step,
        )

        model.bind_event("do_step", self._on_step)

        self._handle = model.get_handle()

    def on_update(self, doc):
        preview = self._latest()
        if preview is None:
            if self._last_pair_index != -1:
                self._last_pair_index = -1
                self._dirty_all()
                return True
            return False

        if preview.pair_index != self._last_pair_index:
            self._last_pair_index = preview.pair_index
            self._dirty_all()
            return True
        return False

    def on_unmount(self, doc):
        doc.remove_data_model("debug_matches")
        self._handle = None
        self._doc = None

    # ── Data accessors ──────────────────────────────────

    def _latest(self) -> Optional[MatchPreview]:
        if self._debug_state:
            return self._debug_state.latest()
        return None

    def _has_preview(self) -> bool:
        return self._latest() is not None

    def _pair_index(self) -> str:
        p = self._latest()
        return str(p.pair_index + 1) if p else "0"

    def _total_pairs(self) -> str:
        p = self._latest()
        return str(p.total_pairs) if p else "0"

    def _ref_label(self) -> str:
        p = self._latest()
        return p.ref_label if p else ""

    def _ref_id(self) -> str:
        p = self._latest()
        return str(p.ref_id) if p else ""

    def _nbr_label(self) -> str:
        p = self._latest()
        return p.nbr_label if p else ""

    def _nbr_id(self) -> str:
        p = self._latest()
        return str(p.nbr_id) if p else ""

    def _match_count(self) -> str:
        p = self._latest()
        return f"{p.match_count:,}" if p else "0"

    # ── Events ──────────────────────────────────────────

    def _set_auto_step(self, value):
        if self._debug_state:
            self._debug_state.set_auto_step(bool(value))
            if bool(value):
                self._debug_state.release_waiters()

    def _on_step(self, handle, event, args):
        if self._debug_state:
            self._debug_state.step_once()

    # ── Helpers ─────────────────────────────────────────

    def _dirty_all(self):
        if self._handle:
            self._handle.dirty_all()
