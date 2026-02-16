"""Debug helpers for match visualization."""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

import numpy as np


@dataclass
class MatchPreview:
    """Compact representation of a single reference/neighbor match preview."""

    ref_id: int
    nbr_id: int
    ref_label: str
    nbr_label: str
    left_image: np.ndarray  # HxWx3 uint8 RGB
    right_image: np.ndarray  # HxWx3 uint8 RGB
    matches: np.ndarray  # Nx4 float32: (xA, yA, xB, yB) in pixel coords
    cert_norm: np.ndarray  # N float32 in [0, 1] for coloring
    match_count: int
    pair_index: int
    total_pairs: int


class MatchDebugState:
    """Controller for live match previews and stepping.

    The pipeline runs on a single background thread (DensifyJob); the UI
    draws on the main thread.  ``submit_preview`` blocks the pipeline
    thread when manual stepping is active until ``step_once`` is called
    from the UI.
    """

    def __init__(self, max_history: int = 4) -> None:
        self._enabled = False
        self._auto_step = True
        self._latest: Optional[MatchPreview] = None
        self._history: Deque[MatchPreview] = deque(maxlen=max_history)
        self._total_pairs: int = 0
        self._lock = threading.Lock()
        self._step_event = threading.Event()
        self._step_event.set()

        # --- Match visibility controls ---
        self._max_visible_matches: int = 0       # 0 = show all
        self._single_match_mode: bool = False    # iterate one match at a time
        self._current_match_index: int = 0       # index for single-match mode

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._enabled = enabled
            if not enabled:
                self._history.clear()
                self._latest = None
                self._auto_step = True
                self._step_event.set()

    def is_enabled(self) -> bool:
        with self._lock:
            return self._enabled

    def set_auto_step(self, auto: bool) -> None:
        with self._lock:
            self._auto_step = auto
            if auto:
                self._step_event.set()

    def is_auto_step(self) -> bool:
        with self._lock:
            return self._auto_step

    def set_total_pairs(self, total: int) -> None:
        with self._lock:
            self._total_pairs = max(0, int(total))

    def total_pairs(self) -> int:
        with self._lock:
            return self._total_pairs

    def submit_preview(self, preview: MatchPreview) -> None:
        """Push a preview and optionally block until the user steps."""
        with self._lock:
            if not self._enabled:
                return
            self._latest = preview
            self._history.append(preview)
            auto = self._auto_step
        if not auto:
            self._step_event.clear()
            self._step_event.wait()

    def step_once(self) -> None:
        self._step_event.set()

    def latest(self) -> Optional[MatchPreview]:
        with self._lock:
            return self._latest

    def history(self) -> List[MatchPreview]:
        with self._lock:
            return list(self._history)

    def release_waiters(self) -> None:
        """Unblock any waiting pipeline thread."""
        self._step_event.set()

    # --- Match visibility API ---

    def set_max_visible_matches(self, n: int) -> None:
        with self._lock:
            self._max_visible_matches = max(0, int(n))

    def max_visible_matches(self) -> int:
        with self._lock:
            return self._max_visible_matches

    def set_single_match_mode(self, enabled: bool) -> None:
        with self._lock:
            self._single_match_mode = enabled
            if enabled:
                self._current_match_index = 0

    def is_single_match_mode(self) -> bool:
        with self._lock:
            return self._single_match_mode

    def current_match_index(self) -> int:
        with self._lock:
            return self._current_match_index

    def set_current_match_index(self, idx: int) -> None:
        with self._lock:
            self._current_match_index = max(0, int(idx))

    def next_match(self, total: int) -> None:
        """Advance to the next match, wrapping around."""
        with self._lock:
            if total > 0:
                self._current_match_index = (self._current_match_index + 1) % total

    def prev_match(self, total: int) -> None:
        """Go to the previous match, wrapping around."""
        with self._lock:
            if total > 0:
                self._current_match_index = (self._current_match_index - 1) % total

    def visible_match_indices(self, total: int) -> List[int]:
        """Return the list of match indices that should be drawn."""
        with self._lock:
            if total <= 0:
                return []
            if self._single_match_mode:
                idx = min(self._current_match_index, total - 1)
                return [idx]
            if self._max_visible_matches <= 0 or self._max_visible_matches >= total:
                return list(range(total))
            return list(range(self._max_visible_matches))
