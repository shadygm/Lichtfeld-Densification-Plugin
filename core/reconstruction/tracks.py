"""Numeric observation tracks; lengths alone suffice for UI-only jobs."""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ObservationTracks:
    lengths: np.ndarray  # uint32 [points]
    camera_ids: np.ndarray | None = None  # int64 [observations]
    pixels: np.ndarray | None = None  # float64 [observations, 2]

    def __post_init__(self):
        if self.lengths.dtype != np.uint32 or self.lengths.ndim != 1:
            raise ValueError("Track lengths must be a uint32 vector")
        if (self.camera_ids is None) != (self.pixels is None):
            raise ValueError("Camera IDs and pixels must be retained together")
        if self.has_observations:
            count = int(self.lengths.sum(dtype=np.int64))
            if self.camera_ids.dtype != np.int64 or self.camera_ids.shape != (count,):
                raise ValueError("Camera IDs must be an int64 observation vector")
            if self.pixels.dtype != np.float64 or self.pixels.shape != (count, 2):
                raise ValueError("Pixels must be float64 [observations, 2]")

    def __len__(self):
        return len(self.lengths)

    @property
    def has_observations(self):
        return self.camera_ids is not None

    @cached_property
    def offsets(self):
        offsets = np.empty(len(self) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(self.lengths, dtype=np.int64, out=offsets[1:])
        return offsets

    def require_observations(self):
        if not self.has_observations:
            raise ValueError("This run retained track lengths only, not observations")

    def point_indices(self):
        return np.repeat(np.arange(len(self), dtype=np.int64), self.lengths)

    def camera_groups(self):
        """Yield camera IDs and stable observation indices without Python rows."""
        self.require_observations()
        order = np.argsort(self.camera_ids, kind="stable")
        ids = self.camera_ids[order]
        boundaries = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(ids)]
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            if start < stop:
                yield int(ids[start]), order[start:stop]

    def select(self, selection):
        """Gather complete tracks in point order, including repeats and empties."""
        indices = np.arange(len(self), dtype=np.int64)[selection]
        if indices.ndim != 1:
            raise ValueError("Point selection must be one-dimensional")
        lengths = self.lengths[indices]
        if not self.has_observations:
            return ObservationTracks(lengths)
        selected = ObservationTracks(lengths)
        count = int(selected.offsets[-1])
        rows = (np.repeat(self.offsets[indices] - selected.offsets[:-1], lengths)
                + np.arange(count, dtype=np.int64))
        return ObservationTracks(lengths, self.camera_ids[rows], self.pixels[rows])

    @classmethod
    def concatenate(cls, parts: Sequence[ObservationTracks]):
        if not parts:
            return cls(np.empty(0, dtype=np.uint32))
        if len(parts) == 1:
            return parts[0]
        full = parts[0].has_observations
        if any(part.has_observations != full for part in parts):
            raise ValueError("Cannot mix full observations and lengths-only tracks")
        lengths = np.concatenate([part.lengths for part in parts])
        if not full:
            return cls(lengths)
        return cls(lengths, np.concatenate([part.camera_ids for part in parts]),
                   np.concatenate([part.pixels for part in parts]))

    @classmethod
    def from_rows(cls, rows):
        """Compatibility at API boundaries; pipeline builders never create rows."""
        lengths = np.fromiter(map(len, rows), dtype=np.uint32, count=len(rows))
        count = int(lengths.sum(dtype=np.int64))
        camera_ids = np.fromiter((camera for row in rows for camera, _, _ in row),
                                 dtype=np.int64, count=count)
        pixels = np.fromiter((value for row in rows for _, x, y in row for value in (x, y)),
                             dtype=np.float64, count=count * 2).reshape(-1, 2)
        return cls(lengths, camera_ids, pixels)


def observation_tracks(tracks) -> ObservationTracks:
    return tracks if isinstance(tracks, ObservationTracks) else ObservationTracks.from_rows(tracks)


class ObservationTrackBuilder:
    """Append one camera's accepted observations in bulk per reference batch."""

    def __init__(self, point_count, capacity, retain_observations=True):
        self.lengths = np.zeros(point_count, dtype=np.uint32)
        self.camera_ids = np.empty((point_count, capacity), dtype=np.int64) if retain_observations else None
        self.pixels = np.empty((point_count, capacity, 2), dtype=np.float64) if retain_observations else None

    def append(self, positions, camera_id, pixels):
        # Triangulation supplies unique positions for each accepted camera batch.
        if self.camera_ids is not None:
            slots = self.lengths[positions]
            self.camera_ids[positions, slots] = camera_id
            self.pixels[positions, slots] = pixels
        self.lengths[positions] += 1

    def finish(self):
        lengths = self.lengths.copy()
        if self.camera_ids is None:
            return ObservationTracks(lengths)
        valid = np.arange(self.camera_ids.shape[1])[None, :] < lengths[:, None]
        return ObservationTracks(lengths, self.camera_ids[valid], self.pixels[valid])
