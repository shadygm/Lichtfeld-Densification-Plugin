"""Reprojection metrics for final points and their recorded image observations."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..cameras.models import CameraRecord
from ..cameras.geometry import reprojection_errors, reprojection_errors_camera
from .tracks import ObservationTracks, observation_tracks


def _summarize(errors: np.ndarray, threshold: float) -> dict:
    finite = errors[np.isfinite(errors)]
    summary = dict(count=int(errors.size), invalid_count=int(errors.size - finite.size))
    if not finite.size:
        summary.update({name: None for name in ('mean', 'median', 'rmse', 'p90', 'p95', 'p99', 'max')})
    else:
        median, p90, p95, p99 = np.percentile(finite, [50, 90, 95, 99])
        summary.update(
            mean=float(finite.mean()), median=float(median),
            rmse=float(np.sqrt(np.mean(finite ** 2))),
            p90=float(p90), p95=float(p95), p99=float(p99), max=float(finite.max()),
        )
    summary['fraction_over_threshold_or_invalid'] = (
        float(np.count_nonzero(~np.isfinite(errors) | (errors > threshold)) / errors.size)
        if errors.size else None
    )
    return summary


def compute_reprojection_metrics(
    xyz: np.ndarray,
    tracks: ObservationTracks,
    cameras: Sequence[CameraRecord],
    threshold_px: float,
) -> dict:
    """Measure Euclidean pixel residuals using full camera models where available.

    Statistics exclude invalid projections; their counts and threshold failures
    are reported separately. Pixels use each camera's original resolution.
    """
    tracks = observation_tracks(tracks)
    if len(xyz) != len(tracks):
        raise ValueError('Each point must have exactly one observation track')
    tracks.require_observations()
    by_id = {camera.uid: camera for camera in cameras}
    point_indices = tracks.point_indices()

    errors_by_camera = []
    point_max = np.full(len(xyz), np.nan, dtype=np.float64)
    for camera_id, observations in tracks.camera_groups():
        if camera_id not in by_id:
            raise ValueError(f'Unknown camera ID in track: {camera_id}')
        indices = point_indices[observations]
        pixels = tracks.pixels[observations]
        points = xyz[indices]
        camera = by_id[camera_id]
        if camera.colmap_camera is not None:
            errors, valid = reprojection_errors_camera(
                camera.colmap_camera, camera.R, camera.t, points, pixels,
            )
        else:
            homogeneous = np.column_stack((points, np.ones(len(points))))
            errors = reprojection_errors(camera.P, homogeneous, pixels)
            valid = np.isfinite(points).all(axis=1) & ((homogeneous @ camera.P.T)[:, 2] > 0)
        errors = np.where(valid & np.isfinite(errors), errors, np.inf).astype(np.float64)
        errors_by_camera.append(errors)
        # fmax ignores the initial NaN, preserving NaN for points with no observations.
        np.fmax.at(point_max, indices, errors)
    errors = np.concatenate(errors_by_camera) if errors_by_camera else np.empty(0)
    lengths = tracks.lengths
    return dict(
        units='original_camera_pixels', threshold_px=float(threshold_px),
        observations=_summarize(errors, threshold_px),
        per_point_max=_summarize(point_max, threshold_px),
        tracks=dict(
            points=int(lengths.size), unobserved_points=int(np.count_nonzero(lengths == 0)),
            single_observation_points=int(np.count_nonzero(lengths == 1)),
            min=int(lengths.min()) if lengths.size else None,
            mean=float(lengths.mean()) if lengths.size else None,
            max=int(lengths.max()) if lengths.size else None,
        ),
    )
