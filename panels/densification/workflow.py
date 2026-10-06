# SPDX-FileCopyrightText: 2025 Shady Gmira
# SPDX-License-Identifier: GPL-3.0-or-later
"""Start jobs and import preview and completed point clouds."""

import os
import shutil
from dataclasses import replace
import lichtfeld as lf
import numpy as np
from ...core.reconstruction.writers import read_points3D_bin_point_cloud
from .job import DensifyJob, DensifyResult


def _format_track_lengths(track_lengths: np.ndarray) -> str:
    if track_lengths.size == 0:
        return "empty"
    return (
        f"min={int(track_lengths.min())}, "
        f"mean={float(track_lengths.mean()):.2f}, "
        f"max={int(track_lengths.max())}"
    )


def _log_sparse_import_track_stats(
    total_track_lengths: np.ndarray,
    kept_track_lengths: np.ndarray,
    min_track_length: int,
) -> None:
    before = int(total_track_lengths.size)
    after = int(kept_track_lengths.size)
    kept_pct = 100.0 if before == 0 else (float(after) / float(before)) * 100.0
    if int(min_track_length) > 0:
        lf.log.info(
            f"COLMAP sparse import validation: min_track_length={int(min_track_length)} loaded "
            f"{after:,}/{before:,} points ({kept_pct:.1f}%). "
            f"Track lengths in written sparse model [{_format_track_lengths(kept_track_lengths)}]"
        )
    else:
        lf.log.info(
            "COLMAP sparse import validation: track filtering disabled "
            f"(min_track_length=0). {before:,} points, "
            f"track lengths [{_format_track_lengths(total_track_lengths)}]"
        )


class DensificationWorkflow:
    def _start_training_after_import(self):
        try:
            lf.start_training()
        except Exception as exc:
            lf.log.warn(f"Failed to start training after densification: {exc}")

    def _is_running(self) -> bool:
        return self.job is not None and self.job.is_running()

    def _on_do_start(self, handle, event, args):
        self._start()

    def _on_do_cancel(self, handle, event, args):
        if self.job:
            self.job.cancel()

    def _start(self):
        if not self._has_training_data():
            lf.log.warn("No training cameras found in scene")
            self.last_result = DensifyResult(
                success=False,
                error="No training cameras found. Please load a dataset first."
            )
            return

        camera_nodes = self._get_effective_camera_nodes()
        if self.config.roi_only_selected and len(camera_nodes) < 2:
            self.last_result = DensifyResult(
                success=False,
                error="ROI densification requires at least 2 selected cameras.",
            )
            return
        if len(camera_nodes) < 2:
            self.last_result = DensifyResult(
                success=False,
                error="Need at least 2 cameras for densification.",
            )
            return
        if not self._capture_base_point_cloud():
            return

        self.last_result = None
        self._pending_start_training = False

        self._active_run_roi_only_selected = bool(self.config.roi_only_selected)

        config = replace(
            self.config,
            output_path=self._get_temp_output_path(),
        )

        self.job = DensifyJob(
            config=config,
            camera_nodes=camera_nodes,
            on_complete=self._on_complete,
            on_error=self._on_error,
            on_sequential_viz=self._on_sequential_viz,
            debug_state=self.debug_state,
        )
        self.job.start()

    def _on_sequential_viz(self, ply_path: str):
        self._pending_import = ply_path

    def _import_output(self, path: str) -> bool:
        if os.path.isdir(path):
            return self._import_sparse_model(path)
        return self._import_ply(path)

    def _on_complete(self, result: DensifyResult):
        base_count = len(self._base_point_cloud_points) if self._base_point_cloud_points is not None else 0
        run_roi_only_selected = self._run_roi_only_selected()
        if run_roi_only_selected and base_count > 0 and result.success:
            result.num_points = base_count + result.num_points
        if run_roi_only_selected:
            lf.log.info(f"Densification complete: {result.num_points:,} points after ROI merge")
        else:
            lf.log.info(f"Densification complete: {result.num_points:,} points after overwrite")
        self.last_result = result
        if self._auto_import and result.output_path:
            self._pending_import = result.output_path
            self._pending_start_training = bool(self._start_training_when_complete and result.success)
        else:
            self._pending_start_training = False

    def _on_error(self, error: Exception):
        lf.log.error(f"Densification failed: {error}")
        self._pending_start_training = False
        if self._preview_override_active:
            self._restore_base_point_cloud()
        self._active_run_roi_only_selected = None
        self.last_result = DensifyResult(success=False, error=str(error))

    def _apply_dense_point_cloud(self, scene, target, point_cloud, points, colors) -> bool:
        if self._run_roi_only_selected():
            points, colors = self._build_roi_merge_arrays(points, colors)
        self._set_point_cloud_data(point_cloud, points, colors)
        lf.log.debug(f"Updated '{target.name}' with {int(points.shape[0]):,} points")
        scene.notify_changed()
        self._preview_override_active = True
        if not self._is_running() and not self._pending_import:
            self._active_run_roi_only_selected = None
        return True

    def _import_ply(self, ply_path: str) -> bool:
        """Import the latest dense PLY into the active point cloud."""
        if not ply_path or not os.path.exists(ply_path):
            lf.log.warn(f"PLY file not found: {ply_path}")
            return False

        try:
            lf.log.debug(f"Loading PLY: {ply_path}")

            scene = lf.get_scene()
            if scene is None:
                lf.log.error("No scene available")
                return False

            target = self._resolve_target_point_cloud_node(scene)
            if not target:
                lf.log.error("No point cloud node found to merge into")
                return False

            dense_points, dense_colors = lf.io.load_point_cloud(ply_path)

            point_cloud = target.point_cloud()
            if not point_cloud:
                lf.log.error(f"Node '{target.name}' has no point cloud data")
                return False

            return self._apply_dense_point_cloud(scene, target, point_cloud, dense_points, dense_colors)

        except Exception as e:
            lf.log.error(f"Failed to import PLY: {e}")
            return False
        finally:
            if not os.environ.get("LFS_KEEP_TEMP"):
                try:
                    if os.path.exists(ply_path):
                        os.remove(ply_path)
                except Exception:
                    lf.log.warn(f"Failed to delete temp file: {ply_path}")

    def _import_sparse_model(self, sparse_dir: str) -> bool:
        """Import final dense COLMAP sparse output into the active point cloud."""
        points3d_path = os.path.join(sparse_dir, "points3D.bin")
        if not os.path.exists(points3d_path):
            lf.log.warn(f"COLMAP points3D.bin not found: {points3d_path}")
            return False

        try:
            lf.log.debug(f"Loading dense sparse model: {sparse_dir}")

            scene = lf.get_scene()
            if scene is None:
                lf.log.error("No scene available")
                return False

            target = self._resolve_target_point_cloud_node(scene)
            if not target:
                lf.log.error("No point cloud node found to merge into")
                return False

            dense_points, dense_colors, all_track_lengths = read_points3D_bin_point_cloud(points3d_path)
            keep = all_track_lengths >= max(0, self.config.min_track_length)
            dense_points, dense_colors = dense_points[keep], dense_colors[keep]
            track_lengths = all_track_lengths[keep]
            _log_sparse_import_track_stats(
                all_track_lengths,
                track_lengths,
                self.config.min_track_length,
            )
            if dense_points.shape[0] == 0:
                lf.log.warn("No dense sparse points remain after track filtering")
                return False

            point_cloud = target.point_cloud()
            if not point_cloud:
                lf.log.error(f"Node '{target.name}' has no point cloud data")
                return False

            return self._apply_dense_point_cloud(scene, target, point_cloud, dense_points, dense_colors)

        except Exception as e:
            lf.log.error(f"Failed to import sparse model: {e}")
            return False
        finally:
            if not os.environ.get("LFS_KEEP_TEMP"):
                try:
                    cleanup_root = sparse_dir
                    parent = os.path.dirname(sparse_dir)
                    if os.path.basename(sparse_dir) == "0" and os.path.basename(parent) == "sparse":
                        cleanup_root = os.path.dirname(parent)
                    shutil.rmtree(cleanup_root, ignore_errors=True)
                except Exception:
                    lf.log.warn(f"Failed to delete temp sparse model: {sparse_dir}")
