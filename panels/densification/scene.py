# SPDX-FileCopyrightText: 2025 Shady Gmira
# SPDX-License-Identifier: GPL-3.0-or-later
"""Camera selection and preservation of the scene point cloud."""

import lichtfeld as lf
import numpy as np
from .job import DensifyResult


class DensificationScene:
    def _has_training_data(self) -> bool:
        try:
            scene = lf.get_scene()
            cameras = [n for n in scene.get_nodes() if n.has_camera]
            return cameras is not None and len(cameras) > 0
        except Exception:
            return False

    def _has_masks(self) -> bool:
        try:
            scene = lf.get_scene()
            for n in scene.get_nodes():
                if n.has_camera and getattr(n, "has_mask", False):
                    return True
            return False
        except Exception:
            return False

    def _get_camera_count(self) -> int:
        try:
            scene = lf.get_scene()
            cameras = [n for n in scene.get_nodes() if n.has_camera]
            return len(cameras)
        except Exception:
            return 0

    def _get_camera_nodes(self):
        try:
            scene = lf.get_scene()
            if scene is None:
                return []
            return [n for n in scene.get_nodes() if n.has_camera]
        except Exception:
            return []

    def _get_selected_camera_nodes(self):
        try:
            scene = lf.get_scene()
            if scene is None:
                return []
            selected_names = list(getattr(lf, "get_selected_node_names", lambda: [])() or [])
            selected_nodes = []
            seen = set()
            for name in selected_names:
                if not name or name in seen:
                    continue
                seen.add(name)
                try:
                    node = scene.get_node(name)
                except Exception:
                    node = None
                if node is not None and getattr(node, "has_camera", False):
                    selected_nodes.append(node)
            return selected_nodes
        except Exception:
            return []

    def _get_selected_camera_count(self) -> int:
        return len(self._get_selected_camera_nodes())

    def _get_effective_camera_nodes(self):
        if self.config.roi_only_selected:
            return self._get_selected_camera_nodes()
        return self._get_camera_nodes()

    def _get_effective_camera_count(self) -> int:
        return len(self._get_effective_camera_nodes())

    def _nns_per_ref_max(self) -> int:
        return max(1, min(10, self._get_effective_camera_count() - 1))

    def _min_track_length_max(self) -> int:
        return max(0, min(self._get_effective_camera_count(), int(self.config.nns_per_ref) + 1))

    def _camera_scope_text(self) -> str:
        total = self._get_camera_count()
        selected = self._get_selected_camera_count()
        if self.config.roi_only_selected:
            if selected >= 2:
                return f"ROI mode: using only the {selected} selected cameras."
            return "ROI mode: select at least 2 cameras in the scene graph."
        return f"Scene mode: using all {total} cameras for densification."

    def _run_roi_only_selected(self) -> bool:
        if self._active_run_roi_only_selected is not None:
            return self._active_run_roi_only_selected
        return bool(self.config.roi_only_selected)

    @staticmethod
    def _coerce_point_cloud_array(value, field_name: str, *, copy: bool = True) -> np.ndarray:
        native = isinstance(value, lf.Tensor)
        arr = np.asarray(value.numpy(copy=copy) if native else value)
        if arr.ndim != 2:
            raise RuntimeError(
                f"Point cloud field '{field_name}' must be 2D, got shape {arr.shape!r}"
            )
        # Native numpy(copy=True) already owns an independent snapshot.
        return np.array(arr, copy=True) if copy and not native else arr

    def _build_roi_merge_arrays(self, dense_points, dense_colors):
        if self._base_point_cloud_points is None or self._base_point_cloud_colors is None:
            raise RuntimeError("ROI merge snapshot is missing.")

        dense_points_np = self._coerce_point_cloud_array(dense_points, "dense_points", copy=False)
        dense_colors_np = self._coerce_point_cloud_array(dense_colors, "dense_colors", copy=False)
        base_colors = self._base_point_cloud_colors
        if dense_colors_np.dtype != base_colors.dtype:
            # Preserve normalized float colors when merging a uint8 dense cloud.
            if dense_colors_np.dtype == np.uint8:
                dense_colors_np = dense_colors_np.astype(np.float32) / 255.0
            if base_colors.dtype == np.uint8:
                base_colors = base_colors.astype(np.float32) / 255.0
        merged_points_np = np.concatenate((dense_points_np, self._base_point_cloud_points), axis=0)
        merged_colors_np = np.concatenate((dense_colors_np, base_colors), axis=0)
        return merged_points_np, merged_colors_np

    def _capture_base_point_cloud(self) -> bool:
        scene = lf.get_scene()
        if scene is None:
            self.last_result = DensifyResult(success=False, error="No scene available.")
            return False

        target = self._find_target_point_cloud_node(scene)
        if target is None:
            self.last_result = DensifyResult(
                success=False,
                error="No point cloud node found to merge into.",
            )
            return False

        point_cloud = target.point_cloud()
        if point_cloud is None:
            self.last_result = DensifyResult(
                success=False,
                error=f"Node '{target.name}' has no point cloud data.",
            )
            return False

        self._target_point_cloud_uuid = target.uuid
        self._base_point_cloud_points = self._coerce_point_cloud_array(
            point_cloud.means,
            "means",
        )
        self._base_point_cloud_colors = self._coerce_point_cloud_array(
            point_cloud.colors,
            "colors",
        )
        self._preview_override_active = False

        return True

    def _resolve_target_point_cloud_node(self, scene):
        if scene is None:
            return None
        if self._target_point_cloud_uuid is not None:
            target = scene.get_node_by_uuid(self._target_point_cloud_uuid)
            return target if target is not None and target.type == lf.scene.NodeType.POINTCLOUD else None
        return self._find_target_point_cloud_node(scene)

    @staticmethod
    def _find_target_point_cloud_node(scene):
        for node in scene.get_nodes():
            if node.type == lf.scene.NodeType.POINTCLOUD:
                return node
        return None

    def _restore_base_point_cloud(self):
        if self._base_point_cloud_points is None or self._base_point_cloud_colors is None:
            return
        try:
            scene = lf.get_scene()
            target = self._resolve_target_point_cloud_node(scene)
            if target is None:
                return
            point_cloud = target.point_cloud()
            if point_cloud is None:
                return
            self._queue_cloud_update(
                target, point_cloud, self._base_point_cloud_points, self._base_point_cloud_colors,
                kind="restore",
            )
        except Exception as exc:
            lf.log.warn(f"Failed to restore original point cloud: {exc}")
