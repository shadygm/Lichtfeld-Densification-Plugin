# SPDX-FileCopyrightText: 2025 Shady Gmira
# SPDX-License-Identifier: GPL-3.0-or-later
"""UI bindings, lifecycle and collapsed sections."""

import lichtfeld as lf
from .job import DensifyStage, DensifyResult


class DensificationBindings:
    def on_mount(self, doc):
        self._doc = doc
        self._scrub_fields.mount(doc)
        self._cleanup_cache()
        self._sync_section_states()

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("densification")
        if model is None:
            return

        # --- Scene state (read-only) ---
        model.bind_func("has_scene", self._has_training_data)
        model.bind_func("camera_count_text",
                        lambda: f"Scene loaded with {self._get_camera_count()} cameras")
        model.bind_func("selected_camera_text",
                        lambda: f"Selected cameras: {self._get_selected_camera_count()}")
        model.bind_func("camera_scope_text", self._camera_scope_text)
        model.bind_func("show_roi_selection_warning",
                        lambda: self.config.roi_only_selected and self._get_selected_camera_count() < 2)

        # --- Quality setting ---
        model.bind("roma_setting",
                    lambda: self.config.roma_setting,
                    self._set_roma_setting)
        model.bind_func("roma_description",
                        lambda: self._ROMA_DESCRIPTIONS.get(self.config.roma_setting, ""))
        model.bind_func("has_masks", self._has_masks)
        model.bind("use_masks",
                    lambda: self.config.use_masks,
                    self._set_use_masks)
        model.bind("roi_only_selected",
                    lambda: self.config.roi_only_selected,
                    self._set_roi_only_selected)

        # --- Slider-bound config values ---
        for prop, spec in self._scrub_specs.items():
            if prop == "voxel_size":
                continue
            model.bind(
                prop,
                lambda prop=prop, spec=spec: spec.fmt % self._get_scrub_field_value(prop),
                lambda value, prop=prop: self._set_scrub_field_value(prop, value),
            )
        model.bind("start_training_when_complete",
                    lambda: self._start_training_when_complete,
                    self._set_start_training_when_complete)
        model.bind("write_colmap", lambda: self._write_colmap, self._set_write_colmap)

        # --- Number-input config values ---
        model.bind("matches_per_ref_str",
                    lambda: str(self.config.matches_per_ref),
                    lambda v: self._set_int_config("matches_per_ref", v, 1000, 30000))
        model.bind("max_points_str",
                    lambda: str(self.config.max_points),
                    lambda v: self._set_int_config("max_points", v, 0, 10000000))

        # --- Distance filter ---
        model.bind("distance_filter_enabled",
                    lambda: self.config.voxel_size > 0.0,
                    self._set_distance_filter_enabled)
        model.bind("voxel_size",
                    lambda: f"{self._voxel_size_ui:.3f}",
                    lambda v: self._set_voxel_size(v))

        # --- Job state (read-only) ---
        model.bind_func("show_idle", lambda: not self._is_running())
        model.bind_func("show_running", self._is_running)
        model.bind_func("stage_text",
                        lambda: self.job.stage.value.capitalize() if self.job else "Idle")
        model.bind_func("progress_value",
                        lambda: f"{max(0.0, min(1.0, self.job.progress / 100.0)):.4f}"
                        if self.job else "0")
        model.bind_func("progress_pct",
                        lambda: f"{int(self.job.progress)}%"
                        if self.job else "0%")
        model.bind_func("progress_status",
                        lambda: self.job.status if self.job else "")

        # --- Result state ---
        model.bind_func("show_results",
                        lambda: self.last_result is not None and self.last_result.success)
        model.bind_func("result_points",
                        lambda: f"{self.last_result.num_points:,}"
                        if self.last_result and self.last_result.success else "0")
        model.bind_func("result_time",
                        lambda: f"{self.last_result.elapsed_time:.1f}s"
                        if self.last_result and self.last_result.success else "")
        model.bind_func("show_error",
                        lambda: self.last_result is not None and not self.last_result.success)
        model.bind_func("error_text",
                        lambda: self.last_result.error or "Unknown error"
                        if self.last_result and not self.last_result.success else "")

        # --- Events ---
        model.bind_event("do_start", self._on_do_start)
        model.bind_event("do_cancel", self._on_do_cancel)
        model.bind_event("toggle_section", self._on_toggle_section)
        model.bind_event("num_step", self._on_num_step)

        self._handle = model.get_handle()

    def on_update(self, doc):
        if self._pending_error is not None:
            self._pending_import = None  # Discard previews queued before the failure.
            self._pending_start_training = False
            if self._preview_override_active:
                self._restore_base_point_cloud()
            self._active_run_roi_only_selected = None
            self.last_result = DensifyResult(success=False, error=self._pending_error)
            self._pending_error = None

        # Handle pending import on main thread
        if self._pending_import:
            output = self._pending_import
            start_training = self._pending_start_training
            self._pending_import = None
            self._pending_start_training = False
            lf.log.info("Loading dense point cloud")
            imported = self._import_output(output)
            if imported and start_training:
                self._start_training_after_import()

        dirty = False

        if self._sync_scrub_specs():
            dirty = True
        if self._scrub_fields.sync_all():
            dirty = True

        # Track running state changes
        running = self._is_running()
        if running != self._last_running:
            self._last_running = running
            self._dirty("show_idle", "show_running")
            dirty = True

        if running and self.job:
            progress = self.job.progress
            status = self.job.status
            stage = self.job.stage.value
            if (progress != self._last_progress or
                    status != self._last_status or
                    stage != self._last_stage):
                self._last_progress = progress
                self._last_status = status
                self._last_stage = stage
                self._dirty("stage_text", "progress_value", "progress_pct", "progress_status")
                dirty = True

        # Track result changes
        has_result = self.last_result is not None and self.last_result.success
        has_error = self.last_result is not None and not self.last_result.success
        if has_result != self._last_has_result or has_error != self._last_has_error:
            self._last_has_result = has_result
            self._last_has_error = has_error
            self._dirty("show_results", "result_points", "result_time",
                        "show_error", "error_text",
                        "show_idle", "show_running")
            dirty = True

        # Track camera count changes
        cam_count = self._get_camera_count()
        if cam_count != self._last_camera_count:
            self._last_camera_count = cam_count
            self._dirty("has_scene", "camera_count_text")
            dirty = True

        selected_cam_count = self._get_selected_camera_count()
        if selected_cam_count != self._last_selected_camera_count:
            self._last_selected_camera_count = selected_cam_count
            self._dirty("selected_camera_text", "camera_scope_text", "show_roi_selection_warning")
            dirty = True

        effective_cam_count = self._get_effective_camera_count()
        if effective_cam_count != self._last_effective_camera_count:
            self._last_effective_camera_count = effective_cam_count
            self._dirty("camera_scope_text", "nns_per_ref")
            dirty = True

        # Track mask availability changes
        has_masks = self._has_masks()
        if has_masks != self._last_has_masks:
            self._last_has_masks = has_masks
            if not has_masks:
                self.config.use_masks = True
            self._dirty("has_masks", "use_masks")
            dirty = True

        if self.job and self.job.stage == DensifyStage.CANCELLED and self.last_result is None:
            self.last_result = self.job.result or DensifyResult(success=False, error="Cancelled")
            self._pending_start_training = False
            if self._preview_override_active:
                self._restore_base_point_cloud()
            self._active_run_roi_only_selected = None
            self._dirty("show_results", "show_error", "error_text", "show_idle", "show_running")
            dirty = True

        return dirty

    def on_scene_changed(self, doc):
        self._last_camera_count = -1
        self._last_selected_camera_count = -1
        self._last_effective_camera_count = -1
        self._last_has_masks = not self._has_masks()  # force redirty next update
        # Keep the ROI snapshot alive while a densification job is active or
        # while sequential previews/final import are still being applied.
        if not self._is_running() and not self._pending_import:
            self._target_point_cloud_name = None
            self._base_point_cloud_points = None
            self._base_point_cloud_colors = None
            self._active_run_roi_only_selected = None
            self._preview_override_active = False
        if self._handle:
            self._dirty(
                "has_scene",
                "camera_count_text",
                "selected_camera_text",
                "camera_scope_text",
                "show_roi_selection_warning",
                "has_masks",
                "use_masks",
                "nns_per_ref",
                "min_track_length",
            )

    def on_unmount(self, doc):
        doc.remove_data_model("densification")
        self._scrub_fields.unmount()
        self._handle = None
        self._doc = None

    def _dirty(self, *fields):
        if not self._handle:
            return
        if not fields:
            self._handle.dirty_all()
            return
        for f in fields:
            self._handle.dirty(f)

    def _get_section_elements(self, name):
        if not self._doc:
            return None, None, None
        header = self._doc.get_element_by_id(f"hdr-{name}")
        arrow = self._doc.get_element_by_id(f"arrow-{name}")
        content = self._doc.get_element_by_id(f"sec-{name}")
        return header, arrow, content

    def _sync_section_states(self):
        for name in ("matching", "cameras", "filtering", "output"):
            header, arrow, content = self._get_section_elements(name)
            if content:
                expanded = name not in self._collapsed
                if expanded:
                    content.set_class("collapsed", False)
                else:
                    content.set_class("collapsed", True)
                if arrow:
                    arrow.set_class("is-expanded", expanded)
                if header:
                    header.set_class("is-expanded", expanded)

    def _on_toggle_section(self, handle, event, args):
        del handle, event
        if not args:
            return
        name = str(args[0])
        expanding = name in self._collapsed
        if expanding:
            self._collapsed.discard(name)
        else:
            self._collapsed.add(name)

        header, arrow, content = self._get_section_elements(name)
        if content:
            content.set_class("collapsed", not expanding)
        if arrow:
            arrow.set_class("is-expanded", expanding)
        if header:
            header.set_class("is-expanded", expanding)
