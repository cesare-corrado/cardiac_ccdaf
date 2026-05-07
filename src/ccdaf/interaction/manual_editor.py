"""
ManualEditor
============

Interactive correction of the ``elemTag`` cell-data array.

Interaction model
-----------------
- User picks triangles on the surface (single-cell picking).
- Picked triangles are highlighted in yellow as a *pending* batch.
- Pressing **X** commits the batch: every pending triangle receives the
  currently active label, the highlight is cleared, and the mesh is
  recolored live.
- Changing the active label clears any pending selection (prevents mixed
  batches being committed by accident).

The class is Qt-free: the GUI (label combobox, buttons) is assembled by
``MainApp``; this module just exposes a small API the GUI binds to.
"""

from __future__ import annotations

from collections import deque
from enum import Enum, auto
from typing import Callable, Deque, Dict, Optional, Set
from scipy.spatial import cKDTree
import numpy as np
import pyvista as pv

from mesh_loader import BODY_LABEL, UNASSIGNED


ALLOWED_LABELS = (11, 13, 15, 17, 19, BODY_LABEL)


class EditState(Enum):
    IDLE       = auto()
    SELECTING  = auto()
    DONE       = auto()


class ManualEditor:
    """Triangle-level editor over an existing ``elemTag`` array."""

    def __init__(
        self,
        mesh: pv.PolyData,
        plotter,
        on_render: Optional[Callable[[], None]] = None,
        on_state: Optional[Callable[[EditState], None]] = None,
        on_commit: Optional[Callable[[], None]] = None,
    ) -> None:
        self.mesh = mesh
        self.plotter = plotter
        self.on_render = on_render
        self.on_state = on_state
        self.on_commit = on_commit

        self._state: EditState = EditState.IDLE
        self._active_label: int = 11
        self._pending: Set[int] = set()
        self._highlight_actor = None
        self._sphere_actor = None
        self._undo_stack: Deque[np.ndarray] = deque(maxlen=3)

        # Bind the commit key once; the callback is a no-op while idle.
        self.plotter.add_key_event("x", self._commit)
        self.plotter.add_key_event("X", self._commit)
        self._prepare_search_index(mesh)
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def state(self) -> EditState:
        return self._state

    @property
    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    def undo(self) -> bool:
        """Restore the elemTag array to the state before the last commit."""
        if not self._undo_stack:
            return False
        tags = self._undo_stack.pop()
        self.mesh.cell_data["elemTag"] = tags
        if self.on_render is not None:
            self.on_render()
        return True

    def reset_undo(self) -> None:
        """Clear the undo history (call after auto-tagging overwrites the mesh)."""
        self._undo_stack.clear()

    def set_active_label(self, label: int) -> None:
        if label not in ALLOWED_LABELS:
            raise ValueError(f"label {label} is not in {ALLOWED_LABELS}")
        if label != self._active_label:
            self._active_label = label
            self._clear_pending()  # don't let a change of label mix batches

    def activate(self) -> None:
        """Enter SELECTING state and enable cell picking."""
        if self._state is EditState.SELECTING:
            return
        self.plotter.enable_trackball_style()
        self._state = EditState.SELECTING
        self._enable_cell_picking()
        self._emit_state()

    def deactivate(self) -> None:
        """Leave SELECTING state but preserve pending selection (unusual)."""
        if self._state is EditState.IDLE:
            return
        self._state = EditState.IDLE
        try:
            self.plotter.disable_picking()
        except Exception:
            pass
        self._emit_state()

    def accept(self) -> None:
        """Commit any pending batch and exit the editor entirely."""
        if self._pending:
            self._commit()
        # assign body_label to unassigned
        tags = np.asarray(self.mesh.cell_data["elemTag"]).copy()
        idx = (tags == UNASSIGNED)
        tags[idx] = BODY_LABEL
        self.mesh.cell_data["elemTag"] = tags   # force VTK update

        self.deactivate()
        self._state = EditState.DONE
        self._emit_state()


    # Inside manual_editor.py
    def fill_holes(self, tagger: 'RegionTagger') -> None:
        """Fill unassigned holes while preserving separation between existing regions."""
        if self.mesh is None:
            return

        # 1. Get current tags from mesh
        tri_label = np.asarray(self.mesh.cell_data["elemTag"]).copy()
        # 2. Identify shared boundaries to prevent regions from merging
        pv_labels = {11, 13, 15, 17, 19} # From LABELS in region_tagger
        multi_tag_vertices = tagger._find_shared_pv_vertices(tri_label, pv_labels)
        
        # 3. Mask triangles on the boundary as UNASSIGNED to create a buffer
        if multi_tag_vertices.size > 0:
            mask_to_unassign = np.any(np.isin(tagger._triangles, multi_tag_vertices), axis=1)
            tri_label[mask_to_unassign] = UNASSIGNED
        
        # 4. Use the tagger's hole-filling logic
        filled_label = tagger._fill_holes(tri_label)
        
        # 5. Write back to mesh, respecting the BODY_LABEL for background
        assigned = tri_label != UNASSIGNED
        self.mesh.cell_data["elemTag"] = filled_label
        
        # 6. Refresh the view
        if self.on_render:
            self.on_render()



    # ------------------------------------------------------------------
    # Picking
    # ------------------------------------------------------------------
    def _prepare_search_index(self, mesh: pv.PolyData) -> None:
        self._centroids = self.mesh.cell_centers().points
        self._search_tree = cKDTree(self._centroids)
        # Mean triangle inradius (r = area / semi-perimeter) used to size
        # the centroid spheres in world units.
        pts = np.asarray(mesh.points, dtype=float)
        tris = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]
        A, B, C = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
        ab = np.linalg.norm(B - A, axis=1)
        bc = np.linalg.norm(C - B, axis=1)
        ca = np.linalg.norm(A - C, axis=1)
        area = 0.5 * np.linalg.norm(np.cross(B - A, C - A), axis=1)
        s = 0.5 * (ab + bc + ca)
        self._sphere_radius = float(np.mean(np.where(s > 0, area / s, 0.0)))



    def _enable_cell_picking(self) -> None:
        self.plotter.enable_point_picking(
            callback=self._on_cell_picked,
            use_picker=True,
            show_message=False,
            show_point=False,
            left_clicking=True,
            pickable_window=False,
        )

    def _on_cell_picked(self, picked, *args, **kwargs) -> None:
        if self._state is not EditState.SELECTING or picked is None:
            return
        cell_id = None
        # Primary path: re-pick at the current screen position using vtkCellPicker.
        # This avoids any dependency on PyVista's internal picker object and the
        # GetPicker/SetPicker API that is not reliably exposed by PyVista's wrapper.
        try:
            import vtk
            event_pos = self.plotter.iren.GetEventPosition()
            cell_picker = vtk.vtkCellPicker()
            cell_picker.SetTolerance(0.001)
            cell_picker.Pick(event_pos[0], event_pos[1], 0, self.plotter.renderer)
            cid = cell_picker.GetCellId()
            if cid >= 0:
                cell_id = int(cid)
        except Exception:
            pass
        # Fallback: ray-trace from camera through the picked world coord.
        if cell_id is None:
            camera_pos = self.plotter.camera_position[0]
            direction = np.array(picked) - np.array(camera_pos)
            far_point = np.array(camera_pos) + direction * 1.5
            _, ind = self.mesh.ray_trace(camera_pos, far_point, first_point=True)
            if len(ind) > 0:
                cell_id = int(ind[0])
            else:
                _, cell_id = self._search_tree.query(picked)
                cell_id = int(cell_id)
        self._pending.add(cell_id)
        self._refresh_highlight()

    # ------------------------------------------------------------------
    # Commit / highlight
    # ------------------------------------------------------------------
    def _commit(self) -> None:
        if not self._pending:
            return
        # Snapshot current state before modifying (for undo).
        self._undo_stack.append(
            np.asarray(self.mesh.cell_data["elemTag"]).copy()
        )
        idx = np.fromiter(self._pending, dtype=np.int64)
        tags = np.asarray(self.mesh.cell_data["elemTag"])
        tags[idx] = self._active_label
        self.mesh.cell_data["elemTag"] = tags   # force VTK update

        self._clear_pending()
        if self.on_render is not None:
            self.on_render()
        if self.on_commit is not None:
            self.on_commit()

    def _clear_pending(self) -> None:
        self._pending.clear()
        for attr in ("_highlight_actor", "_sphere_actor"):
            actor = getattr(self, attr, None)
            if actor is not None:
                try:
                    self.plotter.remove_actor(actor, reset_camera=False)
                except Exception:
                    pass
                setattr(self, attr, None)

    def _refresh_highlight(self) -> None:
        for attr in ("_highlight_actor", "_sphere_actor"):
            actor = getattr(self, attr, None)
            if actor is not None:
                try:
                    self.plotter.remove_actor(actor, reset_camera=False)
                except Exception:
                    pass
                setattr(self, attr, None)
        if not self._pending:
            self.plotter.render()
            return

        idx = np.fromiter(self._pending, dtype=np.int64)

        # Yellow triangle overlay.
        sub = self.mesh.extract_cells(idx)
        self._highlight_actor = self.plotter.add_mesh(
            sub,
            color="yellow",
            opacity=1.0,
            ambient=0.5,
            line_width=5,
            edge_color="yellow",
            reset_camera=False,
            pickable=False,
            name="edit_highlight",
            render_lines_as_tubes=True,
            render_points_as_spheres=True,
        )
        self._highlight_actor.GetMapper().SetRelativeCoincidentTopologyPolygonOffsetParameters(-1, -1)

        # Yellow spheres at triangle centroids sized in world units
        # (radius = mean triangle inradius).
        centers = self._centroids[idx]
        glyphs = pv.PolyData(centers).glyph(
            geom=pv.Sphere(radius=self._sphere_radius),
            scale=False,
            orient=False,
        )
        self._sphere_actor = self.plotter.add_mesh(
            glyphs,
            color="yellow",
            reset_camera=False,
            pickable=False,
            name="edit_highlight_spheres",
        )
        self.plotter.render()
    # ------------------------------------------------------------------
    def _emit_state(self) -> None:
        if self.on_state is not None:
            self.on_state(self._state)


__all__ = ["ManualEditor", "EditState", "ALLOWED_LABELS"]
