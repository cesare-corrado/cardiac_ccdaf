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
  recolored live. The key itself is bound by the host, which routes it
  to :meth:`commit_pending` — clipping shares the key.
- Changing the active label clears any pending selection (prevents mixed
  batches being committed by accident).

The class is Qt-free: the GUI (label combobox, buttons) is assembled by
``MainApp``; this module just exposes a small API the GUI binds to.
"""

from __future__ import annotations

from collections import deque
from enum import Enum, auto
from typing import TYPE_CHECKING, Callable, Deque, Optional, Set
from scipy.spatial import cKDTree
import numpy as np
import pyvista as pv

from ccdaf.core.mesh_loader import BODY_LABEL, UNASSIGNED

if TYPE_CHECKING:
    from ccdaf.core.region_tagger import RegionTagger


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

        # Snake (geodesic tag) sub-mode. X drops anchor vertices; the open
        # geodesic through them is redrawn live; commit tags every triangle
        # touching the path with the active label. Kept separate from the
        # cell-selection picker above — the two are mutually exclusive modes.
        #
        # The snake grows *bidirectionally*, like the PV clip: each new anchor
        # extends whichever endpoint — head or tail — is geodesically closer,
        # so the path stays open (head → tail) and never doubles back.
        self._snake_active: bool = False
        self._snake_tagger = None
        self._snake_anchors: list[int] = []   # picked vertices (for the spheres)
        self._snake_path: list[int] = []      # geodesic vertex ids, head → tail
        self._snake_head: int = -1
        self._snake_tail: int = -1
        # Pre-extend snapshots so a single anchor can be undone — the last
        # anchor may have grown either endpoint, so we restore state wholesale.
        self._snake_history: list[tuple] = []
        self._snake_anchor_actor = None
        self._snake_line_actor = None

        # The X key is bound by the host, not here: clipping wants the same
        # key, and when both tools bound it themselves whoever came last
        # won — or wiped the other's binding outright. The host owns the
        # key and routes it to commit_pending().
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
            self.commit_pending()
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
        
        # 5. Write the filled labels back to the mesh
        self.mesh.cell_data["elemTag"] = filled_label
        
        # 6. Refresh the view
        if self.on_render:
            self.on_render()



    def smooth_label(self, tagger: 'RegionTagger', label: int,
                     dilate: bool = True, erode: bool = False) -> int:
        """Morphologically smooth one label's boundary, undoably.

        ``dilate`` grows the label into its jagged body fringe, ``erode``
        shaves its spikes; both together run dilate-then-erode (a closing)
        that de-jags without net growth. Body is not a smoothing target — it
        is the background, not a region. Snapshots for undo only when it
        actually changes something. Returns the number of cells changed.
        """
        if label not in ALLOWED_LABELS or label == BODY_LABEL:
            return 0
        if not (dilate or erode):
            return 0
        before = np.asarray(self.mesh.cell_data["elemTag"]).copy()
        out = before.copy()
        if dilate:
            out = tagger.dilate_label(out, label)
        if erode:
            out = tagger.erode_label(out, label)
        changed = int(np.count_nonzero(out != before))
        if changed == 0:
            return 0
        self._undo_stack.append(before)
        self.mesh.cell_data["elemTag"] = out
        if self.on_render is not None:
            self.on_render()
        if self.on_commit is not None:
            self.on_commit()
        return changed

    # ------------------------------------------------------------------
    # Snake (geodesic tag) sub-mode
    # ------------------------------------------------------------------
    @property
    def snake_active(self) -> bool:
        return self._snake_active

    @property
    def snake_point_count(self) -> int:
        return len(self._snake_anchors)

    def _snake_reset(self) -> None:
        self._snake_anchors = []
        self._snake_path = []
        self._snake_head = -1
        self._snake_tail = -1
        self._snake_history = []

    def start_snake(self, tagger: 'RegionTagger') -> None:
        """Enter snake mode: X drops geodesic anchors on the surface.

        Left-drag keeps rotating (point picking with ``left_clicking=False``);
        the host routes the X key to :meth:`snake_pick_at_cursor`. Mutually
        exclusive with cell-selection mode — the host turns that off first.
        """
        self._snake_tagger = tagger
        self._snake_reset()
        self._snake_active = True
        self.plotter.enable_trackball_style()
        self.plotter.enable_point_picking(
            callback=lambda *a, **k: None,   # X drives picks, not the click
            use_picker=True,
            show_message=False,
            show_point=False,
            left_clicking=False,
            pickable_window=False,
        )
        self._clear_snake_actors()

    def stop_snake(self) -> None:
        """Leave snake mode, drop anchors, and release the picker."""
        self._snake_active = False
        self._snake_reset()
        self._clear_snake_actors()
        try:
            self.plotter.disable_picking()
        except Exception:
            pass

    def clear_snake(self) -> None:
        """Discard the current anchors without leaving snake mode."""
        self._snake_reset()
        self._clear_snake_actors()
        self.plotter.render()

    def _snake_extend(self, tagger: 'RegionTagger', vid: int) -> int:
        """Grow the snake with a new anchor ``vid``, bidirectionally.

        The new anchor extends whichever endpoint — head or tail — reaches it
        by the shorter geodesic, exactly like the PV clip snake. Pure with
        respect to rendering (no plotter calls), so it is unit-testable.

        Returns the anchor count (>0) on success, ``0`` for a no-op (a repeat
        pick on an endpoint), or ``-1`` when ``vid`` is unreachable from both
        endpoints (a disconnected mesh component).
        """
        vid = int(vid)
        # Snapshot the pre-extend state so this anchor can be undone; recorded
        # only on the branches that actually mutate (the successful ones).
        snap = (list(self._snake_anchors), list(self._snake_path),
                self._snake_head, self._snake_tail)
        # First anchor: seed head == tail.
        if self._snake_tail < 0:
            self._snake_history.append(snap)
            self._snake_tail = self._snake_head = vid
            self._snake_path = [vid]
            self._snake_anchors = [vid]
            return 1
        # Ignore a repeat pick on an existing endpoint.
        if vid == self._snake_head or vid == self._snake_tail:
            return 0
        # Second anchor: the first real geodesic segment (tail → vid).
        if len(self._snake_path) == 1:
            g = tagger.geodesic_path(self._snake_tail, vid)
            if len(g) < 2:
                return -1
            self._snake_history.append(snap)
            self._snake_path = g
            self._snake_head = vid
            self._snake_anchors.append(vid)
            return len(self._snake_anchors)
        # Later anchors: extend the nearer endpoint.
        g1 = tagger.geodesic_path(self._snake_head, vid)   # head → P
        g2 = tagger.geodesic_path(vid, self._snake_tail)   # P → tail
        if not g1 and not g2:
            return -1
        if not g1:
            chose_head = False
        elif not g2:
            chose_head = True
        else:
            chose_head = len(g1) <= len(g2)
        self._snake_history.append(snap)
        if chose_head:
            # Append head → P (skip g1[0] = old head, already at path end).
            self._snake_path = self._snake_path + g1[1:]
            self._snake_head = vid
        else:
            # Prepend P → tail (skip g2[-1] = old tail, already at path start).
            self._snake_path = g2[:-1] + self._snake_path
            self._snake_tail = vid
        self._snake_anchors.append(vid)
        return len(self._snake_anchors)

    def undo_last_point(self) -> int:
        """Remove the most recently added snake anchor, restoring the previous
        geodesic. Returns the anchor count after undo (``0`` once emptied), or
        ``-1`` when there is nothing to undo."""
        if not self._snake_history:
            return -1
        anchors, path, head, tail = self._snake_history.pop()
        self._snake_anchors = anchors
        self._snake_path = path
        self._snake_head = head
        self._snake_tail = tail
        self._redraw_snake()
        return len(self._snake_anchors)

    def snake_pick_at_cursor(self) -> int:
        """Drop a snake anchor at the mouse position and grow the geodesic.

        Returns the anchor count (>0) when one was placed, ``0`` for a no-op
        (body active label — body builds no geodesic — or a repeat pick), or
        ``-1`` when the pick is unreachable from the current snake.
        """
        if not self._snake_active:
            return 0
        if self._active_label == BODY_LABEL or self._active_label not in ALLOWED_LABELS:
            return 0
        interactor = self.plotter.iren.interactor
        click_pos = interactor.GetEventPosition()
        picker = self.plotter.picker
        picker.Pick(click_pos[0], click_pos[1], 0, self.plotter.renderer)
        picked = picker.GetPickPosition()
        if picked == (0.0, 0.0, 0.0):
            return 0
        vid = int(self.mesh.find_closest_point(np.asarray(picked, dtype=float)))
        code = self._snake_extend(self._snake_tagger, vid)
        if code > 0:
            self._redraw_snake()
        return code

    def commit_snake(self, tagger: 'RegionTagger') -> int:
        """Tag every triangle touching the geodesic with the active label.

        The open head → tail geodesic is already built (grown anchor by
        anchor); commit extracts the triangles with at least one vertex on it
        and assigns the active label. Undoable. Returns the number of cells
        changed, or ``0`` when there is nothing to do (fewer than two anchors,
        body label, or no net change).
        """
        if self._active_label == BODY_LABEL or self._active_label not in ALLOWED_LABELS:
            return 0
        if len(self._snake_path) < 2:
            return 0
        cells = tagger.triangles_incident_to(self._snake_path)
        if cells.size == 0:
            return 0
        before = np.asarray(self.mesh.cell_data["elemTag"]).copy()
        tags = before.copy()
        tags[cells] = self._active_label
        changed = int(np.count_nonzero(tags != before))
        if changed == 0:
            self._snake_reset()
            self._clear_snake_actors()
            return 0
        self._undo_stack.append(before)
        self.mesh.cell_data["elemTag"] = tags   # force VTK update
        self._snake_reset()
        self._clear_snake_actors()
        if self.on_render is not None:
            self.on_render()
        if self.on_commit is not None:
            self.on_commit()
        return changed

    def _redraw_snake(self) -> None:
        if self.plotter is None:
            return
        self._clear_snake_actors()
        if not self._snake_anchors:
            self.plotter.render()
            return

        # Anchor spheres at the picked vertices.
        anchor_pts = np.asarray(self.mesh.points[self._snake_anchors], dtype=float)
        glyphs = pv.PolyData(anchor_pts).glyph(
            geom=pv.Sphere(radius=self._sphere_radius * 1.5),
            scale=False,
            orient=False,
        )
        self._snake_anchor_actor = self.plotter.add_mesh(
            glyphs,
            color="lime",
            name="snake_anchors",
            reset_camera=False,
            pickable=False,
        )

        # Geodesic polyline head → tail (green tube).
        if len(self._snake_path) >= 2:
            line_pts = np.asarray(self.mesh.points[self._snake_path], dtype=float)
            poly = pv.lines_from_points(line_pts)
            self._snake_line_actor = self.plotter.add_mesh(
                poly,
                color="lime",
                line_width=6,
                render_lines_as_tubes=True,
                name="snake_line",
                reset_camera=False,
                pickable=False,
            )
        self.plotter.render()

    def _clear_snake_actors(self) -> None:
        for attr in ("_snake_anchor_actor", "_snake_line_actor"):
            actor = getattr(self, attr, None)
            if actor is not None:
                try:
                    self.plotter.remove_actor(actor, reset_camera=False)
                except Exception:
                    pass
                setattr(self, attr, None)

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
    def commit_pending(self) -> None:
        """Commit the pending batch; a no-op with nothing pending."""
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
