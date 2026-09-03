"""
ClippingTool
============

Two clipping operations, driven interactively:

1.  **PV contour clip** — tag-constrained bidirectional Dijkstra snake.
    The user selects a *target tag* (the surface region the snake may
    travel on) and places picks on that region. Between picks the tool
    grows a shortest geodesic path along a subgraph restricted to
    vertices carrying that tag, excluding tag-boundary vertices (those
    incident to ≥ 2 distinct triangle tags). The snake is
    bidirectional: each new pick extends from whichever endpoint —
    head or tail — is closer in geodesic distance. On closure the head
    and tail are joined by one more constrained geodesic segment.
    ``vtkSelectPolyData`` (Dijkstra edge search + seed-anchored
    ``SetClosestPoint``) identifies the PV region, which is then
    discarded. The resulting hole is left open.

2.  **Geometric clip** — either
      * **Sphere**: interactive ``vtkSphereWidget``; triangles whose
        centroid lies inside the sphere are removed; or
      * **Plane**:  interactive ``vtkPlaneWidget``; triangles on the
        seed's side of the plane are removed.
    The resulting hole is likewise left open. Both are driven from a
    *seed* — the anatomical point the widget is first placed on — and
    so serve the mitral valve and the pulmonary veins alike: the seed
    is whichever region the user selected, not the MV by construction.

    Each geometry is remembered per seed key once its widget goes away
    (see ``pose_for`` / ``remember_pose``), so restarting a clip resumes
    where the last one left off instead of jumping back to the seed
    default. ``reset_pose`` is the deliberate way back to that default.

All clips preserve ``elemTag`` on surviving triangles (no cells are
split). Clipped meshes are deliberately non-manifold at the ostium /
annulus: downstream tooling expects open boundaries there.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, List, Optional, Sequence

import numpy as np
import pyvista as pv
import vtk

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as _sp_dijkstra
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
class ClipMode(Enum):
    NONE        = auto()
    PV_CONTOUR  = auto()
    SPHERE      = auto()
    PLANE       = auto()


@dataclass
class ClipResult:
    mesh: pv.PolyData
    n_removed: int


# ---------------------------------------------------------------------------
SNAKE_SPHERE_RADIUS: float = 0.5
ELEM_TAG_ARRAY: str = "elemTag"


# ---------------------------------------------------------------------------
# Tag preprocessing + subgraph
# ---------------------------------------------------------------------------
def _compute_point_tags_and_boundary(
    mesh: pv.PolyData,
) -> tuple[np.ndarray, np.ndarray]:
    """Transfer cell tags to points and flag tag-boundary vertices.

    Uses ``cell_data_to_point_data`` per the spec for the tag transfer
    (interior vertices inherit the unique incident-cell tag; boundary
    vertices receive the average, which is immaterial because they are
    flagged invalid separately).

    A vertex is "boundary" iff it is incident to triangles carrying
    ≥ 2 distinct ``elemTag`` values.

    Returns
    -------
    point_tag   : int array, shape (n_points,)  — -1 on boundary vertices
    boundary   : bool array, shape (n_points,)
    """
    if ELEM_TAG_ARRAY not in mesh.cell_data:
        raise ValueError(f"mesh has no cell_data '{ELEM_TAG_ARRAY}'")

    # Spec-required transfer (averaging); we overwrite boundary entries below.
    try:
        transferred = mesh.cell_data_to_point_data()
        averaged = np.asarray(transferred.point_data[ELEM_TAG_ARRAY])
    except Exception:
        averaged = np.zeros(mesh.n_points, dtype=np.int64)

    faces = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]
    cell_tags = np.asarray(mesh.cell_data[ELEM_TAG_ARRAY], dtype=np.int64)
    n_points = int(mesh.n_points)

    v_ids = faces.reshape(-1)                       # (3*n_cells,)
    c_tags = np.repeat(cell_tags, 3)                # (3*n_cells,)

    # Unique (vertex, tag) pairs → per-vertex distinct-tag count.
    pairs = np.unique(np.stack([v_ids, c_tags], axis=1), axis=0)
    verts_u, counts = np.unique(pairs[:, 0], return_counts=True)
    boundary = np.zeros(n_points, dtype=bool)
    boundary[verts_u[counts > 1]] = True

    # Definitive per-vertex tag for non-boundary points: the tag of any
    # incident cell (they all agree by construction).
    point_tag = np.rint(averaged).astype(np.int64)
    # Override with exact tags for non-boundary vertices via first-incidence.
    first_seen = np.full(n_points, -1, dtype=np.int64)
    # np.unique with return_index picks the lowest index; that's fine since
    # all incident cells share the tag for non-boundary vertices.
    u_first, idx_first = np.unique(v_ids, return_index=True)
    first_seen[u_first] = c_tags[idx_first]
    non_b = ~boundary
    point_tag[non_b] = first_seen[non_b]
    point_tag[boundary] = -1
    return point_tag, boundary


def _build_subgraph(
    mesh: pv.PolyData,
    allowed_mask: np.ndarray,
) -> csr_matrix:
    """Build a symmetric edge-weighted graph over ``mesh`` restricted to
    ``allowed_mask`` vertices. Edge weight = Euclidean length."""
    faces = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]
    pts = np.asarray(mesh.points, dtype=float)

    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    keep = allowed_mask[edges[:, 0]] & allowed_mask[edges[:, 1]]
    edges = edges[keep]
    if edges.size == 0:
        n = int(mesh.n_points)
        return csr_matrix((n, n), dtype=float)

    # Each interior triangle edge is shared by two faces, so ``edges`` lists
    # each undirected edge twice. Canonicalise (min, max) and deduplicate
    # before building the symmetric CSR — otherwise csr_matrix would sum
    # duplicate entries and inflate edge weights.
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)

    lengths = np.linalg.norm(pts[edges[:, 0]] - pts[edges[:, 1]], axis=1)
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.concatenate([lengths, lengths])
    n = int(mesh.n_points)
    return csr_matrix((data, (rows, cols)), shape=(n, n))


def _subgraph_path(graph: csr_matrix, start: int, end: int) -> List[int]:
    """Shortest path (list of vertex ids, start → end) on ``graph``.

    Returns ``[start]`` when start == end, or ``[]`` if disconnected."""
    if start == end:
        return [int(start)]
    dist, pred = _sp_dijkstra(
        graph, indices=int(start), return_predecessors=True,
    )
    if not np.isfinite(dist[end]):
        return []
    path = [int(end)]
    v = int(end)
    while v != int(start):
        v = int(pred[v])
        if v < 0:
            return []
        path.append(v)
    path.reverse()
    return path


# ---------------------------------------------------------------------------
class ClippingTool:
    """Interactive contour + geometric clipping with multi-undo and checks."""

    _TOL_ABS_FLOOR: float = 1e-8
    _CLOSURE_TOL_REL: float = 1e-6

    def __init__(
        self,
        mesh_getter: Callable[[], pv.PolyData],
        mesh_setter: Callable[[pv.PolyData], None],
        plotter,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.get_mesh = mesh_getter
        self.set_mesh = mesh_setter
        self.plotter = plotter
        self.on_status = on_status
        # Fired whenever the live sphere/plane geometry changes, so a panel
        # can show the numbers. Payload: (ClipMode, dict of named floats).
        self.on_pose_changed: Optional[Callable[[ClipMode, dict], None]] = None

        self._mode: ClipMode = ClipMode.NONE

        # Snake state.
        self._target_tag: int = -1
        self._head: int = -1
        self._tail: int = -1
        self._path: List[int] = []
        self._snake_actor = None

        # Per-pick history for "undo last point" while building the snake.
        # Each entry is a pre-mutation snapshot (path, head, tail, pick_count);
        # popping it reverts exactly one placed point. Distinct from the
        # mesh-level ``_history`` used by restore()/undo().
        self._pick_history: List[tuple] = []
        self._pick_count: int = 0

        # Tag-constrained subgraph (cached per PV session).
        self._point_tag: Optional[np.ndarray] = None
        self._boundary: Optional[np.ndarray] = None
        self._allowed: Optional[np.ndarray] = None
        self._subgraph: Optional[csr_matrix] = None
        self._faces: Optional[np.ndarray] = None
        self._cell_tags: Optional[np.ndarray] = None

        # Widgets
        self._sphere_widget: Optional[vtk.vtkSphereWidget] = None
        self._plane_widget: Optional[vtk.vtkImplicitPlaneWidget] = None

        # Clip preview actor (red overlay of to-be-clipped triangles).
        self._preview_actor = None
        # Fixed reference point deciding which half of the plane goes; set
        # by start_plane and held apart from the plane's own origin.
        self._side_seed: Optional[np.ndarray] = None

        # Multi-undo history stack (deep copies).
        self._history: List[pv.PolyData] = []

        # Last geometry per (seed key, mode), captured when a widget goes
        # away. Restarting a clip on the same seed resumes from it rather
        # than from the seed default — reverting a clip should not cost the
        # user the sphere they spent time placing. Scoped to this tool, so a
        # newly loaded mesh (which rebuilds the tool) starts clean.
        self._pose_memory: dict = {}
        self._seed_key: str = ""

    # ==================================================================
    # Common helpers
    # ==================================================================
    @property
    def mode(self) -> ClipMode:
        return self._mode

    @property
    def can_undo(self) -> bool:
        """True when a previous mesh state is available to restore.

        Every clip pushes a snapshot before modifying the mesh,
        so this gates the host's revert/undo button for either clip type."""
        return bool(self._history)

    def _status(self, msg: str) -> None:
        if self.on_status is not None:
            self.on_status(msg)

    @staticmethod
    def _mesh_diag(mesh: pv.PolyData) -> float:
        b = mesh.bounds
        return float(np.linalg.norm([b[1] - b[0], b[3] - b[2], b[5] - b[4]]))

    def _tolerance(self, mesh: pv.PolyData) -> float:
        return max(self._CLOSURE_TOL_REL * self._mesh_diag(mesh),
                   self._TOL_ABS_FLOOR)

    # ==================================================================
    # Undo / history
    # ==================================================================
    def _snapshot(self) -> None:
        m = self.get_mesh()
        if m is None:
            return
        self._history.append(m.copy(deep=True))

    def drop_snapshot(self) -> None:
        """Discard the newest snapshot without touching the mesh.

        Restarting a widget-based clip pushes a snapshot of a mesh that was
        never modified. Resetting the widget is one such restart, and without
        this the undo stack would grow a rung per reset that reverts to the
        state already on screen."""
        if self._history:
            self._history.pop()

    def restore(self) -> None:
        if not self._history:
            return
        prev = self._history.pop()
        self.set_mesh(prev)
        self._status("Clip reverted.")

    def undo(self) -> bool:
        if not self._history:
            self._status("Undo: nothing to undo.")
            return False
        prev = self._history.pop()
        self.set_mesh(prev)
        self._status("Undo: previous mesh state restored.")
        return True

    def cancel(self) -> None:
        # Every exit from a geometric clip funnels through here, so this is
        # the one place the pose has to be saved for a later resume.
        self._capture_pose()
        self._mode = ClipMode.NONE
        self._clear_contour()
        self._clear_subgraph()
        self._clear_preview()
        self._remove_sphere_widget()
        self._remove_plane_widget()
        # Switching a widget off does not repaint on its own, so without this
        # the sphere or plane stays drawn until something else triggers a
        # render — looking to the user like two live widgets at once.
        try:
            self.plotter.render()
        except Exception:
            pass
        try:
            self.plotter.disable_picking()
        except Exception:
            pass

    def refresh(self) -> None:
        # Subgraph is session-scoped; rebuilt on start_pv_contour.
        pass

    # ==================================================================
    # Mesh integrity
    # ==================================================================
    @staticmethod
    def _validate_mesh(mesh: Optional[pv.PolyData]) -> bool:
        if mesh is None:
            return False
        try:
            if not isinstance(mesh, pv.PolyData):
                return False
            if mesh.n_points <= 0 or mesh.n_cells <= 0:
                return False
            faces = np.asarray(mesh.faces)
            if faces.size == 0 or faces.size % 4 != 0:
                return False
            if np.any(faces[::4] != 3):
                return False
            tris = faces.reshape(-1, 4)[:, 1:]
            if np.any(tris < 0) or np.any(tris >= mesh.n_points):
                return False
        except Exception:
            return False
        return True

    # ==================================================================
    # Subgraph management
    # ==================================================================
    def _build_pv_subgraph(self, mesh: pv.PolyData, target_tag: int) -> bool:
        """Preprocess: point tags, boundary mask, allowed mask, subgraph."""
        try:
            point_tag, boundary = _compute_point_tags_and_boundary(mesh)
        except Exception as exc:
            self._status(f"PV clip: tag preprocessing failed ({exc}).")
            return False
        allowed = (point_tag == int(target_tag)) & (~boundary)
        if not np.any(allowed):
            self._status(
                f"PV clip: no vertices carry tag {target_tag} — aborting."
            )
            return False
        self._point_tag = point_tag
        self._boundary = boundary
        self._allowed = allowed
        self._subgraph = _build_subgraph(mesh, allowed)
        # Triangles and their tags, cached from the same mesh as the masks
        # above: a pick is judged by the triangle it lands on, not by the
        # vertex nearest to it (see _pick_vertex_on_target).
        self._faces = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]
        self._cell_tags = np.asarray(mesh.cell_data[ELEM_TAG_ARRAY], dtype=np.int64)
        return True

    def _clear_subgraph(self) -> None:
        self._point_tag = None
        self._boundary = None
        self._allowed = None
        self._subgraph = None
        self._faces = None
        self._cell_tags = None
        self._target_tag = -1

    # ==================================================================
    # PV contour clipping — tag-constrained snake
    # ==================================================================
    def start_pv_contour(self, pv_label: int) -> None:
        """Begin PV clipping on the surface region carrying ``pv_label``.

        ``pv_label`` is the *target tag*: the snake may travel only on
        vertices carrying that tag and not on tag-boundary vertices.
        """
        self.cancel()
        self._mode = ClipMode.PV_CONTOUR
        self._target_tag = int(pv_label)
        self._head = -1
        self._tail = -1
        self._path = []
        self._pick_history = []
        self._pick_count = 0
        self._snapshot()

        mesh = self.get_mesh()
        if mesh is None:
            self._status("PV clip: no mesh loaded.")
            self._mode = ClipMode.NONE
            return
        if not self._build_pv_subgraph(mesh, self._target_tag):
            self._mode = ClipMode.NONE
            return

        # Release first — one picker per render window, and the seed selector
        # and manual editor grab the same one; see ``CCDAF._release_picker``.
        try:
            self.plotter.disable_picking()
        except Exception:
            pass
        self.plotter.enable_point_picking(
            callback=self._on_contour_pick,
            picker="hardware",               # z-buffer pick of the VISIBLE surface
            use_picker=True,
            show_message=False,
            show_point=False,
            pickable_window=False,
            left_clicking=False,
        )
        self._status(
            f"PV clip: pick points on tag {self._target_tag} — the snake "
            "will follow a geodesic restricted to that region."
        )

    def pick_at_cursor(self) -> None:
        """Place a snake point at the mouse position — the host routes the
        X key here. The tool no longer binds keys itself: manual correction
        wants the same key, and whoever bound last used to win."""
        if self._mode is not ClipMode.PV_CONTOUR:
            return
        interactor = self.plotter.iren.interactor
        click_pos = interactor.GetEventPosition()
        picker = self.plotter.picker
        picker.Pick(click_pos[0], click_pos[1], 0, self.plotter.renderer)
        # A miss must not place a point. The hardware picker still reports a
        # world position when the ray hits no geometry — a point on the focal
        # plane, whose nearest mesh vertex is an arbitrary point of the
        # surface, accepted or rejected by pure chance. It holds no dataset in
        # that case, which is exactly what pyvista's own pick observer tests
        # before it forwards an event; the (0,0,0) test below never caught it.
        if hasattr(picker, "GetDataSet") and picker.GetDataSet() is None:
            self._status("PV pick ignored: nothing under the cursor.")
            return
        picked_point = picker.GetPickPosition()
        if picked_point != (0.0, 0.0, 0.0):
            self._on_contour_pick(picked_point)

    # ------------------------------------------------------------------
    def _on_contour_pick(self, picked_point, *args) -> None:
        if self._mode is not ClipMode.PV_CONTOUR or picked_point is None:
            return
        mesh = self.get_mesh()
        if mesh is None or mesh.n_points == 0:
            return
        if self._allowed is None or self._subgraph is None:
            return

        p = self._pick_vertex_on_target(mesh, picked_point)
        if p < 0:
            return

        # A pick that lands on an existing endpoint is a no-op. This also
        # absorbs pyvista firing the callback twice for a single X press:
        # enable_point_picking installs an EndPickEvent observer, and
        # pick_at_cursor both calls picker.Pick() (which fires it) and invokes
        # the callback directly. Without this guard the phantom second call
        # would record a duplicate undo step per placed point. Returns silently
        # so the real pick's status message is preserved.
        if p == self._head or p == self._tail:
            return

        if self._tail < 0:
            self._pick_history.append(self._capture_pick_state())
            self._tail = p
            self._head = p
            self._path = [p]
            self._pick_count += 1
            self._redraw_snake()
            self._status(f"PV clip: snake started ({self._pick_count} point).")
            return

        if len(self._path) == 1:
            path = _subgraph_path(self._subgraph, self._tail, p)
            if len(path) < 2:
                self._status(
                    "PV clip: no tag-restricted geodesic to that point."
                )
                return
            self._pick_history.append(self._capture_pick_state())
            self._path = path
            self._head = p
            self._pick_count += 1
            self._redraw_snake()
            self._status(
                f"PV clip: {self._pick_count} points, "
                f"{len(self._path)} vertices."
            )
            return

        g1 = _subgraph_path(self._subgraph, self._head, p)   # head → P
        g2 = _subgraph_path(self._subgraph, p, self._tail)   # P → tail

        if not g1 and not g2:
            self._status(
                "PV clip: no constrained geodesic from either end — "
                "pick closer to the current snake."
            )
            return
        if not g1:
            chose_head = False
        elif not g2:
            chose_head = True
        else:
            chose_head = len(g1) <= len(g2)

        self._pick_history.append(self._capture_pick_state())
        if chose_head:
            # Append head → P (skip g1[0] = old head, already at path end).
            self._path = self._path + g1[1:]
            self._head = p
        else:
            # Prepend P → tail (skip g2[-1] = old tail, already at path start).
            self._path = g2[:-1] + self._path
            self._tail = p

        self._pick_count += 1
        self._redraw_snake()
        self._status(
            f"PV clip: {self._pick_count} points, {len(self._path)} vertices."
        )

    # ------------------------------------------------------------------
    def _pick_vertex_on_target(self, mesh: pv.PolyData, picked_point) -> int:
        """Resolve a surface hit to a snake vertex on the target region.

        The **triangle** under the hit decides whether the pick is on the
        selected vein. Judging by the nearest *vertex* instead lets a pick on
        the wrong vein through wherever two tagged regions come close — the
        right-vein carina above all, where an RIPV triangle can have an RSPV
        vertex as its nearest: the vertex carries the target tag, so the pick
        was accepted even though the surface clicked belongs to the other vein.

        Returns the vertex id to grow the snake from — the nearest vertex *of
        that triangle* that the subgraph allows — or ``-1`` when the pick is
        rejected, having posted the reason.
        """
        if self._faces is None or self._cell_tags is None:
            return -1
        pt = np.asarray(picked_point, dtype=float)
        cell_id = int(mesh.find_closest_cell(pt))
        if cell_id < 0 or cell_id >= self._cell_tags.size:
            self._status("PV pick ignored: not on the surface.")
            return -1
        cell_tag = int(self._cell_tags[cell_id])
        if cell_tag != self._target_tag:
            self._status(
                f"PV pick ignored: that triangle is tagged {cell_tag}, "
                f"not the target tag {self._target_tag}."
            )
            return -1
        tri = self._faces[cell_id]
        allowed = tri[self._allowed[tri]]
        if allowed.size == 0:
            # Every vertex of the triangle is a tag boundary: the pick sits on
            # the rim of the region, where the snake has nothing to travel on.
            self._status(
                "PV pick ignored: on the region border — pick further inside "
                f"tag {self._target_tag}."
            )
            return -1
        d = np.linalg.norm(np.asarray(mesh.points)[allowed] - pt, axis=1)
        return int(allowed[int(np.argmin(d))])

    # ------------------------------------------------------------------
    def _capture_pick_state(self) -> tuple:
        """Snapshot the snake state prior to a pick, for undo_last_point."""
        return (list(self._path), self._head, self._tail, self._pick_count)

    def undo_last_point(self) -> int:
        """Remove the most recently placed PV snake point.

        Returns the number of placed points remaining afterwards, or -1 when
        there is nothing to undo (or no PV contour is in progress)."""
        if self._mode is not ClipMode.PV_CONTOUR:
            return -1
        if not self._pick_history:
            self._status("PV clip: no points to undo.")
            return -1
        path, head, tail, count = self._pick_history.pop()
        self._path = path
        self._head = head
        self._tail = tail
        self._pick_count = count
        self._redraw_snake()
        if count == 0:
            self._status("PV clip: removed last point — snake is empty.")
        else:
            self._status(f"PV clip: removed last point — {count} left.")
        return count

    # ------------------------------------------------------------------
    def _redraw_snake(self) -> None:
        if self.plotter is None:
            return
        if self._snake_actor is not None:
            try:
                self.plotter.remove_actor(self._snake_actor, reset_camera=False)
            except Exception:
                pass
            self._snake_actor = None

        mesh = self.get_mesh()
        if mesh is None or not self._path:
            return

        pts = np.asarray(mesh.points[self._path], dtype=float)
        cloud = pv.PolyData(pts)
        glyphs = cloud.glyph(
            geom=pv.Sphere(radius=SNAKE_SPHERE_RADIUS),
            scale=False,
            orient=False,
        )
        self._snake_actor = self.plotter.add_mesh(
            glyphs,
            color="blue",
            name="pv_snake",
            reset_camera=False,
            pickable=False,
        )

    def _clear_contour(self) -> None:
        self._head = -1
        self._tail = -1
        self._path = []
        self._pick_history = []
        self._pick_count = 0
        if self._snake_actor is not None:
            try:
                self.plotter.remove_actor(self._snake_actor, reset_camera=False)
            except Exception:
                pass
            self._snake_actor = None

    # ------------------------------------------------------------------
    def _ensure_closed_loop(self, mesh: pv.PolyData) -> List[int]:
        """Close the snake with a tag-restricted geodesic head → tail."""
        if len(self._path) < 2 or self._head < 0 or self._tail < 0:
            return []
        if self._head == self._tail:
            return list(self._path)
        if self._subgraph is None:
            return []
        closing = _subgraph_path(self._subgraph, self._head, self._tail)
        if len(closing) < 2:
            return []
        # Drop the leading head (already at path end); keep the trailing tail
        # to make the closure explicit for downstream selection.
        return list(self._path) + closing[1:]

    # ------------------------------------------------------------------
    def finish_pv_contour(
        self,
        pv_seed_xyz: Sequence[float],
    ) -> Optional[ClipResult]:
        """Close the snake and clip on the PV-seed side.

        The closed vertex-id loop is passed to ``vtkSelectPolyData``
        with ``SetEdgeSearchModeToDijkstra`` and seed-anchored
        ``SetClosestPoint``. Triangles whose majority of vertices lie
        on the PV-seed side are discarded; the resulting open mesh is
        returned.
        """
        if self._mode is not ClipMode.PV_CONTOUR:
            return None
        if len(self._path) < 3 or self._head < 0 or self._tail < 0:
            self._status("PV clip: snake too short — add more points.")
            return None

        mesh = self.get_mesh()
        if mesh is None:
            return None

        loop_ids = self._ensure_closed_loop(mesh)
        
        if len(loop_ids) < 3:
            self._status(
                "PV clip: could not close the snake (no tag-restricted "
                "geodesic between head and tail)."
            )
            self.restore()
            self.cancel()
            return None

        loop_xyz = np.asarray(mesh.points[loop_ids], dtype=float)
        seed = np.asarray(pv_seed_xyz, dtype=float).reshape(3)

        loop_vtk = vtk.vtkPoints()
        for xyz in loop_xyz:
            loop_vtk.InsertNextPoint(float(xyz[0]), float(xyz[1]), float(xyz[2]))

        try:
            sel = vtk.vtkSelectPolyData()
            sel.SetInputData(mesh)
            sel.SetLoop(loop_vtk)
            sel.GenerateSelectionScalarsOn()
            
            if hasattr(sel, "SetSelectionScalarsArrayName"):
                sel.SetSelectionScalarsArrayName("SelectionScalars")            
            
            sel.SetSelectionModeToClosestPointRegion()
            sel.SetClosestPoint(*seed.tolist())
            
            if hasattr(sel, "SetEdgeSearchModeToDijkstra"):
                sel.SetEdgeSearchModeToDijkstra()
            sel.Update()
            selected = pv.wrap(sel.GetOutput())
            
            # 3. Robust scalar retrieval
            # Check for the name, then fallback to active scalars (unnamed arrays)
            if "SelectionScalars" in selected.point_data:
                out_scalars = np.asarray(selected.point_data["SelectionScalars"])
            elif "Selection" in selected.point_data:
                out_scalars = np.asarray(selected.point_data["Selection"])
            elif selected.active_scalars is not None:
                out_scalars = np.asarray(selected.active_scalars)
            else:
                raise RuntimeError("vtkSelectPolyData produced no scalar data.")            
            
            
            if selected is None or selected.n_points == 0:
                raise RuntimeError("empty output")
            if "SelectionScalars" not in selected.point_data:
                raise RuntimeError("no SelectionScalars in output")
            out_scalars = np.asarray(selected.point_data["SelectionScalars"])
        except Exception as exc:
            self._status(f"PV clip: vtkSelectPolyData failed ({exc}).")
            self.restore()
            self.cancel()
            return None

        # vtkSelectPolyData can insert extra points along the loop; its
        # output point count may exceed mesh.n_points. Map each input
        # vertex to its corresponding output scalar via nearest-neighbour
        # so we can classify the ORIGINAL triangles (preserving elemTag).
        pts = np.asarray(mesh.points, dtype=float)
        if out_scalars.size == mesh.n_points:
            scalars_in = out_scalars
        else:
            out_pts = np.asarray(selected.points, dtype=float)
            try:
                tree = cKDTree(out_pts)
                _, idx = tree.query(pts, k=1)
            except Exception as exc:
                self._status(f"PV clip: scalar remap failed ({exc}).")
                self.restore()
                self.cancel()
                return None
            scalars_in = out_scalars[idx]

        d2_to_seed = np.sum((pts - seed) ** 2, axis=1)
        neg_mask = scalars_in < 0.0
        pos_mask = scalars_in > 0.0
        d2_neg = float(np.min(d2_to_seed[neg_mask])) if np.any(neg_mask) else np.inf
        d2_pos = float(np.min(d2_to_seed[pos_mask])) if np.any(pos_mask) else np.inf

        if not np.isfinite(d2_neg) and not np.isfinite(d2_pos):
            self._status("PV clip: degenerate loop (no classified vertices).")
            self.restore()
            self.cancel()
            return None

        pv_side_pt = neg_mask if d2_neg <= d2_pos else pos_mask

        faces = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]
        pv_count = pv_side_pt[faces].sum(axis=1)
        # Discard any triangle with ≥ 1 vertex on the PV side — this is
        # more aggressive than the previous majority rule, but it is the
        # correct behaviour when the loop follows a tag boundary: a
        # majority vote leaves a narrow fringe of PV triangles along the
        # cut. A strict "any PV vertex" rule cleanly separates the two
        # sides because the loop vertices themselves are on the body
        # tag (target_tag) and carry scalar ≥ 0.
        keep_mask = pv_count < 1

        kept_idx = np.where(keep_mask)[0]
        if kept_idx.size == 0 or kept_idx.size == mesh.n_cells:
            self._status("PV clip: loop did not isolate a PV region — aborted.")
            self.restore()
            self.cancel()
            return None

        kept = mesh.extract_cells(kept_idx).extract_surface(algorithm='dataset_surface')

        if not self._validate_mesh(kept):
            self._status(
                "PV clip: post-clip mesh is empty or degenerate — reverted."
            )
            self.restore()
            self.cancel()
            return None

        n_removed = int(mesh.n_cells - int(keep_mask.sum()))
        if n_removed <= 0:
            self._status("PV clip: no triangles were removed — reverted.")
            self.restore()
            self.cancel()
            return None

        self.set_mesh(kept)
        self.cancel()

        self._status(
            f"PV clip done — removed {n_removed} triangles "
            f"(mesh is now open at the PV ostium)."
        )
        return ClipResult(mesh=kept, n_removed=n_removed)

    # ==================================================================
    # Pose memory
    # ==================================================================
    def pose_for(self, seed_key: str, mode: ClipMode) -> Optional[dict]:
        """The remembered geometry for ``seed_key`` in ``mode``, if any.

        Callers use it to decide whether a clip resumes from where the last
        one was left or starts at the seed default; ``None`` means nothing
        has been placed on that seed in that mode yet."""
        entry = self._pose_memory.get((str(seed_key), mode))
        return dict(entry) if entry is not None else None

    def remember_pose(self, seed_key: str, mode: ClipMode, pose: dict) -> None:
        self._pose_memory[(str(seed_key), mode)] = dict(pose)

    def forget_pose(self, seed_key: str, mode: ClipMode) -> None:
        """Drop the remembered geometry — the "reset to default" half.

        Resetting has to erase the memory as well as move the widget, or the
        next start would helpfully restore what the user just asked to be rid
        of."""
        self._pose_memory.pop((str(seed_key), mode), None)

    def _capture_pose(self) -> None:
        """Record the live widget's geometry against the current seed key.

        Called wherever a widget is about to disappear — cancel, mode switch,
        picker hand-off, apply — so every route out of a clip leaves the pose
        behind rather than only the tidy ones."""
        if not self._seed_key:
            return
        if self._sphere_widget is not None:
            self.remember_pose(self._seed_key, ClipMode.SPHERE,
                               self._sphere_pose())
        if self._plane_widget is not None:
            self.remember_pose(self._seed_key, ClipMode.PLANE,
                               self._plane_pose())

    def _sphere_pose(self) -> dict:
        c = self._sphere_widget.GetCenter()
        return {"cx": float(c[0]), "cy": float(c[1]), "cz": float(c[2]),
                "radius": float(self._sphere_widget.GetRadius())}

    def _plane_pose(self) -> dict:
        o = self._plane_widget.GetOrigin()
        n = self._plane_widget.GetNormal()
        return {"ox": float(o[0]), "oy": float(o[1]), "oz": float(o[2]),
                "nx": float(n[0]), "ny": float(n[1]), "nz": float(n[2])}

    def current_pose(self) -> Optional[dict]:
        """The live widget's geometry, or ``None`` when no widget is up."""
        if self._mode is ClipMode.SPHERE and self._sphere_widget is not None:
            return self._sphere_pose()
        if self._mode is ClipMode.PLANE and self._plane_widget is not None:
            return self._plane_pose()
        return None

    def _emit_pose(self) -> None:
        if self.on_pose_changed is None:
            return
        pose = self.current_pose()
        if pose is not None:
            self.on_pose_changed(self._mode, pose)

    # ==================================================================
    # Geometric clipping — sphere
    # ==================================================================
    def start_sphere(
        self,
        center: Sequence[float],
        radius: float,
        seed_key: str = "",
    ) -> None:
        """Raise the sphere widget at ``center`` / ``radius``.

        ``seed_key`` names the anatomical point the sphere belongs to (the
        region whose seed it was placed on); it is what the pose memory is
        filed under. Callers wanting the remembered geometry ask ``pose_for``
        first and pass it in — the tool does not silently override the
        centre it was handed, so "start where I left off" and "start at the
        seed" stay the caller's choice, visible in the call."""
        self.cancel()
        self._mode = ClipMode.SPHERE
        self._seed_key = str(seed_key)
        self._snapshot()

        w = vtk.vtkSphereWidget()
        w.SetInteractor(self.plotter.iren.interactor)
        w.SetRepresentationToSurface()
        w.SetCenter(*[float(c) for c in center])
        w.SetRadius(float(radius))
        w.GetSphereProperty().SetOpacity(0.35)
        w.GetSphereProperty().SetColor(1.0, 0.3, 0.3)
        # Two observers, two jobs: the numbers track the drag continuously,
        # while the red preview — which re-tests every triangle centroid — is
        # recomputed only once the drag ends.
        w.AddObserver("InteractionEvent", self._on_sphere_interaction)
        w.AddObserver("EndInteractionEvent", self._update_sphere_preview)
        w.On()
        self._sphere_widget = w
        self._emit_pose()
        self._status(
            "Sphere: left-drag it to move, right-drag to resize — then "
            "‘Apply clip’ removes everything inside it."
        )

    def set_sphere_pose(self, pose: dict) -> None:
        """Drive the live sphere from typed-in numbers.

        The panel's spin boxes are an equal partner to the mouse, so the
        preview is refreshed exactly as an end-of-drag would."""
        if self._sphere_widget is None:
            return
        self._sphere_widget.SetCenter(
            float(pose["cx"]), float(pose["cy"]), float(pose["cz"]))
        self._sphere_widget.SetRadius(max(float(pose["radius"]),
                                          self._TOL_ABS_FLOOR))
        self._update_sphere_preview()

    def apply_sphere(self) -> Optional[ClipResult]:
        if self._mode is not ClipMode.SPHERE or self._sphere_widget is None:
            return None
        center = np.array(self._sphere_widget.GetCenter(), dtype=float)
        radius = float(self._sphere_widget.GetRadius())
        self._capture_pose()

        mesh = self.get_mesh()
        centroids = self._triangle_centroids(mesh)
        dist = np.linalg.norm(centroids - center, axis=1)
        keep_mask = dist > radius
        return self._finalize_geometric_clip(mesh, keep_mask)

    # ==================================================================
    # Geometric clipping — plane
    # ==================================================================
    def start_plane(
        self,
        origin: Sequence[float],
        normal: Sequence[float],
        seed: Optional[Sequence[float]] = None,
        seed_key: str = "",
    ) -> None:
        """Raise the plane widget at ``origin`` with ``normal``.

        ``seed`` is the anatomical point whose side of the plane gets clipped,
        and is kept apart from ``origin`` on purpose: a resumed plane starts at
        the origin the user left it on, which lies *in* the plane and so says
        nothing about which half to discard. Defaults to ``origin`` — right for
        a first placement, where the two coincide."""
        self.cancel()
        self._mode = ClipMode.PLANE
        self._seed_key = str(seed_key)
        self._snapshot()

        # Fixed reference point for the "which half" question, held while the
        # plane moves.
        ref = origin if seed is None else seed
        self._side_seed = np.asarray(ref, dtype=float).copy()

        w = vtk.vtkImplicitPlaneWidget()
        w.SetInteractor(self.plotter.iren.interactor)
        w.SetPlaceFactor(1.25)
        mesh = self.get_mesh()
        w.SetInputData(mesh)
        w.PlaceWidget(mesh.bounds)
        w.SetOrigin(*[float(o) for o in origin])
        w.SetNormal(*[float(n) for n in normal])
        w.DrawPlaneOff()
        w.OutlineTranslationOff()
        w.AddObserver("InteractionEvent", self._on_plane_interaction)
        w.AddObserver("EndInteractionEvent", self._update_plane_preview)
        w.On()
        self._plane_widget = w
        self._emit_pose()
        self._status(
            "Plane: left-drag the arrowhead to tilt, the rim to slide along "
            "the arrow, the centre ball to shift sideways — then ‘Apply clip’."
        )

    def set_plane_pose(self, pose: dict) -> None:
        """Drive the live plane from typed-in numbers."""
        if self._plane_widget is None:
            return
        normal = np.array([float(pose["nx"]), float(pose["ny"]),
                           float(pose["nz"])], dtype=float)
        if float(np.linalg.norm(normal)) < self._TOL_ABS_FLOOR:
            return
        self._plane_widget.SetOrigin(
            float(pose["ox"]), float(pose["oy"]), float(pose["oz"]))
        self._plane_widget.SetNormal(*normal)
        self._update_plane_preview()

    def apply_plane(self, side_seed: Sequence[float]) -> Optional[ClipResult]:
        """Clip away the half of the mesh that ``side_seed`` sits in."""
        if self._mode is not ClipMode.PLANE or self._plane_widget is None:
            return None
        self._capture_pose()
        origin = np.array(self._plane_widget.GetOrigin(), dtype=float)
        normal = np.array(self._plane_widget.GetNormal(), dtype=float)
        n_norm = float(np.linalg.norm(normal))
        if n_norm < self._TOL_ABS_FLOOR:
            self._status("Clip plane ambiguous — invalid normal.")
            return None
        normal = normal / n_norm

        mesh = self.get_mesh()
        seed = np.asarray(side_seed, dtype=float).reshape(3)

        eps = self._tolerance(mesh)
        seed_side = float((seed - origin) @ normal)
        if abs(seed_side) < eps:
            self._status("Clip plane ambiguous — adjust plane")
            return None

        centroids = self._triangle_centroids(mesh)
        signed = (centroids - origin) @ normal
        keep_mask = (signed * seed_side) < 0.0
        return self._finalize_geometric_clip(mesh, keep_mask)

    # ==================================================================
    # Shared finalization for the geometric clips
    # ==================================================================
    def _finalize_geometric_clip(
        self, mesh: pv.PolyData, keep_mask: np.ndarray,
    ) -> ClipResult:
        kept_count = int(keep_mask.sum())
        if kept_count == 0 or kept_count == mesh.n_cells:
            self._status("Clip: nothing would be clipped — aborted.")
            self.restore()
            self.cancel()
            return ClipResult(mesh=self.get_mesh(), n_removed=0)

        kept = mesh.extract_cells(np.where(keep_mask)[0]).extract_surface(algorithm='dataset_surface')

        if not self._validate_mesh(kept):
            self._status(
                "Clip: post-clip mesh is empty or degenerate — reverted."
            )
            self.restore()
            self.cancel()
            return ClipResult(mesh=self.get_mesh(), n_removed=0)

        self.set_mesh(kept)
        self.cancel()

        n_removed = int(mesh.n_cells - kept_count)
        self._status(
            f"Clip done — removed {n_removed} triangles "
            f"(mesh is now open where the geometry cut it)."
        )
        return ClipResult(mesh=kept, n_removed=n_removed)

    # ==================================================================
    # Geometry helpers
    # ==================================================================
    @staticmethod
    def _triangle_centroids(mesh: pv.PolyData) -> np.ndarray:
        faces = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]
        pts = np.asarray(mesh.points)
        return pts[faces].mean(axis=1)

    # ==================================================================
    # Clip preview (red overlay of triangles that will be removed)
    # ==================================================================
    def _clear_preview(self) -> None:
        if self._preview_actor is not None:
            try:
                self.plotter.remove_actor(self._preview_actor, reset_camera=False)
            except Exception:
                pass
            self._preview_actor = None

    def _on_sphere_interaction(self, obj=None, event=None) -> None:
        """Mid-drag: publish the numbers, leave the overlay alone.

        Re-testing every triangle centroid on each mouse move is what would
        make a large mesh feel sticky, and the overlay is the expensive half —
        the readout is three floats."""
        self._emit_pose()

    def _on_plane_interaction(self, obj=None, event=None) -> None:
        self._emit_pose()

    def _update_sphere_preview(self, obj=None, event=None) -> None:
        self._emit_pose()
        self._clear_preview()
        if self._sphere_widget is None:
            return
        mesh = self.get_mesh()
        if mesh is None:
            return
        center = np.array(self._sphere_widget.GetCenter(), dtype=float)
        radius = float(self._sphere_widget.GetRadius())
        centroids = self._triangle_centroids(mesh)
        clip_mask = np.linalg.norm(centroids - center, axis=1) <= radius
        if not np.any(clip_mask):
            return
        clip_cells = mesh.extract_cells(np.where(clip_mask)[0])
        self._preview_actor = self.plotter.add_mesh(
            clip_cells,
            color="red",
            opacity=0.6,
            lighting=False,
            name="_mv_clip_preview",
            reset_camera=False,
            pickable=False,
        )
        self.plotter.render()

    def _update_plane_preview(self, obj=None, event=None) -> None:
        self._emit_pose()
        self._clear_preview()
        if self._plane_widget is None or self._side_seed is None:
            return
        mesh = self.get_mesh()
        if mesh is None:
            return
        origin = np.array(self._plane_widget.GetOrigin(), dtype=float)
        normal = np.array(self._plane_widget.GetNormal(), dtype=float)
        n_norm = float(np.linalg.norm(normal))
        if n_norm < self._TOL_ABS_FLOOR:
            return
        normal = normal / n_norm
        seed_side = float((self._side_seed - origin) @ normal)
        if abs(seed_side) < self._TOL_ABS_FLOOR:
            return
        centroids = self._triangle_centroids(mesh)
        signed = (centroids - origin) @ normal
        # Triangles on the same side as the seed are the ones that will be clipped.
        clip_mask = (signed * seed_side) >= 0.0
        if not np.any(clip_mask):
            return
        clip_cells = mesh.extract_cells(np.where(clip_mask)[0])
        self._preview_actor = self.plotter.add_mesh(
            clip_cells,
            color="red",
            opacity=0.6,
            lighting=False,
            name="_mv_clip_preview",
            reset_camera=False,
            pickable=False,
        )
        self.plotter.render()

    # ==================================================================
    def _remove_sphere_widget(self) -> None:
        if self._sphere_widget is not None:
            self._sphere_widget.Off()
            self._sphere_widget = None

    def _remove_plane_widget(self) -> None:
        if self._plane_widget is not None:
            self._plane_widget.Off()
            self._plane_widget = None


__all__ = ["ClippingTool", "ClipMode", "ClipResult"]
