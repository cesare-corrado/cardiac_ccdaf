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

2.  **Mitral clip** — either
      * **Sphere**: interactive ``vtkSphereWidget``; triangles whose
        centroid lies inside the sphere are removed; or
      * **Plane**:  interactive ``vtkPlaneWidget``; triangles on the
        "mitral side" of the plane are removed.
    The mitral hole is likewise left open.

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
    MV_SPHERE   = auto()
    MV_PLANE    = auto()


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
    """Interactive PV + mitral clipping with multi-undo and integrity checks."""

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

        # Widgets
        self._sphere_widget: Optional[vtk.vtkSphereWidget] = None
        self._plane_widget: Optional[vtk.vtkImplicitPlaneWidget] = None

        # Clip preview actor (red overlay of to-be-clipped triangles).
        self._preview_actor = None
        # Fixed reference point used to determine the mitral-side for the
        # plane preview (stored on start_mv_plane, equals the initial origin).
        self._mv_seed: Optional[np.ndarray] = None

        # Multi-undo history stack (deep copies).
        self._history: List[pv.PolyData] = []

    # ==================================================================
    # Common helpers
    # ==================================================================
    @property
    def mode(self) -> ClipMode:
        return self._mode

    @property
    def can_undo(self) -> bool:
        """True when a previous mesh state is available to restore.

        Both PV and mitral clips push a snapshot before modifying the mesh,
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
        self._mode = ClipMode.NONE
        self._clear_contour()
        self._clear_subgraph()
        self._clear_preview()
        self._remove_sphere_widget()
        self._remove_plane_widget()
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
        return True

    def _clear_subgraph(self) -> None:
        self._point_tag = None
        self._boundary = None
        self._allowed = None
        self._subgraph = None
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

        self.plotter.enable_point_picking(
            callback=self._on_contour_pick,
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

        p = int(mesh.find_closest_point(np.asarray(picked_point, dtype=float)))

        # Reject picks that leave the allowed region:
        #   * on a tag-boundary vertex, or
        #   * not on the target tag.
        if self._boundary[p]:
            self._status("PV pick ignored: on tag boundary.")
            return
        if int(self._point_tag[p]) != self._target_tag:
            self._status(
                f"PV pick ignored: vertex tag {int(self._point_tag[p])} "
                f"≠ target tag {self._target_tag}."
            )
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
    # Mitral clipping — sphere
    # ==================================================================
    def start_mv_sphere(self, center: Sequence[float], radius: float) -> None:
        self.cancel()
        self._mode = ClipMode.MV_SPHERE
        self._snapshot()

        w = vtk.vtkSphereWidget()
        w.SetInteractor(self.plotter.iren.interactor)
        w.SetRepresentationToSurface()
        w.SetCenter(*[float(c) for c in center])
        w.SetRadius(float(radius))
        w.GetSphereProperty().SetOpacity(0.35)
        w.GetSphereProperty().SetColor(1.0, 0.3, 0.3)
        w.AddObserver("EndInteractionEvent", self._update_sphere_preview)
        w.On()
        self._sphere_widget = w
        self._status("Adjust the sphere, then click ‘Apply clip’.")

    def apply_mv_sphere(self) -> Optional[ClipResult]:
        if self._mode is not ClipMode.MV_SPHERE or self._sphere_widget is None:
            return None
        center = np.array(self._sphere_widget.GetCenter(), dtype=float)
        radius = float(self._sphere_widget.GetRadius())

        mesh = self.get_mesh()
        centroids = self._triangle_centroids(mesh)
        dist = np.linalg.norm(centroids - center, axis=1)
        keep_mask = dist > radius
        return self._finalize_mv_clip(mesh, keep_mask)

    # ==================================================================
    # Mitral clipping — plane
    # ==================================================================
    def start_mv_plane(
        self, origin: Sequence[float], normal: Sequence[float],
    ) -> None:
        self.cancel()
        self._mode = ClipMode.MV_PLANE
        self._snapshot()

        # Store the initial origin as the fixed mitral-side reference point so
        # the preview can determine which half to shade while the plane moves.
        self._mv_seed = np.asarray(origin, dtype=float).copy()

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
        w.AddObserver("EndInteractionEvent", self._update_plane_preview)
        w.On()
        self._plane_widget = w
        self._status("Adjust the plane, then click ‘Apply clip’.")

    def apply_mv_plane(self, mitral_seed: Sequence[float]) -> Optional[ClipResult]:
        if self._mode is not ClipMode.MV_PLANE or self._plane_widget is None:
            return None
        origin = np.array(self._plane_widget.GetOrigin(), dtype=float)
        normal = np.array(self._plane_widget.GetNormal(), dtype=float)
        n_norm = float(np.linalg.norm(normal))
        if n_norm < self._TOL_ABS_FLOOR:
            self._status("Mitral plane ambiguous — invalid normal.")
            return None
        normal = normal / n_norm

        mesh = self.get_mesh()
        seed = np.asarray(mitral_seed, dtype=float).reshape(3)

        eps = self._tolerance(mesh)
        seed_side = float((seed - origin) @ normal)
        if abs(seed_side) < eps:
            self._status("Mitral plane ambiguous — adjust plane")
            return None

        centroids = self._triangle_centroids(mesh)
        signed = (centroids - origin) @ normal
        keep_mask = (signed * seed_side) < 0.0
        return self._finalize_mv_clip(mesh, keep_mask)

    # ==================================================================
    # Shared mitral finalization
    # ==================================================================
    def _finalize_mv_clip(
        self, mesh: pv.PolyData, keep_mask: np.ndarray,
    ) -> ClipResult:
        kept_count = int(keep_mask.sum())
        if kept_count == 0 or kept_count == mesh.n_cells:
            self._status("Mitral clip: nothing would be clipped — aborted.")
            self.restore()
            self.cancel()
            return ClipResult(mesh=self.get_mesh(), n_removed=0)

        kept = mesh.extract_cells(np.where(keep_mask)[0]).extract_surface(algorithm='dataset_surface')

        if not self._validate_mesh(kept):
            self._status(
                "Mitral clip: post-clip mesh is empty or degenerate — reverted."
            )
            self.restore()
            self.cancel()
            return ClipResult(mesh=self.get_mesh(), n_removed=0)

        self.set_mesh(kept)
        self.cancel()

        n_removed = int(mesh.n_cells - kept_count)
        self._status(
            f"Mitral clip done — removed {n_removed} triangles "
            f"(mesh is now open at the mitral annulus)."
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

    def _update_sphere_preview(self, obj=None, event=None) -> None:
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
        self._clear_preview()
        if self._plane_widget is None or self._mv_seed is None:
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
        seed_side = float((self._mv_seed - origin) @ normal)
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
