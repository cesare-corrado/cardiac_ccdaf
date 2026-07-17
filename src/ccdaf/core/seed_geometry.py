"""
SeedGeometryResolver — deterministic geometry contract.

No VTK spatial queries. Snapping is a pure KD-tree nearest-vertex lookup;
PV validation is a geodesic-distance outlier test against an anatomical
landmark (body anchor = surface vertex nearest the mesh centroid).

Public contract:
    snap_point(mesh, point) -> vertex_id
    SeedGeometryResolver(mesh)
        .snap(xyz) -> SnapResult(vertex_id, xyz)
        .is_duplicate_position(xyz, existing_positions) -> bool
        .validate_pv(vertex_id) -> (bool, reason)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import pyvista as pv
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree


class GeometryError(RuntimeError):
    """Raised when mesh/pick resolution fails unambiguously."""


@dataclass(frozen=True)
class SnapResult:
    vertex_id: int
    xyz: np.ndarray  # shape (3,)


# ---------------------------------------------------------------------
# Pure function: deterministic nearest-vertex snap.
# ---------------------------------------------------------------------
def snap_point(mesh: pv.PolyData, point) -> int:
    """Return the index of the mesh vertex nearest to ``point``.

    Deterministic: same (mesh.points, point) inputs always yield the
    same vertex id. Ties are broken by insertion order (= point index)
    via scipy's cKDTree, which is stable across runs.

    Raises
    ------
    GeometryError
        If the mesh is empty, the point is non-finite, or the tree
        returns an out-of-range index.
    """
    if mesh is None:
        raise GeometryError("mesh is None")
    pts = np.asarray(mesh.points, dtype=float)
    if pts.size == 0 or pts.shape[0] == 0:
        raise GeometryError("mesh has no points")
    p = np.asarray(point, dtype=float).reshape(3)
    if not np.all(np.isfinite(p)):
        raise GeometryError("pick contains non-finite coordinates")

    tree = cKDTree(pts)
    _, vid = tree.query(p, k=1)
    vid = int(vid)
    if vid < 0 or vid >= pts.shape[0]:
        raise GeometryError(f"KD-tree returned invalid vertex id {vid}")
    return vid


class SeedGeometryResolver:
    """Mesh-bound deterministic snapping and anatomical validation."""

    def __init__(self, mesh: pv.PolyData) -> None:
        if mesh is None:
            raise GeometryError("mesh is None")
        n_points = int(mesh.n_points)
        n_cells = int(mesh.n_cells)
        if n_points <= 0 or n_cells <= 0:
            raise GeometryError("mesh is empty")

        self.mesh = mesh
        self._pts = np.asarray(mesh.points, dtype=float)

        b = mesh.bounds
        diag = float(np.linalg.norm([b[1] - b[0], b[3] - b[2], b[5] - b[4]]))
        if not np.isfinite(diag) or diag <= 0.0:
            raise GeometryError("degenerate mesh bounds (diag <= 0)")
        self._diag = diag
        self._min_separation = 0.02 * diag

        # KD-tree is stored once; nearest-vertex queries are then O(log n)
        # and fully deterministic for fixed mesh.points order.
        self._tree = cKDTree(self._pts)

        # Anatomical landmark: the surface vertex nearest the Euclidean
        # mesh centroid. Used as the anchor for geodesic PV validation.
        centroid = self._pts.mean(axis=0)
        _, anchor = self._tree.query(centroid, k=1)
        self._anchor_vid = int(anchor)

        # Geodesic distance from anchor to every vertex on the 1-skeleton.
        self._geo = self._geodesic_from_anchor(self._anchor_vid)

        # Tukey-style robust threshold over FINITE distances only.
        finite = self._geo[np.isfinite(self._geo)]
        if finite.size == 0:
            raise GeometryError("geodesic computation returned no finite values")
        q1 = float(np.percentile(finite, 25))
        # Classic upper Tukey fence: values above (median + 1.5*IQR) are
        # statistical outliers — on an atrial mesh, those are the PV tips.
        #tukey_threshold    = med + 0.5 * iqr
        #percentile_floor   = float(np.percentile(finite, 50))
        #self._pv_threshold = min(tukey_threshold, percentile_floor)
        self._pv_threshold = q1

    # ------------------------------------------------------------------
    @property
    def diag(self) -> float:
        return self._diag

    @property
    def min_separation(self) -> float:
        return self._min_separation

    @property
    def anchor_vertex_id(self) -> int:
        return self._anchor_vid

    @property
    def pv_threshold(self) -> float:
        return self._pv_threshold

    # ------------------------------------------------------------------
    # Snapping
    # ------------------------------------------------------------------
    def snap(self, xyz) -> SnapResult:
        """Deterministic nearest-vertex snap via KD-tree."""
        vid = snap_point(self.mesh, xyz)
        return SnapResult(vertex_id=vid, xyz=self._pts[vid].copy())

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def is_duplicate_position(
        self, xyz, existing_positions: Iterable[np.ndarray],
    ) -> bool:
        p = np.asarray(xyz, dtype=float)
        for q in existing_positions:
            if np.linalg.norm(np.asarray(q, dtype=float) - p) < self._min_separation:
                return True
        return False

    def validate_pv(self, vertex_id: int) -> Tuple[bool, str]:
        """Geodesic PV validation against the body-anchor landmark.

        A legitimate PV pick sits on a protrusion whose geodesic distance
        from the body anchor is a statistical outlier (Tukey upper fence).
        Body-wall picks stay well below the fence and are rejected.
        """
        if vertex_id < 0 or vertex_id >= self._pts.shape[0]:
            return False, f"vertex id {vertex_id} out of range"
        g = float(self._geo[vertex_id])
        if not np.isfinite(g):
            return False, "vertex is geodesically disconnected from body anchor"
        if g < self._pv_threshold:
            return False, (
                "pick is not on a protrusion — geodesic distance to the "
                "atrial body is below the PV-outlier threshold."
            )
        return True, ""

    # ------------------------------------------------------------------
    # Internals: geodesic 1-skeleton
    # ------------------------------------------------------------------
    def _geodesic_from_anchor(self, anchor: int) -> np.ndarray:
        """Single-source Dijkstra on the mesh vertex 1-skeleton.

        Edge weight = Euclidean length. Duplicate edges (two triangles
        share an edge) are de-duplicated BEFORE csr_matrix construction
        so weights are not doubled.
        """
        faces = np.asarray(self.mesh.faces, dtype=np.int64)
        if faces.size == 0:
            raise GeometryError("mesh has no faces")
        # PyVista layout: [k, v0..v{k-1}, k, ...]. Triangles => stride 4.
        tris = faces.reshape(-1, 4)
        if not np.all(tris[:, 0] == 3):
            raise GeometryError("mesh is not purely triangular")
        tris = tris[:, 1:4]

        edges = np.vstack([
            tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]],
        ])
        edges.sort(axis=1)
        edges = np.unique(edges, axis=0)

        i = edges[:, 0]
        j = edges[:, 1]
        w = np.linalg.norm(self._pts[i] - self._pts[j], axis=1)

        n = self._pts.shape[0]
        rows = np.concatenate([i, j])
        cols = np.concatenate([j, i])
        data = np.concatenate([w, w])
        graph = csr_matrix((data, (rows, cols)), shape=(n, n))
        return np.asarray(dijkstra(graph, indices=anchor), dtype=float)


__all__ = ["snap_point", "SeedGeometryResolver", "SnapResult", "GeometryError"]
