"""
opening_band
============
Tag a thin band round one opening of an open surface.

A vein clipped right at its ostium leaves no stub to tag: the opening's
edge runs straight along the atrial body, and region growing from a seed
finds no vein to grow into. What anatomy says is there is a ring of vein
tissue at that edge, so the band of faces within a set distance of the
opening's edge is given the vein's label.

The edge needs no drawing. On an open surface every opening is a closed
loop of boundary edges (edges with one face), so a click near the opening
picks its loop, and the band is everything within ``width`` of the loop,
measured along the surface. The click only says which loop; it need not
lie on it. A face nearer another opening is left to that opening, so a
wide band stops halfway to the next vein.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pyvista as pv
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, dijkstra


def _faces(mesh: pv.PolyData) -> np.ndarray:
    return np.asarray(mesh.faces).reshape(-1, 4)[:, 1:].astype(np.int64)


def boundary_loops(mesh: pv.PolyData) -> List[np.ndarray]:
    """The node ids of each opening's edge loop, largest loop first."""
    tri = _faces(mesh)
    edges = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]],
                                    tri[:, [0, 2]]]), axis=1)
    unique, count = np.unique(edges, axis=0, return_counts=True)
    border = unique[count == 1]
    if len(border) == 0:
        return []
    n = mesh.n_points
    graph = sp.coo_matrix((np.ones(len(border)), (border[:, 0], border[:, 1])),
                          shape=(n, n))
    _k, comp = connected_components(graph, directed=False)
    on = np.unique(border)
    loops = [on[comp[on] == c] for c in np.unique(comp[on])]
    return sorted(loops, key=len, reverse=True)


def _edge_graph(mesh: pv.PolyData, tri: np.ndarray) -> sp.csr_matrix:
    edges = np.unique(np.sort(np.concatenate(
        [tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [0, 2]]]), axis=1), axis=0)
    pts = np.asarray(mesh.points, dtype=float)
    w = np.maximum(np.linalg.norm(pts[edges[:, 1]] - pts[edges[:, 0]], axis=1),
                   1e-12)
    n = mesh.n_points
    return sp.coo_matrix((w, (edges[:, 0], edges[:, 1])), shape=(n, n)).tocsr()


def band_faces(mesh: pv.PolyData, point, width: float
               ) -> Tuple[np.ndarray, np.ndarray]:
    """The faces within *width* of the opening nearest *point*, and that
    opening's loop nodes.

    *width* is in mesh units, measured along the surface. Faces touching
    the loop are always in the band, so a width finer than the mesh still
    tags one row. Raises ``ValueError`` on a closed surface.
    """
    loops = boundary_loops(mesh)
    if not loops:
        raise ValueError("this surface has no openings")
    pts = np.asarray(mesh.points, dtype=float)
    p = np.asarray(point, dtype=float)
    loop = min(loops, key=lambda nodes: float(
        np.min(np.linalg.norm(pts[nodes] - p, axis=1))))
    faces = loop_band(mesh, loop, width)
    others = [lp for lp in loops if lp is not loop]
    if others and len(faces):
        # A face nearer another opening is that opening's: a wide band
        # stops halfway to the next vein instead of running into its edge.
        tri = _faces(mesh)
        graph = _edge_graph(mesh, tri)
        mine = dijkstra(graph, directed=False, indices=loop, min_only=True,
                        limit=float(width) * 1.0001)
        theirs = dijkstra(graph, directed=False,
                          indices=np.concatenate(others), min_only=True,
                          limit=float(width) * 1.0001)
        keep = mine[tri[faces]].mean(1) <= theirs[tri[faces]].mean(1)
        faces = faces[keep]
    return faces, loop


def loop_band(mesh: pv.PolyData, loop: np.ndarray, width: float
              ) -> np.ndarray:
    """The faces within *width* (mesh units, along the surface) of the
    edge loop *loop*; faces touching the loop always."""
    tri = _faces(mesh)
    dist = dijkstra(_edge_graph(mesh, tri), directed=False, indices=loop,
                    min_only=True, limit=float(width) * 1.0001)
    inside = np.isfinite(dist)
    touching = np.isin(tri, loop).any(1)
    return np.where(inside[tri].all(1) | touching)[0]


__all__ = ["boundary_loops", "band_faces", "loop_band"]
