"""
test_eikonal.py
===============
Geodesic distance by the Fast Iterative Method.

The contract:

* a plane front is reproduced exactly: the local updates are exact for a
  front that is planar across an element;
* from a point, in a convex box, the result is never shorter than the
  straight line and far closer to it than paths along edges (near a point
  source the method is first order, so a coarse mesh is not exact);
* the result does not depend on the number of threads;
* a node the front cannot reach stays at ``inf`` and is counted;
* the distance ratio is 0 on its first set, 1 on its second, 0.5 where
  both are 0, and ``nan`` where a distance is ``inf``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pytest
import pyvista as pv

from ccdaf.core import eikonal as E
from ccdaf.core import volume_mesh as vm


def _box(n=13, size=(12.0, 9.0, 6.0)):
    grid = pv.ImageData(dimensions=(n, n, n),
                        spacing=tuple(s / (n - 1) for s in size)).cast_to_unstructured_grid()
    grid = grid.triangulate()
    return np.asarray(grid.points, dtype=float), vm.tetrahedra(grid)


def test_a_plane_front_is_exact():
    pts, tets = _box()
    T, report = E.distance(pts, tets, np.where(np.isclose(pts[:, 0], 0.0))[0])
    assert report.converged and report.unreached == 0
    assert np.allclose(T, pts[:, 0], atol=1e-9)


def test_a_point_source_beats_paths_along_edges():
    import scipy.sparse as sp
    from scipy.sparse.csgraph import dijkstra
    pts, tets = _box()
    T, _ = E.distance(pts, tets, [0])
    exact = np.linalg.norm(pts - pts[0], axis=1)
    assert np.all(T >= exact - 1e-9)                 # never shorter than straight
    edges = np.unique(np.sort(np.vstack([tets[:, [a, b]] for a, b in
                                         ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))]), 1), axis=0)
    w = np.linalg.norm(pts[edges[:, 0]] - pts[edges[:, 1]], axis=1)
    g = sp.coo_matrix((w, (edges[:, 0], edges[:, 1])), shape=(len(pts),) * 2).tocsr()
    paths = dijkstra(g + g.T, indices=0)
    far = exact > 3.0
    fim_err = np.mean(np.abs(T - exact)[far] / exact[far])
    path_err = np.mean(np.abs(paths - exact)[far] / exact[far])
    assert fim_err < 0.1 and fim_err < path_err / 3


def test_the_result_does_not_depend_on_the_threads():
    pts, tets = _box(9)
    one, _ = E.distance(pts, tets, [0, 5], threads=1)
    many, _ = E.distance(pts, tets, [0, 5])
    assert np.array_equal(one, many)


def test_an_unreachable_piece_stays_infinite():
    pts, tets = _box(5)
    far = pts + 100.0
    both = np.vstack([pts, far])
    tets2 = np.vstack([tets, tets + len(pts)])
    T, report = E.distance(both, tets2, [0])
    assert report.unreached == len(pts)
    assert np.all(np.isinf(T[len(pts):])) and np.all(np.isfinite(T[:len(pts)]))


def test_a_shared_adjacency_gives_the_same_result():
    pts, tets = _box(7)
    adj = E.adjacency(tets, len(pts))
    a, _ = E.distance(pts, tets, [3], adj=adj)
    b, _ = E.distance(pts, tets, [3])
    assert np.array_equal(a, b)


def test_a_solve_needs_a_source():
    pts, tets = _box(5)
    with pytest.raises(ValueError, match="source"):
        E.distance(pts, tets, [])


def test_the_distance_ratio():
    r = E.distance_ratio(np.array([0.0, 2.0, 0.0, 1.0, np.inf]),
                         np.array([3.0, 0.0, 0.0, 3.0, 1.0]))
    assert r[0] == 0.0 and r[1] == 1.0 and r[2] == 0.5 and np.isclose(r[3], 0.25)
    assert np.isnan(r[4])
