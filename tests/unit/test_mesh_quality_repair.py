"""
test_mesh_quality_repair.py
===========================
Tests for the triangle quality metric and ``improve_quality`` (step 7 of
``clean``):

* the metric is scale-free, scores 1 for equilateral and 0 for
  degenerate, and matches its definition computed independently;
* the analytic quality gradient matches finite differences;
* the repair improves badly-shaped triangles while leaving topology, the
  point count and the point *order* untouched;
* it does not buy triangle shape with geometry: vertices stay on the input
  surface and within the displacement cap;
* frozen vertices (preserved labels, label seams, open boundaries) never
  move;
* ``clean``'s defaults are the documented ones.

Run with pytest:
    pytest tests/unit/test_mesh_quality_repair.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pytest
import pyvista as pv

from ccdaf.core.mesh_postprocessor import (
    PostprocessOptions,
    _quality_gradient,
    _triangle_quality,
    _triangle_quality_from,
    clean,
    improve_quality,
)


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def _tri(mesh):
    return np.asarray(mesh.faces).reshape(-1, 4)[:, 1:].astype(np.int64)


def _poly(points, tri):
    faces = np.hstack([np.full((tri.shape[0], 1), 3, dtype=np.int64), tri])
    return pv.PolyData(np.asarray(points), faces.ravel())


def _quality_from_definition(mesh):
    """The metric written out from its definition, independently of the
    implementation under test."""
    tri = _tri(mesh)
    p = np.asarray(mesh.points, dtype=float)
    a = np.linalg.norm(p[tri[:, 1]] - p[tri[:, 0]], axis=1)
    b = np.linalg.norm(p[tri[:, 2]] - p[tri[:, 1]], axis=1)
    c = np.linalg.norm(p[tri[:, 0]] - p[tri[:, 2]], axis=1)
    s = 0.5 * (a + b + c)
    area = np.sqrt(np.clip(s * (s - a) * (s - b) * (s - c), 0.0, None))
    return 4.0 * np.sqrt(3.0) * area / (a ** 2 + b ** 2 + c ** 2)


def make_sphere(theta=20, phi=20):
    m = pv.Sphere(radius=10.0, theta_resolution=theta,
                  phi_resolution=phi).triangulate()
    return _poly(m.points, _tri(m))


def make_noisy_sphere(theta=20, phi=20, amp=0.35, seed=0):
    """Sphere with radial noise — plenty of badly-shaped triangles."""
    m = make_sphere(theta, phi)
    rng = np.random.default_rng(seed)
    p = np.asarray(m.points, dtype=float)
    r = np.linalg.norm(p, axis=1)[:, None]
    out = _poly(p + (p / r) * rng.normal(scale=amp, size=(p.shape[0], 1)),
                _tri(m))
    out.cell_data["elemTag"] = np.ones(out.n_cells, dtype=np.int32)
    return out


# ---------------------------------------------------------------------
# the metric
# ---------------------------------------------------------------------
def test_equilateral_scores_one():
    tri = np.array([[0, 1, 2]], dtype=np.int64)
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                    [0.5, np.sqrt(3) / 2, 0.0]])
    assert _triangle_quality_from(pts, tri)[0] == pytest.approx(1.0)


def test_degenerate_scores_zero():
    tri = np.array([[0, 1, 2]], dtype=np.int64)
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    assert _triangle_quality_from(pts, tri)[0] == pytest.approx(0.0)


def test_metric_is_scale_invariant():
    mesh = make_noisy_sphere()
    scaled = _poly(np.asarray(mesh.points) * 7.5, _tri(mesh))
    assert _triangle_quality(scaled) == pytest.approx(
        _triangle_quality(mesh), rel=1e-9)


def test_metric_matches_its_definition():
    """Pins the formula: area over summed squared sides, normalised so an
    equilateral triangle scores 1."""
    mesh = make_noisy_sphere()
    assert _triangle_quality(mesh) == pytest.approx(
        _quality_from_definition(mesh), abs=1e-12)


def test_threshold_flags_the_worse_triangles():
    """Raising the threshold can only ever flag more triangles, never
    different ones."""
    q = _triangle_quality(make_noisy_sphere())
    lo, hi = q < 0.6, q < 0.9
    assert lo.sum() < hi.sum()
    assert np.array_equal(lo & hi, lo)


def test_quality_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(3, 3))
    tri = np.array([[0, 1, 2]], dtype=np.int64)
    q0, grad = _quality_gradient(pts, tri)
    h = 1e-6
    numeric = np.zeros((3, 3))
    for k in range(3):
        for c in range(3):
            shifted = pts.copy()
            shifted[k, c] += h
            numeric[k, c] = (_triangle_quality_from(shifted, tri)[0]
                             - q0[0]) / h
    assert grad[0] == pytest.approx(numeric, abs=1e-5)


# ---------------------------------------------------------------------
# improve_quality
# ---------------------------------------------------------------------
def test_repairs_bad_triangles():
    mesh = make_noisy_sphere()
    before = _triangle_quality(mesh)
    after = _triangle_quality(improve_quality(mesh, iterations=30))
    assert after.min() > before.min()
    assert np.sum(after < 0.8) < np.sum(before < 0.8)


def test_preserves_topology_and_point_order():
    """Same contract as smooth(): only coordinates change."""
    mesh = make_noisy_sphere()
    mesh.point_data["Bipolar"] = np.asarray(mesh.points)[:, 0]
    out = improve_quality(mesh, iterations=10)
    assert out.n_points == mesh.n_points
    assert np.array_equal(np.asarray(out.faces), np.asarray(mesh.faces))
    assert np.array_equal(np.asarray(out.point_data["Bipolar"]),
                          np.asarray(mesh.point_data["Bipolar"]))
    assert np.array_equal(np.asarray(out.cell_data["elemTag"]),
                          np.asarray(mesh.cell_data["elemTag"]))


def test_vertices_stay_on_the_input_surface():
    """The guard against buying triangle shape with geometry."""
    mesh = make_noisy_sphere()
    out = improve_quality(mesh, iterations=30)
    _, closest = mesh.find_closest_cell(np.asarray(out.points, dtype=float),
                                        return_closest_point=True)
    off = np.linalg.norm(np.asarray(out.points) - np.asarray(closest), axis=1)
    assert off.max() < 1e-6


def test_respects_the_displacement_cap():
    mesh = make_noisy_sphere()
    tri, p = _tri(mesh), np.asarray(mesh.points, dtype=float)
    e = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    shortest = np.full(mesh.n_points, np.inf)
    lengths = np.linalg.norm(p[e[:, 0]] - p[e[:, 1]], axis=1)
    for col in (0, 1):
        np.minimum.at(shortest, e[:, col], lengths)
    out = improve_quality(mesh, iterations=30, max_shift=0.25)
    moved = np.linalg.norm(np.asarray(out.points, dtype=float) - p, axis=1)
    # cap is relative to the mean incident edge, which is >= the shortest
    assert np.all(moved <= 0.25 * shortest * 3.0 + 1e-9)


def test_never_inverts_a_triangle():
    mesh = make_noisy_sphere()
    out = improve_quality(mesh, iterations=30)
    tri = _tri(mesh)
    before = np.cross(np.asarray(mesh.points)[tri[:, 1]] - np.asarray(mesh.points)[tri[:, 0]],
                      np.asarray(mesh.points)[tri[:, 2]] - np.asarray(mesh.points)[tri[:, 0]])
    after = np.cross(np.asarray(out.points)[tri[:, 1]] - np.asarray(out.points)[tri[:, 0]],
                     np.asarray(out.points)[tri[:, 2]] - np.asarray(out.points)[tri[:, 0]])
    assert np.all(np.sum(before * after, axis=1) > 0.0)


def test_never_makes_things_worse():
    """Keeps the best iterate, so the result cannot be worse than the input
    even when the objective stalls."""
    mesh = make_noisy_sphere()
    for iters in (1, 5, 40):
        out = improve_quality(mesh, iterations=iters)
        assert (np.sum(_triangle_quality(out) < 0.8)
                <= np.sum(_triangle_quality(mesh) < 0.8))


def test_freezes_preserved_labels():
    mesh = make_noisy_sphere()
    tri = _tri(mesh)
    tags = np.asarray(mesh.cell_data["elemTag"]).copy()
    tags[:50] = 11
    mesh.cell_data["elemTag"] = tags
    protected = np.unique(tri[tags == 11].ravel())
    out = improve_quality(mesh, iterations=30, preserve_labels=(11,))
    assert np.array_equal(np.asarray(out.points)[protected],
                          np.asarray(mesh.points)[protected])


def test_freezes_label_seams():
    mesh = make_noisy_sphere()
    tri = _tri(mesh)
    zc = np.asarray(mesh.points)[tri].mean(axis=1)[:, 2]
    mesh.cell_data["elemTag"] = np.where(zc > 0, 1, 2).astype(np.int32)
    e = np.sort(np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]),
                axis=1)
    owner = np.tile(np.arange(tri.shape[0]), 3)
    order = np.lexsort((e[:, 1], e[:, 0]))
    e, owner = e[order], owner[order]
    pair = np.flatnonzero(np.all(e[:-1] == e[1:], axis=1))
    tags = np.asarray(mesh.cell_data["elemTag"])
    seam = np.unique(e[pair[tags[owner[pair]] != tags[owner[pair + 1]]]])
    out = improve_quality(mesh, iterations=30)
    assert np.array_equal(np.asarray(out.points)[seam],
                          np.asarray(mesh.points)[seam])


def test_freezes_open_boundaries():
    """On a clipped mesh these are the PV ostia and mitral-valve rims."""
    mesh = make_noisy_sphere()
    tri = _tri(mesh)
    keep = np.asarray(mesh.points)[tri].mean(axis=1)[:, 2] > 0
    hemi = _poly(mesh.points, tri[keep])
    e = np.sort(np.vstack([hemi.faces.reshape(-1, 4)[:, 1:][:, [0, 1]],
                           hemi.faces.reshape(-1, 4)[:, 1:][:, [1, 2]],
                           hemi.faces.reshape(-1, 4)[:, 1:][:, [2, 0]]]), axis=1)
    uniq, counts = np.unique(e, axis=0, return_counts=True)
    rim = np.unique(uniq[counts == 1])
    out = improve_quality(hemi, iterations=30)
    assert np.array_equal(np.asarray(out.points)[rim],
                          np.asarray(hemi.points)[rim])


def test_zero_iterations_is_a_no_op():
    mesh = make_noisy_sphere()
    out = improve_quality(mesh, iterations=0)
    assert np.array_equal(np.asarray(out.points), np.asarray(mesh.points))


def test_reports_progress():
    mesh = make_noisy_sphere()
    seen: list[tuple[int, int]] = []
    improve_quality(mesh, iterations=8,
                    on_progress=lambda i, n: seen.append((i, n)))
    assert seen, "no progress reported"
    assert all(n == 8 for _, n in seen)
    assert seen[-1][0] == 8, "final tick missing, bar would stall short"


def test_clean_forwards_progress():
    mesh = make_noisy_sphere()
    seen: list[tuple[int, int]] = []
    clean(mesh, quality_threshold=0.8, smooth_iterations=5,
          quality_relaxation=0.05,
          on_progress=lambda i, n: seen.append((i, n)))
    assert seen, "clean did not forward progress"


def test_does_not_mutate_input():
    mesh = make_noisy_sphere()
    before = np.asarray(mesh.points).copy()
    improve_quality(mesh, iterations=20)
    assert np.array_equal(np.asarray(mesh.points), before)


# ---------------------------------------------------------------------
# integration with clean
# ---------------------------------------------------------------------
def test_clean_defaults_are_the_documented_ones():
    opts = PostprocessOptions()
    assert opts.clean_quality_threshold == 0.8      # 1 - 0.2
    assert opts.clean_quality_relaxation == 0.05


def test_clean_still_repairs_quality():
    mesh = make_noisy_sphere()
    out = clean(mesh, quality_threshold=0.8, smooth_iterations=30,
                quality_relaxation=0.05)
    assert (np.sum(_triangle_quality(out) < 0.8)
            < np.sum(_triangle_quality(mesh) < 0.8))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
