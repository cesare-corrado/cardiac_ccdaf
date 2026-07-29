"""
test_mesh_remesh.py
===================
Tests for ``mesh_postprocessor.remesh`` and the ``refine_mode`` switch:

* the target -> [min, max] band widens by the right factors in each of
  the three regimes, and depends only on the geometry;
* ``target_edge`` and ``min_edge``/``max_edge`` are mutually exclusive;
* resampling lands the edge lengths in the band, keeps the surface
  manifold, and improves triangle quality;
* unlike ``refine`` it can also coarsen;
* label seams and open boundaries survive verbatim, and ``elemTag``
  transfers by descent (exact values and dtype), not by proximity;
* ``apply`` dispatches on ``refine_mode`` and rejects a target given
  together with a band.

Uses synthetic sphere meshes (no real EAM data required).

Run with pytest:
    pytest tests/unit/test_mesh_remesh.py

Run as a standalone script:
    python tests/unit/test_mesh_remesh.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pytest
import pyvista as pv

from ccdaf.core.mesh_postprocessor import (
    PostprocessOptions,
    REFINE_ADAPTIVE,
    REFINE_RESAMPLE,
    _band_from_target,
    _mean_edge_length,
    apply,
    remesh,
)


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def _tri(mesh):
    return np.asarray(mesh.faces).reshape(-1, 4)[:, 1:].astype(np.int64)


def _poly(points, tri):
    faces = np.hstack([np.full((tri.shape[0], 1), 3, dtype=np.int64), tri])
    return pv.PolyData(np.asarray(points), faces.ravel())


def _edges(mesh):
    tri = _tri(mesh)
    return np.sort(np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]),
                   axis=1)


def _edge_lengths(mesh):
    e = _edges(mesh)
    # float64 throughout: pv.Sphere hands back float32 points, and summing
    # a few hundred float32 lengths loses ~1e-7 relative on its own.
    p = np.asarray(mesh.points, dtype=float)
    return np.linalg.norm(p[e[:, 0]] - p[e[:, 1]], axis=1)


def _edge_counts(mesh):
    _, counts = np.unique(_edges(mesh), axis=0, return_counts=True)
    return counts


def _quality(mesh):
    """Triangle badness: 0 equilateral, 1 degenerate."""
    tri = _tri(mesh)
    p = np.asarray(mesh.points)
    a = np.linalg.norm(p[tri[:, 1]] - p[tri[:, 0]], axis=1)
    b = np.linalg.norm(p[tri[:, 2]] - p[tri[:, 1]], axis=1)
    c = np.linalg.norm(p[tri[:, 0]] - p[tri[:, 2]], axis=1)
    s = 0.5 * (a + b + c)
    area = np.sqrt(np.clip(s * (s - a) * (s - b) * (s - c), 0.0, None))
    return 1.0 - 4.0 * np.sqrt(3.0) * area / (a ** 2 + b ** 2 + c ** 2)


def make_sphere(theta=30, phi=30, radius=10.0):
    m = pv.Sphere(radius=radius, theta_resolution=theta,
                  phi_resolution=phi).triangulate()
    return _poly(m.points, _tri(m))


def make_tagged_sphere(theta=30, phi=30):
    """Sphere split into two labelled hemispheres, with a point field."""
    m = make_sphere(theta, phi)
    tri = _tri(m)
    zc = np.asarray(m.points)[tri].mean(axis=1)[:, 2]
    m.cell_data["elemTag"] = np.where(zc > 0.0, 1, 2).astype(np.int32)
    m.point_data["Bipolar"] = np.asarray(m.points)[:, 0]
    return m


def _seam(mesh):
    """(unique seam vertices, seam edges) of the elemTag interface."""
    tri = _tri(mesh)
    tags = np.asarray(mesh.cell_data["elemTag"])
    e = _edges(mesh)
    owner = np.tile(np.arange(tri.shape[0]), 3)
    order = np.lexsort((e[:, 1], e[:, 0]))
    e, owner = e[order], owner[order]
    pair = np.flatnonzero(np.all(e[:-1] == e[1:], axis=1))
    seam = pair[tags[owner[pair]] != tags[owner[pair + 1]]]
    return np.unique(e[seam]), e[seam]


def _seam_length(mesh):
    _, edges = _seam(mesh)
    p = np.asarray(mesh.points, dtype=float)
    return float(np.linalg.norm(p[edges[:, 0]] - p[edges[:, 1]],
                                axis=1).sum())


def _boundary_vertices(mesh):
    uniq, counts = np.unique(_edges(mesh), axis=0, return_counts=True)
    return np.unique(uniq[counts == 1])


def _contains_all(haystack, needles, tol=1e-12):
    """Every point of *needles* appears in *haystack* (to *tol*)."""
    from scipy.spatial import cKDTree
    if len(needles) == 0:
        return True
    return bool(cKDTree(haystack).query(needles, k=1)[0].max() <= tol)


# ---------------------------------------------------------------------
# band derivation
# ---------------------------------------------------------------------
def test_band_fine_regime_halves_and_scales_by_1_8():
    """Target well below the mesh's mean edge: [0.5 t, 1.8 t]."""
    mesh = make_sphere()
    avrg = _mean_edge_length(mesh)
    target = 0.3 * avrg
    assert _band_from_target(mesh, target) == pytest.approx(
        (0.50 * target, 1.8 * target))


def test_band_near_regime():
    """Target comparable to the mesh's mean edge: [0.65 t, 1.7 t]."""
    mesh = make_sphere()
    target = _mean_edge_length(mesh)
    assert _band_from_target(mesh, target) == pytest.approx(
        (0.65 * target, 1.7 * target))


def test_band_coarse_regime():
    """Target well above the mesh's mean edge: [0.75 t, 1.8 t]."""
    mesh = make_sphere()
    target = 3.0 * _mean_edge_length(mesh)
    assert _band_from_target(mesh, target) == pytest.approx(
        (0.75 * target, 1.8 * target))


def test_band_regime_switches_at_half_and_one_and_a_half():
    mesh = make_sphere()
    avrg = _mean_edge_length(mesh)
    below = _band_from_target(mesh, 0.499 * avrg)
    above = _band_from_target(mesh, 0.501 * avrg)
    assert below[0] / (0.499 * avrg) == pytest.approx(0.50)
    assert above[0] / (0.501 * avrg) == pytest.approx(0.65)
    coarse = _band_from_target(mesh, 1.501 * avrg)
    assert coarse[0] / (1.501 * avrg) == pytest.approx(0.75)


def test_band_is_deterministic_under_element_reordering():
    """The band is a function of the geometry alone: shuffling the
    triangles must not move a mesh into a different regime."""
    mesh = make_sphere()
    rng = np.random.default_rng(0)
    shuffled = _poly(mesh.points, _tri(mesh)[rng.permutation(mesh.n_cells)])
    target = 0.55 * _mean_edge_length(mesh)
    assert _band_from_target(shuffled, target) == pytest.approx(
        _band_from_target(mesh, target))


# ---------------------------------------------------------------------
# argument handling
# ---------------------------------------------------------------------
def test_target_and_band_together_raise():
    mesh = make_sphere()
    with pytest.raises(ValueError, match="not both"):
        remesh(mesh, target_edge=0.3, max_edge=0.5)
    with pytest.raises(ValueError, match="not both"):
        remesh(mesh, target_edge=0.3, min_edge=0.1)


def test_no_size_at_all_raises():
    with pytest.raises(ValueError):
        remesh(make_sphere())


def test_inverted_band_raises():
    with pytest.raises(ValueError, match="min_edge"):
        remesh(make_sphere(), min_edge=1.0, max_edge=0.5)


def test_explicit_band_is_used_verbatim():
    mesh = make_sphere()
    out = remesh(mesh, min_edge=0.4, max_edge=1.2)
    lengths = _edge_lengths(out)
    assert lengths.max() <= 1.2 * 1.05


# ---------------------------------------------------------------------
# resampling behaviour
# ---------------------------------------------------------------------
def test_edges_land_in_band():
    mesh = make_sphere()
    lo, hi = _band_from_target(mesh, 0.5 * _mean_edge_length(mesh))
    out = remesh(mesh, target_edge=0.5 * _mean_edge_length(mesh))
    lengths = _edge_lengths(out)
    out_of_band = np.mean((lengths < lo) | (lengths > hi))
    assert out_of_band < 0.02, f"{out_of_band:.1%} of edges outside the band"


def test_stays_manifold_and_closed():
    mesh = make_sphere()
    out = remesh(mesh, target_edge=0.5 * _mean_edge_length(mesh))
    counts = _edge_counts(out)
    assert counts.min() == 2 and counts.max() == 2
    tri = _tri(out)
    assert len(np.unique(np.sort(tri, axis=1), axis=0)) == tri.shape[0]


def test_improves_triangle_quality():
    mesh = make_sphere()
    out = remesh(mesh, target_edge=0.5 * _mean_edge_length(mesh))
    assert _quality(out).mean() < _quality(mesh).mean()


def test_preserves_the_surface():
    """A resampled sphere is still a sphere of the same radius."""
    mesh = make_sphere()
    out = remesh(mesh, target_edge=0.5 * _mean_edge_length(mesh))
    r = np.linalg.norm(np.asarray(out.points), axis=1)
    assert abs(r.mean() - 10.0) < 0.05
    assert r.std() < 0.05


def test_coarsens_where_refine_cannot():
    """The point of remesh over refine: the point count can go down."""
    mesh = make_sphere()
    out = remesh(mesh, target_edge=3.0 * _mean_edge_length(mesh))
    assert out.n_points < mesh.n_points
    assert _edge_lengths(out).mean() > _edge_lengths(mesh).mean()


def test_refines_when_target_is_smaller():
    mesh = make_sphere()
    out = remesh(mesh, target_edge=0.4 * _mean_edge_length(mesh))
    assert out.n_points > mesh.n_points


def test_surf_corr_at_one_disables_collapsing():
    """surf_corr is a dot product, so 1.0 vetoes every collapse and the
    point count can only grow."""
    mesh = make_sphere()
    out = remesh(mesh, target_edge=3.0 * _mean_edge_length(mesh),
                 surf_corr=1.0)
    assert out.n_points == mesh.n_points


def test_reports_progress_per_pass():
    mesh = make_sphere()
    seen: list[tuple[int, int]] = []
    remesh(mesh, target_edge=0.5 * _mean_edge_length(mesh), n_passes=6,
           on_progress=lambda i, n: seen.append((i, n)))
    assert seen, "no progress reported"
    assert [i for i, _ in seen] == list(range(1, len(seen) + 1))
    assert all(n == 6 for _, n in seen)
    assert seen[-1][0] <= 6


def test_does_not_mutate_input():
    mesh = make_sphere()
    before_pts = np.asarray(mesh.points).copy()
    before_tri = _tri(mesh).copy()
    remesh(mesh, target_edge=0.5 * _mean_edge_length(mesh))
    assert np.array_equal(np.asarray(mesh.points), before_pts)
    assert np.array_equal(_tri(mesh), before_tri)


# ---------------------------------------------------------------------
# labels and boundaries
# ---------------------------------------------------------------------
def test_seam_curve_survives_verbatim():
    mesh = make_tagged_sphere()
    out = remesh(mesh, target_edge=0.5 * _mean_edge_length(mesh))
    src_seam, _ = _seam(mesh)
    assert _contains_all(np.asarray(out.points),
                         np.asarray(mesh.points)[src_seam]), (
        "a label-seam vertex was moved or collapsed away")
    assert _seam_length(out) == pytest.approx(_seam_length(mesh), rel=1e-9)


def test_seam_may_still_be_refined():
    """Preserving the seam means not redrawing it, not freezing its
    resolution: seam edges can still be split."""
    mesh = make_tagged_sphere()
    out = remesh(mesh, target_edge=0.4 * _mean_edge_length(mesh))
    assert len(_seam(out)[0]) > len(_seam(mesh)[0])


def test_preserve_labels_empty_lets_the_seam_move():
    """``()`` opts out of seam protection, letting the seam be redrawn."""
    mesh = make_tagged_sphere()
    out = remesh(mesh, target_edge=0.5 * _mean_edge_length(mesh),
                 preserve_labels=())
    assert _seam_length(out) != pytest.approx(_seam_length(mesh), rel=1e-6)


def test_preserve_labels_selects_which_seams():
    mesh = make_tagged_sphere()
    out = remesh(mesh, target_edge=0.5 * _mean_edge_length(mesh),
                 preserve_labels=(1,))
    src_seam, _ = _seam(mesh)
    assert _contains_all(np.asarray(out.points),
                         np.asarray(mesh.points)[src_seam])


def test_elem_tag_transfers_exactly():
    mesh = make_tagged_sphere()
    out = remesh(mesh, target_edge=0.5 * _mean_edge_length(mesh))
    tags = np.asarray(out.cell_data["elemTag"])
    assert tags.dtype == np.asarray(mesh.cell_data["elemTag"]).dtype
    assert set(np.unique(tags).tolist()) == {1, 2}
    assert "Bipolar" in out.point_data
    assert out.point_data["Bipolar"].shape[0] == out.n_points


def test_open_boundary_is_frozen_by_default():
    mesh = make_tagged_sphere()
    hemi = _poly(mesh.points, _tri(mesh)[np.asarray(
        mesh.cell_data["elemTag"]) == 1])
    rim = _boundary_vertices(hemi)
    out = remesh(hemi, target_edge=0.5 * _mean_edge_length(hemi))
    assert _contains_all(np.asarray(out.points),
                         np.asarray(hemi.points)[rim]), (
        "fix_boundary=True let a rim vertex move")


def test_fix_boundary_false_lets_the_rim_move():
    mesh = make_tagged_sphere()
    hemi = _poly(mesh.points, _tri(mesh)[np.asarray(
        mesh.cell_data["elemTag"]) == 1])
    rim = _boundary_vertices(hemi)
    out = remesh(hemi, target_edge=0.5 * _mean_edge_length(hemi),
                 fix_boundary=False, preserve_labels=())
    assert not _contains_all(np.asarray(out.points),
                             np.asarray(hemi.points)[rim])


# ---------------------------------------------------------------------
# apply orchestration
# ---------------------------------------------------------------------
def test_defaults_are_the_documented_ones():
    """Defaults are tuned for a clipped atrial wall in mm."""
    opts = PostprocessOptions()
    assert opts.refine_mode == REFINE_RESAMPLE
    assert opts.refine_edge_len == 0.3
    assert opts.remesh_surf_corr == 0.95
    assert opts.remesh_fix_boundary is True
    assert opts.remesh_min_edge == 0.0 and opts.remesh_max_edge == 0.0


def test_apply_adaptive_mode_only_splits():
    mesh = make_sphere()
    opts = PostprocessOptions(do_refine=True,
                              refine_mode=REFINE_ADAPTIVE,
                              refine_edge_len=0.5 * _mean_edge_length(mesh))
    out = apply(mesh, opts)
    assert _contains_all(np.asarray(out.points), np.asarray(mesh.points))


def test_apply_resample_mode_calls_remesh():
    mesh = make_sphere()
    target = 3.0 * _mean_edge_length(mesh)
    opts = PostprocessOptions(do_refine=True,
                              refine_mode=REFINE_RESAMPLE,
                              refine_edge_len=target)
    out = apply(mesh, opts)
    assert out.n_points < mesh.n_points        # adaptive could never do this


def test_apply_resample_band_requires_zero_edge_len():
    mesh = make_sphere()
    opts = PostprocessOptions(do_refine=True,
                              refine_mode=REFINE_RESAMPLE,
                              refine_edge_len=0.4,
                              remesh_min_edge=0.2,
                              remesh_max_edge=0.8)
    with pytest.raises(ValueError, match="refine_edge_len=0"):
        apply(mesh, opts)


def test_apply_resample_accepts_an_explicit_band():
    mesh = make_sphere()
    opts = PostprocessOptions(do_refine=True,
                              refine_mode=REFINE_RESAMPLE,
                              refine_edge_len=0.0,
                              remesh_min_edge=0.4,
                              remesh_max_edge=1.2)
    out = apply(mesh, opts)
    assert _edge_lengths(out).max() <= 1.2 * 1.05


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
