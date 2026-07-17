"""
test_smoothing_and_electrodes.py
================================
Tests for global smoothing and for carrying EAM electrodes along with it:

* ``smooth`` keeps the vertex correspondence (count and order) and the
  arrays riding on it — that correspondence is what the electrode warp
  is built from, so it is a contract, not an implementation detail;
* Taubin holds the enclosed volume where Laplacian shrinks it;
* ``apply`` reports moved vertices only for smoothing, the one step that
  moves the surface rather than re-tessellating it;
* ``displace_electrodes`` keeps every electrode inside the surface's own
  displacement, is scale-invariant (so a mesh in cm behaves like one in
  mm), and honours explicit parameters instead of searching.

Uses synthetic sphere meshes (no real EAM data required).

Run with pytest:
    pytest tests/test_smoothing_and_electrodes.py

Run as a standalone script:
    python tests/test_smoothing_and_electrodes.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pytest
import pyvista as pv
import vtk

from ccdaf.core.eam_loader import (
    displace_electrodes,
    displace_electrodes_by_distance,
    displace_electrodes_for,
)
from ccdaf.core.mesh_postprocessor import (
    PostprocessOptions,
    SMOOTH_LAPLACIAN,
    SMOOTH_TAUBIN,
    apply,
    smooth,
)


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def _tri(mesh):
    return np.asarray(mesh.faces).reshape(-1, 4)[:, 1:].astype(np.int64)


def _poly(points, tri):
    faces = np.hstack([np.full((tri.shape[0], 1), 3, dtype=np.int64), tri])
    return pv.PolyData(np.asarray(points), faces.ravel())


def make_bumpy_sphere(theta=24, phi=24, noise=0.15, seed=0):
    """Sphere with radial noise, standing in for an acquisition-noisy
    surface: something smoothing has actual work to do on."""
    m = pv.Sphere(radius=10.0, theta_resolution=theta, phi_resolution=phi)
    m = _poly(m.points, _tri(m.triangulate()))
    pts = np.asarray(m.points)
    rng = np.random.default_rng(seed)
    radial = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    m.points = pts + radial * rng.normal(scale=noise, size=(len(pts), 1))
    m.point_data["Bipolar"] = np.linalg.norm(np.asarray(m.points), axis=1)
    m.cell_data["elemTag"] = np.ones(m.n_cells, dtype=np.int32)
    return m


def electrodes_near(mesh, n=60, offset=0.4, seed=1):
    """Points floating a little off the surface, as Carto electrodes are."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(mesh.n_points, n, replace=False)
    pts = np.asarray(mesh.points)[idx]
    radial = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    return pts + radial * offset


# ---------------------------------------------------------------------
# smooth
# ---------------------------------------------------------------------
@pytest.mark.parametrize("method", [SMOOTH_TAUBIN, SMOOTH_LAPLACIAN])
def test_smooth_preserves_vertex_correspondence(method):
    mesh = make_bumpy_sphere()
    out = smooth(mesh, method=method, iterations=10)
    assert out.n_points == mesh.n_points
    assert out.n_cells == mesh.n_cells
    np.testing.assert_array_equal(_tri(out), _tri(mesh))


@pytest.mark.parametrize("method", [SMOOTH_TAUBIN, SMOOTH_LAPLACIAN])
def test_smooth_keeps_arrays(method):
    mesh = make_bumpy_sphere()
    out = smooth(mesh, method=method, iterations=10)
    assert "Bipolar" in out.point_data
    assert "elemTag" in out.cell_data
    np.testing.assert_array_equal(out.cell_data["elemTag"],
                                  mesh.cell_data["elemTag"])
    np.testing.assert_allclose(out.point_data["Bipolar"],
                               mesh.point_data["Bipolar"])


@pytest.mark.parametrize("method", [SMOOTH_TAUBIN, SMOOTH_LAPLACIAN])
def test_smooth_actually_moves_the_surface(method):
    mesh = make_bumpy_sphere()
    out = smooth(mesh, method=method, iterations=10)
    assert np.linalg.norm(out.points - mesh.points, axis=1).max() > 0.0


def test_smooth_does_not_mutate_input():
    mesh = make_bumpy_sphere()
    before = np.asarray(mesh.points).copy()
    smooth(mesh, method=SMOOTH_TAUBIN, iterations=10)
    np.testing.assert_array_equal(mesh.points, before)


def test_taubin_preserves_volume_better_than_laplacian_on_a_dense_mesh():
    """The reason Taubin is offered at all: Laplacian deflates the shell as
    iterations grow, which matters when it is later measured.

    The advantage is *not* unconditional, hence the dense mesh here. The
    windowed-sinc filter is only stable for modest iteration counts relative
    to mesh density: 40 iterations holds volume to +0.1% on a 13.5k-point
    Carto surface, loses ~4% on the same surface decimated to 4k, and runs
    away entirely on a very coarse one. Dense is the case we ship for.
    """
    mesh = make_bumpy_sphere(theta=60, phi=60, noise=0.05)
    v0 = mesh.volume
    taubin = abs(smooth(mesh, method=SMOOTH_TAUBIN, iterations=40).volume / v0 - 1)
    lap = abs(smooth(mesh, method=SMOOTH_LAPLACIAN, iterations=40).volume / v0 - 1)
    assert taubin < lap


def test_smooth_rejects_unknown_method():
    with pytest.raises(ValueError):
        smooth(make_bumpy_sphere(), method="nope")


# ---------------------------------------------------------------------
# apply -> on_surface_moved
# ---------------------------------------------------------------------
def test_apply_reports_the_surface_either_side_of_smoothing():
    mesh = make_bumpy_sphere()
    seen = []
    apply(mesh, PostprocessOptions(do_smooth=True, smooth_iterations=10),
          on_surface_moved=lambda a, b: seen.append((a, b)))
    assert len(seen) == 1
    old, new = seen[0]
    assert isinstance(old, pv.PolyData) and isinstance(new, pv.PolyData)
    assert old.n_points == new.n_points == mesh.n_points
    assert not np.allclose(np.asarray(old.points), np.asarray(new.points))


def test_apply_reports_the_pre_smoothing_surface_not_the_original():
    """`old` must be what went into smoothing, i.e. after any earlier steps
    — otherwise the electrodes are moved from a wall that no longer exists.
    """
    mesh = make_bumpy_sphere()
    seen = []
    out = apply(mesh, PostprocessOptions(do_clean=True, do_smooth=True,
                                         smooth_iterations=10),
                on_surface_moved=lambda a, b: seen.append((a, b)))
    old, new = seen[0]
    assert old.n_points == new.n_points == out.n_points


def test_apply_does_not_report_without_smoothing():
    """Only smoothing moves the surface; the other steps re-tessellate it
    rather than move it, so they must not report."""
    mesh = make_bumpy_sphere()
    seen = []
    apply(mesh, PostprocessOptions(do_clean=True),
          on_surface_moved=lambda a, b: seen.append((a, b)))
    assert seen == []


# ---------------------------------------------------------------------
# displace_electrodes
# ---------------------------------------------------------------------
def test_electrodes_stay_within_the_surface_displacement():
    """The load-bearing guarantee: the warp is driven by the surface, so
    nothing embedded in it may move further than the surface did."""
    mesh = make_bumpy_sphere()
    el = electrodes_near(mesh)
    old = np.asarray(mesh.points).copy()
    new = np.asarray(smooth(mesh, method=SMOOTH_TAUBIN, iterations=20).points)
    moved = displace_electrodes(el, old, new, n_centres=400)
    field_max = np.linalg.norm(new - old, axis=1).max()
    assert np.linalg.norm(moved - el, axis=1).max() <= field_max + 1e-9


def test_electrodes_follow_the_surface():
    """They must actually move — and roughly like the wall they sit on."""
    mesh = make_bumpy_sphere()
    el = electrodes_near(mesh)
    old = np.asarray(mesh.points).copy()
    new = np.asarray(smooth(mesh, method=SMOOTH_TAUBIN, iterations=20).points)
    moved = displace_electrodes(el, old, new, n_centres=400)
    step = np.linalg.norm(moved - el, axis=1)
    assert step.max() > 0.0
    assert step.mean() <= np.linalg.norm(new - old, axis=1).mean() * 3.0


def test_displacement_is_scale_invariant():
    """The search space is derived from the mesh, so the same anatomy in
    different units must give the same warp."""
    mesh = make_bumpy_sphere()
    el = electrodes_near(mesh)
    old = np.asarray(mesh.points).copy()
    new = np.asarray(smooth(mesh, method=SMOOTH_TAUBIN, iterations=20).points)
    mm = displace_electrodes(el, old, new, n_centres=400)
    scale = 0.1
    cm = displace_electrodes(el * scale, old * scale, new * scale, n_centres=400)
    np.testing.assert_allclose(cm / scale, mm, atol=1e-6)


def test_no_electrodes_is_a_noop():
    mesh = make_bumpy_sphere()
    old = np.asarray(mesh.points).copy()
    new = np.asarray(smooth(mesh, method=SMOOTH_TAUBIN, iterations=10).points)
    out = displace_electrodes(np.empty((0, 3)), old, new, n_centres=200)
    assert out.shape == (0, 3)


def test_unmoved_surface_leaves_electrodes_alone():
    mesh = make_bumpy_sphere()
    el = electrodes_near(mesh)
    pts = np.asarray(mesh.points)
    np.testing.assert_array_equal(displace_electrodes(el, pts, pts.copy()), el)


def test_mismatched_point_arrays_raise():
    mesh = make_bumpy_sphere()
    el = electrodes_near(mesh)
    pts = np.asarray(mesh.points)
    with pytest.raises(ValueError):
        displace_electrodes(el, pts, pts[:-5])


def test_explicit_parameters_skip_the_search():
    """Given both parameters there is nothing to choose, so no status is
    reported — and the result must differ from another length scale, i.e.
    the arguments are really used."""
    mesh = make_bumpy_sphere()
    el = electrodes_near(mesh)
    old = np.asarray(mesh.points).copy()
    new = np.asarray(smooth(mesh, method=SMOOTH_TAUBIN, iterations=20).points)
    seen = []
    tight = displace_electrodes(el, old, new, n_centres=400, length_scale=1.0,
                                smoothing=0.0, on_status=seen.append)
    wide = displace_electrodes(el, old, new, n_centres=400, length_scale=8.0,
                               smoothing=0.0)
    assert seen == []
    assert not np.allclose(tight, wide)


def test_search_reports_what_it_chose():
    mesh = make_bumpy_sphere()
    el = electrodes_near(mesh)
    old = np.asarray(mesh.points).copy()
    new = np.asarray(smooth(mesh, method=SMOOTH_TAUBIN, iterations=20).points)
    seen = []
    displace_electrodes(el, old, new, n_centres=400, on_status=seen.append)
    assert len(seen) == 1
    assert "length scale" in seen[0]


# ---------------------------------------------------------------------
# displace_electrodes_by_distance — the correspondence-free path, used
# after a segmentation round trip where marching cubes returns an
# unrelated point set. Concentric spheres make the answer analytic: an
# electrode r away from a sphere of radius R must land r away from R'.
# ---------------------------------------------------------------------
def _sphere(radius, theta=40, phi=40):
    return pv.Sphere(radius=radius, theta_resolution=theta,
                     phi_resolution=phi).triangulate()


@pytest.mark.parametrize("offset", [2.0, -2.0, 6.0])
def test_distance_move_keeps_the_electrode_offset_from_the_wall(offset):
    """The invariant: an electrode `offset` from the wall stays `offset`
    from it, on the same side, when the wall moves outward by 1."""
    old, new = _sphere(20.0), _sphere(21.0)
    el = np.array([[20.0 + offset, 0.0, 0.0],
                   [0.0, 20.0 + offset, 0.0],
                   [0.0, 0.0, -(20.0 + offset)]])
    moved = displace_electrodes_by_distance(el, old, new)
    # Each must now sit at 21 + offset from the centre, in its own direction.
    assert np.allclose(np.linalg.norm(moved, axis=1), 21.0 + offset, atol=0.15)
    # and must not have slid appreciably off its own ray. The residual drift
    # is faceting, not tangential motion: the field's gradient is normal to
    # the triangles, not to the ideal sphere they approximate, and the drift
    # halves whenever the tessellation doubles.
    for a, b in zip(el, moved):
        assert np.allclose(a / np.linalg.norm(a), b / np.linalg.norm(b), atol=0.01)


def _flip_winding(mesh):
    """A surface whose triangles wind the other way — what marching cubes
    returns here, since _segmentation_to_polydata calls FlipNormalsOn."""
    n = vtk.vtkPolyDataNormals()
    n.SetInputData(mesh)
    n.ComputePointNormalsOn()
    n.ComputeCellNormalsOff()
    n.AutoOrientNormalsOn()
    n.FlipNormalsOn()
    n.Update()
    return pv.wrap(n.GetOutput())


def test_distance_move_survives_a_surface_wound_the_other_way():
    """The sign of a distance field comes from the winding, and the two
    surfaces need not agree on it. Unhandled, every electrode is driven
    through the wall to the far side instead of moved a fraction of a mm.
    """
    old = _sphere(20.0)
    new = _flip_winding(_sphere(21.0))
    el = np.array([[22.0, 0.0, 0.0], [0.0, 0.0, 18.0]])
    moved = displace_electrodes_by_distance(el, old, new)
    # 2 outside stays 2 outside; 2 inside stays 2 inside.
    assert np.linalg.norm(moved[0]) == pytest.approx(23.0, abs=0.2)
    assert np.linalg.norm(moved[1]) == pytest.approx(19.0, abs=0.2)


def test_distance_move_needs_no_correspondence():
    """The whole point: the two surfaces have unrelated point sets, which
    displace_electrodes would reject outright."""
    old, new = _sphere(20.0, theta=40, phi=40), _sphere(21.0, theta=25, phi=25)
    assert old.n_points != new.n_points
    el = np.array([[22.0, 0.0, 0.0]])
    moved = displace_electrodes_by_distance(el, old, new)
    assert np.linalg.norm(moved[0]) == pytest.approx(23.0, abs=0.2)


def test_distance_move_is_bounded_by_what_the_wall_did():
    """The guarantee that replaces the RBF's rejection constraint: the step
    IS the local change in wall distance, so nothing can move further than
    the surfaces differ — however far out it sits. Note this is a bound, not
    a decay: a wall that moved everywhere moves every electrode with it.
    """
    old, new = _sphere(20.0), _sphere(21.0)   # differ by 1 everywhere
    el = np.array([[22.0, 0.0, 0.0], [0.0, 30.0, 0.0],
                   [0.0, 0.0, 18.0], [300.0, 0.0, 0.0]])
    steps = np.linalg.norm(
        displace_electrodes_by_distance(el, old, new) - el, axis=1)
    assert steps.max() <= 1.0 + 0.15
    # even the one 280 away tracks it — the wall moved there too
    assert steps[-1] == pytest.approx(1.0, abs=0.15)


def test_distance_move_ignores_a_change_the_electrode_is_not_nearest_to():
    """Locality is by closest point, not by distance from the anatomy: bump
    one cap and an electrode over the far side must not notice."""
    old = _sphere(20.0)
    new = old.copy()
    pts = np.asarray(new.points).copy()
    pts[pts[:, 0] > 17.0] *= 1.10          # a bump on the +x cap only
    new.points = pts
    over_bump = np.array([[22.0, 0.0, 0.0]])
    far_side = np.array([[-22.0, 0.0, 0.0]])
    assert np.linalg.norm(
        displace_electrodes_by_distance(over_bump, old, new) - over_bump) > 1.0
    assert np.linalg.norm(
        displace_electrodes_by_distance(far_side, old, new) - far_side) < 0.05


def test_distance_move_leaves_electrodes_alone_when_nothing_changed():
    old = _sphere(20.0)
    el = np.array([[22.0, 0.0, 0.0], [0.0, 18.0, 0.0]])
    moved = displace_electrodes_by_distance(el, old, old.copy())
    assert np.allclose(moved, el, atol=1e-6)


def test_distance_move_does_not_touch_either_mesh():
    """The constraint on this path: it may move electrodes and nothing else.
    The rebuilt surface is the segmentation's output, not ours to adjust."""
    old, new = _sphere(20.0), _sphere(21.0)
    old_pts = np.array(old.points, copy=True)
    new_pts = np.array(new.points, copy=True)
    displace_electrodes_by_distance(np.array([[22.0, 0.0, 0.0]]), old, new)
    assert np.array_equal(old_pts, np.asarray(old.points))
    assert np.array_equal(new_pts, np.asarray(new.points))


def test_distance_move_with_no_electrodes_is_a_noop():
    old, new = _sphere(20.0), _sphere(21.0)
    assert displace_electrodes_by_distance(np.empty((0, 3)), old, new).shape == (0, 3)


def test_distance_move_reports_what_it_did():
    old, new = _sphere(20.0), _sphere(21.0)
    seen = []
    displace_electrodes_by_distance(np.array([[22.0, 0.0, 0.0]]), old, new,
                                    on_status=seen.append)
    assert len(seen) == 1
    assert "distance to the wall" in seen[0]


# ---------------------------------------------------------------------
# displace_electrodes_for — the dispatcher the app uses
# ---------------------------------------------------------------------
def test_dispatcher_uses_the_distance_field():
    old, new = _sphere(20.0), _sphere(21.0)
    el = np.array([[22.0, 0.0, 0.0], [0.0, 0.0, 18.0]])
    np.testing.assert_allclose(
        displace_electrodes_for(el, old, new),
        displace_electrodes_by_distance(el, old, new))


def test_dispatcher_falls_back_to_the_rbf_when_the_distance_field_faults(monkeypatch):
    """Robustness path: the RBF needs the 1:1 correspondence, so it can only
    stand in when the point count survived."""
    mesh = make_bumpy_sphere()
    el = electrodes_near(mesh)
    new = smooth(mesh, method=SMOOTH_TAUBIN, iterations=10)

    def boom(*a, **k):
        raise RuntimeError("distance field unusable")

    monkeypatch.setattr("ccdaf.core.eam_loader.displace_electrodes_by_distance", boom)
    seen = []
    out = displace_electrodes_for(el, mesh, new, on_status=seen.append)
    assert out.shape == el.shape
    assert not np.allclose(out, el)                 # the RBF did the work
    assert any("fell back" in s for s in seen)      # and said so


def test_dispatcher_reraises_when_no_fallback_is_possible(monkeypatch):
    """No correspondence means no RBF; failing loudly beats leaving the
    electrodes silently on the old wall."""
    old = _sphere(20.0)
    new = _sphere(21.0, theta=25, phi=25)
    assert old.n_points != new.n_points

    def boom(*a, **k):
        raise RuntimeError("distance field unusable")

    monkeypatch.setattr("ccdaf.core.eam_loader.displace_electrodes_by_distance", boom)
    with pytest.raises(RuntimeError):
        displace_electrodes_for(np.array([[22.0, 0.0, 0.0]]), old, new)


def test_dispatcher_with_no_electrodes_is_a_noop():
    old, new = _sphere(20.0), _sphere(21.0)
    assert displace_electrodes_for(np.empty((0, 3)), old, new).shape == (0, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
