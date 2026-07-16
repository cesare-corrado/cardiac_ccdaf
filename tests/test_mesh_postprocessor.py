"""
test_mesh_postprocessor.py
==========================
Tests for the public mesh_postprocessor contracts:

* ``clean`` leaks no bookkeeping arrays (RegionId, vtkOriginal*Ids) but
  keeps arrays the input already carried;
* ``clean``'s ``merge_tol`` defaults to coincident-only welding and is an
  absolute distance;
* ``fill_holes`` closes small openings, leaves large (anatomical) ones
  open, and preserves point fields and ``elemTag``.

Uses synthetic sphere meshes (no real EAM data required).

Run with pytest:
    pytest tests/test_mesh_postprocessor.py

Run as a standalone script:
    python tests/test_mesh_postprocessor.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pyvista as pv

from ccdaf.core.mesh_postprocessor import (
    PostprocessOptions,
    apply,
    clean,
    fill_holes,
)


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def _tri(mesh):
    return np.asarray(mesh.faces).reshape(-1, 4)[:, 1:].astype(np.int64)


def _poly(points, tri):
    faces = np.hstack([np.full((tri.shape[0], 1), 3, dtype=np.int64), tri])
    return pv.PolyData(np.asarray(points), faces.ravel())


def _n_boundary_edges(mesh):
    tri = _tri(mesh)
    e = np.sort(np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    return int((counts == 1).sum())


def _n_nonmanifold_edges(mesh):
    tri = _tri(mesh)
    e = np.sort(np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    return int((counts >= 3).sum())


def make_sphere(theta=20, phi=20):
    """Closed triangulated sphere carrying one point field and elemTag,
    and no other arrays."""
    m = pv.Sphere(radius=10.0, theta_resolution=theta, phi_resolution=phi)
    m = _poly(m.points, _tri(m.triangulate()))
    m.point_data["Bipolar"] = np.linalg.norm(np.asarray(m.points), axis=1)
    m.cell_data["elemTag"] = np.ones(m.n_cells, dtype=np.int32)
    return m


def punch_hole(mesh, n_faces, near=(0.0, 0.0, 10.0)):
    """Drop the ``n_faces`` faces closest to ``near``, opening a hole.
    Rebuilds the PolyData by hand so no VTK bookkeeping arrays appear."""
    tri = _tri(mesh)
    centres = np.asarray(mesh.cell_centers().points)
    order = np.argsort(np.linalg.norm(centres - np.asarray(near, dtype=float), axis=1))
    drop = set(int(i) for i in order[:n_faces])
    keep = np.array([i for i in range(tri.shape[0]) if i not in drop], dtype=np.int64)
    out = _poly(mesh.points, tri[keep])
    for name in mesh.point_data:
        out.point_data[name] = np.asarray(mesh.point_data[name])
    for name in mesh.cell_data:
        out.cell_data[name] = np.asarray(mesh.cell_data[name])[keep]
    return out


def make_split_vertex_pyramid(gap):
    """Closed-ish square pyramid whose apex is split into two vertices
    ``gap`` apart. The mesh stays a single connected component (through
    the base), so only the weld tolerance decides whether the two apex
    copies collapse into one."""
    pts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 1.0],          # apex A
        [0.5, 0.5, 1.0 + gap],    # apex A' — gap away from A
    ])
    tri = np.array([
        [0, 1, 4], [1, 2, 4],     # sides on A
        [2, 3, 5], [3, 0, 5],     # sides on A'
        [0, 2, 1], [0, 3, 2],     # base
    ], dtype=np.int64)
    m = _poly(pts, tri)
    m.cell_data["elemTag"] = np.ones(m.n_cells, dtype=np.int32)
    return m


# ---------------------------------------------------------------------
# clean — array leaks
# ---------------------------------------------------------------------
def test_clean_leaks_no_bookkeeping_arrays():
    mesh = make_sphere()
    out = clean(mesh)
    assert set(out.point_data.keys()) == set(mesh.point_data.keys()), (
        f"clean added point arrays: "
        f"{set(out.point_data.keys()) - set(mesh.point_data.keys())}"
    )
    assert set(out.cell_data.keys()) == set(mesh.cell_data.keys()), (
        f"clean added cell arrays: "
        f"{set(out.cell_data.keys()) - set(mesh.cell_data.keys())}"
    )


def test_clean_leaks_no_region_id():
    """RegionId comes from connectivity() inside _keep_main_components and
    must not reach the caller — the EAM field drop-down is built from
    point_data.keys()."""
    mesh = make_sphere()
    out = clean(mesh)
    assert "RegionId" not in out.point_data
    assert "RegionId" not in out.cell_data
    assert "vtkOriginalPointIds" not in out.point_data
    assert "vtkOriginalCellIds" not in out.cell_data


def test_clean_preserves_input_region_id():
    """An input that already carries RegionId keeps it, with the input's
    values — only arrays clean() itself introduced get stripped."""
    mesh = make_sphere()
    mesh.point_data["RegionId"] = np.full(mesh.n_points, 7, dtype=np.int32)
    mesh.cell_data["RegionId"] = np.full(mesh.n_cells, 9, dtype=np.int32)
    out = clean(mesh)
    assert "RegionId" in out.point_data, "input point RegionId was dropped"
    assert "RegionId" in out.cell_data, "input cell RegionId was dropped"
    assert np.all(np.asarray(out.point_data["RegionId"]) == 7), (
        "input point RegionId values were overwritten by connectivity()"
    )
    assert np.all(np.asarray(out.cell_data["RegionId"]) == 9), (
        "input cell RegionId values were overwritten by connectivity()"
    )


def test_clean_preserves_elem_tag():
    mesh = make_sphere()
    out = clean(mesh)
    assert "elemTag" in out.cell_data
    assert np.all(np.asarray(out.cell_data["elemTag"]) == 1)


def _labelled_sphere():
    """Sphere with elemTag=2 on the cap above z=5, elemTag=1 elsewhere."""
    mesh = make_sphere()
    tags = np.ones(mesh.n_cells, dtype=np.int32)
    tags[np.asarray(mesh.cell_centers().points)[:, 2] > 5.0] = 2
    mesh.cell_data["elemTag"] = tags
    return mesh, int((tags == 2).sum())


def test_clean_preserve_labels_keeps_protected_cells():
    """Regression guard on the preserve_labels contract: protected cells are
    never dropped, and the array strip must not take elemTag with it."""
    mesh, n_protected = _labelled_sphere()
    out = clean(mesh, preserve_labels=(2,))
    assert "elemTag" in out.cell_data
    kept = int((np.asarray(out.cell_data["elemTag"]) == 2).sum())
    assert kept >= n_protected, f"protected cells lost: {n_protected} -> {kept}"


def test_clean_preserve_labels_does_not_move_protected_vertices():
    """At the default merge_tol=0 protected vertices keep their exact
    coordinates."""
    from scipy.spatial import cKDTree

    mesh, _ = _labelled_sphere()
    tags = np.asarray(mesh.cell_data["elemTag"])
    protected_pts = np.asarray(mesh.points)[np.unique(_tri(mesh)[tags == 2].ravel())]
    out = clean(mesh, preserve_labels=(2,))
    dist, _ = cKDTree(np.asarray(out.points)).query(protected_pts, k=1)
    assert dist.max() < 1e-9, (
        f"protected vertices moved by up to {dist.max():.3e}"
    )


# ---------------------------------------------------------------------
# clean — merge_tol
# ---------------------------------------------------------------------
def test_clean_merge_tol_default_does_not_weld_near_duplicates():
    """The default must stay 0.0: only exactly coincident points merge, so
    existing behaviour and the preserve_labels contract are unchanged."""
    mesh = make_split_vertex_pyramid(gap=0.05)
    out = clean(mesh, smooth_iterations=0)
    assert out.n_points == mesh.n_points, (
        f"default clean() welded near-duplicates: "
        f"{mesh.n_points} -> {out.n_points}"
    )


def test_clean_merge_tol_welds_near_duplicates_when_asked():
    mesh = make_split_vertex_pyramid(gap=0.05)
    out = clean(mesh, smooth_iterations=0, merge_tol=0.1)
    assert out.n_points == mesh.n_points - 1, (
        f"merge_tol=0.1 should weld the 0.05-apart apex pair: "
        f"{mesh.n_points} -> {out.n_points}"
    )


def test_clean_merge_tol_is_absolute():
    """pyvista's clean() is absolute by default, so merge_tol is a distance
    in mesh units — 0.1 must not weld a pair 0.5 apart, whatever the mesh
    scale."""
    mesh = make_split_vertex_pyramid(gap=0.5)
    out = clean(mesh, smooth_iterations=0, merge_tol=0.1)
    assert out.n_points == mesh.n_points, (
        "merge_tol=0.1 welded a pair 0.5 apart — tolerance is not absolute"
    )


def test_clean_merge_tol_zero_matches_omitted():
    mesh = make_split_vertex_pyramid(gap=0.05)
    a = clean(mesh, smooth_iterations=0)
    b = clean(mesh, smooth_iterations=0, merge_tol=0.0)
    assert a.n_points == b.n_points
    assert a.n_cells == b.n_cells


# ---------------------------------------------------------------------
# fill_holes
# ---------------------------------------------------------------------
def test_fill_holes_closes_small_hole():
    mesh = punch_hole(make_sphere(), n_faces=1)
    assert _n_boundary_edges(mesh) > 0, "test setup: no hole was punched"
    out = fill_holes(mesh, max_size=5.0)
    assert _n_boundary_edges(out) == 0, "small hole was not closed"
    assert _n_nonmanifold_edges(out) == 0, "filling introduced non-manifold edges"


def test_fill_holes_preserves_elem_tag():
    """_fill_small_holes drops cell data; the public wrapper must
    re-transfer it."""
    mesh = punch_hole(make_sphere(), n_faces=1)
    out = fill_holes(mesh, max_size=5.0)
    assert "elemTag" in out.cell_data, "elemTag lost by fill_holes"
    assert len(out.cell_data["elemTag"]) == out.n_cells
    assert np.all(np.asarray(out.cell_data["elemTag"]) == 1)


def test_fill_holes_preserves_elem_tag_values_and_dtype():
    mesh = punch_hole(make_sphere(), n_faces=1)
    mesh.cell_data["elemTag"] = np.full(mesh.n_cells, 3, dtype=np.int32)
    out = fill_holes(mesh, max_size=5.0)
    tags = np.asarray(out.cell_data["elemTag"])
    assert tags.dtype == np.int32, f"elemTag dtype changed to {tags.dtype}"
    assert np.all(tags == 3)


def test_fill_holes_preserves_point_fields():
    mesh = punch_hole(make_sphere(), n_faces=1)
    before = np.asarray(mesh.point_data["Bipolar"], dtype=float)
    out = fill_holes(mesh, max_size=5.0)
    assert "Bipolar" in out.point_data
    after = np.asarray(out.point_data["Bipolar"], dtype=float)
    assert len(after) == out.n_points
    # Values are resampled onto the repaired topology, so the range must
    # stay inside the input's range rather than match element-wise.
    assert after.min() >= before.min() - 1e-9
    assert after.max() <= before.max() + 1e-9


def test_fill_holes_leaves_large_opening_open():
    """Size-based semantics: an opening larger than max_size stays open, so
    anatomical openings (PV ostia, mitral valve) keep their identity.

    max_size=2.0 sits between the one-triangle hole (loop radius 1.09) that
    test_fill_holes_closes_small_hole closes and this 60-triangle opening
    (loop radius 3.25), so the threshold is doing the discriminating rather
    than being too small to close anything."""
    mesh = punch_hole(make_sphere(), n_faces=60)
    assert _n_boundary_edges(mesh) > 0, "test setup: no opening was punched"
    out = fill_holes(mesh, max_size=2.0)
    assert _n_boundary_edges(out) > 0, (
        "a large opening was closed despite max_size=2.0"
    )
    # Same threshold, smaller hole -> closed. Proves 2.0 is not simply
    # below everything.
    small = punch_hole(make_sphere(), n_faces=1)
    assert _n_boundary_edges(fill_holes(small, max_size=2.0)) == 0


def test_fill_holes_large_max_size_closes_everything():
    mesh = punch_hole(make_sphere(), n_faces=60)
    out = fill_holes(mesh, max_size=1e6)
    assert _n_boundary_edges(out) == 0, (
        "a very large max_size should give close-everything behaviour"
    )


def test_fill_holes_leaks_no_arrays():
    mesh = punch_hole(make_sphere(), n_faces=1)
    out = fill_holes(mesh, max_size=5.0)
    assert set(out.point_data.keys()) == set(mesh.point_data.keys())
    assert set(out.cell_data.keys()) == set(mesh.cell_data.keys())


def test_fill_holes_rejects_non_positive_max_size():
    mesh = punch_hole(make_sphere(), n_faces=1)
    for bad in (0.0, -1.0):
        try:
            fill_holes(mesh, max_size=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"max_size={bad} must raise ValueError")


def test_fill_holes_does_not_mutate_input():
    mesh = punch_hole(make_sphere(), n_faces=1)
    n_pts, n_cells = mesh.n_points, mesh.n_cells
    bnd = _n_boundary_edges(mesh)
    fill_holes(mesh, max_size=5.0)
    assert (mesh.n_points, mesh.n_cells) == (n_pts, n_cells)
    assert _n_boundary_edges(mesh) == bnd


# ---------------------------------------------------------------------
# apply orchestration
# ---------------------------------------------------------------------
def test_apply_fill_holes_off_by_default():
    mesh = punch_hole(make_sphere(), n_faces=1)
    opts = PostprocessOptions()
    assert opts.do_fill_holes is False
    assert opts.clean_merge_tol == 0.0
    out = apply(mesh, opts)
    assert _n_boundary_edges(out) == _n_boundary_edges(mesh), (
        "apply() filled holes with every flag off"
    )


def test_apply_honours_do_fill_holes():
    mesh = punch_hole(make_sphere(), n_faces=1)
    opts = PostprocessOptions(do_fill_holes=True, max_hole_size=5.0)
    out = apply(mesh, opts)
    assert _n_boundary_edges(out) == 0
    assert "elemTag" in out.cell_data


def test_apply_clean_then_fill_leaves_no_holes():
    """Hole filling runs after clean, so holes that clean itself opens (by
    dropping non-manifold / degenerate cells) still get closed."""
    mesh = punch_hole(make_sphere(), n_faces=1)
    opts = PostprocessOptions(do_clean=True, do_fill_holes=True,
                              max_hole_size=5.0)
    out = apply(mesh, opts)
    assert _n_boundary_edges(out) == 0
    assert _n_nonmanifold_edges(out) == 0
    assert "elemTag" in out.cell_data
    assert "RegionId" not in out.point_data


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
