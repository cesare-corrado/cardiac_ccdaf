"""
test_field_transfer.py
======================
Tests for carrying fields across a remesh with no vertex correspondence —
what a segmentation round trip produces.

The contract:

* point fields (measurements) are interpolated within the triangle the
  closest surface point lands in, and never leave the source's range;
* cell fields (labels) are copied from the nearest cell, keeping their dtype
  and inventing no value that was not already a label;
* no-data does not spread through interpolation, and does not shrink either;
* surface further from the source than ``max_distance`` — the wall a
  segmentation edit invented — is no-data rather than plausible fiction;
* the source is never modified.

Uses synthetic meshes (no real EAM data required).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pytest
import pyvista as pv

from ccdaf.core.eam_export import polydata_to_carto_dict
from ccdaf.core.field_transfer import transfer_fields


def _sphere(radius=10.0, theta=30, phi=30):
    return pv.Sphere(radius=radius, theta_resolution=theta,
                     phi_resolution=phi).triangulate()


def _source():
    """A sphere carrying a field that is linear in space, so interpolation
    has an exactly checkable answer."""
    m = _sphere()
    pts = np.asarray(m.points)
    m.point_data["linear"] = 2.0 * pts[:, 0] + 3.0
    m.cell_data["elemTag"] = np.ones(m.n_cells, dtype=np.int32)
    return m


def _bare(mesh):
    out = mesh.copy(deep=True)
    for k in list(out.point_data.keys()):
        out.point_data.remove(k)
    for k in list(out.cell_data.keys()):
        out.cell_data.remove(k)
    return out


# ---------------------------------------------------------------------
# point fields — interpolation
# ---------------------------------------------------------------------
def test_point_field_is_interpolated_not_copied():
    """A denser target must resolve a linear field better than a
    nearest-vertex copy could: on the source's own vertices it is exact."""
    src = _source()
    dst = _bare(src.subdivide(1))
    transfer_fields(src, dst)
    expected = 2.0 * np.asarray(dst.points)[:, 0] + 3.0
    # The target's new vertices sit slightly off the source surface (a
    # subdivided sphere bulges), so allow the sagitta, not machine epsilon.
    assert np.abs(np.asarray(dst.point_data["linear"]) - expected).max() < 0.2


def test_transferred_values_never_leave_the_source_range():
    """Interpolation is convex, so it cannot invent an out-of-range value —
    the property that makes it safe for a measurement."""
    src = _source()
    dst = _bare(src.subdivide(1))
    transfer_fields(src, dst)
    a = np.asarray(src.point_data["linear"])
    b = np.asarray(dst.point_data["linear"])
    assert np.nanmin(b) >= np.nanmin(a) - 1e-9
    assert np.nanmax(b) <= np.nanmax(a) + 1e-9


# ---------------------------------------------------------------------
# no-data
# ---------------------------------------------------------------------
def test_an_all_nodata_field_stays_all_nodata():
    src = _source()
    src.point_data["dead"] = np.full(src.n_points, np.nan)
    dst = _bare(src.subdivide(1))
    transfer_fields(src, dst)
    assert np.all(np.isnan(np.asarray(dst.point_data["dead"])))


def test_nodata_does_not_spread_across_whole_triangles():
    """One invalid vertex must not blank every triangle touching it — the
    weights renormalise over the valid ones instead."""
    src = _source()
    vals = np.asarray(src.point_data["linear"]).copy()
    vals[0] = np.nan                       # a single invalid vertex
    src.point_data["linear"] = vals
    dst = _bare(src.subdivide(1))
    transfer_fields(src, dst)
    out = np.asarray(dst.point_data["linear"])
    # Plain interpolation would blank every triangle around vertex 0; with
    # renormalisation almost nothing is lost.
    assert np.isnan(out).sum() < 0.01 * len(out)


# ---------------------------------------------------------------------
# the guard — surface an edit invented
# ---------------------------------------------------------------------
def test_surface_beyond_the_guard_is_nodata_not_invented():
    src = _source()
    dst = _bare(src)
    pts = np.asarray(dst.points).copy()
    far = pts[:, 0] > 9.0                  # push a cap 5 units off the wall
    pts[far] += np.array([5.0, 0.0, 0.0])
    dst.points = pts
    transfer_fields(src, dst, max_distance=2.0)
    out = np.asarray(dst.point_data["linear"])
    assert np.all(np.isnan(out[far]))      # invented wall says so
    assert not np.any(np.isnan(out[~far]))  # the rest is untouched


def test_without_a_guard_invented_surface_inherits_values():
    """The opposite choice, kept honest: max_distance=None means new wall
    silently takes whatever it is nearest to."""
    src = _source()
    dst = _bare(src)
    pts = np.asarray(dst.points).copy()
    far = pts[:, 0] > 9.0
    pts[far] += np.array([5.0, 0.0, 0.0])
    dst.points = pts
    transfer_fields(src, dst, max_distance=None)
    assert not np.any(np.isnan(np.asarray(dst.point_data["linear"])))


def test_a_faithful_remesh_never_trips_the_guard():
    src = _source()
    dst = _bare(src.subdivide(1))
    transfer_fields(src, dst, max_distance=2.0)
    assert not np.any(np.isnan(np.asarray(dst.point_data["linear"])))


# ---------------------------------------------------------------------
# cell fields — labels
# ---------------------------------------------------------------------
def test_cell_labels_are_copied_never_averaged():
    """elemTag is the case that exists: averaging 1 and 11 into 6 would
    invent a region that is not a region."""
    src = _source()
    tags = np.where(np.asarray(src.cell_centers().points)[:, 2] > 0, 11, 1)
    src.cell_data["elemTag"] = tags.astype(np.int32)
    dst = _bare(src.subdivide(1))
    transfer_fields(src, dst)
    out = np.asarray(dst.cell_data["elemTag"])
    assert set(out.tolist()) <= {1, 11}
    assert out.dtype == np.int32


def test_cell_labels_ignore_the_guard():
    """New wall is still part of the body, so its label is a fact, not a
    fabricated measurement."""
    src = _source()
    dst = _bare(src)
    pts = np.asarray(dst.points).copy()
    pts[pts[:, 0] > 9.0] += np.array([5.0, 0.0, 0.0])
    dst.points = pts
    transfer_fields(src, dst, max_distance=2.0)
    assert set(np.asarray(dst.cell_data["elemTag"]).tolist()) == {1}


# ---------------------------------------------------------------------
# housekeeping
# ---------------------------------------------------------------------
def test_source_is_not_modified():
    src = _source()
    pts = np.array(src.points, copy=True)
    vals = np.array(src.point_data["linear"], copy=True)
    keys = (sorted(src.point_data.keys()), sorted(src.cell_data.keys()))
    transfer_fields(src, _bare(src.subdivide(1)), max_distance=2.0)
    assert np.array_equal(pts, np.asarray(src.points))
    assert np.array_equal(vals, np.asarray(src.point_data["linear"]))
    assert keys == (sorted(src.point_data.keys()), sorted(src.cell_data.keys()))


def test_the_renderers_picking_array_is_not_carried_over():
    """`render_idx` is stamped onto whatever mesh is on screen, so a source
    that has been displayed carries it. It is not data, and must not
    reappear as a selectable field on the rebuilt surface."""
    src = _source()
    src.cell_data["render_idx"] = np.arange(src.n_cells)
    dst = _bare(src.subdivide(1))
    transfer_fields(src, dst)
    assert "render_idx" not in dst.cell_data
    assert "elemTag" in dst.cell_data


def test_multi_component_fields_survive():
    src = _source()
    src.point_data["vec"] = np.asarray(src.points) * 2.0
    dst = _bare(src.subdivide(1))
    transfer_fields(src, dst)
    assert np.asarray(dst.point_data["vec"]).shape == (dst.n_points, 3)


def test_a_carto_export_keeps_one_column_per_name():
    """One column per name is the format's premise. A 3-component field would
    contribute three columns under one name and shift every later field's
    ColorsID onto the wrong column — so Unipolar would read back as a normal's
    y-component. A segmentation round trip carries exactly such a field.
    """
    mesh = _source()
    mesh.point_data["Normals"] = np.asarray(mesh.points) * 1.0   # 3-component
    d = polydata_to_carto_dict(mesh)
    assert d["VertexColors"].shape[1] == len(d["ColorsNames"])
    assert len(d["ColorsIDs"]) == len(d["ColorsNames"])
    assert "Normals" not in list(d["ColorsNames"])
    assert "linear" in list(d["ColorsNames"])


def test_a_source_without_fields_is_a_noop():
    src = _bare(_source())
    dst = _bare(_source().subdivide(1))
    transfer_fields(src, dst)
    assert len(dst.point_data) == 0 and len(dst.cell_data) == 0


def test_reports_what_it_did():
    src = _source()
    seen = []
    transfer_fields(src, _bare(src.subdivide(1)), max_distance=2.0,
                    on_status=seen.append)
    assert len(seen) == 1 and "point" in seen[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
