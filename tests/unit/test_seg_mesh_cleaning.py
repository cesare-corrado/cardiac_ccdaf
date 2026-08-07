"""
test_seg_mesh_cleaning.py
=========================
What the segmentation -> mesh converter cleans up before handing the surface
over as the working mesh.

``vtkCleanPolyData`` always ran at the end of ``segmentation_to_polydata``,
but with the default tolerance of 0 it merges only *exactly* coincident
points. Two defects got past it:

* **Marching-cubes slivers.** Where the isosurface grazes a grid vertex, MC
  emits a triangle whose two points sit a rounding error apart — an area
  orders of magnitude below the median, close enough to zero in the float32
  the points are stored in that VTK's OpenGL mapper may skip the cell when it
  builds its picking id map. Every later cell id then shifts and picks land on
  the wrong triangle. Welding at a fraction of the voxel pitch collapses the
  pair and the sliver goes with it.
* **Stray shells.** MC returns every isosurface it finds in one polydata, so a
  speck of stray voxels arrives as a second closed surface beside the anatomy
  and is tagged, picked and exported as if it belonged.

The contract:

* welding is on by default, sized from the voxel pitch, and does not open or
  tear the surface — it stays closed and manifold;
* ``weld_fraction=0`` restores exact-coincidence merging, so the old behaviour
  is still reachable;
* ``drop_stray_shells`` keeps the largest component, drops the specks, and
  keeps anything at or above the size threshold rather than deleting data;
* it reports what it did in both cases — a single-component surface comes back
  untouched, with a report that says so.

Synthetic volumes, built to be small: no display, no example data.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pytest
import pyvista as pv
import SimpleITK as sitk

from ccdaf.core.segmentation import (
    WELD_FRACTION, drop_stray_shells, segmentation_to_polydata,
)

SMOOTH = {"filt_stdev": [1.0, 1.0, 1.0], "filt_rfact": [1.5, 1.5, 1.5]}


def _volume(blocks, size=(40, 40, 40), spacing=(1.0, 1.0, 1.0)):
    """A uint8 volume with a foreground box per ``(slice, slice, slice)``."""
    arr = np.zeros(size[::-1], dtype=np.uint8)
    for zs, ys, xs in blocks:
        arr[zs, ys, xs] = 1
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    return img


def _cube(size=(40, 40, 40), spacing=(1.0, 1.0, 1.0)):
    return _volume([(slice(8, 30), slice(8, 30), slice(8, 30))], size, spacing)


def _mesh(img, **kw) -> pv.PolyData:
    return pv.wrap(segmentation_to_polydata(img, flip=False, **SMOOTH, **kw))


def _areas(mesh: pv.PolyData) -> np.ndarray:
    pts = np.asarray(mesh.points, dtype=float)
    tris = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]
    a, b, c = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def _edge_counts(mesh: pv.PolyData) -> np.ndarray:
    tris = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]
    e = np.sort(np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [0, 2]]]),
                axis=1)
    return np.unique(e, axis=0, return_counts=True)[1]


def _components(mesh: pv.PolyData) -> list:
    ids = np.asarray(mesh.connectivity("all").cell_data["RegionId"])
    return sorted(np.bincount(ids).tolist(), reverse=True)


# ---------------------------------------------------------------------------
# Welding
# ---------------------------------------------------------------------------
def test_welding_is_on_by_default_and_can_be_switched_off():
    img = _cube()
    default = _mesh(img)
    exact = _mesh(img, weld_fraction=0.0)
    assert WELD_FRACTION > 0.0
    # Welding can only ever merge points, never add them.
    assert default.n_points <= exact.n_points
    assert np.array_equal(_mesh(img, weld_fraction=WELD_FRACTION).points,
                          default.points)


def test_welding_leaves_no_degenerate_triangle_behind():
    """A welded pair would leave a triangle with a repeated vertex id — the
    exact thing whose zero area breaks the picking id map."""
    tris = np.asarray(_mesh(_cube()).faces).reshape(-1, 4)[:, 1:]
    repeated = ((tris[:, 0] == tris[:, 1]) | (tris[:, 1] == tris[:, 2])
                | (tris[:, 0] == tris[:, 2]))
    assert not repeated.any()
    assert _areas(_mesh(_cube())).min() > 0.0


def test_welding_keeps_the_surface_closed_and_manifold():
    counts = _edge_counts(_mesh(_cube()))
    assert int(np.count_nonzero(counts == 1)) == 0     # no boundary edge
    assert int(np.count_nonzero(counts > 2)) == 0      # no non-manifold edge


def test_the_weld_distance_follows_the_voxel_pitch():
    """Sized from the pitch, not from the mesh: the same anatomy sampled
    finer must not be welded harder in world units."""
    fine = _cube(size=(80, 80, 80), spacing=(0.5, 0.5, 0.5))
    coarse = _cube(size=(40, 40, 40), spacing=(1.0, 1.0, 1.0))
    # Nothing to assert on counts across resolutions; assert the geometry is
    # untouched at the scale that matters — the surface stays where it was.
    for img in (fine, coarse):
        welded, exact = _mesh(img), _mesh(img, weld_fraction=0.0)
        pitch = float(min(img.GetSpacing()))
        moved = np.abs(np.asarray(welded.bounds) - np.asarray(exact.bounds))
        assert moved.max() <= WELD_FRACTION * pitch + 1e-6


# ---------------------------------------------------------------------------
# Stray shells
# ---------------------------------------------------------------------------
def _speck_volume():
    """The anatomy plus a speck too big for the Gaussian to dissolve.

    Sized so the speck is well under the 1% default threshold (~0.6% of the
    cells), which is what a real brush slip looks like next to an atrium.
    """
    return _volume([(slice(6, 56), slice(6, 56), slice(6, 56)),
                    (slice(62, 66), slice(62, 66), slice(62, 66))],
                   size=(70, 70, 70))


def test_a_speck_arrives_as_a_second_shell():
    """The defect itself: without the pass, both surfaces are in the mesh."""
    assert len(_components(_mesh(_speck_volume()))) == 2


def test_the_speck_is_dropped_and_reported():
    poly = segmentation_to_polydata(_speck_volume(), flip=False, **SMOOTH)
    before = pv.wrap(poly).n_cells
    out, shells = drop_stray_shells(poly)
    mesh = pv.wrap(out)

    assert len(_components(mesh)) == 1
    assert shells.n_components == 2
    assert len(shells.dropped) == 1
    assert shells.dropped_cells > 0
    assert mesh.n_cells == before - shells.dropped_cells
    assert mesh.n_cells == shells.kept[0]


def test_the_dropped_shell_takes_its_points_with_it():
    """Specified-region extraction keeps the whole point set; orphan points
    would survive as unused vertices."""
    out, _ = drop_stray_shells(
        segmentation_to_polydata(_speck_volume(), flip=False, **SMOOTH))
    mesh = pv.wrap(out)
    used = np.unique(np.asarray(mesh.faces).reshape(-1, 4)[:, 1:])
    assert used.size == mesh.n_points


def test_no_region_id_array_is_left_on_the_mesh():
    out, _ = drop_stray_shells(
        segmentation_to_polydata(_speck_volume(), flip=False, **SMOOTH))
    assert "RegionId" not in pv.wrap(out).point_data
    assert "RegionId" not in pv.wrap(out).cell_data


def test_a_single_surface_is_returned_untouched():
    poly = segmentation_to_polydata(_cube(), flip=False, **SMOOTH)
    out, shells = drop_stray_shells(poly)
    assert out is poly
    assert shells.n_components == 1
    assert shells.dropped == ()
    assert shells.kept == (poly.GetNumberOfCells(),)


def test_a_component_too_big_to_be_a_speck_is_kept():
    """Deleting a second structure that large would be deleting data — it is
    kept and reported instead, for the caller to warn about."""
    poly = segmentation_to_polydata(
        _volume([(slice(6, 20), slice(6, 20), slice(6, 20)),
                 (slice(24, 36), slice(24, 36), slice(24, 36))]),
        flip=False, **SMOOTH)
    out, shells = drop_stray_shells(poly)
    assert shells.n_components == 2
    assert shells.dropped == ()
    assert len(shells.kept) == 2
    assert out is poly
    assert len(_components(pv.wrap(out))) == 2


@pytest.mark.parametrize("min_fraction,expect_dropped", [(1.0, 1), (0.0, 0)])
def test_the_threshold_decides(min_fraction, expect_dropped):
    poly = segmentation_to_polydata(
        _volume([(slice(6, 20), slice(6, 20), slice(6, 20)),
                 (slice(24, 36), slice(24, 36), slice(24, 36))]),
        flip=False, **SMOOTH)
    _, shells = drop_stray_shells(poly, min_fraction=min_fraction)
    assert len(shells.dropped) == expect_dropped
    assert shells.kept[0] == max(shells.kept)


def test_the_kept_surface_is_still_closed():
    out, _ = drop_stray_shells(
        segmentation_to_polydata(_speck_volume(), flip=False, **SMOOTH))
    counts = _edge_counts(pv.wrap(out))
    assert int(np.count_nonzero(counts == 1)) == 0
    assert int(np.count_nonzero(counts > 2)) == 0
