"""
test_segmentation_core.py
=========================
Tests for the mesh ↔ volume conversion in ``ccdaf.core.segmentation`` —
the segmentation round trip, runnable at last with no GUI class around it.

The contract:

* voxelising a closed surface fills its interior (volume ≈ analytic),
  on a grid padded by one background voxel per side;
* the MIRTK flip is applied to a copy — the caller's mesh never moves —
  and flipping twice is the identity;
* the rebuilt surface lands on the source wall (within the smoothing's
  reach) and comes back all-triangle;
* label masking meshes one label; the binary mask takes everything > 0;
* array edits synced back keep the reference image's geometry;
* a missing image raises rather than meshing nothing.

Uses synthetic spheres (no real data, no display).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pytest
import pyvista as pv
import SimpleITK as sitk

from ccdaf.core.segmentation import (
    binary_mask_image,
    label_mask_image,
    negate_xy_inplace,
    segmentation_to_polydata,
    sitk_to_vtk_image,
    sync_sitk_from_array,
    voxelise_polydata,
    vtk_image_to_sitk,
)

RADIUS = 8.0


def _sphere(center=(0.0, 0.0, 0.0)) -> pv.PolyData:
    return pv.Sphere(radius=RADIUS, center=center,
                     theta_resolution=60, phi_resolution=60)


def test_voxelise_fills_the_interior():
    img = voxelise_polydata(_sphere(), (1.0, 1.0, 1.0), flip=False)
    arr = sitk.GetArrayFromImage(img)
    volume = float(arr.sum())          # 1mm voxels: count == mm^3
    analytic = 4.0 / 3.0 * np.pi * RADIUS ** 3
    assert abs(volume - analytic) / analytic < 0.05
    # One voxel of background on every side — no foreground on any face.
    assert arr[0].max() == 0 and arr[-1].max() == 0
    assert arr[:, 0].max() == 0 and arr[:, -1].max() == 0
    assert arr[:, :, 0].max() == 0 and arr[:, :, -1].max() == 0


def test_voxelise_rejects_nonpositive_spacing():
    with pytest.raises(ValueError):
        voxelise_polydata(_sphere(), (1.0, 0.0, 1.0), flip=False)


def test_flip_copies_and_matches_pre_negated_mesh():
    center = (5.0, 3.0, 0.0)
    mesh = _sphere(center)
    before = np.asarray(mesh.points).copy()
    flipped = voxelise_polydata(mesh, (1.0, 1.0, 1.0), flip=True)
    assert np.array_equal(np.asarray(mesh.points), before), \
        "flip must act on a copy, not the caller's mesh"

    negated = _sphere(center)
    negate_xy_inplace(negated)
    reference = voxelise_polydata(negated, (1.0, 1.0, 1.0), flip=False)
    assert np.array_equal(sitk.GetArrayFromImage(flipped),
                          sitk.GetArrayFromImage(reference))
    assert flipped.GetOrigin() == reference.GetOrigin()


def test_negate_xy_twice_is_identity():
    mesh = _sphere((2.0, -1.0, 4.0))
    before = np.asarray(mesh.points).copy()
    negate_xy_inplace(mesh)
    negate_xy_inplace(mesh)
    assert np.allclose(np.asarray(mesh.points), before)


def test_round_trip_lands_on_the_source_wall():
    img = voxelise_polydata(_sphere(), (1.0, 1.0, 1.0), flip=False)
    poly = segmentation_to_polydata(
        img, flip=False, filt_stdev=[1.0] * 3, filt_rfact=[1.5] * 3)
    surf = pv.wrap(poly)
    assert surf.n_points > 0
    assert surf.is_all_triangles
    radii = np.linalg.norm(np.asarray(surf.points), axis=1)
    # Rebuilt from a 1mm rasterisation with sigma-1 smoothing: the wall
    # must sit on the sphere to within grid+smoothing reach.
    assert abs(float(radii.mean()) - RADIUS) < 0.5
    assert float(np.abs(radii - RADIUS).max()) < 1.5


def test_label_selects_one_body():
    # Two spheres, labels 1 and 2, far apart on one grid.
    a = voxelise_polydata(_sphere((0, 0, 0)), (1.0, 1.0, 1.0), flip=False)
    arr = sitk.GetArrayFromImage(a).astype(np.int16)
    nz, ny, nx = arr.shape
    two = np.zeros((nz, ny, 2 * nx), dtype=np.int16)
    two[:, :, :nx] = arr
    two[:, :, nx:] = arr * 2
    img = sitk.GetImageFromArray(two)
    img.SetSpacing((1.0, 1.0, 1.0))

    only_two = segmentation_to_polydata(
        img, flip=False, filt_stdev=[1.0] * 3, filt_rfact=[1.5] * 3, label=2)
    both = segmentation_to_polydata(
        img, flip=False, filt_stdev=[1.0] * 3, filt_rfact=[1.5] * 3)
    xs = np.asarray(pv.wrap(only_two).points)[:, 0]
    assert xs.min() > nx / 2.0, "label=2 must mesh only the second body"
    assert pv.wrap(both).n_points > pv.wrap(only_two).n_points


def test_masks_binarise_and_keep_geometry():
    arr = np.zeros((4, 4, 4), dtype=np.int16)
    arr[1, 1, 1] = 1
    arr[2, 2, 2] = 5
    img = sitk.GetImageFromArray(arr)
    img.SetOrigin((1.0, 2.0, 3.0))
    img.SetSpacing((0.5, 0.5, 2.0))

    b = sitk.GetArrayFromImage(binary_mask_image(img))
    assert b.dtype == np.uint8 and b.sum() == 2 and set(b.ravel()) == {0, 1}
    m = sitk.GetArrayFromImage(label_mask_image(img, 5))
    assert m.sum() == 1 and m[2, 2, 2] == 1
    assert label_mask_image(img, 5).GetOrigin() == img.GetOrigin()


def test_sync_keeps_reference_geometry():
    ref = sitk.GetImageFromArray(np.zeros((3, 3, 3), dtype=np.int16))
    ref.SetOrigin((10.0, 20.0, 30.0))
    ref.SetSpacing((2.0, 2.0, 2.0))
    edited = np.arange(27, dtype=np.int32).reshape(3, 3, 3)
    out = sync_sitk_from_array(edited, ref)
    assert out.GetOrigin() == ref.GetOrigin()
    assert out.GetSpacing() == ref.GetSpacing()
    assert out.GetDirection() == ref.GetDirection()
    assert np.array_equal(sitk.GetArrayFromImage(out), edited)


def test_image_conversions_round_trip():
    arr = np.random.default_rng(7).integers(0, 3, (5, 6, 7)).astype(np.uint8)
    img = sitk.GetImageFromArray(arr)
    img.SetOrigin((1.0, -2.0, 3.5))
    img.SetSpacing((0.5, 1.0, 1.5))
    back = vtk_image_to_sitk(sitk_to_vtk_image(img))
    assert np.array_equal(sitk.GetArrayFromImage(back), arr)
    assert back.GetOrigin() == img.GetOrigin()
    assert back.GetSpacing() == img.GetSpacing()


def test_missing_image_raises():
    with pytest.raises(RuntimeError):
        binary_mask_image(None)
    with pytest.raises(RuntimeError):
        segmentation_to_polydata(
            None, flip=False, filt_stdev=[1.0] * 3, filt_rfact=[1.5] * 3)
