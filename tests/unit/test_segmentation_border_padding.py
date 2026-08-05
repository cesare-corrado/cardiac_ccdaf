"""
test_segmentation_border_padding.py
===================================
A label that runs into the volume border still comes back closed.

Marching cubes needs background on the far side of a wall to put a
triangle there. Where the anatomy reaches the edge of the volume — a scan
cropped tight around the atrium, or a mask edited out to the border — the
signed distance stayed positive all the way to the last voxel plane and
the surface was left open: the export came back with holes exactly on the
faces the segmentation touched.

The contract:

* the mask is padded with background before the distance transforms, by
  one plane plus the smoothing kernel's reach, per axis;
* a body touching every face of its volume meshes closed, and meshes to
  the same surface as the same body with room around it — cropping the
  volume must not change the mesh;
* padding does not move the result: a body already clear of the border
  meshes exactly as before, in the same world position;
* ``pad=0`` still meshes the mask as stored — it is what pins the
  behaviour the padding fixes.

Synthetic volumes only (no real data, no display).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pytest
import pyvista as pv
import SimpleITK as sitk

from ccdaf.core.segmentation import border_padding, segmentation_to_polydata

SMOOTH = dict(filt_stdev=[0.5] * 3, filt_rfact=[1.5] * 3)


def _ball(shape=(24, 24, 24), radius=8.0, label=1):
    """A ball centred in *shape*, as a labelled uint8 volume."""
    zz, yy, xx = np.indices(shape)
    c = (np.asarray(shape) - 1) / 2.0
    r = np.sqrt((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2)
    arr = np.where(r <= radius, label, 0).astype(np.uint8)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    return img


def _cropped_to_labels(img):
    """*img* cropped tight to its foreground — the clinical border case."""
    arr = sitk.GetArrayFromImage(img)
    z, y, x = np.nonzero(arr)
    box = arr[z.min():z.max() + 1, y.min():y.max() + 1, x.min():x.max() + 1]
    out = sitk.GetImageFromArray(box)
    out.SetSpacing(img.GetSpacing())
    out.SetDirection(img.GetDirection())
    out.SetOrigin(img.TransformIndexToPhysicalPoint(
        (int(x.min()), int(y.min()), int(z.min()))))
    return out


def _open_edges(poly) -> int:
    surf = pv.wrap(poly)
    edges = surf.extract_feature_edges(
        boundary_edges=True, feature_edges=False,
        manifold_edges=False, non_manifold_edges=False)
    return edges.n_cells


def _touches_every_face(img) -> bool:
    a = sitk.GetArrayFromImage(img)
    return bool(a[0].any() and a[-1].any() and a[:, 0].any()
                and a[:, -1].any() and a[:, :, 0].any() and a[:, :, -1].any())


# ---------------------------------------------------------------------------
# How much padding
# ---------------------------------------------------------------------------
def test_one_plane_of_background_when_smoothing_is_off():
    assert border_padding([0.0, 0.0, 0.0], [1.5, 1.5, 1.5]) == [1, 1, 1]
    assert border_padding([0.5, 0.5, 0.5], [0.0, 0.0, 0.0]) == [1, 1, 1]


def test_the_smoothing_kernel_reach_is_added_on_top():
    """The kernel reads stdev*factor voxels either side; the closure has to
    sit that far in, or the truncated kernel at the image edge shapes it."""
    assert border_padding([0.5] * 3, [1.5] * 3) == [2, 2, 2]      # 1 + ceil(0.75)
    assert border_padding([2.0] * 3, [2.0] * 3) == [5, 5, 5]      # 1 + 4


def test_each_axis_gets_its_own_width():
    assert border_padding([0.5, 2.0, 4.0], [2.0, 2.0, 2.0]) == [2, 5, 9]


# ---------------------------------------------------------------------------
# What it fixes
# ---------------------------------------------------------------------------
def test_a_body_touching_every_face_meshes_closed():
    img = _cropped_to_labels(_ball())
    assert _touches_every_face(img)
    assert _open_edges(segmentation_to_polydata(img, flip=False, **SMOOTH)) == 0


def test_without_padding_that_body_is_open():
    """Pins the defect: pad=0 is the old behaviour, holes and all."""
    img = _cropped_to_labels(_ball())
    assert _open_edges(
        segmentation_to_polydata(img, flip=False, pad=0, **SMOOTH)) > 0


def test_cropping_the_volume_no_longer_changes_the_mesh():
    """The same body with and without room around it — one surface."""
    roomy = _ball()
    tight = _cropped_to_labels(roomy)
    a = pv.wrap(segmentation_to_polydata(roomy, flip=False, **SMOOTH))
    b = pv.wrap(segmentation_to_polydata(tight, flip=False, **SMOOTH))
    assert a.n_points == b.n_points
    assert np.allclose(np.asarray(a.points), np.asarray(b.points), atol=1e-9)


def test_a_single_label_touching_the_border_is_closed_too():
    """The 3D preview meshes one label at a time; padding follows it."""
    img = _cropped_to_labels(_ball(label=3))
    poly = segmentation_to_polydata(img, flip=False, label=3, **SMOOTH)
    assert pv.wrap(poly).n_points > 0
    assert _open_edges(poly) == 0


# ---------------------------------------------------------------------------
# What it must not disturb
# ---------------------------------------------------------------------------
def test_a_body_clear_of_the_border_is_meshed_unchanged():
    img = _ball()
    padded = pv.wrap(segmentation_to_polydata(img, flip=False, **SMOOTH))
    plain = pv.wrap(segmentation_to_polydata(img, flip=False, pad=0, **SMOOTH))
    assert padded.n_points == plain.n_points
    assert np.allclose(np.asarray(padded.points), np.asarray(plain.points),
                       atol=1e-9)


def test_padding_keeps_the_world_position():
    """Padding shifts the origin with the voxels. An off-centre volume is
    where a lost shift would show up as a surface in the wrong place."""
    img = _ball()
    img.SetOrigin((-13.5, 7.25, 42.0))
    img.SetSpacing((0.8, 1.1, 1.4))
    surf = pv.wrap(segmentation_to_polydata(img, flip=False, **SMOOTH))
    size = np.asarray(img.GetSize())
    spacing = np.asarray(img.GetSpacing())
    expected = np.asarray(img.GetOrigin()) + (size - 1) * spacing / 2.0
    assert np.allclose(np.asarray(surf.center), expected, atol=0.5)


@pytest.mark.parametrize("flip", [False, True])
def test_flip_still_applies_after_padding(flip):
    img = _cropped_to_labels(_ball())
    surf = pv.wrap(segmentation_to_polydata(img, flip=flip, **SMOOTH))
    reference = pv.wrap(segmentation_to_polydata(img, flip=False, **SMOOTH))
    xy = np.asarray(surf.points)[:, :2]
    expected = np.asarray(reference.points)[:, :2] * (-1.0 if flip else 1.0)
    assert np.allclose(xy, expected)
