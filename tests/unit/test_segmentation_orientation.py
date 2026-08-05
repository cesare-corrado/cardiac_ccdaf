"""
test_segmentation_orientation.py
================================
Guards the NIfTI orientation handling in ``ccdaf.core.segmentation`` and the
load/save wiring that uses it.

The bug this pins: a segmentation stored in any orientation other than LPS
rendered its 3D surface and its slice planes in two different places. Marching
cubes honours the image's direction matrix; the segmentation view does not —
it places every slice, crosshair, brush stroke and plane gizmo at
``origin + index * spacing``. A RAS volume (the direction VTK's NIfTI writer
produces by default) therefore drew its planes mirrored about the origin,
132 mm away from the surface on the case that surfaced this.

The contract:

* a volume is re-indexed into LPS on load, which makes the view's placement
  rule true rather than merely assumed;
* re-indexing is lossless — labels keep their world positions, so the surface
  marching cubes builds does not move by so much as a float;
* the orientation it arrived in is remembered and written back on save, so the
  round trip through the app is invisible to the rest of the pipeline;
* an LPS volume is left completely alone;
* an oblique volume is detected, because a permutation cannot straighten it.

Synthetic volumes only — no real data, no display.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pytest
import SimpleITK as sitk

from ccdaf.core.segmentation import (
    LPS,
    is_axis_aligned,
    orientation_code,
    reorient_to_lps,
    restore_orientation,
    segmentation_to_polydata,
)

# ITK reports direction cosines in LPS, so RAS is a 180° turn about z.
RAS_DIRECTION = (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)
LAS_DIRECTION = (1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)
OBLIQUE_DIRECTION = tuple(
    np.array([[np.cos(np.deg2rad(15)), -np.sin(np.deg2rad(15)), 0.0],
              [np.sin(np.deg2rad(15)), np.cos(np.deg2rad(15)), 0.0],
              [0.0, 0.0, 1.0]]).ravel()
)


def _volume(direction=RAS_DIRECTION) -> sitk.Image:
    """An asymmetric labelled blob, so any missed flip shows up."""
    arr = np.zeros((14, 16, 18), dtype=np.uint8)   # (Z, Y, X)
    arr[3:9, 4:11, 5:14] = 1
    arr[3:5, 4:6, 5:7] = 2                          # corner marker
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.5, 1.25, 2.0))
    img.SetOrigin((95.0, 53.0, -14.0))
    img.SetDirection(direction)
    return img


def _naive_matches_world(img: sitk.Image) -> bool:
    """True when ``origin + index * spacing`` is where the voxel really is.

    That equality is what the whole segmentation view assumes.
    """
    origin = np.array(img.GetOrigin())
    spacing = np.array(img.GetSpacing())
    nx, ny, nz = img.GetSize()
    for idx in [(0, 0, 0), (nx - 1, 0, 0), (0, ny - 1, 0), (0, 0, nz - 1),
                (nx - 1, ny - 1, nz - 1), (nx // 2, ny // 2, nz // 2)]:
        true = np.array(img.TransformIndexToPhysicalPoint(idx))
        if not np.allclose(origin + np.array(idx) * spacing, true):
            return False
    return True


def _label_world_positions(img: sitk.Image) -> set:
    """Every non-zero voxel as ``(label, rounded world xyz)``."""
    arr = sitk.GetArrayFromImage(img)
    out = set()
    for k, j, i in np.argwhere(arr > 0):
        w = img.TransformIndexToPhysicalPoint((int(i), int(j), int(k)))
        out.add((int(arr[k, j, i]), tuple(np.round(w, 6))))
    return out


# ---------------------------------------------------------------- codes


def test_identity_direction_is_lps():
    img = _volume(direction=tuple(np.eye(3).ravel()))
    assert orientation_code(img) == LPS


@pytest.mark.parametrize("direction,expected",
                         [(RAS_DIRECTION, "RAS"), (LAS_DIRECTION, "LAS")])
def test_flipped_axes_report_their_code(direction, expected):
    assert orientation_code(_volume(direction)) == expected


@pytest.mark.parametrize("direction,aligned", [
    (tuple(np.eye(3).ravel()), True),
    (RAS_DIRECTION, True),
    (LAS_DIRECTION, True),
    ((0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0), True),   # a permutation
    (OBLIQUE_DIRECTION, False),
])
def test_is_axis_aligned(direction, aligned):
    assert is_axis_aligned(_volume(direction)) is aligned


# ---------------------------------------------------------------- reorient


def test_lps_volume_is_left_alone():
    img = _volume(direction=tuple(np.eye(3).ravel()))
    out, code = reorient_to_lps(img)
    assert code == LPS
    assert out is img


@pytest.mark.parametrize("direction", [RAS_DIRECTION, LAS_DIRECTION])
def test_reorient_makes_the_views_placement_rule_true(direction):
    """The whole point: the view's ``origin + index * spacing`` becomes real."""
    img = _volume(direction)
    assert not _naive_matches_world(img)        # the bug, before
    out, _ = reorient_to_lps(img)
    assert orientation_code(out) == LPS
    assert _naive_matches_world(out)            # fixed, after


@pytest.mark.parametrize("direction", [RAS_DIRECTION, LAS_DIRECTION])
def test_reorient_is_lossless(direction):
    """Every label keeps its value and its world position."""
    img = _volume(direction)
    out, _ = reorient_to_lps(img)
    assert _label_world_positions(out) == _label_world_positions(img)
    assert np.allclose(sorted(out.GetSpacing()), sorted(img.GetSpacing()))


@pytest.mark.parametrize("direction", [RAS_DIRECTION, LAS_DIRECTION])
def test_surface_does_not_move(direction):
    """Marching cubes already worked in world coordinates — keep it that way.

    This is what makes the fix safe for anything downstream of the converter:
    the exported mesh is bit-for-bit where it was.
    """
    img = _volume(direction)
    kw = dict(flip=False, filt_stdev=[0, 0, 0], filt_rfact=[0, 0, 0])
    before = segmentation_to_polydata(img, **kw)
    after = segmentation_to_polydata(reorient_to_lps(img)[0], **kw)
    assert after.GetNumberOfPoints() == before.GetNumberOfPoints()
    assert np.allclose(after.GetBounds(), before.GetBounds())


def test_oblique_volume_survives_but_stays_oblique():
    """A permutation cannot rotate axes, so the caller must be warned."""
    img = _volume(OBLIQUE_DIRECTION)
    out, _ = reorient_to_lps(img)
    assert not is_axis_aligned(out)


# ---------------------------------------------------------------- restore


@pytest.mark.parametrize("direction", [RAS_DIRECTION, LAS_DIRECTION,
                                       tuple(np.eye(3).ravel())])
def test_restore_orientation_round_trips_exactly(direction):
    img = _volume(direction)
    out, code = reorient_to_lps(img)
    back = restore_orientation(out, code)
    assert np.array_equal(sitk.GetArrayFromImage(back), sitk.GetArrayFromImage(img))
    assert np.allclose(back.GetOrigin(), img.GetOrigin())
    assert np.allclose(back.GetDirection(), img.GetDirection())
    assert np.allclose(back.GetSpacing(), img.GetSpacing())


@pytest.mark.parametrize("code", [None, "", LPS])
def test_restore_orientation_no_ops(code):
    img = _volume(direction=tuple(np.eye(3).ravel()))
    assert restore_orientation(img, code) is img


# ---------------------------------------------------------------- app wiring


def _host():
    """Stand-in for the window: constructing the real one needs a GL context."""
    host = MagicMock()
    host._seg_undo_stack = []
    return host


@pytest.mark.parametrize("direction,expected", [
    (RAS_DIRECTION, "RAS"),
    (LAS_DIRECTION, "LAS"),
    (tuple(np.eye(3).ravel()), LPS),
])
def test_set_segmentation_records_and_normalises(direction, expected):
    from ccdaf.app.ccdaf import CCDAF

    host = _host()
    CCDAF._set_segmentation(host, _volume(direction))

    assert host._seg_orientation == expected
    assert orientation_code(host._seg_sitk) == LPS
    # The geometry the slice views were handed is the re-indexed one.
    assert host._seg_origin == tuple(host._seg_sitk.GetOrigin())
    assert host._seg_spacing == tuple(host._seg_sitk.GetSpacing())
    assert _naive_matches_world(host._seg_sitk)
    assert host._seg_array.shape == sitk.GetArrayFromImage(host._seg_sitk).shape


def test_set_segmentation_warns_on_oblique(monkeypatch):
    from ccdaf.app import ccdaf as app_mod

    warned = []
    monkeypatch.setattr(app_mod.QtWidgets.QMessageBox, "warning",
                        lambda *a, **k: warned.append(a))
    CCDAF = app_mod.CCDAF
    CCDAF._set_segmentation(_host(), _volume(OBLIQUE_DIRECTION))
    assert warned, "an oblique volume must not be re-indexed silently"

    warned.clear()
    CCDAF._set_segmentation(_host(), _volume(RAS_DIRECTION))
    assert not warned, "an axis-aligned volume is handled exactly, no warning"


@pytest.mark.parametrize("direction,expected", [
    (RAS_DIRECTION, "RAS"),
    (LAS_DIRECTION, "LAS"),
    (tuple(np.eye(3).ravel()), LPS),
])
def test_save_writes_back_the_original_orientation(tmp_path, direction, expected):
    from ccdaf.app.ccdaf import CCDAF

    img = _volume(direction)
    host = _host()
    CCDAF._set_segmentation(host, img)

    fn = tmp_path / "seg.nii"
    assert CCDAF._write_segmentation(host, str(fn)) is True

    reloaded = sitk.ReadImage(str(fn))
    assert orientation_code(reloaded) == expected
    assert np.array_equal(sitk.GetArrayFromImage(reloaded),
                          sitk.GetArrayFromImage(img))
    assert np.allclose(reloaded.GetOrigin(), img.GetOrigin())
    assert np.allclose(reloaded.GetDirection(), img.GetDirection())


def test_save_reports_failure_rather_than_raising(monkeypatch):
    from ccdaf.app import ccdaf as app_mod

    reported = []
    monkeypatch.setattr(app_mod.QtWidgets.QMessageBox, "critical",
                        lambda *a, **k: reported.append(a))
    host = _host()
    app_mod.CCDAF._set_segmentation(host, _volume())
    assert app_mod.CCDAF._write_segmentation(host, "/nonexistent-dir/seg.nii") is False
    assert reported, "an unwritable path must be reported, not swallowed"


@pytest.mark.parametrize("code,expected", [(LPS, ""), ("RAS", " (reoriented RAS → LPS)")])
def test_orientation_note(code, expected):
    from ccdaf.app.ccdaf import CCDAF

    host = MagicMock()
    host._seg_orientation = code
    assert CCDAF._seg_orientation_note(host) == expected
