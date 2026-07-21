"""
test_relabel_halfspace.py
========================
`relabel_halfspace` — recolour one label on the normal-positive side of an
arbitrary plane, for clipping a segmentation along an oblique anatomical cut
the three axis-locked planes cannot express.

The contract:

* only voxels equal to ``from_label`` on the side the normal points to change,
  to ``to_label``; the other half and every other label are untouched;
* flipping the normal relabels the opposite half;
* the plane may be oblique (works off the axes);
* a zero normal or an absent ``from_label`` is a no-op, and the input array is
  never modified.

Uses synthetic arrays (no real data).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np

from ccdaf.core.segmentation import relabel_halfspace


ORIGIN = (0.0, 0.0, 0.0)
SPACING = (1.0, 1.0, 1.0)


def _filled(nz, ny, nx, label=1):
    return np.full((nz, ny, nx), label, dtype=np.int32)


def test_positive_x_side_is_relabelled():
    arr = _filled(4, 4, 10)                       # (Z, Y, X)
    # plane at x=4.5, normal +x -> voxels with x-index 5..9 relabelled
    out = relabel_halfspace(arr, ORIGIN, SPACING, point=(4.5, 0, 0),
                            normal=(1, 0, 0), from_label=1, to_label=7)
    assert np.all(out[:, :, :5] == 1)             # x < 4.5 untouched
    assert np.all(out[:, :, 5:] == 7)             # x > 4.5 relabelled


def test_flipping_normal_relabels_the_other_half():
    arr = _filled(4, 4, 10)
    out = relabel_halfspace(arr, ORIGIN, SPACING, point=(4.5, 0, 0),
                            normal=(-1, 0, 0), from_label=1, to_label=7)
    assert np.all(out[:, :, :5] == 7)             # x < 4.5 now relabelled
    assert np.all(out[:, :, 5:] == 1)


def test_only_from_label_changes():
    arr = _filled(4, 4, 10, label=1)
    arr[:, :, 7] = 3                              # a stripe of a different label
    out = relabel_halfspace(arr, ORIGIN, SPACING, point=(4.5, 0, 0),
                            normal=(1, 0, 0), from_label=1, to_label=7)
    assert np.all(out[:, :, 7] == 3)             # label 3 untouched even on the +x side
    assert out[0, 0, 6] == 7 and out[0, 0, 8] == 7


def test_oblique_plane_is_not_axis_aligned():
    arr = _filled(1, 20, 20)
    # 45-degree plane in the x-y plane through the centre; normal (1,1,0)
    out = relabel_halfspace(arr, ORIGIN, SPACING, point=(9.5, 9.5, 0),
                            normal=(1, 1, 0), from_label=1, to_label=5)
    # a voxel clearly on the +normal side and one clearly on the - side
    assert out[0, 19, 19] == 5                    # x+y large -> positive side
    assert out[0, 0, 0] == 1                      # x+y small -> negative side
    # the split is diagonal, so both labels are present in roughly balanced parts
    assert 0.3 < (out == 5).mean() < 0.7


def test_zero_normal_and_absent_label_are_noops():
    arr = _filled(3, 3, 3)
    assert np.array_equal(
        relabel_halfspace(arr, ORIGIN, SPACING, (1, 1, 1), (0, 0, 0), 1, 9), arr)
    assert np.array_equal(
        relabel_halfspace(arr, ORIGIN, SPACING, (1, 1, 1), (1, 0, 0), 4, 9), arr)


def test_input_is_not_mutated_and_origin_spacing_respected():
    arr = _filled(2, 2, 6)
    before = arr.copy()
    # origin shifts the world, spacing stretches it: plane at world x=10
    out = relabel_halfspace(arr, origin=(4.0, 0, 0), spacing=(2.0, 1, 1),
                            point=(10.0, 0, 0), normal=(1, 0, 0),
                            from_label=1, to_label=8)
    assert np.array_equal(arr, before)           # untouched
    # world x of index i is 4 + 2*i: 4,6,8,10,12,14. Plane at 10, strictly
    # greater is relabelled -> index 3 (world 10) stays, 4 and 5 change.
    assert np.all(out[:, :, :4] == 1)
    assert np.all(out[:, :, 4:] == 8)
