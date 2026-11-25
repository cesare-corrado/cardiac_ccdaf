"""
test_volume_fibres.py
=====================
Atrial fibres through the wall (:mod:`ccdaf.core.uac.volume_fibres`): the
turn from the endocardial to the epicardial fibre with depth, the sheet,
and the filling of tetrahedra with no direction through the wall. The
full run needs volume UAC and the atlas, and is validated on real atria.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pytest

from ccdaf.core.uac import volume_fibres as vf

N = np.array([[0.0, 0.0, 1.0]])           # through the wall
ENDO = np.array([[1.0, 0.0, 0.0]])
EPI = np.array([[0.0, 1.0, 0.0]])         # 90 degrees from the endocardial


@pytest.mark.parametrize("depth, expected_deg", [(0.0, 0), (0.5, 45), (1.0, 90)])
def test_the_fibre_turns_linearly_with_depth(depth, expected_deg):
    f, angle = vf.blend(ENDO, EPI, N, np.array([depth]))
    assert angle[0] == pytest.approx(90)
    got = np.degrees(np.arctan2(f[0, 1], f[0, 0]))
    assert got == pytest.approx(expected_deg, abs=1e-9)


def test_a_fibre_is_a_line_not_an_arrow():
    # The epicardial fibre pointing the other way is the same fibre: the
    # turn takes the shorter way, 30 degrees, not 150.
    epi = np.array([[-np.cos(np.radians(30)), -np.sin(np.radians(30)), 0.0]])
    _f, angle = vf.blend(ENDO, epi, N, np.array([1.0]))
    assert angle[0] == pytest.approx(30)


def test_the_fibre_lies_in_the_wall():
    tilted = np.array([[1.0, 0.0, 0.7]])  # a fibre with a part through the wall
    f, _ = vf.blend(tilted, EPI, N, np.array([0.3]))
    assert abs(np.dot(f[0], N[0])) < 1e-12
    assert np.linalg.norm(f[0]) == pytest.approx(1.0)


def test_missing_sheets_take_their_neighbours():
    v = np.array([[0, 0, 1.0], [np.nan] * 3, [0, 0, -1.0], [np.nan] * 3])
    a = np.array([0, 1, 2])
    b = np.array([1, 2, 3])
    filled = vf._fill_from_neighbours(v, a, b)
    assert filled == 2
    assert np.allclose(np.abs(v[:, 2]), 1.0)      # signs aligned, not cancelled


def test_a_volume_without_uac_is_refused():
    import pyvista as pv
    grid = pv.ImageData(dimensions=(3, 3, 3)).cast_to_unstructured_grid().triangulate()
    ok, why = vf.availability(grid)
    assert not ok and "Compute the volume UAC" in why
    assert not vf.availability(pv.Sphere())[0]
