"""
test_volume_uac.py
==================
Volume UAC (:mod:`ccdaf.core.uac.volume`) pieces that need no surface UAC
run: when a volume can be given UAC, the fold test, and what is written.
The full run is validated on real atria (see the module notes); a
synthetic atrium with four veins and an appendage laid out as the template
expects is beyond a unit test.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pytest

from ccdaf.core import atrial_labels as al
from ccdaf.core import tag_map
from ccdaf.core.atrial_surfaces import label_atria
from ccdaf.core.chamber_extract import extract_chambers
from ccdaf.core.surface_labels import LABEL_MASK_FIELD, node_label_mask
from ccdaf.core.uac import volume as vu
from test_atrial_surfaces import LA_MAP, _shell


@pytest.fixture(scope="module")
def labelled():
    grid = extract_chambers(_shell(), LA_MAP, ["LA"]).grid
    labels = label_atria(grid, tag_map.read(grid))
    grid.point_data[LABEL_MASK_FIELD] = node_label_mask(grid, labels)
    return grid


def test_a_volume_without_surface_labels_is_refused():
    ok, why = vu.availability(_shell())
    assert not ok and "Label the atrial surfaces" in why


def test_a_volume_without_regions_is_refused(labelled):
    grid = labelled.copy()
    grid.cell_data["elemTag"] = np.full(grid.n_cells, al.LA.body)
    ok, why = vu.availability(grid)
    assert not ok and "Tag the endocardium" in why
    with pytest.raises(vu.VolumeUacError):
        vu.compute(grid)


def test_a_surface_is_refused():
    import pyvista as pv
    ok, _why = vu.availability(pv.Sphere())
    assert not ok


def test_the_jacobian_of_the_coordinates_themselves_is_one():
    X = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]])
    tet = np.array([[0, 1, 2, 3]])
    j = vu._jacobians(X, tet, X[:, 0], X[:, 1], X[:, 2])
    assert j[0] == pytest.approx(1.0)
    # Swapping two coordinates turns the tetrahedron inside out.
    assert vu._jacobians(X, tet, X[:, 1], X[:, 0], X[:, 2])[0] == pytest.approx(-1.0)


def test_write_adds_every_field(labelled):
    grid = labelled.copy()
    n, m = grid.n_points, grid.n_cells
    run = vu.VolumeUac(alpha=np.zeros(n), beta=np.zeros(n), depth=np.zeros(n),
                       alpha_tet=np.zeros(m), beta_tet=np.zeros(m),
                       half=np.zeros(m, np.int32), atrium="LA",
                       epicardium=vu.FROM_ENDO, endo_run=None,
                       folded_mask=np.zeros(m, bool))
    vu.write(grid, run)
    for name in vu.written_names():
        assert name in grid.point_data or name in grid.cell_data
