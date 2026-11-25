"""
test_atrial_surfaces.py
=======================
Chamber extraction and atrial surface labelling on a synthetic atrium: a
spherical shell two prisms thick, with cone-shaped plugs through the wall
tagged as a vein cap and a valve.

* extraction keeps the tissue, drops the plugs, and records their rims;
* the recorded rims split the boundary into epicardium (outer sphere),
  endocardium (inner sphere) and the rim (the plug holes' walls);
* without recorded rims the openings are found from the shape;
* a second, separate shell gives two endocardia named from the tags;
* the names can be swapped, and the legend calls the base "rim".
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pytest
import pyvista as pv
from PyQt5 import QtWidgets

from ccdaf.app.ccdaf import _surface_label_legend
from ccdaf.core import tag_map
from ccdaf.core.atrial_surfaces import (
    AtrialOptions, has_recorded_rims, label_atria, swap_atria,
)
from ccdaf.core.chamber_extract import (
    OPENING_FIELD, available_chambers, extract_chambers,
)
from ccdaf.core.surface_labels import (
    BASE, EPI, LA_ENDO, LABEL_MASK_FIELD, RA_ENDO, _as_boundary,
    face_labels_from_mask, node_label_mask,
)
from ccdaf.gui.atrial_labels_dialog import AtrialLabelsDialog
from ccdaf.gui.chamber_extract_dialog import ChamberExtractDialog

R_IN, R_MID, R_OUT = 20.0, 22.0, 24.0
LA_MAP = {1: "LA body", 11: "LSPV cap", 7: "MV", 2: "RA body", 16: "SVC cap",
          8: "TV"}


def _shell(centre=(0.0, 0.0, 0.0), body=1, plugs=((11, (0, 0, 1)), (7, (1, 0, 0))),
           cone_deg=25.0):
    """A conforming tetrahedral shell; tets under each plug's cone get its tag."""
    sphere = pv.Sphere(radius=1.0, theta_resolution=36, phi_resolution=36)
    unit = np.asarray(sphere.points, dtype=float)
    tri = np.asarray(sphere.faces).reshape(-1, 4)[:, 1:]
    n = len(unit)
    points = np.vstack([unit * r for r in (R_IN, R_MID, R_OUT)]) + centre
    tets, tags = [], []
    centres = unit[tri].mean(1)
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    face_tag = np.full(len(tri), body)
    for tag, axis in plugs:
        axis = np.asarray(axis, float) / np.linalg.norm(axis)
        face_tag[centres @ axis > np.cos(np.radians(cone_deg))] = tag
    for layer in range(2):
        for f, t in zip(np.sort(tri, axis=1), face_tag):
            a, b, c = f + layer * n
            a2, b2, c2 = a + n, b + n, c + n
            # Split each prism from its smallest index: neighbours agree on
            # every shared quad's diagonal, so the mesh is conforming.
            tets += [(a, b, c, a2), (b, c, a2, b2), (c, a2, b2, c2)]
            tags += [t] * 3
    cells = np.hstack([np.full((len(tets), 1), 4), np.asarray(tets)]).ravel()
    grid = pv.UnstructuredGrid(cells, np.full(len(tets), pv.CellType.TETRA),
                               points)
    grid.cell_data["elemTag"] = np.asarray(tags)
    return grid


def _radius_of(labels, grid, key, centre=(0.0, 0.0, 0.0)):
    surface = _as_boundary(grid)
    tri = np.asarray(surface.faces).reshape(-1, 4)[:, 1:]
    c = np.asarray(surface.points)[tri].mean(1) - centre
    return np.linalg.norm(c[labels.face_labels == key], axis=1)


# ---------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------
def test_extraction_keeps_tissue_and_records_the_rims():
    grid = _shell()
    ex = extract_chambers(grid, LA_MAP, ["LA"])
    assert set(np.unique(ex.grid.cell_data["elemTag"])) == {1}
    assert ex.tissue_tags == [1]
    assert ex.opening_tags == [7, 11]
    assert has_recorded_rims(ex.grid)
    assert set(np.unique(ex.grid.point_data[OPENING_FIELD])) == {0, 7, 11}
    assert tag_map.read(ex.grid) == tag_map.normalise(LA_MAP)


def test_extraction_drops_a_stale_labelling():
    grid = _shell()
    grid.point_data[LABEL_MASK_FIELD] = np.ones(grid.n_points, dtype=np.int32)
    ex = extract_chambers(grid, LA_MAP, ["LA"])
    assert LABEL_MASK_FIELD not in ex.grid.point_data


def test_extraction_refuses_a_chamber_without_tissue():
    with pytest.raises(ValueError):
        extract_chambers(_shell(), LA_MAP, ["LV"])


def test_available_chambers_follow_the_tags_present():
    assert available_chambers(LA_MAP, [1, 7, 11]) == ["LA"]
    assert available_chambers(LA_MAP, [1, 2]) == ["LA", "RA"]
    assert available_chambers(None, [1, 2]) == []


def test_a_shared_tag_is_split_between_the_chambers():
    # A bar: LA body at one end, RA body at the other, a shared tag between.
    # Not an atrium, so the tags around it decide: each shared tet goes to
    # the body nearer along the bar.
    bar = pv.ImageData(dimensions=(21, 3, 3)).cast_to_unstructured_grid().triangulate()
    x = np.asarray(bar.cell_centers().points)[:, 0]
    bar.cell_data["elemTag"] = np.where(x < 3, 1, np.where(x > 17, 2, 26))
    names = {1: "LA body", 2: "RA body", 26: "BB"}
    ex = extract_chambers(bar, names, ["LA"])
    kept_x = np.asarray(ex.grid.cell_centers().points)[:, 0]
    assert kept_x.max() < 10.5 and kept_x.max() > 9
    kept, dropped = ex.split[26]
    assert kept > 0 and dropped > 0
    assert "split" in ex.summary()


# ---------------------------------------------------------------------
# Labelling
# ---------------------------------------------------------------------
def test_recorded_rims_split_endo_and_epi():
    ex = extract_chambers(_shell(), LA_MAP, ["LA"])
    labels = label_atria(ex.grid, tag_map.read(ex.grid))
    assert labels.method == "tags"
    assert labels.naming_confident
    assert np.allclose(_radius_of(labels, ex.grid, EPI), R_OUT, atol=0.3)
    assert np.allclose(_radius_of(labels, ex.grid, LA_ENDO), R_IN, atol=0.3)
    rim = _radius_of(labels, ex.grid, BASE)
    assert len(rim) and rim.min() > R_IN - 0.3 and rim.max() < R_OUT + 0.3
    assert labels.areas[RA_ENDO] == 0.0
    assert "RA endocardium" not in labels.summary()


def test_the_labels_survive_the_node_mask():
    ex = extract_chambers(_shell(), LA_MAP, ["LA"])
    labels = label_atria(ex.grid, tag_map.read(ex.grid))
    mask = node_label_mask(ex.grid, labels)
    surface = _as_boundary(ex.grid)
    back = face_labels_from_mask(
        surface, mask[np.asarray(surface.point_data["vtkOriginalPointIds"])])
    assert np.array_equal(back, labels.face_labels)


def test_openings_are_found_from_the_shape_without_recorded_rims():
    ex = extract_chambers(_shell(), LA_MAP, ["LA"])
    bare = ex.grid.copy()
    del bare.point_data[OPENING_FIELD]
    labels = label_atria(bare, tag_map.read(bare),
                         AtrialOptions(voxel_mm=1.0, closing_retry_mm=()))
    assert labels.method == "openings"
    assert len(labels.openings) == 2
    epi = _radius_of(labels, bare, EPI)
    endo = _radius_of(labels, bare, LA_ENDO)
    assert np.median(epi) > R_OUT - 0.5 and np.median(endo) < R_IN + 0.5


def test_two_atria_are_named_from_their_tags():
    left = _shell()
    right = _shell(centre=(60.0, 0.0, 0.0), body=2,
                   plugs=((16, (0, 0, 1)), (8, (-1, 0, 0))))
    both = left.merge(right)
    ex = extract_chambers(both, LA_MAP, ["LA", "RA"])
    labels = label_atria(ex.grid, tag_map.read(ex.grid))
    assert labels.naming_confident
    assert np.allclose(_radius_of(labels, ex.grid, LA_ENDO), R_IN, atol=0.3)
    assert np.allclose(
        _radius_of(labels, ex.grid, RA_ENDO, centre=(60.0, 0.0, 0.0)),
        R_IN, atol=0.3)


def test_without_tag_names_the_atrium_is_a_guess():
    ex = extract_chambers(_shell(), LA_MAP, ["LA"])
    grid = ex.grid.copy()
    grid.cell_data["elemTag"] = np.full(grid.n_cells, 99)
    tag_map.write(grid, None)
    labels = label_atria(grid, None)
    assert not labels.naming_confident
    # Two openings: fewer than a left atrium's, so the guess is the right.
    assert labels.areas[RA_ENDO] > 0


def test_a_lone_default_tag_is_not_evidence():
    # A mesh loaded without tags gets elemTag 1 everywhere, which is also
    # the LA body's number in ccdaf's scheme: that must not name it.
    ex = extract_chambers(_shell(), LA_MAP, ["LA"])
    grid = ex.grid.copy()
    grid.cell_data["elemTag"] = np.ones(grid.n_cells, dtype=int)
    tag_map.write(grid, None)
    assert not label_atria(grid, None).naming_confident


def test_swapping_exchanges_the_atria():
    ex = extract_chambers(_shell(), LA_MAP, ["LA"])
    labels = label_atria(ex.grid, tag_map.read(ex.grid))
    swapped = swap_atria(labels)
    assert np.array_equal(swapped.face_labels == RA_ENDO,
                          labels.face_labels == LA_ENDO)
    assert swapped.areas[RA_ENDO] == labels.areas[LA_ENDO]
    assert np.array_equal(swapped.node_sets[RA_ENDO], labels.node_sets[LA_ENDO])


def test_the_legend_calls_the_base_a_rim_on_an_atrium():
    _, _, names = _surface_label_legend(np.array([BASE, EPI, LA_ENDO]))
    assert list(names.values()) == ["rim", "epicardium", "LA endocardium"]
    _, _, names = _surface_label_legend(np.array([BASE, EPI]))
    assert names[0] == "base"


# ---------------------------------------------------------------------
# Dialogs
# ---------------------------------------------------------------------
@pytest.fixture(scope="module")
def app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_the_extract_dialog_ticks_the_atria(app):
    dlg = ChamberExtractDialog(["LV", "LA", "RA"], {**LA_MAP, 5: "LV"},
                               [1, 2, 5])
    assert dlg.chambers() == ["LA", "RA"]
    dlg._boxes["LA"].setChecked(False)
    dlg._boxes["RA"].setChecked(False)
    assert not dlg.btn_ok.isEnabled()


def test_the_labelling_dialog_labels_and_swaps(app):
    ex = extract_chambers(_shell(), LA_MAP, ["LA"])
    dlg = AtrialLabelsDialog(ex.grid, tag_map.read(ex.grid), run_now=False)
    assert not dlg.cmb_unit.isEnabled()          # rims recorded: no voxels
    dlg.run()
    assert dlg.result() is not None and dlg.btn_ok.isEnabled()
    dlg.btn_swap.click()
    assert dlg.result().areas[RA_ENDO] > 0
