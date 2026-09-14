"""
test_tissue_property_dialog.py
==============================
Actions → Assign tissue property: the dialog, and the save dialog's handling
of the array it writes.

The contract:

* only fields a tissue property can be read from are offered;
* unticking a tissue adds an ID box for it, pre-filled with its ``elemTag``
  number or, when that is taken, the smallest unused one;
* threshold rows are linked — each row's *from* is the previous row's *to*;
* changing the number of rows keeps the IDs already typed;
* Auto makes the mean and SD read-only, and Estimate fills them;
* Apply returns a ``tissueTag`` for every element, with no negative ID, and
  a blocking problem keeps the dialog open with nothing returned;
* every control carries a tooltip;
* the save dialog ticks ``tissueTag`` whenever the mesh has one.

Runs headless with the offscreen Qt platform.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pytest
import pyvista as pv
from PyQt5 import QtWidgets

from ccdaf.gui.save_mesh_dialog import SaveMeshDialog
from ccdaf.gui.tissue_property_dialog import TissuePropertyDialog


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def _ventricles() -> pv.UnstructuredGrid:
    """An LV (elemTag 1) with a bright corner, and a masked RV (elemTag 2)."""
    grid = pv.ImageData(dimensions=(10, 10, 10)).triangulate()
    centres = np.asarray(grid.cell_centers().points)
    grid.cell_data["elemTag"] = np.where(centres[:, 0] < 3.0, 2, 1).astype(np.int32)
    rng = np.random.default_rng(3)
    points = np.asarray(grid.points)
    lge = 0.12 + 0.03 * rng.standard_normal(grid.n_points)
    lge[np.linalg.norm(points - 9.0, axis=1) < 3.5] += 0.5
    lge[points[:, 0] <= 3.0] = 0.0
    grid.point_data["lge"] = lge
    grid.cell_data["fiber"] = np.tile([1.0, 0.0, 0.0], (grid.n_cells, 1))
    return grid


@pytest.fixture
def answers(monkeypatch):
    """Answer every question Yes, and record every warning shown."""
    shown = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Yes))
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning",
                        staticmethod(lambda _parent, _title, text, *a, **k: shown.append(text)))
    return shown


def test_only_eligible_fields_are_offered(qapp):
    dlg = TissuePropertyDialog(_ventricles())
    offered = [dlg.cmb_field.itemData(i) for i in range(dlg.cmb_field.count())]
    assert offered == [("lge", "point")]


def test_unticking_a_tissue_adds_its_id_box(qapp):
    dlg = TissuePropertyDialog(_ventricles())
    assert dlg.grp_excluded.isHidden()
    dlg.chk_tissue[2].setChecked(False)
    assert not dlg.grp_excluded.isHidden()
    # elemTag 2 is taken by the scar row, so the RV gets the next free number.
    assert dlg.options().excluded_ids == {2: 3}
    assert dlg.options().tissue == (1,)


def test_threshold_rows_are_linked(qapp):
    dlg = TissuePropertyDialog(_ventricles())
    dlg.rad_thresholds.setChecked(True)
    dlg._rows[0]["to"].setValue(0.3)
    assert dlg._rows[1]["from"].text() == "0.3000"
    edges = dlg.options().edges
    assert edges[1] == pytest.approx(0.3) and len(edges) == 3


def test_changing_the_number_of_rows_keeps_typed_ids(qapp):
    dlg = TissuePropertyDialog(_ventricles())
    dlg._rows[0]["id"].setValue(7)
    dlg.spn_n.setValue(3)
    options = dlg.options()
    assert options.region_ids == (7, 2, 3)
    assert options.ks == (2.0, 3.0, 4.0)


def test_auto_makes_mean_and_sd_read_only(qapp):
    dlg = TissuePropertyDialog(_ventricles())
    assert dlg.spn_mean.isReadOnly() and dlg.spn_sd.isReadOnly()
    dlg.chk_auto.setChecked(False)
    assert not dlg.spn_mean.isReadOnly() and not dlg.spn_sd.isReadOnly()
    assert not dlg.btn_estimate.isEnabled()


def test_estimate_fills_mean_and_sd(qapp, answers):
    dlg = TissuePropertyDialog(_ventricles())
    dlg.chk_tissue[2].setChecked(False)
    dlg._estimate()
    assert answers == []
    assert dlg.spn_sd.value() > 0.0
    assert "Healthy" in dlg.lbl_estimate.text()


def test_apply_returns_a_complete_tissue_tag(qapp, answers):
    grid = _ventricles()
    dlg = TissuePropertyDialog(grid)
    dlg.chk_tissue[2].setChecked(False)
    dlg.accept()
    result = dlg.result()
    assert answers == []
    assert result is not None
    assert len(result.tags) == grid.n_cells
    assert result.tags.min() >= 0
    assert set(np.unique(result.tags[grid.cell_data["elemTag"] == 2])) == {3}
    assert "tissueTag" not in grid.cell_data        # writing is the caller's job


def test_a_blocking_problem_keeps_the_dialog_open(qapp, answers):
    dlg = TissuePropertyDialog(_ventricles())       # the masked RV stays ticked
    dlg.accept()
    assert dlg.result() is None
    assert any("no spread" in text for text in answers)


def test_controls_have_tooltips(qapp):
    dlg = TissuePropertyDialog(_ventricles())
    names = ["cmb_field", "lbl_range", "rad_sd", "rad_thresholds", "spn_n", "chk_auto",
             "spn_mean", "spn_sd", "btn_estimate", "btn_advanced", "spn_seed", "spn_m",
             "spn_tolerance", "spn_healthy", "btn_apply"]
    for name in names:
        assert getattr(dlg, name).toolTip().strip(), name
    for box in dlg.chk_tissue.values():
        assert box.toolTip().strip()
    for row in dlg._rows:
        for key in ("id", "k", "from", "to"):
            assert row[key].toolTip().strip(), key
    dlg.chk_tissue[2].setChecked(False)
    for spin in dlg._excluded_spins.values():
        assert spin.toolTip().strip()


def test_the_save_dialog_ticks_tissue_tag(qapp):
    dlg = SaveMeshDialog(point_fields=["LAT"], cell_fields=["elemTag", "tissueTag"])
    assert "tissueTag" in dlg.selected_fields()
    assert "LAT" not in dlg.selected_fields()
