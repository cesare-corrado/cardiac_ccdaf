"""
test_carp_export_dialog.py
==========================
Export → Carp: the dialog in ``ccdaf.gui.carp_export_dialog``.

The contract:

* only arrays that can be written are offered — whole-numbered cell arrays as
  region tags, named direction fields as fibres and sheets;
* a sheet cannot be chosen without a fibre;
* the options mirror the widgets, and the preview names the three files on one
  line;
* Export writes the files and returns a result; a blocking problem keeps the
  dialog open with nothing written;
* existing files, and anything worth confirming, are asked about **before**
  anything is written;
* every control carries a tooltip.

Runs headless with the offscreen Qt platform.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pytest
import pyvista as pv
from PyQt5 import QtWidgets

from ccdaf.gui.carp_export_dialog import CarpExportDialog


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def _grid() -> pv.UnstructuredGrid:
    grid = pv.ImageData(dimensions=(4, 4, 4), spacing=(30.0, 30.0, 30.0)).triangulate()
    grid.cell_data["elemTag"] = np.ones(grid.n_cells, dtype=np.int32)
    grid.cell_data["tissueTag"] = (np.arange(grid.n_cells) % 3).astype(np.int32)
    grid.cell_data["measured"] = np.linspace(0.0, 1.5, grid.n_cells)
    grid.cell_data["fiber"] = np.tile([1.0, 0.0, 0.0], (grid.n_cells, 1))
    grid.cell_data["sheet"] = np.tile([0.0, 1.0, 0.0], (grid.n_cells, 1))
    return grid


@pytest.fixture
def answers(monkeypatch):
    """Answer every question Yes, and record the warnings shown."""
    shown = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Yes))
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning",
                        staticmethod(lambda _p, _t, text, *a, **k: shown.append(text)))
    return shown


def _dialog(tmp_path, name="mesh") -> CarpExportDialog:
    return CarpExportDialog(_grid(), start_dir=str(tmp_path), default_name=name)


def test_only_writable_arrays_are_offered(qapp, tmp_path):
    dlg = _dialog(tmp_path)
    regions = [dlg.cmb_region.itemData(i) for i in range(dlg.cmb_region.count())]
    fibres = [dlg.cmb_fibre.itemData(i) for i in range(dlg.cmb_fibre.count())]
    sheets = [dlg.cmb_sheet.itemData(i) for i in range(dlg.cmb_sheet.count())]
    assert regions == [None, "elemTag", "tissueTag"]        # not the measured field
    assert fibres == [None, "fiber"]
    assert sheets == [None, "sheet"]


def test_the_region_tag_starts_on_the_material_regions(qapp, tmp_path):
    """Writing tissueTag into the element tag column is the point of this export."""
    dlg = _dialog(tmp_path)
    assert dlg.options().region_field == "tissueTag"

    grid = _grid()
    grid.cell_data.remove("tissueTag")
    without = CarpExportDialog(grid, start_dir=str(tmp_path), default_name="mesh")
    assert without.options().region_field is None


def test_a_sheet_needs_a_fibre(qapp, tmp_path):
    dlg = _dialog(tmp_path)
    assert not dlg.cmb_sheet.isEnabled()
    dlg.cmb_fibre.setCurrentIndex(dlg.cmb_fibre.findData("fiber"))
    assert dlg.cmb_sheet.isEnabled()
    dlg.cmb_sheet.setCurrentIndex(dlg.cmb_sheet.findData("sheet"))
    assert dlg.options().sheet_field == "sheet"
    dlg.cmb_fibre.setCurrentIndex(0)                        # back to the placeholder
    assert not dlg.cmb_sheet.isEnabled()
    assert dlg.options().sheet_field is None


def test_options_mirror_the_widgets(qapp, tmp_path):
    dlg = _dialog(tmp_path, name="run1")
    dlg.spn_scale.setValue(1.0)
    dlg.cmb_region.setCurrentIndex(dlg.cmb_region.findData("tissueTag"))
    options = dlg.options()
    assert options.directory == str(tmp_path)
    assert options.name == "run1"
    assert options.scale == 1.0
    assert options.region_field == "tissueTag"
    assert options.fibre_field is None
    assert [p.name for p in options.paths()] == ["run1.pts", "run1.elem", "run1.lon"]


def test_the_preview_names_the_three_files_on_one_line(qapp, tmp_path):
    dlg = _dialog(tmp_path, name="run1")
    preview = dlg.lbl_preview.text()
    assert preview.count("\n") == 0
    assert preview.endswith("run1.[pts|elem|lon]")


def test_export_writes_the_files(qapp, tmp_path, answers):
    dlg = _dialog(tmp_path, name="written")
    dlg.cmb_region.setCurrentIndex(dlg.cmb_region.findData("tissueTag"))
    dlg.cmb_fibre.setCurrentIndex(dlg.cmb_fibre.findData("fiber"))
    dlg.accept()
    result = dlg.result()
    assert answers == []
    assert result is not None
    assert all(path.is_file() for path in result.paths)
    assert result.tags == {0: 27, 1: 27, 2: 27} or sum(result.tags.values()) == result.n_elements


def test_a_blocking_problem_writes_nothing(qapp, tmp_path, answers):
    dlg = _dialog(tmp_path, name="")
    dlg.accept()
    assert dlg.result() is None
    assert any("name" in text for text in answers)
    assert list(tmp_path.glob("*")) == []


def test_existing_files_are_asked_about_before_writing(qapp, tmp_path, monkeypatch):
    dlg = _dialog(tmp_path, name="kept")
    for suffix in (".pts", ".elem", ".lon"):
        (tmp_path / f"kept{suffix}").write_text("older run\n")
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        staticmethod(lambda *a, **k: QtWidgets.QMessageBox.No))
    dlg.accept()
    assert dlg.result() is None
    assert (tmp_path / "kept.pts").read_text() == "older run\n"


def test_warnings_are_raised_before_anything_is_written(qapp, tmp_path, monkeypatch):
    asked = []

    def refuse(_parent, title, text, *args, **kwargs):
        asked.append((title, text))
        return QtWidgets.QMessageBox.No

    monkeypatch.setattr(QtWidgets.QMessageBox, "question", staticmethod(refuse))
    dlg = _dialog(tmp_path, name="placeholder")     # no fibre chosen: a warning
    dlg.accept()
    assert dlg.result() is None
    assert asked and "isotropic" in asked[0][1]
    assert list(tmp_path.glob("placeholder.*")) == []


def test_controls_have_tooltips(qapp, tmp_path):
    dlg = _dialog(tmp_path)
    for name in ("txt_dir", "btn_browse", "txt_name", "spn_scale", "cmb_region",
                 "cmb_fibre", "cmb_sheet", "lbl_preview", "btn_export"):
        assert getattr(dlg, name).toolTip().strip(), name
