"""
test_tag_map.py
===============
What each elemTag number means, and the legends that read it.

* the map survives a VTK file (ASCII and binary), a surface extraction
  and a pickle bundle;
* the proposal names an atrial tagging and a labelled ventricular volume,
  and nothing else;
* the Regions legend shows the names ("LV", "RV") in their own colours,
  while a map that agrees with the atrial scheme keeps the atrial legend;
* the surfaceLabel legend names the surfaces instead of numbering them;
* the dialog returns what was typed, blank rows left out;
* known names bring a chamber and a role, which actions read;
* a cohort's {name: tag} file translates into ccdaf's names, and a map
  saved to a file reads back unchanged.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pytest
import pyvista as pv
from PyQt5 import QtWidgets

from ccdaf.app.ccdaf import (
    SURFACE_LABEL_COLORS, _region_legend,
    _surface_label_legend,
)
from ccdaf.core import atrial_labels as al
from ccdaf.core import tag_map
from ccdaf.core.eam_export import export_binary
from ccdaf.core.eam_loader import read_bundle
from ccdaf.core.surface_labels import BASE, EPI, LV_ENDO, RV_ENDO, UNLABELLED
from ccdaf.gui.tag_map_dialog import TagMapDialog
from ccdaf.io.vtkfunctions import writevtk

FOUR_CHAMBER = {1: "LV", 2: "RV", 3: "LA body", 4: "RA body", 7: "Cap / valve"}


def _names(mapping):
    return {k: v.name for k, v in mapping.items()}


def _tagged_sphere(tags=(1, 2)):
    mesh = pv.Sphere(theta_resolution=8, phi_resolution=8)
    mesh.cell_data["elemTag"] = np.resize(np.asarray(tags), mesh.n_cells)
    return mesh


# ---------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------
def test_encode_decode_round_trip():
    assert tag_map.decode(tag_map.encode(FOUR_CHAMBER)) == tag_map.normalise(FOUR_CHAMBER)


def test_blank_names_are_dropped_on_decode():
    assert _names(tag_map.decode('{"1": "LV", "2": " "}')) == {1: "LV"}


def test_malformed_text_reads_as_no_map():
    mesh = _tagged_sphere()
    mesh.field_data[tag_map.FIELD] = np.array(["not json"])
    assert tag_map.read(mesh) is None


@pytest.mark.parametrize("binary", [False, True])
def test_the_map_survives_a_vtk_file(tmp_path, binary):
    mesh = _tagged_sphere()
    tag_map.write(mesh, FOUR_CHAMBER)
    path = str(tmp_path / "m.vtk")
    writevtk(mesh, path, binary=binary)
    assert tag_map.read(pv.read(path)) == tag_map.normalise(FOUR_CHAMBER)


def test_the_map_survives_a_volume_file_and_its_boundary(tmp_path):
    grid = pv.ImageData(dimensions=(3, 3, 3)).cast_to_unstructured_grid().triangulate()
    grid.cell_data["elemTag"] = np.ones(grid.n_cells, dtype=int)
    tag_map.write(grid, {1: "LA body"})
    path = str(tmp_path / "v.vtk")
    writevtk(grid, path)
    back = pv.read(path)
    assert _names(tag_map.read(back)) == {1: "LA body"}
    assert _names(tag_map.read(
        back.extract_surface(algorithm="dataset_surface"))) == {1: "LA body"}


def test_writing_nothing_removes_the_map():
    mesh = _tagged_sphere()
    tag_map.write(mesh, FOUR_CHAMBER)
    tag_map.write(mesh, None)
    assert tag_map.read(mesh) is None


def test_the_map_survives_a_bundle(tmp_path):
    mesh = _tagged_sphere()
    path = str(tmp_path / "b.pkl")
    export_binary(path, mesh, include_elem_tag=True, tag_map=FOUR_CHAMBER)
    back, *_ = read_bundle(path)
    assert tag_map.read(back) == tag_map.normalise(FOUR_CHAMBER)


def test_a_bundle_keeps_chamber_and_role(tmp_path):
    mesh = _tagged_sphere()
    m = {1: tag_map.TagEntry("LSPV ring", "LA", tag_map.APPENDAGE)}
    path = str(tmp_path / "b.pkl")
    export_binary(path, mesh, include_elem_tag=True, tag_map=m)
    back, *_ = read_bundle(path)
    assert tag_map.read(back) == m


def test_a_bundle_without_a_map_reads_as_none(tmp_path):
    mesh = _tagged_sphere()
    path = str(tmp_path / "b.pkl")
    export_binary(path, mesh, include_elem_tag=True)
    back, *_ = read_bundle(path)
    assert tag_map.read(back) is None


# ---------------------------------------------------------------------
# Proposal
# ---------------------------------------------------------------------
def test_an_atrial_tagging_names_itself():
    assert _names(tag_map.propose([1, 11, 19])) == {1: "LA body", 11: "LSPV",
                                                    19: "LAA"}


def test_a_labelled_ventricular_volume_gets_lv_and_rv():
    assert _names(tag_map.propose([1, 2], ventricles=(2, 1))) == {2: "LV", 1: "RV"}


def test_a_bare_one_two_is_left_unnamed():
    # Two atrial bodies or two ventricles: nothing says which.
    assert tag_map.propose([1, 2]) == {}


def test_ventricles_that_do_not_cover_the_tags_are_ignored():
    assert tag_map.propose([1, 2, 3], ventricles=(1, 2)) == {}


def test_every_atrial_name_is_offered():
    for name in al.NAMES.values():
        assert name in tag_map.MEANINGS
    assert tag_map.MEANINGS[:2] == (tag_map.LV, tag_map.RV)


# ---------------------------------------------------------------------
# Legends
# ---------------------------------------------------------------------
def test_a_ventricular_map_names_and_colours_the_legend():
    ordered, colours, names = _region_legend(np.array([1, 2]),
                                             mapping={1: "LV", 2: "RV"})
    assert ordered == [1, 2]
    assert list(names.values()) == ["LV", "RV"]
    assert colours == [tag_map.VENTRICLE_COLORS["LV"],
                       tag_map.VENTRICLE_COLORS["RV"]]


def test_a_four_chamber_map_uses_atrial_colours_for_atrial_meanings():
    ordered, colours, names = _region_legend(np.array([1, 2, 3, 4, 7]),
                                             mapping=FOUR_CHAMBER)
    assert [names[i] for i in range(5)] == ["LV", "RV", "LA body", "RA body",
                                            "Cap / valve"]
    assert colours[2] == al.COLORS[al.LA.body]
    assert colours[3] == al.COLORS[al.RA.body]
    assert colours[4] == tag_map.VOCABULARY[tag_map.CAP_NAME].color


def test_a_map_that_agrees_with_the_atrial_scheme_keeps_the_atrial_legend():
    tags = np.array([1, 11])
    plain = _region_legend(tags)
    mapped = _region_legend(tags, mapping={1: "LA body", 11: "LSPV"})
    assert mapped == plain
    assert len(plain[0]) == len(al.LA.values)


def test_a_map_that_disagrees_overrides_the_atrial_scheme():
    _, _, names = _region_legend(np.array([1, 11]),
                                 mapping={1: "LV", 11: "Cap / valve"})
    assert list(names.values()) == ["LV", "Cap / valve"]


def test_a_derived_field_ignores_the_map():
    _, _, names = _region_legend(np.array([1, 2]), anatomical=False,
                                 mapping={1: "LV", 2: "RV"})
    assert list(names.values()) == ["1", "2"]


def test_unmapped_tags_keep_their_numbers():
    _, _, names = _region_legend(np.array([1, 2, 9]), mapping={1: "LV"})
    assert list(names.values()) == ["LV", "2", "9"]


def test_the_surface_label_legend_names_the_surfaces():
    ordered, colours, names = _surface_label_legend(
        np.array([BASE, EPI, LV_ENDO, RV_ENDO, EPI]))
    assert ordered == [BASE, EPI, LV_ENDO, RV_ENDO]
    assert list(names.values()) == ["base", "epicardium", "LV endocardium",
                                    "RV endocardium"]
    assert colours == [SURFACE_LABEL_COLORS[t] for t in ordered]


def test_the_surface_label_legend_names_unlabelled_faces():
    ordered, _, names = _surface_label_legend(np.array([UNLABELLED, EPI]))
    assert names[0] == "unlabelled"


# ---------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------
@pytest.fixture(scope="module")
def app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_the_dialog_starts_from_the_current_map(app):
    dlg = TagMapDialog({1: 10, 2: 5}, current={1: "LV", 2: "RV"},
                       proposed={1: "RV", 2: "LV"})
    assert _names(dlg.mapping()) == {1: "LV", 2: "RV"}


def test_the_dialog_starts_from_the_proposal_without_a_map(app):
    dlg = TagMapDialog({1: 10, 2: 5}, current=None, proposed={1: "LV"})
    assert _names(dlg.mapping()) == {1: "LV"}


def test_the_dialog_takes_free_text_and_drops_blank_rows(app):
    dlg = TagMapDialog({1: 10, 2: 5, 3: 1}, current=None)
    dlg._names[1].setCurrentText("Aorta wall")
    dlg._names[2].setCurrentText("RV")
    assert dlg.mapping() == {1: tag_map.TagEntry("Aorta wall"),
                             2: tag_map.TagEntry("RV", "RV", tag_map.WALL)}


def test_proposed_names_refill_the_boxes(app):
    dlg = TagMapDialog({1: 10, 2: 5}, current={1: "X", 2: "Y"},
                       proposed={1: "LV", 2: "RV"})
    dlg.reset_button.click()
    assert _names(dlg.mapping()) == {1: "LV", 2: "RV"}


def test_picking_a_known_name_fills_chamber_and_role(app):
    dlg = TagMapDialog({21: 10}, current=None)
    dlg._names[21].setCurrentText("LSPV ring")
    assert dlg.mapping() == {21: tag_map.TagEntry("LSPV ring", "LA", tag_map.VEIN)}


def test_the_chamber_and_role_can_be_changed(app):
    dlg = TagMapDialog({18: 10}, current={18: "LSPV ring"})
    dlg._roles[18].setCurrentText(tag_map.APPENDAGE)
    assert dlg.mapping()[18].role == tag_map.APPENDAGE


def test_the_dialog_loads_and_saves_a_file(app, tmp_path):
    path = tmp_path / "names.json"
    first = TagMapDialog({1: 10, 3: 5}, current={1: "LV", 3: "LA body"})
    first.save_file(str(path))
    second = TagMapDialog({1: 7, 3: 2, 9: 1}, current={9: "AV"})
    second.load_file(str(path))
    assert _names(second.mapping()) == {1: "LV", 3: "LA body", 9: "AV"}
    assert "2 of 3" in second.status.text()


# ---------------------------------------------------------------------
# Vocabulary, files and lookups
# ---------------------------------------------------------------------
COHORT_FILE = {"LV": 1, "RV": 2, "LA": 3, "RA": 4, "Ao": 5, "PArt": 6,
               "MV": 7, "TV": 8, "AV": 9, "PV": 10, "LSPV": 11, "LAA": 15,
               "SVC": 16, "LAA_ring": 18, "LSPV_ring": 21, "FEC_LV": 25,
               "FEC": 25, "BB": 26, "AV_plane": 27}


def test_a_known_name_brings_its_chamber_and_role():
    assert tag_map.entry("FEC_LV") == tag_map.TagEntry("FEC_LV", "LV", tag_map.LAYER)
    assert tag_map.entry("Septum") == tag_map.TagEntry("Septum")


def test_a_cohort_file_translates_into_ccdaf_names(tmp_path):
    path = tmp_path / "tags_lvrv.json"
    path.write_text(json.dumps(COHORT_FILE))
    m = tag_map.load_file(str(path))
    assert m[3].name == "LA body"
    assert m[10].name == "Pulmonary valve"
    # The cohort's bare vein names are the closing caps.
    assert m[11] == tag_map.TagEntry("LSPV cap", "LA", tag_map.CAP)
    assert m[21] == tag_map.TagEntry("LSPV ring", "LA", tag_map.VEIN)
    # FEC_LV and FEC share 25: the first wins.
    assert m[25].name == "FEC_LV"


def test_a_saved_file_reads_back_with_its_changes(tmp_path):
    m = tag_map.normalise(FOUR_CHAMBER)
    m[18] = tag_map.TagEntry("LSPV ring", "LA", tag_map.APPENDAGE)
    path = str(tmp_path / "m.json")
    tag_map.save_file(path, m)
    assert tag_map.load_file(path) == m


def test_a_names_only_file_reads():
    assert tag_map.from_json({"1": "LV"}) == {1: tag_map.entry("LV")}


def test_tags_of_a_chamber_keep_tissue_only():
    m = tag_map.from_json(COHORT_FILE)
    assert tag_map.tags_of(m, "LA") == [3, 18, 21, 26]
    assert tag_map.tags_of(m, "LA", roles=[tag_map.CAP]) == [11, 15]
    assert tag_map.tags_of(m, "LV") == [1, 25]


def test_the_atrial_region_of_each_tag():
    region = tag_map.atrial_region
    assert region("LSPV ring") == "LSPV"
    assert region("LAA ring") == "LAA"
    assert region("LA body") == "LA body"
    assert region(tag_map.TagEntry("LSPV ring", "LA", tag_map.APPENDAGE)) == "LAA"
    assert region("LSPV cap") == ""
    assert region("BB") == ""          # both atria: split it first


def test_every_name_has_its_own_colour_except_rings():
    # A ring is the vein it was cut from, so it takes the vein's colour.
    colours = [m.color for n, m in tag_map.VOCABULARY.items()
               if not n.endswith(" ring")]
    assert len(colours) == len(set(colours))


def test_a_long_tagging_keeps_its_colours_apart():
    _, colours, _ = _region_legend(np.arange(1, 30))
    assert len(set(colours)) == len(colours)
