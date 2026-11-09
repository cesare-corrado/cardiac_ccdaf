"""
test_uvc.py
===========
Universal ventricular coordinates, and their window.

The test heart is the labelled cube of the fibre tests: two blind holes
from the top, the LV hole inside tag-1 elements and the RV hole against
the tag boundary, so part of its wall is LV tissue: a septum.

The contract:

* the action needs labels and exactly two element tags, and says why not;
* the LV tag is the one that touches the LV endocardium;
* the RV septum is the RV-endocardial faces of LV elements, and with the
  free wall it makes up the RV endocardium;
* the surfaces, built from candidate faces only, are node for node those
  of the full face table;
* labels with the epicardium inside an endocardium are refused, not
  turned into plausible-looking coordinates;
* nu follows the tags, z runs 0 at the apex to 1 at the base, rho 0 on
  the endocardium to 1 outside, phi stays within [-pi, pi], all four are
  written as point arrays;
* a node on an endocardium and the outer wall at once is held by neither
  for rho, takes a value from the tissue around it, and is reported;
* the automatic RV apex is the RV-endocardial node farthest from its base
  rim along the RV endocardium, the LV apex the shared fibre location, and
  a run without apices uses exactly those;
* a picked apex is used instead of the automatic one;
* the fibres put their automatic apex at the epicardial end of the UVC's
  LV apex;
* the window opens with both apices found and marked, runs with exactly
  the apices shown, can pick either one, and goes back to automatic.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pytest
from PyQt5 import QtWidgets

from ccdaf.core import uvc
from ccdaf.core import ventricular_fibres as vf
from ccdaf.core.surface_labels import (
    BASE, EPI, LABEL_MASK_FIELD, LV_ENDO, MASK_BITS, RV_ENDO,
)
from test_ventricular_fibres import SIZE, _labelled


@pytest.fixture()
def grid():
    return _labelled()


# ---------------------------------------------------------------------
# Inputs and surfaces
# ---------------------------------------------------------------------
def test_a_labelled_tagged_volume_is_available(grid):
    ok, why = uvc.availability(grid)
    assert ok, why


def test_missing_labels_or_tags_are_refused_with_a_reason(grid):
    no_labels = grid.copy()
    del no_labels.point_data[LABEL_MASK_FIELD]
    assert "Label the surfaces" in uvc.availability(no_labels)[1]
    no_tags = grid.copy()
    del no_tags.cell_data["elemTag"]
    assert "element tags" in uvc.availability(no_tags)[1]
    three = grid.copy()
    tags = np.asarray(three.cell_data["elemTag"]).copy()
    tags[:5] = 7
    three.cell_data["elemTag"] = tags
    assert "exactly two" in uvc.availability(three)[1]


def test_the_lv_tag_touches_the_lv_endocardium(grid):
    points, tets, mask, tags = uvc.inputs(grid)
    assert uvc.lv_rv_tags(tets, tags, mask) == (1, 2)


def test_the_septum_is_the_rv_endocardium_on_lv_elements(grid):
    points, tets, mask, tags = uvc.inputs(grid)
    s = uvc.surfaces(tets, mask, tags == 1)
    assert len(s.rv_septum) and len(s.rv_free_endo) and len(s.interface)
    assert np.isin(s.rv_septum, s.rv_endo).all()
    assert np.array_equal(np.union1d(s.rv_septum, s.rv_free_endo), s.rv_endo)
    # Septal nodes belong to LV elements.
    assert s.lv_nodes[s.rv_septum].all()


def _reference_surfaces(tets, mask, is_lv):
    """The surfaces from the full face table: what the candidate-face
    construction must reproduce."""
    from ccdaf.core.volume_mesh import face_table
    uniq, inv, counts, owner = face_table(tets)
    lv_count = np.bincount(inv, weights=is_lv[owner].astype(float), minlength=len(uniq))
    one = np.empty(len(uniq), dtype=np.int64)
    one[inv] = owner
    faces = uniq[counts == 1]
    face_lv = is_lv[one[counts == 1]]
    fm = mask[faces[:, 0]] & mask[faces[:, 1]] & mask[faces[:, 2]]

    def of(bit):
        return np.unique(faces[(fm & MASK_BITS[bit]) != 0])

    rv = (fm & MASK_BITS[RV_ENDO]) != 0
    lvb = uniq[lv_count == 1]
    lm = mask[lvb[:, 0]] & mask[lvb[:, 1]] & mask[lvb[:, 2]]
    return {"epi": of(EPI), "lv_endo": of(LV_ENDO), "rv_endo": of(RV_ENDO),
            "rv_septum": np.unique(faces[rv & face_lv]),
            "rv_free_endo": np.unique(faces[rv & ~face_lv]),
            "lv_outer": np.unique(lvb[(lm & (MASK_BITS[LV_ENDO] | MASK_BITS[BASE])) == 0]),
            "interface": np.unique(uniq[(counts == 2) & (lv_count == 1)])}


@pytest.mark.parametrize("n", [9, 13])
def test_candidate_faces_give_the_full_table_surfaces(n):
    points, tets, mask, tags = uvc.inputs(_labelled(n))
    lean = uvc.surfaces(tets, mask, tags == 1)
    ref = _reference_surfaces(tets, mask, tags == 1)
    for key, nodes in ref.items():
        assert np.array_equal(getattr(lean, key), nodes), key


def test_swapped_labels_are_refused(grid):
    points, tets, mask, tags = uvc.inputs(grid)
    swapped = mask.copy()
    epi, rv = (mask & MASK_BITS[EPI]) != 0, (mask & MASK_BITS[RV_ENDO]) != 0
    swapped &= ~(MASK_BITS[EPI] | MASK_BITS[RV_ENDO])
    swapped[epi] |= MASK_BITS[RV_ENDO]
    swapped[rv] |= MASK_BITS[EPI]
    with pytest.raises(ValueError, match="swapped"):
        uvc.compute(points, tets, swapped, tags)


# ---------------------------------------------------------------------
# The coordinates
# ---------------------------------------------------------------------
def test_the_four_coordinates(grid):
    points, tets, mask, tags = uvc.inputs(grid)
    run = uvc.compute(points, tets, mask, tags)
    f = run.fields
    s = uvc.surfaces(tets, mask, tags == 1)
    assert np.array_equal(f["ven"], np.where(s.lv_nodes, -1.0, 1.0))
    for key in ("z", "rho"):
        assert np.nanmin(f[key]) >= 0.0 and np.nanmax(f[key]) <= 1.0
    assert np.all(np.abs(f["phi"]) <= np.pi + 1e-12)
    assert not any(np.isnan(v).any() for v in f.values())
    lv = s.lv_nodes
    assert np.allclose(f["z"][run.lv_apex.nodes], 0.0)
    assert np.allclose(f["z"][s.base], 1.0)
    assert np.allclose(f["rho"][np.intersect1d(s.lv_endo, np.where(lv)[0])], 0.0)
    assert np.allclose(f["rho"][np.setdiff1d(s.rv_free_endo, np.where(lv)[0])], 0.0)
    assert "LV tag 1, RV tag 2" in run.details()
    uvc.write(grid, run)
    for name in uvc.written_names():
        assert grid.point_data[name].shape == (grid.n_points,)


def test_a_pinch_node_is_released_for_rho(grid):
    """A node on the LV endocardium and the LV outer wall at once, or on
    the RV free wall and the epicardium, is in neither rho source."""
    points, tets, mask, tags = uvc.inputs(grid)
    s = uvc.surfaces(tets, mask, tags == 1)
    lv_pinch = int(s.lv_endo[s.lv_nodes[s.lv_endo]][0])
    rv_pinch = int(s.rv_free_endo[~s.lv_nodes[s.rv_free_endo]][0])
    s.lv_outer = np.union1d(s.lv_outer, [lv_pinch])
    s.epi = np.union1d(s.epi, [rv_pinch])
    sources, released = uvc.transmural_sources(s)
    assert sorted(released.tolist()) == sorted([lv_pinch, rv_pinch])
    assert lv_pinch not in sources["lv endo"] and lv_pinch not in sources["lv outer"]
    assert rv_pinch not in sources["rv free endo"] and rv_pinch not in sources["epi"]
    # Everything else is untouched.
    assert len(sources["lv endo"]) == len(s.lv_endo) - 1
    # A node of the other ventricle on the same pair is left alone: it
    # takes the other ventricle's rho.
    lv_node_on_rv_pair = int(s.rv_septum[0])
    s.rv_free_endo = np.union1d(s.rv_free_endo, [lv_node_on_rv_pair])
    s.epi = np.union1d(s.epi, [lv_node_on_rv_pair])
    _sources, released = uvc.transmural_sources(s)
    assert lv_node_on_rv_pair not in released


def test_a_clean_heart_releases_nothing(grid):
    run = uvc.compute(*uvc.inputs(grid))
    assert len(run.released) == 0 and "Released" not in run.details()


def test_the_automatic_apices(grid):
    points, tets, mask, tags = uvc.inputs(grid)
    lv, rv = uvc.automatic_apices(points, tets, mask)
    assert lv.automatic and rv.automatic
    rv_endo = np.where(mask & MASK_BITS[RV_ENDO])[0]
    base = np.where(mask & MASK_BITS[BASE])[0]
    assert rv.endo in rv_endo and rv.endo not in base
    # The RV hole bottom (z = 1) is the farthest from the rim at z = SIZE.
    assert np.isclose(points[rv.endo, 2], 1.0)
    sets = vf.surface_sets(mask)
    assert lv.endo == uvc.lv_apex_ends(points, sets[LV_ENDO], sets[EPI], sets[BASE])[0]
    auto = uvc.compute(points, tets, mask, tags)
    given = uvc.compute(points, tets, mask, tags, lv_apex=lv, rv_apex=rv)
    for key in uvc.UVC_FIELDS:
        assert np.allclose(auto.fields[key], given.fields[key])


def test_a_picked_apex_is_used(grid):
    points, tets, mask, tags = uvc.inputs(grid)
    picked = uvc.apex_at(points, tets, mask, [1.5, 3.5, 2.0], "lv")
    assert not picked.automatic
    assert picked.endo in np.where(mask & MASK_BITS[LV_ENDO])[0]
    run = uvc.compute(points, tets, mask, tags, lv_apex=picked)
    assert run.lv_apex is picked and run.rv_apex.automatic
    assert np.allclose(run.fields["z"][picked.nodes], 0.0)


def test_the_fibres_share_the_lv_apex(grid):
    points, tets, mask = vf.inputs(grid)
    sets = vf.surface_sets(mask)
    _endo, epi = uvc.lv_apex_ends(points, sets[LV_ENDO], sets[EPI], sets[BASE])
    assert vf.find_apex(points, tets, mask).node == epi


# ---------------------------------------------------------------------
# The window
# ---------------------------------------------------------------------
@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def test_the_window_runs_in_the_background_and_reports(qapp, grid):
    from ccdaf.gui.uvc_dialog import UvcDialog, UvcWorker
    dlg = UvcDialog(grid)
    marks = []
    dlg.apices_changed.connect(lambda lv, rv: marks.append((lv, rv)))
    qapp.processEvents()                                  # the apices are found
    assert "automatic:" in dlg.lbl_lv.text() and "automatic:" in dlg.lbl_rv.text()
    assert marks and all(a is not None for a in marks[-1])
    assert dlg.btn_run.isEnabled()
    worker = UvcWorker(*dlg.inputs(), *dlg.apices())
    got, failed = [], []
    worker.done.connect(got.append)
    worker.failed.connect(failed.append)
    worker.start()
    start = time.time()
    while worker.isRunning() and time.time() - start < 60:
        qapp.processEvents()
        time.sleep(0.01)
    qapp.processEvents()
    assert not failed and got
    assert got[0].lv_apex is dlg.apices()[0] and got[0].rv_apex is dlg.apices()[1]
    dlg.show_run(got[0])
    assert "UVC for" in dlg.report.toPlainText()
    assert dlg.matches(grid)


def test_the_window_picks_either_apex_and_goes_back_to_automatic(qapp, grid):
    from ccdaf.gui.uvc_dialog import UvcDialog
    dlg = UvcDialog(grid)
    qapp.processEvents()
    auto_lv, auto_rv = dlg.apices()
    asked = []
    dlg.pick_requested.connect(asked.append)
    dlg.btn_pick_rv.setChecked(True)
    assert asked[-1] == "rv" and dlg.picking() == "rv"
    dlg.set_apex_at([4.5, 3.5, 2.0])
    assert "picked" in dlg.lbl_rv.text() and "automatic" in dlg.lbl_lv.text()
    assert dlg.apices()[1] is not auto_rv
    dlg.btn_pick_lv.setChecked(True)                   # switches, not adds
    assert dlg.picking() == "lv" and not dlg.btn_pick_rv.isChecked()
    runs = []
    dlg.run_requested.connect(lambda lv, rv: runs.append((lv, rv)))
    dlg.btn_run.click()
    assert runs and runs[-1][0] is auto_lv and not runs[-1][1].automatic
    assert dlg.picking() == "" and asked[-1] == ""
    dlg.use_automatic()
    assert "automatic" in dlg.lbl_rv.text() and dlg.apices() == (auto_lv, auto_rv)


def test_controls_have_tooltips(qapp, grid):
    from ccdaf.gui.uvc_dialog import UvcDialog
    dlg = UvcDialog(grid)
    for w in (dlg.btn_pick_lv, dlg.btn_pick_rv, dlg.btn_auto, dlg.btn_run):
        assert w.toolTip()


def test_size_is_what_the_fixture_says():
    assert SIZE == 8.0
