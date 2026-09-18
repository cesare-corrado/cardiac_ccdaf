"""
test_ventricular_fibres.py
==========================
The fibre pipeline behind Actions > Generate fibres, and its window.

The contract:

* the surface sets come from the label mask, and a mesh is refused, with a
  reason, until it has all four surfaces;
* the apex is the epicardial node farthest from the base, never a base
  node, and a click becomes the nearest epicardial node;
* a node on the epicardium and an endocardium at once (a pinch point, no
  wall between them) is held by no transmural solve, and the run finishes;
* a run writes ``fiber``, ``sheet``, the four fields and a digest, and a
  second run on the same mesh, labels and apex reuses the fields;
* when the mesh is replaced, fibres are kept if nothing changed, kept and
  marked as carried if the geometry changed (a remesh interpolates them on
  purpose), and removed if the labels changed on the same geometry; fibres
  the mesh arrived with are left alone;
* the window runs, locks while running, and reports.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pytest
import pyvista as pv
from PyQt5 import QtWidgets

from ccdaf.core import ventricular_fibres as vf
from ccdaf.core.ldrb import FibreAngles
from ccdaf.core.surface_labels import (
    BASE, EPI, LABEL_MASK_FIELD, LV_ENDO, MASK_BITS, RV_ENDO, Plane,
    label_boundary, node_label_mask,
)

N, SIZE = 9, 8.0


def _labelled() -> pv.UnstructuredGrid:
    """A cube with two blind holes drilled from the top, labelled.

    The top is the base, the outside the epicardium, the hole inside one
    material the LV and the hole straddling both the RV.
    """
    grid = pv.ImageData(dimensions=(N, N, N),
                        spacing=(SIZE / (N - 1),) * 3).cast_to_unstructured_grid()
    grid = grid.triangulate()
    step = SIZE / (N - 1)
    c = np.asarray(grid.cell_centers().points)
    ix, iy, iz = (np.floor(c[:, k] / step).astype(int) for k in range(3))
    holes = ((ix == 1) | (ix == 4)) & (iy == 3) & (iz >= 1)
    out = grid.extract_cells(np.where(~holes)[0])
    c = np.asarray(out.cell_centers().points)
    out.cell_data["elemTag"] = np.where(c[:, 0] < 4 * step, 1, 2).astype(np.int32)
    labels = label_boundary(out, Plane(origin=[SIZE / 2, SIZE / 2, SIZE],
                                       normal=[0.0, 0.0, 1.0]))
    out.point_data[LABEL_MASK_FIELD] = node_label_mask(out, labels)
    return out


@pytest.fixture()
def grid():
    return _labelled()


# ---------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------
def test_a_labelled_volume_is_available(grid):
    ok, why = vf.availability(grid)
    assert ok, why


def test_an_unlabelled_volume_is_refused_with_a_reason(grid):
    del grid.point_data[LABEL_MASK_FIELD]
    ok, why = vf.availability(grid)
    assert not ok and "Label the surfaces" in why


def test_a_missing_surface_is_named(grid):
    mask = np.asarray(grid.point_data[LABEL_MASK_FIELD]).copy()
    mask &= ~MASK_BITS[RV_ENDO]
    grid.point_data[LABEL_MASK_FIELD] = mask
    ok, why = vf.availability(grid)
    assert not ok and "RV endocardium" in why


def test_a_surface_is_refused(grid):
    ok, why = vf.availability(grid.extract_surface())
    assert not ok and "volume" in why


def test_the_apex_is_the_epicardial_node_farthest_from_the_base(grid):
    points, tets, mask = vf.inputs(grid)
    apex = vf.find_apex(points, tets, mask)
    sets = vf.surface_sets(mask)
    # The base is the top (z = SIZE), so the apex lies on the bottom face.
    assert np.isclose(points[apex.node, 2], 0.0)
    assert np.isin(apex.nodes, sets[EPI]).all()
    assert not np.isin(apex.nodes, sets[BASE]).any()
    assert apex.automatic and apex.node in apex.nodes


def test_a_click_becomes_the_nearest_epicardial_node(grid):
    points, tets, mask = vf.inputs(grid)
    target = np.array([SIZE / 2, SIZE / 2, -3.0])        # below the bottom face
    apex = vf.apex_at(points, tets, mask, target)
    assert not apex.automatic
    assert np.allclose(points[apex.node], [SIZE / 2, SIZE / 2, 0.0])


def test_a_pinch_node_is_released_not_refused(grid):
    """One node on the epicardium and the LV at once: no solve can hold it
    at 1 and 0, so none holds it."""
    mask = np.asarray(grid.point_data[LABEL_MASK_FIELD]).copy()
    sets = vf.surface_sets(mask)
    pinch = int(np.setdiff1d(sets[EPI], sets[BASE])[0])
    mask[pinch] |= MASK_BITS[LV_ENDO]
    epi, lv, rv, released = vf.transmural_sets(vf.surface_sets(mask))
    assert released.tolist() == [pinch]
    assert pinch not in epi and pinch not in lv

    points, tets, _ = vf.inputs(grid)
    run = vf.generate(points, tets, mask, vf.find_apex(points, tets, mask))
    assert run.released.tolist() == [pinch]
    assert run.conservation_error() < 1e-4
    assert "Released 1 node" in run.details()


# ---------------------------------------------------------------------
# A run
# ---------------------------------------------------------------------
def test_a_run_writes_fibres_fields_and_digest(grid):
    points, tets, mask = vf.inputs(grid)
    apex = vf.find_apex(points, tets, mask)
    run = vf.generate(points, tets, mask, apex)
    assert all(r.converged for r in run.reports.values())
    assert run.conservation_error() < 1e-4
    vf.write(grid, run)
    for name in (vf.FIBRE_FIELD, vf.SHEET_FIELD):
        assert grid.cell_data[name].shape == (grid.n_cells, 3)
    for name in vf.FIELD_NAMES.values():
        assert grid.point_data[name].shape == (grid.n_points,)
    assert vf.stamp(grid) is not None
    assert vf.provenance(grid) == vf.GENERATED
    assert np.allclose(np.sum(run.fibres.fibre * run.fibres.sheet, axis=1), 0.0, atol=1e-9)
    assert np.allclose(grid.point_data["ldrb_ab"][apex.nodes], 0.0)


def test_a_second_run_reuses_the_fields(grid):
    points, tets, mask = vf.inputs(grid)
    apex = vf.find_apex(points, tets, mask)
    vf.write(grid, vf.generate(points, tets, mask, apex))
    cached = vf.cached_fields(grid, apex)
    assert cached is not None
    again = vf.generate(points, tets, mask, apex, FibreAngles(alpha_endo=60.0),
                        cached=cached)
    assert again.reused and not again.reports
    assert "reused" in again.details()


def test_another_apex_does_not_reuse(grid):
    points, tets, mask = vf.inputs(grid)
    apex = vf.find_apex(points, tets, mask)
    vf.write(grid, vf.generate(points, tets, mask, apex))
    other = vf.apex_at(points, tets, mask, [SIZE, SIZE, 0.0])
    assert other.node != apex.node
    assert vf.cached_fields(grid, other) is None


def test_the_result_survives_a_save(grid, tmp_path):
    points, tets, mask = vf.inputs(grid)
    apex = vf.find_apex(points, tets, mask)
    vf.write(grid, vf.generate(points, tets, mask, apex))
    path = tmp_path / "fibres.vtk"
    grid.save(path)
    back = pv.read(path)
    assert vf.cached_fields(back, apex) is not None
    assert vf.reconcile(back) == vf.KEPT


# ---------------------------------------------------------------------
# Staleness
# ---------------------------------------------------------------------
def _with_fibres(grid):
    points, tets, mask = vf.inputs(grid)
    vf.write(grid, vf.generate(points, tets, mask, vf.find_apex(points, tets, mask)))
    return grid


def test_an_unchanged_mesh_keeps_its_result(grid):
    """A relabelling that gives the same labels, or a tissue tag: nothing
    the fibres depend on moved."""
    grid = _with_fibres(grid)
    grid.cell_data["tissueTag"] = np.ones(grid.n_cells, dtype=np.int32)
    assert vf.reconcile(grid, vf.stamp(grid)) == vf.KEPT
    assert vf.FIBRE_FIELD in grid.cell_data and vf.stamp(grid) is not None


def test_new_labels_on_the_same_geometry_remove_the_result(grid):
    grid = _with_fibres(grid)
    mask = np.asarray(grid.point_data[LABEL_MASK_FIELD]).copy()
    node = int(np.where(mask == MASK_BITS[EPI])[0][0])
    mask[node] = MASK_BITS[RV_ENDO]              # a genuine change of label
    grid.point_data[LABEL_MASK_FIELD] = mask
    assert vf.reconcile(grid) == vf.REMOVED
    assert not any(n in grid.cell_data or n in grid.point_data
                   for n in vf.written_names())
    assert vf.stamp(grid) is None and vf.provenance(grid) == vf.NO_FIBRES


def test_a_changed_geometry_carries_the_fibres(grid):
    """A remesh interpolates the fibres onto the new mesh on purpose, and a
    clean keeps each surviving element's own. They stay, marked as carried,
    without the stamp: the next run must solve, not reuse."""
    grid = _with_fibres(grid)
    previous = vf.stamp(grid)
    cleaned = pv.UnstructuredGrid(grid.extract_cells(np.arange(grid.n_cells - 10)))
    for key in list(cleaned.field_data.keys()):
        del cleaned.field_data[key]
    assert vf.reconcile(cleaned, previous) == vf.CARRIED
    assert vf.FIBRE_FIELD in cleaned.cell_data
    assert vf.stamp(cleaned) is None
    assert vf.provenance(cleaned) == vf.CARRIED
    points, tets, mask = vf.inputs(cleaned)
    assert vf.cached_fields(cleaned, vf.find_apex(points, tets, mask)) is None
    assert "interpolation" in vf.describe_provenance(cleaned)


def test_carried_fibres_stay_carried_until_generated_again(grid, tmp_path):
    grid = _with_fibres(grid)
    previous = vf.stamp(grid)
    moved = grid.copy(deep=True)
    moved.points = np.asarray(moved.points) * 1.01
    for key in list(moved.field_data.keys()):
        del moved.field_data[key]
    assert vf.reconcile(moved, previous) == vf.CARRIED
    moved.save(tmp_path / "carried.vtk")
    back = pv.read(tmp_path / "carried.vtk")
    assert vf.provenance(back) == vf.CARRIED
    assert vf.reconcile(back) is None            # nothing of ours to judge
    points, tets, mask = vf.inputs(back)
    vf.write(back, vf.generate(points, tets, mask, vf.find_apex(points, tets, mask)))
    assert vf.provenance(back) == vf.GENERATED


def test_fibres_the_mesh_arrived_with_are_left_alone(grid):
    grid.cell_data[vf.FIBRE_FIELD] = np.tile([1.0, 0.0, 0.0], (grid.n_cells, 1))
    assert vf.reconcile(grid) is None
    assert vf.FIBRE_FIELD in grid.cell_data
    assert vf.provenance(grid) == vf.FOREIGN


# ---------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------
def test_segments_follow_the_directions(grid):
    grid = _with_fibres(grid)
    lines = vf.segments(grid, vf.FIBRE_FIELD, count=200)
    assert lines.n_lines == 200
    pts = np.asarray(lines.points)
    d = pts[1::2] - pts[0::2]
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    fib = np.asarray(grid.cell_data[vf.FIBRE_FIELD])
    # Every segment is parallel to some element's fibre.
    best = np.abs(d @ fib.T).max(axis=1)
    assert np.all(best > 1 - 1e-9)


def test_more_segments_than_elements_draws_one_per_element(grid):
    grid = _with_fibres(grid)
    assert vf.segments(grid, vf.FIBRE_FIELD, count=10 ** 7).n_lines == grid.n_cells


# ---------------------------------------------------------------------
# The window
# ---------------------------------------------------------------------
@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def _wait(qapp, dlg, worker, timeout=60.0):
    start = time.time()
    while worker.isRunning() and time.time() - start < timeout:
        qapp.processEvents()
        time.sleep(0.01)
    qapp.processEvents()


def test_the_dialog_opens_with_an_apex(qapp, grid):
    from ccdaf.gui.fibre_dialog import FibreDialog
    dlg = FibreDialog(grid)
    seen = []
    dlg.apex_changed.connect(seen.append)
    qapp.processEvents()
    assert dlg.apex() is not None and seen and seen[-1] is dlg.apex()
    assert dlg.btn_run.isEnabled()
    assert dlg.matches(grid)


def test_the_defaults_are_glrulefibers(qapp, grid):
    from ccdaf.gui.fibre_dialog import FibreDialog
    dlg = FibreDialog(grid)
    dlg.spn_alpha_endo.setValue(10.0)
    dlg.reset_angles()
    assert dlg.angles() == FibreAngles()


def test_generate_asks_to_run_with_the_apex_and_angles(qapp, grid):
    from ccdaf.gui.fibre_dialog import FibreDialog
    dlg = FibreDialog(grid)
    qapp.processEvents()
    asked = []
    dlg.run_requested.connect(lambda a, b: asked.append((a, b)))
    dlg.spn_beta_epi.setValue(30.0)
    dlg.btn_run.click()
    assert asked and asked[0][0] is dlg.apex() and asked[0][1].beta_epi == 30.0


def test_a_click_replaces_the_apex(qapp, grid):
    from ccdaf.gui.fibre_dialog import FibreDialog
    dlg = FibreDialog(grid)
    qapp.processEvents()
    dlg.set_apex_at([0.0, 0.0, 0.0])
    assert not dlg.apex().automatic
    assert "picked" in dlg.lbl_apex.text()


def test_running_locks_the_inputs(qapp, grid):
    from ccdaf.gui.fibre_dialog import FibreDialog
    dlg = FibreDialog(grid)
    qapp.processEvents()
    dlg.set_running(True)
    assert not dlg.btn_run.isEnabled() and not dlg.spn_alpha_endo.isEnabled()
    dlg.set_running(False)
    assert dlg.btn_run.isEnabled() and dlg.spn_alpha_endo.isEnabled()


def test_the_worker_runs_off_the_gui_thread(qapp, grid):
    from ccdaf.gui.fibre_dialog import FibreWorker
    points, tets, mask = vf.inputs(grid)
    worker = FibreWorker(points, tets, mask, vf.find_apex(points, tets, mask),
                         FibreAngles())
    got, failed, status = [], [], []
    worker.done.connect(got.append)
    worker.failed.connect(failed.append)
    worker.status.connect(status.append)
    worker.start()
    _wait(qapp, None, worker)
    assert not failed and got and len(got[0].fibres.fibre) == grid.n_cells
    assert any("Laplace solve" in s for s in status)


def test_a_failure_is_reported_not_raised(qapp, grid):
    from ccdaf.gui.fibre_dialog import FibreWorker
    points, tets, mask = vf.inputs(grid)
    apex = vf.find_apex(points, tets, mask)
    bad = vf.Apex(node=apex.node, nodes=vf.surface_sets(mask)[BASE][:3])  # base held at 0 too
    worker = FibreWorker(points, tets, mask, bad, FibreAngles())
    failed = []
    worker.failed.connect(failed.append)
    worker.start()
    _wait(qapp, None, worker)
    assert failed and "held at 1 and at 0" in failed[0]


def test_dialog_controls_have_tooltips(qapp, grid):
    from ccdaf.gui.fibre_dialog import FibreDialog
    dlg = FibreDialog(grid)
    for w in (dlg.btn_auto, dlg.btn_pick, dlg.btn_defaults, dlg.btn_run,
              dlg.spn_alpha_endo, dlg.spn_alpha_epi, dlg.spn_beta_endo,
              dlg.spn_beta_epi):
        assert w.toolTip()


def test_existing_fibres_are_announced(qapp, grid):
    from ccdaf.gui.fibre_dialog import FibreDialog
    grid.cell_data[vf.FIBRE_FIELD] = np.tile([1.0, 0.0, 0.0], (grid.n_cells, 1))
    dlg = FibreDialog(grid)
    texts = [w.text() for w in dlg.findChildren(QtWidgets.QLabel)]
    assert any("from elsewhere" in t and "replaces them" in t for t in texts)


def test_the_window_says_where_the_fibres_came_from(qapp, grid):
    from ccdaf.gui.fibre_dialog import FibreDialog
    assert not FibreDialog(grid).lbl_note.text()          # none yet
    grid = _with_fibres(grid)
    assert "generated here" in FibreDialog(grid).lbl_note.text()
    dlg = FibreDialog(grid)
    dlg.set_note("Mesh and labels unchanged: fibres kept.")
    assert "fibres kept" in dlg.lbl_note.text()


# ---------------------------------------------------------------------
# The application's sequence
# ---------------------------------------------------------------------
def test_replacing_the_volume_checks_staleness_before_adopting_it():
    """Stale fibres must be gone before the view or a save can see them."""
    import inspect
    import re

    from ccdaf.app.ccdaf import CCDAF
    source = inspect.getsource(CCDAF._replace_volume)
    order = re.findall(r"reconcile|set_volume|_populate_fields", source)
    assert order.index("reconcile") < order.index("set_volume")
    assert order.index("reconcile") < order.index("_populate_fields")


def test_every_outcome_has_a_note():
    from ccdaf.app.ccdaf import CCDAF
    for outcome in (vf.KEPT, vf.CARRIED, vf.REMOVED):
        assert CCDAF.FIBRE_NOTES[outcome]


def test_the_fibre_controls_follow_the_mesh(qapp):
    from ccdaf.gui.visualisation_widget import VisualisationWidget
    w = VisualisationWidget()
    assert not w.chk_fibres.isEnabled()
    w.set_fibre_directions(["fiber", "sheet"])
    w.chk_fibres.setChecked(True)
    assert w.show_fibres() and w.fibre_direction() == "fiber"
    w.set_fibre_directions([])
    assert not w.show_fibres() and w.chk_fibres.isChecked()   # the tick survives


def test_a_remesh_that_drops_the_labels_says_so(monkeypatch):
    """With the boundary adapted the labels cannot be carried, and the
    transfer's own line is overwritten by the remesh's: the final message
    has to say it, or the labelling vanishes unannounced."""
    from unittest.mock import MagicMock

    import ccdaf.app.ccdaf as app
    from ccdaf.app.ccdaf import CCDAF
    from ccdaf.core.volume_mesh import VOLUME
    from ccdaf.core.volume_postprocessor import RemeshOptions

    source = _labelled()
    remeshed = source.copy(deep=True)
    del remeshed.point_data[LABEL_MASK_FIELD]
    monkeypatch.setattr(app, "remesh_volume", lambda grid, options, **kw: remeshed)

    for freeze, drops in ((False, True), (True, False)):
        host = MagicMock()
        host.loader.kind = VOLUME
        host.loader.grid = source
        host.volume_postproc.options.return_value = RemeshOptions(
            freeze_boundary=freeze)
        if not drops:
            monkeypatch.setattr(app, "remesh_volume",
                                lambda grid, options, **kw: source.copy(deep=True))
        CCDAF._action_remesh_volume(host)
        said = host.statusBar().showMessage.call_args[0][0]
        assert ("label the surfaces again" in said) is drops
        assert ("label the surfaces again"
                in host.volume_postproc.set_status.call_args[0][0]) is drops
