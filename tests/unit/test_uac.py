"""
test_uac.py
===========
Universal atrial coordinates, and their window.

The test atrium is a sphere of radius 25 mm, open at the bottom (the mitral
valve) and at four vein openings, each ringed by a band of its vein's label,
with an appendage patch on the anterior-lateral side. Small (6 000 points),
so a full run takes seconds.

The contract:

* the action needs a surface with all six labels, and says why not;
* a sphere with holes has genus 0, a torus genus 1;
* the repair removes a duplicated flap and a sliver lying on an opening,
  and keeps a duplicate whose removal would open a hole;
* a thin tunnel is cut and both openings closed, the patch is marked and
  the genus drops to 0; a wide one is refused with its loop length;
* every available solver gives SciPy's answer, and CHOLMOD stays right
  when only the values of the matrix change;
* the template warp moves landmarks onto their targets without folding a
  triangle and keeps hard points where they are told;
* a full run: no folded triangles, every ring near its template target,
  alpha and beta in [0, 1] on every input point, each vein and the
  appendage in its template place;
* the flat map lies in z = 0, holds points on the cut twice, and maps each
  point back to an input point;
* an atrium without a valve opening is refused; when no cut works, the
  message names the problem (touching labels, a twisted layout);
* the window runs in the background, reports, and enables saving.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pytest
import pyvista as pv
import scipy.sparse as sp
from PyQt5 import QtWidgets

from ccdaf.core import uac
from ccdaf.core.uac import solvers, template as tpl
from ccdaf.core.uac.flatten import collapsed_patches, signed_areas
from ccdaf.core.uac.mesh_ops import remove_duplicates, remove_rim_slivers, topology, unique_edges
from ccdaf.core.uac.slim import slim
from ccdaf.core.uac.tunnels import TunnelError, remove_tunnels

DIRS = {tpl.LSPV: (-0.55, -0.35, 0.75), tpl.LIPV: (-0.65, -0.65, 0.2),
        tpl.RSPV: (0.55, -0.35, 0.75), tpl.RIPV: (0.65, -0.65, 0.2),
        tpl.LAA: (-0.75, 0.6, 0.25)}


def _atrium(res: int = 90, radius: float = 25.0, dirs=None) -> pv.PolyData:
    s = pv.Sphere(radius=radius, theta_resolution=res, phi_resolution=res).triangulate().clean()
    P = np.asarray(s.points)
    t = s.faces.reshape(-1, 4)[:, 1:]
    c = P[t].mean(1)
    u = c / np.linalg.norm(c, axis=1, keepdims=True)
    tag = np.ones(len(t), int)
    keep = u[:, 2] > -0.75                                  # mitral valve opening
    for L, d in (dirs or DIRS).items():
        d = np.array(d) / np.linalg.norm(d)
        ang = np.arccos(np.clip(u @ d, -1, 1))
        if L == tpl.LAA:
            tag[ang < 0.32] = L
        else:
            tag[ang < 0.30] = L
            keep &= ang > 0.14                              # the vein's opening
    m = pv.PolyData(P, np.c_[np.full(keep.sum(), 3), t[keep]].ravel())
    m.cell_data["elemTag"] = tag[keep]
    return m.clean()


@pytest.fixture(scope="module")
def atrium():
    return _atrium()


@pytest.fixture(scope="module")
def run(atrium):
    return uac.compute(*uac.inputs(atrium), uac.Options(iterations=40))


@pytest.fixture(scope="module")
def done(run, atrium):
    """The atrium as the application keeps it after a run: repaired, with
    the coordinates."""
    return uac.write(atrium, run)


def _faces(m):
    return m.faces.reshape(-1, 4)[:, 1:]


# ---------------------------------------------------------------------
# Availability and topology
# ---------------------------------------------------------------------
def test_availability_says_why_not(atrium):
    assert uac.availability(None)[0] is False
    assert "volume" in uac.availability(pv.ImageData(dimensions=(3, 3, 3)).cast_to_unstructured_grid())[1]
    bare = pv.Sphere()
    assert "elemTag" in uac.availability(bare)[1]
    partial = atrium.copy()
    partial.cell_data["elemTag"] = np.where(partial.cell_data["elemTag"] == tpl.LAA, 1,
                                            partial.cell_data["elemTag"])
    ok, why = uac.availability(partial)
    assert not ok and "LAA" in why
    assert uac.availability(atrium) == (True, "")


def test_topology_counts_tunnels(atrium):
    assert topology(_faces(atrium)) == (0, 5)
    torus = pv.ParametricTorus(ringradius=10, crosssectionradius=2).triangulate().clean()
    assert topology(_faces(torus))[0] == 1


def _grid(n=6):
    """A flat n x n grid of unit squares, two triangles each."""
    xs, ys = np.meshgrid(np.arange(n + 1.0), np.arange(n + 1.0))
    P = np.c_[xs.ravel(), ys.ravel(), np.zeros(xs.size)]
    t = []
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i
            t += [(a, a + 1, a + n + 2), (a, a + n + 2, a + n + 1)]
    return P, np.array(t)


def test_a_duplicated_flap_is_removed_and_a_needed_copy_kept():
    P, t = _grid()
    # a flap: a new point on edge (0, 1) and its triangle, listed twice
    P = np.vstack([P, [0.5, 1e-4, 0.0]])
    c = len(P) - 1
    flap = np.array([[0, 1, c], [1, 0, c]])
    t2 = np.vstack([t, flap])
    out, tag, n = remove_duplicates(t2, np.ones(len(t2), int))
    assert n == 2 and len(out) == len(t) and (unique_edges(out)[2] <= 2).all()
    # a real triangle listed twice: one copy must stay, or the grid gets a hole
    t3 = np.vstack([t, t[:1]])
    out, _, n = remove_duplicates(t3, np.ones(len(t3), int))
    assert n == 1 and len(out) == len(t)


def test_a_sliver_on_an_opening_is_removed():
    P, t = _grid()
    # replace the bottom-left triangle (0, 1, 8) by a sliver whose far corner
    # sits just above the open edge (0, 1)
    P = np.vstack([P, [0.5, 1e-3, 0.0]])
    c = len(P) - 1
    t = np.vstack([[[0, 1, c], [1, 8, c], [c, 8, 0]], t[1:]])
    out, _, n = remove_rim_slivers(P, t, np.ones(len(t), int))
    assert n == 1 and not ((out == 0) & (out[:, [1, 2, 0]] == 1)).all(1).any()
    assert topology(out)[0] == 0


def test_a_patch_squeezed_to_a_point_is_reported_not_counted_as_folded():
    P, t = _grid()
    U = P[:, :2].copy() / 10.0
    spike = np.unique(t[:2])                     # two triangles, squeezed to one point
    U[spike] = U[spike[0]] + 1e-9 * np.random.default_rng(0).standard_normal((len(spike), 2))
    tags = np.ones(len(t), int)
    real = np.ones(len(t), bool)
    mask, patches = collapsed_patches(U, t, P, tags, real)
    assert set(np.where(mask)[0]) >= {0, 1}
    assert len(patches) == 1 and patches[0]["label"] == 1 and patches[0]["size_mm"] > 0
    assert not collapsed_patches(U, t, P, tags, np.zeros(len(t), bool))[0].any()  # not tissue
    assert not collapsed_patches(P[:, :2], t, P, tags, real)[0].any()           # a sound map


def _holed_torus(cross):
    torus = pv.ParametricTorus(ringradius=12, crosssectionradius=cross, u_res=120, v_res=40,
                               w_res=2).triangulate().clean()
    P = np.asarray(torus.points)
    t = _faces(torus)
    keep = np.linalg.norm(P[t].mean(1) - np.array([12 + cross, 0, 0]), axis=1) > 0.6 * cross
    return P, t[keep]


def test_a_thin_tunnel_is_cut_and_closed():
    P, t = _holed_torus(1.5)                              # loop round it ~9.4 mm
    assert topology(t) == (1, 1)
    tag = np.ones(len(t), int)
    P2, t2, tag2, origin, patch, rep = remove_tunnels(P, t, tag, np.arange(len(P)))
    assert topology(t2)[0] == 0
    assert patch.sum() > 0 and len(patch) == len(t2) and not patch[: len(t)].any()
    assert len(rep) == 1 and rep[0]["loop_mm"] < 15
    assert (origin[: len(P)] == np.arange(len(P))).all()
    assert ((origin == -1) | (origin < len(P))).all()


def test_a_wide_tunnel_is_refused():
    P, t = _holed_torus(3.5)                              # loop round it ~22 mm
    with pytest.raises(TunnelError, match="mm long"):
        remove_tunnels(P, t, np.ones(len(t), int), np.arange(len(P)))


# ---------------------------------------------------------------------
# Solvers and the warp
# ---------------------------------------------------------------------
def _spd(n=400, seed=1):
    rng = np.random.default_rng(seed)
    A = sp.random(n, n, density=0.01, random_state=seed)
    A = A @ A.T + sp.eye(n) * n * 0.01
    return sp.csr_matrix(A), rng.standard_normal(n)


def test_every_available_solver_matches_scipy():
    from scipy.sparse.linalg import spsolve
    A, b = _spd()
    ref = spsolve(A.tocsc(), b)
    for name in solvers.available():
        x = solvers.SpdSolver(name).solve(A, b)
        assert np.allclose(x, ref, rtol=1e-8, atol=1e-10), name


def test_a_reused_factorisation_follows_new_values():
    if "cholmod" not in solvers.available():
        pytest.skip("CHOLMOD not installed")
    from scipy.sparse.linalg import spsolve
    A, b = _spd()
    s = solvers.SpdSolver("cholmod")
    s.solve(A, b)
    A2 = A.copy()
    A2.data = A2.data * 1.7 + 0.1 * (A2.indices == np.repeat(np.arange(A2.shape[0]), np.diff(A2.indptr)))
    assert np.allclose(s.solve(A2, b), spsolve(A2.tocsc(), b), rtol=1e-8, atol=1e-10)


def test_the_warp_reaches_targets_without_folding():
    disk = pv.Disc(inner=0.0, outer=1.0, r_res=12, c_res=48).triangulate().clean()
    R = np.asarray(disk.points)[:, :2]
    t = _faces(disk)
    rim = np.where(np.isclose(np.linalg.norm(R, axis=1), 1.0))[0]
    centre = int(np.argmin(np.linalg.norm(R, axis=1)))
    U, hist, kept = slim(R, t, np.array([centre]), np.array([[0.4, 0.2]]), rim, R[rim], iters=80)
    assert kept.all()
    assert np.linalg.norm(U[centre] - [0.4, 0.2]) < 1e-3
    assert np.allclose(U[rim], R[rim])
    assert (np.sign(signed_areas(U, t)) == np.sign(signed_areas(R, t))).all()


# ---------------------------------------------------------------------
# A full run
# ---------------------------------------------------------------------
def test_a_run_is_fold_free_and_on_template(run, done):
    a = run.attempt
    assert a.folded_real == 0 and a.collapsed == []
    assert a.worst_miss < 0.02
    assert run.solver == solvers.available()[0]
    assert run.alpha.shape == (done.n_points,) and run.beta.shape == (done.n_points,)
    assert np.isfinite(run.alpha).all() and np.isfinite(run.beta).all()
    assert run.alpha.min() >= -1e-6 and run.alpha.max() <= 1 + 1e-6
    assert run.beta.min() >= -1e-6 and run.beta.max() <= 1 + 1e-6


def test_veins_and_appendage_land_in_their_template_places(run, done):
    t = _faces(done)
    tag = np.asarray(done.cell_data["elemTag"])
    places = {tpl.LIPV: tpl.LIPV_C, tpl.RIPV: tpl.RIPV_C, tpl.LAA: tpl.LAA_C,
              tpl.LSPV: (tpl.LSPV_C[0], 0.55), tpl.RSPV: (tpl.RSPV_C[0], 0.55)}
    for L, (a0, b0) in places.items():
        pts = np.unique(t[tag == L])
        centre = (np.median(run.alpha[pts]), np.median(run.beta[pts]))
        assert abs(centre[0] - a0) < 0.08 and abs(centre[1] - b0) < 0.08, (tpl.NAMES[L], centre)
    body = np.unique(t[tag == tpl.BODY])
    assert (run.beta[body] < 0.5).any() and (run.beta[body] > 0.5).any()


def test_the_flat_map_from_the_stored_values_is_the_run_s_own(run, done):
    flat = uac.flat_mesh(done)
    assert np.allclose(np.asarray(flat.points)[:, 2], 0)
    src = np.asarray(flat.point_data["uac_source"])
    assert src.min() >= 0 and src.max() < done.n_points
    assert len(src) > len(np.unique(src))                  # cut points appear once per half
    uv = np.asarray(flat.points)[:, :2]
    assert uv.min() >= -1e-6 and uv.max() <= 1 + 1e-6
    # the run's own two halves: each atrium point at the same flat positions
    n0 = done.n_points
    want: dict = {}
    for h in run.attempt.halves:
        ok = h.nodes < n0
        for p, q in zip(h.nodes[ok], h.uv[ok]):
            want.setdefault(int(p), []).append(q)
    got: dict = {}
    for p, q in zip(src, uv):
        got.setdefault(int(p), []).append(q)
    assert got.keys() == want.keys()
    for p in got:
        g = np.array(sorted(map(tuple, got[p])))
        w = np.array(sorted(map(tuple, want[p])))
        assert len(g) == len(w) and np.abs(g - w).max() < 1e-12, p
    assert flat.n_cells == done.n_cells


def test_write_returns_the_repaired_atrium_and_a_second_repair_finds_nothing(run, atrium, done):
    assert set(uac.written_names()) <= set(done.point_data.keys())
    assert uac.ALPHA not in atrium.point_data.keys()          # the input is not changed
    assert "uac_alpha" in run.details() and run.summary().startswith("UAC computed")
    again = uac.prepare(*uac.inputs(done))
    assert not again.changed and (again.t == _faces(done)).all()


def test_a_closed_valve_is_refused():
    m = _atrium()
    t = _faces(m)
    tag = np.asarray(m.cell_data["elemTag"])
    # no opening next to body tissue: give the tissue round the valve a vein's label
    P = np.asarray(m.points)
    low = P[t].mean(1)[:, 2] < -15
    tag = np.where(low, tpl.LIPV, tag)
    with pytest.raises(uac.UacError):
        uac.compute(P, t, tag, uac.Options(iterations=5))


def test_left_and_right_veins_swapped_are_refused():
    from ccdaf.core.uac.cut import CutError, check_sides
    P, t, tag = uac.inputs(_atrium())
    check_sides(P, t, tag)                                   # the right way round: fine
    swapped = tag.copy()
    for a, b in ((tpl.LSPV, tpl.RSPV), (tpl.LIPV, tpl.RIPV)):
        swapped[tag == a], swapped[tag == b] = b, a
    with pytest.raises(CutError, match="swapped"):
        check_sides(P, t, swapped)
    with pytest.raises(uac.UacError, match="LSPV .* mm, RSPV"):
        uac.compute(P, t, swapped, uac.Options(iterations=5))


def test_the_diagnosis_names_the_problem():
    from ccdaf.core.uac.cut import anatomy, diagnose
    dirs = dict(DIRS)
    dirs[tpl.LSPV], dirs[tpl.RSPV] = (-0.2, -0.35, 0.9), (0.2, -0.35, 0.9)   # bands meet
    m = _atrium(dirs=dirs)
    P, t, tag = uac.inputs(m)
    msg = diagnose(P, t, tag, anatomy(P, t, tag), [])
    assert msg.startswith("No valid cut") and "labels touch" in msg
    P, t, tag = uac.inputs(_atrium())
    twisted = [{"where": {tpl.LSPV: 0, tpl.LIPV: 1, tpl.RSPV: 1, tpl.RIPV: 0, tpl.LAA: 1},
                "sides_ok": True}]
    msg = diagnose(P, t, tag, anatomy(P, t, tag), twisted)
    assert "twisted" in msg and "LSPV" in msg and "LIPV" in msg and "labels touch" not in msg


# ---------------------------------------------------------------------
# The window
# ---------------------------------------------------------------------
@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def test_the_window_runs_in_the_background_and_reports(qapp, atrium):
    from ccdaf.gui.uac_dialog import UacDialog, UacWorker
    dlg = UacDialog(atrium)
    assert dlg.replace_note.isHidden()           # nothing to replace yet
    assert dlg.btn_run.isEnabled() and not dlg.btn_save.isEnabled() and not dlg.btn_stop.isEnabled()
    for w in (dlg.btn_run, dlg.btn_stop, dlg.btn_save):
        assert w.toolTip()
    worker = UacWorker(*dlg.inputs())
    got, failed, said = [], [], []
    worker.done.connect(got.append)
    worker.failed.connect(failed.append)
    worker.status.connect(said.append)
    worker.start()
    start = time.time()
    while worker.isRunning() and time.time() - start < 300:
        qapp.processEvents()
        time.sleep(0.02)
    qapp.processEvents()
    assert not failed and got and said
    dlg.show_run(got[0])
    assert "Solver:" in dlg.report.toPlainText()
    assert dlg.matches(atrium)
    done = uac.write(atrium, got[0])
    dlg.adopt(done, before=atrium)
    assert not dlg.replace_note.isHidden() and dlg.btn_save.isEnabled() and dlg.btn_undo.isEnabled()
    assert dlg.matches(done) and dlg.before() is atrium
    reopened = UacDialog(done)                    # a surface from a file with the arrays
    assert not reopened.replace_note.isHidden() and reopened.btn_save.isEnabled()
    assert not reopened.btn_undo.isEnabled()
    changed = atrium.copy()
    changed.points = changed.points * 1.01
    assert not dlg.matches(changed)


def test_the_window_can_stop_a_run(qapp, atrium):
    from ccdaf.gui.uac_dialog import UacWorker
    worker = UacWorker(*uac.inputs(atrium))
    failed = []
    worker.failed.connect(failed.append)
    worker.stop()                                        # asked before the warp starts
    worker.start()
    start = time.time()
    while worker.isRunning() and time.time() - start < 120:
        qapp.processEvents()
        time.sleep(0.02)
    qapp.processEvents()
    assert failed == ["Stopped."]


def test_the_save_dialog_ticks_the_coordinates(qapp):
    from ccdaf.gui.save_mesh_dialog import SaveMeshDialog
    dlg = SaveMeshDialog(point_fields=["LAT", "uac_alpha", "uac_beta"], cell_fields=["elemTag"])
    assert {"uac_alpha", "uac_beta", "elemTag"} <= set(dlg.selected_fields())
    assert "LAT" not in dlg.selected_fields()
