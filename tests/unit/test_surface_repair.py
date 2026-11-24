"""
test_surface_repair.py
======================
The shared connectivity repairs (``ccdaf.core.surface_repair``): each one
fixes what it is for, moves no point, and carries every point and cell
field onto the repaired surface.
"""
from __future__ import annotations

import numpy as np
import pytest
import pyvista as pv

from ccdaf.core import surface_repair as sr
from ccdaf.core.uac.mesh_ops import topology, unique_edges


def _poly(P, t, tag=None):
    m = pv.PolyData(np.asarray(P, float), np.c_[np.full(len(t), 3), t].ravel())
    m.point_data["LAT"] = np.asarray(m.points)[:, 0] * 10.0     # a smooth point field
    m.point_data["region"] = np.arange(m.n_points, dtype=np.int32)
    m.cell_data["elemTag"] = (np.ones(len(t), np.int32) if tag is None
                              else np.asarray(tag, np.int32))
    m.cell_data["cellid"] = np.arange(len(t), dtype=np.int64)
    return m


def _faces(m):
    return m.faces.reshape(-1, 4)[:, 1:]


def _grid(n=6):
    xs, ys = np.meshgrid(np.arange(n + 1.0), np.arange(n + 1.0))
    P = np.c_[xs.ravel(), ys.ravel(), np.zeros(xs.size)]
    t = []
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i
            t += [(a, a + 1, a + n + 2), (a, a + n + 2, a + n + 1)]
    return P, np.array(t)


def _carried(src, out):
    """Every surviving point keeps its own field values."""
    idx = out.point_data["region"]
    assert np.allclose(out.point_data["LAT"], src.point_data["LAT"][idx])
    assert np.allclose(out.points, src.points[idx])          # no point moved


ALL = (sr.remove_duplicates, sr.trim_rims, sr.flip_edges, sr.trim_rims, sr.split_pinches,
       sr.join_islands, sr.cut_tunnels)


def test_a_second_run_changes_nothing():
    P, t = _grid()
    once, rep = sr.run(_poly(P, t), *ALL)
    assert rep["ears_removed"] == 2              # the grid's two one-triangle corners
    twice, rep2 = sr.run(once, *ALL)
    assert not any(rep2.values())
    assert twice is not once and twice.n_cells == once.n_cells
    assert np.allclose(twice.points, once.points) and (_faces(twice) == _faces(once)).all()


def test_a_duplicated_flap_goes_and_fields_follow():
    P, t = _grid()
    P = np.vstack([P, [0.5, 1e-4, 0.0]])
    c = len(P) - 1
    m = _poly(P, np.vstack([t, [[0, 1, c], [1, 0, c]]]))
    out, rep = sr.run(m, sr.remove_duplicates)
    assert rep["duplicates_removed"] == 2
    assert out.n_cells == len(t) and out.n_points == len(P) - 1      # the flap's tip went too
    assert (unique_edges(_faces(out))[2] <= 2).all()
    _carried(m, out)
    assert (out.cell_data["cellid"] == np.arange(len(t))).all()


def test_ears_and_rim_slivers_go_with_their_tips():
    P, t = _grid()
    P = np.vstack([P, [0.5, -0.5, 0.0], [3.5, 1e-3, 0.0]])
    ear, sl = len(P) - 2, len(P) - 1
    # an ear below edge (0, 1); a sliver replacing triangle (3, 4, 11)
    t = np.vstack([[[1, 0, ear]], [[3, 4, sl], [4, 11, sl], [sl, 11, 3]],
                   np.delete(t, 6, axis=0)])
    m = _poly(P, t)
    out, rep = sr.run(m, sr.trim_rims)
    assert rep["ears_removed"] >= 1 and rep["rim_slivers_removed"] >= 1
    assert ear not in out.point_data["region"]
    assert topology(_faces(out))[0] == 0
    _carried(m, out)


def test_flips_fix_a_thin_pair_and_keep_points_and_labels():
    P, t = _grid(4)
    P[6, :2] += [0.45, 0.45]                     # push a point so a diagonal is bad
    m = _poly(P, t, tag=np.r_[np.ones(len(t) // 2), 2 * np.ones(len(t) - len(t) // 2)])
    out, rep = sr.run(m, sr.flip_edges)
    assert rep["flips"] >= 1
    assert out.n_cells == m.n_cells and out.n_points == m.n_points
    assert np.allclose(out.points, m.points)
    assert (np.bincount(out.cell_data["elemTag"]) == np.bincount(m.cell_data["elemTag"])).all()
    again, rep2 = sr.run(out, sr.flip_edges)
    assert rep2["flips"] == 0                    # a second pass has nothing left to do


def test_a_pinched_point_gets_a_copy_with_its_values():
    # two fans sharing point 0 only (a bow tie)
    P = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [-1, 0, 0], [-1, -1, 0.0]])
    t = np.array([[0, 1, 2], [0, 3, 4]])
    m = _poly(P, t)
    out, rep = sr.run(m, sr.split_pinches)
    assert rep["pinches_split"] == 1 and out.n_points == 6
    _carried(m, out)


def test_islands_take_the_label_round_them():
    P, t = _grid()
    tag = np.ones(len(t), int)
    tag[20] = 11
    m = _poly(P, t, tag)
    out, rep = sr.run(m, sr.join_islands)
    # the island is not the largest piece of label 11 only if 11 has another piece
    assert rep["island_triangles_relabelled"] == 0
    tag[[0, 1, 2, 3]] = 11                       # a bigger piece of 11 elsewhere
    out, rep = sr.run(_poly(P, t, tag), sr.join_islands)
    assert rep["island_triangles_relabelled"] == 1 and out.cell_data["elemTag"][20] == 1


def _holed_torus(cross):
    torus = pv.ParametricTorus(ringradius=12, crosssectionradius=cross, u_res=120, v_res=40,
                               w_res=2).triangulate().clean()
    P = np.asarray(torus.points)
    t = _faces(torus)
    keep = np.linalg.norm(P[t].mean(1) - np.array([12 + cross, 0, 0]), axis=1) > 0.6 * cross
    used = np.unique(t[keep])
    remap = np.full(len(P), -1)
    remap[used] = np.arange(len(used))
    return P[used], remap[t[keep]]


def test_a_thin_tunnel_is_cut_and_closed_with_the_label_round_it():
    P, t = _holed_torus(1.5)
    tag = np.where(P[t].mean(1)[:, 0] > 0, 13, 1)
    # a second, separate piece with no tunnel must not stop the search
    s = pv.Sphere(radius=3, center=(40, 0, 0)).triangulate()
    both = _poly(np.vstack([P, s.points]), np.vstack([t, _faces(s) + len(P)]),
                 np.r_[tag, np.ones(s.n_cells, int)])
    out, rep = sr.run(both, sr.cut_tunnels)
    assert topology(_faces(out))[0] == 0 and len(rep["tunnels"]) == 1
    new = out.point_data["region"].size - both.n_points
    assert new > 0
    patch = np.arange(out.n_cells) >= both.n_cells
    assert patch.any() and set(np.unique(out.cell_data["elemTag"][patch])) <= {1, 13}
    assert np.isfinite(out.point_data["LAT"]).all()
    old = np.arange(both.n_points)
    assert np.allclose(out.point_data["LAT"][old], both.point_data["LAT"])


def test_a_wide_tunnel_is_left_alone_unless_strict():
    P, t = _holed_torus(3.5)
    m = _poly(P, t)
    out, rep = sr.run(m, sr.cut_tunnels)
    assert rep["tunnels_left"] == 1 and out.n_cells == m.n_cells
    with pytest.raises(sr.TunnelError):
        sr.run(m, lambda e: sr.cut_tunnels(e, strict=True))


def test_frozen_labels_are_never_dropped_or_re_cut():
    P, t = _grid(4)
    P[6, :2] += [0.45, 0.45]
    m = _poly(P, t, tag=np.full(len(t), 5))
    out, rep = sr.run(m, sr.flip_edges, sr.trim_rims, frozen_labels=(5,))
    assert rep["flips"] == 0 and rep["ears_removed"] == 0
    assert (_faces(out) == t).all()


# ---------------------------------------------------------------------
# Clean and the Flip edges step
# ---------------------------------------------------------------------
def _sphere_with_flap():
    s = pv.Sphere(radius=10, theta_resolution=30, phi_resolution=30).triangulate().clean()
    P = np.asarray(s.points)
    t = _faces(s)
    a, b = t[0, 0], t[0, 1]
    P = np.vstack([P, 0.5 * (P[a] + P[b]) + 1e-4])
    c = len(P) - 1
    return _poly(P, np.vstack([t, [[a, b, c], [b, a, c]]]))


def test_clean_drops_a_flap_without_opening_a_hole():
    from ccdaf.core.mesh_postprocessor import clean
    rep = {}
    out = clean(_sphere_with_flap(), quality_threshold=0.0, report=rep)
    assert rep["duplicates_removed"] == 2
    assert topology(_faces(out)) == (0, 0)                  # still closed


def test_clean_flips_by_default_and_the_step_alone_does_too():
    from ccdaf.core.mesh_postprocessor import PostprocessOptions, apply, clean, flip_edges
    P, t = _grid(4)
    P[6, :2] += [0.45, 0.45]
    m = _poly(P, t)
    rep = {}
    clean(m, quality_threshold=0.0, report=rep)
    assert rep["flips"] >= 1
    rep = {}
    clean(m, quality_threshold=0.0, flip=False, report=rep)
    assert "flips" not in rep
    rep = {}
    out = flip_edges(m, report=rep)
    assert rep["flips"] >= 1 and np.allclose(out.points, m.points)
    rep = {}
    apply(m, PostprocessOptions(do_flip_edges=True), report=rep)
    assert rep["flips"] >= 1


def test_clean_cuts_a_thin_tunnel():
    from ccdaf.core.mesh_postprocessor import clean
    P, t = _holed_torus(1.5)
    rep = {}
    out = clean(_poly(P, t), quality_threshold=0.0, report=rep)
    assert len(rep["tunnels"]) == 1 and topology(_faces(out))[0] == 0


@pytest.fixture(scope="module")
def qapp():
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def test_the_panel_offers_flips_inside_clean_and_on_their_own(qapp):
    from ccdaf.gui.postprocessing_widget import PostprocessingWidget, _describe
    w = PostprocessingWidget(lambda: None, lambda m: None)
    assert w.chk_clean_flip.isChecked() and not w.chk_flip.isChecked()
    assert w.chk_clean_flip.text() != w.chk_flip.text()
    w.chk_clean.setChecked(False)
    assert not w.chk_clean_flip.isEnabled()
    w.chk_clean.setChecked(True)
    assert w.chk_clean_flip.isEnabled()
    assert w.chk_clean_flip.toolTip() and w.chk_flip.toolTip()
    assert _describe({"flips": 3, "tunnels": [{}], "tunnels_left": 1}) == (
        "1 thin tunnels cut and closed, 3 edges flipped, 1 tunnel(s) wider than 15 mm left")


# ---------------------------------------------------------------------
# Clipping tidies its rim
# ---------------------------------------------------------------------
def test_a_clip_rim_is_tidied():
    from ccdaf.interaction.clipping_tool import tidy_rim
    P, t = _grid()
    P = np.vstack([P, [0.5, -0.5, 0.0]])
    m = _poly(P, np.vstack([t, [[1, 0, len(P) - 1]]]))
    out, words = tidy_rim(m)
    assert "ears" in words and out.n_points < m.n_points
    again, words2 = tidy_rim(out)
    assert words2 == "" and again.n_cells == out.n_cells
