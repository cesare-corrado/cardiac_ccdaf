"""
test_atrial_fibres.py
=====================
Actions > Atrial fibres: atlas fibres carried onto an atrium through the
UAC, extrapolated beyond the atlas, filled where the flat map is squeezed,
and the bilayer thickness checked. No test uses the network.
"""
from __future__ import annotations

import numpy as np
import pytest
import pyvista as pv

from ccdaf.core.uac import fibres as fb


def _square(n=20, top=1.0):
    """A grid on the unit square (a row on beta = 0.5, so no triangle
    straddles the two halves) with alpha = x, beta = y. Columns stop at
    alpha = *top*."""
    xs = np.linspace(0, top, int(round(n * top)) + 1)
    ys = np.linspace(0, 1, n + 1)
    X, Y = np.meshgrid(xs, ys)
    uv = np.c_[X.ravel(), Y.ravel()]
    nx = len(xs)
    t = []
    for j in range(n):
        for i in range(nx - 1):
            a = j * nx + i
            t += [(a, a + 1, a + nx + 1), (a, a + nx + 1, a + nx)]
    return uv, np.array(t)


def _mesh(P, t, uv, tag=1):
    m = pv.PolyData(np.asarray(P, float), np.c_[np.full(len(t), 3), t].ravel())
    m.point_data["uac_alpha"], m.point_data["uac_beta"] = uv[:, 0], uv[:, 1]
    m.cell_data["elemTag"] = np.full(len(t), tag, np.int32)
    return m


def _bent(uv):
    """A patient: stretched along alpha and bent out of the plane."""
    a, b = uv[:, 0], uv[:, 1]
    return np.c_[60 * a, 40 * b, 15 * np.sin(2 * a)]


def _expected(m):
    """The direction in which alpha grows on each triangle."""
    P = np.asarray(m.points)
    t = m.faces.reshape(-1, 4)[:, 1:]
    a = np.asarray(m.point_data["uac_alpha"])
    e1, e2 = P[t[:, 1]] - P[t[:, 0]], P[t[:, 2]] - P[t[:, 0]]
    da1, da2 = a[t[:, 1]] - a[t[:, 0]], a[t[:, 2]] - a[t[:, 0]]
    G = np.stack([np.c_[(e1 * e1).sum(1), (e1 * e2).sum(1)],
                  np.c_[(e1 * e2).sum(1), (e2 * e2).sum(1)]], 1)
    c = np.linalg.solve(G, np.c_[da1, da2][..., None])[..., 0]
    g = c[:, :1] * e1 + c[:, 1:] * e2                       # gradient of alpha
    return g / np.linalg.norm(g, axis=1, keepdims=True)


def _angle(a, b):
    return np.degrees(np.arccos(np.clip(np.abs((a * b).sum(1)), 0, 1)))


@pytest.fixture
def atlas():
    """A flat atlas whose fibres all run along alpha; it stops at alpha 0.8."""
    uv, t = _square(top=0.8)
    A = _mesh(np.c_[50 * uv, np.zeros(len(uv))], t, uv)
    A.cell_data["fibre"] = np.tile([1.0, 0.0, 0.0], (A.n_cells, 1))
    return A


def test_fibres_follow_the_coordinates_onto_another_shape(atlas):
    uv, t = _square()
    m = _mesh(_bent(uv), t, uv)
    f, rep = fb.map_layer(m, "endo", atlas=atlas)
    assert np.isfinite(f).all() and np.allclose(np.linalg.norm(f, axis=1), 1)
    assert np.percentile(_angle(f, _expected(m)), 99) < 1.0
    # the part beyond alpha 0.8 had no atlas: extrapolated, and still right
    assert 15 < rep.extrapolated["body"] < 25
    assert any("Beyond the atlas" in s for s in rep.lines())


def test_a_squeezed_triangle_takes_its_neighbours_fibre(atlas):
    uv, t = _square()
    uv = uv.copy()
    k = int(np.argmin(np.linalg.norm(uv - [0.4, 0.3], axis=1)))
    nb = np.unique(t[(t == k).any(1)])
    uv[k] = uv[nb[nb != k][0]]                      # collapse two flat points
    m = _mesh(_bent(_square()[0]), t, uv)
    f, rep = fb.map_layer(m, "endo", atlas=atlas)
    assert rep.filled >= 1 and np.isfinite(f).all()


def test_apply_writes_the_layers_and_the_thickness(atlas, monkeypatch):
    monkeypatch.setattr(fb.fibre_atlas, "load", lambda s, layer: atlas)
    uv, t = _square()
    m = _mesh(_bent(uv), t, uv)
    out, reps = fb.apply(m, "endo")
    assert "fibre_endo" in out.cell_data and "fibre_epi" not in out.cell_data
    assert "fibre_endo" not in m.cell_data                    # input unchanged
    out, reps = fb.apply(m, "bilayer")
    assert {"fibre_endo", "fibre_epi"} <= set(out.cell_data.keys()) and len(reps) == 2
    assert np.allclose(out.point_data["thickness"], 0.1)
    th = np.linspace(0.5, 2.0, m.n_points)
    out, _ = fb.apply(m, "bilayer", th)
    assert np.allclose(out.point_data["thickness"], th)
    with pytest.raises(ValueError):
        fb.apply(m, "bilayer", -1.0)
    bare = m.copy()
    del bare.point_data["uac_alpha"]
    with pytest.raises(ValueError, match="compute UAC"):
        fb.apply(bare, "endo")


def test_a_thickness_file_must_give_every_point_a_value(tmp_path):
    good = tmp_path / "t.txt"
    np.savetxt(good, np.full(5, 1.5))
    assert np.allclose(fb.read_thickness(str(good), 5), 1.5)
    head = tmp_path / "h.dat"
    head.write_text("5\n1\n2\n3\n4\n5\n")                   # a count line first
    assert np.allclose(fb.read_thickness(str(head), 5), [1, 2, 3, 4, 5])
    np.save(tmp_path / "t.npy", np.ones(5))
    assert len(fb.read_thickness(str(tmp_path / "t.npy"), 5)) == 5
    with pytest.raises(ValueError, match="holds 5 values; the mesh has 6"):
        fb.read_thickness(str(good), 6)
    bad = tmp_path / "b.txt"
    bad.write_text("1 2 x")
    with pytest.raises(ValueError, match="not a list"):
        fb.read_thickness(str(bad), 3)


def test_fibre_segments_are_drawn_on_a_surface():
    from ccdaf.core.ventricular_fibres import segments
    uv, t = _square()
    m = _mesh(_bent(uv), t, uv)
    m.cell_data["fibre_endo"] = _expected(m)
    lines = segments(m, "fibre_endo", 100)
    assert lines.n_lines == 100


def test_carp_export_offers_the_atrial_fibre_fields():
    from ccdaf.core.carp_export import FIBRE_FIELDS, direction_fields
    from ccdaf.core.field_transfer import AXIAL_CELL_FIELDS
    uv, t = _square()
    m = _mesh(_bent(uv), t, uv)
    m.cell_data["fibre_epi"] = _expected(m)
    assert direction_fields(m, FIBRE_FIELDS) == ["fibre_epi"]
    assert {"fibre_endo", "fibre_epi"} <= AXIAL_CELL_FIELDS


@pytest.fixture(scope="module")
def qapp():
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def test_the_window_offers_the_thickness_only_for_a_bilayer(qapp, tmp_path, monkeypatch):
    from PyQt5 import QtWidgets
    from ccdaf.gui.atrial_fibre_dialog import AtrialFibreDialog
    uv, t = _square()
    m = _mesh(_bent(uv), t, uv)
    m.point_data["wall"] = np.full(m.n_points, 2.0)
    m.point_data["LAT"] = np.linspace(-5, 5, m.n_points)     # negative: not a thickness
    d = AtrialFibreDialog(m)
    assert not d.grp_thick.isEnabled()
    d.cmb_layer.setCurrentIndex(2)
    assert d.grp_thick.isEnabled() and d.spn_fixed.value() == pytest.approx(0.1)
    assert [d.cmb_field.itemData(i) for i in range(d.cmb_field.count())] == ["wall"]
    warned = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *a: warned.append(a[2]))
    bad = tmp_path / "t.txt"
    np.savetxt(bad, np.ones(3))
    d.rb_file.setChecked(True)
    d.txt_file.setText(str(bad))
    d._accept()
    assert warned and "the mesh has" in warned[0] and d.result() is None
    d.rb_field.setChecked(True)
    d._accept()
    layer, th = d.result()
    assert layer == "bilayer" and np.allclose(th, 2.0)
