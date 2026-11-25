"""
test_carp_bilayer.py
====================
The CARP bilayer export (:func:`ccdaf.core.carp_export.write_bilayer`):
an endocardial layer, an epicardial copy moved outwards by the thickness,
and one line element per node, with their tags and fibres.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pytest
import pyvista as pv
from PyQt5 import QtWidgets

from ccdaf.core.carp_export import (
    CarpExportOptions, bilayer_availability, write_carp,
)

R = 20.0                      # mm


def _atrium():
    """A hemisphere (open like an atrium) with bilayer fibres and tags."""
    s = pv.Sphere(radius=R, theta_resolution=24, phi_resolution=24,
                  end_phi=120).triangulate().clean()
    n = s.n_cells
    s.cell_data["elemTag"] = np.where(np.arange(n) % 2, 1, 11).astype(np.int32)
    c = np.asarray(s.cell_centers().points)
    s.cell_data["fibre_endo"] = np.cross(c, [0, 0, 1.0]) + 1e-9
    s.cell_data["fibre_epi"] = np.cross(c, [1.0, 0, 0]) + 1e-9
    s.point_data["thickness"] = np.full(s.n_points, 0.1)
    return s


def _read(prefix):
    P = np.loadtxt(f"{prefix}.pts", skiprows=1)
    rows = [ln.split() for ln in open(f"{prefix}.elem").read().splitlines()[1:]]
    lon = np.loadtxt(f"{prefix}.lon", skiprows=1)
    return P, rows, lon, open(f"{prefix}.lon").readline().strip()


def _options(tmp_path, **kw):
    base = dict(directory=str(tmp_path), name="bi", scale=1000.0,
                region_field="elemTag", bilayer=True,
                thickness_field="thickness", mm_per_unit=1.0)
    base.update(kw)
    return CarpExportOptions(**base)


def test_the_bilayer_has_two_layers_and_one_line_per_node(tmp_path):
    s = _atrium()
    res = write_carp(s, _options(tmp_path))
    P, rows, lon, header = _read(tmp_path / "bi")
    n = s.n_points
    assert len(P) == 2 * n
    tr = [r for r in rows if r[0] == "Tr"]
    ln = np.array([[int(x) for x in r[1:4]] for r in rows if r[0] == "Ln"])
    assert len(tr) == 2 * s.n_cells and len(ln) == n
    assert np.array_equal(ln[:, 1] - ln[:, 0], np.full(n, n))
    assert header == "1" and len(lon) == len(rows)
    assert res.n_elements == len(rows)


def test_the_epicardium_lies_outside_by_the_thickness(tmp_path):
    s = _atrium()
    write_carp(s, _options(tmp_path))
    P, *_ = _read(tmp_path / "bi")
    n = s.n_points
    r_endo = np.linalg.norm(P[:n], axis=1)
    r_epi = np.linalg.norm(P[n:], axis=1)
    assert np.allclose(r_epi - r_endo, 100.0, atol=0.5)        # 0.1 mm in um


def test_tags_follow_the_layers(tmp_path):
    s = _atrium()
    res = write_carp(s, _options(tmp_path, epi_tag_offset=100, line_tag=32))
    assert set(res.tags) == {1, 11, 101, 111, 32}
    assert res.tags[32] == s.n_points


def test_fibres_follow_the_layers_and_the_lines(tmp_path):
    s = _atrium()
    write_carp(s, _options(tmp_path))
    P, rows, lon, _ = _read(tmp_path / "bi")
    m, n = s.n_cells, s.n_points

    def unit(v):
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    assert np.allclose(lon[:m], unit(np.asarray(s.cell_data["fibre_endo"])), atol=1e-5)
    assert np.allclose(lon[m:2 * m], unit(np.asarray(s.cell_data["fibre_epi"])), atol=1e-5)
    line = unit(P[n:] - P[:n])
    assert np.allclose(np.abs(np.sum(lon[2 * m:] * line, 1)), 1.0, atol=1e-5)


def test_a_fixed_thickness_without_a_field(tmp_path):
    s = _atrium()
    write_carp(s, _options(tmp_path, thickness_field=None, thickness_mm=0.2))
    P, *_ = _read(tmp_path / "bi")
    n = s.n_points
    d = np.linalg.norm(P[n:] - P[:n], axis=1)
    assert np.allclose(d, 200.0, atol=0.01)


def test_a_mesh_in_micrometres_moves_by_the_thickness_in_mm(tmp_path):
    s = _atrium()
    s.points = np.asarray(s.points) * 1000.0
    write_carp(s, _options(tmp_path, scale=1.0, mm_per_unit=1e-3))
    P, *_ = _read(tmp_path / "bi")
    n = s.n_points
    assert np.allclose(np.linalg.norm(P[n:] - P[:n], axis=1), 100.0, atol=0.01)


def test_clashing_tags_are_reported(tmp_path):
    s = _atrium()
    res = write_carp(s, _options(tmp_path, epi_tag_offset=10, line_tag=11),
                     dry_run=True)
    text = " ".join(res.warnings)
    assert "share a tag" in text and "line tag 11" in text


def test_no_bilayer_without_both_fibres():
    s = _atrium()
    del s.cell_data["fibre_epi"]
    ok, why = bilayer_availability(s)
    assert not ok and "bilayer" in why
    ok, _ = bilayer_availability(pv.ImageData(dimensions=(3, 3, 3))
                                 .cast_to_unstructured_grid().triangulate())
    assert not ok


@pytest.fixture(scope="module")
def app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_the_dialog_offers_the_bilayer_as_a_fibre_choice(app, tmp_path):
    from ccdaf.gui.carp_export_dialog import BILAYER_ENTRY, CarpExportDialog
    dlg = CarpExportDialog(_atrium(), start_dir=str(tmp_path), default_name="bi")
    idx = dlg.cmb_fibre.findData(BILAYER_ENTRY)
    assert idx >= 0
    assert dlg.box_bilayer.isHidden()
    dlg.cmb_fibre.setCurrentIndex(idx)
    assert not dlg.box_bilayer.isHidden()
    assert not dlg.cmb_sheet.isEnabled()
    opts = dlg.options()
    assert opts.bilayer and opts.thickness_field == "thickness" and opts.fibre_field is None
    assert (opts.epi_tag_offset, opts.line_tag) == (100, 32)


def test_without_a_thickness_field_the_layers_are_0_1_mm_apart(app, tmp_path):
    from ccdaf.gui.carp_export_dialog import BILAYER_ENTRY, CarpExportDialog
    s = _atrium()
    del s.point_data["thickness"]
    dlg = CarpExportDialog(s, start_dir=str(tmp_path), default_name="bi")
    dlg.cmb_fibre.setCurrentIndex(dlg.cmb_fibre.findData(BILAYER_ENTRY))
    opts = dlg.options()
    assert opts.thickness_field is None and opts.thickness_mm == 0.1


def test_no_bilayer_choice_without_both_fibres(app, tmp_path):
    from ccdaf.gui.carp_export_dialog import BILAYER_ENTRY, CarpExportDialog
    s = _atrium()
    del s.cell_data["fibre_epi"]
    dlg = CarpExportDialog(s, start_dir=str(tmp_path), default_name="bi")
    assert dlg.cmb_fibre.findData(BILAYER_ENTRY) < 0
