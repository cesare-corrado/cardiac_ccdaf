"""
test_clipping_accept_gate.py
============================
Reaching clipping without hunting for *Accept tagging*.

Clipping is only meaningful once the tagging is settled, so the panel's
controls wait for it. The gate itself was invisible: ticking **Clipping
active** with the tagging unaccepted left every button grey and said nothing,
which reads as a broken panel rather than a missing step — reported by a user
who reloaded a mesh to start from clipping.

Two routes open it now:

- a mesh that arrives already tagged opens it on load (and again after a
  plotter rebuild, which resets the panel);
- ticking the box accepts the tagging — silently when accepting changes no
  tag, after a confirmation when it would commit picks or paint unassigned
  triangles body, since that fill is not undoable.

Qt-only: the host is stubbed down to the parts these paths touch.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pyvista as pv
import pytest
from PyQt5 import QtWidgets

from ccdaf.app.ccdaf import CCDAF, PV_NAMES
from ccdaf.gui.clipping_widget import ClippingWidget
from ccdaf.core.mesh_loader import BODY_LABEL, UNASSIGNED


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _mesh(tags=None) -> pv.PolyData:
    mesh = pv.Sphere(theta_resolution=12, phi_resolution=12).triangulate()
    if tags is None:
        tags = np.full(mesh.n_cells, BODY_LABEL, dtype=np.int32)
    mesh.cell_data["elemTag"] = np.asarray(tags, dtype=np.int32)
    return mesh


def _tagged(mesh: pv.PolyData) -> pv.PolyData:
    """A mesh carrying a real tagging: a PV label, nothing unassigned."""
    tags = np.full(mesh.n_cells, BODY_LABEL, dtype=np.int32)
    tags[: mesh.n_cells // 4] = 11
    mesh.cell_data["elemTag"] = tags
    return mesh


class _Host:
    """The parts of the window the clipping-gate paths reach for."""

    def __init__(self, mesh: pv.PolyData):
        self._sync_clipping_gate = lambda **kw: CCDAF._sync_clipping_gate(self, **kw)
        self._accept_for_clipping = lambda: CCDAF._accept_for_clipping(self)
        self._pending_accept_changes = lambda: CCDAF._pending_accept_changes(self)
        self._accept_tagging = lambda: CCDAF._accept_tagging(self)
        self._stray_note = CCDAF._stray_note
        self._on_clipping_toggled = lambda on: CCDAF._on_clipping_toggled(self, on)

        self.loader = MagicMock(mesh=mesh, path=None)
        self.clipping_widget = ClippingWidget(
            region_names=list(PV_NAMES) + ["MV"])
        self.clipping_widget.clipping_toggled.connect(self._on_clipping_toggled)
        self.manual_widget = MagicMock()
        self.editor = MagicMock(pending_count=0, snake_active=False)
        self.clipper = None
        # The island guard is the tagger's; it leaves clean tags alone.
        self.tagger = MagicMock(
            reduce_to_single_components=lambda tags: np.asarray(tags))
        self._mark_dirty = MagicMock()
        self._render_mesh = MagicMock()
        self.statusBar = MagicMock()

    def status_text(self) -> str:
        return " ".join(str(c.args[0]) for c in self.statusBar().showMessage.mock_calls
                        if c.args)


@pytest.fixture
def host(qapp):
    return _Host(_mesh())


# ----------------------------------------------------------------------
# A mesh that arrives tagged
# ----------------------------------------------------------------------
def test_untagged_mesh_leaves_the_gate_shut(host):
    """Body everywhere is how every mesh starts — that is not a tagging."""
    assert host._sync_clipping_gate() is False
    assert not host.clipping_widget.is_accepted()


def test_tagged_mesh_opens_the_gate(host):
    _tagged(host.loader.mesh)
    assert host._sync_clipping_gate() is True
    assert host.clipping_widget.is_accepted()


def test_gate_stays_shut_while_triangles_are_unassigned(host):
    _tagged(host.loader.mesh)
    tags = np.asarray(host.loader.mesh.cell_data["elemTag"]).copy()
    tags[-1] = UNASSIGNED
    host.loader.mesh.cell_data["elemTag"] = tags
    assert host._sync_clipping_gate() is False
    assert not host.clipping_widget.is_accepted()


def test_opening_the_gate_on_load_leaves_the_file_clean(host):
    """Nothing to fix means nothing changed: no unsaved-changes prompt."""
    _tagged(host.loader.mesh)
    host._sync_clipping_gate(announce=True)
    host._mark_dirty.assert_not_called()
    assert "Clipping is available" in host.status_text()


def test_the_gate_runs_the_island_guard(host):
    """A reloaded mesh reaches export through here, so the guard runs here."""
    _tagged(host.loader.mesh)
    tags = np.asarray(host.loader.mesh.cell_data["elemTag"])
    fixed = tags.copy()
    fixed[fixed == 11] = BODY_LABEL          # pretend the patch was an island
    host.tagger.reduce_to_single_components = lambda t: fixed

    assert host._sync_clipping_gate() is True
    assert np.array_equal(
        np.asarray(host.loader.mesh.cell_data["elemTag"]), fixed)
    host._mark_dirty.assert_called_once()
    assert "stray label cell" in host.status_text()


# ----------------------------------------------------------------------
# Ticking the box
# ----------------------------------------------------------------------
def test_ticking_accepts_silently_when_nothing_would_change(host):
    """The reported case: no pending picks, nothing unassigned — just go."""
    host.clipping_widget.chk_active.setChecked(True)

    assert host.clipping_widget.chk_active.isChecked()
    assert host.clipping_widget.is_accepted()
    assert host.clipping_widget.btn_start.isEnabled()
    assert host.clipping_widget.btn_start.isEnabled()
    assert "Clipping is active" in host.status_text()


def test_ticking_asks_before_consuming_unassigned_triangles(host, monkeypatch):
    tags = np.asarray(host.loader.mesh.cell_data["elemTag"]).copy()
    tags[:5] = UNASSIGNED
    host.loader.mesh.cell_data["elemTag"] = tags
    asked = {}

    def _question(parent, title, text, *a, **kw):
        asked["text"] = text
        return QtWidgets.QMessageBox.Yes

    monkeypatch.setattr(QtWidgets.QMessageBox, "question", _question)
    host.clipping_widget.chk_active.setChecked(True)

    assert "5 still-unassigned triangles" in asked["text"]
    assert host.clipping_widget.is_accepted()
    assert host.clipping_widget.btn_start.isEnabled()


def test_declining_withdraws_the_tick_instead_of_greying_out(host, monkeypatch):
    """The failure being fixed: never leave the box on over a dead panel."""
    host.editor.pending_count = 3
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "question",
        lambda *a, **kw: QtWidgets.QMessageBox.No)

    host.clipping_widget.chk_active.setChecked(True)

    assert not host.clipping_widget.chk_active.isChecked()
    assert not host.clipping_widget.is_accepted()
    assert not host.clipping_widget.btn_start.isEnabled()
    host.editor.accept.assert_not_called()
    assert "accept the tagging" in host.status_text()


def test_ticking_with_no_mesh_says_so_and_unticks(qapp):
    host = _Host(_mesh())
    host.loader.mesh = None
    host.editor = None

    host.clipping_widget.chk_active.setChecked(True)

    assert not host.clipping_widget.chk_active.isChecked()
    assert "Load a mesh" in host.status_text()


def test_an_already_accepted_panel_does_not_re_accept(host):
    """Re-ticking is not a second accept — the mesh has moved on since."""
    host.clipping_widget.set_enabled_after_accept()
    host.clipping_widget.chk_active.setChecked(True)
    host.clipping_widget.chk_active.setChecked(False)
    host.clipping_widget.chk_active.setChecked(True)

    host.editor.accept.assert_not_called()
    assert host.clipping_widget.btn_start.isEnabled()
