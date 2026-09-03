"""
test_manual_label_sync.py
=========================
Guards the one thing the manual-correction panel promises: the combo box says
which label a pick will get.

The bug this pins: the panel pushes a label into the editor only when the
combo box *changes* (``currentIndexChanged``), while every rebuilt editor
started on its own default, 11 (LSPV). Pick 17 in the panel, then do anything
that rebuilds the editor — load a mesh, load an EAM mapping, come back from
the segmentation view — and the panel read 17 while the editor tagged 11. The
only way out was to cycle the combo box through other labels and back, which
is exactly what the user who reported it did.

The contract:

* ``ManualEditor`` is *told* its starting label; it no longer assumes one;
* the host builds every editor through ``_new_manual_editor``, which seeds it
  from the panel — so no build site can forget (checked over the source, since
  three of the four need a live GL context to reach);
* entering selection or snake mode re-adopts the panel's label as a backstop;
* ``set_active_label`` writes the label whether or not it changed, so a
  desync can always be repaired.

No display, no rendering: the plotter is a mock, and the host methods are
called unbound against a stand-in holding just what they touch.
"""

import re
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pyvista as pv
import pytest
from PyQt5 import QtWidgets

from ccdaf.app.ccdaf import CCDAF
from ccdaf.gui.manual_correction_widget import ManualCorrectionWidget
from ccdaf.interaction.manual_editor import ALLOWED_LABELS, ManualEditor
from ccdaf.core.mesh_loader import BODY_LABEL


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _mesh() -> pv.PolyData:
    mesh = pv.Sphere(theta_resolution=12, phi_resolution=12).triangulate()
    mesh.cell_data["elemTag"] = np.full(mesh.n_cells, BODY_LABEL, dtype=np.int32)
    return mesh


class _Host:
    """The parts of the window the label-sync paths reach for."""

    def __init__(self, panel: ManualCorrectionWidget, mesh: pv.PolyData):
        self._new_manual_editor = lambda m: CCDAF._new_manual_editor(self, m)
        self._build_mesh_tools = lambda: CCDAF._build_mesh_tools(self)
        self._action_edit_toggle = lambda on: CCDAF._action_edit_toggle(self, on)
        self._action_snake_toggle = lambda on: CCDAF._action_snake_toggle(self, on)
        self._replace_mesh = MagicMock()
        # Rebuilding the tools also re-reads the clipping gate from the tags;
        # that path has its own tests, so here it only has to be reachable.
        self._sync_clipping_gate = MagicMock()
        self.loader = MagicMock(mesh=mesh, path=None)
        self.plotter = MagicMock()
        self.manual_widget = panel
        self.tagger = MagicMock()
        self.editor = None
        self.clipper = None
        self._render_mesh = MagicMock()
        self._on_edit_committed = MagicMock()
        self._focus_3d = MagicMock()
        self._take_picker = MagicMock()
        self.statusBar = MagicMock()


@pytest.fixture
def host(qapp):
    panel = ManualCorrectionWidget([(lbl, str(lbl)) for lbl in ALLOWED_LABELS])
    return _Host(panel, _mesh())


def _select_label(panel: ManualCorrectionWidget, label: int) -> int:
    """Point the combo box at ``label`` the way the user does."""
    panel.set_label_index(ALLOWED_LABELS.index(label))
    assert panel.current_label() == label
    return label


def _tag_first_triangle(host: _Host) -> int:
    """Pick a triangle and commit it, as selection mode + X does."""
    mesh = host.loader.mesh
    host.editor.activate()
    host.editor.add_cell_at_point(
        np.asarray(mesh.extract_cells(0).points, dtype=float).mean(axis=0))
    host.editor.commit_pending()
    return int(np.asarray(mesh.cell_data["elemTag"])[0])


# ---------------------------------------------------------------------------
# The editor is told its label
# ---------------------------------------------------------------------------
def test_editor_starts_on_the_label_it_is_given():
    editor = ManualEditor(mesh=_mesh(), plotter=None, active_label=17)
    assert editor.active_label == 17


def test_editor_still_defaults_when_nobody_says():
    assert ManualEditor(mesh=_mesh(), plotter=None).active_label == 11


def test_editor_refuses_a_label_outside_the_allowed_set():
    with pytest.raises(ValueError):
        ManualEditor(mesh=_mesh(), plotter=None, active_label=42)


def test_set_active_label_writes_even_when_it_looks_unchanged():
    """The repair path: a desynced editor is told a label it thinks it holds."""
    editor = ManualEditor(mesh=_mesh(), plotter=None, active_label=11)
    editor.set_active_label(11)
    assert editor.active_label == 11
    editor.set_active_label(19)
    assert editor.active_label == 19


# ---------------------------------------------------------------------------
# Every build site goes through the factory
# ---------------------------------------------------------------------------
def test_new_editor_follows_the_panel(host):
    label = _select_label(host.manual_widget, 17)   # chosen before any editor
    editor = host._new_manual_editor(host.loader.mesh)
    assert editor.active_label == label


def test_rebuilding_the_mesh_tools_keeps_the_chosen_label(host):
    """The reported path: pick 17, rebuild the tools, tag — it must be 17."""
    label = _select_label(host.manual_widget, 17)
    host._build_mesh_tools()
    assert host.editor.active_label == label
    assert _tag_first_triangle(host) == label


def test_the_only_editor_built_by_hand_is_the_factorys(host):
    """Three of the four build sites need a live GL context to exercise, so
    the guard is over the source: nothing constructs a ManualEditor except
    ``_new_manual_editor``, which is what seeds the label from the panel."""
    src = (Path(__file__).resolve().parents[2]
           / "src" / "ccdaf" / "app" / "ccdaf.py").read_text()
    bodies = re.split(r"\n    def ", src)
    offenders = [
        b.split("(", 1)[0] for b in bodies
        if "ManualEditor(" in b and not b.startswith("_new_manual_editor")
    ]
    assert offenders == [], (
        f"these build a ManualEditor directly: {offenders} — "
        "use _new_manual_editor so the panel's label is carried over")


# ---------------------------------------------------------------------------
# Entering a picking mode re-adopts the panel's label
# ---------------------------------------------------------------------------
# The stand-in never wires ``label_changed`` to the host, so moving the combo
# box leaves the editor behind — the desync, staged.
def test_selection_mode_adopts_the_panel_label(host):
    host._build_mesh_tools()
    label = _select_label(host.manual_widget, 15)
    assert host.editor.active_label != label

    host._action_edit_toggle(True)
    assert host.editor.active_label == label
    assert _tag_first_triangle(host) == label


def test_snake_mode_adopts_the_panel_label(host):
    host._build_mesh_tools()
    label = _select_label(host.manual_widget, 13)
    assert host.editor.active_label != label

    host._action_snake_toggle(True)
    assert host.editor.active_label == label
