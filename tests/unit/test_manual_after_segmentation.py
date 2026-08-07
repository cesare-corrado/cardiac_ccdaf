"""
test_manual_after_segmentation.py
=================================
Guards the mesh-side tools when a new working mesh is adopted — above all when
that mesh is *born* in the converter.

The bug this pins: a segmentation loaded from a NIfTI with no mesh ever open
produced a surface that had no ``ManualEditor`` behind it.
``_action_seg_to_vtk`` closes the segmentation (which tears the tools down)
*before* it adopts the converted surface, so ``_exit_segmentation_mode`` saw
``loader.mesh is None`` and skipped ``_build_mesh_tools``; ``_replace_mesh``
then refreshed only an editor that already existed. Tagging enables the
manual-correction panel regardless, so every control looked live and silently
did nothing — no triangle selection, no snake. The clipper was missing the
same way, which would have killed the clipping step next.

The contract, all of it on ``_replace_mesh`` — the one place every new working
mesh passes through (converter, post-processing, clipping):

* an editor always comes out of it, bound to the new mesh, whether or not one
  went in;
* a clipper is built when there is none, but a live one is left alone: it owns
  the clip undo stack and calls ``_replace_mesh`` itself;
* the panel is switched on when the editor is born there, so "enabled panel"
  and "there is an editor" cannot drift apart again;
* the new editor tags with the label the panel shows, not its own default.

``_replace_mesh`` is called unbound against a stand-in host holding just the
attributes it touches. Constructing the real window needs a live GL context,
which aborts VTK on a headless box; the end-to-end sequence is exercised by
the validation script instead. No display, no rendering.
"""

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
from ccdaf.interaction.clipping_tool import ClippingTool
from ccdaf.interaction.manual_editor import ALLOWED_LABELS, ManualEditor
from ccdaf.core.mesh_loader import BODY_LABEL


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _mesh(tag: int = BODY_LABEL) -> pv.PolyData:
    """A tagged triangular surface — stands in for the converted segmentation."""
    mesh = pv.Sphere(theta_resolution=12, phi_resolution=12).triangulate()
    mesh.cell_data["elemTag"] = np.full(mesh.n_cells, tag, dtype=np.int32)
    return mesh


class _Host:
    """The parts of the window ``_replace_mesh`` reaches for.

    ``_replace_mesh`` is the real method, bound here — the clipper is handed
    it as its ``mesh_setter``, so it has to be reachable on the host.
    """

    def __init__(self, panel: ManualCorrectionWidget):
        self._replace_mesh = lambda mesh: CCDAF._replace_mesh(self, mesh)
        self._new_manual_editor = lambda mesh: CCDAF._new_manual_editor(self, mesh)
        self.loader = MagicMock(mesh=None, path=None)
        self.plotter = MagicMock()
        self.manual_widget = panel
        self.mesh_info = MagicMock()
        self.tagger = None
        self.editor = None
        self.clipper = None
        self._populate_fields = MagicMock()
        self._render_mesh = MagicMock()
        self._on_edit_committed = MagicMock()
        self.statusBar = MagicMock()


@pytest.fixture
def host(qapp):
    """A host with the panel as it is before tagging: controls disabled."""
    panel = ManualCorrectionWidget(
        [(lbl, str(lbl)) for lbl in ALLOWED_LABELS])
    return _Host(panel)


def _adopt(host: _Host, mesh: pv.PolyData) -> None:
    host._replace_mesh(mesh)


def _tag_first_triangle(host: _Host) -> int:
    """Pick a triangle and commit it, as selection mode + X does."""
    mesh = host.loader.mesh
    host.editor.activate()
    host.editor.add_cell_at_point(
        np.asarray(mesh.extract_cells(0).points, dtype=float).mean(axis=0))
    host.editor.commit_pending()
    return int(np.asarray(mesh.cell_data["elemTag"])[0])


# ---------------------------------------------------------------------------
# An editor always comes out of it
# ---------------------------------------------------------------------------
def test_editor_is_built_when_the_mesh_is_born_here(host):
    mesh = _mesh()
    _adopt(host, mesh)
    assert isinstance(host.editor, ManualEditor)
    assert host.editor.mesh is mesh


def test_existing_editor_is_rebound_to_the_new_mesh(host):
    _adopt(host, _mesh())
    first = host.editor
    second_mesh = _mesh()
    _adopt(host, second_mesh)
    assert host.editor is not first
    assert host.editor.mesh is second_mesh


def test_tagger_follows_the_new_mesh(host):
    mesh = _mesh()
    _adopt(host, mesh)
    assert host.tagger is not None
    assert host.tagger.mesh is mesh


# ---------------------------------------------------------------------------
# The clipper: built when missing, never replaced when live
# ---------------------------------------------------------------------------
def test_clipper_is_built_when_there_is_none(host):
    _adopt(host, _mesh())
    assert isinstance(host.clipper, ClippingTool)


def test_live_clipper_is_left_alone(host):
    """It owns the clip undo stack and calls _replace_mesh from its own
    mesh_setter — rebuilding it there would drop the clips already applied."""
    _adopt(host, _mesh())
    clipper = host.clipper
    _adopt(host, _mesh())
    assert host.clipper is clipper


# ---------------------------------------------------------------------------
# Panel and editor cannot drift apart
# ---------------------------------------------------------------------------
def test_panel_goes_live_with_the_new_editor(host):
    assert not host.manual_widget.btn_edit_toggle.isEnabled()
    _adopt(host, _mesh())
    assert host.manual_widget.btn_edit_toggle.isEnabled()
    assert host.manual_widget.btn_snake.isEnabled()
    assert host.editor is not None


def test_an_enabled_panel_always_has_an_editor(host):
    """The invariant that broke: tagging enables the panel unconditionally."""
    _adopt(host, _mesh())
    host.manual_widget.set_active(True)          # what tagging does
    assert host.manual_widget.btn_edit_toggle.isEnabled()
    assert host.editor is not None


def test_undo_starts_empty_on_the_new_mesh(host):
    _adopt(host, _mesh())
    assert not host.manual_widget.btn_undo.isEnabled()
    assert not host.editor.can_undo


# ---------------------------------------------------------------------------
# The new editor follows the panel
# ---------------------------------------------------------------------------
def test_new_editor_tags_with_the_label_the_panel_shows(host):
    host.manual_widget.set_label_index(2)        # chosen before any editor
    label = host.manual_widget.current_label()
    assert label != 11                           # not the editor's own default
    _adopt(host, _mesh())
    assert _tag_first_triangle(host) == label


def test_editing_the_adopted_mesh_writes_its_tags(host):
    mesh = _mesh()
    _adopt(host, mesh)
    assert _tag_first_triangle(host) == host.manual_widget.current_label()
    assert host.editor.can_undo
