"""
test_selection_keys.py
======================
Guards the manual-correction key map: **X picks, C commits**.

Selection mode used to pick with the mouse and commit with **X** — the same
key the snake and the PV contour use to *drop a point*. One key meant two
opposite things ("add this" vs "done adding") depending on a mode the user
could not see, and the mouse did the picking in one tool while the keyboard
did it in the others. Now X is the pick key everywhere and C is the commit
key, so the two halves of an edit are never the same keystroke.

The contract:

* **C** commits the pending triangle batch to the active label;
* **X** never commits — with a batch pending and no snake in flight, X does
  not write tags;
* both keys are bound by the host, once per plotter, so a view rebuild cannot
  leave one of them dead;
* C stays out of the way of the tools it does not own: no editor, a snake in
  flight, or nothing pending, and it does nothing.

The host methods are called unbound against a stand-in holding just what they
touch — constructing the window needs a live GL context.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pyvista as pv
import pytest

from ccdaf.app.ccdaf import CCDAF
from ccdaf.core.mesh_loader import BODY_LABEL
from ccdaf.interaction.manual_editor import ManualEditor

CCDAF_SRC = (Path(__file__).resolve().parents[2]
             / "src" / "ccdaf" / "app" / "ccdaf.py").read_text()


def _mesh() -> pv.PolyData:
    mesh = pv.Sphere(theta_resolution=16, phi_resolution=16).triangulate()
    mesh.cell_data["elemTag"] = np.full(mesh.n_cells, BODY_LABEL, dtype=np.int32)
    return mesh


class _Host:
    """The parts of the window the key routing reaches for."""

    def __init__(self, editor):
        self._on_x_key = lambda: CCDAF._on_x_key(self)
        self._on_c_key = lambda: CCDAF._on_c_key(self)
        self.editor = editor
        self.clipper = None
        self.clipping_widget = MagicMock()
        self.statusBar = MagicMock()


@pytest.fixture
def mesh():
    return _mesh()


@pytest.fixture
def editor(mesh):
    ed = ManualEditor(mesh=mesh, plotter=MagicMock(), active_label=17)
    ed.activate()
    return ed


@pytest.fixture
def host(editor):
    host = _Host(editor)
    host.clipping_widget.is_clipping_enabled.return_value = False
    return host


def _pick(editor: ManualEditor, mesh: pv.PolyData, cell_id: int) -> None:
    """Land a pick on triangle ``cell_id`` — a point inside its face."""
    tris = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]
    a, b, c = mesh.points[tris[cell_id]]
    editor._on_cell_picked(tuple(float(v) for v in 0.5 * a + 0.3 * b + 0.2 * c))


# ---------------------------------------------------------------------------
# C commits
# ---------------------------------------------------------------------------
def test_c_commits_the_pending_batch(host, editor, mesh):
    _pick(editor, mesh, 100)
    _pick(editor, mesh, 200)
    assert editor.pending_count == 2

    host._on_c_key()

    tags = np.asarray(mesh.cell_data["elemTag"])
    assert tags[100] == 17 and tags[200] == 17
    assert editor.pending_count == 0
    assert editor.can_undo


def test_c_reports_what_it_committed(host, editor, mesh):
    _pick(editor, mesh, 100)
    host._on_c_key()
    msg = host.statusBar().showMessage.call_args.args[0]
    assert "1 triangle" in msg and "17" in msg


def test_c_does_nothing_with_nothing_pending(host, editor, mesh):
    before = np.asarray(mesh.cell_data["elemTag"]).copy()
    host._on_c_key()
    assert np.array_equal(np.asarray(mesh.cell_data["elemTag"]), before)
    assert not editor.can_undo
    assert not host.statusBar().showMessage.called


def test_c_is_inert_without_an_editor(host):
    host.editor = None
    host._on_c_key()          # must not raise


def test_c_leaves_the_snake_alone(host, editor, mesh):
    """The snake commits from its own button — C must not fire it."""
    _pick(editor, mesh, 100)
    editor._snake_active = True
    before = np.asarray(mesh.cell_data["elemTag"]).copy()

    host._on_c_key()

    assert np.array_equal(np.asarray(mesh.cell_data["elemTag"]), before)
    assert editor.pending_count == 1


# ---------------------------------------------------------------------------
# X no longer commits
# ---------------------------------------------------------------------------
def test_x_does_not_commit_the_batch(host, editor, mesh):
    """The bug this pins: X used to mean 'commit' here and 'pick' everywhere
    else. It must no longer write tags."""
    _pick(editor, mesh, 100)
    before = np.asarray(mesh.cell_data["elemTag"]).copy()

    host._on_x_key()

    assert np.array_equal(np.asarray(mesh.cell_data["elemTag"]), before)
    assert not editor.can_undo


# ---------------------------------------------------------------------------
# Both keys are bound, once per plotter
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key,slot", [("x", "_on_x_key"), ("c", "_on_c_key")])
def test_key_is_bound_by_the_host(key, slot):
    """Binding happens where the plotter is built, which needs a live GL
    context — so the guard is over the source: both keys are registered, in
    upper and lower case, on the one method that owns them."""
    assert f'for key in ("{key}", "{key.upper()}"):' in CCDAF_SRC
    assert f"self.plotter.add_key_event(key, self.{slot})" in CCDAF_SRC
