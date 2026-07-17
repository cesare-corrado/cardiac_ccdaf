"""
test_manual_editor.py
=====================
Tests for the manual editor's commit path, constructible without a live
plotter now that the host owns the X key and routes it to
``commit_pending`` — the editor no longer binds keys at construction.

The contract:

* a pending batch commits to the active label, and nothing else moves;
* committing with nothing pending is a no-op — the host fires it blind
  whenever clipping does not claim the key;
* undo restores the tags from before the last commit;
* changing the active label drops the pending batch, so two labels can
  never mix in one commit.

Uses a synthetic mesh; no display, no Qt.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pyvista as pv

from ccdaf.interaction.manual_editor import ManualEditor


def _mesh() -> pv.PolyData:
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    faces = np.hstack([[3, 0, 1, 2], [3, 1, 3, 2]])
    mesh = pv.PolyData(pts, faces)
    mesh.cell_data["elemTag"] = np.array([7, 7], dtype=np.int64)
    return mesh


def _editor(mesh: pv.PolyData, calls: list) -> ManualEditor:
    return ManualEditor(
        mesh=mesh,
        plotter=None,
        on_render=lambda: calls.append("render"),
        on_commit=lambda: calls.append("commit"),
    )


def test_commit_applies_the_active_label():
    calls: list = []
    mesh = _mesh()
    editor = _editor(mesh, calls)
    editor._pending.add(1)
    editor.commit_pending()
    assert list(mesh.cell_data["elemTag"]) == [7, 11]
    assert calls == ["render", "commit"]
    assert editor.can_undo


def test_commit_with_nothing_pending_is_a_noop():
    calls: list = []
    mesh = _mesh()
    editor = _editor(mesh, calls)
    editor.commit_pending()
    assert list(mesh.cell_data["elemTag"]) == [7, 7]
    assert calls == []
    assert not editor.can_undo


def test_undo_restores_the_previous_tags():
    calls: list = []
    mesh = _mesh()
    editor = _editor(mesh, calls)
    editor._pending.add(0)
    editor.commit_pending()
    assert list(mesh.cell_data["elemTag"]) == [11, 7]
    assert editor.undo()
    assert list(mesh.cell_data["elemTag"]) == [7, 7]
    assert not editor.can_undo


def test_label_change_drops_the_pending_batch():
    calls: list = []
    mesh = _mesh()
    editor = _editor(mesh, calls)
    editor._pending.add(0)
    editor.set_active_label(13)
    editor.commit_pending()
    assert list(mesh.cell_data["elemTag"]) == [7, 7]
