"""
test_clip_undo.py
=================
Undo for the clipping tool:

* **Undo last point** (``ClippingTool.undo_last_point``) — while building the
  PV contour snake, pop the most recently placed geodesic point and restore
  the exact prior snake state (path, head, tail, count). Bidirectional growth
  means the last point may have extended either endpoint, so undo works from
  whole-state snapshots taken before each accepted pick.

* **Mesh-history undo for mitral** (``ClippingTool.can_undo`` / ``restore``) —
  every clip pushes a pre-clip snapshot; after a mitral clip is finalized the
  snapshot survives, so ``can_undo`` is True and ``restore`` brings the mesh
  back. This is what lets the host keep its revert button live for mitral.

A minimal fake plotter stands in for the Qt/VTK plotter so the snake logic is
exercised headlessly (only actor add/remove and picking toggles are touched).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pyvista as pv

from ccdaf.interaction.clipping_tool import ClippingTool, ClipMode


class _FakePlotter:
    """Just enough plotter surface for the snake / finalize code paths."""

    def enable_point_picking(self, **kwargs):
        pass

    def disable_picking(self):
        pass

    def add_mesh(self, mesh, name=None, **kwargs):
        return object()

    def remove_actor(self, actor, reset_camera=False):
        pass

    def render(self):
        pass


def _tagged_sphere(tag: int = 11):
    """A triangulated sphere whose every cell carries a single tag.

    With one tag no vertex is a tag boundary, so the whole surface is an
    allowed snake region — every pick lands validly."""
    mesh = pv.Sphere(theta_resolution=24, phi_resolution=24).triangulate()
    mesh.cell_data["elemTag"] = np.full(mesh.n_cells, tag, dtype=np.int32)
    return mesh


def _pv_tool_on_sphere(tag: int = 11):
    mesh = _tagged_sphere(tag)
    holder = {"mesh": mesh}
    tool = ClippingTool(
        mesh_getter=lambda: holder["mesh"],
        mesh_setter=lambda m: holder.__setitem__("mesh", m),
        plotter=_FakePlotter(),
    )
    return mesh, holder, tool


def _pick(tool, mesh, vid: int) -> None:
    tool._on_contour_pick(tuple(np.asarray(mesh.points[vid], dtype=float)))


# ---------------------------------------------------------------------------
# undo_last_point — PV snake
# ---------------------------------------------------------------------------
def test_undo_last_point_restores_previous_snake_step_by_step():
    mesh, _, tool = _pv_tool_on_sphere(11)
    tool.start_pv_contour(pv_label=11)

    v0 = 0
    far = mesh.n_points - 1
    mid = mesh.n_points // 2

    _pick(tool, mesh, v0)
    assert tool._pick_count == 1
    _pick(tool, mesh, far)
    assert tool._pick_count == 2
    two_path = list(tool._path)
    two_head, two_tail = tool._head, tool._tail

    _pick(tool, mesh, mid)
    assert tool._pick_count == 3
    assert tool._path != two_path                     # the third pick changed it

    # Undo the third point → exactly the two-point snake is back.
    assert tool.undo_last_point() == 2
    assert tool._path == two_path
    assert tool._head == two_head and tool._tail == two_tail

    # Undo to one point, then to empty, then nothing left to undo.
    assert tool.undo_last_point() == 1
    assert tool._path == [v0]
    assert tool._head == v0 and tool._tail == v0
    assert tool.undo_last_point() == 0
    assert tool._path == [] and tool._head == -1 and tool._tail == -1
    assert tool.undo_last_point() == -1               # empty pick history


def test_duplicate_pick_records_single_undo_step():
    """pyvista fires the pick callback twice per X press (EndPickEvent
    observer + pick_at_cursor's explicit call). Each placed point must still
    cost exactly one undo — the phantom repeat on the same vertex is a no-op."""
    mesh, _, tool = _pv_tool_on_sphere(11)
    tool.start_pv_contour(pv_label=11)

    v0, far = 0, mesh.n_points - 1
    _pick(tool, mesh, v0)
    _pick(tool, mesh, v0)                              # phantom repeat
    assert tool._pick_count == 1
    _pick(tool, mesh, far)
    _pick(tool, mesh, far)                             # phantom repeat
    assert tool._pick_count == 2

    # One undo drops the second point and restores the single-point snake.
    assert tool.undo_last_point() == 1
    assert tool._path == [v0]
    assert tool.undo_last_point() == 0
    assert tool._path == []


def test_undo_last_point_is_noop_without_pv_contour():
    _, _, tool = _pv_tool_on_sphere(11)
    assert tool.mode is ClipMode.NONE
    assert tool.undo_last_point() == -1


def test_start_pv_contour_clears_stale_pick_history():
    mesh, _, tool = _pv_tool_on_sphere(11)
    tool.start_pv_contour(pv_label=11)
    _pick(tool, mesh, 0)
    _pick(tool, mesh, mesh.n_points - 1)
    assert tool._pick_count == 2

    # A fresh session starts empty — no carry-over undo.
    tool.start_pv_contour(pv_label=11)
    assert tool._pick_count == 0
    assert tool._pick_history == []
    assert tool.undo_last_point() == -1


# ---------------------------------------------------------------------------
# mesh-history undo — mitral clip
# ---------------------------------------------------------------------------
def test_mitral_clip_is_undoable_via_history():
    mesh = _tagged_sphere(1)
    holder = {"mesh": mesh}
    tool = ClippingTool(
        mesh_getter=lambda: holder["mesh"],
        mesh_setter=lambda m: holder.__setitem__("mesh", m),
        plotter=_FakePlotter(),
    )
    original_cells = mesh.n_cells

    tool._mode = ClipMode.MV_SPHERE
    tool._snapshot()                                  # pre-clip snapshot
    assert tool.can_undo

    keep_mask = np.ones(mesh.n_cells, dtype=bool)
    keep_mask[:10] = False                            # remove 10 triangles
    res = tool._finalize_mv_clip(holder["mesh"], keep_mask)

    assert res.n_removed == 10
    assert holder["mesh"].n_cells == original_cells - 10
    # The snapshot survives the finalize, so the clip can be undone.
    assert tool.can_undo

    tool.restore()
    assert holder["mesh"].n_cells == original_cells
    assert not tool.can_undo


def test_can_undo_reflects_history_stack():
    _, _, tool = _pv_tool_on_sphere(11)
    assert not tool.can_undo
    tool._snapshot()
    assert tool.can_undo
    tool.restore()
    assert not tool.can_undo
