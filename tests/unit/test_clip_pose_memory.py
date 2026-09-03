"""
test_clip_pose_memory.py
========================
The clipping panel's rework, in three parts.

* **Pose memory** — a sphere or plane is remembered per (region, mode) when
  its widget goes away, so reverting a clip and pressing *Start* again brings
  the geometry back where it was rather than at the region's seed default.
  Reverting a clip should cost the clip, not the placement work.

* **Undo / reset** — the one button reads through the live mode: it pops a
  contour point, or puts a sphere/plane back at the seed default. A reset is a
  widget move, not a second clip, so it must not leave a rung on the mesh undo
  stack.

* **A refused clip stays live** — a plane through its own seed cannot say which
  half to drop, so the apply is refused. The widget is still up and still
  draggable, so the buttons that drive it must stay enabled.

* **The selection re-points a live clip** — a sphere left on screen under a
  panel that now reads "plane" looks like two live widgets, so changing region
  or mode drops the pending clip and raises the new one.

* **The impossible pairing** — MV carries no tagged surface region, so the
  contour snake has nothing to travel on there. The panel disables the pairing
  in whichever combo the user did *not* just touch, so neither choice is
  silently overruled.
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
from ccdaf.gui.clipping_widget import (
    ClippingWidget, MODE_CONTOUR, MODE_SPHERE, MODE_PLANE,
)
from ccdaf.interaction.clipping_tool import ClippingTool, ClipMode


# ---------------------------------------------------------------------------
# Pose memory, on the tool itself
# ---------------------------------------------------------------------------
class _FakeSphereWidget:
    def __init__(self, center, radius):
        self._c, self._r = tuple(center), float(radius)

    def GetCenter(self):
        return self._c

    def GetRadius(self):
        return self._r

    def Off(self):
        pass


class _FakePlaneWidget:
    def __init__(self, origin, normal):
        self._o, self._n = tuple(origin), tuple(normal)

    def GetOrigin(self):
        return self._o

    def GetNormal(self):
        return self._n

    def Off(self):
        pass


def _bare_tool() -> ClippingTool:
    mesh = pv.Sphere(theta_resolution=8, phi_resolution=8).triangulate()
    mesh.cell_data["elemTag"] = np.full(mesh.n_cells, 11, dtype=np.int32)
    holder = {"mesh": mesh}
    return ClippingTool(
        mesh_getter=lambda: holder["mesh"],
        mesh_setter=lambda m: holder.__setitem__("mesh", m),
        plotter=MagicMock(),
    )


def test_cancel_files_the_sphere_pose_under_its_region():
    tool = _bare_tool()
    tool._mode = ClipMode.SPHERE
    tool._seed_key = "MV"
    tool._sphere_widget = _FakeSphereWidget((1.0, 2.0, 3.0), 4.5)

    tool.cancel()

    pose = tool.pose_for("MV", ClipMode.SPHERE)
    assert pose == {"cx": 1.0, "cy": 2.0, "cz": 3.0, "radius": 4.5}


def test_poses_are_kept_apart_per_region_and_mode():
    tool = _bare_tool()
    tool._mode = ClipMode.SPHERE
    tool._seed_key = "LSPV"
    tool._sphere_widget = _FakeSphereWidget((1.0, 0.0, 0.0), 2.0)
    tool.cancel()

    tool._mode = ClipMode.SPHERE
    tool._seed_key = "RIPV"
    tool._sphere_widget = _FakeSphereWidget((9.0, 0.0, 0.0), 3.0)
    tool.cancel()

    assert tool.pose_for("LSPV", ClipMode.SPHERE)["cx"] == 1.0
    assert tool.pose_for("RIPV", ClipMode.SPHERE)["cx"] == 9.0
    # A plane on the same region is a separate memory, still empty.
    assert tool.pose_for("LSPV", ClipMode.PLANE) is None


def test_plane_pose_records_origin_and_normal():
    tool = _bare_tool()
    tool._mode = ClipMode.PLANE
    tool._seed_key = "MV"
    tool._plane_widget = _FakePlaneWidget((1.0, 2.0, 3.0), (0.0, 0.0, 1.0))
    tool.cancel()

    assert tool.pose_for("MV", ClipMode.PLANE) == {
        "ox": 1.0, "oy": 2.0, "oz": 3.0, "nx": 0.0, "ny": 0.0, "nz": 1.0}


def test_forget_pose_erases_the_memory_not_just_the_widget():
    """Reset has to erase, or the next Start would restore what was rejected."""
    tool = _bare_tool()
    tool.remember_pose("MV", ClipMode.SPHERE,
                       {"cx": 1.0, "cy": 1.0, "cz": 1.0, "radius": 2.0})
    tool.forget_pose("MV", ClipMode.SPHERE)
    assert tool.pose_for("MV", ClipMode.SPHERE) is None


def test_pose_for_hands_out_a_copy():
    tool = _bare_tool()
    tool.remember_pose("MV", ClipMode.SPHERE,
                       {"cx": 0.0, "cy": 0.0, "cz": 0.0, "radius": 2.0})
    tool.pose_for("MV", ClipMode.SPHERE)["radius"] = 99.0
    assert tool.pose_for("MV", ClipMode.SPHERE)["radius"] == 2.0


def test_drop_snapshot_pops_one_rung_without_touching_the_mesh():
    tool = _bare_tool()
    before = tool.get_mesh().n_cells
    tool._snapshot()
    assert tool.can_undo
    tool.drop_snapshot()
    assert not tool.can_undo
    assert tool.get_mesh().n_cells == before


# ---------------------------------------------------------------------------
# The panel
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture
def panel(qapp):
    return ClippingWidget(["LSPV", "LIPV", "RSPV", "RIPV", "MV"])


def _region_item(panel, name):
    return panel.cmb_region.model().item(panel.cmb_region.findData(name))


def _mode_item(panel, mode):
    return panel.cmb_mode.model().item(panel.cmb_mode.findData(mode))


def test_mv_is_unselectable_while_the_mode_is_contour(panel):
    assert panel.selected_mode() == MODE_CONTOUR
    assert not _region_item(panel, "MV").isEnabled()
    assert _region_item(panel, "LSPV").isEnabled()


def test_choosing_a_geometry_frees_mv(panel):
    panel.cmb_mode.setCurrentIndex(panel.cmb_mode.findData(MODE_SPHERE))
    assert _region_item(panel, "MV").isEnabled()


def test_contour_is_unselectable_while_the_region_is_mv(panel):
    panel.cmb_mode.setCurrentIndex(panel.cmb_mode.findData(MODE_SPHERE))
    panel.cmb_region.setCurrentIndex(panel.cmb_region.findData("MV"))
    assert not _mode_item(panel, MODE_CONTOUR).isEnabled()


def test_the_readout_follows_the_mode(panel):
    assert panel.stack.currentIndex() == 0            # contour: nothing
    panel.cmb_mode.setCurrentIndex(panel.cmb_mode.findData(MODE_SPHERE))
    assert panel.stack.currentIndex() == 1
    panel.cmb_mode.setCurrentIndex(panel.cmb_mode.findData(MODE_PLANE))
    assert panel.stack.currentIndex() == 2


def test_showing_a_pose_does_not_echo_it_back_as_an_edit(panel):
    """Otherwise the 3D widget would drive itself in a loop through the panel."""
    seen = []
    panel.sphere_pose_edited.connect(lambda *a: seen.append(a))
    panel.set_sphere_pose({"cx": 1.0, "cy": 2.0, "cz": 3.0, "radius": 4.0})
    assert seen == []
    assert panel.sphere_pose() == {
        "cx": 1.0, "cy": 2.0, "cz": 3.0, "radius": 4.0}


def test_typing_a_radius_is_emitted_as_an_edit(panel):
    seen = []
    panel.sphere_pose_edited.connect(lambda *a: seen.append(a))
    panel.set_sphere_pose({"cx": 0.0, "cy": 0.0, "cz": 0.0, "radius": 1.0})
    panel.sph_r.setValue(7.5)
    assert seen and seen[-1] == (0.0, 0.0, 0.0, 7.5)


def test_the_diameter_tracks_the_radius(panel):
    panel.set_sphere_pose({"cx": 0.0, "cy": 0.0, "cz": 0.0, "radius": 3.0})
    assert "6.000" in panel.lbl_diam.text()
    panel.sph_r.setValue(5.0)
    assert "10.000" in panel.lbl_diam.text()


def test_start_carries_the_region_and_the_mode(panel):
    seen = []
    panel.start_requested.connect(lambda r, m: seen.append((r, m)))
    panel.cmb_mode.setCurrentIndex(panel.cmb_mode.findData(MODE_PLANE))
    panel.cmb_region.setCurrentIndex(panel.cmb_region.findData("MV"))
    # Start is gated on an accepted tagging plus the activation tick.
    panel.set_enabled_after_accept()
    panel.chk_active.setChecked(True)
    panel.btn_start.click()
    assert seen == [("MV", MODE_PLANE)]


# ---------------------------------------------------------------------------
# The host: start resumes, reset returns to the default
# ---------------------------------------------------------------------------
class _FakeClipper:
    """Records what the host asked for, without a VTK interactor in sight."""

    def __init__(self):
        self.mode = ClipMode.NONE
        self._memory = {}
        self.starts = []
        self.dropped = 0
        # What apply_* hands back: None stands for a refused clip.
        self.result = None
        self.can_undo = False

    def pose_for(self, key, mode):
        entry = self._memory.get((key, mode))
        return dict(entry) if entry else None

    def remember_pose(self, key, mode, pose):
        self._memory[(key, mode)] = dict(pose)

    def forget_pose(self, key, mode):
        self._memory.pop((key, mode), None)

    def drop_snapshot(self):
        self.dropped += 1

    def start_sphere(self, center, radius, seed_key=""):
        self.mode = ClipMode.SPHERE
        self.starts.append(("sphere", tuple(center), radius, seed_key))

    def start_plane(self, origin, normal, seed=None, seed_key=""):
        self.mode = ClipMode.PLANE
        self.starts.append(
            ("plane", tuple(origin), tuple(normal), tuple(seed), seed_key))

    def apply_sphere(self):
        return self.result

    def apply_plane(self, side_seed):
        return self.result

    def cancel(self):
        self.mode = ClipMode.NONE


class _Host:
    """The parts of the window the clip-start paths reach for."""

    def __init__(self, panel: ClippingWidget):
        self._seed_xyz = lambda name: CCDAF._seed_xyz(self, name)
        self._require_seed = lambda name: CCDAF._require_seed(self, name)
        self._default_sphere_pose = \
            lambda seed: CCDAF._default_sphere_pose(self, seed)
        self._default_plane_pose = \
            lambda seed: CCDAF._default_plane_pose(self, seed)
        self._start_sphere_clip = lambda region, **kw: \
            CCDAF._start_sphere_clip(self, region, **kw)
        self._start_plane_clip = lambda region, **kw: \
            CCDAF._start_plane_clip(self, region, **kw)
        self._after_geometric_start = \
            lambda: CCDAF._after_geometric_start(self)
        self._action_clip_undo_reset = \
            lambda: CCDAF._action_clip_undo_reset(self)
        self._action_clip_apply = lambda: CCDAF._action_clip_apply(self)
        self._action_clip_selection_changed = lambda region, mode: \
            CCDAF._action_clip_selection_changed(self, region, mode)
        self._after_clip_settled = lambda: CCDAF._after_clip_settled(self)

        mesh = pv.Sphere(theta_resolution=8, phi_resolution=8).triangulate()
        self.loader = MagicMock(mesh=mesh)
        self.plotter = MagicMock()
        self.clipping_widget = panel
        self.clipper = _FakeClipper()
        self._focus_3d = MagicMock()
        self._take_picker = MagicMock()
        self._render_mesh = MagicMock()
        self.statusBar = MagicMock()
        seeds = {name: MagicMock(xyz=np.array([float(i), 0.0, 0.0]))
                 for i, name in enumerate(
                     ("LSPV", "LIPV", "RSPV", "RIPV", "MV"))}
        self._seed_selector = lambda: MagicMock(seeds=seeds)


@pytest.fixture
def host(panel):
    return _Host(panel)


def test_first_sphere_lands_on_the_seed(host):
    host._start_sphere_clip("MV")
    kind, center, radius, key = host.clipper.starts[-1]
    assert kind == "sphere" and key == "MV"
    assert center == (4.0, 0.0, 0.0)               # the MV seed
    assert radius > 0.0


def test_start_resumes_the_remembered_sphere(host):
    """The point of the whole exercise: a revert must not cost the placement."""
    host.clipper.remember_pose(
        "MV", ClipMode.SPHERE,
        {"cx": 11.0, "cy": 12.0, "cz": 13.0, "radius": 4.0})

    host._start_sphere_clip("MV")

    _, center, radius, _ = host.clipper.starts[-1]
    assert center == (11.0, 12.0, 13.0)
    assert radius == 4.0


def test_reset_goes_back_to_the_seed_default(host):
    host.clipper.remember_pose(
        "MV", ClipMode.SPHERE,
        {"cx": 11.0, "cy": 12.0, "cz": 13.0, "radius": 4.0})

    host._start_sphere_clip("MV", reset=True)

    _, center, _, _ = host.clipper.starts[-1]
    assert center == (4.0, 0.0, 0.0)
    assert host.clipper.pose_for("MV", ClipMode.SPHERE) is None


def test_resetting_a_sphere_does_not_add_an_undo_rung(host):
    """A reset moves a widget; it is not a clip to be reverted separately."""
    host.clipping_widget.cmb_mode.setCurrentIndex(
        host.clipping_widget.cmb_mode.findData(MODE_SPHERE))
    host.clipping_widget.cmb_region.setCurrentIndex(
        host.clipping_widget.cmb_region.findData("MV"))
    host._start_sphere_clip("MV")

    host._action_clip_undo_reset()

    assert host.clipper.dropped == 1


def test_a_resumed_plane_still_knows_which_half_to_clip(host):
    """The remembered origin lies *in* the plane, so it cannot be the
    reference — the region's seed is passed separately for that."""
    host.clipper.remember_pose(
        "MV", ClipMode.PLANE,
        {"ox": 20.0, "oy": 0.0, "oz": 0.0,
         "nx": 1.0, "ny": 0.0, "nz": 0.0})

    host._start_plane_clip("MV")

    kind, origin, normal, seed, key = host.clipper.starts[-1]
    assert kind == "plane"
    assert origin == (20.0, 0.0, 0.0)
    assert seed == (4.0, 0.0, 0.0)                 # the MV seed, not the origin
    assert seed != origin


def test_a_missing_seed_stops_the_clip(host, monkeypatch):
    warned = MagicMock()
    monkeypatch.setattr(
        "ccdaf.app.ccdaf.QtWidgets.QMessageBox.warning", warned)
    host._seed_selector = lambda: MagicMock(seeds={})

    host._start_sphere_clip("MV")

    assert host.clipper.starts == []
    warned.assert_called_once()


def test_a_refused_clip_leaves_its_buttons_live(host):
    """A plane through its own seed is refused — and stays there to be moved.

    Settling the panel here would strand the user on a widget they can drag
    but no longer apply, with only Start to get out of it."""
    host._start_plane_clip("MV")
    assert host.clipping_widget.btn_apply.isEnabled()
    assert host.clipping_widget.btn_undo_reset.isEnabled()

    host.clipper.result = None                    # the tool refuses
    host._action_clip_apply()

    assert host.clipping_widget.btn_apply.isEnabled()
    assert host.clipping_widget.btn_undo_reset.isEnabled()
    host._render_mesh.assert_not_called()


def test_a_committed_clip_settles_the_panel(host):
    host._start_plane_clip("MV")
    host.clipper.result = object()                # the tool commits
    host.clipper.can_undo = True
    host._action_clip_apply()

    assert not host.clipping_widget.btn_apply.isEnabled()
    assert not host.clipping_widget.btn_undo_reset.isEnabled()
    assert host.clipping_widget.btn_revert.isEnabled()
    host._render_mesh.assert_called_once()


def test_starting_a_sphere_takes_the_surface_picker(host):
    """A widget needs no picks, but it must not leave X on another tool."""
    host._start_sphere_clip("MV")
    host._take_picker.assert_called_with("clip")


def test_changing_the_mode_swaps_a_live_widget(host):
    host._start_sphere_clip("MV")
    assert host.clipper.mode is ClipMode.SPHERE

    host._action_clip_selection_changed("MV", MODE_PLANE)

    assert host.clipper.mode is ClipMode.PLANE
    assert host.clipper.starts[-1][0] == "plane"


def test_changing_the_region_moves_a_live_widget_to_the_new_seed(host):
    host._start_sphere_clip("MV")
    host._action_clip_selection_changed("LSPV", MODE_SPHERE)

    kind, center, _, key = host.clipper.starts[-1]
    assert kind == "sphere" and key == "LSPV"
    assert center == (0.0, 0.0, 0.0)               # the LSPV seed


def test_switching_away_and_back_keeps_the_placement(host):
    host.clipper.remember_pose(
        "MV", ClipMode.SPHERE,
        {"cx": 11.0, "cy": 12.0, "cz": 13.0, "radius": 4.0})
    host._start_sphere_clip("MV")

    host._action_clip_selection_changed("MV", MODE_PLANE)
    host._action_clip_selection_changed("MV", MODE_SPHERE)

    _, center, radius, _ = host.clipper.starts[-1]
    assert center == (11.0, 12.0, 13.0) and radius == 4.0


def test_switching_to_contour_drops_the_widget_without_starting_a_snake(host):
    """Auto-starting a contour would take the X key without being asked."""
    host._start_sphere_clip("MV")
    n_starts = len(host.clipper.starts)

    host._action_clip_selection_changed("LSPV", MODE_CONTOUR)

    assert host.clipper.mode is ClipMode.NONE
    assert len(host.clipper.starts) == n_starts
    assert not host.clipping_widget.btn_apply.isEnabled()


def test_re_pointing_a_clip_leaves_no_revert_rung(host):
    host._start_sphere_clip("MV")
    host._action_clip_selection_changed("MV", MODE_PLANE)
    assert host.clipper.dropped == 1


def test_nothing_happens_when_no_clip_is_in_flight(host):
    host._action_clip_selection_changed("MV", MODE_SPHERE)
    assert host.clipper.starts == []
    assert host.clipper.dropped == 0
