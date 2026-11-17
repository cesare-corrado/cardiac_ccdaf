"""
test_egm_display.py
===================
Tests for the electrogram view: the pick helpers in ``io.carto_egm``, the
``EGMPanel`` widget, and the app's picking / selection hooks.

Uses the synthetic Carto map of ``test_carto_egm`` (3 splines x 4
electrodes, three placements, one with an undecidable missing spline).
The app methods run unbound on a stub host, as elsewhere in this suite:
a real window needs a GL context.

Offscreen Qt, no plotter.
"""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pytest
from PyQt5 import QtWidgets

from ccdaf.io.carto_egm import (
    load_carto_egm, mapping_catheter, nearest_bipole, placement_bipoles,
    scan_carto_egm,
)
from test_carto_egm import MAP, carto_map, electrode_xyz  # noqa: F401 (fixture)


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture
def egm(carto_map):  # noqa: F811
    return load_carto_egm(scan_carto_egm(str(carto_map), MAP), extra_channels=["V1"])


# --- helpers ---------------------------------------------------------------

def test_placement_bipoles(egm):
    first, second, has_point = placement_bipoles(egm, 0)
    # bipoles 20A_1-2, 20A_5-6, 20A_7-8, 20A_11-12 at placement 0
    assert first[0].tolist() == electrode_xyz(0, 1)
    assert second[0].tolist() == electrode_xyz(0, 2)
    assert first[3].tolist() == electrode_xyz(0, 11)
    assert has_point.tolist() == [True, False, False, True]


def test_nearest_bipole_among_points(egm):
    # Point 3 is bipole 1 (20A_5-6) at placement 1.
    assert nearest_bipole(egm, np.array(electrode_xyz(1, 5)) + 0.01) == (1, 1)


def test_nearest_bipole_uses_displayed_point_positions(egm):
    # The app passes the electrodes as displayed (moved by smoothing); a point
    # is found where it is drawn, not where Carto put it.
    car = np.array([electrode_xyz(p, k) for _, _, p, k in
                    [(1, 1000, 0, 1), (2, 1000, 0, 11), (3, 2000, 1, 5),
                     (4, 2000, 1, 11), (5, 3000, 2, 7)]], dtype=float)
    moved = car + np.array([50.0, 0.0, 0.0])
    assert nearest_bipole(egm, moved[4], car_points=moved) == (2, 2)


def test_non_point_bipoles_pickable_only_on_show(egm):
    # 20A_5-6 at placement 0 is not a point: unreachable until placement 0
    # is the one on show.
    xyz = np.array(electrode_xyz(0, 5))
    assert nearest_bipole(egm, xyz) != (0, 1)
    assert nearest_bipole(egm, xyz, placement=0) == (0, 1)


# --- panel -------------------------------------------------------------------

def _panel(qapp, egm):
    from ccdaf.gui.egm_panel import EGMPanel
    p = EGMPanel()
    p.resize(600, 700)
    p.set_egm(egm, lat_by_row=[5.0, 5.0, 5.0, 5.0, -10000.0])
    p.show_bipole(0, 1)
    return p


def test_panel_rows_grouped_per_bipole(qapp, egm):
    p = _panel(qapp, egm)
    assert p.row_labels() == [
        "20A_1-2", "20A_1", "20A_2", "20A_5-6", "20A_5", "20A_6",
        "20A_7-8", "20A_7", "20A_8", "20A_11-12", "20A_11", "20A_12", "V1"]
    assert p.row_bipoles()[:4] == [0, 0, 0, 1] and p.row_bipoles()[-1] is None


def test_panel_type_ticks_hide_rows(qapp, egm):
    from ccdaf.gui.egm_panel import UNI, EXTRA
    p = _panel(qapp, egm)
    p._ticks[UNI].setChecked(False)
    p._ticks[EXTRA].setChecked(False)
    assert p.row_labels() == ["20A_1-2", "20A_5-6", "20A_7-8", "20A_11-12"]


def test_panel_marks_a_channel_missing_from_the_placement(qapp, egm):
    p = _panel(qapp, egm)
    p.show_bipole(2, 0)                       # V1 is not in placement 3000's file
    labels = [str(t.get_text()) for t in p._ax.get_yticklabels()]
    assert labels[-1] == "V1 (not recorded)"


def test_panel_draws_a_saturated_channel(qapp, egm):
    # A trace stuck at the recorder's limit is a real recording, not a gap.
    col = egm["channel_names"].index("V1")
    egm["traces"][0, col] = -32768
    p = _panel(qapp, egm)
    p.show_bipole(0, 0)
    labels = [str(t.get_text()) for t in p._ax.get_yticklabels()]
    assert labels[-1] == "V1"


def test_reference_follows_toolbar_zoom_after_redraws(qapp, egm):
    # Zoom and pan act on the trace axes; the reference strip must follow,
    # also after the panel has been redrawn (a new pick, a gain change).
    p = _panel(qapp, egm)
    p.show_bipole(0, 0)
    p.show_bipole(1, 1)
    p._ax.set_xlim(-20.0, 30.0)                # what the toolbar does
    assert p._ref_ax.get_xlim() == pytest.approx((-20.0, 30.0))
    p._bump("Bipolar", 1.5)                    # a redraw keeps the range
    assert p._ax.get_xlim() == pytest.approx((-20.0, 30.0))
    p._ax.set_xlim(0.0, 10.0)
    assert p._ref_ax.get_xlim() == pytest.approx((0.0, 10.0))


def test_pan_and_zoom_move_time_only(qapp, egm):
    p = _panel(qapp, egm)
    p.show_bipole(0, 0)
    rows = p._ax.get_ylim()
    p._ax.set_xlim(-1990.0, -1960.0)
    p._ax.set_ylim(rows[0] + 2.0, rows[1] + 2.0)      # a vertical pan
    assert p._ax.get_ylim() == pytest.approx(rows)
    assert p._ax.get_xlim() == pytest.approx((-1990.0, -1960.0))
    assert p._ref_ax.get_xlim() == pytest.approx((-1990.0, -1960.0))
    p._bump("Bipolar", 1.5)                            # still locked after a redraw
    p._ax.set_ylim(-1.0, 0.0)
    assert p._ax.get_ylim() == pytest.approx(rows)
    # The toolbar's own code paths: a diagonal pan drag and a zoom box.
    ax = p._ax
    x0 = ax.get_xlim()
    cx, cy = ax.transAxes.transform((0.5, 0.5))
    ax.start_pan(cx, cy, 1)
    ax.drag_pan(1, None, cx + 40, cy + 40)
    ax.end_pan()
    assert ax.get_ylim() == pytest.approx(rows)
    assert ax.get_xlim() != pytest.approx(x0)            # time did move
    ax._set_view_from_bbox((cx - 30, cy - 30, cx + 30, cy + 30))
    assert ax.get_ylim() == pytest.approx(rows)
    assert p._ref_ax.get_xlim() == pytest.approx(ax.get_xlim())


def test_scale_bar_value():
    from ccdaf.gui.egm_panel import SCALE_BAR_MAX_ROW, scale_bar_mv
    assert scale_bar_mv(1.0) == 0.5            # 1 mV would be a full row
    assert scale_bar_mv(0.01) == 50.0          # a very small trace scale
    assert scale_bar_mv(1e4) == 0.01           # nothing fits: the smallest
    for u in (0.003, 0.4, 2.5, 37.0):
        assert scale_bar_mv(u) * u <= SCALE_BAR_MAX_ROW


def test_scale_bars_follow_the_gain(qapp, egm):
    from ccdaf.gui.egm_panel import BIP, REF, SCALE_BAR_MAX_ROW
    p = _panel(qapp, egm)
    before = p.scale_bars()
    p._bump(BIP, 10.0)
    after = p.scale_bars()
    assert after[BIP] < before[BIP] and after[REF] == before[REF]
    assert after[BIP] * p._units_per_mv(BIP) <= SCALE_BAR_MAX_ROW
    texts = [t.get_text() for t in p._ax.texts]
    assert f"{after[BIP]:g} mV" in texts       # drawn on the first bipolar row


def test_readout_gives_channel_time_and_mv(qapp, egm):
    # Synthetic traces: every sample is 100 * placement + the channel's
    # position in the ECG file (20A_1-2 is 13th from 0, CS1-CS2 0th), 50
    # samples, reference annotation at sample 2000 (so t runs -2000..-1951).
    p = _panel(qapp, egm)
    p.show_bipole(1, 0)
    assert p._row_coord(-1990.0, 0.0) == "20A_1-2   t = -1990 ms   0.339 mV"
    assert p._ref_coord(-1980.2, 0.0) == "CS1-CS2   t = -1980 ms   0.300 mV"
    assert p._ref_coord(0.0, 0.0) == "CS1-CS2   t = 0 ms"      # past the end
    p.show_bipole(2, 0)
    v1 = p.row_labels().index("V1")
    assert p._row_coord(0.0, -float(v1)) == "V1   not recorded"


def test_panel_click_selects_the_bipole(qapp, egm):
    p = _panel(qapp, egm)
    got = []
    p.bipole_clicked.connect(got.append)
    row = p.row_labels().index("20A_7")        # a unipole row of bipole 2
    p._on_click(types.SimpleNamespace(inaxes=p._ax, ydata=-row))
    p._on_click(types.SimpleNamespace(inaxes=p._ax, ydata=-(len(p.row_labels()) - 1)))
    assert got == [2]                          # the V1 row belongs to no bipole


def test_panel_highlights_the_selected_group(qapp, egm):
    p = _panel(qapp, egm)
    # Bipole 1 is rows 3..5, so its band runs from y = -5.5 to y = -2.5.
    # (The WOI span is the other patch, in axes units vertically.)
    band = [r for r in p._ax.patches if abs(r.get_height() - 3.0) < 1e-6]
    assert len(band) == 1 and band[0].get_y() == pytest.approx(-5.5)


def test_panel_gain_and_zoom(qapp, egm):
    from ccdaf.gui.egm_panel import BIP, GAIN_STEP, WOI_MARGIN_MS
    p = _panel(qapp, egm)
    p._bump(BIP, GAIN_STEP)
    assert p._gain[BIP] == pytest.approx(GAIN_STEP)
    p.zoom_full()
    t = p._time_axis()
    assert p._ax.get_xlim() == pytest.approx((t[0], t[-1]))
    p.zoom_woi()
    assert p._ax.get_xlim() == pytest.approx((-151 - WOI_MARGIN_MS, 89 + WOI_MARGIN_MS))
    assert p._ref_ax.get_xlim() == pytest.approx(p._ax.get_xlim())


def test_panel_without_egm_is_empty(qapp):
    from ccdaf.gui.egm_panel import EGMPanel
    p = EGMPanel()
    p.set_egm(None)
    p.show_bipole(0, 0)                        # ignored, no crash
    assert p.row_labels() == []


# --- app hooks (stub host) ----------------------------------------------------

class _Host:
    def __init__(self, egm, car_points=None):
        self._eam_data = types.SimpleNamespace(egm=egm)
        self._eam_electrode_points = car_points
        self._egm_sel = None
        self._egm_picking = False
        self.egm_panel = MagicMock()
        self.plotter = MagicMock()
        self.drawn = 0

    def _egm(self):
        return self._eam_data.egm

    def _draw_egm_selection(self):
        self.drawn += 1

    def _egm_select(self, placement, bipole):
        import ccdaf.app.ccdaf as app
        app.CCDAF._egm_select(self, placement, bipole)


def test_pick_selects_and_panel_click_moves_within_placement(qapp, egm):
    import ccdaf.app.ccdaf as app
    host = _Host(egm)
    app.CCDAF._on_egm_pick(host, np.array(electrode_xyz(1, 11)))
    assert host._egm_sel == (1, 3)
    host.egm_panel.show_bipole.assert_called_with(1, 3)
    app.CCDAF._on_egm_bipole_clicked(host, 1)
    assert host._egm_sel == (1, 1) and host.drawn == 2


def test_panel_click_before_any_pick_does_nothing(qapp, egm):
    import ccdaf.app.ccdaf as app
    host = _Host(egm)
    app.CCDAF._on_egm_bipole_clicked(host, 1)
    assert host._egm_sel is None


def test_stop_picking_releases_the_picker(qapp, egm):
    import ccdaf.app.ccdaf as app
    host = _Host(egm)
    host._egm_picking = True
    app.CCDAF._stop_egm_picking(host)
    assert host._egm_picking is False
    host.plotter.disable_picking.assert_called_once()
    host.egm_panel.set_pick_checked.assert_called_with(False)


def test_mapping_catheter_of_nothing():
    assert mapping_catheter(None) is None and mapping_catheter({}) is None


# --- drawing the selection: nothing left behind --------------------------------

class _DrawHost:
    """What _draw_egm_selection touches; the plotter hands out fresh actors."""
    def __init__(self, egm, show_direction=True):
        import pyvista as pv
        import ccdaf.app.ccdaf as app
        self._eam_data = types.SimpleNamespace(egm=egm)
        self._egm_sel = (0, 0)
        self._egm_actors = []
        self._view = app.VIEWS["egm"]
        self.loader = types.SimpleNamespace(mesh=pv.Sphere(radius=20.0))
        self.egm_panel = MagicMock()
        self.egm_panel.show_direction.return_value = show_direction
        self.plotter = MagicMock()
        self.plotter.add_mesh.side_effect = lambda *a, **k: object()
        self.plotter.add_silhouette.side_effect = lambda *a, **k: object()

    def _egm(self):
        return self._eam_data.egm

    def _focus_3d(self):
        pass


def test_redraw_removes_every_actor_including_outlines(egm):
    import ccdaf.app.ccdaf as app
    host = _DrawHost(egm)
    app.CCDAF._draw_egm_selection(host)
    first = list(host._egm_actors)
    # placement 0: blue (1 other point), red (2), picked + arrow, each outlined
    assert len(first) == 8
    host._egm_sel = (1, 1)
    app.CCDAF._draw_egm_selection(host)
    removed = [c.args[0] for c in host.plotter.remove_actor.call_args_list]
    assert removed == first                     # outlines went too
    assert not set(map(id, host._egm_actors)) & set(map(id, first))


def test_show_bipole_off_draws_no_arrow(egm):
    import ccdaf.app.ccdaf as app
    host = _DrawHost(egm, show_direction=False)
    app.CCDAF._draw_egm_selection(host)
    assert len(host._egm_actors) == 6           # the arrow and its outline gone


def test_leaving_the_view_clears_the_selection_actors(egm):
    import ccdaf.app.ccdaf as app
    host = _DrawHost(egm)
    app.CCDAF._draw_egm_selection(host)
    host._view = app.VIEWS["general"]
    app.CCDAF._draw_egm_selection(host)
    assert host._egm_actors == [] and host.plotter.remove_actor.call_count == 8


# --- surface opacity --------------------------------------------------------------

def _opacity_host(slider, fibres=False, volume=False):
    import ccdaf.app.ccdaf as app
    host = types.SimpleNamespace(
        plotter=MagicMock(), _mesh_actor=MagicMock(),
        vis_widget=MagicMock(), loader=types.SimpleNamespace(
            kind=app.VOLUME if volume else "surface"))
    host.vis_widget.surface_opacity.return_value = slider
    host.vis_widget.show_fibres.return_value = fibres
    return host


def test_slider_sets_the_surface_opacity():
    import ccdaf.app.ccdaf as app
    host = _opacity_host(0.4)
    app.CCDAF._apply_surface_opacity(host)
    assert host._mesh_actor.prop.opacity == pytest.approx(0.4)
    host.plotter.enable_depth_peeling.assert_called_once()
    host = _opacity_host(1.0)
    app.CCDAF._apply_surface_opacity(host)
    host.plotter.disable_depth_peeling.assert_called_once()


def test_fibres_keep_their_see_through_and_a_lower_slider_wins():
    import ccdaf.app.ccdaf as app
    host = _opacity_host(1.0, fibres=True, volume=True)
    app.CCDAF._apply_surface_opacity(host)
    assert host._mesh_actor.prop.opacity == pytest.approx(app.FIBRE_SURFACE_OPACITY)
    host = _opacity_host(0.1, fibres=True, volume=True)
    app.CCDAF._apply_surface_opacity(host)
    assert host._mesh_actor.prop.opacity == pytest.approx(0.1)


def test_visualisation_opacity_slider(qapp):
    from ccdaf.gui.visualisation_widget import MIN_OPACITY_PCT, VisualisationWidget
    w = VisualisationWidget()
    assert w.surface_opacity() == 1.0
    got = []
    w.opacity_changed.connect(got.append)
    w.sld_opacity.setValue(40)
    assert w.surface_opacity() == pytest.approx(0.4) and got == [pytest.approx(0.4)]
    w.sld_opacity.setValue(0)
    assert w.surface_opacity() == pytest.approx(MIN_OPACITY_PCT / 100.0)


def test_panel_show_bipole_tick(qapp, egm):
    p = _panel(qapp, egm)
    assert p.show_direction() is True
    got = []
    p.direction_toggled.connect(got.append)
    p.chk_direction.setChecked(False)
    assert p.show_direction() is False and got == [False]
