"""
test_dark_view_contrast.py
==========================
Black on black is invisible, and pyvista's defaults land there twice.

The 3D viewport is painted black, while pyvista's theme font colour is black:
anything that takes its colour from that theme and nothing else is drawn but
cannot be seen. Two places did.

* The **Regions colour bar** labels its bands through ``annotations`` rather
  than tick numbers, and pyvista paints title, tick and annotation text from
  the one ``color`` value — so leaving it unset hid the region names, which
  are the entire content of that bar.
* The **segmentation plane widget** takes its normal arrow and outline from
  the same theme colour. The arrow is how the user reads which half-space
  ``Apply plane relabel`` will act on, so an invisible one is a guess.

Both are exercised unbound against a stand-in host: constructing the real
window needs a live GL context, which aborts VTK on a headless box.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pyvista as pv

from ccdaf.app.ccdaf import CCDAF
from ccdaf.core.mesh_loader import BODY_LABEL


def _tagged_mesh() -> pv.PolyData:
    mesh = pv.Sphere(theta_resolution=8, phi_resolution=8).triangulate()
    mesh.cell_data["elemTag"] = np.full(mesh.n_cells, BODY_LABEL,
                                        dtype=np.int32)
    return mesh


class _RenderHost:
    """The parts of the window ``_render_mesh`` reaches for."""

    def __init__(self):
        self.loader = MagicMock(mesh=_tagged_mesh())
        self.plotter = MagicMock()
        self._mesh_actor = None
        self._view = MagicMock(is_multiview=False)
        self._focus_3d = MagicMock()
        self._clear_scalar_bars = MagicMock()
        self._enable_bar_interaction = MagicMock()


def test_the_regions_bar_names_a_font_colour():
    host = _RenderHost()
    CCDAF._render_mesh(host)

    args = host.plotter.add_mesh.call_args.kwargs["scalar_bar_args"]
    assert args["color"] == "white"


def test_the_regions_bar_still_labels_through_annotations():
    """The colour is on the annotation text, so the annotations must remain.

    If the bar ever grew tick labels instead, the fix would have to move with
    them — this pins what the colour is actually making readable."""
    host = _RenderHost()
    CCDAF._render_mesh(host)

    kwargs = host.plotter.add_mesh.call_args.kwargs
    assert kwargs["scalar_bar_args"]["n_labels"] == 0
    assert kwargs["annotations"]


class _SegHost:
    """The parts of the window the segmentation plane toggle reaches for."""

    def __init__(self):
        self.plotter = MagicMock()
        self._seg_plane_widget = None
        self._seg_array = np.zeros((4, 5, 6), dtype=np.int16)
        self._seg_origin = (0.0, 0.0, 0.0)
        self._seg_spacing = (1.0, 1.0, 1.0)
        self._seg_plane_normal = None
        self._seg_plane_point = None
        self._focus_3d = MagicMock()
        self.seg_widget = MagicMock()
        self.statusBar = MagicMock()


def test_the_segmentation_plane_names_its_own_colour():
    host = _SegHost()
    CCDAF._action_seg_plane_toggle(host, True)

    assert host.plotter.add_plane_widget.call_args.kwargs["color"] == "white"


def test_toggling_the_plane_off_still_clears_it():
    """The colour change rides on the creation path only."""
    host = _SegHost()
    CCDAF._action_seg_plane_toggle(host, True)
    host._seg_plane_widget = MagicMock()
    CCDAF._action_seg_plane_toggle(host, False)

    host.plotter.clear_plane_widgets.assert_called_once()
    assert host._seg_plane_widget is None
