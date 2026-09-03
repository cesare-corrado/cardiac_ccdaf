"""
test_tooltips.py
================
Regression guard for hover help (Qt tooltips) on the side-panel controls.

Every interactive control a user hovers should carry a ``setToolTip`` string.
This test constructs each panel widget headlessly and asserts that the named
controls have a non-empty tooltip, so a future control added without help is
caught here rather than in the GUI.

Menu ``QAction``s are intentionally excluded — menu entries do not show hover
tooltips unless ``menu.setToolTipsVisible(True)`` is set — as is the per-item
mapping radio (its label is the mapping name).

Runs headless: needs a QApplication with the offscreen platform (the project's
standard pytest invocation sets ``QT_QPA_PLATFORM=offscreen``).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pytest
from PyQt5 import QtWidgets

from ccdaf.gui.seed_widget import SeedWidget
from ccdaf.gui.manual_correction_widget import ManualCorrectionWidget
from ccdaf.gui.clipping_widget import ClippingWidget
from ccdaf.gui.segmentation_widget import SegmentationWidget
from ccdaf.gui.tagging_widget import TaggingWidget
from ccdaf.gui.visualisation_widget import VisualisationWidget


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def _widgets():
    """(widget, [control attribute names that must have a tooltip])."""
    return [
        (SeedWidget(),
         ["btn_start", "btn_undo", "btn_reset", "btn_save", "btn_load"]),
        (ManualCorrectionWidget([(11, "LSPV"), (13, "LIPV"), (1, "body")]),
         ["cmb_label", "btn_edit_toggle", "btn_fill_holes", "chk_dilate",
          "chk_erode", "btn_smooth", "btn_snake", "btn_snake_undo_point",
          "btn_snake_clear", "btn_snake_commit", "btn_accept", "btn_undo"]),
        (ClippingWidget(["LSPV", "LIPV", "RSPV", "RIPV"]),
         ["chk_active", "cmb_pv", "btn_pv_start", "btn_pv_undo_point",
          "btn_pv_finish", "btn_mv_sphere", "btn_mv_plane", "btn_apply",
          "btn_revert"]),
        (SegmentationWidget(),
         ["btn_dilate", "btn_erode", "btn_moprh_opening", "btn_moprh_closing",
          "spn_kernel_x", "spn_kernel_y", "spn_kernel_z", "btn_fill",
          "rb_sphere", "rb_square", "rb_cyl", "spn_radius", "rb_2d", "rb_3d",
          "spn_depth", "btn_paint", "btn_convert_all", "btn_plane",
          "btn_plane_apply", "btn_undo",
          "spn_stdev_x", "spn_rfac_x"]),
        (TaggingWidget(),
         ["btn_tag", "chk_disable_tag"]),
        (VisualisationWidget(),
         ["cmb_field", "cmb_cmap", "chk_auto", "spn_min", "spn_max",
          "spn_iso", "chk_electrodes"]),
    ]


def test_panel_controls_have_tooltips(qapp):
    missing = []
    for widget, names in _widgets():
        cls = type(widget).__name__
        for name in names:
            control = getattr(widget, name)
            if not control.toolTip().strip():
                missing.append(f"{cls}.{name}")
    assert not missing, "controls missing a tooltip: " + ", ".join(missing)


def test_multi_clause_tooltips_break_into_lines(qapp):
    """The mitral tooltips carry three or four separate points.

    Run together they are a wall of text, and a plain ``\\n`` will not break
    them: Qt honours line breaks in a tooltip only when it reads the string as
    rich text, which it decides by finding a tag. ``<br>`` is both the break
    and the thing that makes Qt look for one.
    """
    panel = ClippingWidget(["LSPV", "LIPV", "RSPV", "RIPV"])
    for name in ("btn_mv_sphere", "btn_mv_plane", "chk_active"):
        tip = getattr(panel, name).toolTip()
        assert "<br>" in tip, f"{name} tooltip runs its clauses together"
        assert "\n" not in tip, (
            f"{name} tooltip mixes a newline into rich text, where it is "
            "ignored")
