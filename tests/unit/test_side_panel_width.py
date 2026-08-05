"""
test_side_panel_width.py
========================
The side panel is wide enough to read.

The panel is a scroll area whose inner column is laid out at whatever its
widest section needs. When the area is narrower than that, the column is
not squeezed — it is cut off, and with the horizontal scroll bar off there
was no way to reach the missing part: the ``nElem`` box, two of the three
bounding-box boxes and the edge mean/max simply were not on screen.

So two things are pinned here:

* ``_measure_side_width`` sizes the panel from its sections rather than a
  hard-coded number, counts the sections that start hidden (segmentation,
  visualisation — they appear later and must not squeeze the panel then),
  and stays inside half the screen so a small display keeps a usable 3D
  view;
* a mesh-info value box is wide enough for the longest string
  ``update_info`` can put in it, so no statistic is shown elided.

Qt only — no display beyond the offscreen platform, no plotter, no mesh.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pytest
from PyQt5 import QtGui, QtWidgets

from ccdaf.app.ccdaf import CCDAF
from ccdaf.gui.mesh_info_widget import (
    _BOX_BORDER_PX, _BOX_PADDING_PX, _WIDEST_VALUE, MeshInfoWidget,
)


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _section(width: int) -> QtWidgets.QFrame:
    """A section frame whose minimum width is *width*, near enough."""
    frame = QtWidgets.QFrame()
    lay = QtWidgets.QHBoxLayout(frame)
    lay.setContentsMargins(0, 0, 0, 0)
    filler = QtWidgets.QLabel()
    filler.setFixedWidth(width)
    lay.addWidget(filler)
    return frame


class _Host:
    """Just enough of the window for the unbound measuring method."""

    def __init__(self, sections):
        self._sections = sections


def _measure(sections):
    scroll = QtWidgets.QScrollArea()
    layout = QtWidgets.QVBoxLayout()
    return CCDAF._measure_side_width(_Host(sections), scroll, layout), scroll


def _screen_cap():
    screen = QtWidgets.QApplication.primaryScreen()
    return screen.availableGeometry().width() // 2


# ---------------------------------------------------------------------------
# Panel width
# ---------------------------------------------------------------------------
def test_panel_fits_its_widest_section(qapp):
    sections = {"narrow": _section(120), "wide": _section(260)}
    width, _ = _measure(sections)
    assert width >= sections["wide"].minimumSizeHint().width()


def test_panel_leaves_room_for_the_scroll_bar(qapp):
    """The bar eats viewport width; a panel sized to the bare content
    would put the last column half under it."""
    sections = {"only": _section(260)}
    width, scroll = _measure(sections)
    bar = scroll.verticalScrollBar().sizeHint().width()
    assert width >= sections["only"].minimumSizeHint().width() + bar


def test_sections_hidden_at_startup_are_measured_too(qapp):
    """Segmentation and visualisation start hidden. Measuring only what is
    visible would size the panel too narrow the moment they appear."""
    hidden = _section(260)
    hidden.setVisible(False)
    with_hidden, _ = _measure({"visible": _section(120), "hidden": hidden})
    without, _ = _measure({"visible": _section(120)})
    assert with_hidden > without


def test_panel_never_takes_more_than_half_the_screen(qapp):
    """Past that point the as-needed scroll bar takes over, rather than the
    3D view being squeezed off a small display."""
    width, _ = _measure({"huge": _section(10_000)})
    assert width <= _screen_cap()


def test_no_sections_is_not_a_crash(qapp):
    width, _ = _measure({})
    assert width >= 0


# ---------------------------------------------------------------------------
# Mesh-info boxes
# ---------------------------------------------------------------------------
def _boxes(widget: MeshInfoWidget):
    return (widget.box_nodes, widget.box_cells,
            widget.box_bbox_x, widget.box_bbox_y, widget.box_bbox_z,
            widget.box_emin, widget.box_emean, widget.box_emax)


@pytest.fixture
def info(qapp):
    """Held by the fixture: a dropped widget takes its boxes down with it."""
    return MeshInfoWidget()


def test_every_box_fits_the_longest_value_it_can_show(qapp, info):
    widget = info
    for box in _boxes(widget):
        box.setText(_WIDEST_VALUE)
    widget.resize(widget.minimumSizeHint())
    widget.show()
    qapp.processEvents()
    for box in _boxes(widget):
        need = QtGui.QFontMetrics(box.font()).horizontalAdvance(box.text())
        assert need <= box.contentsRect().width(), f"{box.text()} elided"
    widget.close()


@pytest.mark.parametrize("value", [
    "-1.234e-05",     # %.4g at its longest
    "12845312",       # eight-digit point count
    "1.234e+02",
])
def test_widest_value_is_not_beaten_by_a_real_one(qapp, info, value):
    """``_WIDEST_VALUE`` is the yardstick the boxes are sized to; if a
    format change makes some value longer, this is where it shows up."""
    box = info.box_nodes
    fm = QtGui.QFontMetrics(box.font())
    assert fm.horizontalAdvance(value) <= fm.horizontalAdvance(_WIDEST_VALUE)


def test_box_is_measured_in_the_font_it_is_drawn_in(qapp, info):
    """The minimum width comes from QFontMetrics on the widget font, so the
    monospace request has to live on the widget: set only in the stylesheet
    it would draw one font and measure another. Which family the request
    resolves to is the desktop's business, hence the flags, not the name."""
    box = info.box_nodes
    font = box.font()
    assert font.fixedPitch()
    assert font.styleHint() == QtGui.QFont.Monospace
    chrome = 2 * (_BOX_PADDING_PX + _BOX_BORDER_PX)
    need = QtGui.QFontMetrics(font).horizontalAdvance(_WIDEST_VALUE)
    assert box.minimumWidth() >= need + chrome
