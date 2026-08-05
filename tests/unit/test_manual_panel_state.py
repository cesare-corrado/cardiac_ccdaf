"""
test_manual_panel_state.py
==========================
Which manual-correction controls are live, and when.

**Fill Holes** and **Smooth active label** are whole-mesh operations on the
active label: they neither read nor write the pending selection, so they do not
belong to selection mode. They used to be enabled and disabled by its toggle,
which meant arming a picker just to press them and disarming it afterwards.
They now follow the panel itself — available whenever manual correction is,
whatever selection mode is doing.

Selection mode and the snake stay mutually exclusive; that is the host's job,
not the panel's, and is not retested here.

Qt only — no display beyond the offscreen platform, no mesh.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pytest
from PyQt5 import QtWidgets

from ccdaf.gui.manual_correction_widget import ManualCorrectionWidget
from ccdaf.interaction.manual_editor import ALLOWED_LABELS


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def panel(qapp):
    """An active panel — the state after a mesh is adopted or tagging runs."""
    w = ManualCorrectionWidget([(lbl, str(lbl)) for lbl in ALLOWED_LABELS])
    w.set_active(True)
    return w


def test_cleanup_tools_are_live_with_selection_mode_off(panel):
    assert not panel.btn_edit_toggle.isChecked()
    assert panel.btn_fill_holes.isEnabled()
    assert panel.btn_smooth.isEnabled()


def test_leaving_selection_mode_keeps_them_live(panel):
    panel.btn_edit_toggle.setChecked(True)
    panel.btn_edit_toggle.setChecked(False)
    assert panel.btn_fill_holes.isEnabled()
    assert panel.btn_smooth.isEnabled()


def test_they_stay_live_in_selection_mode_too(panel):
    panel.btn_edit_toggle.setChecked(True)
    assert panel.btn_fill_holes.isEnabled()
    assert panel.btn_smooth.isEnabled()


def test_they_still_emit_their_requests_with_selection_mode_off(panel):
    fired: list = []
    panel.fill_holes_requested.connect(lambda: fired.append("fill"))
    panel.smooth_requested.connect(lambda d, e: fired.append(("smooth", d, e)))
    panel.btn_fill_holes.click()
    panel.chk_erode.setChecked(True)
    panel.btn_smooth.click()
    assert fired == ["fill", ("smooth", True, True)]


def test_an_inactive_panel_offers_neither(qapp):
    w = ManualCorrectionWidget([(lbl, str(lbl)) for lbl in ALLOWED_LABELS])
    assert not w.btn_fill_holes.isEnabled()
    assert not w.btn_smooth.isEnabled()
    w.set_active(True)
    w.set_active(False)
    assert not w.btn_fill_holes.isEnabled()
    assert not w.btn_smooth.isEnabled()


def test_reset_state_disables_them(panel):
    panel.reset_state()
    assert not panel.btn_fill_holes.isEnabled()
    assert not panel.btn_smooth.isEnabled()
    assert not panel.btn_edit_toggle.isEnabled()
