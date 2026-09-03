"""
ClippingWidget
==============
Side-panel widget for mesh clipping controls (PV contours, mitral valve).

All user actions are exposed as signals.  The host enables/disables
individual controls via the provided setter methods.
"""
from __future__ import annotations

from typing import List, Optional

from PyQt5 import QtCore, QtWidgets


def _tip(*lines: str) -> str:
    """One tooltip per line, as rich text.

    Qt only honours line breaks in a tooltip when it reads the string as rich
    text, and it decides that by looking for a tag — a bare newline in plain
    text is collapsed into the auto-wrapping. The wrapper is what makes the
    tag mandatory: a tooltip built from clauses ("what it is", "how to drive
    it", "what it needs") reads as a list rather than one long sentence.
    """
    return "<br>".join(lines)


class ClippingWidget(QtWidgets.QGroupBox):

    pv_start_requested   = QtCore.pyqtSignal(str)   # emits selected pv_name
    pv_undo_point_requested = QtCore.pyqtSignal()
    pv_finish_requested  = QtCore.pyqtSignal()
    mv_sphere_requested  = QtCore.pyqtSignal()
    mv_plane_requested   = QtCore.pyqtSignal()
    clip_apply_requested = QtCore.pyqtSignal()
    clip_revert_requested = QtCore.pyqtSignal()
    clipping_toggled     = QtCore.pyqtSignal(bool)

    def __init__(self,
                 pv_names: List[str],
                 parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        # Set by set_enabled_after_accept / reset_state; the start buttons
        # need both this and the activation checkbox.
        self._accepted = False
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.chk_active = QtWidgets.QCheckBox("Clipping active")
        self.chk_active.setChecked(False)
        self.chk_active.setToolTip(_tip(
            "While unchecked, clipping is dormant and the X key belongs to "
            "manual correction. Both tools want X — this decides the owner.",
            "Ticking it accepts the tagging first if that has not happened "
            "yet, so clipping never waits on a step you cannot see.",
        ))
        self.chk_active.toggled.connect(self._on_active_toggled)
        layout.addWidget(self.chk_active)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("PV:"))
        self.cmb_pv = QtWidgets.QComboBox()
        self.cmb_pv.setToolTip(
            "Which pulmonary vein the contour runs on — the snake is confined "
            "to this tagged region."
        )
        for name in pv_names:
            self.cmb_pv.addItem(name, userData=name)
        row.addWidget(self.cmb_pv, 1)
        layout.addLayout(row)

        self.btn_pv_start = QtWidgets.QPushButton("Start PV contour")
        self.btn_pv_start.setToolTip(
            "Begin a PV clip on the selected vein: press X to drop points; a "
            "geodesic contour follows on that tag."
        )
        self.btn_pv_start.clicked.connect(
            lambda: self.pv_start_requested.emit(str(self.cmb_pv.currentData()))
        )
        self.btn_pv_start.setEnabled(False)
        layout.addWidget(self.btn_pv_start)

        self.btn_pv_undo_point = QtWidgets.QPushButton("Undo last point")
        self.btn_pv_undo_point.setToolTip(
            "Remove the most recently placed geodesic point while building the "
            "PV contour, and redraw the snake through the remaining points."
        )
        self.btn_pv_undo_point.clicked.connect(self.pv_undo_point_requested.emit)
        self.btn_pv_undo_point.setEnabled(False)
        layout.addWidget(self.btn_pv_undo_point)

        self.btn_pv_finish = QtWidgets.QPushButton("Close & Clip PV")
        self.btn_pv_finish.setToolTip(
            "Close the PV contour into a loop and clip off the vein cuff inside it."
        )
        self.btn_pv_finish.clicked.connect(self.pv_finish_requested.emit)
        self.btn_pv_finish.setEnabled(False)
        layout.addWidget(self.btn_pv_finish)

        row = QtWidgets.QHBoxLayout()
        self.btn_mv_sphere = QtWidgets.QPushButton("Mitral: sphere")
        self.btn_mv_sphere.setToolTip(_tip(
            "Show an adjustable sphere at the mitral seed.",
            "Left-drag it to move, right-drag to resize.",
            "Apply clip removes every triangle whose centre falls inside.",
            "Needs the MV seed.",
        ))
        self.btn_mv_sphere.clicked.connect(self.mv_sphere_requested.emit)
        self.btn_mv_sphere.setEnabled(False)
        row.addWidget(self.btn_mv_sphere)
        self.btn_mv_plane = QtWidgets.QPushButton("Mitral: plane")
        self.btn_mv_plane.setToolTip(_tip(
            "Show an adjustable cutting plane at the mitral seed.",
            "Left-drag the arrowhead to tilt it, the plane's rim to slide it "
            "along the arrow, the centre ball to shift it sideways.",
            "Apply clip removes the mitral side.",
            "Needs the MV seed.",
        ))
        self.btn_mv_plane.clicked.connect(self.mv_plane_requested.emit)
        self.btn_mv_plane.setEnabled(False)
        row.addWidget(self.btn_mv_plane)
        layout.addLayout(row)

        self.btn_apply = QtWidgets.QPushButton("Apply clip")
        self.btn_apply.setToolTip(
            "Apply the pending mitral clip (sphere or plane) to the mesh."
        )
        self.btn_apply.clicked.connect(self.clip_apply_requested.emit)
        self.btn_apply.setEnabled(False)
        layout.addWidget(self.btn_apply)

        self.btn_revert = QtWidgets.QPushButton("Reject / revert clip")
        self.btn_revert.setToolTip(
            "Discard the pending clip, or undo the last applied clip, restoring "
            "the previous mesh."
        )
        self.btn_revert.clicked.connect(self.clip_revert_requested.emit)
        self.btn_revert.setEnabled(False)
        layout.addWidget(self.btn_revert)

    def selected_pv(self) -> str:
        return str(self.cmb_pv.currentData())

    def is_clipping_enabled(self) -> bool:
        return bool(self.chk_active.isChecked())

    def is_accepted(self) -> bool:
        """Whether the panel has been told the tagging is accepted.

        The host asks before honouring a tick of the activation checkbox: an
        unaccepted tagging is something it can put right, not a reason to
        leave the panel dead."""
        return self._accepted

    def set_active_checked(self, checked: bool) -> None:
        """Set the activation checkbox without re-entering the toggle handler.

        Used to withdraw a tick the host could not honour. Going through the
        signal would ask the host again about the answer it just gave."""
        self.chk_active.blockSignals(True)
        self.chk_active.setChecked(checked)
        self.chk_active.blockSignals(False)
        self._sync_start_buttons()

    def set_pv_undo_point_enabled(self, enabled: bool) -> None:
        self.btn_pv_undo_point.setEnabled(enabled)

    def set_pv_finish_enabled(self, enabled: bool) -> None:
        self.btn_pv_finish.setEnabled(enabled)

    def set_clip_apply_enabled(self, enabled: bool) -> None:
        self.btn_apply.setEnabled(enabled)

    def set_clip_revert_enabled(self, enabled: bool) -> None:
        self.btn_revert.setEnabled(enabled)

    def set_enabled_after_accept(self) -> None:
        """Allow PV and mitral controls once tagging has been accepted.

        The start buttons still wait for the activation checkbox."""
        self._accepted = True
        self._sync_start_buttons()

    def clear_in_flight(self) -> None:
        """Put the controls back to "no clip in progress".

        Used when another tool takes the shared picker away mid-clip: the
        clip is abandoned, so the buttons that only mean something during one
        must go with it. Activation and the accepted-tagging state are the
        user's, and survive — only the in-flight controls are cleared."""
        self.btn_pv_undo_point.setEnabled(False)
        self.btn_pv_finish.setEnabled(False)
        self.btn_apply.setEnabled(False)

    def reset_state(self) -> None:
        """Disable all controls — used by teardown after plotter rebuild.

        The activation checkbox is the user's choice and survives."""
        self._accepted = False
        self.btn_pv_start.setEnabled(False)
        self.btn_pv_undo_point.setEnabled(False)
        self.btn_pv_finish.setEnabled(False)
        self.btn_mv_sphere.setEnabled(False)
        self.btn_mv_plane.setEnabled(False)
        self.btn_apply.setEnabled(False)
        self.btn_revert.setEnabled(False)

    def _sync_start_buttons(self) -> None:
        on = self._accepted and self.is_clipping_enabled()
        self.btn_pv_start.setEnabled(on)
        self.btn_mv_sphere.setEnabled(on)
        self.btn_mv_plane.setEnabled(on)

    def _on_active_toggled(self, on: bool) -> None:
        self._sync_start_buttons()
        if not on:
            # Whatever was mid-flight is being abandoned by the host.
            self.btn_pv_undo_point.setEnabled(False)
            self.btn_pv_finish.setEnabled(False)
            self.btn_apply.setEnabled(False)
        self.clipping_toggled.emit(on)


__all__ = ["ClippingWidget"]
