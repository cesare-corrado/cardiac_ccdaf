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


class ClippingWidget(QtWidgets.QGroupBox):

    pv_start_requested   = QtCore.pyqtSignal(str)   # emits selected pv_name
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
        self.chk_active.setToolTip(
            "While unchecked, clipping is dormant and the X key belongs to "
            "manual correction. Both tools want X — this decides the owner."
        )
        self.chk_active.toggled.connect(self._on_active_toggled)
        layout.addWidget(self.chk_active)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("PV:"))
        self.cmb_pv = QtWidgets.QComboBox()
        for name in pv_names:
            self.cmb_pv.addItem(name, userData=name)
        row.addWidget(self.cmb_pv, 1)
        layout.addLayout(row)

        self.btn_pv_start = QtWidgets.QPushButton("Start PV contour")
        self.btn_pv_start.clicked.connect(
            lambda: self.pv_start_requested.emit(str(self.cmb_pv.currentData()))
        )
        self.btn_pv_start.setEnabled(False)
        layout.addWidget(self.btn_pv_start)

        self.btn_pv_finish = QtWidgets.QPushButton("Close & Clip PV")
        self.btn_pv_finish.clicked.connect(self.pv_finish_requested.emit)
        self.btn_pv_finish.setEnabled(False)
        layout.addWidget(self.btn_pv_finish)

        row = QtWidgets.QHBoxLayout()
        self.btn_mv_sphere = QtWidgets.QPushButton("Mitral: sphere")
        self.btn_mv_sphere.clicked.connect(self.mv_sphere_requested.emit)
        self.btn_mv_sphere.setEnabled(False)
        row.addWidget(self.btn_mv_sphere)
        self.btn_mv_plane = QtWidgets.QPushButton("Mitral: plane")
        self.btn_mv_plane.clicked.connect(self.mv_plane_requested.emit)
        self.btn_mv_plane.setEnabled(False)
        row.addWidget(self.btn_mv_plane)
        layout.addLayout(row)

        self.btn_apply = QtWidgets.QPushButton("Apply clip")
        self.btn_apply.clicked.connect(self.clip_apply_requested.emit)
        self.btn_apply.setEnabled(False)
        layout.addWidget(self.btn_apply)

        self.btn_revert = QtWidgets.QPushButton("Reject / revert clip")
        self.btn_revert.clicked.connect(self.clip_revert_requested.emit)
        self.btn_revert.setEnabled(False)
        layout.addWidget(self.btn_revert)

    def selected_pv(self) -> str:
        return str(self.cmb_pv.currentData())

    def is_clipping_enabled(self) -> bool:
        return bool(self.chk_active.isChecked())

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

    def reset_state(self) -> None:
        """Disable all controls — used by teardown after plotter rebuild.

        The activation checkbox is the user's choice and survives."""
        self._accepted = False
        self.btn_pv_start.setEnabled(False)
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
            self.btn_pv_finish.setEnabled(False)
            self.btn_apply.setEnabled(False)
        self.clipping_toggled.emit(on)


__all__ = ["ClippingWidget"]
