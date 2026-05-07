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

    def __init__(self,
                 pv_names: List[str],
                 parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

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

    def set_pv_finish_enabled(self, enabled: bool) -> None:
        self.btn_pv_finish.setEnabled(enabled)

    def set_clip_apply_enabled(self, enabled: bool) -> None:
        self.btn_apply.setEnabled(enabled)

    def set_clip_revert_enabled(self, enabled: bool) -> None:
        self.btn_revert.setEnabled(enabled)

    def set_enabled_after_accept(self) -> None:
        """Enable PV and mitral controls once tagging has been accepted."""
        self.btn_pv_start.setEnabled(True)
        self.btn_mv_sphere.setEnabled(True)
        self.btn_mv_plane.setEnabled(True)

    def reset_state(self) -> None:
        """Disable all controls — used by teardown after plotter rebuild."""
        self.btn_pv_start.setEnabled(False)
        self.btn_pv_finish.setEnabled(False)
        self.btn_mv_sphere.setEnabled(False)
        self.btn_mv_plane.setEnabled(False)
        self.btn_apply.setEnabled(False)
        self.btn_revert.setEnabled(False)


__all__ = ["ClippingWidget"]
