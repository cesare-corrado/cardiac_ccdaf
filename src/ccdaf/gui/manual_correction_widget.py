"""
ManualCorrectionWidget
======================
Side-panel widget for manual mesh label correction.

Signals communicate user intent; the host connects them to the actual
editor logic.  State helpers (``set_active``, ``reset_state``,
``on_accepted``) let the host update widget appearance without coupling
to individual buttons.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from PyQt5 import QtCore, QtWidgets


class ManualCorrectionWidget(QtWidgets.QGroupBox):

    label_changed      = QtCore.pyqtSignal(int)
    edit_toggled       = QtCore.pyqtSignal(bool)
    fill_holes_requested = QtCore.pyqtSignal()
    accept_requested   = QtCore.pyqtSignal()
    undo_requested     = QtCore.pyqtSignal()

    def __init__(self,
                 label_entries: List[Tuple[int, str]],
                 parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Label:"))
        self.cmb_label = QtWidgets.QComboBox()
        for lbl, name in label_entries:
            self.cmb_label.addItem(f"{lbl} — {name}", userData=int(lbl))
        self.cmb_label.currentIndexChanged.connect(
            lambda _: self.label_changed.emit(int(self.cmb_label.currentData()))
        )
        row.addWidget(self.cmb_label, 1)
        layout.addLayout(row)

        self.btn_edit_toggle = QtWidgets.QPushButton("Activate selection mode")
        self.btn_edit_toggle.setCheckable(True)
        self.btn_edit_toggle.toggled.connect(self._on_edit_toggled)
        self.btn_edit_toggle.setEnabled(False)
        layout.addWidget(self.btn_edit_toggle)

        self.btn_fill_holes = QtWidgets.QPushButton("Fill Holes (Protect Boundaries)")
        self.btn_fill_holes.clicked.connect(self.fill_holes_requested.emit)
        self.btn_fill_holes.setEnabled(False)
        layout.addWidget(self.btn_fill_holes)

        self.btn_accept = QtWidgets.QPushButton("Accept tagging")
        self.btn_accept.clicked.connect(self.accept_requested.emit)
        self.btn_accept.setEnabled(False)
        layout.addWidget(self.btn_accept)

        self.btn_undo = QtWidgets.QPushButton("Undo last edit")
        self.btn_undo.setToolTip("Undo the last committed batch (up to 3 levels).")
        self.btn_undo.clicked.connect(self.undo_requested.emit)
        self.btn_undo.setEnabled(False)
        layout.addWidget(self.btn_undo)

        layout.addWidget(QtWidgets.QLabel(
            "<i>Click triangles, then press <b>X</b> to commit the batch.</i>"
        ))

    def _on_edit_toggled(self, on: bool) -> None:
        self.btn_edit_toggle.setText(
            "Deactivate selection mode" if on else "Activate selection mode"
        )
        self.btn_fill_holes.setEnabled(on)
        self.edit_toggled.emit(on)

    def current_label(self) -> int:
        return int(self.cmb_label.currentData())

    def set_label_index(self, index: int) -> None:
        self.cmb_label.setCurrentIndex(index)

    def set_active(self, enabled: bool) -> None:
        """Enable/disable the editing controls (called after mesh load or tagging)."""
        self.btn_edit_toggle.setEnabled(enabled)
        self.btn_fill_holes.setEnabled(enabled)
        self.btn_accept.setEnabled(enabled)

    def set_undo_enabled(self, enabled: bool) -> None:
        self.btn_undo.setEnabled(enabled)

    def reset_state(self) -> None:
        """Disable all controls — used by teardown after plotter rebuild."""
        self.btn_edit_toggle.blockSignals(True)
        self.btn_edit_toggle.setChecked(False)
        self.btn_edit_toggle.blockSignals(False)
        self.btn_edit_toggle.setText("Activate selection mode")
        self.btn_edit_toggle.setEnabled(False)
        self.btn_fill_holes.setEnabled(False)
        self.btn_accept.setEnabled(False)
        self.btn_undo.setEnabled(False)

    def on_accepted(self) -> None:
        """Update appearance after tagging is accepted without re-triggering the toggle signal."""
        self.btn_edit_toggle.blockSignals(True)
        self.btn_edit_toggle.setChecked(False)
        self.btn_edit_toggle.blockSignals(False)
        self.btn_edit_toggle.setText("Activate selection mode")
        self.btn_edit_toggle.setEnabled(True)
        self.btn_accept.setEnabled(True)
        self.btn_fill_holes.setEnabled(False)


__all__ = ["ManualCorrectionWidget"]
