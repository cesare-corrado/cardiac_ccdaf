"""
SeedWidget
==========
Side-panel widget for seed selection controls.
"""
from __future__ import annotations

from typing import Optional

from PyQt5 import QtCore, QtWidgets


class SeedWidget(QtWidgets.QGroupBox):

    start_requested = QtCore.pyqtSignal()
    undo_requested  = QtCore.pyqtSignal()
    reset_requested = QtCore.pyqtSignal()
    save_requested  = QtCore.pyqtSignal()
    load_requested  = QtCore.pyqtSignal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.btn_start = QtWidgets.QPushButton("Start seed selection")
        self.btn_start.clicked.connect(self.start_requested.emit)
        self.btn_start.setEnabled(False)
        layout.addWidget(self.btn_start)

        row = QtWidgets.QHBoxLayout()
        self.btn_undo = QtWidgets.QPushButton("Undo")
        self.btn_undo.clicked.connect(self.undo_requested.emit)
        self.btn_undo.setEnabled(False)
        row.addWidget(self.btn_undo)
        self.btn_reset = QtWidgets.QPushButton("Reset")
        self.btn_reset.clicked.connect(self.reset_requested.emit)
        self.btn_reset.setEnabled(False)
        row.addWidget(self.btn_reset)
        layout.addLayout(row)

        row = QtWidgets.QHBoxLayout()
        self.btn_save = QtWidgets.QPushButton("Save seeds…")
        self.btn_save.setToolTip(
            "Save the six seeds as names and coordinates — no vertex ids, "
            "so they reload onto a clipped or refined mesh."
        )
        self.btn_save.clicked.connect(self.save_requested.emit)
        self.btn_save.setEnabled(False)
        row.addWidget(self.btn_save)
        self.btn_load = QtWidgets.QPushButton("Load seeds…")
        self.btn_load.setToolTip(
            "Load saved seeds; each is snapped to the current surface by "
            "nearest point."
        )
        self.btn_load.clicked.connect(self.load_requested.emit)
        self.btn_load.setEnabled(False)
        row.addWidget(self.btn_load)
        layout.addLayout(row)

        self.lbl_prompt = QtWidgets.QLabel("Load a mesh to begin.")
        self.lbl_prompt.setWordWrap(True)
        self.lbl_prompt.setStyleSheet("QLabel { padding: 6px; }")
        layout.addWidget(self.lbl_prompt)

        self.lbl_progress = QtWidgets.QLabel("Seeds: 0 / 6")
        layout.addWidget(self.lbl_progress)

    def set_start_enabled(self, enabled: bool) -> None:
        self.btn_start.setEnabled(enabled)

    def set_undo_enabled(self, enabled: bool) -> None:
        self.btn_undo.setEnabled(enabled)

    def set_reset_enabled(self, enabled: bool) -> None:
        self.btn_reset.setEnabled(enabled)

    def set_save_enabled(self, enabled: bool) -> None:
        self.btn_save.setEnabled(enabled)

    def set_load_enabled(self, enabled: bool) -> None:
        self.btn_load.setEnabled(enabled)

    def set_prompt(self, text: str) -> None:
        self.lbl_prompt.setText(text)

    def set_progress(self, text: str) -> None:
        self.lbl_progress.setText(text)


__all__ = ["SeedWidget"]
