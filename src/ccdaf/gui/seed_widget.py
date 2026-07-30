"""
SeedWidget
==========
Side-panel widget for seed selection controls.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from PyQt5 import QtCore, QtWidgets


class SeedWidget(QtWidgets.QGroupBox):

    start_requested = QtCore.pyqtSignal()
    undo_requested  = QtCore.pyqtSignal()
    reset_requested = QtCore.pyqtSignal()
    save_requested  = QtCore.pyqtSignal()
    load_requested  = QtCore.pyqtSignal()
    type_changed    = QtCore.pyqtSignal(str)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        type_row = QtWidgets.QHBoxLayout()
        type_row.addWidget(QtWidgets.QLabel("Seed type:"))
        self.combo_type = QtWidgets.QComboBox()
        self.combo_type.setToolTip(
            "Choose which set of surface points to pick. 'seed' is the "
            "six-seed workflow used for tagging; other choices pick "
            "landmark sets that are only saved/exported."
        )
        self.combo_type.currentIndexChanged.connect(self._on_type_changed)
        type_row.addWidget(self.combo_type, 1)
        layout.addLayout(type_row)

        self.btn_start = QtWidgets.QPushButton("Start seed selection")
        self.btn_start.setToolTip(
            "Begin picking the six seeds (LSPV, LIPV, RSPV, RIPV, LAA, MV) in "
            "order by clicking on the surface."
        )
        self.btn_start.clicked.connect(self.start_requested.emit)
        self.btn_start.setEnabled(False)
        layout.addWidget(self.btn_start)

        row = QtWidgets.QHBoxLayout()
        self.btn_undo = QtWidgets.QPushButton("Undo")
        self.btn_undo.setToolTip("Remove the last placed seed so you can re-pick it.")
        self.btn_undo.clicked.connect(self.undo_requested.emit)
        self.btn_undo.setEnabled(False)
        row.addWidget(self.btn_undo)
        self.btn_reset = QtWidgets.QPushButton("Reset")
        self.btn_reset.setToolTip("Clear all placed seeds and start the selection over.")
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

    def set_seed_types(self, items: List[Tuple[str, str]]) -> None:
        """Populate the seed-type dropdown from ``(type_id, label)`` pairs.

        Signals are blocked while filling so populating does not emit a
        spurious ``type_changed`` before the caller is wired up.
        """
        self.combo_type.blockSignals(True)
        self.combo_type.clear()
        for type_id, label in items:
            self.combo_type.addItem(label, type_id)
        self.combo_type.blockSignals(False)

    def current_seed_type(self) -> Optional[str]:
        return self.combo_type.currentData()

    def set_type_enabled(self, enabled: bool) -> None:
        self.combo_type.setEnabled(enabled)

    def _on_type_changed(self) -> None:
        type_id = self.combo_type.currentData()
        if type_id is not None:
            self.type_changed.emit(str(type_id))

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
