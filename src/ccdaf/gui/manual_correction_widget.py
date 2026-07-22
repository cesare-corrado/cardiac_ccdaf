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
    smooth_requested   = QtCore.pyqtSignal(bool, bool)   # (dilate, erode)
    snake_toggled      = QtCore.pyqtSignal(bool)
    snake_undo_point_requested = QtCore.pyqtSignal()
    snake_clear_requested  = QtCore.pyqtSignal()
    snake_commit_requested = QtCore.pyqtSignal()
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

        # Boundary smoothing of the *active* label. Dilate fills the jagged
        # body fringe, erode shaves spikes; the button applies whichever are
        # ticked (both = a closing) one pass per click, so the user smooths by
        # eye. Body does nothing — it is the background, not a region.
        smooth_row = QtWidgets.QHBoxLayout()
        self.chk_dilate = QtWidgets.QCheckBox("Dilate")
        self.chk_dilate.setChecked(True)
        self.chk_erode = QtWidgets.QCheckBox("Erode")
        self.btn_smooth = QtWidgets.QPushButton("Smooth active label")
        self.btn_smooth.setToolTip(
            "Smooth the boundary of the label selected above, one pass per "
            "click. Dilate grows it into the jagged fringe, Erode shaves "
            "spikes; both ticked de-jags without net growth. Body does nothing."
        )
        self.btn_smooth.clicked.connect(
            lambda: self.smooth_requested.emit(
                self.chk_dilate.isChecked(), self.chk_erode.isChecked())
        )
        smooth_row.addWidget(self.chk_dilate)
        smooth_row.addWidget(self.chk_erode)
        smooth_row.addWidget(self.btn_smooth, 1)
        layout.addLayout(smooth_row)
        self.btn_smooth.setEnabled(False)

        # Snake (geodesic tag). Toggle on, press X to drop points on the
        # surface; the open geodesic through them is drawn live. Commit tags
        # every triangle touching that line with the label selected above.
        # Body builds no geodesic — pick a PV/LAA label. Mutually exclusive
        # with selection mode (both drive the surface picker).
        self.btn_snake = QtWidgets.QPushButton("Snake tag: off")
        self.btn_snake.setCheckable(True)
        self.btn_snake.setToolTip(
            "Toggle geodesic tagging. Press X to drop points on the surface; "
            "the open geodesic between them is drawn live. Commit tags every "
            "triangle touching that line with the selected label. Body builds "
            "no geodesic."
        )
        self.btn_snake.toggled.connect(self._on_snake_toggled)
        self.btn_snake.setEnabled(False)
        layout.addWidget(self.btn_snake)

        snake_row = QtWidgets.QHBoxLayout()
        self.btn_snake_undo_point = QtWidgets.QPushButton("Undo last point")
        self.btn_snake_undo_point.setToolTip(
            "Remove the most recently dropped snake point and redraw the "
            "geodesic through the remaining points."
        )
        self.btn_snake_undo_point.clicked.connect(self.snake_undo_point_requested.emit)
        self.btn_snake_undo_point.setEnabled(False)
        self.btn_snake_clear = QtWidgets.QPushButton("Clear snake")
        self.btn_snake_clear.clicked.connect(self.snake_clear_requested.emit)
        self.btn_snake_clear.setEnabled(False)
        self.btn_snake_commit = QtWidgets.QPushButton("Commit snake")
        self.btn_snake_commit.clicked.connect(self.snake_commit_requested.emit)
        self.btn_snake_commit.setEnabled(False)
        snake_row.addWidget(self.btn_snake_undo_point)
        snake_row.addWidget(self.btn_snake_clear)
        snake_row.addWidget(self.btn_snake_commit, 1)
        layout.addLayout(snake_row)

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
        self.btn_smooth.setEnabled(on)
        self.edit_toggled.emit(on)

    def _on_snake_toggled(self, on: bool) -> None:
        self.btn_snake.setText("Snake tag: on (press X)" if on else "Snake tag: off")
        self.btn_snake_undo_point.setEnabled(on)
        self.btn_snake_clear.setEnabled(on)
        self.btn_snake_commit.setEnabled(on)
        self.snake_toggled.emit(on)

    def uncheck_edit_toggle(self) -> None:
        """Programmatically leave selection mode without re-emitting the signal."""
        self.btn_edit_toggle.blockSignals(True)
        self.btn_edit_toggle.setChecked(False)
        self.btn_edit_toggle.blockSignals(False)
        self.btn_edit_toggle.setText("Activate selection mode")

    def uncheck_snake(self) -> None:
        """Programmatically leave snake mode without re-emitting the signal."""
        self.btn_snake.blockSignals(True)
        self.btn_snake.setChecked(False)
        self.btn_snake.blockSignals(False)
        self.btn_snake.setText("Snake tag: off")
        self.btn_snake_undo_point.setEnabled(False)
        self.btn_snake_clear.setEnabled(False)
        self.btn_snake_commit.setEnabled(False)

    def current_label(self) -> int:
        return int(self.cmb_label.currentData())

    def set_label_index(self, index: int) -> None:
        self.cmb_label.setCurrentIndex(index)

    def set_active(self, enabled: bool) -> None:
        """Enable/disable the editing controls (called after mesh load or tagging)."""
        self.btn_edit_toggle.setEnabled(enabled)
        self.btn_fill_holes.setEnabled(enabled)
        self.btn_smooth.setEnabled(enabled)
        self.btn_snake.setEnabled(enabled)
        self.btn_accept.setEnabled(enabled)
        if not enabled:
            self.uncheck_snake()

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
        self.btn_smooth.setEnabled(False)
        self.uncheck_snake()
        self.btn_snake.setEnabled(False)
        self.btn_accept.setEnabled(False)
        self.btn_undo.setEnabled(False)

    def on_accepted(self) -> None:
        """Update appearance after tagging is accepted without re-triggering the toggle signal."""
        self.btn_edit_toggle.blockSignals(True)
        self.btn_edit_toggle.setChecked(False)
        self.btn_edit_toggle.blockSignals(False)
        self.btn_edit_toggle.setText("Activate selection mode")
        self.btn_edit_toggle.setEnabled(True)
        self.uncheck_snake()
        self.btn_snake.setEnabled(True)
        self.btn_accept.setEnabled(True)
        self.btn_fill_holes.setEnabled(False)


__all__ = ["ManualCorrectionWidget"]
