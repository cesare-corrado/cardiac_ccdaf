"""
ManualCorrectionWidget
======================
Side-panel widget for manual mesh label correction.

Signals communicate user intent; the host connects them to the actual
editor logic.  State helpers (``set_active``, ``reset_state``,
``on_accepted``) let the host update widget appearance without coupling
to individual buttons.

Which labels the panel offers follows the seed type chosen in the Seed
selection panel, pushed in through :meth:`set_label_entries`. A seed type
with no labels of its own — a landmark set, an anatomy whose tagging is
not defined yet — leaves nothing to correct, so the whole panel is
disabled and says which seed type it is following. Correcting a tagging
belongs to the seed set that produced it.
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

        # Whether the active seed type offers any label at all. Separate
        # from set_active's "the mesh is ready": a panel can be ready and
        # still have nothing to apply.
        self._has_labels: bool = bool(label_entries)
        # The last label actually offered, kept across seed types that
        # offer none. Without it, switching to a label-less type and back
        # silently re-points the editor at whichever label sorts first,
        # which is the change this panel exists not to make behind the
        # user's back.
        self._last_label: Optional[int] = (
            int(label_entries[0][0]) if label_entries else None)

        self.lbl_follows = QtWidgets.QLabel()
        self.lbl_follows.setWordWrap(True)
        self.lbl_follows.setToolTip(
            "Manual correction acts on the seed type selected in the Seed "
            "selection panel. Change it there to correct a different set."
        )
        layout.addWidget(self.lbl_follows)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Label:"))
        self.cmb_label = QtWidgets.QComboBox()
        self.cmb_label.setToolTip(
            "Active label for tagging — selection mode and the snake both apply "
            "this label to the triangles they pick."
        )
        for lbl, name in label_entries:
            self.cmb_label.addItem(f"{lbl} — {name}", userData=int(lbl))
        self.cmb_label.currentIndexChanged.connect(self._on_label_index_changed)
        row.addWidget(self.cmb_label, 1)
        layout.addLayout(row)

        self.btn_edit_toggle = QtWidgets.QPushButton("Activate selection mode")
        self.btn_edit_toggle.setToolTip(
            "Toggle triangle selection: press X over a triangle to add it to a "
            "pending batch, then press C to commit the batch to the active label."
        )
        self.btn_edit_toggle.setCheckable(True)
        self.btn_edit_toggle.toggled.connect(self._on_edit_toggled)
        self.btn_edit_toggle.setEnabled(False)
        layout.addWidget(self.btn_edit_toggle)

        self.btn_fill_holes = QtWidgets.QPushButton("Fill Holes (Protect Boundaries)")
        self.btn_fill_holes.setToolTip(
            "Fill unassigned triangles by region growing, keeping existing "
            "region boundaries separate so neighbouring labels do not merge."
        )
        self.btn_fill_holes.clicked.connect(self.fill_holes_requested.emit)
        self.btn_fill_holes.setEnabled(False)
        layout.addWidget(self.btn_fill_holes)

        # Boundary smoothing of the *active* label. Dilate fills the jagged
        # body fringe, erode shaves spikes; the button applies whichever are
        # ticked (both = a closing) one pass per click, so the user smooths by
        # eye. Body smooths like any other label: it has no boundary of its
        # own, so growing it erodes every PV label at once, and shrinking it
        # dilates them.
        smooth_row = QtWidgets.QHBoxLayout()
        self.chk_dilate = QtWidgets.QCheckBox("Dilate")
        self.chk_dilate.setToolTip(
            "Include a dilation pass when smoothing — grows the active label "
            "into its jagged fringe. With body selected this erodes every PV "
            "label at once."
        )
        self.chk_dilate.setChecked(True)
        self.chk_erode = QtWidgets.QCheckBox("Erode")
        self.chk_erode.setToolTip(
            "Include an erosion pass when smoothing — shaves spikes off the "
            "active label. With body selected this dilates every PV label at "
            "once."
        )
        self.btn_smooth = QtWidgets.QPushButton("Smooth active label")
        self.btn_smooth.setToolTip(
            "Smooth the boundary of the label selected above, one pass per "
            "click. Dilate grows it into the jagged fringe, Erode shaves "
            "spikes; both ticked runs dilate-then-erode, which de-jags with "
            "little net growth. Body counts as a label: growing it smooths "
            "every region's boundary at once."
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
            "triangle touching that line with the selected label (body included)."
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
        self.btn_snake_clear.setToolTip(
            "Discard the current snake's points without leaving snake mode."
        )
        self.btn_snake_clear.clicked.connect(self.snake_clear_requested.emit)
        self.btn_snake_clear.setEnabled(False)
        self.btn_snake_commit = QtWidgets.QPushButton("Commit snake")
        self.btn_snake_commit.setToolTip(
            "Tag every triangle touching the current geodesic with the active "
            "label."
        )
        self.btn_snake_commit.clicked.connect(self.snake_commit_requested.emit)
        self.btn_snake_commit.setEnabled(False)
        snake_row.addWidget(self.btn_snake_undo_point)
        snake_row.addWidget(self.btn_snake_clear)
        snake_row.addWidget(self.btn_snake_commit, 1)
        layout.addLayout(snake_row)

        self.btn_accept = QtWidgets.QPushButton("Accept tagging")
        self.btn_accept.setToolTip(
            "Finish manual correction: commit any pending batch and assign the "
            "body label to all still-unassigned triangles."
        )
        self.btn_accept.clicked.connect(self.accept_requested.emit)
        self.btn_accept.setEnabled(False)
        layout.addWidget(self.btn_accept)

        self.btn_undo = QtWidgets.QPushButton("Undo last edit")
        self.btn_undo.setToolTip("Undo the last committed batch (up to 3 levels).")
        self.btn_undo.clicked.connect(self.undo_requested.emit)
        self.btn_undo.setEnabled(False)
        layout.addWidget(self.btn_undo)

        layout.addWidget(QtWidgets.QLabel(
            "<i>Press <b>X</b> over a triangle to pick it, "
            "then <b>C</b> to commit the batch.</i>"
        ))

    def _on_label_index_changed(self, _index: int) -> None:
        """Announce a label change, unless the combo is empty.

        Clearing the combo fires this with no current data; that is the
        panel being repopulated, not the user choosing a region."""
        label = self.current_label()
        if label is not None:
            self._last_label = int(label)
            self.label_changed.emit(int(label))

    def _on_edit_toggled(self, on: bool) -> None:
        # Fill Holes and Smooth are whole-mesh operations on the active label —
        # they neither read nor write the pending selection, so they stay
        # available whether or not selection mode is on. Tying them to it only
        # forced a pointless round trip through the toggle.
        self.btn_edit_toggle.setText(
            "Deactivate selection mode" if on else "Activate selection mode"
        )
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

    def current_label(self) -> Optional[int]:
        """The active label, or ``None`` when this seed type offers none."""
        data = self.cmb_label.currentData()
        return None if data is None else int(data)

    def set_label_index(self, index: int) -> None:
        self.cmb_label.setCurrentIndex(index)

    def set_label_entries(self,
                          label_entries: List[Tuple[int, str]],
                          follows: str = "") -> None:
        """Offer exactly *label_entries*, naming the seed type they belong to.

        The current label is kept when the new set still carries it, so
        switching seed type and back does not silently re-point the editor
        at a different region. An empty set disables the panel: there is
        no label to apply, and every control here applies one.

        Signals are blocked while refilling — the intermediate states of a
        clear-and-repopulate are not label changes the host should act on.
        """
        previous = self.current_label()
        if previous is not None:
            self._last_label = int(previous)
        wanted = self._last_label

        self.cmb_label.blockSignals(True)
        self.cmb_label.clear()
        for lbl, name in label_entries:
            self.cmb_label.addItem(f"{lbl} — {name}", userData=int(lbl))
        if wanted is not None:
            idx = self.cmb_label.findData(int(wanted))
            if idx >= 0:
                self.cmb_label.setCurrentIndex(idx)
        self.cmb_label.blockSignals(False)

        self._has_labels = bool(label_entries)
        self.cmb_label.setEnabled(self._has_labels)
        self.lbl_follows.setText(
            f"Follows seed type: <b>{follows}</b>" if self._has_labels
            else (f"Follows seed type: <b>{follows}</b><br>"
                  f"<i>This seed type has no labels to correct.</i>")
        )
        if not self._has_labels:
            self.set_active(False)
            self.set_undo_enabled(False)
        elif self.current_label() != wanted:
            # The remembered label is not in the new set, so the combo
            # landed on whatever is first. Say so, or the editor keeps
            # applying a label this seed type does not have.
            self._last_label = int(self.current_label())
            self.label_changed.emit(int(self._last_label))

    def set_active(self, enabled: bool) -> None:
        """Enable/disable the editing controls (called after mesh load or tagging).

        A seed type offering no labels can never be active: every control
        here applies the label the dropdown holds, and there is none."""
        enabled = bool(enabled) and self._has_labels
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
        self.btn_edit_toggle.setEnabled(self._has_labels)
        self.uncheck_snake()
        self.btn_snake.setEnabled(self._has_labels)
        self.btn_accept.setEnabled(self._has_labels)
        # Fill Holes stays live after accept. Accept leaves nothing
        # unassigned, but the operation is not only a fill: it first drops the
        # triangles straddling two regions, then regrows, which pulls touching
        # labels apart again. That is still worth reaching for once tagging is
        # accepted, so the button follows the panel like the other whole-mesh
        # tools rather than parking here.


__all__ = ["ManualCorrectionWidget"]
