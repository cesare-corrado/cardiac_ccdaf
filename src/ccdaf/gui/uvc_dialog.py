"""
UvcDialog
=========
Compute universal ventricular coordinates: choose the apices, run.

The window opens with both apices already found and marked in the 3D
view (:func:`ccdaf.core.uvc.automatic_apices`, cheap on any mesh), and the
run uses exactly the apices shown. Either can be replaced by clicking an
endocardium; **Automatic** brings the found ones back.

As in the fibre window, the dialog only asks for picking through a signal;
the application owns the picker and the markers. The run happens on a
worker thread (:class:`UvcWorker`).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt5 import QtCore, QtWidgets

from ccdaf.core import uvc
from ccdaf.core import ventricular_fibres as vf


class UvcWorker(QtCore.QThread):
    """Run :func:`ccdaf.core.uvc.compute` off the GUI thread, on plain arrays."""

    status = QtCore.pyqtSignal(str)
    done = QtCore.pyqtSignal(object)        # a UvcRun
    failed = QtCore.pyqtSignal(str)

    def __init__(self, points: np.ndarray, tets: np.ndarray, mask: np.ndarray,
                 tags: np.ndarray, lv_apex: Optional[uvc.Apex],
                 rv_apex: Optional[uvc.Apex],
                 parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self._args = (points, tets, mask, tags)
        self._apices = (lv_apex, rv_apex)

    def run(self) -> None:                    # noqa: D401 - Qt override
        try:
            result = uvc.compute(*self._args, lv_apex=self._apices[0],
                                 rv_apex=self._apices[1], on_status=self.status.emit)
        except Exception as exc:               # reported, never raised in a thread
            self.failed.emit(str(exc))
            return
        self.done.emit(result)


def _describe(apex: Optional[uvc.Apex]) -> str:
    if apex is None:
        return "not found"
    how = "automatic" if apex.automatic else "picked"
    return f"{how}: {len(apex.nodes)} node(s) across the wall"


class UvcDialog(QtWidgets.QDialog):
    """Apices and the run, for one labelled, tagged volume. Non-modal."""

    #: The apices to mark changed: ``(lv, rv)``, either may be ``None``.
    apices_changed = QtCore.pyqtSignal(object, object)
    #: Start picking an apex (``"lv"`` or ``"rv"``), or stop (``""``).
    pick_requested = QtCore.pyqtSignal(str)
    #: Compute with ``(lv, rv)``.
    run_requested = QtCore.pyqtSignal(object, object)

    def __init__(self, grid, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Compute UVC")
        self._points, self._tets, self._mask, self._tags = uvc.inputs(grid)
        self._digest = vf.digest(self._points, self._tets, self._mask)
        lv_tag, rv_tag = uvc.lv_rv_tags(self._tets, self._tags, self._mask)
        self._auto = {"lv": None, "rv": None}
        self._shown = {"lv": None, "rv": None}
        self._auto_error = ""

        layout = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            "<i>Universal ventricular coordinates: transventricular (nu), "
            "apicobasal (z), transmural (rho) and rotational (phi), each from "
            "geodesic distances to the labelled surfaces.</i>")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        tags = QtWidgets.QLabel(f"Element tags: LV {lv_tag}, RV {rv_tag}.")
        tags.setToolTip("The LV tag is the one whose elements touch the LV "
                        "endocardium.")
        layout.addWidget(tags)

        box = QtWidgets.QGroupBox("Apices")
        grid_layout = QtWidgets.QGridLayout(box)
        self.lbl_lv = QtWidgets.QLabel()
        self.lbl_rv = QtWidgets.QLabel()
        self.btn_pick_lv = QtWidgets.QPushButton("Pick LV apex")
        self.btn_pick_rv = QtWidgets.QPushButton("Pick RV apex")
        for btn, which in ((self.btn_pick_lv, "lv"), (self.btn_pick_rv, "rv")):
            btn.setCheckable(True)
            btn.setToolTip(
                f"Click the {'LV' if which == 'lv' else 'RV'} endocardium in the "
                f"3D view; the line from the nearest endocardial node to the "
                f"epicardium becomes the apex.\nPress again to stop picking.")
            btn.toggled.connect(lambda on, w=which: self._on_pick_toggled(w, on))
        self.btn_auto = QtWidgets.QPushButton("Automatic")
        self.btn_auto.setToolTip(
            "Forget picked apices. LV: the LV-endocardial node farthest from the "
            "centre of the base. RV: the RV-endocardial node farthest from its "
            "base rim along the RV endocardium. Each joined to the nearest "
            "epicardial node.")
        self.btn_auto.clicked.connect(self.use_automatic)
        grid_layout.addWidget(QtWidgets.QLabel("LV"), 0, 0)
        grid_layout.addWidget(self.lbl_lv, 0, 1)
        grid_layout.addWidget(self.btn_pick_lv, 0, 2)
        grid_layout.addWidget(QtWidgets.QLabel("RV"), 1, 0)
        grid_layout.addWidget(self.lbl_rv, 1, 1)
        grid_layout.addWidget(self.btn_pick_rv, 1, 2)
        grid_layout.addWidget(self.btn_auto, 2, 2)
        grid_layout.setColumnStretch(1, 1)
        layout.addWidget(box)

        self.report = QtWidgets.QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setMinimumHeight(180)
        self.report.setLineWrapMode(QtWidgets.QPlainTextEdit.WidgetWidth)
        layout.addWidget(self.report)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        self.btn_run = buttons.addButton("Compute", QtWidgets.QDialogButtonBox.ActionRole)
        self.btn_run.setToolTip("Compute the four coordinates and write uvc_ven, "
                                "uvc_z, uvc_rho and uvc_phi onto the mesh.")
        self.btn_run.clicked.connect(self._on_run)
        buttons.rejected.connect(self.reject)
        self.btn_close = buttons.button(QtWidgets.QDialogButtonBox.Close)
        layout.addWidget(buttons)
        QtCore.QTimer.singleShot(0, self.use_automatic)
        self.adjustSize()

    # -----------------------------------------------------------------
    def matches(self, grid) -> bool:
        """Whether *grid* is still the mesh and labels this window measured."""
        try:
            return np.array_equal(vf.digest(*vf.inputs(grid)), self._digest)
        except Exception:
            return False

    def inputs(self):
        """``(points, tets, mask, tags)`` the window was opened on."""
        return self._points, self._tets, self._mask, self._tags

    def digest(self) -> np.ndarray:
        return self._digest

    def picking(self) -> str:
        """Which apex is being picked, or ``""``."""
        if self.btn_pick_lv.isChecked():
            return "lv"
        if self.btn_pick_rv.isChecked():
            return "rv"
        return ""

    def set_apex_at(self, position) -> None:
        """Make the endocardial line nearest *position* the apex being picked."""
        which = self.picking()
        if not which:
            return
        try:
            apex = uvc.apex_at(self._points, self._tets, self._mask, position, which)
        except ValueError as exc:
            self.set_status(f"No apex there: {exc}")
            return
        self._shown[which] = apex
        self._refresh()

    def use_automatic(self) -> None:
        """Show the automatic apices, finding them on first use."""
        self._stop_picking()
        if self._auto["lv"] is None and not self._auto_error:
            try:
                lv, rv = uvc.automatic_apices(self._points, self._tets, self._mask)
                self._auto = {"lv": lv, "rv": rv}
            except ValueError as exc:
                self._auto_error = str(exc)
        if self._auto_error:
            self.set_status(f"No automatic apex: {self._auto_error}\nPick both apices.")
        self._shown = dict(self._auto)
        self._refresh()

    def apices(self):
        """``(lv, rv)`` the next run will use."""
        return self._shown["lv"], self._shown["rv"]

    def show_run(self, run) -> None:
        """A run finished: report it."""
        self.set_status(run.details())

    def set_running(self, running: bool) -> None:
        for w in (self.btn_run, self.btn_auto, self.btn_pick_lv, self.btn_pick_rv,
                  self.btn_close):
            w.setEnabled(not running)
        if not running:
            self.btn_run.setEnabled(all(self._shown.values()))

    def set_status(self, text: str) -> None:
        self.report.setPlainText(text)

    def uncheck_pick(self) -> None:
        """Put the pick buttons back without emitting (the picker was taken)."""
        for btn in (self.btn_pick_lv, self.btn_pick_rv):
            btn.blockSignals(True)
            btn.setChecked(False)
            btn.blockSignals(False)

    # -----------------------------------------------------------------
    def _refresh(self) -> None:
        self.lbl_lv.setText(_describe(self._shown["lv"]))
        self.lbl_rv.setText(_describe(self._shown["rv"]))
        self.btn_run.setEnabled(all(self._shown.values()))
        self.apices_changed.emit(self._shown["lv"], self._shown["rv"])

    def _on_pick_toggled(self, which: str, on: bool) -> None:
        other = self.btn_pick_rv if which == "lv" else self.btn_pick_lv
        if on and other.isChecked():
            other.blockSignals(True)
            other.setChecked(False)
            other.blockSignals(False)
        self.pick_requested.emit(which if on else "")

    def _stop_picking(self) -> None:
        if self.picking():
            self.uncheck_pick()
            self.pick_requested.emit("")

    def _on_run(self) -> None:
        self._stop_picking()
        if all(self._shown.values()):
            self.run_requested.emit(self._shown["lv"], self._shown["rv"])

    def reject(self) -> None:
        self._stop_picking()
        super().reject()


__all__ = ["UvcDialog", "UvcWorker"]
