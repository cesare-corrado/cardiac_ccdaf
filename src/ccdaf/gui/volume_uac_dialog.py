"""
volume_uac_dialog
=================
Actions > Compute UAC… on an atrial volume: α, β and the depth through
the wall (see :mod:`ccdaf.core.uac.volume`).

One choice, where the epicardium's α and β come from: carried through the
wall from the endocardium (the default), or the epicardium's own surface
UAC (as in Roney et al. 2023). The run takes about a minute, so it runs
in a thread; the window stays open and shows the report. Nothing is
written until OK.
"""
from __future__ import annotations

from typing import Optional

from PyQt5 import QtCore, QtWidgets

from ccdaf.core.uac import volume as vuac


class VolumeUacWorker(QtCore.QThread):
    """Runs :func:`ccdaf.core.uac.volume.compute` off the GUI thread."""

    status = QtCore.pyqtSignal(str)
    done = QtCore.pyqtSignal(object)        # a VolumeUac
    failed = QtCore.pyqtSignal(str)

    def __init__(self, grid, epicardium: str, parent=None) -> None:
        super().__init__(parent)
        self._grid = grid
        self._epicardium = epicardium
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:                    # noqa: D401 - Qt override
        try:
            result = vuac.compute(self._grid, self._epicardium,
                                  on_status=self.status.emit,
                                  should_stop=lambda: self._stop)
        except vuac.VolumeUacError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:              # report, never crash the app
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        else:
            self.done.emit(result)


class VolumeUacDialog(QtWidgets.QDialog):
    """Compute volume UAC; :meth:`result` is the accepted run."""

    def __init__(self, grid, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Compute volume UAC")
        self._grid = grid
        self._result = None
        self._worker: Optional[VolumeUacWorker] = None

        layout = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            "<i>Surface UAC on the endocardium, carried through the wall in "
            "each half of the atrium, and the depth from the endocardium (0) "
            "to the epicardium (1). Nothing is written until you press "
            "OK.</i>")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QtWidgets.QFormLayout()
        self.cmb_epi = QtWidgets.QComboBox()
        self.cmb_epi.addItem("From the endocardium", vuac.FROM_ENDO)
        self.cmb_epi.addItem("Its own surface UAC", vuac.OWN)
        self.cmb_epi.setToolTip(
            "From the endocardium: α and β run straight out through the wall; "
            "the epicardium fits the template approximately, with no folds of "
            "its own.\nIts own surface UAC: the epicardium fits the template "
            "exactly, but any folds of its surface map are carried into the "
            "wall. Takes about twice as long.")
        form.addRow("Epicardium α, β", self.cmb_epi)
        layout.addLayout(form)

        row = QtWidgets.QHBoxLayout()
        self.btn_run = QtWidgets.QPushButton("Compute")
        self.btn_run.clicked.connect(self.run)
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop)
        row.addWidget(self.btn_run)
        row.addWidget(self.btn_stop)
        row.addStretch(1)
        layout.addLayout(row)

        self.report = QtWidgets.QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setMinimumSize(520, 200)
        layout.addWidget(self.report)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self.btn_ok = buttons.button(QtWidgets.QDialogButtonBox.Ok)
        self.btn_ok.setEnabled(False)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ------------------------------------------------------------------
    def epicardium(self) -> str:
        return str(self.cmb_epi.currentData())

    def result(self):
        return self._result

    def matches(self, grid) -> bool:
        return grid is self._grid

    def run(self) -> None:
        if self._worker is not None:
            return
        self._result = None
        self.btn_ok.setEnabled(False)
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.cmb_epi.setEnabled(False)
        self.report.setPlainText("Starting…")
        self._worker = VolumeUacWorker(self._grid, self.epicardium(), self)
        self._worker.status.connect(self.report.setPlainText)
        self._worker.done.connect(self._done)
        self._worker.failed.connect(self._failed)
        self._worker.finished.connect(self._finished)
        self._worker.start()

    def stop(self) -> None:
        if self._worker is not None:
            self._worker.stop()
            self.report.setPlainText("Stopping…")

    def wait(self) -> None:
        """Block until a run in progress ends (for closing and tests)."""
        if self._worker is not None:
            self._worker.wait()
            QtWidgets.QApplication.processEvents()

    def _done(self, result) -> None:
        self._result = result
        self.report.setPlainText(result.details())
        self.btn_ok.setEnabled(True)

    def _failed(self, message: str) -> None:
        self.report.setPlainText(f"Could not compute volume UAC: {message}")

    def _finished(self) -> None:
        self._worker = None
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.cmb_epi.setEnabled(True)

    def reject(self) -> None:
        self.stop()
        self.wait()
        super().reject()


__all__ = ["VolumeUacDialog", "VolumeUacWorker"]
