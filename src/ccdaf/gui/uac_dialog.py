"""
UacDialog
=========
Compute universal atrial coordinates on a labelled left-atrial surface.

There is nothing to choose: the window says what will happen (and
whether it replaces coordinates the surface already has), runs it on a
worker thread (:class:`UacWorker`) with progress in the report, and can stop
it. When a run finishes the application replaces the surface with its
repaired copy carrying ``uac_alpha`` and ``uac_beta``; **Undo** puts the
previous surface back. **Save flat map…** works on any surface that has the
two arrays, from a run or from a file.
"""
from __future__ import annotations

import hashlib
import threading
from typing import Optional

import numpy as np
from PyQt5 import QtCore, QtWidgets

from ccdaf.core import uac


def digest(points: np.ndarray, triangles: np.ndarray, tags: np.ndarray) -> str:
    """A fingerprint of the mesh and its labels, to spot a change mid-run."""
    h = hashlib.sha1()
    for a in (np.ascontiguousarray(points, np.float64), np.ascontiguousarray(triangles, np.int64),
              np.ascontiguousarray(tags, np.int64)):
        h.update(a.tobytes())
    return h.hexdigest()


class UacWorker(QtCore.QThread):
    """Run :func:`ccdaf.core.uac.compute` off the GUI thread, on plain arrays."""

    status = QtCore.pyqtSignal(str)
    done = QtCore.pyqtSignal(object)        # a UacRun
    failed = QtCore.pyqtSignal(str)

    def __init__(self, points, triangles, tags,
                 parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self._args = (points, triangles, tags)
        self._stop = threading.Event()

    def stop(self) -> None:
        """Ask the run to stop at its next check (within a second or two)."""
        self._stop.set()

    def run(self) -> None:                    # noqa: D401 - Qt override
        try:
            result = uac.compute(*self._args, on_status=self.status.emit,
                                 should_stop=self._stop.is_set)
        except Exception as exc:               # reported, never raised in a thread
            self.failed.emit(str(exc))
            return
        self.done.emit(result)


class UacDialog(QtWidgets.QDialog):
    """The run, its report and the flat-map export, for one surface. Non-modal."""

    run_requested = QtCore.pyqtSignal()
    stop_requested = QtCore.pyqtSignal()
    undo_requested = QtCore.pyqtSignal()

    def __init__(self, mesh, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Compute UAC")
        self._mesh = mesh
        self._before = None                    # the surface before the last run
        self._inputs = uac.inputs(mesh)
        self._digest = digest(*self._inputs)

        layout = QtWidgets.QVBoxLayout(self)
        if uac.which_atrium(self._inputs[2]) == "RA":
            axes = ("Right atrium. Universal atrial coordinates: alpha from the lateral "
                    "tricuspid valve (0) over the roof (0.5) to the septal tricuspid valve "
                    "(1), beta from the IVC (0) to the SVC (1). Venae cavae, coronary sinus "
                    "and appendage land in the same place for every patient. ")
        else:
            axes = ("Left atrium. Universal atrial coordinates: alpha from septal (0) to "
                    "lateral (1), beta from the posterior mitral valve (0) over the roof "
                    "(0.5) to the anterior mitral valve (1). Veins and appendage land in "
                    "the same place for every patient. ")
        intro = QtWidgets.QLabel(
            "<i>" + axes +
            "Fully automatic, from the elemTag labels; a run takes "
            "a few minutes. The surface is repaired first (label islands, duplicate "
            "triangles, ears and slivers on openings, edge flips, pinched points, thin "
            "tunnels) and the repaired surface replaces it.</i>")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        solvers = uac.available_solvers()
        lbl = QtWidgets.QLabel(f"Solver: {solvers[0]}.")
        lbl.setToolTip("The fastest installed sparse solver is used: CHOLMOD "
                       "(scikit-sparse), else PARDISO (pypardiso), else SciPy.")
        layout.addWidget(lbl)
        self.replace_note = QtWidgets.QLabel(
            f"<b>This surface already has {uac.ALPHA} and {uac.BETA}; Compute will "
            f"replace them.</b>")
        self.replace_note.setWordWrap(True)
        self.replace_note.setVisible(self.has_uac(mesh))
        layout.addWidget(self.replace_note)

        self.report = QtWidgets.QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setMinimumHeight(220)
        self.report.setLineWrapMode(QtWidgets.QPlainTextEdit.WidgetWidth)
        layout.addWidget(self.report)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        self.btn_run = buttons.addButton("Compute", QtWidgets.QDialogButtonBox.ActionRole)
        self.btn_run.setToolTip("Compute alpha and beta and write uac_alpha and uac_beta "
                                "onto the surface.")
        self.btn_run.clicked.connect(self.run_requested.emit)
        self.btn_stop = buttons.addButton("Stop", QtWidgets.QDialogButtonBox.ActionRole)
        self.btn_stop.setToolTip("Stop the run; nothing is written.")
        self.btn_stop.clicked.connect(self.stop_requested.emit)
        self.btn_save = buttons.addButton("Save flat map…", QtWidgets.QDialogButtonBox.ActionRole)
        self.btn_save.setToolTip("Save the flat square map as its own mesh (VTK), drawn "
                                 "from the surface's uac_alpha and uac_beta. Points on the "
                                 "cut appear once per half; uac_source gives each flat "
                                 "point's point on the atrium.")
        self.btn_save.clicked.connect(self._on_save)
        self.btn_undo = buttons.addButton("Undo", QtWidgets.QDialogButtonBox.ActionRole)
        self.btn_undo.setToolTip("Put back the surface as it was before the last Compute: "
                                 "its own triangles and labels, without the repairs and "
                                 "without the new uac_alpha and uac_beta.")
        self.btn_undo.clicked.connect(self.undo_requested.emit)
        buttons.rejected.connect(self.reject)
        self.btn_close = buttons.button(QtWidgets.QDialogButtonBox.Close)
        layout.addWidget(buttons)
        self.set_running(False)
        self.set_status("Press Compute to start.")
        self.adjustSize()

    # -----------------------------------------------------------------
    @staticmethod
    def has_uac(mesh) -> bool:
        """Whether *mesh* already carries the coordinates a run writes."""
        names = set(mesh.point_data.keys())
        return any(n in names for n in uac.written_names())

    def inputs(self):
        return self._inputs

    def adopt(self, mesh, before=None) -> None:
        """Follow the surface the application now shows. ``before``: the
        surface it replaced, which Undo puts back (``None`` keeps none)."""
        self._mesh = mesh
        self._inputs = uac.inputs(mesh)
        self._digest = digest(*self._inputs)
        self._before = before
        self.replace_note.setVisible(self.has_uac(mesh))
        self.set_running(False)

    def before(self):
        return self._before

    def matches(self, mesh) -> bool:
        """Whether *mesh* is still the surface and labels this window read."""
        try:
            return digest(*uac.inputs(mesh)) == self._digest
        except Exception:
            return False

    def show_run(self, run) -> None:
        self.set_status(run.details())

    def set_running(self, running: bool) -> None:
        self.btn_run.setEnabled(not running)
        self.btn_close.setEnabled(not running)
        self.btn_stop.setEnabled(running)
        self.btn_save.setEnabled(not running and uac.has_coordinates(self._mesh))
        self.btn_undo.setEnabled(not running and self._before is not None)

    def set_status(self, text: str) -> None:
        self.report.setPlainText(text)

    def _on_save(self) -> None:
        if not uac.has_coordinates(self._mesh):
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save flat UAC map", "uac_flat.vtk", "VTK files (*.vtk)")
        if not path:
            return
        try:
            uac.flat_mesh(self._mesh).save(path, binary=False)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Save flat UAC map", str(exc))


__all__ = ["UacDialog", "UacWorker", "digest"]
