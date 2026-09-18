"""
FibreDialog
===========
Generate ventricular fibres: confirm the apex, set the angles, run.

The window opens with the apex already found, so the common case is a
glance and Generate. The apex can be replaced by clicking the epicardium
in the 3D view; the dialog only asks for that through a signal, and the
application owns the picker and the marker, the same division the surface
labelling window follows.

The run itself happens on a worker thread (:class:`FibreWorker`): four
Laplace solves on a 3-million-element ventricle take over a minute, and a
window that stops repainting for that long looks crashed. The dialog stays
open afterwards, so trying other angles is a second press of Generate,
and that one reuses the solved fields and takes seconds.

Only the published rule is offered. The linear variant exists in the core
but is not yet checked against reference fibres, and a rule that has not
been checked should not be one click from a saved mesh.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
from PyQt5 import QtCore, QtWidgets

from ccdaf.core import ventricular_fibres as vf
from ccdaf.core.ldrb import FibreAngles


class FibreWorker(QtCore.QThread):
    """Run :func:`ccdaf.core.ventricular_fibres.generate` off the GUI thread.

    It is given plain arrays, never a VTK object: VTK is not safe to touch
    from two threads, and the numerics do not need it.
    """

    status = QtCore.pyqtSignal(str)
    done = QtCore.pyqtSignal(object)        # a FibreRun
    failed = QtCore.pyqtSignal(str)

    def __init__(self, points: np.ndarray, tets: np.ndarray, mask: np.ndarray,
                 apex: vf.Apex, angles: FibreAngles,
                 cached: Optional[Dict[str, np.ndarray]] = None,
                 parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self._args = (points, tets, mask, apex, angles)
        self._cached = cached

    def run(self) -> None:                    # noqa: D401 - Qt override
        try:
            result = vf.generate(*self._args, cached=self._cached,
                                 on_status=self.status.emit)
        except Exception as exc:               # reported, never raised in a thread
            self.failed.emit(str(exc))
            return
        self.done.emit(result)


def _angle_spin(value: float, tip: str) -> QtWidgets.QDoubleSpinBox:
    box = QtWidgets.QDoubleSpinBox()
    box.setDecimals(1)
    box.setRange(-90.0, 90.0)
    box.setSingleStep(5.0)
    box.setSuffix(" °")
    box.setValue(value)
    box.setKeyboardTracking(False)
    box.setToolTip(tip)
    return box


class FibreDialog(QtWidgets.QDialog):
    """Apex, angles and the run, for one labelled volume.

    Shown **non-modally**, so the apex can be clicked in the 3D view while
    the window is open.
    """

    #: The apex changed (an :class:`~ccdaf.core.ventricular_fibres.Apex`):
    #: redraw the marker.
    apex_changed = QtCore.pyqtSignal(object)
    #: Start (True) or stop (False) picking the apex in the view.
    pick_requested = QtCore.pyqtSignal(bool)
    #: Generate with ``(apex, angles)``.
    run_requested = QtCore.pyqtSignal(object, object)

    def __init__(self, grid, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Generate fibres")
        self._points, self._tets, self._mask = vf.inputs(grid)
        self._digest = vf.digest(self._points, self._tets, self._mask)
        self._apex: Optional[vf.Apex] = None

        layout = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            "<i>Rule-based fibres (Bayer et al. 2012), the same rule as "
            "GlRuleFibers. Four Laplace problems are solved on the labelled "
            "surfaces, then a fibre and a sheet direction are set in every "
            "element.</i>")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        # Where the mesh's fibres came from, and what has happened to them
        # since: generated here, carried from a previous mesh, or brought
        # by the file. Empty when there are none.
        self.lbl_note = QtWidgets.QLabel()
        self.lbl_note.setWordWrap(True)
        layout.addWidget(self.lbl_note)
        self.refresh_provenance(grid)

        # --- apex -----------------------------------------------------
        apex_box = QtWidgets.QGroupBox("Apex")
        apex_layout = QtWidgets.QVBoxLayout(apex_box)
        self.lbl_apex = QtWidgets.QLabel("—")
        self.lbl_apex.setWordWrap(True)
        apex_layout.addWidget(self.lbl_apex)
        row = QtWidgets.QHBoxLayout()
        self.btn_auto = QtWidgets.QPushButton("Find automatically")
        self.btn_auto.setToolTip(
            "The epicardial node farthest from the base plane, with the "
            "epicardial nodes within one mean edge of it.")
        self.btn_auto.clicked.connect(self.find_apex)
        self.btn_pick = QtWidgets.QPushButton("Pick on surface")
        self.btn_pick.setCheckable(True)
        self.btn_pick.setToolTip(
            "Click the epicardium in the 3D view; the nearest epicardial "
            "node becomes the apex.\nPress again to stop picking.")
        self.btn_pick.toggled.connect(self.pick_requested.emit)
        row.addWidget(self.btn_auto)
        row.addWidget(self.btn_pick)
        row.addStretch(1)
        apex_layout.addLayout(row)
        layout.addWidget(apex_box)

        # --- angles ---------------------------------------------------
        angle_box = QtWidgets.QGroupBox("Angles")
        grid_layout = QtWidgets.QGridLayout(angle_box)
        d = FibreAngles()
        self.spn_alpha_endo = _angle_spin(
            d.alpha_endo, "Helix angle at the endocardium.")
        self.spn_alpha_epi = _angle_spin(
            d.alpha_epi, "Helix angle at the epicardium.")
        self.spn_beta_endo = _angle_spin(
            d.beta_endo, "Sheet (transverse) angle at the endocardium.")
        self.spn_beta_epi = _angle_spin(
            d.beta_epi, "Sheet (transverse) angle at the epicardium.")
        grid_layout.addWidget(QtWidgets.QLabel(""), 0, 0)
        grid_layout.addWidget(QtWidgets.QLabel("endocardium"), 0, 1)
        grid_layout.addWidget(QtWidgets.QLabel("epicardium"), 0, 2)
        grid_layout.addWidget(QtWidgets.QLabel("helix α"), 1, 0)
        grid_layout.addWidget(self.spn_alpha_endo, 1, 1)
        grid_layout.addWidget(self.spn_alpha_epi, 1, 2)
        grid_layout.addWidget(QtWidgets.QLabel("sheet β"), 2, 0)
        grid_layout.addWidget(self.spn_beta_endo, 2, 1)
        grid_layout.addWidget(self.spn_beta_epi, 2, 2)
        self.btn_defaults = QtWidgets.QPushButton("Defaults")
        self.btn_defaults.setToolTip("GlRuleFibers' defaults: α 40 / −50, β −65 / 25.")
        self.btn_defaults.clicked.connect(self.reset_angles)
        grid_layout.addWidget(self.btn_defaults, 3, 2)
        layout.addWidget(angle_box)

        # --- report ---------------------------------------------------
        self.report = QtWidgets.QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setMinimumHeight(180)
        self.report.setLineWrapMode(QtWidgets.QPlainTextEdit.WidgetWidth)
        layout.addWidget(self.report)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        self.btn_run = buttons.addButton("Generate", QtWidgets.QDialogButtonBox.ActionRole)
        self.btn_run.setToolTip(
            "Solve (or reuse) the Laplace fields and write 'fiber' and "
            "'sheet' onto the mesh.")
        self.btn_run.clicked.connect(self._on_run)
        buttons.rejected.connect(self.reject)
        self.btn_close = buttons.button(QtWidgets.QDialogButtonBox.Close)
        layout.addWidget(buttons)

        QtCore.QTimer.singleShot(0, self.find_apex)
        self.adjustSize()

    # -----------------------------------------------------------------
    def apex(self) -> Optional[vf.Apex]:
        return self._apex

    def angles(self) -> FibreAngles:
        return FibreAngles(alpha_endo=float(self.spn_alpha_endo.value()),
                           alpha_epi=float(self.spn_alpha_epi.value()),
                           beta_endo=float(self.spn_beta_endo.value()),
                           beta_epi=float(self.spn_beta_epi.value()))

    def matches(self, grid) -> bool:
        """Whether *grid* is still the mesh and labels this window measured."""
        try:
            return np.array_equal(vf.digest(*vf.inputs(grid)), self._digest)
        except Exception:
            return False

    def inputs(self):
        """``(points, tets, mask)`` the window was opened on."""
        return self._points, self._tets, self._mask

    # -----------------------------------------------------------------
    def find_apex(self) -> None:
        self._stop_picking()
        self._adopt(lambda: vf.find_apex(self._points, self._tets, self._mask))

    def set_apex_at(self, position) -> None:
        """Make the epicardial node nearest *position* the apex."""
        self._adopt(lambda: vf.apex_at(self._points, self._tets, self._mask,
                                       position))

    def reset_angles(self) -> None:
        d = FibreAngles()
        for box, value in ((self.spn_alpha_endo, d.alpha_endo),
                           (self.spn_alpha_epi, d.alpha_epi),
                           (self.spn_beta_endo, d.beta_endo),
                           (self.spn_beta_epi, d.beta_epi)):
            box.setValue(value)

    def set_running(self, running: bool) -> None:
        """Lock the inputs while a run is in flight."""
        for w in (self.btn_run, self.btn_auto, self.btn_pick, self.btn_defaults,
                  self.spn_alpha_endo, self.spn_alpha_epi, self.spn_beta_endo,
                  self.spn_beta_epi, self.btn_close):
            w.setEnabled(not running)
        if not running:
            self.btn_run.setEnabled(self._apex is not None)

    def set_status(self, text: str) -> None:
        self.report.setPlainText(text)

    def set_note(self, text: str) -> None:
        """Show *text* above the controls, or hide the line for ``""``."""
        self.lbl_note.setText(f"<b>{text}</b>" if text else "")
        self.lbl_note.setVisible(bool(text))

    def refresh_provenance(self, grid) -> None:
        """Say where *grid*'s fibres came from."""
        self.set_note(vf.describe_provenance(grid))

    def uncheck_pick(self) -> None:
        """Put the pick button back without emitting (the picker was taken)."""
        self.btn_pick.blockSignals(True)
        self.btn_pick.setChecked(False)
        self.btn_pick.blockSignals(False)

    # -----------------------------------------------------------------
    def _adopt(self, make: Callable[[], vf.Apex]) -> None:
        try:
            apex = make()
        except ValueError as exc:
            self._apex = None
            self.lbl_apex.setText(f"No apex: {exc}")
            self.btn_run.setEnabled(False)
            self.apex_changed.emit(None)
            return
        self._apex = apex
        self.lbl_apex.setText(apex.describe().capitalize() + ".")
        self.btn_run.setEnabled(True)
        self.apex_changed.emit(apex)

    def _stop_picking(self) -> None:
        if self.btn_pick.isChecked():
            self.btn_pick.setChecked(False)       # emits pick_requested(False)

    def _on_run(self) -> None:
        if self._apex is None:
            return
        self._stop_picking()
        self.run_requested.emit(self._apex, self.angles())

    def reject(self) -> None:
        self._stop_picking()
        super().reject()


__all__ = ["FibreDialog", "FibreWorker"]
