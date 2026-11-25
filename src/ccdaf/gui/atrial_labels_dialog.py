"""
AtrialLabelsDialog
==================
Actions > Label atrial surfaces…: split an atrial volume's boundary into
rim, epicardium and the endocardium of each atrium present.

The window runs the labelling as soon as it opens, so the common case is
a glance and OK. A volume taken out of a tagged mesh by Actions > Extract
chambers carries its rims and is labelled from them; any other volume is
labelled from its shape, which needs the mesh units and takes longer. See
:mod:`ccdaf.core.atrial_surfaces`.

Shown non-modally, like the ventricular window, so the preview in the 3D
view can be turned and zoomed. Nothing is written until OK.
"""
from __future__ import annotations

from typing import Mapping, Optional

from PyQt5 import QtCore, QtWidgets

from ccdaf.core.atrial_surfaces import (
    AtrialOptions, has_recorded_rims, label_atria, swap_atria,
)
from ccdaf.core.orifice_labels import UNITS, guess_unit, rms_radius
from ccdaf.core.surface_labels import SurfaceLabels


class AtrialLabelsDialog(QtWidgets.QDialog):
    """Label an atrial volume; :meth:`result` is the accepted labelling."""

    #: Kept for the application's labelling slot, which also serves the
    #: ventricular window; an atrium has no plane to drag.
    plane_edit_requested = QtCore.pyqtSignal(bool)
    #: A fresh labelling to preview, or ``None`` to clear the overlays.
    preview_requested = QtCore.pyqtSignal(object)

    def __init__(self, dataset, mapping: Optional[Mapping[int, object]] = None,
                 parent: Optional[QtWidgets.QWidget] = None,
                 run_now: bool = True) -> None:
        super().__init__(parent)
        self.setWindowTitle("Label atrial surfaces")
        self._dataset = dataset
        self._mapping = mapping
        self._labels: Optional[SurfaceLabels] = None
        self._tagged = has_recorded_rims(dataset)

        layout = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            "<i>The boundary is split into the rim (where endocardium meets "
            "epicardium at each opening), the epicardium, and the "
            "endocardium of each atrium. Nothing is written until you press "
            "OK.</i>")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QtWidgets.QFormLayout()
        method = ("Recorded rims (from Extract chambers)" if self._tagged
                  else "Openings found from the shape")
        self.lbl_method = QtWidgets.QLabel(method)
        self.lbl_method.setToolTip(
            "A volume taken out of a tagged mesh with Actions > Extract "
            "chambers remembers where its caps and valves were, and the rims "
            "come from that. Any other volume is voxelised and its openings "
            "found where the blood pools meet the outside.")
        form.addRow("Method", self.lbl_method)

        unit_row = QtWidgets.QHBoxLayout()
        self.cmb_unit = QtWidgets.QComboBox()
        self.cmb_unit.addItems(list(UNITS))
        guessed = guess_unit(dataset)
        self.cmb_unit.setCurrentText(guessed)
        self.cmb_unit.setToolTip(
            "The length unit of the mesh coordinates, guessed from its size. "
            "The shape method works in millimetres.")
        unit_row.addWidget(self.cmb_unit)
        unit_row.addWidget(QtWidgets.QLabel(
            f"RMS radius {rms_radius(dataset) * UNITS[guessed]:.1f} mm"))
        unit_row.addStretch(1)
        form.addRow("Mesh units", unit_row)

        self.spn_voxel = QtWidgets.QDoubleSpinBox()
        self.spn_voxel.setRange(0.1, 2.0)
        self.spn_voxel.setSingleStep(0.1)
        self.spn_voxel.setDecimals(2)
        self.spn_voxel.setValue(AtrialOptions.voxel_mm)
        self.spn_voxel.setSuffix(" mm")
        self.spn_voxel.setToolTip(
            "Voxel edge for the shape method. It must be about half the "
            "thinnest wall, or the wall leaks on voxels.")
        form.addRow("Voxel size", self.spn_voxel)
        for w in (self.cmb_unit, self.spn_voxel):
            w.setEnabled(not self._tagged)
        layout.addLayout(form)

        row = QtWidgets.QHBoxLayout()
        self.btn_find = QtWidgets.QPushButton("Find surfaces")
        self.btn_find.clicked.connect(self.run)
        self.btn_swap = QtWidgets.QPushButton("Swap LA and RA")
        self.btn_swap.setToolTip(
            "Exchange the two endocardia's names (or rename the only one), "
            "when the naming was guessed wrong.")
        self.btn_swap.setEnabled(False)
        self.btn_swap.clicked.connect(self._swap)
        row.addWidget(self.btn_find)
        row.addWidget(self.btn_swap)
        row.addStretch(1)
        layout.addLayout(row)

        self.report = QtWidgets.QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setMinimumSize(520, 220)
        layout.addWidget(self.report)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self.btn_ok = buttons.button(QtWidgets.QDialogButtonBox.Ok)
        self.btn_ok.setEnabled(False)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        if run_now:
            QtCore.QTimer.singleShot(0, self.run)

    # ------------------------------------------------------------------
    def options(self) -> AtrialOptions:
        return AtrialOptions(mm_per_unit=UNITS[self.cmb_unit.currentText()],
                             voxel_mm=float(self.spn_voxel.value()))

    def result(self) -> Optional[SurfaceLabels]:
        return self._labels

    def run(self) -> None:
        """Label the volume with the current settings and show the report."""
        self.report.setPlainText("Labelling…")
        QtWidgets.QApplication.processEvents()
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            labels = label_atria(self._dataset, self._mapping, self.options())
        except (ValueError, MemoryError) as exc:
            self._labels = None
            self.btn_ok.setEnabled(False)
            self.btn_swap.setEnabled(False)
            self.report.setPlainText(f"Could not label the surfaces: {exc}")
            self.preview_requested.emit(None)
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self._show(labels)

    def _show(self, labels: SurfaceLabels) -> None:
        self._labels = labels
        self.btn_ok.setEnabled(True)
        self.btn_swap.setEnabled(True)
        text = labels.details()
        if not labels.naming_confident:
            text += ("\n\nCheck which atrium is which before using this: "
                     "the names are a guess.")
        self.report.setPlainText(text)
        self.preview_requested.emit(labels)

    def _swap(self) -> None:
        if self._labels is not None:
            self._show(swap_atria(self._labels))


__all__ = ["AtrialLabelsDialog"]
