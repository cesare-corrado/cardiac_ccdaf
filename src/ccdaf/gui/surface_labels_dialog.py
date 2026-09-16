"""
SurfaceLabelsDialog
===================
Confirm the basal plane, then label the boundary surfaces.

The dialog opens with an answer rather than a blank form: the cut plane is
detected and the labelling is run straight away, so the common case is a
glance and OK. The plane is still shown as six editable numbers, because
detection maximises coplanar area and knows nothing about anatomy, so on a
mesh flat at both ends it can pick the end the cavities do not open onto.

Nothing is written until OK. The report shown is the one the core
produces, including its refusal: a plane in the wrong place, or a mesh
that still has its valve orifices, leaves the boundary in one piece
instead of three, and that is said in words rather than left to appear
later as a Laplace solve that cannot be trusted.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from PyQt5 import QtCore, QtWidgets

from ccdaf.core.surface_labels import (
    LabelOptions, Plane, SurfaceLabels, detect_base_plane,
    label_boundary_or_snap,
)


def _spin(value: float, *, decimals: int = 3, step: float = 0.1,
          low: float = -1.0e6, high: float = 1.0e6) -> QtWidgets.QDoubleSpinBox:
    box = QtWidgets.QDoubleSpinBox()
    box.setDecimals(decimals)
    box.setRange(low, high)
    box.setSingleStep(step)
    box.setValue(value)
    box.setKeyboardTracking(False)
    box.setMinimumWidth(90)
    return box


class SurfaceLabelsDialog(QtWidgets.QDialog):
    """Pick the basal plane and label base, epicardium, LV and RV.

    Shown **non-modally**: a modal dialog blocks the mouse everywhere else,
    and hiding it does not lift that, so the plane could not be dragged in
    the 3D view. The window holds no VTK of its own — it asks through
    signals and the application owns the gizmo and the overlays, which is
    the same division every other panel here follows.
    """

    #: Show (True) or take away (False) the draggable plane in the view.
    plane_edit_requested = QtCore.pyqtSignal(bool)
    #: A fresh labelling to preview, or ``None`` to clear the overlays.
    preview_requested = QtCore.pyqtSignal(object)

    def __init__(self, dataset, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Label ventricular surfaces")
        self._dataset = dataset
        self._labels: Optional[SurfaceLabels] = None

        layout = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            "<i>The basal cut plane is detected, then the boundary is split "
            "into base, epicardium, LV endocardium and RV endocardium. "
            "Nothing is written until you press OK.</i>")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        # --- the plane ------------------------------------------------
        plane_box = QtWidgets.QGroupBox("Basal plane")
        plane_layout = QtWidgets.QVBoxLayout(plane_box)
        self._origin: List[QtWidgets.QDoubleSpinBox] = []
        self._normal: List[QtWidgets.QDoubleSpinBox] = []
        for title, boxes, step in (("point on the plane", self._origin, 1.0),
                                   ("normal", self._normal, 0.05)):
            row = QtWidgets.QHBoxLayout()
            label = QtWidgets.QLabel(title)
            label.setToolTip(
                "A point the plane passes through, in mesh units."
                if boxes is self._origin else
                "The plane's normal. Its length does not matter; only its "
                "direction is used, and the sign is irrelevant.")
            row.addWidget(label)
            for _ in range(3):
                box = _spin(0.0, step=step)
                box.valueChanged.connect(self._on_plane_edited)
                row.addWidget(box)
                boxes.append(box)
            row.addStretch(1)
            plane_layout.addLayout(row)

        button_row = QtWidgets.QHBoxLayout()
        self.btn_detect = QtWidgets.QPushButton("Detect again")
        self.btn_detect.setToolTip(
            "Find the largest flat, coplanar patch of the boundary.\n"
            "It maximises area and knows nothing about anatomy, so on a mesh "
            "flat at both ends it can pick the wrong one.")
        self.btn_detect.clicked.connect(self._detect)
        self.btn_check = QtWidgets.QPushButton("Check this plane")
        self.btn_check.setToolTip(
            "Label the surfaces with the plane above and report the result, "
            "without writing anything.")
        self.btn_check.clicked.connect(self._check)
        self.btn_modify = QtWidgets.QPushButton("Modify plane")
        self.btn_modify.setCheckable(True)
        self.btn_modify.setToolTip(
            "Show a draggable plane in the 3D view.\n"
            "Drag its centre to move it and its arrow to turn it; the "
            "numbers above follow as you go.\n"
            "Press the button again to put the plane away, then "
            "'Check this plane'.")
        self.btn_modify.toggled.connect(self.plane_edit_requested.emit)
        button_row.addWidget(self.btn_detect)
        button_row.addWidget(self.btn_modify)
        button_row.addWidget(self.btn_check)
        button_row.addStretch(1)
        plane_layout.addLayout(button_row)
        layout.addWidget(plane_box)

        # --- the rule -------------------------------------------------
        rule_row = QtWidgets.QHBoxLayout()
        tol_label = QtWidgets.QLabel("distance (× mean edge)")
        tol_tip = (
            "A face joins the base when all three of its nodes lie within "
            "this many mean edge lengths of the plane.\n"
            "A planar cut is exactly planar, so the value barely matters: "
            "0.5 and 1.0 select the same faces on a real cut.")
        tol_label.setToolTip(tol_tip)
        self.spn_tol = _spin(LabelOptions().tol_edges, decimals=2, step=0.1,
                             low=0.01, high=10.0)
        self.spn_tol.setToolTip(tol_tip)
        cos_label = QtWidgets.QLabel("| cos angle | >")
        cos_tip = (
            "How parallel a face must be to the plane to join the base.\n"
            "Without it, faces tangent to the plane are picked up; with the "
            "distance test alone removed, the selection scatters into 28 "
            "patches on the example instead of 1.")
        cos_label.setToolTip(cos_tip)
        self.spn_cos = _spin(LabelOptions().cos_min, decimals=2, step=0.01,
                             low=0.01, high=1.0)
        self.spn_cos.setToolTip(cos_tip)
        for widget in (tol_label, self.spn_tol, cos_label, self.spn_cos):
            rule_row.addWidget(widget)
        rule_row.addStretch(1)
        layout.addLayout(rule_row)

        # --- the report -----------------------------------------------
        self.report = QtWidgets.QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setMinimumHeight(170)
        self.report.setLineWrapMode(QtWidgets.QPlainTextEdit.WidgetWidth)
        layout.addWidget(self.report)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self.btn_ok = buttons.button(QtWidgets.QDialogButtonBox.Ok)
        self.btn_ok.setToolTip("Write the labels onto the mesh.")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        QtCore.QTimer.singleShot(0, self._detect_and_check)
        self.adjustSize()

    # -----------------------------------------------------------------
    def options(self) -> LabelOptions:
        return LabelOptions(tol_edges=float(self.spn_tol.value()),
                            cos_min=float(self.spn_cos.value()))

    def plane(self) -> Optional[Plane]:
        normal = np.array([b.value() for b in self._normal], dtype=float)
        if float(np.linalg.norm(normal)) < 1e-12:
            return None
        return Plane(origin=[b.value() for b in self._origin], normal=normal)

    def result(self) -> Optional[SurfaceLabels]:
        return self._labels

    # -----------------------------------------------------------------
    def set_plane(self, plane: Plane) -> None:
        """Adopt *plane*, as the gizmo does while it is being dragged."""
        self._set_plane(plane)
        self._on_plane_edited()

    def _set_plane(self, plane: Plane) -> None:
        for box, value in zip(self._origin, plane.origin):
            box.blockSignals(True)
            box.setValue(float(value))
            box.blockSignals(False)
        for box, value in zip(self._normal, plane.normal):
            box.blockSignals(True)
            box.setValue(float(value))
            box.blockSignals(False)

    def _on_plane_edited(self, *_args) -> None:
        """An edited plane invalidates the last answer until re-checked."""
        self._labels = None
        self.btn_ok.setEnabled(False)

    def _detect_and_check(self) -> None:
        if self._detect():
            self._check()

    def _detect(self) -> bool:
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            found = detect_base_plane(self._dataset, self.options())
        except ValueError as exc:
            self._fail(str(exc))
            return False
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        if found is None:
            self._fail("No flat cut could be found on this mesh.")
            return False
        self._set_plane(found)
        self._on_plane_edited()
        return True

    def _check(self) -> None:
        plane = self.plane()
        if plane is None:
            self._fail("The normal has no length, so it names no plane.")
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            labels, snapped = label_boundary_or_snap(
                self._dataset, plane, self.options())
        except ValueError as exc:
            self._fail(str(exc))
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        prefix = ""
        if snapped:
            # The boxes follow the plane that was used. Leaving them on the
            # plane that failed would make the report describe something
            # the window is not showing.
            # How far the plane moved, measured across it. The snapped
            # origin is some face centre lying anywhere on the cut, so the
            # straight-line distance between the two origins says more
            # about which face was sampled than about the move: on the
            # example it reads 45 for a plane 6 away.
            moved = abs(float(
                (np.asarray(labels.plane.origin) - np.asarray(plane.origin))
                @ np.asarray(plane.normal)))
            self._set_plane(labels.plane)
            prefix = (f"Nothing lay on the plane you set, so it snapped to "
                      f"the flat face {moved:.2f} away.\n\n")

        self._labels = labels
        self.btn_ok.setEnabled(True)
        text = prefix + labels.details()
        if not labels.naming_confident:
            text += ("\n\nCheck which cavity is which before using this: the "
                     "LV and RV may be the wrong way round.")
        self.report.setPlainText(text)
        self.preview_requested.emit(labels)

    def _fail(self, message: str) -> None:
        self._labels = None
        self.btn_ok.setEnabled(False)
        self.report.setPlainText(message)
        # A refused plane leaves nothing to look at: a stale highlight from
        # the last good answer would say the opposite of the report.
        self.preview_requested.emit(None)


__all__ = ["SurfaceLabelsDialog"]
