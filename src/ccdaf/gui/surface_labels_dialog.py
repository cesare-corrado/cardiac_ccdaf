"""
SurfaceLabelsDialog
===================
Find the base, then label the boundary surfaces.

Two methods, one window. A truncated mesh has a flat basal cut, shown as
six editable numbers. A mesh that still has its valve openings has a ring
at each opening instead, shown as a list of the openings found. The
method is chosen automatically when the window opens (flat cut first,
openings if the cut does not split the surface), and can be forced.

The dialog opens with an answer rather than a blank form, so the common
case is a glance and OK. The plane stays editable because detection
maximises coplanar area and knows nothing about anatomy, so on a mesh
flat at both ends it can pick the end the cavities do not open onto.

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

from ccdaf.core.orifice_labels import (
    UNITS, Opening, OrificeOptions, find_openings, guess_unit,
    label_open_boundary, rms_radius,
)
from ccdaf.core.surface_labels import (
    LabelOptions, Plane, SurfaceLabels, detect_base_plane,
    label_boundary_or_snap,
)

#: Entries of the method box, in order.
AUTOMATIC, FLAT_CUT, OPENINGS = "Automatic", "Flat cut", "Valve openings"


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
    """Find the base (a flat cut or valve openings) and label the surfaces.

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
        self._openings: List[Opening] = []
        self._last_error = ""
        #: Whether the detected flat cut splits the mesh into three, i.e.
        #: the mesh is truncated. Learnt by the automatic run on opening and
        #: kept, because it is a property of the mesh, not of the controls.
        self._truncated: Optional[bool] = None

        layout = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            "<i>The base is found, then the boundary is split into base, "
            "epicardium, LV endocardium and RV endocardium. "
            "Nothing is written until you press OK.</i>")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        # --- method and units -------------------------------------------
        top = QtWidgets.QFormLayout()
        self.cmb_method = QtWidgets.QComboBox()
        self.cmb_method.addItems([AUTOMATIC, FLAT_CUT, OPENINGS])
        self.cmb_method.setToolTip(
            "How the base is found.\n"
            "Flat cut: a truncated mesh, cut by one plane.\n"
            "Valve openings: a mesh that still has its valves open; the "
            "base is a ring at each opening.\n"
            "Automatic tries the flat cut first and falls back to the "
            "openings when the cut does not split the surface into three.")
        self.cmb_method.activated.connect(self._on_method_chosen)
        top.addRow("Method", self.cmb_method)

        unit_row = QtWidgets.QHBoxLayout()
        self.cmb_unit = QtWidgets.QComboBox()
        self.cmb_unit.addItems(list(UNITS))
        guessed = guess_unit(dataset)
        self.cmb_unit.setCurrentText(guessed)
        unit_tip = (
            "The length unit of the mesh coordinates. The valve-opening "
            "method works in millimetres, so it needs to know.\n"
            "Guessed from the mesh size: a ventricle's RMS radius about "
            "its centre is about 45 mm. The flat cut does not use it.")
        self.cmb_unit.setToolTip(unit_tip)
        self.cmb_unit.currentTextChanged.connect(self._on_unit_changed)
        self.lbl_unit = QtWidgets.QLabel(
            f"guessed: RMS radius "
            f"{rms_radius(dataset) * UNITS[guessed]:.1f} mm")
        self.lbl_unit.setToolTip(unit_tip)
        unit_row.addWidget(self.cmb_unit)
        unit_row.addWidget(self.lbl_unit)
        unit_row.addStretch(1)
        top.addRow("Mesh units", unit_row)
        layout.addLayout(top)

        # --- the plane ------------------------------------------------
        plane_box = QtWidgets.QGroupBox("Flat cut: basal plane")
        self.grp_plane = plane_box
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
        plane_layout.addLayout(rule_row)
        layout.addWidget(plane_box)

        # --- the openings -----------------------------------------------
        self.grp_openings = QtWidgets.QGroupBox("Valve openings")
        open_layout = QtWidgets.QVBoxLayout(self.grp_openings)
        self.lst_openings = QtWidgets.QListWidget()
        self.lst_openings.setToolTip(
            "The openings found, largest first within each blood pool.\n"
            "A mitral and an aortic valve side by side can come back as "
            "one opening; that is fine, the base there is one ring.")
        self.lst_openings.setMaximumHeight(110)
        open_layout.addWidget(self.lst_openings)
        open_buttons = QtWidgets.QHBoxLayout()
        self.btn_find = QtWidgets.QPushButton("Detect again")
        self.btn_find.setToolTip(
            "Find the blood pools and their openings again, with the units "
            "above, then label.")
        self.btn_find.clicked.connect(self._find_and_check)
        self.btn_remove = QtWidgets.QPushButton("Remove selected")
        self.btn_remove.setToolTip(
            "Drop the selected opening, for one found where there is none. "
            "Then press 'Check these openings'.")
        self.btn_remove.clicked.connect(self._remove_opening)
        self.btn_check_openings = QtWidgets.QPushButton("Check these openings")
        self.btn_check_openings.setToolTip(
            "Build a ring at each opening listed and report the result, "
            "without writing anything.")
        self.btn_check_openings.clicked.connect(self._check_openings)
        for button in (self.btn_find, self.btn_remove,
                       self.btn_check_openings):
            open_buttons.addWidget(button)
        open_buttons.addStretch(1)
        open_layout.addLayout(open_buttons)
        layout.addWidget(self.grp_openings)
        self.grp_openings.hide()

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

        QtCore.QTimer.singleShot(0, self._automatic)
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

    def orifice_options(self) -> OrificeOptions:
        return OrificeOptions(mm_per_unit=UNITS[self.cmb_unit.currentText()])

    def openings(self) -> List[Opening]:
        return list(self._openings)

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

    def _show(self, method: str) -> None:
        """Show the controls of *method* (FLAT_CUT or OPENINGS) only."""
        if method != FLAT_CUT and self.btn_modify.isChecked():
            self.btn_modify.setChecked(False)     # puts the gizmo away
        self.grp_plane.setVisible(method == FLAT_CUT)
        self.grp_openings.setVisible(method == OPENINGS)
        self.adjustSize()

    def _on_method_chosen(self, _index: int = 0) -> None:
        method = self.cmb_method.currentText()
        if method == AUTOMATIC:
            self._automatic()
        elif method == FLAT_CUT:
            self._show(FLAT_CUT)
            self._detect_and_check()
        else:
            self._show(OPENINGS)
            self._find_and_check()

    def _automatic(self) -> None:
        """Flat cut first; the openings when the cut splits nothing."""
        self._show(FLAT_CUT)
        reason = ""
        if self._detect(report=False):
            if self._check(report=False):
                self._truncated = True
                self._prepend("Automatic: the mesh is truncated, so the "
                              "base is its flat cut.\n\n")
                return
            reason = self._last_error
        else:
            reason = self._last_error
        self._truncated = False
        self._show(OPENINGS)
        if self._find_and_check(report=False):
            self._prepend(
                "Automatic: no flat cut splits this mesh "
                f"({reason.rstrip('.')}), so the base was found at its "
                "valve openings.\n\n")
            return
        self._fail(f"No method worked.\n\nAs a truncated mesh: {reason}"
                   f"\n\nWith valve openings: {self._last_error}")

    def _on_unit_changed(self, _text: str = "") -> None:
        """New units make the openings meaningless until found again."""
        self._openings = []
        self.lst_openings.clear()
        if not self.grp_openings.isHidden():
            self._fail("The units changed. Press 'Detect again' to find the "
                       "openings with them.")

    def _is_truncated(self) -> bool:
        """Whether a detected flat cut splits the mesh into three.

        Normally already known from the automatic run. When it is not, it
        is found once here, without touching the plane boxes.
        """
        if self._truncated is None:
            self._busy(True)
            try:
                plane = detect_base_plane(self._dataset, self.options())
                self._truncated = plane is not None and bool(
                    label_boundary_or_snap(self._dataset, plane,
                                           self.options()))
            except ValueError:
                self._truncated = False
            finally:
                self._busy(False)
        return self._truncated

    def _refuse_if_truncated(self, report: bool) -> bool:
        """Refuse the openings on a truncated mesh; True when refused.

        Forced onto a truncated mesh, the openings method finds each
        cavity's mouth on the cut and still splits the surface into three,
        so its own check passes while the answer is wrong: measured on the
        truncated example, the whole flat base came out as epicardium, and
        so did 38% of the LV endocardium.
        """
        if not self._is_truncated():
            return False
        self._failed(
            "This mesh has a flat basal cut that splits it into three "
            "surfaces, so it is truncated. Use the Flat cut method: the "
            "valve-opening method would label the cut as epicardium.",
            report)
        return True

    def _find_and_check(self, *_args, report: bool = True) -> bool:
        if self._refuse_if_truncated(report):
            return False
        self._busy(True)
        try:
            found = find_openings(self._dataset, self.orifice_options())
        except ValueError as exc:
            self._openings = []
            self._list_openings()
            return self._failed(str(exc), report)
        finally:
            self._busy(False)
        self._openings = found.openings
        self._list_openings()
        if not self._openings:
            return self._failed("No valve openings were found.", report)
        return self._check_openings(report=report)

    def _list_openings(self) -> None:
        self.lst_openings.clear()
        for opening in self._openings:
            self.lst_openings.addItem(opening.describe())

    def _remove_opening(self) -> None:
        row = self.lst_openings.currentRow()
        if row < 0 or row >= len(self._openings):
            return
        del self._openings[row]
        self._list_openings()
        self._on_plane_edited()
        self.report.setPlainText(
            "Opening removed. Press 'Check these openings' to label with "
            "the rest.")
        self.preview_requested.emit(None)

    def _check_openings(self, *_args, report: bool = True) -> bool:
        if self._refuse_if_truncated(report):
            return False
        if not self._openings:
            return self._failed("There are no openings to build rings at.",
                                report)
        self._busy(True)
        try:
            labels = label_open_boundary(self._dataset, self._openings,
                                         self.orifice_options())
        except ValueError as exc:
            return self._failed(str(exc), report)
        finally:
            self._busy(False)
        self._accept_labels(labels, "")
        return True

    def _on_plane_edited(self, *_args) -> None:
        """An edited plane invalidates the last answer until re-checked."""
        self._labels = None
        self.btn_ok.setEnabled(False)

    def _detect_and_check(self) -> None:
        if self._detect():
            self._check()

    def _detect(self, *_args, report: bool = True) -> bool:
        self._busy(True)
        try:
            found = detect_base_plane(self._dataset, self.options())
        except ValueError as exc:
            return self._failed(str(exc), report)
        finally:
            self._busy(False)
        if found is None:
            return self._failed("No flat cut could be found on this mesh.",
                                report)
        self._set_plane(found)
        self._on_plane_edited()
        return True

    def _check(self, *_args, report: bool = True) -> bool:
        plane = self.plane()
        if plane is None:
            return self._failed(
                "The normal has no length, so it names no plane.", report)
        self._busy(True)
        try:
            labels, snapped = label_boundary_or_snap(
                self._dataset, plane, self.options())
        except ValueError as exc:
            return self._failed(str(exc), report)
        finally:
            self._busy(False)

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
        self._accept_labels(labels, prefix)
        return True

    def _accept_labels(self, labels: SurfaceLabels, prefix: str) -> None:
        self._labels = labels
        self.btn_ok.setEnabled(True)
        text = prefix + labels.details()
        if not labels.naming_confident:
            text += ("\n\nCheck which cavity is which before using this: the "
                     "LV and RV may be the wrong way round.")
        self.report.setPlainText(text)
        self.preview_requested.emit(labels)

    def _prepend(self, text: str) -> None:
        self.report.setPlainText(text + self.report.toPlainText())

    def _busy(self, on: bool) -> None:
        if on:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        else:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _failed(self, message: str, report: bool) -> bool:
        """Record a failure; say it unless a fallback is still to come."""
        self._last_error = message
        if report:
            self._fail(message)
        return False

    def _fail(self, message: str) -> None:
        self._labels = None
        self.btn_ok.setEnabled(False)
        self.report.setPlainText(message)
        # A refused plane leaves nothing to look at: a stale highlight from
        # the last good answer would say the opposite of the report.
        self.preview_requested.emit(None)


__all__ = ["SurfaceLabelsDialog"]
