"""
SegmentationWidget
==================

Side-panel widget exposing voxel-segmentation tools to the user:

* morphology operations (dilate, erode, opening, closing) with per-axis
  structuring-element radii
* hole filling
* manual paint mode with configurable brush (shape, radius, depth)
* surface (marching-cubes) parameters shared by the "Update 3D" and
  "Export to VTK" actions

The widget is intentionally decoupled from the rest of the GUI and
communicates exclusively through Qt signals.
"""

from __future__ import annotations

from typing import Optional, Tuple

from PyQt5 import QtCore, QtWidgets


class SegmentationWidget(QtWidgets.QGroupBox):
    """Side-panel controls for binary/multi-label segmentation."""

    morphology_requested = QtCore.pyqtSignal(str)            # 'dilate' | 'erode'
    fill_holes_requested = QtCore.pyqtSignal()
    convert_all_requested = QtCore.pyqtSignal(int, int)      # actual_label, new_label
    update_3d_requested = QtCore.pyqtSignal()
    paint_mode_changed = QtCore.pyqtSignal(bool)
    label_changed = QtCore.pyqtSignal(int)
    brush_changed = QtCore.pyqtSignal(str, int, str, int)    # shape, radius, depth_mode, depth
    undo_requested = QtCore.pyqtSignal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # --- Morphology -------------------------------------------------
        layout.addWidget(QtWidgets.QLabel("<b>Morphology</b>"))
        row = QtWidgets.QHBoxLayout()
        self.btn_dilate = QtWidgets.QPushButton("Binary Dilate")
        self.btn_erode = QtWidgets.QPushButton("Binary Erode")
        self.btn_dilate.clicked.connect(lambda: self.morphology_requested.emit("dilate"))
        self.btn_erode.clicked.connect(lambda: self.morphology_requested.emit("erode"))

        self.btn_moprh_opening = QtWidgets.QPushButton("Morph. opening")
        self.btn_moprh_closing = QtWidgets.QPushButton("Morph. closing")
        self.btn_moprh_opening.clicked.connect(lambda: self.morphology_requested.emit("morph_open"  ))
        self.btn_moprh_closing.clicked.connect(lambda: self.morphology_requested.emit("morph_close"))

        row.addWidget(self.btn_dilate)
        row.addWidget(self.btn_erode)
        layout.addLayout(row)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.btn_moprh_opening)
        row.addWidget(self.btn_moprh_closing)
        layout.addLayout(row)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Kernel radius (vx):"))
        self.spn_kernel_x = QtWidgets.QSpinBox()
        self.spn_kernel_y = QtWidgets.QSpinBox()
        self.spn_kernel_z = QtWidgets.QSpinBox()
        for sp, lbl in ((self.spn_kernel_x, "x"),
                        (self.spn_kernel_y, "y"),
                        (self.spn_kernel_z, "z")):
            sp.setRange(0, 50)
            sp.setValue(1)
            sp.setPrefix(f"{lbl}=")
            sp.setToolTip(
                f"Structuring-element radius along {lbl} (in voxels). "
                f"Set per-axis to handle anisotropic spacing."
            )
            row.addWidget(sp, 1)
        layout.addLayout(row)

        # --- Cleanup ---------------------------------------------------
        layout.addWidget(QtWidgets.QLabel("<b>Cleanup</b>"))
        self.btn_fill = QtWidgets.QPushButton("Fill Holes")
        self.btn_fill.clicked.connect(self.fill_holes_requested.emit)
        layout.addWidget(self.btn_fill)

        # --- Manual edit -----------------------------------------------
        layout.addWidget(QtWidgets.QLabel("<b>Manual Edit</b>"))

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Actual label:"))
        self.spn_actual_label = QtWidgets.QSpinBox()
        self.spn_actual_label.setRange(0, 8)
        self.spn_actual_label.setValue(1)
        self.spn_actual_label.setToolTip("Only voxels with this label will be repainted.")
        self.spn_actual_label.valueChanged.connect(self.label_changed.emit)
        row.addWidget(self.spn_actual_label, 1)
        layout.addLayout(row)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("New label:"))
        self.spn_new_label = QtWidgets.QSpinBox()
        self.spn_new_label.setRange(0, 8)
        self.spn_new_label.setValue(2)
        self.spn_new_label.setToolTip("Label to assign to the painted voxels.")
        row.addWidget(self.spn_new_label, 1)
        layout.addLayout(row)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Brush:"))
        self.rb_sphere = QtWidgets.QRadioButton("Sphere")
        self.rb_square = QtWidgets.QRadioButton("Square")
        self.rb_cyl = QtWidgets.QRadioButton("Cylinder")
        self.rb_sphere.setChecked(True)
        self._brush_group = QtWidgets.QButtonGroup(self)
        for rb in (self.rb_sphere, self.rb_square, self.rb_cyl):
            self._brush_group.addButton(rb)
            row.addWidget(rb)
            rb.toggled.connect(self._emit_brush)
        layout.addLayout(row)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Brush radius:"))
        self.spn_radius = QtWidgets.QSpinBox()
        self.spn_radius.setRange(1, 50)
        self.spn_radius.setValue(3)
        self.spn_radius.valueChanged.connect(self._emit_brush)
        row.addWidget(self.spn_radius, 1)
        layout.addLayout(row)

        row = QtWidgets.QHBoxLayout()
        self.rb_2d = QtWidgets.QRadioButton("2D (current plane)")
        self.rb_3d = QtWidgets.QRadioButton("3D (extended)")
        self.rb_2d.setChecked(True)
        self._depth_group = QtWidgets.QButtonGroup(self)
        for rb in (self.rb_2d, self.rb_3d):
            self._depth_group.addButton(rb)
            row.addWidget(rb)
            rb.toggled.connect(self._emit_brush)
        layout.addLayout(row)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("3D depth:"))
        self.spn_depth = QtWidgets.QSpinBox()
        self.spn_depth.setRange(1, 200)
        self.spn_depth.setValue(3)
        self.spn_depth.valueChanged.connect(self._emit_brush)
        row.addWidget(self.spn_depth, 1)
        layout.addLayout(row)

        self.btn_paint = QtWidgets.QPushButton("Activate paint mode")
        self.btn_paint.setCheckable(True)
        self.btn_paint.toggled.connect(self._on_paint_toggled)
        layout.addWidget(self.btn_paint)

        self.btn_convert_all = QtWidgets.QPushButton("Convert All")
        self.btn_convert_all.setToolTip(
            "Replace every voxel whose value equals 'Actual label' with 'New label'."
        )
        self.btn_convert_all.clicked.connect(self._on_convert_all_clicked)
        layout.addWidget(self.btn_convert_all)

        self.btn_undo = QtWidgets.QPushButton("Undo")
        self.btn_undo.setEnabled(False)
        self.btn_undo.clicked.connect(self.undo_requested.emit)
        layout.addWidget(self.btn_undo)

        # --- Surface params (shared by Update 3D + Export to VTK) ------
        layout.addWidget(QtWidgets.QLabel("<b>Image smoothing</b>"))

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("gaussian std (vx):"))
        self.spn_stdev_x = QtWidgets.QDoubleSpinBox()
        self.spn_stdev_y = QtWidgets.QDoubleSpinBox()
        self.spn_stdev_z = QtWidgets.QDoubleSpinBox()
        for sp, lbl in ((self.spn_stdev_x, "sx"),
                        (self.spn_stdev_y, "sy"),
                        (self.spn_stdev_z, "sz")):
            sp.setRange(0.0, 50.0)
            sp.setSingleStep(0.1)
            sp.setDecimals(3)
            sp.setValue(0.5)
            sp.setPrefix(f"{lbl}=")
            sp.setToolTip(
                f"Gaussian filter standard deviation along {lbl} (in voxels). "
                f"Set per-axis to handle anisotropic spacing."
            )
            row.addWidget(sp, 1)
        layout.addLayout(row)


        row = QtWidgets.QHBoxLayout()

        row.addWidget(QtWidgets.QLabel("radius factor (vx):"))
        self.spn_rfac_x = QtWidgets.QDoubleSpinBox()
        self.spn_rfac_y = QtWidgets.QDoubleSpinBox()
        self.spn_rfac_z = QtWidgets.QDoubleSpinBox()
        for sp, lbl in ((self.spn_rfac_x, "rx"),
                        (self.spn_rfac_y, "ry"),
                        (self.spn_rfac_z, "rz")):
            sp.setRange(0.0, 50.0)
            sp.setDecimals(3)
            sp.setSingleStep(0.1)
            sp.setValue(1.5)
            sp.setPrefix(f"{lbl}=")
            sp.setToolTip(
                f"Gaussian filter radius factor (kernel size) {lbl} (in voxels). "
                f"Set per-axis to handle anisotropic spacing."
            )
            row.addWidget(sp, 1)
        layout.addLayout(row)





        # Note: the "Update 3D" button itself lives as an overlay anchored
        # to the bottom-left of the 3D viewport (created by MainApp).

    # -- helpers --------------------------------------------------------
    def _on_convert_all_clicked(self) -> None:
        actual = self.actual_label()
        new = self.new_label()
        reply = QtWidgets.QMessageBox.question(
            self,
            "Confirm Convert All",
            f"Convert all voxels with label {actual} to label {new}?\n\nThis action can be undone.",
            QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Cancel,
        )
        if reply == QtWidgets.QMessageBox.Ok:
            self.convert_all_requested.emit(actual, new)

    def set_undo_enabled(self, enabled: bool) -> None:
        self.btn_undo.setEnabled(enabled)

    def _on_paint_toggled(self, on: bool) -> None:
        self.btn_paint.setText("Deactivate paint mode" if on else "Activate paint mode")
        self.paint_mode_changed.emit(on)

    def _emit_brush(self, *_) -> None:
        shape = self.brush_shape()
        depth_mode = "3d" if self.rb_3d.isChecked() else "2d"
        self.brush_changed.emit(shape, self.spn_radius.value(),
                                depth_mode, self.spn_depth.value())

    def brush_shape(self) -> str:
        if self.rb_sphere.isChecked():
            return "sphere"
        if self.rb_square.isChecked():
            return "square"
        return "cylinder"

    def is_3d_brush(self) -> bool:
        return self.rb_3d.isChecked()

    def actual_label(self) -> int:
        return int(self.spn_actual_label.value())

    def new_label(self) -> int:
        return int(self.spn_new_label.value())

    def label(self) -> int:
        return self.new_label()

    def brush_radius(self) -> int:
        return int(self.spn_radius.value())

    def brush_depth(self) -> int:
        return int(self.spn_depth.value())

    def kernel_radius(self) -> Tuple[int, int, int]:
        """Per-axis structuring element radius in voxels."""
        return (int(self.spn_kernel_x.value()),
                int(self.spn_kernel_y.value()),
                int(self.spn_kernel_z.value()))

    def gfilt_standard_deviation(self) -> Tuple[int, int, int]:
        """Per-axis Gaussian filter standard devioation in voxels."""
        return (float(self.spn_stdev_x.value()),
                float(self.spn_stdev_y.value()),
                float(self.spn_stdev_z.value()))


    def gfilt_radius_factor(self) -> Tuple[int, int, int]:
        """Per-axis Gaussian filter standard devioation in voxels."""
        return (float(self.spn_rfac_x.value()),
                float(self.spn_rfac_y.value()),
                float(self.spn_rfac_z.value()))


__all__ = ["SegmentationWidget"]
