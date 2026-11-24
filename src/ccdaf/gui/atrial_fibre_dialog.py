"""
atrial_fibre_dialog.py
======================
Actions > Atrial fibres: the choices for mapping the average atlas fibres
onto a left atrium that has universal atrial coordinates.

The window only gathers the choices and checks them; the mapping is
``ccdaf.core.uac.fibres.apply``.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt5 import QtWidgets

from ccdaf.core.uac import fibres

DEFAULT_THICKNESS_MM = 0.1     # the separation of the published bilayer models
LAYERS = (("endocardium", "endo"), ("epicardium", "epi"),
          ("bilayer (endocardium and epicardium)", "bilayer"))


class AtrialFibreDialog(QtWidgets.QDialog):
    def __init__(self, mesh, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Atrial fibres")
        self._mesh = mesh
        self._result = None
        v = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            "Map the average atlas fibres (Roney et al., 2021) onto this atrium "
            "through its UAC. The endocardium gives cell field fibre_endo, the "
            "epicardium fibre_epi, a bilayer both and a point field thickness.")
        intro.setWordWrap(True)
        v.addWidget(intro)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Layer:"))
        self.cmb_layer = QtWidgets.QComboBox()
        for text, key in LAYERS:
            self.cmb_layer.addItem(text, key)
        row.addWidget(self.cmb_layer, 1)
        v.addLayout(row)

        self.grp_thick = QtWidgets.QGroupBox("Bilayer thickness (distance between the layers)")
        g = QtWidgets.QGridLayout(self.grp_thick)
        self.rb_fixed = QtWidgets.QRadioButton("Fixed")
        self.rb_fixed.setChecked(True)
        self.spn_fixed = QtWidgets.QDoubleSpinBox()
        self.spn_fixed.setRange(0.0, 20.0)
        self.spn_fixed.setDecimals(2)
        self.spn_fixed.setSingleStep(0.05)
        self.spn_fixed.setSuffix(" mm")
        self.spn_fixed.setValue(DEFAULT_THICKNESS_MM)
        self.spn_fixed.setToolTip("0.1 mm is the separation of the published bilayer models.")
        g.addWidget(self.rb_fixed, 0, 0)
        g.addWidget(self.spn_fixed, 0, 1, 1, 2)

        self.rb_field = QtWidgets.QRadioButton("Point field")
        self.cmb_field = QtWidgets.QComboBox()
        for name in fibres.thickness_fields(mesh):
            self.cmb_field.addItem(name, name)
        if not self.cmb_field.count():
            self.rb_field.setEnabled(False)
            self.rb_field.setToolTip("The mesh has no point field with one value of 0 or "
                                     "more at every point.")
        g.addWidget(self.rb_field, 1, 0)
        g.addWidget(self.cmb_field, 1, 1, 1, 2)

        self.rb_file = QtWidgets.QRadioButton("File")
        self.txt_file = QtWidgets.QLineEdit()
        self.txt_file.setPlaceholderText("one value (mm) per point: .txt, .dat, .csv or .npy")
        btn = QtWidgets.QPushButton("Browse…")
        btn.clicked.connect(self._browse)
        g.addWidget(self.rb_file, 2, 0)
        g.addWidget(self.txt_file, 2, 1)
        g.addWidget(btn, 2, 2)
        v.addWidget(self.grp_thick)

        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok
                                        | QtWidgets.QDialogButtonBox.Cancel)
        bb.button(QtWidgets.QDialogButtonBox.Ok).setText("Map fibres")
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)
        self.cmb_layer.currentIndexChanged.connect(self._sync)
        for w in (self.rb_fixed, self.rb_field, self.rb_file):
            w.toggled.connect(self._sync)
        self._sync()

    # -- choices ----------------------------------------------------------
    def layer(self) -> str:
        return str(self.cmb_layer.currentData())

    def result(self):
        """``(layer, thickness)`` once accepted: thickness is ``None``
        (one layer), a number, or one value per point."""
        return self._result

    def _sync(self, *_a) -> None:
        bil = self.layer() == "bilayer"
        self.grp_thick.setEnabled(bil)
        self.spn_fixed.setEnabled(bil and self.rb_fixed.isChecked())
        self.cmb_field.setEnabled(bil and self.rb_field.isChecked())
        self.txt_file.setEnabled(bil and self.rb_file.isChecked())

    def _browse(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Thickness per point", "", "Values (*.txt *.dat *.csv *.npy);;All files (*)")
        if path:
            self.txt_file.setText(path)
            self.rb_file.setChecked(True)

    def thickness(self):
        """The bilayer thickness chosen, checked against the mesh. Raises
        ``ValueError`` saying what is wrong."""
        if self.rb_field.isChecked():
            name = self.cmb_field.currentData()
            return np.asarray(self._mesh.point_data[name], float).copy()
        if self.rb_file.isChecked():
            path = self.txt_file.text().strip()
            if not path:
                raise ValueError("Choose a thickness file.")
            return fibres.read_thickness(path, self._mesh.n_points)
        return float(self.spn_fixed.value())

    def _accept(self) -> None:
        th = None
        if self.layer() == "bilayer":
            try:
                th = self.thickness()
            except (ValueError, OSError) as exc:
                QtWidgets.QMessageBox.warning(self, "Atrial fibres", str(exc))
                return
        self._result = (self.layer(), th)
        self.accept()
