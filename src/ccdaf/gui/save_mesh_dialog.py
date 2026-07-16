"""
SaveMeshDialog
==============
The Save-mesh file dialog, with a **Choose fields…** button that decides
which of the mesh's fields are written, and an **ASCII** box that decides
how they are encoded.

Saving otherwise keeps the project format's own fields alone, which silently
drops an EAM mapping's measured Carto fields. The button makes that choice
visible at the moment of saving, rather than leaving the user to discover the
loss afterwards.

Non-native so the widgets can be injected into the dialog's own layout; the
same approach as :mod:`ccdaf.gui.eam_load_dialog`.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

from PyQt5 import QtWidgets

from ccdaf.core.mesh_loader import DEFAULT_SAVE_FIELDS
from ccdaf.gui.field_select_dialog import FieldSelectDialog


class SaveMeshDialog(QtWidgets.QFileDialog):

    def __init__(self,
                 point_fields: Sequence[str],
                 cell_fields: Sequence[str],
                 start_dir: str = "",
                 parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent, "Save mesh", start_dir,
                         "VTK (*.vtk);;All files (*)")
        self.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)
        # The caller already confirms overwriting; leave that to it rather
        # than asking twice.
        self.setOption(QtWidgets.QFileDialog.DontConfirmOverwrite, True)
        self.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        self.setDefaultSuffix("vtk")

        self._point_fields = [str(f) for f in point_fields]
        self._cell_fields = [str(f) for f in cell_fields]
        # A mesh may not carry every default field — a Carto mapping arrives
        # as bare geometry with no Normals. Offer them anyway; saving computes
        # what is missing rather than quietly writing a file without it.
        self._derived: List[str] = [
            f for f in DEFAULT_SAVE_FIELDS
            if f not in self._point_fields and f not in self._cell_fields
        ]
        self._cell_fields += self._derived
        self._selected: List[str] = [
            f for f in self._cell_fields + self._point_fields
            if f in DEFAULT_SAVE_FIELDS
        ]

        self.btn_fields = QtWidgets.QPushButton("Choose fields…")
        self.btn_fields.setToolTip(
            "Pick which of the mesh's fields are written. Only the project "
            "format's own fields (elemTag, Normals) are kept by default; an "
            "EAM mapping's Carto fields are dropped unless ticked here."
        )
        self.btn_fields.clicked.connect(self._choose_fields)

        self.lbl_fields = QtWidgets.QLabel()
        self._refresh_label()

        self.chk_ascii = QtWidgets.QCheckBox("ASCII")
        self.chk_ascii.setChecked(True)
        self.chk_ascii.setToolTip(
            "Write a text VTK rather than a binary one. The project format is "
            "read as ASCII; untick for a smaller, faster file where whatever "
            "reads it back does not care."
        )

        layout = self.layout()
        if isinstance(layout, QtWidgets.QGridLayout):
            row = layout.rowCount()
            layout.addWidget(self.btn_fields, row, 0)
            layout.addWidget(self.lbl_fields, row, 1, 1, layout.columnCount() - 1)
            layout.addWidget(self.chk_ascii, row + 1, 1)

    # -- internals ------------------------------------------------------
    def _choose_fields(self) -> None:
        dlg = FieldSelectDialog(self._point_fields, self._cell_fields,
                                ticked=self._selected, derived=self._derived,
                                parent=self)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self._selected = dlg.selected_fields()
            self._refresh_label()

    def _refresh_label(self) -> None:
        if not self._selected:
            self.lbl_fields.setText("writing: no fields")
        else:
            self.lbl_fields.setText("writing: " + ", ".join(self._selected))

    # -- queries --------------------------------------------------------
    def selected_fields(self) -> List[str]:
        return list(self._selected)

    def selected_binary(self) -> bool:
        return not self.chk_ascii.isChecked()

    def selected_path(self) -> str:
        files = self.selectedFiles()
        return files[0] if files else ""


__all__ = ["SaveMeshDialog"]
