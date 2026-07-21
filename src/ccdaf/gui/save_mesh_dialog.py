"""
SaveMeshDialog
==============
The Save-mesh file dialog, with a **Format** dropdown (a VTK surface or a
pickle bundle), a **Choose fields…** button that decides which of the
mesh's fields are written, and an **ASCII** box that decides how a VTK is
encoded.

Saving otherwise keeps the project format's own fields alone, which silently
drops an EAM mapping's measured Carto fields. The button makes that choice
visible at the moment of saving, rather than leaving the user to discover the
loss afterwards.

The pickle bundle carries the surface as the Carto reader's dict, plus the
seeds and elemTag a VTK does not — so a mesh saved that way reloads with its
seeds and tagging. ASCII is a VTK-only choice (a pickle is binary), so it is
greyed for the bundle.

Non-native so the widgets can be injected into the dialog's own layout; the
same approach as :mod:`ccdaf.gui.eam_load_dialog`.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

from PyQt5 import QtWidgets

from ccdaf.core.mesh_loader import DEFAULT_SAVE_FIELDS
from ccdaf.gui.field_select_dialog import FieldSelectDialog

FORMAT_VTK = "vtk"
FORMAT_PICKLE = "pkl"


def _hide_type_filter(dialog: QtWidgets.QFileDialog) -> None:
    """Hide the built-in 'Files of type' row of a non-native file dialog.

    A dialog that carries its own format control leaves this dropdown
    showing a single 'All files (*)' — a one-entry combo that only adds
    noise. Names are Qt's own for the non-native dialog.
    """
    combo = dialog.findChild(QtWidgets.QComboBox, "fileTypeCombo")
    if combo is not None:
        combo.hide()
    label = dialog.findChild(QtWidgets.QLabel, "fileTypeLabel")
    if label is not None:
        label.hide()


class SaveMeshDialog(QtWidgets.QFileDialog):

    def __init__(self,
                 point_fields: Sequence[str],
                 cell_fields: Sequence[str],
                 start_dir: str = "",
                 parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent, "Save mesh", start_dir,
                         "All files (*)")
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

        self.cmb_format = QtWidgets.QComboBox()
        self.cmb_format.addItem("VTK surface (*.vtk)", FORMAT_VTK)
        self.cmb_format.addItem("Pickle bundle (*.pkl)", FORMAT_PICKLE)
        self.cmb_format.setToolTip(
            "VTK writes the surface for the downstream pipeline. The pickle "
            "bundle also carries the seeds and tagging, so the mesh reloads "
            "with them."
        )
        self.cmb_format.currentIndexChanged.connect(self._on_format_changed)

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
            "Write a text VTK rather than a binary one. ASCII cannot store NaN, "
            "so no-data is written as the Carto sentinel (±10000) and folded "
            "back to NaN when ccdaf reloads it; in ParaView those points show as "
            "±10000, not blank. Untick for binary, which keeps NaN natively "
            "(blank no-data in ParaView) in a smaller file. A pickle bundle is "
            "binary regardless."
        )

        layout = self.layout()
        if isinstance(layout, QtWidgets.QGridLayout):
            row = layout.rowCount()
            layout.addWidget(QtWidgets.QLabel("Format:"), row, 0)
            layout.addWidget(self.cmb_format, row, 1)
            layout.addWidget(self.btn_fields, row + 1, 0)
            layout.addWidget(self.lbl_fields, row + 1, 1, 1,
                             layout.columnCount() - 1)
            layout.addWidget(self.chk_ascii, row + 2, 1)

        _hide_type_filter(self)

    # -- internals ------------------------------------------------------
    def _on_format_changed(self, *_args) -> None:
        is_pkl = self.selected_format() == FORMAT_PICKLE
        self.setDefaultSuffix("pkl" if is_pkl else "vtk")
        # ASCII is a VTK encoding choice; a pickle is binary.
        self.chk_ascii.setEnabled(not is_pkl)

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

    def selected_format(self) -> str:
        return str(self.cmb_format.currentData())

    def selected_path(self) -> str:
        files = self.selectedFiles()
        return files[0] if files else ""


__all__ = ["SaveMeshDialog", "FORMAT_VTK", "FORMAT_PICKLE"]
