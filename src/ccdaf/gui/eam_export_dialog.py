"""
EAMExportDialog
===============
Pick where a loaded EAM mapping is written, under what name, and in which
format.

The suffix follows the format rather than the typed name, so a file cannot
end up claiming to be something it is not.
"""
from __future__ import annotations

import os
from typing import Optional

from PyQt5 import QtWidgets

from ccdaf.core.eam_export import EXPORT_BINARY, EXPORT_SUFFIX, EXPORT_VTK


_FORMAT_LABELS = (
    ("Binary (.pkl)", EXPORT_BINARY),
    ("VTK (.vtk)", EXPORT_VTK),
)


class EAMExportDialog(QtWidgets.QDialog):

    def __init__(self,
                 start_dir: str = "",
                 default_name: str = "",
                 parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export EAM")
        self.setMinimumWidth(460)

        layout = QtWidgets.QVBoxLayout(self)
        grid = QtWidgets.QGridLayout()

        grid.addWidget(QtWidgets.QLabel("directory"), 0, 0)
        self.txt_dir = QtWidgets.QLineEdit(start_dir)
        grid.addWidget(self.txt_dir, 0, 1)
        btn_browse = QtWidgets.QPushButton("Browse…")
        btn_browse.clicked.connect(self._browse)
        grid.addWidget(btn_browse, 0, 2)

        grid.addWidget(QtWidgets.QLabel("file name"), 1, 0)
        self.txt_name = QtWidgets.QLineEdit(default_name)
        self.txt_name.setToolTip(
            "Without a suffix — the format below decides that."
        )
        grid.addWidget(self.txt_name, 1, 1, 1, 2)

        grid.addWidget(QtWidgets.QLabel("format"), 2, 0)
        self.cmb_format = QtWidgets.QComboBox()
        for label, key in _FORMAT_LABELS:
            self.cmb_format.addItem(label, key)
        self.cmb_format.setToolTip(
            "Binary: a pickled {'surface', 'electrodes'} dictionary, as the "
            "reference Carto pipeline dumps.\n"
            "VTK: the surface with every field, for ParaView (electrodes are "
            "not included — they are separate geometry)."
        )
        self.cmb_format.currentIndexChanged.connect(self._refresh_preview)
        grid.addWidget(self.cmb_format, 2, 1, 1, 2)

        self.chk_ascii = QtWidgets.QCheckBox("ASCII")
        self.chk_ascii.setToolTip(
            "Write a text VTK rather than a binary one. Only applies to the "
            "VTK format — a pickle has no text form."
        )
        grid.addWidget(self.chk_ascii, 3, 1, 1, 2)
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)
        self.cmb_format.currentIndexChanged.connect(self._sync_ascii)
        self._sync_ascii()

        self.lbl_preview = QtWidgets.QLabel()
        self.lbl_preview.setWordWrap(True)
        layout.addWidget(self.lbl_preview)
        self.txt_dir.textChanged.connect(self._refresh_preview)
        self.txt_name.textChanged.connect(self._refresh_preview)
        self._refresh_preview()

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # -- internals ------------------------------------------------------
    def _browse(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Export directory", self.txt_dir.text())
        if d:
            self.txt_dir.setText(d)

    def _refresh_preview(self, *_args) -> None:
        self.lbl_preview.setText(f"writes: {self.selected_path()}")

    def _sync_ascii(self, *_args) -> None:
        """A pickle has no encoding to choose, so do not pretend otherwise."""
        self.chk_ascii.setEnabled(self.selected_format() == EXPORT_VTK)

    # -- queries --------------------------------------------------------
    def selected_format(self) -> str:
        return str(self.cmb_format.currentData())

    def selected_binary(self) -> bool:
        return not self.chk_ascii.isChecked()

    def selected_path(self) -> str:
        """Directory + name + the format's suffix."""
        name = self.txt_name.text().strip()
        if not name:
            return ""
        suffix = EXPORT_SUFFIX[self.selected_format()]
        stem = name[:-len(suffix)] if name.lower().endswith(suffix) else name
        return os.path.join(self.txt_dir.text().strip(), stem + suffix)


__all__ = ["EAMExportDialog"]
