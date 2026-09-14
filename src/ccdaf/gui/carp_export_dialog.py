"""
CarpExportDialog
================
**Export → Carp…**: where the three files go, the scale factor that takes the
mesh to micrometres, and which of the mesh's arrays ride in them.

Like the tissue-property dialog, this one holds no method of its own. It
assembles a :class:`~ccdaf.core.carp_export.CarpExportOptions`, checks it
against the mesh without writing anything, puts any warnings to the user, and
only then writes. The result is read back with :meth:`result`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt5 import QtCore, QtWidgets

from ccdaf.core.carp_export import (
    CarpExportOptions, CarpExportResult, DEFAULT_SCALE, DEFAULT_TAG,
    FIBRE_FIELDS, PLACEHOLDER_FIBRE, SHEET_FIELDS, SUFFIXES, direction_fields,
    region_fields, write_carp,
)
from ccdaf.core.tissue_property import TISSUE_TAG

#: What the "no array" entry of a drop-down carries.
NONE_ENTRY = None


class _ScaleSpin(QtWidgets.QDoubleSpinBox):
    """A scale factor shown as people write it: ``1000``, not ``1000.0000``.

    The precision stays available for a fractional factor; only the trailing
    zeros go.
    """

    def textFromValue(self, value: float) -> str:      # noqa: N802 (Qt's name)
        text = f"{value:.6f}".rstrip("0").rstrip(".")
        return text or "0"


class CarpExportDialog(QtWidgets.QDialog):

    def __init__(self, dataset, start_dir: str = "", default_name: str = "",
                 parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export to CARP")
        self.setSizeGripEnabled(True)
        self._dataset = dataset
        self._result: Optional[CarpExportResult] = None

        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QGridLayout()
        row = 0

        form.addWidget(QtWidgets.QLabel("directory"), row, 0)
        self.txt_dir = QtWidgets.QLineEdit(start_dir)
        self.txt_dir.setToolTip("Where the three files are written.")
        form.addWidget(self.txt_dir, row, 1)
        self.btn_browse = QtWidgets.QPushButton("Browse…")
        self.btn_browse.setToolTip("Choose the destination directory.")
        self.btn_browse.clicked.connect(self._browse)
        form.addWidget(self.btn_browse, row, 2)
        row += 1

        form.addWidget(QtWidgets.QLabel("file name"), row, 0)
        self.txt_name = QtWidgets.QLineEdit(default_name)
        self.txt_name.setToolTip(
            "Without a suffix: it is the prefix of all three files.")
        form.addWidget(self.txt_name, row, 1, 1, 2)
        row += 1

        form.addWidget(QtWidgets.QLabel("scale factor"), row, 0)
        scale_row = QtWidgets.QHBoxLayout()
        self.spn_scale = _ScaleSpin()
        self.spn_scale.setDecimals(6)
        self.spn_scale.setRange(1e-6, 1e9)
        self.spn_scale.setValue(DEFAULT_SCALE)
        self.spn_scale.setMaximumWidth(140)
        self.spn_scale.setToolTip(
            "Point coordinates are multiplied by this. CARP works in "
            "micrometres, so a mesh in millimetres needs 1000.")
        scale_row.addWidget(self.spn_scale)
        scale_row.addWidget(QtWidgets.QLabel("(CARP reads micrometres)"))
        scale_row.addStretch(1)
        form.addLayout(scale_row, row, 1, 1, 2)
        row += 1

        form.addWidget(QtWidgets.QLabel("region tag"), row, 0)
        self.cmb_region = QtWidgets.QComboBox()
        self.cmb_region.addItem(f"(none — every element {DEFAULT_TAG})", NONE_ENTRY)
        for name in region_fields(dataset):
            self.cmb_region.addItem(name, name)
        # Start on the material regions when the mesh carries them: writing
        # them into the element tag column is what this export is for.
        preferred = self.cmb_region.findData(TISSUE_TAG)
        if preferred >= 0:
            self.cmb_region.setCurrentIndex(preferred)
        self.cmb_region.setToolTip(
            "Cell array written as the .elem region column, which is what a "
            "simulation assigns conductivities and cell models by. Only whole-"
            f"numbered arrays are offered; with none, every element is written "
            f"as {DEFAULT_TAG}.")
        form.addWidget(self.cmb_region, row, 1, 1, 2)
        row += 1

        form.addWidget(QtWidgets.QLabel("fibres"), row, 0)
        self.cmb_fibre = QtWidgets.QComboBox()
        self.cmb_fibre.addItem(f"placeholder {PLACEHOLDER_FIBRE}", NONE_ENTRY)
        for name in direction_fields(dataset, FIBRE_FIELDS):
            self.cmb_fibre.addItem(name, name)
        self.cmb_fibre.setToolTip(
            "Direction written to .lon. With no field the placeholder is "
            "written for every element, which is only valid where "
            "conductivity is isotropic.")
        form.addWidget(self.cmb_fibre, row, 1, 1, 2)
        row += 1

        form.addWidget(QtWidgets.QLabel("sheets"), row, 0)
        self.cmb_sheet = QtWidgets.QComboBox()
        self.cmb_sheet.addItem("(none)", NONE_ENTRY)
        for name in direction_fields(dataset, SHEET_FIELDS):
            self.cmb_sheet.addItem(name, name)
        self.cmb_sheet.setToolTip(
            "A second direction per element, making the .lon header 2. Needs "
            "a fibre direction as well.")
        form.addWidget(self.cmb_sheet, row, 1, 1, 2)
        form.setColumnStretch(1, 1)
        layout.addLayout(form)

        self.lbl_preview = QtWidgets.QLabel()
        self.lbl_preview.setWordWrap(True)
        self.lbl_preview.setToolTip("The files this writes.")
        layout.addWidget(self.lbl_preview)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self.btn_export = buttons.button(QtWidgets.QDialogButtonBox.Ok)
        self.btn_export.setText("Export")
        self.btn_export.setToolTip("Write the three files.")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.txt_dir.textChanged.connect(self._refresh_preview)
        self.txt_name.textChanged.connect(self._refresh_preview)
        self.cmb_fibre.currentIndexChanged.connect(self._sync_sheets)
        self._sync_sheets()
        self._refresh_preview()

    # -- internals --------------------------------------------------------
    def _browse(self) -> None:
        chosen = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Export directory", self.txt_dir.text())
        if chosen:
            self.txt_dir.setText(chosen)

    def _sync_sheets(self, *_args) -> None:
        """A sheet direction means nothing without a fibre to go with it."""
        has_fibre = self.cmb_fibre.currentData() is not NONE_ENTRY
        self.cmb_sheet.setEnabled(has_fibre and self.cmb_sheet.count() > 1)
        if not has_fibre:
            self.cmb_sheet.setCurrentIndex(0)

    def _refresh_preview(self, *_args) -> None:
        name = self.txt_name.text().strip()
        if not name:
            self.lbl_preview.setText("writes: give the files a name")
            return
        # One line, not three paths: a long directory would otherwise wrap
        # into a block that pushes the whole dialog taller.
        prefix = Path(self.txt_dir.text().strip()) / name
        joined = "|".join(suffix.lstrip(".") for suffix in SUFFIXES)
        self.lbl_preview.setText(f"writes: {prefix}.[{joined}]")

    # -- queries ----------------------------------------------------------
    def options(self) -> CarpExportOptions:
        """The dialog's current inputs."""
        return CarpExportOptions(
            directory=self.txt_dir.text().strip(),
            name=self.txt_name.text().strip(),
            scale=self.spn_scale.value(),
            region_field=self.cmb_region.currentData(),
            fibre_field=self.cmb_fibre.currentData(),
            sheet_field=self.cmb_sheet.currentData() if self.cmb_sheet.isEnabled() else None,
        )

    def result(self) -> Optional[CarpExportResult]:
        """What the export wrote, or ``None`` if it was cancelled."""
        return self._result

    def _run(self, options: CarpExportOptions, *, dry_run: bool) -> CarpExportResult:
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            return write_carp(self._dataset, options, dry_run=dry_run)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def accept(self) -> None:  # type: ignore[override]
        options = self.options()
        try:
            options.validate()
            planned = self._run(options, dry_run=True)      # nothing written yet
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Cannot export", str(exc))
            return

        existing = options.existing()
        if existing:
            listed = "\n".join(str(path) for path in existing)
            reply = QtWidgets.QMessageBox.question(
                self, "Overwrite files",
                f"These already exist:\n\n{listed}\n\nOverwrite them?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No)
            if reply != QtWidgets.QMessageBox.Yes:
                return

        if planned.warnings:
            text = "\n\n".join(f"• {note}" for note in planned.warnings)
            reply = QtWidgets.QMessageBox.question(
                self, "Check before writing", f"{text}\n\nWrite the files anyway?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No)
            if reply != QtWidgets.QMessageBox.Yes:
                return

        try:
            self._result = self._run(options, dry_run=False)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Export failed", str(exc))
            return
        except OSError as exc:
            QtWidgets.QMessageBox.critical(self, "Export failed", str(exc))
            return
        super().accept()


__all__ = ["CarpExportDialog"]
