"""
EAMLoadDialog
=============

A ``QFileDialog`` (non-native, so it keeps its own toolbar / up-arrow and
lets us install a custom model) for picking an electroanatomical-mapping
(EAM) source file. It navigates folders natively; an extra **EAM format**
drop-down content-filters the file area:

* ``None``           — show every file (and all folders).
* ``Carto-Study``    — show folders plus files matching the Carto *study*
                       names from :func:`extract_lists_for_loader`.
* ``Carto-mappings`` — show folders plus files matching the Carto *mapping*
                       names.

Matching rule: a file is shown when its name is one of the
``extract_lists_for_loader`` entries for the selected key. Those entries are
*file names*, so ``Carto-Study`` surfaces the study ``.xml`` and
``Carto-mappings`` the ``<map>.mesh`` files; the ``_car.txt``, the
``<map>_Points_Export.xml`` summaries and the per-point dumps are excluded.

Folders are always shown, so navigation works in every mode.
``extract_lists_for_loader`` is run per directory (cached per directory +
mode; the cache is cleared whenever the format changes) and re-evaluated as
the user browses.

The dialog performs no data loading. After ``exec_()`` returns
``QDialog.Accepted`` read the result via :meth:`selected_path`,
:meth:`selected_directory`, and :meth:`selected_format`.
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Set

from PyQt5 import QtCore, QtWidgets

from ccdaf.io.carto_functions import extract_lists_for_loader


FORMAT_NONE           = "None"
FORMAT_CARTO_STUDY    = "Carto-Study"
FORMAT_CARTO_MAPPINGS = "Carto-mappings"
_FORMATS = (FORMAT_NONE, FORMAT_CARTO_STUDY, FORMAT_CARTO_MAPPINGS)

# Maps a Carto format to the extract_lists_for_loader() dict key it filters by.
_FORMAT_KEY = {
    FORMAT_CARTO_STUDY:    "studies",
    FORMAT_CARTO_MAPPINGS: "mappings",
}


class _CartoFilterProxy(QtCore.QSortFilterProxyModel):
    """Row filter for the dialog's file view.

    Directories are always accepted (navigation stays intact). In ``None``
    mode every file is accepted; in a Carto mode a file is accepted only if
    its name is one of the ``extract_lists_for_loader`` entries (which are
    file names) for the active key in that file's own directory.
    """

    def __init__(self, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self._mode: str = FORMAT_NONE
        # Per-directory allowed-name sets, computed lazily; keyed by dir path.
        self._cache: Dict[str, Set[str]] = {}

    def set_mode(self, mode: str) -> None:
        if mode != self._mode:
            self._mode = mode
            self._cache.clear()      # allowed set depends on the mode
            self.invalidateFilter()

    # -- helpers --------------------------------------------------------
    def _allowed_for(self, dirpath: str) -> Set[str]:
        """Set of accepted file names for *dirpath* under the current mode."""
        if dirpath in self._cache:
            return self._cache[dirpath]
        key = _FORMAT_KEY.get(self._mode)
        allowed: Set[str] = set()
        if key is not None and dirpath and os.path.isdir(dirpath):
            try:
                allowed = set(extract_lists_for_loader(dirpath).get(key, []))
            except Exception:
                allowed = set()
        self._cache[dirpath] = allowed
        return allowed

    def accepts_file(self, dirpath: str, filename: str) -> bool:
        """Whether *filename* (in *dirpath*) passes the current filter."""
        if self._mode == FORMAT_NONE:
            return True
        return filename in self._allowed_for(dirpath)

    # -- QSortFilterProxyModel ------------------------------------------
    def filterAcceptsRow(self, source_row: int,
                         source_parent: QtCore.QModelIndex) -> bool:
        model = self.sourceModel()
        if model is None:
            return True
        index = model.index(source_row, 0, source_parent)
        if model.isDir(index):
            return True
        return self.accepts_file(model.filePath(source_parent),
                                 model.fileName(index))


class EAMLoadDialog(QtWidgets.QFileDialog):
    """Native-style file picker with an EAM-format-aware content filter."""

    def __init__(self, start_dir: str = "",
                 parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Load EAM")
        # Non-native so the proxy model and the injected combo take effect.
        self.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)
        self.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        self.setAcceptMode(QtWidgets.QFileDialog.AcceptOpen)
        self.setNameFilter("All files (*)")
        if start_dir:
            self.setDirectory(start_dir)

        self._proxy = _CartoFilterProxy(self)
        self.setProxyModel(self._proxy)

        # EAM-format drop-down, appended as a new row of the dialog's grid.
        self._cmb_format = QtWidgets.QComboBox()
        for fmt in _FORMATS:
            self._cmb_format.addItem(fmt, userData=fmt)
        self._cmb_format.currentIndexChanged.connect(self._on_format_changed)

        layout = self.layout()
        if isinstance(layout, QtWidgets.QGridLayout):
            row = layout.rowCount()
            layout.addWidget(QtWidgets.QLabel("EAM format:"), row, 0)
            layout.addWidget(self._cmb_format, row, 1)

        # The EAM-format combo above is the content filter here, so the
        # built-in 'Files of type' row (a single 'All files (*)') is
        # redundant — hide it. Names are Qt's own for the non-native dialog.
        combo = self.findChild(QtWidgets.QComboBox, "fileTypeCombo")
        if combo is not None:
            combo.hide()
        lbl = self.findChild(QtWidgets.QLabel, "fileTypeLabel")
        if lbl is not None:
            lbl.hide()

        # Re-evaluate the filter whenever the browsed directory changes.
        self.directoryEntered.connect(lambda _path: self._proxy.invalidateFilter())

        self._on_format_changed()

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------
    def selected_format(self) -> str:
        return str(self._cmb_format.currentData())

    def selected_directory(self) -> str:
        return self.directory().absolutePath()

    def selected_path(self) -> Optional[str]:
        """Absolute path of the chosen file, or ``None``."""
        files = self.selectedFiles()
        return files[0] if files else None

    # ------------------------------------------------------------------
    def _on_format_changed(self, *_args) -> None:
        self._proxy.set_mode(self.selected_format())


__all__ = [
    "EAMLoadDialog",
    "FORMAT_NONE",
    "FORMAT_CARTO_STUDY",
    "FORMAT_CARTO_MAPPINGS",
]
