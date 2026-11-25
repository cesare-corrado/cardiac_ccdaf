"""
tag_map_dialog
==============
Actions > Tag names…: say what each ``elemTag`` number of the mesh is.

One row per tag present: the number with its legend colour, how many
elements carry it, and three editable boxes: the name, the chamber and the
role. Picking a name ccdaf knows fills in its chamber and role; both stay
editable (a ring kept as appendage tissue, for instance). The rows start
from the map already on the mesh, or from
:func:`ccdaf.core.tag_map.propose` when there is none. A blank name means
"no name": the legend then shows the number and actions ignore the tag.

**Load…** reads names from a JSON file (ccdaf's own, or a cohort's
``{name: tag}`` file) and **Save…** writes them, so the names given to the
first mesh of a cohort can be reused on the rest. See
:mod:`ccdaf.core.tag_map`.
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

from PyQt5 import QtCore, QtGui, QtWidgets

from ccdaf.core import tag_map

_NAME, _CHAMBER, _ROLE = 2, 3, 4


class TagMapDialog(QtWidgets.QDialog):
    """Edit what each tag is; :meth:`mapping` is the result."""

    def __init__(self, counts: Mapping[int, int],
                 current: Optional[Mapping[int, object]],
                 proposed: Optional[Mapping[int, object]] = None,
                 colors: Optional[Mapping[int, str]] = None,
                 start_dir: str = "",
                 parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Tag names")
        self._tags: Sequence[int] = sorted(int(t) for t in counts)
        self._proposed = tag_map.normalise(proposed)
        self._start_dir = start_dir
        start = tag_map.normalise(current) if current else dict(self._proposed)

        intro = QtWidgets.QLabel(
            "Say what each element tag of this mesh is. The legend shows the "
            "name; actions read the chamber and the role. Leave a name blank "
            "to show the number.")
        intro.setWordWrap(True)

        self.table = QtWidgets.QTableWidget(len(self._tags), 5, self)
        self.table.setHorizontalHeaderLabels(
            ["Tag", "Elements", "Name", "Chamber", "Role"])
        self.table.verticalHeader().setVisible(False)
        self._names: Dict[int, QtWidgets.QComboBox] = {}
        self._chambers: Dict[int, QtWidgets.QComboBox] = {}
        self._roles: Dict[int, QtWidgets.QComboBox] = {}
        for row, tag in enumerate(self._tags):
            item = QtWidgets.QTableWidgetItem(str(tag))
            if colors and tag in colors:
                pix = QtGui.QPixmap(12, 12)
                pix.fill(QtGui.QColor(colors[tag]))
                item.setIcon(QtGui.QIcon(pix))
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
            self.table.setItem(row, 0, item)
            count = QtWidgets.QTableWidgetItem(f"{int(counts[tag]):,}")
            count.setFlags(count.flags() & ~QtCore.Qt.ItemIsEditable)
            self.table.setItem(row, 1, count)

            name = QtWidgets.QComboBox(self)
            name.setEditable(True)
            name.addItem("")
            name.addItems(tag_map.MEANINGS)
            chamber = QtWidgets.QComboBox(self)
            chamber.addItems(tag_map.CHAMBERS)
            role = QtWidgets.QComboBox(self)
            role.addItems(tag_map.ROLES)
            self.table.setCellWidget(row, _NAME, name)
            self.table.setCellWidget(row, _CHAMBER, chamber)
            self.table.setCellWidget(row, _ROLE, role)
            self._names[tag] = name
            self._chambers[tag] = chamber
            self._roles[tag] = role
            self._set_row(tag, start.get(tag))
            name.currentTextChanged.connect(
                lambda text, t=tag: self._name_changed(t, text))
        self.table.resizeColumnsToContents()
        header = self.table.horizontalHeader()
        # Every column can be dragged wider; the name and chamber start
        # wide enough for the longest entries ("Pulmonary valve", "LV+LA").
        header.setSectionResizeMode(QtWidgets.QHeaderView.Interactive)
        header.setStretchLastSection(True)
        self.table.setColumnWidth(_NAME, max(self.table.columnWidth(_NAME), 190))
        self.table.setColumnWidth(_CHAMBER,
                                  max(self.table.columnWidth(_CHAMBER), 100))

        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self.reset_button = buttons.addButton(
            "Proposed names", QtWidgets.QDialogButtonBox.ResetRole)
        self.reset_button.setToolTip(
            "Fill in what ccdaf can tell by itself: the atrial scheme, or LV "
            "and RV on a labelled ventricular volume.")
        self.reset_button.clicked.connect(self._fill_proposed)
        self.load_button = buttons.addButton(
            "Load…", QtWidgets.QDialogButtonBox.ActionRole)
        self.load_button.setToolTip(
            "Read names from a JSON file: one saved here, or a cohort's "
            "{name: tag} file.")
        self.load_button.clicked.connect(self._ask_load)
        self.save_button = buttons.addButton(
            "Save…", QtWidgets.QDialogButtonBox.ActionRole)
        self.save_button.setToolTip(
            "Write these names to a JSON file, to reuse them on other meshes "
            "of the same cohort.")
        self.save_button.clicked.connect(self._ask_save)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addWidget(self.table)
        layout.addWidget(self.status)
        layout.addWidget(buttons)
        self.resize(720, min(640, 180 + 32 * len(self._tags)))

    # ------------------------------------------------------------------
    def _set_row(self, tag: int, value) -> None:
        e = tag_map.entry(value) if value is not None else tag_map.TagEntry("")
        name = self._names[tag]
        name.blockSignals(True)
        name.setCurrentText(e.name)
        name.blockSignals(False)
        self._set_combo(self._chambers[tag], e.chamber)
        self._set_combo(self._roles[tag], e.role)

    @staticmethod
    def _set_combo(box: QtWidgets.QComboBox, text: str) -> None:
        i = box.findText(text)
        box.setCurrentIndex(i if i >= 0 else 0)

    def _name_changed(self, tag: int, text: str) -> None:
        """A known name brings its chamber and role; a new one clears them."""
        e = tag_map.entry(text)
        self._set_combo(self._chambers[tag], e.chamber)
        self._set_combo(self._roles[tag], e.role)

    def _fill_proposed(self) -> None:
        for tag in self._tags:
            self._set_row(tag, self._proposed.get(tag))

    # ------------------------------------------------------------------
    def load_file(self, path: str) -> None:
        """Fill the rows from a JSON file; tags it does not list keep
        their names. Raises ``OSError`` or ``ValueError``."""
        loaded = tag_map.load_file(path)
        used = [t for t in self._tags if t in loaded]
        for tag in used:
            self._set_row(tag, loaded[tag])
        extra = len(set(loaded) - set(self._tags))
        msg = f"Loaded names for {len(used)} of {len(self._tags)} tags."
        if extra:
            msg += f" {extra} tags in the file are not in this mesh."
        self.status.setText(msg)

    def save_file(self, path: str) -> None:
        """Write the named rows to a JSON file. Raises ``OSError``."""
        tag_map.save_file(path, self.mapping())
        self.status.setText(f"Saved {len(self.mapping())} tag names.")

    def _ask_load(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load tag names", self._start_dir,
            "Tag names (*.json);;All files (*)")
        if not path:
            return
        try:
            self.load_file(path)
        except (OSError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(
                self, "Load tag names", f"Could not read {path}:\n{exc}")

    def _ask_save(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save tag names", self._start_dir, "Tag names (*.json)")
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        try:
            self.save_file(path)
        except OSError as exc:
            QtWidgets.QMessageBox.warning(
                self, "Save tag names", f"Could not write {path}:\n{exc}")

    # ------------------------------------------------------------------
    def mapping(self) -> tag_map.TagMap:
        """What was entered, rows with a blank name left out."""
        out: tag_map.TagMap = {}
        for tag in self._tags:
            name = self._names[tag].currentText().strip()
            if name:
                out[tag] = tag_map.TagEntry(
                    name, self._chambers[tag].currentText(),
                    self._roles[tag].currentText())
        return out


__all__ = ["TagMapDialog"]
