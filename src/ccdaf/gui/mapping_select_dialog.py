"""
MappingSelectDialog
===================
Modal pop-up that lists the mappings of a Carto study as radio buttons and
returns the one the user picks.

Single-selection for now (radio buttons). The layout is deliberately built
from a button group so switching to multi-select (check boxes) later is a
small change.
"""
from __future__ import annotations

from typing import List, Optional

from PyQt5 import QtWidgets


class MappingSelectDialog(QtWidgets.QDialog):

    def __init__(self, map_names: List[str],
                 parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select mapping")
        self.setMinimumWidth(340)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("Choose a mapping to load:"))

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QtWidgets.QWidget()
        inner_layout = QtWidgets.QVBoxLayout(inner)

        self._group = QtWidgets.QButtonGroup(self)
        self._radios: list[tuple[QtWidgets.QRadioButton, str]] = []
        for i, name in enumerate(map_names):
            rb = QtWidgets.QRadioButton(str(name))
            if i == 0:
                rb.setChecked(True)
            self._group.addButton(rb, i)
            inner_layout.addWidget(rb)
            self._radios.append((rb, str(name)))
        inner_layout.addStretch(1)
        scroll.setWidget(inner)
        layout.addWidget(scroll, 1)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_mapping(self) -> Optional[str]:
        for rb, name in self._radios:
            if rb.isChecked():
                return name
        return None


__all__ = ["MappingSelectDialog"]
