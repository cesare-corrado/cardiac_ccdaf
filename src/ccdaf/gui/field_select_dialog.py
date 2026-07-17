"""
FieldSelectDialog
=================
Modal pop-up listing every field on a mesh as a check box, and returning
the ticked ones.

Fields are labelled with the association they are stored under — point or
cell — because that decides what a value means: a point field is
interpolated across each triangle, a cell field is flat over it. ``elemTag``
is cell data; an EAM mapping's Carto fields are point data.

A field named in ``derived`` is not on the mesh at all: it is computed from
the geometry if ticked. ``Normals`` on a Carto mapping is the case that
motivates this — the mapping arrives as bare geometry, but the downstream
project format expects normals.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

from PyQt5 import QtWidgets


class FieldSelectDialog(QtWidgets.QDialog):

    def __init__(self,
                 point_fields: Sequence[str],
                 cell_fields: Sequence[str],
                 ticked: Optional[Iterable[str]] = None,
                 derived: Optional[Iterable[str]] = None,
                 parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Choose fields")
        self.setMinimumWidth(320)
        ticked = set(ticked or ())
        derived = set(derived or ())

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("Fields to write:"))

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QtWidgets.QWidget()
        inner_layout = QtWidgets.QVBoxLayout(inner)

        self._boxes: List[tuple] = []
        for names, kind in ((point_fields, "point"), (cell_fields, "cell")):
            if not names:
                continue
            head = QtWidgets.QLabel(f"<b>{kind} fields</b>")
            inner_layout.addWidget(head)
            for name in names:
                name = str(name)
                is_derived = name in derived
                box = QtWidgets.QCheckBox(
                    f"{name} (computed)" if is_derived else name)
                box.setChecked(name in ticked)
                box.setToolTip(
                    f"Not on the mesh — computed from the geometry and "
                    f"written as {kind} data."
                    if is_derived else f"Stored on the {kind}s of the mesh."
                )
                inner_layout.addWidget(box)
                self._boxes.append((box, name))
        inner_layout.addStretch(1)
        scroll.setWidget(inner)
        layout.addWidget(scroll, 1)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_fields(self) -> List[str]:
        return [name for box, name in self._boxes if box.isChecked()]


__all__ = ["FieldSelectDialog"]
