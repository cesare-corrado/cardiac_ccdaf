"""
MeshInfoWidget
==============

Read-only side-panel widget displaying quick statistics about the
current working mesh: number of nodes/elements, axis-aligned bounding-
box size, and min/mean/max edge length. Refresh by calling
``update_info(mesh)`` whenever the working mesh changes.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pyvista as pv
from PyQt5 import QtCore, QtWidgets


_BOX_STYLE = (
    "QLabel { background-color: white; color: black; "
    "border: 1px solid #888; padding: 2px 6px; "
    "font-family: monospace; }"
)


def _box(min_width: int = 50) -> QtWidgets.QLabel:
    lbl = QtWidgets.QLabel("—")
    lbl.setAlignment(QtCore.Qt.AlignCenter)
    lbl.setStyleSheet(_BOX_STYLE)
    lbl.setMinimumWidth(min_width)
    lbl.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
    return lbl


class MeshInfoWidget(QtWidgets.QWidget):

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)

        # Single grid so boxes share column geometry across the three
        # rows: column 0 holds the left-hand label, columns 1..3 hold
        # the value boxes. The leftmost box therefore sits in the same
        # horizontal position on every row; the bounding-box and edge-
        # length trios share the same three columns as well.
        grid = QtWidgets.QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(4)
        grid.setColumnStretch(0, 0)
        for c in (1, 2, 3):
            grid.setColumnStretch(c, 1)

        # Row 0: Npt: <box>  nElem <box>  (nElem label in col 2, box col 3)
        grid.addWidget(QtWidgets.QLabel("Npt:"), 0, 0)
        self.box_nodes = _box(50)
        grid.addWidget(self.box_nodes, 0, 1)
        lbl_nelem = QtWidgets.QLabel("nElem")
        lbl_nelem.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        grid.addWidget(lbl_nelem, 0, 2)
        self.box_cells = _box(50)
        grid.addWidget(self.box_cells, 0, 3)

        # Row 1: Bounding box: <dx> <dy> <dz>
        grid.addWidget(QtWidgets.QLabel("Bounding box:"), 1, 0)
        self.box_bbox_x = _box(50)
        self.box_bbox_y = _box(50)
        self.box_bbox_z = _box(50)
        grid.addWidget(self.box_bbox_x, 1, 1)
        grid.addWidget(self.box_bbox_y, 1, 2)
        grid.addWidget(self.box_bbox_z, 1, 3)

        # Row 2: edge len(min,mean,max) <emin> <emean> <emax>
        grid.addWidget(QtWidgets.QLabel("edge len(min,mean,max)"), 2, 0)
        self.box_emin = _box(50)
        self.box_emean = _box(50)
        self.box_emax = _box(50)
        grid.addWidget(self.box_emin, 2, 1)
        grid.addWidget(self.box_emean, 2, 2)
        grid.addWidget(self.box_emax, 2, 3)

        self.setToolTip(
            "Statistics of the current working mesh. Updates automatically "
            "after load, post-processing, clipping, or any other step that "
            "changes topology."
        )

    # -----------------------------------------------------------------
    def update_info(self, mesh: Optional[pv.PolyData]) -> None:
        boxes = (self.box_nodes, self.box_cells,
                 self.box_bbox_x, self.box_bbox_y, self.box_bbox_z,
                 self.box_emin, self.box_emean, self.box_emax)
        if mesh is None or mesh.n_points == 0:
            for b in boxes:
                b.setText("—")
            return

        self.box_nodes.setText(f"{mesh.n_points}")
        self.box_cells.setText(f"{mesh.n_cells}")

        b = mesh.bounds
        dx, dy, dz = b[1] - b[0], b[3] - b[2], b[5] - b[4]
        self.box_bbox_x.setText(f"{dx:.4g}")
        self.box_bbox_y.setText(f"{dy:.4g}")
        self.box_bbox_z.setText(f"{dz:.4g}")

        emin, emean, emax = _edge_stats(mesh)
        if emin is None:
            self.box_emin.setText("—")
            self.box_emean.setText("—")
            self.box_emax.setText("—")
        else:
            self.box_emin.setText(f"{emin:.4g}")
            self.box_emean.setText(f"{emean:.4g}")
            self.box_emax.setText(f"{emax:.4g}")


def _edge_stats(mesh: pv.PolyData):
    try:
        faces = np.asarray(mesh.faces).reshape(-1, 4)
        if faces.size == 0 or np.any(faces[:, 0] != 3):
            return None, None, None
        tri = faces[:, 1:]
        p = np.asarray(mesh.points)
        e0 = np.linalg.norm(p[tri[:, 1]] - p[tri[:, 0]], axis=1)
        e1 = np.linalg.norm(p[tri[:, 2]] - p[tri[:, 1]], axis=1)
        e2 = np.linalg.norm(p[tri[:, 0]] - p[tri[:, 2]], axis=1)
        e = np.concatenate([e0, e1, e2])
        if e.size == 0:
            return None, None, None
        return float(e.min()), float(e.mean()), float(e.max())
    except Exception:
        return None, None, None


__all__ = ["MeshInfoWidget"]
