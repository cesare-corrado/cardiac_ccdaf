"""
MeshInfoWidget
==============

Read-only side-panel widget displaying quick statistics about the
current working mesh: number of nodes/elements, axis-aligned bounding-
box size, and min/mean/max edge length. Refresh by calling
``update_info(mesh)`` whenever the working mesh changes.

For a volume, pass the tetrahedral grid as well. The counts and edge
lengths then describe the volume, because that is the working mesh — the
surface is a view of it — and a fourth row appears with the boundary
triangle count and how many tetrahedra are inverted. Reporting the
boundary's counts as if they were the mesh's would understate it by a
factor of four on a typical ventricle.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pyvista as pv
from PyQt5 import QtCore, QtGui, QtWidgets


_BOX_PADDING_PX = 6        # left/right padding in _BOX_STYLE
_BOX_BORDER_PX = 1         # border width in _BOX_STYLE

#: Widest string a box ever has to show. Values are formatted ``%.4g``,
#: whose longest form is a negative number in exponent notation
#: ("-1.234e-05", 10 characters); an eight-digit node or element count is
#: the same length. Sizing the boxes to fit it is what stops the numbers
#: being elided when the side panel is at its minimum width.
_WIDEST_VALUE = "-1.234e-05"

_BOX_STYLE = (
    "QLabel { background-color: white; color: black; "
    f"border: {_BOX_BORDER_PX}px solid #888; padding: 2px {_BOX_PADDING_PX}px; "
    "}"
)


def _box() -> QtWidgets.QLabel:
    lbl = QtWidgets.QLabel("—")
    lbl.setAlignment(QtCore.Qt.AlignCenter)
    lbl.setStyleSheet(_BOX_STYLE)
    # Monospace set on the widget rather than in the stylesheet: the
    # minimum width below is measured with QFontMetrics, which reads the
    # widget font, so a family set only in CSS would be measured against
    # the wrong glyphs. The style hints ride along so that a desktop
    # without the named fixed font still substitutes a fixed-pitch one.
    font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
    font.setStyleHint(QtGui.QFont.Monospace, QtGui.QFont.PreferMatch)
    font.setFixedPitch(True)
    lbl.setFont(font)
    lbl.setMinimumWidth(
        QtGui.QFontMetrics(font).horizontalAdvance(_WIDEST_VALUE)
        + 2 * (_BOX_PADDING_PX + _BOX_BORDER_PX)
    )
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
        self.box_nodes = _box()
        grid.addWidget(self.box_nodes, 0, 1)
        lbl_nelem = QtWidgets.QLabel("nElem")
        lbl_nelem.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        grid.addWidget(lbl_nelem, 0, 2)
        self.box_cells = _box()
        grid.addWidget(self.box_cells, 0, 3)

        # Row 1: Bounding box: <dx> <dy> <dz>
        grid.addWidget(QtWidgets.QLabel("Bounding box:"), 1, 0)
        self.box_bbox_x = _box()
        self.box_bbox_y = _box()
        self.box_bbox_z = _box()
        grid.addWidget(self.box_bbox_x, 1, 1)
        grid.addWidget(self.box_bbox_y, 1, 2)
        grid.addWidget(self.box_bbox_z, 1, 3)

        # Row 2: edge len(min,mean,max) <emin> <emean> <emax>
        # (row 3, volumes only, is built after it)
        grid.addWidget(QtWidgets.QLabel("edge len(min,mean,max)"), 2, 0)
        self.box_emin = _box()
        self.box_emean = _box()
        self.box_emax = _box()
        grid.addWidget(self.box_emin, 2, 1)
        grid.addWidget(self.box_emean, 2, 2)
        grid.addWidget(self.box_emax, 2, 3)

        # Row 3: volumes only — the boundary and the inverted count.
        self.lbl_boundary = QtWidgets.QLabel("boundary tri:")
        grid.addWidget(self.lbl_boundary, 3, 0)
        self.box_boundary = _box()
        grid.addWidget(self.box_boundary, 3, 1)
        self.lbl_inverted = QtWidgets.QLabel("inverted")
        self.lbl_inverted.setAlignment(QtCore.Qt.AlignRight
                                       | QtCore.Qt.AlignVCenter)
        grid.addWidget(self.lbl_inverted, 3, 2)
        self.box_inverted = _box()
        grid.addWidget(self.box_inverted, 3, 3)
        self._volume_row = (self.lbl_boundary, self.box_boundary,
                            self.lbl_inverted, self.box_inverted)
        for w in self._volume_row:
            w.setVisible(False)
        self.box_inverted.setToolTip(
            "Tetrahedra whose four nodes wind the other way. Harmless to "
            "view, but the remesher refuses them, so they are normalised "
            "before it sees them."
        )

        self.setToolTip(
            "Statistics of the current working mesh. Updates automatically "
            "after load, post-processing, clipping, or any other step that "
            "changes topology."
        )

    # -----------------------------------------------------------------
    def update_info(self, mesh: Optional[pv.PolyData], volume=None) -> None:
        """Show *mesh*'s statistics, or *volume*'s when there is one.

        *mesh* is still the surface — for a volume, its boundary — so the
        caller passes both and this decides which numbers are the mesh's.
        """
        boxes = (self.box_nodes, self.box_cells,
                 self.box_bbox_x, self.box_bbox_y, self.box_bbox_z,
                 self.box_emin, self.box_emean, self.box_emax,
                 self.box_boundary, self.box_inverted)
        if mesh is None or mesh.n_points == 0:
            for b in boxes:
                b.setText("—")
            for w in self._volume_row:
                w.setVisible(False)
            return

        for w in self._volume_row:
            w.setVisible(volume is not None)

        subject = volume if volume is not None else mesh
        self.box_nodes.setText(f"{subject.n_points}")
        self.box_cells.setText(f"{subject.n_cells}")
        if volume is not None:
            self.box_boundary.setText(f"{mesh.n_cells}")
            self.box_inverted.setText(f"{_inverted(volume)}")

        b = subject.bounds
        dx, dy, dz = b[1] - b[0], b[3] - b[2], b[5] - b[4]
        self.box_bbox_x.setText(f"{dx:.4g}")
        self.box_bbox_y.setText(f"{dy:.4g}")
        self.box_bbox_z.setText(f"{dz:.4g}")

        emin, emean, emax = (_tet_edge_stats(volume) if volume is not None
                             else _edge_stats(mesh))
        if emin is None:
            self.box_emin.setText("—")
            self.box_emean.setText("—")
            self.box_emax.setText("—")
        else:
            self.box_emin.setText(f"{emin:.4g}")
            self.box_emean.setText(f"{emean:.4g}")
            self.box_emax.setText(f"{emax:.4g}")


def _inverted(volume) -> int:
    """How many tetrahedra wind the wrong way; 0 if that cannot be told."""
    try:
        from ccdaf.core.volume_mesh import inverted_count
        return inverted_count(volume)
    except Exception:
        return 0


def _tet_edge_stats(volume):
    """Min / mean / max over the six edges of every tetrahedron.

    The volume's edges, not the boundary's: they are what a remesh target
    size is compared against, and a wall three elements thick has interior
    edges much shorter than the surface suggests.
    """
    try:
        from ccdaf.core.volume_mesh import tetrahedra
        tets = tetrahedra(volume)
        if tets.size == 0:
            return None, None, None
        p = np.asarray(volume.points)
        pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
        e = np.concatenate([
            np.linalg.norm(p[tets[:, j]] - p[tets[:, i]], axis=1)
            for i, j in pairs])
        if e.size == 0:
            return None, None, None
        return float(e.min()), float(e.mean()), float(e.max())
    except Exception:
        return None, None, None


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
