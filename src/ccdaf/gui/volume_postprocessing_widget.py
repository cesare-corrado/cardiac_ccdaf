"""
VolumePostprocessingWidget
==========================
Side-panel widget for adapting a tetrahedral volume.

A separate panel from the surface one rather than a mode of it, because
the operations are not the same operations. The surface panel offers
decimate / refine / clean / fill holes / smooth, each rebuilding a
surface; a volume is adapted in one pass by MMG3D, and what the user
chooses is a size, not a sequence of stages. Sharing a panel would mean
half its controls greyed out whichever mesh was open.

The one control with a sharp edge is **Adapt the boundary too**. Left
unticked — the default — the boundary is frozen and comes back vertex for
vertex identical, so the anatomy is untouched and only the interior
changes. Ticked, the surface is re-approximated within the Hausdorff
tolerance: on a ventricle that moved the wall by half a millimetre on
average and removed a third of its boundary vertices. That is sometimes
exactly what is wanted and is never what someone wants by accident, so it
is off by default and the tooltip says what it costs.
"""
from __future__ import annotations

from typing import Optional

from PyQt5 import QtCore, QtWidgets

from ccdaf.core.volume_postprocessor import RemeshOptions


def _tip(*lines: str) -> str:
    """One tooltip per line, as rich text — see the clipping panel."""
    return "<br>".join(lines)


def _size_box(maximum: float = 1.0e6) -> QtWidgets.QDoubleSpinBox:
    box = QtWidgets.QDoubleSpinBox()
    box.setDecimals(3)
    box.setRange(0.0, maximum)
    box.setSingleStep(0.1)
    box.setSpecialValueText("auto")      # 0 reads as "leave it to MMG"
    box.setKeyboardTracking(False)
    box.setMinimumWidth(80)
    return box


class VolumePostprocessingWidget(QtWidgets.QGroupBox):

    remesh_requested = QtCore.pyqtSignal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QtWidgets.QLabel(
            "<i>Adapts the tetrahedra. Labels, fibres and point fields are "
            "carried onto the new elements.</i>"))

        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(6)

        # -- size: one target, or a band -------------------------------
        target_tip = _tip(
            "One uniform target edge length, in mesh units.",
            "<b>auto</b> (0) leaves the size to MMG, which keeps roughly "
            "what the mesh already has and only repairs quality.",
            "Mutually exclusive with the min/max band below.",
        )
        lbl_target = QtWidgets.QLabel("target edge")
        lbl_target.setToolTip(target_tip)
        grid.addWidget(lbl_target, 0, 0)
        self.spn_target = _size_box()
        self.spn_target.setToolTip(target_tip)
        self.spn_target.valueChanged.connect(self._sync_size_gate)
        grid.addWidget(self.spn_target, 0, 1)

        band_tip = _tip(
            "An edge-length band instead of one target: elements may vary "
            "between them.",
            "Mutually exclusive with the target above — MMG refuses both.",
        )
        for col, (label, name) in enumerate((("min edge", "spn_min"),
                                             ("max edge", "spn_max"))):
            lbl = QtWidgets.QLabel(label)
            lbl.setToolTip(band_tip)
            grid.addWidget(lbl, 1 + col, 0)
            box = _size_box()
            box.setToolTip(band_tip)
            box.valueChanged.connect(self._sync_size_gate)
            grid.addWidget(box, 1 + col, 1)
            setattr(self, name, box)

        # -- knobs that only mean something while the boundary adapts ---
        grad_tip = _tip(
            "Gradation: the largest ratio allowed between the lengths of "
            "two adjacent edges. It does not set the size; it limits how "
            "fast the size may change.",
            "Smaller grades more gently and costs elements — 1.05 against "
            "MMG's default 1.3 gave 2.6&times; as many on a ventricle; "
            "3.0 gave a quarter fewer.",
            "<b>auto</b> leaves MMG's own value (1.3).",
            "Only applies while <i>Adapt the boundary too</i> is ticked: "
            "with a frozen boundary the size does not vary, so there is "
            "nothing to grade and the value has no effect.",
        )
        self.lbl_gradation = QtWidgets.QLabel("gradation")
        self.lbl_gradation.setToolTip(grad_tip)
        grid.addWidget(self.lbl_gradation, 3, 0)
        self.spn_gradation = _size_box(maximum=10.0)
        self.spn_gradation.setToolTip(grad_tip)
        grid.addWidget(self.spn_gradation, 3, 1)

        self.lbl_hausd = QtWidgets.QLabel("boundary tol.")
        self.spn_hausdorff = _size_box(maximum=1.0e3)
        hausd_tip = _tip(
            "Hausdorff distance: how far the adapted boundary may stray "
            "from the original, in mesh units.",
            "Only applies while <i>Adapt the boundary too</i> is ticked — "
            "a frozen boundary does not move at all.",
        )
        self.lbl_hausd.setToolTip(hausd_tip)
        self.spn_hausdorff.setToolTip(hausd_tip)
        grid.addWidget(self.lbl_hausd, 4, 0)
        grid.addWidget(self.spn_hausdorff, 4, 1)
        layout.addLayout(grid)

        # -- the boundary ----------------------------------------------
        self.chk_adapt_boundary = QtWidgets.QCheckBox("Adapt the boundary too")
        self.chk_adapt_boundary.setToolTip(_tip(
            "Unticked (default): the boundary is <b>frozen</b> and comes "
            "back vertex for vertex identical. Only the interior changes, "
            "so the anatomy is untouched.",
            "Ticked: the surface is re-approximated within the tolerance "
            "above. On a ventricle this moved the wall by ~0.5&nbsp;mm on "
            "average and removed a third of its boundary vertices.",
            "Anything anchored to the old surface moves with it.",
        ))
        self.chk_adapt_boundary.toggled.connect(self._sync_boundary_gate)
        layout.addWidget(self.chk_adapt_boundary)

        self.btn_apply = QtWidgets.QPushButton("Remesh volume")
        self.btn_apply.setToolTip(
            "Adapt the tetrahedra to the sizes above. The mesh is replaced; "
            "File → Save data writes the result.")
        self.btn_apply.clicked.connect(self.remesh_requested.emit)
        layout.addWidget(self.btn_apply)

        self.lbl_status = QtWidgets.QLabel()
        self.lbl_status.setWordWrap(True)
        layout.addWidget(self.lbl_status)

        self._sync_size_gate()
        self._sync_boundary_gate()

    # -----------------------------------------------------------------
    def _sync_size_gate(self, *_args) -> None:
        """A target and a band cannot both be set — MMG refuses both.

        Whichever the user is filling in disables the other, so the
        impossible pairing cannot be typed rather than being reported
        after the fact.
        """
        has_target = self.spn_target.value() > 0.0
        has_band = self.spn_min.value() > 0.0 or self.spn_max.value() > 0.0
        self.spn_min.setEnabled(not has_target)
        self.spn_max.setEnabled(not has_target)
        self.spn_target.setEnabled(not has_band)

    def _sync_boundary_gate(self, *_args) -> None:
        """Two controls only mean something while the boundary adapts.

        The tolerance is obvious: a frozen boundary does not move, so how
        far it may move is moot. Gradation is less obvious and was worth
        measuring — it constrains how fast a *varying* size may change,
        and on this path the size only varies when MMG derives it from
        surface curvature, which it does only while adapting the
        boundary. With the boundary frozen, changing gradation gave
        byte-identical meshes. Leaving it live would offer a control that
        does nothing.
        """
        adapting = self.chk_adapt_boundary.isChecked()
        for widget in (self.lbl_hausd, self.spn_hausdorff,
                       self.lbl_gradation, self.spn_gradation):
            widget.setEnabled(adapting)

    # -----------------------------------------------------------------
    def options(self) -> RemeshOptions:
        # The boundary knobs are not sent while the boundary is frozen:
        # MMG ignores them there, and a value in the file that had no
        # effect on the result is a lie about what produced it.
        adapting = self.chk_adapt_boundary.isChecked()
        return RemeshOptions(
            target_edge=float(self.spn_target.value()),
            min_edge=float(self.spn_min.value()),
            max_edge=float(self.spn_max.value()),
            hausdorff=float(self.spn_hausdorff.value()) if adapting else 0.0,
            gradation=float(self.spn_gradation.value()) if adapting else 0.0,
            freeze_boundary=not adapting,
        )

    def set_status(self, text: str) -> None:
        self.lbl_status.setText(text)

    def set_busy(self, busy: bool) -> None:
        """Disable Apply while a remesh runs — it is not re-entrant."""
        self.btn_apply.setEnabled(not busy)
        self.btn_apply.setText("Remeshing…" if busy else "Remesh volume")


__all__ = ["VolumePostprocessingWidget"]
