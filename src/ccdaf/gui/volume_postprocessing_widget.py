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

The panel holds three actions, because they are three different jobs,
and it lists them in the order they are meant to be run: **Clean
volume** repairs connectivity and moves no vertex, **Remesh volume**
changes element sizes, and **Improve quality** repairs the shape of what
is left. Each has its own button rather than a single "fix it" so that
what ran is what was asked for, and cleaning first is not a style
preference: a remesh carries every topological defect straight through,
and the same defects cost 11 seconds to repair on a 290,000-element mesh
against 9 minutes on its 14-million-element descendant.

The clean half has one control of the same kind: **Repair a non-manifold
boundary**, with its weld limit. Welding a pinhole shut adds a sliver of
material, so the limit is on the panel rather than buried — set it to
*report only* and every pinhole is counted instead of filled.

One repair is deliberately **not** on the panel: separating material
that only touches. It is exact and the mesh it makes is better
described, but its only measured effect on a real workflow was to break
*Actions → Label ventricular surfaces*, which relies on non-manifold
edges acting as accidental cuts in the boundary. A control whose effect
is to break a working pipeline does not belong on a panel, however right
the operation is, so it lives in ``CleanOptions.separate_touching`` for
scripts and tests. It comes back here when the labelling no longer needs
the accident.

The quality half has **Let wall nodes slide**. It is on, unlike the
remesh's boundary control, because 81% of the badly shaped elements on
the example ventricle touch the wall — freezing it would leave the panel
offering a repair that cannot reach most of what it is for. What makes
that safe is that the motion is tangential and bounded: measured there,
the wall ended up within 0.07 mm of itself and the surface area changed
in the fifth decimal place.
"""
from __future__ import annotations

from typing import Optional

from PyQt5 import QtCore, QtWidgets

from ccdaf.core.volume_clean import CleanOptions
from ccdaf.core.volume_postprocessor import RemeshOptions
from ccdaf.core.volume_quality import QualityOptions


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


def _section():
    """An empty section of the panel: its widget and its layout."""
    box = QtWidgets.QWidget()
    inner = QtWidgets.QVBoxLayout(box)
    inner.setContentsMargins(0, 0, 0, 0)
    return box, inner


class VolumePostprocessingWidget(QtWidgets.QGroupBox):

    remesh_requested = QtCore.pyqtSignal()
    clean_requested = QtCore.pyqtSignal()
    quality_requested = QtCore.pyqtSignal()
    wall_check_requested = QtCore.pyqtSignal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QtWidgets.QLabel(
            "<i>Clean, then remesh, then improve quality. Labels, fibres "
            "and point fields are carried onto the new elements.</i>"))

        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(6)


        # The three jobs, in the order they are meant to be run: repair
        # the connectivity, then change the sizes, then repair the shape
        # of what is left. Each is built into its own layout and the
        # layouts are added in that order, so the panel reads as the
        # workflow rather than as the order these were written in.
        clean_box, clean = _section()
        remesh_box, remesh = _section()
        quality_box, quality = _section()

        # -- clean: connectivity, not geometry -------------------------
        clean.addWidget(QtWidgets.QLabel(
            "<i>Removes stray elements, repairs a boundary that is not "
            "manifold, and reports the mesh's topology. Moves no "
            "vertex.</i>"))

        clean_grid = QtWidgets.QGridLayout()
        clean_grid.setHorizontalSpacing(6)

        weld_tip = _tip(
            "Absolute distance within which two points are welded into "
            "one, in mesh units.",
            "<b>exact</b> (0) merges only points that are already "
            "identical, so no vertex moves. A positive value welds "
            "near-duplicates; the first point of each group is the one "
            "that stays, so the result is still a point the mesh had.",
        )
        lbl_weld = QtWidgets.QLabel("merge points")
        lbl_weld.setToolTip(weld_tip)
        clean_grid.addWidget(lbl_weld, 0, 0)
        self.spn_merge_tol = _size_box(maximum=1.0e3)
        self.spn_merge_tol.setDecimals(4)
        self.spn_merge_tol.setSingleStep(0.001)
        self.spn_merge_tol.setSpecialValueText("exact")
        self.spn_merge_tol.setToolTip(weld_tip)
        clean_grid.addWidget(self.spn_merge_tol, 0, 1)

        frac_tip = _tip(
            "The smallest share of the elements a detached piece may hold "
            "and still be kept.",
            "A stray element is not cosmetic: a piece carrying no boundary "
            "condition makes a Laplace solve singular. The example "
            "ventricle has three, each a single tetrahedron held to the "
            "body by nodes alone.",
            "The largest piece is always kept, so this cannot empty the "
            "mesh — a separately meshed second body is safe.",
        )
        lbl_frac = QtWidgets.QLabel("min. piece")
        lbl_frac.setToolTip(frac_tip)
        clean_grid.addWidget(lbl_frac, 1, 0)
        self.spn_min_component = QtWidgets.QDoubleSpinBox()
        self.spn_min_component.setDecimals(4)
        self.spn_min_component.setRange(0.0, 1.0)
        self.spn_min_component.setSingleStep(0.01)
        self.spn_min_component.setValue(CleanOptions().min_component_fraction)
        self.spn_min_component.setKeyboardTracking(False)
        self.spn_min_component.setMinimumWidth(80)
        self.spn_min_component.setToolTip(frac_tip)
        clean_grid.addWidget(self.spn_min_component, 1, 1)
        weld_limit_tip = _tip(
            "The largest element a weld may add, as a multiple of the "
            "elements already at the contact.",
            "A pinhole is a passage through the wall that has closed to "
            "a point: welding it shut costs about one element. The limit "
            "is what keeps the repair from filling a passage that is "
            "genuinely open — that would need an element many times the "
            "local one, and is reported instead.",
            "<b>0</b> welds nothing: touching material is still "
            "separated, and every pinhole is reported.",
        )
        lbl_weld_limit = QtWidgets.QLabel("weld limit")
        lbl_weld_limit.setToolTip(weld_limit_tip)
        clean_grid.addWidget(lbl_weld_limit, 2, 0)
        self.spn_weld_limit = QtWidgets.QDoubleSpinBox()
        self.spn_weld_limit.setDecimals(2)
        self.spn_weld_limit.setRange(0.0, 100.0)
        self.spn_weld_limit.setSingleStep(0.5)
        self.spn_weld_limit.setSpecialValueText("report only")
        self.spn_weld_limit.setValue(CleanOptions().max_weld_volume)
        self.spn_weld_limit.setKeyboardTracking(False)
        self.spn_weld_limit.setMinimumWidth(80)
        self.spn_weld_limit.setToolTip(weld_limit_tip)
        clean_grid.addWidget(self.spn_weld_limit, 2, 1)
        clean.addLayout(clean_grid)


        self.chk_repair_boundary = QtWidgets.QCheckBox(
            "Repair a non-manifold boundary")
        self.chk_repair_boundary.setChecked(CleanOptions().repair_boundary)
        self.chk_repair_boundary.setToolTip(_tip(
            "Separate material that only touches, and weld shut a "
            "pinhole that has no thickness left.",
            "Both render as a hole in a wall that has none, and both "
            "let a surface label leak from epicardium to endocardium. "
            "Separating is exact: no element is removed and no "
            "coordinate moves. Welding adds a sliver of material where "
            "the wall already had none, bounded by the weld limit, and "
            "the report says how much.",
            "Until the boundary is manifold, the tunnel and cavity "
            "counts cannot be derived at all.",
        ))
        self.chk_repair_boundary.toggled.connect(self._sync_repair_gate)
        clean.addWidget(self.chk_repair_boundary)


        self.chk_fix_inverted = QtWidgets.QCheckBox("Reorient inverted elements")
        self.chk_fix_inverted.setChecked(CleanOptions().fix_inverted)
        self.chk_fix_inverted.setToolTip(_tip(
            "Swap two nodes of any tetrahedron whose signed volume is "
            "negative.",
            "The same four points and the same shape, so no geometry "
            "changes; the remesher refuses a mesh that still has them.",
        ))
        clean.addWidget(self.chk_fix_inverted)

        self.btn_clean = QtWidgets.QPushButton("Clean volume")
        self.btn_clean.setToolTip(
            "Drop stray and degenerate elements, make the boundary "
            "manifold, then report the topology. A tunnel that is still "
            "open is reported, never filled: closing one would invent "
            "material that was never imaged.")
        self.btn_clean.clicked.connect(self.clean_requested.emit)
        clean.addWidget(self.btn_clean)

        self.btn_wall = QtWidgets.QPushButton("Check the wall")
        self.btn_wall.setToolTip(_tip(
            "Measure the wall and report what is wrong with it. Changes "
            "nothing.",
            "It counts the <b>handles</b> of the boundary — the ways "
            "through the wall — and compares them with what the valve "
            "openings imply: a cavity opened twice must have one, "
            "because you can go in through one opening and out through "
            "the other. Anything beyond that is a hole.",
            "Measured on a <i>fully repaired copy</i>, because neither "
            "count is defined on a boundary that is not manifold, and "
            "the clean leaves one deliberately. Your mesh is untouched.",
        ))
        self.btn_wall.clicked.connect(self.wall_check_requested.emit)
        clean.addWidget(self.btn_wall)

        # -- size: one target, or a band -------------------------------
        rule = QtWidgets.QFrame()
        rule.setFrameShape(QtWidgets.QFrame.HLine)
        rule.setFrameShadow(QtWidgets.QFrame.Sunken)
        remesh.addWidget(rule)
        remesh.addWidget(QtWidgets.QLabel(
            "<i>Changes element sizes. Adapts the tetrahedra to the "
            "sizes below.</i>"))

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
            "<b>auto</b> uses a fifth of the element size, which is what "
            "the validated runs used. It does <i>not</i> fall back to "
            "MMG's own default of 0.01 mesh units — on a heart in "
            "millimetres that asks for the surface to within 10&nbsp;µm "
            "and does not finish.",
            "Cost rises steeply as it tightens: at a 1.5&nbsp;mm target, "
            "23&nbsp;s at 0.3, 27&nbsp;s at 0.1, 56&nbsp;s at 0.05.",
            "Only applies while <i>Adapt the boundary too</i> is ticked — "
            "a frozen boundary does not move at all.",
        )
        self.lbl_hausd.setToolTip(hausd_tip)
        self.spn_hausdorff.setToolTip(hausd_tip)
        grid.addWidget(self.lbl_hausd, 4, 0)
        grid.addWidget(self.spn_hausdorff, 4, 1)
        remesh.addLayout(grid)

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
        remesh.addWidget(self.chk_adapt_boundary)

        self.btn_apply = QtWidgets.QPushButton("Remesh volume")
        self.btn_apply.setToolTip(
            "Adapt the tetrahedra to the sizes above. The mesh is replaced; "
            "File → Save data writes the result.")
        self.btn_apply.clicked.connect(self.remesh_requested.emit)
        remesh.addWidget(self.btn_apply)

        # -- quality: shape, not size and not connectivity -------------
        rule2 = QtWidgets.QFrame()
        rule2.setFrameShape(QtWidgets.QFrame.HLine)
        rule2.setFrameShadow(QtWidgets.QFrame.Sunken)
        quality.addWidget(rule2)
        quality.addWidget(QtWidgets.QLabel(
            "<i>Repairs the shape of the worst elements only. Moves "
            "nodes; keeps the wall.</i>"))

        quality_grid = QtWidgets.QGridLayout()
        quality_grid.setHorizontalSpacing(6)
        thr_tip = _tip(
            "Elements whose shape measure is <b>above</b> this are the "
            "ones repaired. Lower is better in this measure: 0 is a "
            "regular tetrahedron, 1 is flat.",
            "0.8 is 'the bad ones': 889 of 290,508 on the example "
            "ventricle. Do not read 0.2 as strict — it marks 48% of an "
            "ordinary mesh, and a repair turned loose on a whole mesh "
            "shrinks it.",
        )
        lbl_thr = QtWidgets.QLabel("repair above")
        lbl_thr.setToolTip(thr_tip)
        quality_grid.addWidget(lbl_thr, 0, 0)
        self.spn_quality = QtWidgets.QDoubleSpinBox()
        self.spn_quality.setDecimals(2)
        self.spn_quality.setRange(0.05, 1.99)
        self.spn_quality.setSingleStep(0.05)
        self.spn_quality.setValue(QualityOptions().threshold)
        self.spn_quality.setKeyboardTracking(False)
        self.spn_quality.setMinimumWidth(80)
        self.spn_quality.setToolTip(thr_tip)
        quality_grid.addWidget(self.spn_quality, 0, 1)
        quality.addLayout(quality_grid)

        self.chk_slide_boundary = QtWidgets.QCheckBox(
            "Let wall nodes slide")
        self.chk_slide_boundary.setChecked(QualityOptions().slide_boundary)
        self.chk_slide_boundary.setToolTip(_tip(
            "A node on the wall may move, but only in its own tangent "
            "plane: the part of each step along the surface normal is "
            "removed.",
            "Needed because 81% of the badly shaped elements on the "
            "example ventricle touch the wall and 17% have all four "
            "nodes on it. Measured there, the wall ended up within "
            "0.07&nbsp;mm of where it was, 1.3&nbsp;µm at the 99th "
            "percentile, and the surface area changed by 0.00007%.",
            "Nodes on a sharp edge, such as the rim of a valve opening, "
            "never move whatever this is set to.",
            "Unticked, the wall is frozen vertex for vertex and only "
            "flips and interior nodes can help an element that touches "
            "it.",
        ))
        quality.addWidget(self.chk_slide_boundary)

        self.btn_quality = QtWidgets.QPushButton("Improve quality")
        self.btn_quality.setToolTip(
            "Flip, smooth and shift only where elements are worse than "
            "the threshold. The rest of the mesh is left alone, and a "
            "round that does not reduce the count is undone.")
        self.btn_quality.clicked.connect(self.quality_requested.emit)
        quality.addWidget(self.btn_quality)


        layout.addWidget(clean_box)
        layout.addWidget(remesh_box)
        layout.addWidget(quality_box)

        self.lbl_status = QtWidgets.QLabel()
        self.lbl_status.setWordWrap(True)
        layout.addWidget(self.lbl_status)

        self._sync_size_gate()
        self._sync_boundary_gate()
        self._sync_repair_gate()

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
    def _sync_repair_gate(self, *_args) -> None:
        """How much a weld may add is moot while nothing is being welded."""
        self.spn_weld_limit.setEnabled(
            self.chk_repair_boundary.isChecked())

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

    def clean_options(self) -> CleanOptions:
        # The weld limit is not sent while the repair is off: a value in
        # the file that had no effect on the result is a lie about what
        # produced it, which is the same reason the boundary knobs above
        # are dropped while the boundary is frozen.
        repairing = self.chk_repair_boundary.isChecked()
        return CleanOptions(
            merge_tol=float(self.spn_merge_tol.value()),
            min_component_fraction=float(self.spn_min_component.value()),
            fix_inverted=self.chk_fix_inverted.isChecked(),
            repair_boundary=repairing,
            max_weld_volume=(float(self.spn_weld_limit.value())
                             if repairing else 0.0),
        )

    def quality_options(self) -> QualityOptions:
        return QualityOptions(
            threshold=float(self.spn_quality.value()),
            slide_boundary=self.chk_slide_boundary.isChecked(),
        )

    def set_status(self, text: str) -> None:
        self.lbl_status.setText(text)

    def set_busy(self, busy: bool, *, cleaning: bool = False) -> None:
        """Disable both actions while one runs — neither is re-entrant.

        Both buttons go, not just the one pressed: they act on the same
        working volume, so starting the other mid-run would operate on a
        mesh that is about to be replaced.
        """
        self.btn_apply.setEnabled(not busy)
        self.btn_clean.setEnabled(not busy)
        self.btn_quality.setEnabled(not busy)
        self.btn_wall.setEnabled(not busy)
        self.btn_apply.setText(
            "Remeshing…" if busy and not cleaning else "Remesh volume")
        self.btn_clean.setText(
            "Cleaning…" if busy and cleaning else "Clean volume")


__all__ = ["VolumePostprocessingWidget"]
