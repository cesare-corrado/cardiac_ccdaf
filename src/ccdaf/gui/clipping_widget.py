"""
ClippingWidget
==============
Side-panel widget for mesh clipping controls.

The panel is a *region* and a *mode*, then one set of buttons that mean
whatever those two say they mean. Region names an anatomical point —
the four pulmonary veins or the mitral valve — and mode picks how it is
cut: a contour, a sphere, or a plane — all three constrained to the
selected region's tag, so a clip never reaches into a neighbouring
region that merely happens to share the geometry. Start / Undo-reset
/ Apply therefore read the same in every combination, instead of one
button per pairing.

One pairing is impossible and is greyed out rather than left to fail:
**MV + contour**. The contour snake walks a tagged surface region, and
MV is deliberately excluded from region tagging, so there is nothing for
it to walk on.

Changing region or mode while a clip is in flight re-points that clip, so the
widget in the 3D view always matches what the panel says.

Below the buttons sits a mode-dependent readout: nothing for a contour,
centre + radius for a sphere, origin + normal for a plane. The boxes are
editable — typing a number drives the 3D widget exactly as dragging it
would — so a radius can be set to a value rather than eyeballed.

All user actions are exposed as signals. The host enables/disables
individual controls via the provided setter methods.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from PyQt5 import QtCore, QtWidgets


#: Mode identifiers, as carried by ``start_requested`` and the mode combo.
MODE_CONTOUR = "contour"
MODE_SPHERE = "sphere"
MODE_PLANE = "plane"

#: The region that has no tagged surface, and so no contour.
NO_CONTOUR_REGION = "MV"


def _tip(*lines: str) -> str:
    """One tooltip per line, as rich text.

    Qt only honours line breaks in a tooltip when it reads the string as rich
    text, and it decides that by looking for a tag — a bare newline in plain
    text is collapsed into the auto-wrapping. The wrapper is what makes the
    tag mandatory: a tooltip built from clauses ("what it is", "how to drive
    it", "what it needs") reads as a list rather than one long sentence.
    """
    return "<br>".join(lines)


def _coord_box() -> QtWidgets.QDoubleSpinBox:
    """A spin box wide enough for a mesh coordinate.

    The range is deliberately huge: mesh units are whatever the source data
    used, and a box that silently clamps a pasted coordinate would move the
    clip without saying so."""
    box = QtWidgets.QDoubleSpinBox()
    box.setDecimals(3)
    box.setRange(-1.0e9, 1.0e9)
    box.setSingleStep(1.0)
    box.setKeyboardTracking(False)   # emit on commit, not per keystroke
    box.setMinimumWidth(70)
    return box


class ClippingWidget(QtWidgets.QGroupBox):

    # region name, mode id
    start_requested       = QtCore.pyqtSignal(str, str)
    undo_reset_requested  = QtCore.pyqtSignal()
    apply_requested       = QtCore.pyqtSignal()
    revert_requested      = QtCore.pyqtSignal()
    clipping_toggled      = QtCore.pyqtSignal(bool)
    # Region or mode changed — the host re-points a clip already in flight.
    selection_changed     = QtCore.pyqtSignal(str, str)
    # cx, cy, cz, radius
    sphere_pose_edited    = QtCore.pyqtSignal(float, float, float, float)
    # ox, oy, oz, nx, ny, nz
    plane_pose_edited     = QtCore.pyqtSignal(
        float, float, float, float, float, float)

    def __init__(self,
                 region_names: List[str],
                 parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        # Set by set_enabled_after_accept / reset_state; the start button
        # needs both this and the activation checkbox.
        self._accepted = False
        # Guards the widget→panel→widget round trip while a pose is being
        # written in from the 3D widget.
        self._syncing = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.chk_active = QtWidgets.QCheckBox("Clipping active")
        self.chk_active.setChecked(False)
        self.chk_active.setToolTip(_tip(
            "While unchecked, clipping is dormant and the X key belongs to "
            "manual correction. Both tools want X — this decides the owner.",
            "Ticking it accepts the tagging first if that has not happened "
            "yet, so clipping never waits on a step you cannot see.",
        ))
        self.chk_active.toggled.connect(self._on_active_toggled)
        layout.addWidget(self.chk_active)

        # --- Row 1: region ---------------------------------------------
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Region:"))
        self.cmb_region = QtWidgets.QComboBox()
        self.cmb_region.setToolTip(_tip(
            "The anatomical point this clip works on.",
            "Every mode is confined to that region: a contour follows its "
            "tag, a sphere or plane starts from its seed and removes only "
            "triangles carrying its label (the body label, for MV).",
        ))
        for name in region_names:
            self.cmb_region.addItem(name, userData=name)
        self.cmb_region.currentIndexChanged.connect(self._on_selection_changed)
        row.addWidget(self.cmb_region, 1)
        layout.addLayout(row)

        # --- Row 2: mode -----------------------------------------------
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Mode:"))
        self.cmb_mode = QtWidgets.QComboBox()
        self.cmb_mode.setToolTip(_tip(
            "<b>Contour</b> — drop points with X; a geodesic snake follows "
            "the region's tag, and the loop's inside is clipped.",
            "<b>Sphere</b> — triangles of the selected region whose centre "
            "falls inside the sphere are clipped; other regions inside it "
            "are left alone.",
            "<b>Plane</b> — the selected region's triangles on the seed's "
            "side of the plane are clipped.",
        ))
        self.cmb_mode.addItem("Contour", userData=MODE_CONTOUR)
        self.cmb_mode.addItem("Sphere", userData=MODE_SPHERE)
        self.cmb_mode.addItem("Plane", userData=MODE_PLANE)
        self.cmb_mode.currentIndexChanged.connect(self._on_mode_changed)
        self.cmb_mode.currentIndexChanged.connect(self._on_selection_changed)
        row.addWidget(self.cmb_mode, 1)
        layout.addLayout(row)

        # --- Row 3: the three verbs ------------------------------------
        row = QtWidgets.QHBoxLayout()
        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_start.setToolTip(_tip(
            "Begin a clip on the selected region in the selected mode.",
            "<b>Contour</b> — press X on the region to drop points.",
            "<b>Sphere / plane</b> — the widget appears where it was last "
            "left for this region, or at the region's seed the first time.",
        ))
        self.btn_start.clicked.connect(
            lambda: self.start_requested.emit(
                self.selected_region(), self.selected_mode())
        )
        self.btn_start.setEnabled(False)
        row.addWidget(self.btn_start)

        self.btn_undo_reset = QtWidgets.QPushButton("Undo / reset")
        self.btn_undo_reset.setToolTip(_tip(
            "<b>Contour</b> — remove the most recently placed point and "
            "redraw the snake through the rest.",
            "<b>Sphere / plane</b> — put the widget back at the region's "
            "seed default, forgetting where it had been dragged to.",
        ))
        self.btn_undo_reset.clicked.connect(self.undo_reset_requested.emit)
        self.btn_undo_reset.setEnabled(False)
        row.addWidget(self.btn_undo_reset)

        self.btn_apply = QtWidgets.QPushButton("Apply clip")
        self.btn_apply.setToolTip(_tip(
            "<b>Contour</b> — close the contour into a loop and clip off the "
            "cuff inside it.",
            "<b>Sphere / plane</b> — commit the pending clip to the mesh.",
        ))
        self.btn_apply.clicked.connect(self.apply_requested.emit)
        self.btn_apply.setEnabled(False)
        row.addWidget(self.btn_apply)
        layout.addLayout(row)

        # --- Row 4: mesh-level revert ----------------------------------
        self.btn_revert = QtWidgets.QPushButton("Revert clip")
        self.btn_revert.setToolTip(_tip(
            "Discard the clip in progress, or undo the last applied one, "
            "restoring the previous mesh.",
            "Applied clips revert one at a time, newest first.",
            "The sphere or plane you were using is remembered — Start brings "
            "it back where it was.",
        ))
        self.btn_revert.clicked.connect(self.revert_requested.emit)
        self.btn_revert.setEnabled(False)
        layout.addWidget(self.btn_revert)

        # --- Mode-dependent readout ------------------------------------
        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(self._build_blank_page())     # MODE_CONTOUR
        self.stack.addWidget(self._build_sphere_page())    # MODE_SPHERE
        self.stack.addWidget(self._build_plane_page())     # MODE_PLANE
        layout.addWidget(self.stack)

        self._sync_mode_gate()
        self._on_mode_changed()

    # ------------------------------------------------------------------
    # Readout pages
    # ------------------------------------------------------------------
    def _build_blank_page(self) -> QtWidgets.QWidget:
        """The contour has no geometry to show — an empty, zero-height page.

        A page rather than a hidden stack keeps the panel's height steady as
        the modes are cycled."""
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        return page

    def _build_sphere_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Centre:"))
        self.sph_cx = _coord_box()
        self.sph_cy = _coord_box()
        self.sph_cz = _coord_box()
        for box in (self.sph_cx, self.sph_cy, self.sph_cz):
            box.setToolTip(
                "Sphere centre, in mesh units. Editable — the sphere follows."
            )
            box.valueChanged.connect(self._emit_sphere_pose)
            row.addWidget(box, 1)
        lay.addLayout(row)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Radius:"))
        self.sph_r = _coord_box()
        # A sphere of non-positive radius clips nothing; the floor keeps the
        # box from offering a state the clip cannot act on.
        self.sph_r.setRange(1.0e-6, 1.0e9)
        self.sph_r.setToolTip(
            "Sphere radius, in mesh units. Editable — the sphere follows."
        )
        self.sph_r.valueChanged.connect(self._emit_sphere_pose)
        row.addWidget(self.sph_r, 1)
        self.lbl_diam = QtWidgets.QLabel("⌀ —")
        self.lbl_diam.setToolTip(
            "Diameter — the sphere across, for comparing against a measured "
            "annulus or ostium."
        )
        row.addWidget(self.lbl_diam)
        lay.addLayout(row)
        return page

    def _build_plane_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Centre:"))
        self.pln_ox = _coord_box()
        self.pln_oy = _coord_box()
        self.pln_oz = _coord_box()
        for box in (self.pln_ox, self.pln_oy, self.pln_oz):
            box.setToolTip(
                "A point the plane passes through, in mesh units. Editable — "
                "the plane follows."
            )
            box.valueChanged.connect(self._emit_plane_pose)
            row.addWidget(box, 1)
        lay.addLayout(row)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Normal:"))
        self.pln_nx = _coord_box()
        self.pln_ny = _coord_box()
        self.pln_nz = _coord_box()
        for box in (self.pln_nx, self.pln_ny, self.pln_nz):
            box.setRange(-1.0, 1.0)
            box.setSingleStep(0.05)
            box.setToolTip(
                "The plane normal's components. Editable; the direction is "
                "what counts, so the length is normalised for you."
            )
            box.valueChanged.connect(self._emit_plane_pose)
            row.addWidget(box, 1)
        lay.addLayout(row)
        return page

    # ------------------------------------------------------------------
    # Pose in / out
    # ------------------------------------------------------------------
    def set_sphere_pose(self, pose: Dict[str, float]) -> None:
        """Show a sphere geometry without echoing it back as an edit."""
        self._syncing = True
        try:
            self.sph_cx.setValue(float(pose["cx"]))
            self.sph_cy.setValue(float(pose["cy"]))
            self.sph_cz.setValue(float(pose["cz"]))
            self.sph_r.setValue(float(pose["radius"]))
        finally:
            self._syncing = False
        self.lbl_diam.setText(f"⌀ {2.0 * float(pose['radius']):.3f}")

    def set_plane_pose(self, pose: Dict[str, float]) -> None:
        """Show a plane geometry without echoing it back as an edit."""
        self._syncing = True
        try:
            self.pln_ox.setValue(float(pose["ox"]))
            self.pln_oy.setValue(float(pose["oy"]))
            self.pln_oz.setValue(float(pose["oz"]))
            self.pln_nx.setValue(float(pose["nx"]))
            self.pln_ny.setValue(float(pose["ny"]))
            self.pln_nz.setValue(float(pose["nz"]))
        finally:
            self._syncing = False

    def sphere_pose(self) -> Dict[str, float]:
        return {"cx": self.sph_cx.value(), "cy": self.sph_cy.value(),
                "cz": self.sph_cz.value(), "radius": self.sph_r.value()}

    def plane_pose(self) -> Dict[str, float]:
        return {"ox": self.pln_ox.value(), "oy": self.pln_oy.value(),
                "oz": self.pln_oz.value(), "nx": self.pln_nx.value(),
                "ny": self.pln_ny.value(), "nz": self.pln_nz.value()}

    def _emit_sphere_pose(self) -> None:
        if self._syncing:
            return
        self.lbl_diam.setText(f"⌀ {2.0 * self.sph_r.value():.3f}")
        self.sphere_pose_edited.emit(
            self.sph_cx.value(), self.sph_cy.value(),
            self.sph_cz.value(), self.sph_r.value())

    def _emit_plane_pose(self) -> None:
        if self._syncing:
            return
        self.plane_pose_edited.emit(
            self.pln_ox.value(), self.pln_oy.value(), self.pln_oz.value(),
            self.pln_nx.value(), self.pln_ny.value(), self.pln_nz.value())

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------
    def selected_region(self) -> str:
        return str(self.cmb_region.currentData())

    def selected_mode(self) -> str:
        return str(self.cmb_mode.currentData())

    def is_clipping_enabled(self) -> bool:
        return bool(self.chk_active.isChecked())

    def is_accepted(self) -> bool:
        """Whether the panel has been told the tagging is accepted.

        The host asks before honouring a tick of the activation checkbox: an
        unaccepted tagging is something it can put right, not a reason to
        leave the panel dead."""
        return self._accepted

    def set_active_checked(self, checked: bool) -> None:
        """Set the activation checkbox without re-entering the toggle handler.

        Used to withdraw a tick the host could not honour. Going through the
        signal would ask the host again about the answer it just gave."""
        self.chk_active.blockSignals(True)
        self.chk_active.setChecked(checked)
        self.chk_active.blockSignals(False)
        self._sync_start_button()

    # ------------------------------------------------------------------
    # Enablement
    # ------------------------------------------------------------------
    def set_undo_reset_enabled(self, enabled: bool) -> None:
        self.btn_undo_reset.setEnabled(enabled)

    def set_apply_enabled(self, enabled: bool) -> None:
        self.btn_apply.setEnabled(enabled)

    def set_revert_enabled(self, enabled: bool) -> None:
        self.btn_revert.setEnabled(enabled)

    def set_enabled_after_accept(self) -> None:
        """Allow the clipping controls once tagging has been accepted.

        Start still waits for the activation checkbox."""
        self._accepted = True
        self._sync_start_button()

    def clear_in_flight(self) -> None:
        """Put the controls back to "no clip in progress".

        Used when another tool takes the shared picker away mid-clip: the
        clip is abandoned, so the buttons that only mean something during one
        must go with it. Activation and the accepted-tagging state are the
        user's, and survive — only the in-flight controls are cleared."""
        self.btn_undo_reset.setEnabled(False)
        self.btn_apply.setEnabled(False)

    def reset_state(self) -> None:
        """Disable all controls — used by teardown after plotter rebuild.

        The activation checkbox is the user's choice and survives."""
        self._accepted = False
        self.btn_start.setEnabled(False)
        self.btn_undo_reset.setEnabled(False)
        self.btn_apply.setEnabled(False)
        self.btn_revert.setEnabled(False)

    # ------------------------------------------------------------------
    # Internal gating
    # ------------------------------------------------------------------
    def _sync_start_button(self) -> None:
        self.btn_start.setEnabled(self._accepted and self.is_clipping_enabled())

    def _sync_mode_gate(self) -> None:
        """Grey out MV + contour, in whichever order the user gets there.

        The snake walks a tagged region and MV carries no tag, so the pairing
        has nothing to run on. Disabling the item says that up front; leaving
        it selectable would only trade the grey for an error box. Whichever
        combo the user changes last, the *other* one is the one adjusted, so
        neither selection is silently overruled."""
        region_is_mv = self.selected_region() == NO_CONTOUR_REGION
        mode_is_contour = self.selected_mode() == MODE_CONTOUR

        no_contour_tip = (
            f"{NO_CONTOUR_REGION} carries no tagged surface region, so the "
            f"contour snake has nothing to travel on. Use a sphere or a plane."
        )

        # Mode combo: forbid contour while MV is the region.
        item = self.cmb_mode.model().item(
            self.cmb_mode.findData(MODE_CONTOUR))
        if item is not None:
            item.setEnabled(not region_is_mv)
            item.setToolTip(no_contour_tip if region_is_mv else "")

        # Region combo: forbid MV while contour is the mode.
        idx = self.cmb_region.findData(NO_CONTOUR_REGION)
        if idx >= 0:
            item = self.cmb_region.model().item(idx)
            if item is not None:
                item.setEnabled(not mode_is_contour)
                item.setToolTip(no_contour_tip if mode_is_contour else "")

    def _on_selection_changed(self) -> None:
        """Announce the new region/mode pair.

        A clip already in flight is *about* this pair, so the host re-points it
        rather than leaving a sphere on screen that the panel no longer
        describes."""
        self._sync_mode_gate()
        self.selection_changed.emit(self.selected_region(), self.selected_mode())

    def _on_mode_changed(self) -> None:
        mode = self.selected_mode()
        self.stack.setCurrentIndex(
            {MODE_CONTOUR: 0, MODE_SPHERE: 1, MODE_PLANE: 2}.get(mode, 0))
        self._sync_mode_gate()

    def _on_active_toggled(self, on: bool) -> None:
        self._sync_start_button()
        if not on:
            # Whatever was mid-flight is being abandoned by the host.
            self.btn_undo_reset.setEnabled(False)
            self.btn_apply.setEnabled(False)
        self.clipping_toggled.emit(on)


__all__ = [
    "ClippingWidget",
    "MODE_CONTOUR",
    "MODE_SPHERE",
    "MODE_PLANE",
    "NO_CONTOUR_REGION",
]
