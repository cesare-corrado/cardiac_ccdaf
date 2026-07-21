"""
CCDAF — Cardiac Clinical Data Analysis Framework
=================================================

Workflow

    LOAD  ->  SEEDS (6)  ->  AUTO TAG  ->  accept? ─no─► MANUAL EDIT
                                                │
                                              yes
                                                ▼
                          CLIP (PV contours, mitral sphere/plane)
                                                │
                                              yes
                                                ▼
                                              SAVE

Segmentation workflow (parallel)

    LOAD .nii / CREATE FROM MESH  ->  EDIT (morphology, brush)
                                                │
                                                ▼
                                EXPORT TO VTK / SAVE .nii

Run
---
    python ccdaf.py [path/to/mesh.vtk]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import warnings
import numpy as np

# Compat shim — VTK's vtk.util.numpy_support module references aliases
# (`numpy.bool`, `numpy.int`, etc.) that were removed in numpy ≥1.24.
# Restoring them as their builtin equivalents is safe (that's what they
# always meant) and unblocks numpy_to_vtk / vtk_to_numpy on older VTKs.
# `hasattr` itself raises FutureWarning on numpy 1.20–1.23 for some of
# these names, so the probe is wrapped in a warnings filter.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    warnings.simplefilter("ignore", DeprecationWarning)
    for _name, _builtin in (("bool", bool), ("int", int), ("float", float),
                            ("complex", complex), ("object", object),
                            ("str", str), ("long", int), ("unicode", str)):
        if not hasattr(np, _name):
            setattr(np, _name, _builtin)

import pyvista as pv
from pyvistaqt import QtInteractor

import SimpleITK as sitk
import vtk

from PyQt5 import QtCore, QtGui, QtWidgets

from ccdaf.core.mesh_loader import MeshLoader, BODY_LABEL
from ccdaf.interaction.seed_selector import SeedSelector, Seed, SEED_ORDER, SEED_PROMPT, SEED_COLOR
from ccdaf.core.region_tagger import RegionTagger, LABELS
from ccdaf.interaction.manual_editor import ManualEditor, ALLOWED_LABELS
from ccdaf.interaction.clipping_tool import ClippingTool, ClipMode
from ccdaf.gui.postprocessing_widget import PostprocessingWidget
from ccdaf.gui.segmentation_widget import SegmentationWidget
from ccdaf.gui.mesh_info_widget import MeshInfoWidget
from ccdaf.gui.seed_widget import SeedWidget
from ccdaf.gui.tagging_widget import TaggingWidget
from ccdaf.gui.manual_correction_widget import ManualCorrectionWidget
from ccdaf.gui.clipping_widget import ClippingWidget
from ccdaf.gui.eam_load_dialog import (
    EAMLoadDialog, FORMAT_CARTO_STUDY, FORMAT_CARTO_MAPPINGS,
)
from ccdaf.gui.visualisation_widget import VisualisationWidget
from ccdaf.gui.mapping_select_dialog import MappingSelectDialog
from ccdaf.gui.save_mesh_dialog import SaveMeshDialog, FORMAT_PICKLE
from ccdaf.gui.eam_export_dialog import EAMExportDialog
from ccdaf.core.eam_export import EXPORT_BINARY, export_binary, export_vtk
from ccdaf.io.carto_functions import extract_map_list_names
from ccdaf.core.eam_loader import (
    EAMData, load_carto_mapping, displace_electrodes_for, read_bundle,
)
from ccdaf.core.field_transfer import guard_distance, transfer_fields
from ccdaf.core.segmentation import (
    binary_mask_image, negate_xy_inplace, segmentation_to_polydata,
    sync_sitk_from_array, voxelise_polydata,
)
from ccdaf.core.seed_io import load_seeds, save_seeds
from ccdaf.app.views import VIEWS, ViewSpec, title_actor_name


# ---------------------------------------------------------------------------
LABEL_COLORS: Dict[int, str] = {
    BODY_LABEL: "#d9d9d9",
    11: "#e41a1c",
    13: "#377eb8",
    15: "#4daf4a",
    17: "#984ea3",
    19: "#ff7f00",
}


# Colors for segmentation labels 0–8.
# Label 0 is background; its 2D slice color is taken from viridis(0) = "#440154".
SEG_LABEL_COLORS: Dict[int, str] = {
    0: "#440154",   # background — viridis at 0
    1: "#e41a1c",   # red
    2: "#377eb8",   # blue
    3: "#4daf4a",   # green
    4: "#984ea3",   # purple
    5: "#ff7f00",   # orange
    6: "#a65628",   # brown
    7: "#f781bf",   # pink
    8: "#17becf",   # cyan
}
# Ordered list for building a 9-entry discrete colormap (index == label value).
_SEG_COLOR_LIST = [SEG_LABEL_COLORS[i] for i in range(9)]

PV_NAMES = ("LSPV", "LIPV", "RSPV", "RIPV")

# Fields whose values are labels rather than measurements. A colour ramp
# over these would be meaningless, so the visualisation widget hands them
# to the region path (discrete colours + names) instead. Further label
# fields belong here rather than in a paradigm of their own.
CATEGORICAL_FIELDS = ("elemTag",)

# EAM scalar bar: horizontal, along the bottom. Sizes are fractions of the
# viewport, so the height giving a 20:1 bar on screen depends on the window's
# own aspect — see _eam_bar_default_geom().
EAM_BAR_WIDTH  = 0.6       # fraction of the viewport width
EAM_BAR_ASPECT = 20.0      # length : height, as seen on screen
EAM_BAR_POS_X  = 0.2
EAM_BAR_POS_Y  = 0.05
# Out-of-range colors: values past a manual min/max are flagged rather than
# clamped to the map's end colors. They only ever show where data actually
# falls outside the range, so an auto range makes them disappear by itself.
EAM_BELOW_COLOR = "brown"
EAM_ABOVE_COLOR = "magenta"
# EAM electrodes, drawn as Gaussian points.
EAM_ELECTRODE_COLOR = (100.0 / 255.0, 100.0 / 255.0, 100.0 / 255.0)
EAM_ELECTRODE_RADIUS_FRAC = 0.008     # of the mesh's bounding-box diagonal


# The three slice orientations of a segmentation volume. Where each one is
# drawn is not decided here — that is the segmentation view's layout, in
# ccdaf.app.views.
ORIENTATIONS = ("axial", "sagittal", "coronal")


# ---------------------------------------------------------------------------
class CCDAF(QtWidgets.QMainWindow):
    """Qt main window hosting the PyVista view and the full workflow."""

    def __init__(self, initial_data: Optional[str] = None) -> None:
        super().__init__()
        self.setWindowTitle("CCDAF — Cardiac Clinical Data Analysis Framework")
        screen = QtWidgets.QApplication.primaryScreen()
        avail = screen.availableGeometry() if screen is not None else None
        if avail is not None:
            w = min(1360, max(800, avail.width() - 80))
            h = min(860, max(600, avail.height() - 80))
            self.resize(w, h)
            self.setMinimumSize(640, 480)
        else:
            self.resize(1360, 860)

        # ---- mesh state -------------------------------------------------
        self.loader = MeshLoader()
        self.selector: Optional[SeedSelector] = None
        self.tagger: Optional[RegionTagger] = None
        self.editor: Optional[ManualEditor] = None
        self.clipper: Optional[ClippingTool] = None
        self._mesh_actor = None

        # ---- EAM (electroanatomical mapping) state ---------------------
        self._eam_directory: Optional[str] = None
        self._eam_format: Optional[str] = None
        self._eam_selected_file: Optional[str] = None
        self._eam_data = None                              # EAMData once loaded
        self._eam_electrode_points: Optional[np.ndarray] = None
        self._eam_electrode_actor = None
        # Title of the scalar bar we last built, and where the user dragged it
        # to — a re-render rebuilds the bar, and would otherwise reset it.
        self._eam_bar_title: Optional[str] = None
        self._eam_bar_geom = None

        # ---- segmentation state ----------------------------------------
        self._seg_sitk: Optional[sitk.Image] = None
        self._seg_array: Optional[np.ndarray] = None  # shape (Z, Y, X), sitk convention
        self._seg_origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._seg_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
        self._seg_undo_stack: list = []  # up to 2 snapshots of _seg_array
        # The mesh this segmentation was voxelised from, with the flip and
        # spacing used — what converting it back needs to carry the fields
        # and electrodes over. None ⇔ the segmentation came off disk.
        self._seg_source: Optional[dict] = None
        self._transfer_note: Optional[str] = None
        self._seg_idx: Dict[str, int] = {"axial": 0, "sagittal": 0, "coronal": 0}
        self._slice_actors: Dict[str, object] = {}
        self._paint_active: bool = False
        # What the live plotter was built for — layout facts (multi-view?
        # where is each pane?) come from its structure, task identity from
        # its name. Written only by _build_plotter.
        self._view: ViewSpec = VIEWS["general"]
        self._splitter: Optional[QtWidgets.QSplitter] = None
        self.plotter: Optional[QtInteractor] = None
        self._seg_3d_actors: list = []
        self._seg_vtk_actor = None
        self._slice_observer_ids: list = []  # (event, observer_id) pairs
        self.btn_update3d_overlay: Optional[QtWidgets.QPushButton] = None

        # left-panel collapsible sections
        self._sections: Dict[str, QtWidgets.QWidget] = {}
        self._section_actions: Dict[str, QtWidgets.QAction] = {}

        if initial_data is not None:
            self.recent_folder: Path = Path(initial_data).resolve().parent
        else:
            self.recent_folder = Path.cwd()

        self._build_ui()

        if initial_data is not None:
            _p = initial_data.lower()
            if _p.endswith(".nii") or _p.endswith(".nii.gz"):
                try:
                    img = sitk.ReadImage(initial_data)
                    self._set_segmentation(img)
                    self.statusBar().showMessage(
                        f"Loaded segmentation {Path(initial_data).name}"
                    )
                except Exception as exc:
                    QtWidgets.QMessageBox.critical(self, "Load failed", str(exc))
            else:
                self._load_mesh(initial_data)

    # ==================================================================
    # UI
    # ==================================================================
    def _build_ui(self) -> None:
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)

        # --- Menu bar ---------------------------------------------------
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")

        self.act_load = QtWidgets.QAction("&Load mesh…", self)
        self.act_load.setShortcut(QtGui.QKeySequence.Open)
        self.act_load.triggered.connect(self._action_load)
        file_menu.addAction(self.act_load)

        self.act_save = QtWidgets.QAction("&Save mesh…", self)
        self.act_save.setShortcut(QtGui.QKeySequence.Save)
        self.act_save.setEnabled(False)
        self.act_save.triggered.connect(self._action_save)
        file_menu.addAction(self.act_save)

        self.act_close = QtWidgets.QAction("&Close", self)
        self.act_close.setShortcut(QtGui.QKeySequence.Close)
        self.act_close.setEnabled(False)
        self.act_close.triggered.connect(self._action_close)
        file_menu.addAction(self.act_close)

        file_menu.addSeparator()
        act_quit = QtWidgets.QAction("&Quit", self)
        act_quit.setShortcut(QtGui.QKeySequence.Quit)
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        # --- Segmentation menu -----------------------------------------
        seg_menu = menubar.addMenu("&Segmentation")

        self.act_seg_load = QtWidgets.QAction("&Load segmentation…", self)
        self.act_seg_load.triggered.connect(self._action_seg_load)
        seg_menu.addAction(self.act_seg_load)

        self.act_seg_save = QtWidgets.QAction("&Save segmentation…", self)
        self.act_seg_save.setEnabled(False)
        self.act_seg_save.triggered.connect(self._action_seg_save)
        seg_menu.addAction(self.act_seg_save)

        seg_menu.addSeparator()
        self.act_seg_from_mesh = QtWidgets.QAction("Create from polydata…", self)
        self.act_seg_from_mesh.setEnabled(False)
        self.act_seg_from_mesh.triggered.connect(self._action_seg_from_mesh)
        seg_menu.addAction(self.act_seg_from_mesh)

        self.act_seg_to_vtk = QtWidgets.QAction("Export to VTK…", self)
        self.act_seg_to_vtk.setEnabled(False)
        self.act_seg_to_vtk.triggered.connect(self._action_seg_to_vtk)
        seg_menu.addAction(self.act_seg_to_vtk)

        seg_menu.addSeparator()
        self.act_seg_close = QtWidgets.QAction("&Close segmentation", self)
        self.act_seg_close.setEnabled(False)
        self.act_seg_close.triggered.connect(self._action_seg_close)
        seg_menu.addAction(self.act_seg_close)

        # --- EAM menu (electroanatomical mapping) ----------------------
        eam_menu = menubar.addMenu("&EAM")
        self.act_eam_load = QtWidgets.QAction("&Load EAM…", self)
        self.act_eam_load.triggered.connect(self._action_load_eam)
        eam_menu.addAction(self.act_eam_load)
        self.act_eam_export = QtWidgets.QAction("&Export…", self)
        self.act_eam_export.setEnabled(False)
        self.act_eam_export.triggered.connect(self._action_export_eam)
        eam_menu.addAction(self.act_eam_export)

        # Visualise menu — toggles the left-panel sections.
        self.visualise_menu = menubar.addMenu("&Visualise")

        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)

        self._splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, central)
        self._splitter.setChildrenCollapsible(False)
        root.addWidget(self._splitter, 1)

        side_scroll = QtWidgets.QScrollArea()
        side_scroll.setWidgetResizable(True)
        side_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        side_scroll.setFrameShape(QtWidgets.QFrame.StyledPanel)
        side_scroll.setMinimumWidth(280)

        side = QtWidgets.QWidget()
        side_scroll.setWidget(side)
        self._splitter.addWidget(side_scroll)
        v = QtWidgets.QVBoxLayout(side)
        v.setAlignment(QtCore.Qt.AlignTop)

        # --- Mesh info --------------------------------------------------
        self.mesh_info = MeshInfoWidget()
        body = self._register_section(v, "meshinfo", "Mesh info")
        body.addWidget(self.mesh_info)

        # --- Post-processing -------------------------------------------
        self.postproc = PostprocessingWidget(
            mesh_getter=lambda: self.loader.mesh,
            mesh_setter=self._replace_mesh,
            on_status=self.statusBar().showMessage,
            on_surface_moved=self._on_surface_moved,
        )
        self.postproc.mesh_changed.connect(self._on_postproc_applied)
        self.postproc.setTitle("")
        body = self._register_section(v, "postproc", "Mesh post-processing")
        body.addWidget(self.postproc)

        # --- Seeds ------------------------------------------------------
        self.seed_widget = SeedWidget()
        self.seed_widget.start_requested.connect(self._action_start_seeds)
        self.seed_widget.undo_requested.connect(self._action_undo_seed)
        self.seed_widget.reset_requested.connect(self._action_reset_seeds)
        self.seed_widget.save_requested.connect(self._action_save_seeds)
        self.seed_widget.load_requested.connect(self._action_load_seeds)
        body = self._register_section(v, "seeds", "Seed selection")
        body.addWidget(self.seed_widget)

        # --- Tagging ----------------------------------------------------
        self.tagging_widget = TaggingWidget()
        self.tagging_widget.tagging_requested.connect(self._action_run_tagging)
        body = self._register_section(v, "tagging", "Tagging")
        body.addWidget(self.tagging_widget)

        # --- Manual edit (mesh) ----------------------------------------
        label_entries = [(lbl, _label_name(lbl)) for lbl in ALLOWED_LABELS]
        self.manual_widget = ManualCorrectionWidget(label_entries=label_entries)
        self.manual_widget.label_changed.connect(self._action_label_changed)
        self.manual_widget.edit_toggled.connect(self._action_edit_toggle)
        self.manual_widget.fill_holes_requested.connect(self._action_fill_holes)
        self.manual_widget.smooth_requested.connect(self._action_smooth_boundary)
        self.manual_widget.accept_requested.connect(self._action_edit_accept)
        self.manual_widget.undo_requested.connect(self._action_undo_edit)
        body = self._register_section(v, "manual", "Manual correction")
        body.addWidget(self.manual_widget)

        # --- Clipping ---------------------------------------------------
        self.clipping_widget = ClippingWidget(pv_names=list(PV_NAMES))
        self.clipping_widget.pv_start_requested.connect(self._action_pv_start)
        self.clipping_widget.pv_finish_requested.connect(self._action_pv_finish)
        self.clipping_widget.mv_sphere_requested.connect(self._action_mv_sphere_start)
        self.clipping_widget.mv_plane_requested.connect(self._action_mv_plane_start)
        self.clipping_widget.clip_apply_requested.connect(self._action_clip_apply)
        self.clipping_widget.clip_revert_requested.connect(self._action_clip_revert)
        self.clipping_widget.clipping_toggled.connect(self._on_clipping_toggled)
        body = self._register_section(v, "clipping", "Clipping")
        body.addWidget(self.clipping_widget)

        # --- Segmentation widget (hidden until volume present) ---------
        self.seg_widget = SegmentationWidget()
        self.seg_widget.morphology_requested.connect(self._action_seg_morphology)
        self.seg_widget.fill_holes_requested.connect(self._action_seg_fill_holes)
        self.seg_widget.convert_all_requested.connect(self._action_seg_convert_all)
        self.seg_widget.paint_mode_changed.connect(self._action_seg_paint_toggled)
        self.seg_widget.undo_requested.connect(self._action_seg_undo)
        # update_3d_requested is wired by the overlay button instead — see
        # _create_update3d_overlay() / _enter_segmentation_mode().
        body = self._register_section(v, "segmentation", "Segmentation")
        body.addWidget(self.seg_widget)
        self._set_section_visible("segmentation", False)

        # --- EAM display (hidden until an EAM mapping is loaded) --------
        self.vis_widget = VisualisationWidget()
        self.vis_widget.settings_changed.connect(self._render_field)
        self.vis_widget.electrodes_toggled.connect(self._on_electrodes_toggled)
        body = self._register_section(v, "visualisation", "Visualisation")
        body.addWidget(self.vis_widget)
        self._set_section_visible("visualisation", False)

        v.addStretch(1)

        # --- Plotter — start in the general (single 3D) view -----------
        self._build_plotter("general")

        self.statusBar().showMessage("Ready.")

    # ------------------------------------------------------------------
    # Plotter (re)construction
    # ------------------------------------------------------------------
    def _build_plotter(self, view: str) -> None:
        """Create or replace the plotter widget for the named view purpose.

        Tears down the previous QtInteractor (if any), inserts a fresh one
        into the splitter, and styles every pane the view defines. Stale
        actor refs are cleared so callers must re-render after switching.
        """
        spec = VIEWS[view]
        # Tear down any previous plotter.
        if self.plotter is not None:
            try:
                self.plotter.close()
            except Exception:
                pass
            try:
                old = self.plotter.interactor
                old.setParent(None)
                old.deleteLater()
            except Exception:
                pass
            self.plotter = None

        kwargs = {} if spec.groups is None else {"groups": spec.groups}
        self.plotter = QtInteractor(self, shape=spec.shape, **kwargs)
        # From here the live plotter is the new view — the single write site.
        self._view = spec
        self.plotter.interactor.setMinimumSize(480, 360)
        self._splitter.addWidget(self.plotter.interactor)
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setSizes([320, 1000])

        # The X key has one owner: the app. PV clipping and manual correction
        # both want it, and when each bound it itself, whoever came last won —
        # or cleared the other's binding outright. Bound once per plotter
        # (bindings die with it); _on_x_key routes by state at press time.
        for key in ("x", "X"):
            self.plotter.add_key_event(key, self._on_x_key)

        if not spec.is_multiview:
            self.plotter.set_background("black")
            self.plotter.add_axes()
        else:
            for role, loc in spec.roles.items():
                self.plotter.subplot(*loc)
                self.plotter.set_background("black")
                self.plotter.add_axes()
                title = spec.titles.get(role)
                if title:
                    self.plotter.add_text(title, font_size=9, color="white",
                                          name=title_actor_name(loc))
            self._focus_3d()

        # Stale references — old actors lived on the destroyed plotter.
        self._mesh_actor = None
        self._slice_actors = {}

    def _focus_3d(self) -> None:
        """Switch the active subplot to the 3D pane, when the view has one."""
        loc = self._view.roles.get("3d")
        if self._view.is_multiview and loc is not None and self.plotter is not None:
            self.plotter.subplot(*loc)

    def _on_x_key(self) -> None:
        """Route the X key to whichever tool it belongs to right now.

        An in-progress PV contour takes it — but only while the clipping
        panel's checkbox says clipping is active. Otherwise it commits the
        manual-correction batch, which is a no-op with nothing pending.
        """
        if (self.clipper is not None
                and self.clipping_widget.is_clipping_enabled()
                and self.clipper.mode is ClipMode.PV_CONTOUR):
            self.clipper.pick_at_cursor()
            return
        if self.editor is not None:
            self.editor.commit_pending()

    def _on_clipping_toggled(self, enabled: bool) -> None:
        """Deactivating the clipping panel abandons any clip in flight."""
        if enabled or self.clipper is None:
            return
        if self.clipper.mode is not ClipMode.NONE:
            self.clipper.cancel()
            self.plotter.render()
            self.statusBar().showMessage(
                "Clipping deactivated — the clip in progress was cancelled.")

    # ------------------------------------------------------------------
    # "Update 3D" overlay button (anchored bottom-left of the QtInteractor)
    # ------------------------------------------------------------------
    def _create_update3d_overlay(self) -> None:
        """Make a Qt button that floats on top of the 3D quadrant."""
        if self.btn_update3d_overlay is not None:
            self._destroy_update3d_overlay()
        btn = QtWidgets.QPushButton("Update 3D", parent=self.plotter.interactor)
        btn.setStyleSheet(
            "QPushButton {"
            " background-color: rgba(40, 40, 40, 200);"
            " color: white; border: 1px solid #888;"
            " border-radius: 3px; padding: 4px 10px; }"
            "QPushButton:hover { background-color: rgba(70, 70, 70, 220); }"
        )
        btn.clicked.connect(self._action_seg_update_3d)
        btn.adjustSize()
        btn.show()
        btn.raise_()
        self.btn_update3d_overlay = btn
        # Listen to plotter resize to keep the button anchored.
        self.plotter.interactor.installEventFilter(self)
        self._reposition_update3d_overlay()

    def _destroy_update3d_overlay(self) -> None:
        if self.btn_update3d_overlay is None:
            return
        try:
            self.plotter.interactor.removeEventFilter(self)
        except Exception:
            pass
        try:
            self.btn_update3d_overlay.setParent(None)
            self.btn_update3d_overlay.deleteLater()
        except Exception:
            pass
        self.btn_update3d_overlay = None

    def _reposition_update3d_overlay(self) -> None:
        """Anchor the overlay to the bottom-left of the 3D quadrant.

        With the segmentation view's 2×2 layout and its 3D pane at (1, 0),
        the bottom-left of that subplot coincides with the bottom-left of
        the QtInteractor widget. Only that view creates this overlay; a
        view placing 3D elsewhere would have to generalise this anchoring.
        """
        if self.btn_update3d_overlay is None or self.plotter is None:
            return
        host = self.plotter.interactor
        margin = 8
        b = self.btn_update3d_overlay
        b.move(margin, host.height() - b.height() - margin)
        b.raise_()

    def eventFilter(self, obj, event):  # type: ignore[override]
        if (self.btn_update3d_overlay is not None
                and self.plotter is not None
                and obj is self.plotter.interactor
                and event.type() == QtCore.QEvent.Resize):
            self._reposition_update3d_overlay()
        return super().eventFilter(obj, event)

    def _build_mesh_tools(self) -> None:
        """(Re)bind the mesh-side tools to whichever plotter is current.

        The clipper and the editor both hold the plotter they were built
        against, so both die with it and both have to be remade whenever it
        is rebuilt. Only the clipper was, which left manual correction bound
        to nothing after any trip through the segmentation: tagging re-enables
        the button regardless, so it looked live and swallowed every click.
        """
        if self.loader.mesh is None:
            return
        self.clipper = ClippingTool(
            mesh_getter=lambda: self.loader.mesh,
            mesh_setter=self._replace_mesh,
            plotter=self.plotter,
            on_status=self.statusBar().showMessage,
        )
        self.editor = ManualEditor(
            mesh=self.loader.mesh,
            plotter=self.plotter,
            on_render=self._render_mesh,
            on_state=lambda s: None,
            on_commit=self._on_edit_committed,
        )
        self.manual_widget.set_active(True)
        self.manual_widget.set_undo_enabled(False)

    def _enter_segmentation_mode(self) -> None:
        if self._view.name == "segmentation":
            return
        # Force-stop any active mesh-side picker — its callbacks are bound
        # to the plotter we're about to destroy.
        self._teardown_mesh_tools(rebuild_clipper=False)
        self._build_plotter("segmentation")
        if self.loader.mesh is not None:
            self._render_field()
            self._build_mesh_tools()
        self._create_update3d_overlay()

    def _exit_segmentation_mode(self) -> None:
        if self._view.name != "segmentation":
            return
        self._destroy_update3d_overlay()
        self._uninstall_slice_observers()
        self._teardown_mesh_tools(rebuild_clipper=False)
        self._paint_active = False
        self._seg_3d_actors = []
        self._build_plotter("general")
        if self.loader.mesh is not None:
            self._render_field()
            self.plotter.reset_camera()
            self._build_mesh_tools()

    def _teardown_mesh_tools(self, *, rebuild_clipper: bool) -> None:
        """Drop refs to selector/editor/clipper bound to the old plotter."""
        if self.selector is not None:
            try:
                self.selector.stop()
            except Exception:
                pass
            self.selector = None
        if self.editor is not None:
            try:
                self.editor.deactivate()
            except Exception:
                pass
            self.editor = None
        self.clipper = None
        # Reset UI controls that depended on those tools.
        self.seed_widget.set_undo_enabled(False)
        self.seed_widget.set_save_enabled(False)
        self.tagging_widget.set_seeds_complete(False)
        self.manual_widget.reset_state()
        self.clipping_widget.reset_state()

    @staticmethod
    def _section(text: str) -> QtWidgets.QLabel:
        return QtWidgets.QLabel(f"<b>{text}</b>")

    # ------------------------------------------------------------------
    def _register_section(self,
                          side_layout: QtWidgets.QVBoxLayout,
                          key: str,
                          title: str) -> QtWidgets.QVBoxLayout:
        frame = QtWidgets.QFrame()
        frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        outer = QtWidgets.QVBoxLayout(frame)
        outer.setContentsMargins(6, 4, 6, 6)
        outer.setSpacing(4)

        hdr = QtWidgets.QHBoxLayout()
        hdr.setContentsMargins(0, 0, 0, 0)
        hdr.addWidget(QtWidgets.QLabel(f"<b>{title}</b>"), 1)
        btn_close = QtWidgets.QToolButton()
        btn_close.setText("✕")
        btn_close.setAutoRaise(True)
        btn_close.setToolTip(
            f'Hide \"{title}\" panel (toggle back via the Visualise menu)'
        )
        btn_close.clicked.connect(
            lambda _checked=False, k=key: self._set_section_visible(k, False)
        )
        hdr.addWidget(btn_close)
        outer.addLayout(hdr)

        body = QtWidgets.QVBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        outer.addLayout(body)

        side_layout.addWidget(frame)
        side_layout.addSpacing(4)
        self._sections[key] = frame

        act = QtWidgets.QAction(title, self)
        act.setCheckable(True)
        act.setChecked(True)
        act.toggled.connect(
            lambda on, k=key: self._set_section_visible(k, on)
        )
        self.visualise_menu.addAction(act)
        self._section_actions[key] = act

        return body

    def _set_section_visible(self, key: str, visible: bool) -> None:
        frame = self._sections.get(key)
        if frame is None:
            return
        frame.setVisible(visible)
        act = self._section_actions.get(key)
        if act is not None and act.isChecked() != visible:
            act.blockSignals(True)
            act.setChecked(visible)
            act.blockSignals(False)

    # ==================================================================
    # File actions (mesh)
    # ==================================================================
    def _action_load(self) -> None:
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load mesh", str(self.recent_folder),
            "All files (*);;"
            "Meshes (*.vtk *.vtp *.ply *.stl *.obj);;"
            "Bundle (*.pkl *.pickle)",
        )
        if not fn:
            return
        if Path(fn).suffix.lower() in (".pkl", ".pickle"):
            self._load_bundle(fn)
        else:
            self._load_mesh(fn)

    def _adopt_mesh(self, mesh: pv.PolyData, source_name: str) -> None:
        """Common setup once ``mesh`` is the working mesh, whatever its source.

        The loader's ``mesh``/``path`` must already be set. Rebuilds the
        mesh-side tools against the current plotter, refreshes the panels,
        and renders — everything a fresh mesh needs and nothing EAM- or
        seed-specific, which the callers add.
        """
        self.tagger = RegionTagger(mesh)
        self.editor = None
        self.clipper = ClippingTool(
            mesh_getter=lambda: self.loader.mesh,
            mesh_setter=self._replace_mesh,
            plotter=self.plotter,
            on_status=self.statusBar().showMessage,
        )
        self._reset_view()
        # A plain mesh has fields too (elemTag, and whatever the file carried),
        # so the visualisation panel applies to it just as much as to a mapping.
        self._populate_fields()
        self._set_section_visible("visualisation", True)
        self._render_mesh()
        self._focus_3d()
        self.plotter.reset_camera()
        self.mesh_info.update_info(mesh)
        self.seed_widget.set_prompt("Mesh loaded. Click 'Start seed selection'.")
        self.statusBar().showMessage(f"Loaded {source_name}")

        # Manual editor live from the start (see the X-key routing).
        self.editor = ManualEditor(
            mesh=self.loader.mesh,
            plotter=self.plotter,
            on_render=self._render_mesh,
            on_state=lambda s: None,
            on_commit=self._on_edit_committed,
        )
        self.manual_widget.set_active(True)
        self.manual_widget.set_undo_enabled(False)

        self.seed_widget.set_start_enabled(True)
        self.seed_widget.set_reset_enabled(True)
        self.seed_widget.set_load_enabled(True)
        self.act_save.setEnabled(True)
        self.act_seg_from_mesh.setEnabled(True)
        self._sync_close_action()

    def _load_mesh(self, filename: str) -> None:
        try:
            mesh = self.loader.load(filename)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Load failed", str(exc))
            return
        self.recent_folder = Path(filename).resolve().parent
        self._reset_eam_state()
        self._adopt_mesh(mesh, Path(filename).name)

    def _load_bundle(self, filename: str) -> None:
        """Load a File → Save pickle bundle: mesh, tagging, seeds, electrodes."""
        try:
            mesh, seeds, electrodes = read_bundle(filename)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Load failed", str(exc))
            return
        self.recent_folder = Path(filename).resolve().parent
        self.loader.mesh = mesh
        self.loader.path = filename
        self._reset_eam_state()
        self._adopt_mesh(mesh, Path(filename).name)

        # Electrodes, if the bundle carried them: restore the state EAM
        # export needs and draw them.
        elec_points = self._electrode_points_from_record(electrodes)
        if elec_points is not None and len(elec_points):
            self._eam_data = EAMData(
                mesh=mesh,
                field_names=list(mesh.point_data.keys()),
                electrode_points=elec_points,
                map_name=Path(filename).stem,
                electrodes=electrodes,
            )
            self._eam_electrode_points = elec_points
            self.vis_widget.set_electrodes_available(True)
            self.act_eam_export.setEnabled(True)
            self._render_field()

        # Seeds last, so their markers land on the final view.
        if seeds:
            self._apply_loaded_seeds(seeds, Path(filename).name)

    @staticmethod
    def _electrode_points_from_record(electrodes):
        """The (N, 3) electrode coordinates from a raw Carto record, or None."""
        if not electrodes:
            return None
        data = np.asarray(electrodes.get("data"), dtype=float)
        if data.ndim == 2 and data.shape[0] > 0 and data.shape[1] >= 4:
            return data[:, 1:4]
        return None

    def _reset_eam_state(self) -> None:
        """Drop any loaded mapping's state — a fresh non-EAM mesh has none."""
        self._eam_data = None
        self._eam_electrode_points = None
        self._eam_electrode_actor = None
        self._eam_bar_title = None
        self._eam_bar_geom = None
        self.vis_widget.set_electrodes_available(False)
        self.act_eam_export.setEnabled(False)

    def _sync_close_action(self) -> None:
        """File → Close applies whenever a mesh or a segmentation is open."""
        self.act_close.setEnabled(self.loader.mesh is not None
                                  or self._seg_array is not None)

    def _action_close(self) -> None:
        """File → Close: unload everything, back to the startup state."""
        if self._seg_array is not None:
            self._action_seg_close()

        # EAM state.
        self._eam_directory = None
        self._eam_format = None
        self._eam_selected_file = None
        self._eam_data = None
        self._eam_electrode_points = None
        self._eam_electrode_actor = None
        self._eam_bar_title = None
        self._eam_bar_geom = None
        self._eam_warp_note = None
        self.vis_widget.set_electrodes_available(False)
        self.act_eam_export.setEnabled(False)

        # Mesh state and the tools bound to it.
        self._teardown_mesh_tools(rebuild_clipper=False)
        self.tagger = None
        self.loader.mesh = None
        self.loader.path = None
        self._seg_source = None
        self._transfer_note = None

        # The view and the panels, as the app starts.
        self._reset_view()
        self.plotter.render()
        self.mesh_info.update_info(None)
        self.vis_widget.set_fields([], [])
        self._set_section_visible("visualisation", False)
        self.manual_widget.set_active(False)
        self.seed_widget.set_start_enabled(False)
        self.seed_widget.set_reset_enabled(False)
        self.seed_widget.set_load_enabled(False)
        self.seed_widget.set_prompt("Load a mesh to begin.")
        self.seed_widget.set_progress("Seeds: 0 / 6")
        self.act_save.setEnabled(False)
        self.act_seg_from_mesh.setEnabled(False)
        self._sync_close_action()
        self.statusBar().showMessage("Closed.")

    def _action_save(self) -> None:
        if self.loader.mesh is None:
            return
        mesh = self.loader.mesh
        dlg = SaveMeshDialog(
            point_fields=list(mesh.point_data.keys()),
            cell_fields=list(mesh.cell_data.keys()),
            start_dir=str(self.recent_folder), parent=self,
        )
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        fn = dlg.selected_path()
        if not fn:
            return
        fields = dlg.selected_fields()
        self.recent_folder = Path(fn).resolve().parent
        if Path(fn).exists():
            reply = QtWidgets.QMessageBox.question(
                self,
                "Overwrite file",
                f"{fn} already exists. Overwrite?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if reply != QtWidgets.QMessageBox.Yes:
                return

        try:
            if dlg.selected_format() == FORMAT_PICKLE:
                self._save_bundle(fn, fields)
            else:
                binary = dlg.selected_binary()
                self.loader.save(fn, fields=fields, binary=binary)
                written = ", ".join(fields) if fields else "no fields"
                self.statusBar().showMessage(
                    f"Saved to {fn} ({'binary' if binary else 'ASCII'}) "
                    f"— wrote {written}")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Save failed", str(exc))

    def _collect_seeds(self) -> "Optional[dict]":
        """Seeds as ``{name: xyz}`` when the selection is complete, else None.

        Shared by the pickle-bundle save and the EAM binary export so both
        carry the seeds the same way.
        """
        if self.selector is not None and self.selector.is_complete:
            return {name: s.xyz for name, s in self.selector.seeds.items()}
        return None

    def _save_bundle(self, fn: str, fields: "list[str]") -> None:
        """Write the pickle bundle: surface (chosen point fields), seeds,
        electrodes and elemTag, so the mesh reloads with all of them."""
        keep = set(fields)
        surface = self.loader.mesh.copy(deep=True)
        # polydata_to_carto_dict turns every 1-D point field into a colour
        # column, so honour the field selection by dropping the rest first.
        for name in list(surface.point_data.keys()):
            if name not in keep:
                surface.point_data.remove(name)

        seeds = self._collect_seeds()

        electrodes = self._eam_data.electrodes if self._eam_data else None
        export_binary(
            fn, surface,
            electrodes=electrodes,
            electrode_points=self._eam_electrode_points,
            seeds=seeds,
            include_elem_tag=("elemTag" in keep),
        )
        parts = []
        if seeds:
            parts.append(f"{len(seeds)} seeds")
        if electrodes is not None:
            parts.append("electrodes")
        extra = f" (+ {', '.join(parts)})" if parts else ""
        self.statusBar().showMessage(f"Saved bundle to {fn}{extra}")

    # ==================================================================
    # EAM actions
    # ==================================================================
    def _action_load_eam(self) -> None:
        """Pick an EAM source, resolve the map name, and load the mapping.

        * ``Carto-Study``    — the picked study XML feeds
          ``extract_map_list_names``; the user then chooses one mapping from a
          radio-button pop-up.
        * ``Carto-mappings`` — the map name is the picked ``.mesh`` file's stem.
        """
        dlg = EAMLoadDialog(start_dir=str(self.recent_folder), parent=self)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        path = dlg.selected_path()
        if not path:
            return
        self._eam_selected_file = path
        self._eam_directory = str(Path(path).parent)
        self._eam_format = dlg.selected_format()
        self.recent_folder = Path(self._eam_directory)

        if self._eam_format == FORMAT_CARTO_STUDY:
            try:
                map_names = extract_map_list_names(path)
            except Exception as exc:
                QtWidgets.QMessageBox.critical(
                    self, "EAM", f"Could not read the study file:\n{exc}")
                return
            if not map_names:
                QtWidgets.QMessageBox.warning(
                    self, "EAM", "This study lists no mappings.")
                return
            sel = MappingSelectDialog(map_names, parent=self)
            if sel.exec_() != QtWidgets.QDialog.Accepted:
                return
            map_name = sel.selected_mapping()
        elif self._eam_format == FORMAT_CARTO_MAPPINGS:
            map_name = Path(path).stem
        else:
            QtWidgets.QMessageBox.information(
                self, "EAM",
                "Loading is supported for the Carto-Study and Carto-mappings "
                "formats. Pick one of those to load a mapping.")
            return

        if not map_name:
            QtWidgets.QMessageBox.warning(self, "EAM", "No mapping selected.")
            return
        self._load_eam_mapping(self._eam_directory, map_name)

    def _load_eam_mapping(self, directory: str, map_name: str) -> None:
        """Read ``<map_name>.mesh`` + electrodes and adopt them as the working
        mesh, then show the EAM panel and render the first field + electrodes."""
        self.statusBar().showMessage(f"Loading EAM mapping '{map_name}'…")
        QtWidgets.QApplication.processEvents()
        try:
            eam = load_carto_mapping(directory, map_name)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, "EAM load failed",
                f"Could not load mapping '{map_name}':\n{exc}")
            self.statusBar().showMessage("EAM load failed.")
            return

        mesh = eam.mesh
        self._eam_data = eam
        self._eam_electrode_points = eam.electrode_points
        self.vis_widget.set_electrodes_available(
            self._eam_electrode_points is not None
            and len(self._eam_electrode_points) > 0)

        # Adopt as the working mesh and (re)build the mesh-side tools.
        self.loader.mesh = mesh
        self.loader.path = None
        self.tagger = RegionTagger(mesh)
        self.clipper = ClippingTool(
            mesh_getter=lambda: self.loader.mesh,
            mesh_setter=self._replace_mesh,
            plotter=self.plotter,
            on_status=self.statusBar().showMessage,
        )
        self.editor = ManualEditor(
            mesh=self.loader.mesh,
            plotter=self.plotter,
            on_render=self._render_mesh,
            on_state=lambda s: None,
            on_commit=self._on_edit_committed,
        )
        self.manual_widget.set_active(True)
        self.manual_widget.set_undo_enabled(False)
        self.seed_widget.set_start_enabled(True)
        self.seed_widget.set_reset_enabled(True)
        self.seed_widget.set_prompt(
            "EAM mapping loaded. Click 'Start seed selection'.")
        self.seed_widget.set_load_enabled(True)
        self.act_save.setEnabled(True)
        self.act_seg_from_mesh.setEnabled(True)
        self.mesh_info.update_info(mesh)

        # EAM display panel: populate fields and reveal it.
        self._populate_fields()
        if eam.field_names:
            self.vis_widget.select_field(eam.field_names[0])
        self._set_section_visible("visualisation", True)
        self.act_eam_export.setEnabled(True)
        self._sync_close_action()

        # Render coloured by the first field, with electrode spheres.
        self._reset_view()
        self._render_field()
        self._focus_3d()
        self.plotter.reset_camera()
        self.plotter.render()
        self.statusBar().showMessage(
            f"Loaded EAM mapping '{map_name}': {mesh.n_points} points, "
            f"{len(self._eam_electrode_points)} electrodes.")

    def _clear_scalar_bars(self) -> None:
        """Drop every scalar bar on the plotter.

        The mesh is always the single 'atrium' actor, so at most one bar is
        ever meant to be on screen. Bars are keyed by title, so without this
        each new title (another EAM field, or Regions) stacks a further bar.

        An interactive bar is two props: the actor pyvista tracks by title,
        and the widget's representation, which is what actually draws it.
        Taking the mesh actor away makes pyvista forget the bar its mapper
        fed — it drops the actor and the title, and leaves the representation
        on screen with nothing left that names it. So the widgets are
        disabled from their own dict rather than through the titles pyvista
        still admits to; going by title alone leaves the old bar painted
        beside the new one.
        """
        widgets = getattr(self.plotter.scalar_bars, "_scalar_bar_widgets", {})
        for widget in list(widgets.values()):
            try:
                widget.SetEnabled(0)
            except Exception:
                pass
        for title in list(self.plotter.scalar_bars.keys()):
            try:
                self.plotter.remove_scalar_bar(title, render=False)
            except Exception:
                pass
        widgets.clear()

    def _eam_bar_widget(self, title: Optional[str]):
        """The interactive widget behind a scalar bar, or None.

        pyvista keeps these in a private dict on the ScalarBars collection;
        there is no public accessor.
        """
        if title is None:
            return None
        widgets = getattr(self.plotter.scalar_bars, "_scalar_bar_widgets", {})
        return widgets.get(title)

    def _remember_eam_bar_geom(self) -> None:
        """Note where the user dragged/resized the EAM bar, so the next render
        restores it instead of snapping back to the default corner."""
        widget = self._eam_bar_widget(self._eam_bar_title)
        if widget is None:
            return
        try:
            rep = widget.GetRepresentation()
            self._eam_bar_geom = (tuple(rep.GetPosition()),
                                  tuple(rep.GetPosition2()))
        except Exception:
            pass

    def _eam_bar_default_geom(self):
        """Where a fresh EAM bar goes: horizontal, EAM_BAR_ASPECT:1 on screen.

        VTK sizes the bar in viewport fractions, so the height fraction that
        reads as 20:1 depends on the window's aspect. A 2x2 segmentation grid
        halves both sides and so leaves the ratio — and this formula — intact.
        """
        try:
            win_w, win_h = (float(v) for v in self.plotter.window_size)
        except Exception:
            win_w = win_h = 0.0
        if win_w <= 0.0 or win_h <= 0.0:
            win_w = win_h = 1.0
        height = EAM_BAR_WIDTH * (win_w / win_h) / EAM_BAR_ASPECT
        return (EAM_BAR_POS_X, EAM_BAR_POS_Y), (EAM_BAR_WIDTH, height)

    def _enable_bar_interaction(self, title: str, geometry=None) -> None:
        """Let the mouse move and freely resize an interactive scalar bar.

        The widget's representation is what interaction drives, and it starts
        out disagreeing with the actor pyvista placed — until
        ``BuildRepresentation`` copies the representation onto the actor,
        resize drags are swallowed. ``geometry`` (``(px, py), (w, h)``) sets
        the representation explicitly; without it the representation adopts
        the actor's current position, so the bar does not jump. Orientation
        is left to the representation's AutoOrient, which follows the bar's
        aspect — so a bar dragged tall turns vertical.
        """
        widget = self._eam_bar_widget(title)
        if widget is None:      # non-interactive bar: add_mesh's size stands
            return
        try:
            widget.SetResizable(True)
            widget.SetRepositionable(True)
            rep = widget.GetRepresentation()
            rep.ProportionalResizeOff()    # width and height move independently
            rep.SetShowBorderToActive()    # handles appear on hover, to grab
            if geometry is None:
                actor = self.plotter.scalar_bars[title]
                rep.SetPosition(actor.GetPosition())
                rep.SetPosition2(actor.GetPosition2())
            else:
                (px, py), (w, h) = geometry
                rep.SetPosition(px, py)
                rep.SetPosition2(w, h)
            rep.BuildRepresentation()
        except Exception:
            pass

    def _place_eam_scalar_bar(self, title: str) -> None:
        """Give the field's scalar bar our geometry and mouse interaction.

        The geometry passed to ``add_mesh`` is ignored once the widget's
        representation takes over, so it is set on the representation —
        the user's dragged position when there is one, the default corner
        otherwise.
        """
        self._eam_bar_title = title
        try:
            # pyvista draws an 'above'/'below' swatch whenever the lookup table
            # carries an out-of-range colour, and offers no kwarg to stop it —
            # the swatch would show even where no data is out of range.
            bar = self.plotter.scalar_bars[title]
            bar.DrawAboveRangeSwatchOff()
            bar.DrawBelowRangeSwatchOff()
        except Exception:
            pass
        self._enable_bar_interaction(
            title, geometry=self._eam_bar_geom or self._eam_bar_default_geom())

    def _populate_fields(self, keep_selection: bool = False) -> None:
        """Offer every field the current mesh carries, point and cell alike."""
        mesh = self.loader.mesh
        if mesh is None:
            return
        previous = self.vis_widget.current_field() if keep_selection else None
        self.vis_widget.set_fields(
            point_fields=list(mesh.point_data.keys()),
            cell_fields=[k for k in mesh.cell_data.keys() if k != "render_idx"],
            categorical=CATEGORICAL_FIELDS,
        )
        if previous is not None:
            self.vis_widget.select_field(previous)

    def _render_field(self, *_args) -> None:
        """Draw whichever field the visualisation widget has selected.

        Label fields go through the region path — discrete colours and names —
        because a continuous ramp over labels says nothing. Everything else is
        a measurement and gets the colour map.
        """
        if self.loader.mesh is None:
            return
        if self.vis_widget.current_field() is None:
            self._render_mesh()
            return
        if self.vis_widget.is_categorical():
            self._render_mesh()
        else:
            self._render_scalar_field()

    def _render_scalar_field(self, *_args) -> None:
        """Colour the working mesh by the selected measured field and (re)draw
        the EAM electrodes. Falls back to a plain surface when the field is
        absent or all no-data.

        Works for a field on either association: point values are interpolated
        across each triangle by the mapper, cell values are flat over it.

        The range is the field's own when *auto* is ticked, otherwise the
        widget's; values outside a manual range are flagged brown (below) and
        magenta (above).
        """
        mesh = self.loader.mesh
        if mesh is None:
            return
        self._focus_3d()
        # Bars first: removing the mesh actor makes pyvista forget the bar it
        # fed, and a forgotten bar can no longer be cleared by title.
        self._remember_eam_bar_geom()
        self._clear_scalar_bars()
        try:
            self.plotter.remove_actor("atrium", reset_camera=False)
        except Exception:
            pass
        self._mesh_actor = None

        field = self.vis_widget.current_field()
        arr = None
        if field and field in mesh.point_data:
            arr = np.asarray(mesh.point_data[field], dtype=float)
        elif field and field in mesh.cell_data:
            arr = np.asarray(mesh.cell_data[field], dtype=float)
        if arr is not None and np.isfinite(arr).any():
            finite = arr[np.isfinite(arr)]
            if self.vis_widget.is_auto():
                lo, hi = float(finite.min()), float(finite.max())
                self.vis_widget.set_range_display(lo, hi)
            else:
                lo, hi = self.vis_widget.clim()
            if hi <= lo:   # constant field, or min/max typed the wrong way round
                lo, hi = min(lo, hi) - 0.5, max(lo, hi) + 0.5
            n_bands = self.vis_widget.n_isolines()
            # The bar spans the data as well as the chosen range, so data left
            # outside the range stays visible on it as a flat brown/magenta run.
            bar_lo = min(lo, float(finite.min()))
            bar_hi = max(hi, float(finite.max()))
            lut = _eam_lookup_table(self.vis_widget.current_cmap(),
                                    lo, hi, bar_lo, bar_hi, n_bands)
            # Interactive scalar bars are unsupported on multi-renderer
            # plotters — disable them on any multi-view layout.
            interactive = not self._view.is_multiview
            (px, py), (bw, bh) = self._eam_bar_geom or self._eam_bar_default_geom()
            self._mesh_actor = self.plotter.add_mesh(
                mesh, scalars=field, cmap=lut, clim=(bar_lo, bar_hi),
                nan_color="lightgrey", show_edges=False, name="atrium",
                reset_camera=False,
                scalar_bar_args={
                    "title": str(field),
                    "n_labels": 3,        # min, mid, max
                    # n_colors is left out on purpose: the bar then takes the
                    # table's own entry count, so the brown/magenta runs land
                    # on the boundaries rather than being resampled across them.
                    # 'vertical' must be explicit: pyvista only sets it from a
                    # vertical theme, and its None branches misplace the bar.
                    "vertical": False,
                    "width": bw,
                    "height": bh,
                    "position_x": px,
                    "position_y": py,
                    # The theme's font is black and the 3D view is black.
                    "color": "white",
                    "interactive": interactive,
                },
            )
            self._place_eam_scalar_bar(str(field))
        else:
            self._mesh_actor = self.plotter.add_mesh(
                mesh, color="lightgrey", show_edges=False, name="atrium",
                reset_camera=False,
            )
        self._draw_electrodes()
        self.plotter.render()

    def _draw_electrodes(self) -> None:
        """(Re)draw the EAM electrode positions as Gaussian points.

        ``points_gaussian`` is pyvista's equivalent of ParaView's Point
        Gaussian representation; spheres are its sphere shader preset. The
        default gaussian-blur preset washes the electrodes out against the
        surface, leaving them visible only where they overhang the background.
        """
        try:
            self.plotter.remove_actor("eam_electrodes", reset_camera=False)
        except Exception:
            pass
        self._eam_electrode_actor = None
        pts = self._eam_electrode_points
        if pts is None or len(pts) == 0 or self.loader.mesh is None:
            return
        if not self.vis_widget.show_electrodes():
            return   # the actor is already removed above; stay hidden
        self._focus_3d()
        b = self.loader.mesh.bounds
        diag = float(np.linalg.norm([b[1]-b[0], b[3]-b[2], b[5]-b[4]]))
        radius = max(EAM_ELECTRODE_RADIUS_FRAC * diag, 1e-6)
        cloud = pv.PolyData(np.asarray(pts, dtype=float))
        # The Gaussian mapper scales by point_size * dataset.length / 1300;
        # invert that so the blobs keep a fixed size in world units.
        length = float(cloud.length) or 1.0
        point_size = radius * 1300.0 / length
        self._eam_electrode_actor = self.plotter.add_mesh(
            cloud, style="points_gaussian", color=EAM_ELECTRODE_COLOR,
            point_size=point_size, emissive=False,
            render_points_as_spheres=True, name="eam_electrodes",
            reset_camera=False, pickable=False,
        )

    def _on_electrodes_toggled(self, _on: bool) -> None:
        """Show/hide the electrode actor without re-rendering the field.

        A direct redraw works in every view — the actor otherwise lingers
        across the region view, which never calls _draw_electrodes."""
        self._draw_electrodes()
        self.plotter.render()

    # ==================================================================
    # Seed actions
    # ==================================================================
    def _action_start_seeds(self) -> None:
        if self.loader.mesh is None:
            return
        if self.selector is not None:
            self.selector.stop()

        self._focus_3d()
        self.selector = SeedSelector(
            mesh=self.loader.mesh,
            plotter=self.plotter,
            on_progress=self._on_seed_progress,
            on_complete=self._on_seeds_complete,
        )
        self.selector.start()
        self.seed_widget.set_undo_enabled(True)
        self.tagging_widget.set_seeds_complete(False)
        self.statusBar().showMessage("Seed selection active — click on the mesh.")

    def _action_undo_seed(self) -> None:
        if self.selector is None:
            return
        removed = self.selector.undo_last()
        if removed is not None and not self.selector.is_active:
            self.selector.resume()
        self.tagging_widget.set_seeds_complete(False)

    def _action_reset_seeds(self) -> None:
        if self.selector is None:
            return
        self.selector.reset()
        self.selector.start()
        self.tagging_widget.set_seeds_complete(False)

    def _on_seed_progress(self, next_name: str, done: int, total: int) -> None:
        self.seed_widget.set_progress(f"Seeds: {done} / {total}")
        # Saving means something only for a complete selection; progress is
        # emitted on every transition, so this tracks undo and reset too.
        self.seed_widget.set_save_enabled(done == total)
        if done < total and next_name:
            color = SEED_COLOR.get(next_name, "#ffffff")
            self.seed_widget.set_prompt(
                f"<span style='color:{color};font-weight:bold'>Next: {next_name}"
                f"</span><br>{SEED_PROMPT[next_name]}"
            )
        else:
            self.seed_widget.set_prompt("All seeds collected.")

    def _on_seeds_complete(self, seeds: Dict[str, Seed]) -> None:
        self.tagging_widget.set_seeds_complete(True)
        self.statusBar().showMessage("All seeds collected. Ready to tag.")

    def _action_save_seeds(self) -> None:
        if self.selector is None or not self.selector.is_complete:
            return
        stem = Path(self.loader.path).stem if self.loader.path else "mesh"
        default = str(Path(self.recent_folder) / f"{stem}.seeds.json")
        fn, filt = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save seeds", default,
            "Seeds JSON (*.json);;Surface + seeds pickle (*.pkl)",
        )
        if not fn:
            return
        if not fn.lower().endswith((".json", ".pkl", ".pickle")):
            fn += ".pkl" if "pickle" in filt else ".json"
        self.recent_folder = Path(fn).resolve().parent
        seeds = {name: s.xyz for name, s in self.selector.seeds.items()}
        try:
            save_seeds(fn, seeds, mesh=self.loader.mesh)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Save seeds failed", str(exc))
            return
        self.statusBar().showMessage(f"Saved {len(seeds)} seeds to {fn}")

    def _action_load_seeds(self) -> None:
        if self.loader.mesh is None:
            return
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load seeds", str(self.recent_folder),
            "Seed files (*.json *.pkl *.pickle);;All files (*)",
        )
        if not fn:
            return
        self.recent_folder = Path(fn).resolve().parent
        try:
            positions = load_seeds(fn)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Load seeds failed", str(exc))
            return
        self._apply_loaded_seeds(positions, Path(fn).name)

    def _apply_loaded_seeds(self, positions: Dict[str, np.ndarray],
                            source_name: str) -> None:
        """Rebuild the seed selection from saved coordinates on the live mesh.

        Each coordinate is snapped to the current surface; a validation
        failure stops there, keeps the earlier seeds and resumes picking.
        Shared by the seed panel's Load and the pickle-bundle load.
        """
        if self.selector is not None:
            self.selector.stop()
        self._focus_3d()
        self.tagging_widget.set_seeds_complete(False)
        self.selector = SeedSelector(
            mesh=self.loader.mesh,
            plotter=self.plotter,
            on_progress=self._on_seed_progress,
            on_complete=self._on_seeds_complete,
        )
        problems = self.selector.apply_positions(positions)
        self.seed_widget.set_undo_enabled(True)
        if problems:
            self.selector.resume()
            QtWidgets.QMessageBox.warning(
                self, "Seeds partially loaded",
                "\n".join(problems)
                + "\n\nThe seeds before the failure are placed — pick the "
                  "remaining ones in the 3D view.",
            )
        else:
            self.statusBar().showMessage(
                f"Loaded {len(SEED_ORDER)} seeds from {source_name} "
                "(snapped to the current surface).")

    # ==================================================================
    # Tagging
    # ==================================================================
    def _action_run_tagging(self) -> None:
        if self.tagger is None or self.selector is None:
            return
        if not self.selector.is_complete:
            QtWidgets.QMessageBox.warning(
                self, "Seeds incomplete",
                "Please complete seed selection before tagging.",
            )
            return
        try:
            cfg = self.tagger.config
            factors = self.tagging_widget.radius_factors()
            cfg.lspv_radius_factor = factors["LSPV"]
            cfg.lipv_radius_factor = factors["LIPV"]
            cfg.rspv_radius_factor = factors["RSPV"]
            cfg.ripv_radius_factor = factors["RIPV"]
            cfg.laa_radius_factor  = factors["LAA"]
            cfg._validate()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid radius factor", str(exc))
            return

        self.statusBar().showMessage("Running geodesic tagging…")
        QtWidgets.QApplication.processEvents()
        try:
            self.tagger.tag(self.selector.seeds_for_tagging())
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Tagging failed", str(exc))
            return
        self._render_mesh()
        self.statusBar().showMessage("Tagging complete — review & correct if needed.")

        # Auto-tagging overwrites the mesh — clear any stale undo history.
        if self.editor is not None:
            self.editor.reset_undo()
        self.manual_widget.set_undo_enabled(False)

        self.manual_widget.set_active(True)
        self.manual_widget.set_label_index(0)
        if self.editor:
            self.editor.set_active_label(self.manual_widget.current_label())

    # ==================================================================
    # Manual editor (mesh)
    # ==================================================================
    def _action_label_changed(self, label: int) -> None:
        if self.editor is None:
            return
        self.editor.set_active_label(label)

    def _action_edit_toggle(self, on: bool) -> None:
        if self.editor is None:
            return
        self._focus_3d()
        if on:
            self.editor.activate()
        else:
            self.editor.deactivate()

    def _action_edit_accept(self) -> None:
        if self.editor is None:
            return
        self.editor.accept()
        self.editor.deactivate()
        # Guard the export: reduce each PV label to a single connected patch.
        # tag() enforces this, but a manual edit or a segmentation round trip
        # can strand a stray cell that tag() never re-checks, and CemrgApp's
        # label check rejects a label split across two regions. Runs on accept
        # so no accepted tagging carries an island; re-runs on every re-accept.
        stray = 0
        if self.tagger is not None and self.loader.mesh is not None:
            tags = np.asarray(self.loader.mesh.cell_data["elemTag"])
            cleaned = self.tagger.reduce_to_single_components(tags)
            stray = int(np.count_nonzero(cleaned != tags))
            if stray:
                self.loader.mesh.cell_data["elemTag"] = cleaned
        self.manual_widget.on_accepted()
        self._render_mesh()
        self.clipping_widget.set_enabled_after_accept()
        note = (f" — reassigned {stray} stray label cell{'s' if stray != 1 else ''} "
                "to keep each region connected") if stray else ""
        self.statusBar().showMessage(
            f"Tagging accepted{note}. Proceed to clipping.")

    def _on_edit_committed(self) -> None:
        if self.editor is not None:
            self.manual_widget.set_undo_enabled(self.editor.can_undo)

    def _action_undo_edit(self) -> None:
        if self.editor is None:
            return
        self.editor.undo()
        self.manual_widget.set_undo_enabled(self.editor.can_undo)

    def _action_fill_holes(self) -> None:
        if self.editor and self.tagger:
            self.editor.fill_holes(self.tagger)
            self.statusBar().showMessage("Holes filled while preserving boundaries.")

    def _action_smooth_boundary(self, dilate: bool, erode: bool) -> None:
        """Smooth the active label's boundary — one dilate/erode pass per click."""
        if self.editor is None or self.tagger is None:
            return
        label = self.manual_widget.current_label()
        if label == BODY_LABEL:
            self.statusBar().showMessage(
                "Select a PV label to smooth — body is the background.")
            return
        ops = "+".join(w for w, on in (("dilate", dilate), ("erode", erode)) if on)
        if not ops:
            self.statusBar().showMessage("Tick Dilate and/or Erode first.")
            return
        n = self.editor.smooth_label(self.tagger, label, dilate, erode)
        self.manual_widget.set_undo_enabled(self.editor.can_undo)
        self.statusBar().showMessage(
            f"Smoothed label {label} ({ops}): {n} cell{'s' if n != 1 else ''} changed"
            if n else f"Label {label} ({ops}): nothing to smooth.")

    # ==================================================================
    # Clipping — PV
    # ==================================================================
    def _action_pv_start(self, pv_name: str) -> None:
        if self.clipper is None:
            return
        pv_label = LABELS[pv_name]
        if pv_label not in np.unique(self.loader.mesh.cell_data["elemTag"]):
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid PV selection",
                f"{pv_name} region is not present in the current tagging."
            )
            return
        self._focus_3d()
        self.clipper.start_pv_contour(pv_label=pv_label)
        self.clipping_widget.set_pv_finish_enabled(True)
        self.clipping_widget.set_clip_revert_enabled(True)

    def _action_pv_finish(self) -> None:
        if self.clipper is None or self.selector is None:
            return
        pv_name = self.clipping_widget.selected_pv()
        seed = self.selector.seeds.get(pv_name)
        if seed is None:
            QtWidgets.QMessageBox.warning(
                self, "Missing PV seed",
                f"No {pv_name} seed available — cannot disambiguate clip side.",
            )
            return
        res = self.clipper.finish_pv_contour(pv_seed_xyz=seed.xyz)
        if res is None:
            return
        self._render_mesh()
        self.clipping_widget.set_pv_finish_enabled(False)

    # ==================================================================
    # Clipping — mitral
    # ==================================================================
    def _mitral_seed_xyz(self) -> Optional[np.ndarray]:
        if self.selector is None or "MV" not in self.selector.seeds:
            return None
        return self.selector.seeds["MV"].xyz

    def _action_mv_sphere_start(self) -> None:
        if self.clipper is None:
            return
        seed = self._mitral_seed_xyz()
        if seed is None:
            QtWidgets.QMessageBox.warning(
                self, "No mitral seed",
                "Select the mitral seed (MV) during seed selection first.",
            )
            return
        self._focus_3d()
        b = self.loader.mesh.bounds
        diag = float(np.linalg.norm([b[1]-b[0], b[3]-b[2], b[5]-b[4]]))
        self.clipper.start_mv_sphere(center=seed, radius=0.05 * diag)
        self.clipping_widget.set_clip_apply_enabled(True)
        self.clipping_widget.set_clip_revert_enabled(True)

    def _action_mv_plane_start(self) -> None:
        if self.clipper is None:
            return
        seed = self._mitral_seed_xyz()
        if seed is None:
            QtWidgets.QMessageBox.warning(
                self, "No mitral seed",
                "Select the mitral seed (MV) during seed selection first.",
            )
            return
        self._focus_3d()
        c = np.asarray(self.loader.mesh.center, dtype=float)
        normal = seed - c
        if np.linalg.norm(normal) < 1e-9:
            normal = np.array([0.0, 0.0, 1.0])
        normal /= np.linalg.norm(normal)
        self.clipper.start_mv_plane(origin=seed, normal=normal)
        self.clipping_widget.set_clip_apply_enabled(True)
        self.clipping_widget.set_clip_revert_enabled(True)

    def _action_clip_apply(self) -> None:
        if self.clipper is None:
            return
        reply = QtWidgets.QMessageBox.question(
            self,
            "Confirm clip",
            "This will permanently modify the mesh. Continue?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return
        mode = self.clipper.mode
        if mode is ClipMode.MV_SPHERE:
            self.clipper.apply_mv_sphere()
        elif mode is ClipMode.MV_PLANE:
            seed = self._mitral_seed_xyz()
            self.clipper.apply_mv_plane(mitral_seed=seed)
        else:
            return
        self._render_mesh()
        self.clipping_widget.set_clip_apply_enabled(False)
        self.clipping_widget.set_clip_revert_enabled(False)
        self.clipping_widget.set_pv_finish_enabled(False)

    def _action_clip_revert(self) -> None:
        if self.clipper is None:
            return
        self.clipper.cancel()
        self.clipper.restore()
        self._render_mesh()
        self.clipping_widget.set_clip_apply_enabled(False)
        self.clipping_widget.set_pv_finish_enabled(False)
        self.clipping_widget.set_clip_revert_enabled(False)

    # ==================================================================
    # Mesh rendering (3D quadrant)
    # ==================================================================
    def _replace_mesh(self, new_mesh: pv.PolyData) -> None:
        self.loader.mesh = new_mesh
        # Post-processing can add or drop arrays, so re-offer what the new
        # mesh actually has — keeping the user on their field if it survived.
        self._populate_fields(keep_selection=True)
        self.tagger = RegionTagger(new_mesh)
        if self.editor is not None:
            self.editor = ManualEditor(
                mesh=new_mesh,
                plotter=self.plotter,
                on_render=self._render_mesh,
                on_state=lambda s: None,
                on_commit=self._on_edit_committed,
            )
            self.manual_widget.set_undo_enabled(False)
        self.mesh_info.update_info(new_mesh)

    def _action_export_eam(self) -> None:
        """Write the loaded mapping out as it currently stands — repairs,
        smoothing and the electrode displacement included."""
        if self._eam_data is None or self.loader.mesh is None:
            return
        dlg = EAMExportDialog(
            start_dir=str(self.recent_folder),
            default_name=str(self._eam_data.map_name).strip().replace(" ", "_"),
            parent=self,
        )
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        path = dlg.selected_path()
        if not path:
            QtWidgets.QMessageBox.warning(self, "Export", "Give the file a name.")
            return
        if not os.path.isdir(os.path.dirname(path)):
            QtWidgets.QMessageBox.warning(
                self, "Export", f"No such directory:\n{os.path.dirname(path)}")
            return
        if os.path.exists(path):
            reply = QtWidgets.QMessageBox.question(
                self, "Overwrite file", f"{path} already exists. Overwrite?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if reply != QtWidgets.QMessageBox.Yes:
                return
        try:
            if dlg.selected_format() == EXPORT_BINARY:
                # Carry the seeds and tagging too, as the pickle bundle does:
                # the reference downstream reads only surface/electrodes and
                # ignores the extra keys, while ccdaf reads them back — so an
                # export/reload keeps the work rather than resetting elemTag to
                # body and dropping the seeds.
                export_binary(path, self.loader.mesh,
                              electrodes=self._eam_data.electrodes,
                              electrode_points=self._eam_electrode_points,
                              seeds=self._collect_seeds(),
                              include_elem_tag=True)
            else:
                export_vtk(path, self.loader.mesh, binary=dlg.selected_binary())
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Export failed", str(exc))
            return
        self.statusBar().showMessage(f"Exported to {path}")

    def _on_surface_moved(self, old_mesh, new_mesh) -> None:
        """Carry the EAM electrodes when the wall moves under them.

        One path for both callers — post-processing's smoothing, and a
        segmentation round trip — because the electrodes are read from the
        two surfaces as shapes, which is all a marching-cubes result has in
        common with the Carto mesh it came from.

        Neither mesh is altered; this only moves the electrodes. No-op when
        no mapping is loaded.
        """
        pts = self._eam_electrode_points
        if self._eam_data is None or pts is None or len(pts) == 0:
            return
        if (old_mesh is None or new_mesh is None
                or old_mesh.n_cells == 0 or new_mesh.n_cells == 0):
            return
        self._eam_warp_note = None
        try:
            self._eam_electrode_points = displace_electrodes_for(
                pts, old_mesh, new_mesh,
                on_status=lambda msg: setattr(self, "_eam_warp_note", msg))
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self, "EAM",
                f"The surface moved, but the electrodes could not follow "
                f"it:\n{exc}\n\nThey are left where they were.")

    def _on_postproc_applied(self) -> None:
        # Post-processing keeps the point fields, so an EAM mapping stays
        # displayable — put back whichever view the user chose rather than
        # dropping them into the region view.
        self._render_field()
        self._focus_3d()
        self.plotter.reset_camera()
        # Report the warp last. This runs from mesh_changed, and the widget
        # posts its own "done" message straight after emitting it — so defer
        # by one event-loop pass or the note is buried immediately.
        note = getattr(self, "_eam_warp_note", None)
        if note:
            self._eam_warp_note = None
            QtCore.QTimer.singleShot(
                0, lambda: self.statusBar().showMessage(note, 20000))

    def _reset_view(self) -> None:
        """Clear the 3D viewport — other panes (e.g. slice views) survive."""
        self._focus_3d()
        # plotter.clear() forgets the scalar-bar widgets without disabling
        # them, which leaves their representations drawn over the next mesh.
        self._clear_scalar_bars()
        self.plotter.clear()
        self.plotter.set_background("black")
        self.plotter.add_axes()
        # clear() also wiped the pane's title — put back the one the view
        # defines for the 3D role, if any.
        title = self._view.titles.get("3d")
        if title:
            self.plotter.add_text(
                title, font_size=9, color="white",
                name=title_actor_name(self._view.roles["3d"]),
            )
        self._mesh_actor = None

    def _render_mesh(self) -> None:
        mesh = self.loader.mesh
        if mesh is None:
            return
        self._focus_3d()
        # Same ordering as _render_scalar_field, and for the same reason.
        self._clear_scalar_bars()
        if self._mesh_actor is not None:
            try:
                self.plotter.remove_actor(self._mesh_actor, reset_camera=False)
            except Exception:
                pass
            self._mesh_actor = None

        tags       = np.asarray(mesh.cell_data["elemTag"], dtype=int)
        all_tags   = sorted(LABEL_COLORS.keys())
        all_colors = [LABEL_COLORS[t] for t in all_tags]
        tag_to_idx = {tag: i for i, tag in enumerate(all_tags)}
        indexed_tags = np.array([tag_to_idx.get(t, 0) for t in tags])
        mesh.cell_data["render_idx"] = indexed_tags

        annotations = {j: _label_name(t) for j, t in enumerate(all_tags)}
        cmap = _discrete_cmap(all_colors)

        self._mesh_actor = self.plotter.add_mesh(
            mesh,
            scalars="render_idx",
            cmap=cmap,
            clim=(-0.5, len(all_tags) - 0.5),
            categories=True,
            annotations=annotations,
            show_edges=True,
            edge_color="black",
            line_width=0.5,
            pickable=True,
            name="atrium",
            reset_camera=False,
            scalar_bar_args={
                "title": "Regions",
                "n_labels": 0,
                "label_font_size": 18,
                # No "fmt": this bar labels regions through `annotations` and
                # draws no tick labels. VTK still formats them internally, and
                # applies the format to a double — "%d" is invalid there and
                # aborts (fatally, from VTK 9.6: "invalid format specifier").
                "shadow": True,
                # Interactive scalar bars are unsupported on multi-renderer
                # plotters — disable them on any multi-view layout.
                "interactive": not self._view.is_multiview,
            }
        )
        sbar = self.plotter.scalar_bar
        sbar.SetUnconstrainedFontSize(True)
        sbar.SetAnnotationTextScaling(False)
        # The Regions bar was interactive but its representation was never
        # built, so the widget and the actor disagreed about its geometry
        # and resize drags — the vertical ones visibly — were swallowed.
        # Same treatment as the field bar, keeping the default vertical slot.
        self._enable_bar_interaction("Regions")

        self.plotter.render()

    # ==================================================================
    # Segmentation — file actions
    # ==================================================================
    def _action_seg_load(self) -> None:
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load segmentation", str(self.recent_folder),
            "NIfTI (*.nii *.nii.gz);;All files (*)",
        )
        if not fn:
            return
        self.recent_folder = Path(fn).resolve().parent
        try:
            img = sitk.ReadImage(fn)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Load failed", str(exc))
            return
        self._set_segmentation(img)
        # A segmentation off disk has no source surface, so nothing can be
        # carried onto the mesh it converts to — whatever is loaded is
        # unrelated geometry, and matching against it would be nonsense.
        self._seg_source = None
        self.statusBar().showMessage(f"Loaded segmentation {Path(fn).name}")

    def _action_seg_save(self) -> None:
        if self._seg_array is None:
            return
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save segmentation", str(self.recent_folder),
            "NIfTI (*.nii *.nii.gz);;All files (*)",
        )
        if not fn:
            return
        self.recent_folder = Path(fn).resolve().parent
        try:
            self._sync_sitk_from_array()
            sitk.WriteImage(self._seg_sitk, fn)
            self.statusBar().showMessage(f"Saved segmentation to {fn}")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Save failed", str(exc))

    def _action_seg_from_mesh(self) -> None:
        if self.loader.mesh is None:
            QtWidgets.QMessageBox.warning(self, "No mesh", "Load a mesh first.")
            return
        opts = self._prompt_voxelise_options()
        if opts is None:
            return
        spacing, flip = opts
        self.statusBar().showMessage("Voxelising mesh…")
        QtWidgets.QApplication.processEvents()
        try:
            img = voxelise_polydata(self.loader.mesh, spacing, flip=flip)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Voxelisation failed", str(exc))
            return
        self._set_segmentation(img)
        # Remember what this segmentation was made from. Converting it back
        # needs all three: the surface to read the fields off, the flip to
        # undo (the export prompt asks again, and may disagree), and the
        # spacing, which sets how far a rebuilt wall can honestly stray.
        self._seg_source = {
            "mesh": self.loader.mesh.copy(deep=True),
            "flip": bool(flip),
            "spacing": tuple(float(s) for s in spacing),
        }
        self.statusBar().showMessage(
            f"Voxelisation complete — {img.GetSize()} @ spacing {spacing} (flip={flip})."
        )

    def _action_seg_to_vtk(self) -> None:
        if self._seg_array is None:
            return

        flip = self._prompt_export_flip()
        if flip is None:
            return

        # Optionally save the segmentation as NIfTI first.
        reply = QtWidgets.QMessageBox.question(
            self, "Save segmentation?",
            "Save the segmentation as a NIfTI file before converting?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if reply == QtWidgets.QMessageBox.Yes:
            fn, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save segmentation", str(self.recent_folder),
                "NIfTI (*.nii *.nii.gz);;All files (*)",
            )
            if fn:
                self.recent_folder = Path(fn).resolve().parent
                try:
                    self._sync_sitk_from_array()
                    sitk.WriteImage(self._seg_sitk, fn)
                    self.statusBar().showMessage(f"Saved segmentation to {fn}")
                except Exception as exc:
                    QtWidgets.QMessageBox.critical(self, "Save failed", str(exc))
                    return

        # Convert segmentation to VTK polydata via marching cubes.
        self.statusBar().showMessage("Running marching cubes…")
        QtWidgets.QApplication.processEvents()
        try:
            self._sync_sitk_from_array()
            poly = segmentation_to_polydata(
                self._seg_sitk,
                flip=flip,
                filt_stdev=list(self.seg_widget.gfilt_standard_deviation()),
                filt_rfact=list(self.seg_widget.gfilt_radius_factor()),
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Conversion failed", str(exc))
            return

        # Render the converted surface in the 3D quadrant while still in
        # segmentation mode, then close (reverts plotter to 1×1).
        self._action_seg_update_3d()
        self._action_seg_close()

        # Replace the previously loaded mesh with the converted polydata so
        # that all subsequent operations (tagging, clipping, …) act on it.
        new_mesh = pv.wrap(poly)
        new_mesh.cell_data["elemTag"] = np.full(
            new_mesh.n_cells, BODY_LABEL, dtype=np.int32
        )
        # Marching cubes attaches its own point normals, for shading. They are
        # not a field of this mesh: they would show up in the visualisation
        # list beside the measured quantities, and be written as three unnamed
        # columns of a Carto export. Saving recomputes normals from the
        # geometry when asked, exactly as it does for a Carto mapping.
        if "Normals" in new_mesh.point_data:
            new_mesh.point_data.remove("Normals")
        self._carry_source_onto(new_mesh, export_flip=flip)
        self._replace_mesh(new_mesh)
        self._render_field()
        self.act_save.setEnabled(True)
        self.plotter.reset_camera()
        self.plotter.render()
        self.seed_widget.set_start_enabled(True)
        self.seed_widget.set_reset_enabled(True)
        self.seed_widget.set_prompt("Mesh loaded. Click 'Start seed selection'.")
        notes = [n for n in (getattr(self, "_transfer_note", None),
                             getattr(self, "_eam_warp_note", None)) if n]
        self._eam_warp_note = self._transfer_note = None
        self.statusBar().showMessage(
            " ".join(["Segmentation converted and visualised."] + notes),
            20000 if notes else 0)

    # ------------------------------------------------------------------
    def _carry_source_onto(self, new_mesh: pv.PolyData, *,
                           export_flip: bool) -> None:
        """Carry the voxelised surface's fields and electrodes onto the
        surface rebuilt from the segmentation.

        Does nothing unless this segmentation was made from a mesh here: one
        loaded from disk has no source to read, and the mesh that happens to
        be open is unrelated geometry.

        The two flip prompts are independent, so the round trip mirrors the
        anatomy whenever they disagree. The source is brought into the
        rebuilt surface's frame first — otherwise every match is made against
        a mirror image, and the result looks plausible and is wrong.
        """
        src = self._seg_source
        if src is None:
            return
        source_mesh = src["mesh"]
        electrodes = self._eam_electrode_points

        if bool(src["flip"]) != bool(export_flip):
            source_mesh = source_mesh.copy(deep=True)
            negate_xy_inplace(source_mesh)
            if electrodes is not None and len(electrodes):
                electrodes = np.asarray(electrodes, dtype=float).copy()
                electrodes[:, :2] *= -1.0
                self._eam_electrode_points = electrodes

        max_distance = guard_distance(source_mesh, src["spacing"])

        self._on_surface_moved(source_mesh, new_mesh)
        self._transfer_note = None
        try:
            transfer_fields(
                source_mesh, new_mesh, max_distance=max_distance,
                on_status=lambda msg: setattr(self, "_transfer_note", msg))
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self, "Field transfer",
                f"The surface was rebuilt, but its fields could not be "
                f"carried over:\n{exc}\n\nThe new surface has geometry only.")

    # ------------------------------------------------------------------
    def _prompt_voxelise_options(self) -> Optional[Tuple[Tuple[float, float, float], bool]]:
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Voxelisation options")
        form = QtWidgets.QFormLayout(dlg)
        spins = []
        for axis in ("x", "y", "z"):
            sp = QtWidgets.QDoubleSpinBox()
            sp.setRange(0.01, 100.0)
            sp.setDecimals(3)
            sp.setSingleStep(0.1)
            sp.setValue(1.0)
            form.addRow(f"Spacing {axis}:", sp)
            spins.append(sp)
        chk_flip = QtWidgets.QCheckBox("Flip X/Y (MIRTK orientation)")
        form.addRow(chk_flip)
        bb = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        form.addRow(bb)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return None
        spacing = (float(spins[0].value()), float(spins[1].value()), float(spins[2].value()))
        return spacing, bool(chk_flip.isChecked())

    def _prompt_export_flip(self) -> Optional[bool]:
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Export options")
        form = QtWidgets.QFormLayout(dlg)
        chk_flip = QtWidgets.QCheckBox("Flip X/Y (MIRTK orientation)")
        form.addRow(chk_flip)
        form.addRow(QtWidgets.QLabel(
            "<i>Smoothing iterations and relaxation factor"
            "<br>are taken from the Segmentation panel.</i>"
        ))
        bb = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        form.addRow(bb)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return None
        return bool(chk_flip.isChecked())

    # ==================================================================
    # Segmentation — state synchronisation
    # ==================================================================
    def _set_segmentation(self, img: sitk.Image) -> None:
        """Adopt a new SITK image as the active segmentation."""
        self._seg_sitk = img
        self._seg_array = sitk.GetArrayFromImage(img).astype(np.int32)  # (Z, Y, X)
        self._seg_origin = tuple(img.GetOrigin())
        self._seg_spacing = tuple(img.GetSpacing())
        self._seg_undo_stack.clear()
        self.seg_widget.set_undo_enabled(False)

        # Centre slice indices.
        nz, ny, nx = self._seg_array.shape
        self._seg_idx = {"axial": nz // 2, "sagittal": nx // 2, "coronal": ny // 2}

        # Switch the plotter to 4-quadrant layout (rebuilds widget).
        self._enter_segmentation_mode()

        self.act_seg_save.setEnabled(True)
        self.act_seg_to_vtk.setEnabled(True)
        self.act_seg_close.setEnabled(True)
        self.act_save.setEnabled(False)
        self._set_section_visible("segmentation", True)
        #close other sections to tyding up left panel
        for other_sec in ["meshinfo","postproc","seeds","tagging","manual","clipping"]:
            self._set_section_visible(other_sec, False)
        self._wire_slice_pickers()
        self._refresh_slices(reset_camera=True)
        self._action_seg_update_3d()
        self._sync_close_action()

    def _action_seg_close(self) -> None:
        """Drop the active segmentation and revert the plotter to 1×1."""
        self._seg_sitk = None
        self._seg_array = None
        self._seg_undo_stack.clear()
        self.seg_widget.set_undo_enabled(False)
        self._slice_actors = {}
        self.act_seg_save.setEnabled(False)
        self.act_seg_to_vtk.setEnabled(False)
        self.act_seg_close.setEnabled(False)
        self.act_save.setEnabled(self.loader.mesh is not None)
        self._set_section_visible("segmentation", False)
        self._exit_segmentation_mode()
        for other_sec in ["meshinfo","postproc","seeds","tagging","manual","clipping"]:
            self._set_section_visible(other_sec, True)
        self._sync_close_action()
        self.statusBar().showMessage("Segmentation closed.")

    def _sync_sitk_from_array(self) -> None:
        """Push numpy edits back into the SITK image (preserves geometry)."""
        if self._seg_array is None or self._seg_sitk is None:
            return
        self._seg_sitk = sync_sitk_from_array(self._seg_array, self._seg_sitk)

    # ==================================================================
    # Segmentation — slice rendering
    # ==================================================================
    def _slice_imagedata(self, axis: str) -> Optional[pv.ImageData]:
        """Build a 2D pv.ImageData lying in its true world plane.

        Returns a one-cell-thick ImageData oriented perpendicular to the
        requested axis. Cell-data layout matches VTK's flat ordering
        (X fastest, then Y, then Z).
        """
        if self._seg_array is None:
            return None
        arr = self._seg_array  # shape (Z, Y, X)
        ox, oy, oz = self._seg_origin
        sx, sy, sz = self._seg_spacing
        nz, ny, nx = arr.shape

        if axis == "axial":
            k = int(np.clip(self._seg_idx["axial"], 0, nz - 1))
            slc = arr[k, :, :]                              # (Y, X), C-order: outer Y, inner X
            grid = pv.ImageData(
                dimensions=(nx + 1, ny + 1, 1),
                spacing=(sx, sy, 1.0),
                origin=(ox, oy, oz + k * sz),
            )
        elif axis == "sagittal":
            i = int(np.clip(self._seg_idx["sagittal"], 0, nx - 1))
            slc = arr[:, :, i]                              # (Z, Y), C-order: outer Z, inner Y
            # Cell index for dims=(1, ny+1, nz+1) is iy + iz*ny — matches.
            grid = pv.ImageData(
                dimensions=(1, ny + 1, nz + 1),
                spacing=(1.0, sy, sz),
                origin=(ox + i * sx, oy, oz),
            )
        else:  # coronal
            j = int(np.clip(self._seg_idx["coronal"], 0, ny - 1))
            slc = arr[:, j, :]                              # (Z, X), C-order: outer Z, inner X
            # Cell index for dims=(nx+1, 1, nz+1) is ix + iz*nx — matches.
            grid = pv.ImageData(
                dimensions=(nx + 1, 1, nz + 1),
                spacing=(sx, 1.0, sz),
                origin=(ox, oy + j * sy, oz),
            )
        grid.cell_data["label"] = slc.ravel(order="C").astype(np.float32)
        return grid

    def _refresh_slices(self, reset_camera: bool = False) -> None:
        if self._seg_array is None:
            return
        for axis in ORIENTATIONS:
            loc = self._view.roles[axis]
            self.plotter.subplot(*loc)
            prev = self._slice_actors.get(axis)
            if prev is not None:
                try:
                    self.plotter.remove_actor(prev, reset_camera=False)
                except Exception:
                    pass
            grid = self._slice_imagedata(axis)
            if grid is None:
                continue
            actor = self.plotter.add_mesh(
                grid,
                scalars="label",
                cmap=_discrete_cmap(_SEG_COLOR_LIST),
                clim=(0, 8),
                n_colors=9,
                show_edges=False,
                show_scalar_bar=False,
                name=f"slice_{axis}",
                reset_camera=False,
                pickable=True,
            )
            self._slice_actors[axis] = actor
            # Overlay slice index & axis label.
            idx = self._seg_idx[axis]
            self.plotter.add_text(
                f"{self._view.titles[axis]}  idx={idx}",
                font_size=9, color="yellow",
                name=title_actor_name(loc),
            )
            # Lock the slice view to its canonical orthographic orientation.
            self._set_slice_camera(axis, reset=reset_camera)

        # Crosshair lines on each slice indicating the other two planes.
        self._refresh_slice_crosshairs()
        # Translucent plane overlays on the 3D quadrant.
        self._refresh_3d_plane_overlays(reset_camera=reset_camera)
        self.plotter.render()

    def _set_slice_camera(self, axis: str, *, reset: bool) -> None:
        """Force the slice subplot to its canonical orthographic view.

        Called on every refresh so accidental drags can never leave a
        slice view rotated/panned out of alignment.
        """
        ren = self.plotter.renderer
        cam = ren.GetActiveCamera()
        cam.ParallelProjectionOn()
        ox, oy, oz = self._seg_origin
        sx, sy, sz = self._seg_spacing
        nz, ny, nx = self._seg_array.shape

        if axis == "axial":
            k = int(np.clip(self._seg_idx["axial"], 0, nz - 1))
            cx = ox + 0.5 * nx * sx
            cy = oy + 0.5 * ny * sy
            cz = oz + k * sz
            cam.SetFocalPoint(cx, cy, cz)
            cam.SetPosition(cx, cy, cz + max(nx * sx, ny * sy))
            cam.SetViewUp(0.0, 1.0, 0.0)
        elif axis == "sagittal":
            i = int(np.clip(self._seg_idx["sagittal"], 0, nx - 1))
            cx = ox + i * sx
            cy = oy + 0.5 * ny * sy
            cz = oz + 0.5 * nz * sz
            cam.SetFocalPoint(cx, cy, cz)
            cam.SetPosition(cx + max(ny * sy, nz * sz), cy, cz)
            cam.SetViewUp(0.0, 0.0, 1.0)
        else:  # coronal
            j = int(np.clip(self._seg_idx["coronal"], 0, ny - 1))
            cx = ox + 0.5 * nx * sx
            cy = oy + j * sy
            cz = oz + 0.5 * nz * sz
            cam.SetFocalPoint(cx, cy, cz)
            cam.SetPosition(cx, cy + max(nx * sx, nz * sz), cz)
            cam.SetViewUp(0.0, 0.0, 1.0)

        if reset:
            ren.ResetCamera()
        ren.ResetCameraClippingRange()

    # ------------------------------------------------------------------
    # Crosshair indicators on the three 2D slice quadrants.
    # ------------------------------------------------------------------
    # Color convention: each axis owns one colour everywhere.
    _AXIS_COLOR = {"axial": "#ff5555", "sagittal": "#55aaff", "coronal": "#55ff77"}

    def _refresh_slice_crosshairs(self) -> None:
        """Draw two crosshair lines per slice, marking the other two planes."""
        if self._seg_array is None:
            return
        ox, oy, oz = self._seg_origin
        sx, sy, sz = self._seg_spacing
        nz, ny, nx = self._seg_array.shape
        ix = int(np.clip(self._seg_idx["sagittal"], 0, nx - 1))
        iy = int(np.clip(self._seg_idx["coronal"], 0, ny - 1))
        iz = int(np.clip(self._seg_idx["axial"], 0, nz - 1))
        x_w = ox + ix * sx
        y_w = oy + iy * sy
        z_w = oz + iz * sz
        x_min, x_max = ox, ox + nx * sx
        y_min, y_max = oy, oy + ny * sy
        z_min, z_max = oz, oz + nz * sz

        # axis -> list of (other_axis, p0, p1) lines to draw.
        # Draw each line slightly off the slab so it isn't co-planar
        # (z-fighting with the image).
        eps = 1e-3
        plan = {
            "axial": [
                # Sagittal indicator: vertical line at x = x_w.
                ("sagittal", (x_w, y_min, z_w + eps), (x_w, y_max, z_w + eps)),
                # Coronal indicator: horizontal line at y = y_w.
                ("coronal",  (x_min, y_w, z_w + eps), (x_max, y_w, z_w + eps)),
            ],
            "sagittal": [
                # Coronal indicator: vertical line at y = y_w (in YZ plane).
                ("coronal",  (x_w + eps, y_w, z_min), (x_w + eps, y_w, z_max)),
                # Axial indicator: horizontal line at z = z_w.
                ("axial",    (x_w + eps, y_min, z_w), (x_w + eps, y_max, z_w)),
            ],
            "coronal": [
                # Sagittal indicator: vertical line at x = x_w.
                ("sagittal", (x_w, y_w + eps, z_min), (x_w, y_w + eps, z_max)),
                # Axial indicator: horizontal line at z = z_w.
                ("axial",    (x_min, y_w + eps, z_w), (x_max, y_w + eps, z_w)),
            ],
        }

        for axis, lines in plan.items():
            self.plotter.subplot(*self._view.roles[axis])
            for other_axis, p0, p1 in lines:
                actor_name = f"crosshair_{axis}_{other_axis}"
                try:
                    self.plotter.remove_actor(actor_name, reset_camera=False)
                except Exception:
                    pass
                line = pv.Line(p0, p1)
                self.plotter.add_mesh(
                    line,
                    color=self._AXIS_COLOR[other_axis],
                    line_width=2,
                    name=actor_name,
                    pickable=False,
                    reset_camera=False,
                    show_scalar_bar=False,
                )

    # ------------------------------------------------------------------
    # Translucent plane overlays on the 3D quadrant.
    # ------------------------------------------------------------------
    def _refresh_3d_plane_overlays(self, *, reset_camera: bool) -> None:
        """Show the three slice planes in 3D as α=0.5 textured quads."""
        if self._seg_array is None or self.plotter is None:
            return
        self._focus_3d()

        ox, oy, oz = self._seg_origin
        sx, sy, sz = self._seg_spacing
        nz, ny, nx = self._seg_array.shape

        # Reuse the same slice ImageData built for the 2D views.
        plane_specs = {
            "axial":    self._slice_imagedata("axial"),
            "sagittal": self._slice_imagedata("sagittal"),
            "coronal":  self._slice_imagedata("coronal"),
        }
        for axis, grid in plane_specs.items():
            actor_name = f"plane3d_{axis}"
            border_name = f"plane3d_{axis}_border"
            for name in (actor_name, border_name):
                try:
                    self.plotter.remove_actor(name, reset_camera=False)
                except Exception:
                    pass
            if grid is None:
                continue
            self.plotter.add_mesh(
                grid,
                scalars="label",
                cmap=_discrete_cmap(_SEG_COLOR_LIST),
                clim=(0, 8),
                n_colors=9,
                opacity=0.5,
                show_edges=False,
                show_scalar_bar=False,
                name=actor_name,
                reset_camera=False,
                pickable=False,
            )
            self.plotter.add_mesh(
                grid.outline(),
                color=self._AXIS_COLOR[axis],
                line_width=3,
                name=border_name,
                reset_camera=False,
                pickable=False,
            )

        if reset_camera:
            self.plotter.reset_camera()

    # ==================================================================
    # Segmentation — picking on slice planes
    # ==================================================================
    def _wire_slice_pickers(self) -> None:
        """Enable a single picker that dispatches to the clicked subplot.

        PyVista uses one shared picker per render window — re-enabling
        per-subplot raises ``Picking is already enabled``. Instead we
        capture each slice's renderer reference, then dispatch on click
        using FindPokedRenderer to identify which quadrant the click hit.
        """
        # Capture each slice subplot's renderer.
        self._slice_renderers: Dict[str, object] = {}
        for axis in ORIENTATIONS:
            self.plotter.subplot(*self._view.roles[axis])
            self._slice_renderers[axis] = self.plotter.renderer

        # Reset any previously enabled picker (safe even if none).
        try:
            self.plotter.disable_picking()
        except Exception:
            pass

        self.plotter.enable_point_picking(
            callback=self._on_slice_pick_dispatch,
            show_message=False,
            show_point=False,
            left_clicking=True,
            use_picker=True,
        )
        self._install_slice_observers()

    def _install_slice_observers(self) -> None:
        """Lock slice quadrants and bind mouse-wheel to step slice index.

        Uses high-priority observers on the interactor:
          * MouseWheelForward/Backward over a slice quadrant → step slice;
            the default zoom is suppressed via ``cmd.AbortFlagOn()`` so
            the trackball style (priority 0) never runs.
          * MouseMove while a drag state is active and the cursor is over
            a slice quadrant → terminate the drag (``StopState()``) and
            abort the event so no camera motion is applied.
          * Left/Middle/RightButtonPress is *not* observed — pyvista's
            picker fires on press and the trackball's drag-state setup
            is harmless if the matching MouseMove never reaches it.
        """
        self._uninstall_slice_observers()
        try:
            iren = self.plotter.interactor
        except Exception:
            return

        def axis_for(poked):
            for ax, ren in self._slice_renderers.items():
                if ren is poked:
                    return ax
            return None

        # Forward-declare to allow capturing IDs in callbacks.
        wheel_fwd_id = {"id": None}
        wheel_bwd_id = {"id": None}
        move_id      = {"id": None}
        btn_rgt_id   = {"id": None}
        btn_mid_id   = {"id": None}
        lbt_id       = {"id": None}

        def on_wheel_fwd(caller, _evt):
            x, y = caller.GetEventPosition()
            ax = axis_for(caller.FindPokedRenderer(x, y))
            if ax is None:
                return
            self._step_slice(ax, +1)
            cmd = caller.GetCommand(wheel_fwd_id["id"])
            if cmd is not None:
                cmd.AbortFlagOn()

        def on_wheel_bwd(caller, _evt):
            x, y = caller.GetEventPosition()
            ax = axis_for(caller.FindPokedRenderer(x, y))
            if ax is None:
                return
            self._step_slice(ax, -1)
            cmd = caller.GetCommand(wheel_bwd_id["id"])
            if cmd is not None:
                cmd.AbortFlagOn()

        def on_move(caller, _evt):
            style = caller.GetInteractorStyle()
            if style is None:
                return
            try:
                state = style.GetState()
            except Exception:
                return
            if state == 0:
                return  # idle hover — let it through
            x, y = caller.GetEventPosition()
            ax = axis_for(caller.FindPokedRenderer(x, y))
            if ax is None:
                return
            # Terminate the drag interaction so subsequent moves are idle.
            try:
                style.StopState()
            except Exception:
                pass
            # Re-snap the canonical slice camera in case any motion happened.
            self.plotter.subplot(*self._view.roles[ax])
            self._set_slice_camera(ax, reset=False)
            cmd = caller.GetCommand(move_id["id"])
            if cmd is not None:
                cmd.AbortFlagOn()

        def on_right_btn(caller, _evt):
            """Abort right-button press over slice viewports (prevents zoom/dolly)."""
            x, y = caller.GetEventPosition()
            if axis_for(caller.FindPokedRenderer(x, y)) is not None:
                cmd = caller.GetCommand(btn_rgt_id["id"])
                if cmd is not None:
                    cmd.AbortFlagOn()

        def on_mid_btn(caller, _evt):
            """Abort middle-button press over slice viewports (prevents pan)."""
            x, y = caller.GetEventPosition()
            if axis_for(caller.FindPokedRenderer(x, y)) is not None:
                cmd = caller.GetCommand(btn_mid_id["id"])
                if cmd is not None:
                    cmd.AbortFlagOn()

        def on_left_btn(caller, _evt):
            """Pick slice coordinate and abort so the trackball never enters ROTATE state."""
            if self._seg_array is None:
                return
            x, y = caller.GetEventPosition()
            poked = caller.FindPokedRenderer(x, y)
            ax = axis_for(poked)
            if ax is None:
                return  # 3D viewport — let all normal interactions proceed

            # Display → world via the slice renderer's camera (parallel projection).
            coord = vtk.vtkCoordinate()
            coord.SetCoordinateSystemToDisplay()
            coord.SetValue(float(x), float(y), 0.0)
            world = coord.GetComputedWorldValue(poked)
            world_xyz = np.array(world, dtype=float)

            # Override the axis perpendicular to the slice with the exact plane position
            # so brush painting and crosshair navigation use the correct layer.
            ox, oy, oz = self._seg_origin
            sx, sy, sz = self._seg_spacing
            if ax == "axial":
                world_xyz[2] = oz + self._seg_idx["axial"] * sz
            elif ax == "sagittal":
                world_xyz[0] = ox + self._seg_idx["sagittal"] * sx
            elif ax == "coronal":
                world_xyz[1] = oy + self._seg_idx["coronal"] * sy

            self._on_slice_clicked(ax, world_xyz)

            # Abort: the trackball never sees this press, so it cannot enter ROTATE state.
            cmd = caller.GetCommand(lbt_id["id"])
            if cmd is not None:
                cmd.AbortFlagOn()

        # Priority 10 — runs before the trackball style's priority-0 observers.
        wheel_fwd_id["id"] = iren.AddObserver("MouseWheelForwardEvent",  on_wheel_fwd, 10.0)
        wheel_bwd_id["id"] = iren.AddObserver("MouseWheelBackwardEvent", on_wheel_bwd, 10.0)
        move_id["id"]      = iren.AddObserver("MouseMoveEvent",          on_move,      10.0)
        btn_rgt_id["id"]   = iren.AddObserver("RightButtonPressEvent",   on_right_btn, 10.0)
        btn_mid_id["id"]   = iren.AddObserver("MiddleButtonPressEvent",  on_mid_btn,   10.0)
        lbt_id["id"]       = iren.AddObserver("LeftButtonPressEvent",    on_left_btn,  10.0)
        self._slice_observer_ids = [
            ("MouseWheelForwardEvent",  wheel_fwd_id["id"]),
            ("MouseWheelBackwardEvent", wheel_bwd_id["id"]),
            ("MouseMoveEvent",          move_id["id"]),
            ("RightButtonPressEvent",   btn_rgt_id["id"]),
            ("MiddleButtonPressEvent",  btn_mid_id["id"]),
            ("LeftButtonPressEvent",    lbt_id["id"]),
        ]

    def _uninstall_slice_observers(self) -> None:
        if self.plotter is None or not self._slice_observer_ids:
            self._slice_observer_ids = []
            return
        try:
            iren = self.plotter.interactor
            for _evt, oid in self._slice_observer_ids:
                try:
                    iren.RemoveObserver(oid)
                except Exception:
                    pass
        except Exception:
            pass
        self._slice_observer_ids = []

    def _step_slice(self, axis: str, direction: int) -> None:
        """Advance or retreat the active slice along *axis*."""
        if self._seg_array is None:
            return
        nz, ny, nx = self._seg_array.shape
        limits = {"axial": nz, "sagittal": nx, "coronal": ny}
        cap = limits[axis]
        new_idx = int(np.clip(self._seg_idx[axis] + direction, 0, cap - 1))
        if new_idx == self._seg_idx[axis]:
            return
        self._seg_idx[axis] = new_idx
        self._refresh_slices()
        self.statusBar().showMessage(
            f"{axis} slice → {new_idx} / {cap - 1}"
        )

    def _on_slice_pick_dispatch(self, world_xyz, *args) -> None:
        """Route a pick event to the slice whose renderer was clicked."""
        if self._seg_array is None or not self._slice_renderers:
            return
        # Resolve which renderer the click landed in.
        poked = None
        try:
            iren = self.plotter.interactor  # QVTKRenderWindowInteractor
            x, y = iren.GetEventPosition()
            poked = iren.FindPokedRenderer(x, y)
        except Exception:
            poked = self.plotter.renderer
        for axis, ren in self._slice_renderers.items():
            if ren is poked:
                self._on_slice_clicked(axis, np.asarray(world_xyz))
                return
        # Click outside the three slice quadrants (e.g. on the 3D view) — ignore.

    def _on_slice_clicked(self, axis: str, world_xyz: np.ndarray) -> None:
        if self._seg_array is None:
            return
        if self._paint_active:
            self._apply_brush(world_xyz, axis)
        else:
            self._update_other_slices_from_click(axis, world_xyz)
        self._refresh_slices()

    def _update_other_slices_from_click(self, axis: str, world_xyz: np.ndarray) -> None:
        """Move the orthogonal slice indices to where the user clicked."""
        ix, iy, iz = self._world_to_index(world_xyz)
        if axis == "axial":
            self._seg_idx["sagittal"] = ix
            self._seg_idx["coronal"] = iy
        elif axis == "sagittal":
            self._seg_idx["axial"] = iz
            self._seg_idx["coronal"] = iy
        elif axis == "coronal":
            self._seg_idx["axial"] = iz
            self._seg_idx["sagittal"] = ix
        self.statusBar().showMessage(
            f"Slice idx — axial:{self._seg_idx['axial']} "
            f"sagittal:{self._seg_idx['sagittal']} "
            f"coronal:{self._seg_idx['coronal']}"
        )

    def _world_to_index(self, world_xyz: np.ndarray) -> Tuple[int, int, int]:
        """Convert world XYZ to clamped (i, j, k) voxel indices."""
        ox, oy, oz = self._seg_origin
        sx, sy, sz = self._seg_spacing
        nz, ny, nx = self._seg_array.shape
        ix = int(np.clip(round((world_xyz[0] - ox) / sx), 0, nx - 1))
        iy = int(np.clip(round((world_xyz[1] - oy) / sy), 0, ny - 1))
        iz = int(np.clip(round((world_xyz[2] - oz) / sz), 0, nz - 1))
        return ix, iy, iz

    # ==================================================================
    # Segmentation — brush painting
    # ==================================================================
    def _action_seg_paint_toggled(self, on: bool) -> None:
        self._paint_active = on
        msg = "Paint mode active — click a slice to apply brush." if on \
            else "Paint mode off — clicks navigate slices."
        self.statusBar().showMessage(msg)

    def _apply_brush(self, world_xyz: np.ndarray, axis: str) -> None:
        self._seg_push_undo()
        ix, iy, iz = self._world_to_index(world_xyz)
        radius = self.seg_widget.brush_radius()
        depth_3d = self.seg_widget.is_3d_brush()
        depth = self.seg_widget.brush_depth()
        shape = self.seg_widget.brush_shape()
        actual_label = self.seg_widget.actual_label()
        new_label    = self.seg_widget.new_label()

        nz, ny, nx = self._seg_array.shape

        # Index-space half-extents along each axis.
        hx, hy, hz = radius, radius, radius
        if not depth_3d:
            # 2D: collapse extent perpendicular to the click plane.
            if axis == "axial":
                hz = 0
            elif axis == "sagittal":
                hx = 0
            elif axis == "coronal":
                hy = 0
        else:
            # 3D: extend `depth` voxels along the orthogonal axis (centred on click).
            half = max(0, depth // 2)
            if axis == "axial":
                hz = half
            elif axis == "sagittal":
                hx = half
            elif axis == "coronal":
                hy = half

        x0, x1 = max(0, ix - hx), min(nx - 1, ix + hx)
        y0, y1 = max(0, iy - hy), min(ny - 1, iy + hy)
        z0, z1 = max(0, iz - hz), min(nz - 1, iz + hz)

        zz, yy, xx = np.ogrid[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1]
        dx = (xx - ix).astype(np.float32)
        dy = (yy - iy).astype(np.float32)
        dz = (zz - iz).astype(np.float32)

        if shape == "sphere":
            mask = (dx * dx + dy * dy + dz * dz) <= float(radius * radius)
        elif shape == "square":
            mask = np.ones_like(dz + dy + dx, dtype=bool)
        else:  # cylinder — radial in the click plane, full extent along orthogonal axis.
            if axis == "axial":
                mask = (dx * dx + dy * dy) <= float(radius * radius)
                mask = np.broadcast_to(mask, (z1 - z0 + 1, y1 - y0 + 1, x1 - x0 + 1)).copy()
            elif axis == "sagittal":
                mask = (dy * dy + dz * dz) <= float(radius * radius)
                mask = np.broadcast_to(mask, (z1 - z0 + 1, y1 - y0 + 1, x1 - x0 + 1)).copy()
            else:
                mask = (dx * dx + dz * dz) <= float(radius * radius)
                mask = np.broadcast_to(mask, (z1 - z0 + 1, y1 - y0 + 1, x1 - x0 + 1)).copy()

        sub = self._seg_array[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1]
        paint_mask = mask & (sub == actual_label)
        sub[paint_mask] = new_label
        self._seg_array[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1] = sub
        self.statusBar().showMessage(
            f"Painted {shape} r={radius} ({'3D' if depth_3d else '2D'}) "
            f"label {actual_label}→{new_label} at ({ix},{iy},{iz})"
        )

    # ==================================================================
    # Segmentation — undo
    # ==================================================================
    def _seg_push_undo(self) -> None:
        if self._seg_array is None:
            return
        self._seg_undo_stack.append(self._seg_array.copy())
        if len(self._seg_undo_stack) > 2:
            self._seg_undo_stack.pop(0)
        self.seg_widget.set_undo_enabled(True)

    def _action_seg_undo(self) -> None:
        if not self._seg_undo_stack:
            return
        self._seg_array = self._seg_undo_stack.pop()
        self._sync_sitk_from_array()
        self._refresh_slices()
        self.seg_widget.set_undo_enabled(bool(self._seg_undo_stack))
        self.statusBar().showMessage("Undo applied.")

    # ==================================================================
    # Segmentation — morphology / fill holes / 3D refresh
    # ==================================================================
    def _action_seg_morphology(self, op: str) -> None:
        if self._seg_array is None:
            return
        self._sync_sitk_from_array()
        radius = list(self.seg_widget.kernel_radius())
        if all(r == 0 for r in radius):
            self.statusBar().showMessage("Kernel radius is 0 on every axis — nothing to do.")
            return
        self._seg_push_undo()
        try:
            mask = binary_mask_image(self._seg_sitk)
            if op == "dilate":
                out = sitk.BinaryDilate(mask, radius, kernelType=sitk.sitkBox)
            elif op == "erode":
                out = sitk.BinaryErode(mask, radius, kernelType=sitk.sitkBox)
            elif op == "morph_open":
                out = sitk.BinaryDilate(sitk.BinaryErode(mask, radius,kernelType=sitk.sitkBox), radius,kernelType=sitk.sitkBox)
            elif op == "morph_close":
                out = sitk.BinaryErode(sitk.BinaryDilate(mask, radius,kernelType=sitk.sitkBox), radius,kernelType=sitk.sitkBox)
            else:
                raise Exception(f"{op} not known")
            self._seg_sitk = sitk.Cast(out, self._seg_sitk.GetPixelID())

        except Exception as exc:
            self._seg_undo_stack.pop()
            self.seg_widget.set_undo_enabled(bool(self._seg_undo_stack))
            QtWidgets.QMessageBox.critical(self, "Morphology failed", str(exc))
            return
        self._seg_array = sitk.GetArrayFromImage(self._seg_sitk).astype(np.int32)
        self._refresh_slices()
        self.statusBar().showMessage(
            f"Binary {op} (radius x={radius[0]}, y={radius[1]}, z={radius[2]}) applied."
        )

    def _action_seg_fill_holes(self) -> None:
        if self._seg_array is None:
            return
        self._sync_sitk_from_array()
        self._seg_push_undo()
        try:
            mask = binary_mask_image(self._seg_sitk)
            filled = sitk.BinaryFillhole(mask)
            self._seg_sitk = sitk.Cast(filled, self._seg_sitk.GetPixelID())
        except Exception as exc:
            self._seg_undo_stack.pop()
            self.seg_widget.set_undo_enabled(bool(self._seg_undo_stack))
            QtWidgets.QMessageBox.critical(self, "Fill holes failed", str(exc))
            return
        self._seg_array = sitk.GetArrayFromImage(self._seg_sitk).astype(np.int32)
        self._refresh_slices()
        self.statusBar().showMessage("Holes filled.")

    def _action_seg_convert_all(self, actual_label: int, new_label: int) -> None:
        if self._seg_array is None:
            return
        count = int(np.sum(self._seg_array == actual_label))
        if count == 0:
            self.statusBar().showMessage(
                f"No voxels with label {actual_label} found — nothing to do."
            )
            return
        self._seg_push_undo()
        self._seg_array[self._seg_array == actual_label] = new_label
        self._sync_sitk_from_array()
        self._refresh_slices()
        self.statusBar().showMessage(
            f"Converted {count} voxel(s) from label {actual_label} to label {new_label}."
        )

    def _action_seg_update_3d(self) -> None:
        """Refresh the 3D-quadrant rendering from the current voxel volume.

        Renders one surface actor per present label (1–8), each coloured by
        SEG_LABEL_COLORS. Rendering-only: does not touch the mesh, tagger, or
        save state.
        """
        if self._seg_array is None:
            return

        present_labels = [int(v) for v in np.unique(self._seg_array) if int(v) > 0]
        self.statusBar().showMessage(
            f"Rebuilding 3D rendering (labels {present_labels})…"
        )
        QtWidgets.QApplication.processEvents()
        self._sync_sitk_from_array()

        filt_stdev = list(self.seg_widget.gfilt_standard_deviation())
        filt_rfact = list(self.seg_widget.gfilt_radius_factor())

        polys: Dict[int, vtk.vtkPolyData] = {}
        for lbl in present_labels:
            try:
                polys[lbl] = segmentation_to_polydata(
                    self._seg_sitk,
                    flip=False,
                    filt_stdev=filt_stdev,
                    filt_rfact=filt_rfact,
                    label=lbl,
                )
            except Exception as exc:
                QtWidgets.QMessageBox.critical(
                    self, f"Update 3D failed (label {lbl})", str(exc)
                )
                return

        self._focus_3d()
        if self._mesh_actor is not None:
            try:
                self.plotter.remove_actor(self._mesh_actor, reset_camera=False)
            except Exception:
                pass
            self._mesh_actor = None
        for actor in self._seg_3d_actors:
            try:
                self.plotter.remove_actor(actor, reset_camera=False)
            except Exception:
                pass
        self._seg_3d_actors = []

        for lbl, poly in polys.items():
            color = SEG_LABEL_COLORS.get(lbl, "#d9d9d9")
            actor = self.plotter.add_mesh(
                pv.wrap(poly),
                color=color,
                show_edges=False,
                name=f"seg_surface_lbl{lbl}",
                reset_camera=False,
                show_scalar_bar=False,
            )
            self._seg_3d_actors.append(actor)

        self.plotter.reset_camera()
        self.plotter.render()
        self.statusBar().showMessage("3D rendering updated from segmentation.")



# ---------------------------------------------------------------------------
def _discrete_cmap(hex_colors):
    from matplotlib.colors import ListedColormap
    return ListedColormap(hex_colors)


def _eam_lookup_table(cmap_name: str, lo: float, hi: float,
                      bar_lo: float, bar_hi: float, n_bands: int):
    """Colour table for an EAM field, spanning ``bar_lo..bar_hi``.

    The table covers the *union* of the data range and the chosen min/max, so
    the bar keeps showing data that falls outside the chosen range instead of
    rescaling to it. Inside ``lo..hi`` it is ``cmap_name`` in ``n_bands``
    steps; below ``lo`` a flat brown and above ``hi`` a flat magenta. When the
    chosen range covers the data, ``bar_lo/bar_hi`` are ``lo/hi`` and neither
    flat band exists — the ordinary fixed-range colour map.
    """
    from matplotlib import colormaps
    from pyvista import Color

    span, inner = bar_hi - bar_lo, hi - lo
    # Enough entries that the coloured part still resolves n_bands steps, since
    # it only occupies inner/span of the table. Capped: a very narrow range
    # over a wide bar would otherwise ask for a huge table.
    total = n_bands
    if span > inner > 0:
        total = int(round(n_bands * span / inner))
    total = int(min(max(total, n_bands), 4096))

    cmap = colormaps[cmap_name]
    below, above = Color(EAM_BELOW_COLOR).float_rgb, Color(EAM_ABOVE_COLOR).float_rgb

    lut = pv.LookupTable()
    lut.SetNumberOfTableValues(total)
    lut.SetTableRange(bar_lo, bar_hi)
    for i in range(total):
        value = bar_lo + (i + 0.5) * span / total
        if value < lo:
            rgb = below
        elif value > hi:
            rgb = above
        else:
            frac = 0.0 if inner <= 0 else (value - lo) / inner
            band = min(int(frac * n_bands), n_bands - 1)   # quantise to n_bands
            rgb = cmap((band + 0.5) / n_bands)[:3]
        lut.SetTableValue(i, rgb[0], rgb[1], rgb[2], 1.0)
    lut.SetNanColor(*Color("lightgrey").float_rgb, 1.0)
    return lut


def _label_name(tag: int) -> str:
    inverse = {v: k for k, v in LABELS.items()}
    return inverse.get(tag, "body" if tag == BODY_LABEL else str(tag))


# ---------------------------------------------------------------------------
def main() -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    initial = sys.argv[1] if len(sys.argv) > 1 else None
    win = CCDAF(initial_data=initial)
    win.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
