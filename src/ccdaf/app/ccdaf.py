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
from vtk.util import numpy_support

from PyQt5 import QtCore, QtGui, QtWidgets

from ccdaf.core.mesh_loader import MeshLoader, BODY_LABEL
from ccdaf.interaction.seed_selector import SeedSelector, Seed, SEED_ORDER, SEED_PROMPT, SEED_COLOR
from ccdaf.core.region_tagger import RegionTagger, LABELS
from ccdaf.interaction.manual_editor import ManualEditor, EditState, ALLOWED_LABELS
from ccdaf.interaction.clipping_tool import ClippingTool, ClipMode
from ccdaf.gui.postprocessing_widget import PostprocessingWidget
from ccdaf.gui.segmentation_widget import SegmentationWidget
from ccdaf.gui.mesh_info_widget import MeshInfoWidget
from ccdaf.gui.seed_widget import SeedWidget
from ccdaf.gui.tagging_widget import TaggingWidget
from ccdaf.gui.manual_correction_widget import ManualCorrectionWidget
from ccdaf.gui.clipping_widget import ClippingWidget


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


# Subplot positions for the 2x2 view.
#   (0,0) Top-Left     Axial    XY
#   (0,1) Top-Right    Sagittal YZ
#   (1,1) Bottom-Right Coronal  XZ
#   (1,0) Bottom-Left  3D
SUBPLOT_3D = (1, 0)
SUBPLOT_AXIAL = (0, 0)
SUBPLOT_SAGITTAL = (0, 1)
SUBPLOT_CORONAL = (1, 1)

ORIENTATIONS = ("axial", "sagittal", "coronal")
SUBPLOT_FOR = {
    "axial": SUBPLOT_AXIAL,
    "sagittal": SUBPLOT_SAGITTAL,
    "coronal": SUBPLOT_CORONAL,
}
TITLE_FOR = {
    "axial": "Axial (XY)",
    "sagittal": "Sagittal (YZ)",
    "coronal": "Coronal (XZ)",
    "3d": "3D",
}


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

        # ---- segmentation state ----------------------------------------
        self._seg_sitk: Optional[sitk.Image] = None
        self._seg_array: Optional[np.ndarray] = None  # shape (Z, Y, X), sitk convention
        self._seg_origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._seg_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
        self._seg_undo_stack: list = []  # up to 2 snapshots of _seg_array
        self._seg_idx: Dict[str, int] = {"axial": 0, "sagittal": 0, "coronal": 0}
        self._slice_actors: Dict[str, object] = {}
        self._paint_active: bool = False
        self._segmentation_mode: bool = False  # True ⇔ 2×2 plotter active
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
        )
        self.postproc.mesh_changed.connect(self._on_postproc_applied)
        self.postproc.setTitle("")
        body = self._register_section(v, "postproc", "Post-processing")
        body.addWidget(self.postproc)

        # --- Seeds ------------------------------------------------------
        self.seed_widget = SeedWidget()
        self.seed_widget.start_requested.connect(self._action_start_seeds)
        self.seed_widget.undo_requested.connect(self._action_undo_seed)
        self.seed_widget.reset_requested.connect(self._action_reset_seeds)
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

        v.addStretch(1)

        # --- Plotter — start in single-renderer (3D) mode --------------
        self._build_plotter((1, 1))

        self.statusBar().showMessage("Ready.")

    # ------------------------------------------------------------------
    # Plotter (re)construction
    # ------------------------------------------------------------------
    def _build_plotter(self, shape: Tuple[int, int]) -> None:
        """Create or replace the plotter widget at the requested shape.

        Tears down the previous QtInteractor (if any), inserts a fresh one
        into the splitter, and styles all subplots. Stale actor refs are
        cleared so callers must re-render after switching.
        """
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

        self.plotter = QtInteractor(self, shape=shape)
        self.plotter.interactor.setMinimumSize(480, 360)
        self._splitter.addWidget(self.plotter.interactor)
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setSizes([320, 1000])

        if shape == (1, 1):
            self.plotter.set_background("black")
            self.plotter.add_axes()
        else:
            for row in range(shape[0]):
                for col in range(shape[1]):
                    self.plotter.subplot(row, col)

                    self.plotter.set_background("black")
                    self.plotter.add_axes()
                    title = self._title_for_subplot(row, col)
                    self.plotter.add_text(title, font_size=9, color="white",
                                          name=f"_title_{row}_{col}")
                                          
            self.plotter.subplot(*SUBPLOT_3D)

        # Stale references — old actors lived on the destroyed plotter.
        self._mesh_actor = None
        self._slice_actors = {}

    def _focus_3d(self) -> None:
        """Switch the active subplot to the 3D quadrant when in 2×2 mode."""
        if self._segmentation_mode and self.plotter is not None:
            self.plotter.subplot(*SUBPLOT_3D)

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

        With a 2×2 layout and the 3D viewport at (1, 0), the bottom-left
        of that subplot coincides with the bottom-left half of the
        QtInteractor widget.
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

    def _enter_segmentation_mode(self) -> None:
        if self._segmentation_mode:
            return
        # Force-stop any active mesh-side picker — its callbacks are bound
        # to the plotter we're about to destroy.
        self._teardown_mesh_tools(rebuild_clipper=False)
        self._segmentation_mode = True
        self._build_plotter((2, 2))
        if self.loader.mesh is not None:
            self._render_mesh()
            self.clipper = ClippingTool(
                mesh_getter=lambda: self.loader.mesh,
                mesh_setter=self._replace_mesh,
                plotter=self.plotter,
                on_status=self.statusBar().showMessage,
            )
        self._create_update3d_overlay()

    def _exit_segmentation_mode(self) -> None:
        if not self._segmentation_mode:
            return
        self._destroy_update3d_overlay()
        self._uninstall_slice_observers()
        self._teardown_mesh_tools(rebuild_clipper=False)
        self._segmentation_mode = False
        self._paint_active = False
        self._seg_3d_actors = []
        self._build_plotter((1, 1))
        if self.loader.mesh is not None:
            self._render_mesh()
            self.plotter.reset_camera()
            self.clipper = ClippingTool(
                mesh_getter=lambda: self.loader.mesh,
                mesh_setter=self._replace_mesh,
                plotter=self.plotter,
                on_status=self.statusBar().showMessage,
            )

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
        self.tagging_widget.set_seeds_complete(False)
        self.manual_widget.reset_state()
        self.clipping_widget.reset_state()

    @staticmethod
    def _title_for_subplot(row: int, col: int) -> str:
        if (row, col) == SUBPLOT_AXIAL:
            return TITLE_FOR["axial"]
        if (row, col) == SUBPLOT_SAGITTAL:
            return TITLE_FOR["sagittal"]
        if (row, col) == SUBPLOT_CORONAL:
            return TITLE_FOR["coronal"]
        return TITLE_FOR["3d"]

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
            "Mesh files (*.vtk *.vtp *.ply *.stl *.obj);;All files (*)",
        )
        if fn:
            self._load_mesh(fn)

    def _load_mesh(self, filename: str) -> None:
        try:
            mesh = self.loader.load(filename)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Load failed", str(exc))
            return
        self.recent_folder = Path(filename).resolve().parent

        self.tagger = RegionTagger(mesh)
        self.editor = None
        self.clipper = ClippingTool(
            mesh_getter=lambda: self.loader.mesh,
            mesh_setter=self._replace_mesh,
            plotter=self.plotter,
            on_status=self.statusBar().showMessage,
        )
        self._reset_view()
        self._render_mesh()
        self._focus_3d()
        self.plotter.reset_camera()
        self.mesh_info.update_info(mesh)
        self.seed_widget.set_prompt("Mesh loaded. Click 'Start seed selection'.")
        self.statusBar().showMessage(f"Loaded {Path(filename).name}")
        
        ################ manual editor active from beginning ######
        self.editor = ManualEditor(
            mesh=self.loader.mesh,
            plotter=self.plotter,
            on_render=self._render_mesh,
            on_state=lambda s: None,
            on_commit=self._on_edit_committed,
        )
        self.manual_widget.set_active(True)
        self.manual_widget.set_undo_enabled(False)
        ################ manual editor active from beginning ######

        self.seed_widget.set_start_enabled(True)
        self.seed_widget.set_reset_enabled(True)
        self.act_save.setEnabled(True)
        self.act_seg_from_mesh.setEnabled(True)

    def _action_save(self) -> None:
        if self.loader.mesh is None:
            return
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save mesh", str(self.recent_folder), "VTK (*.vtk);;All files (*)",
        )
        if not fn:
            return
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
            self.loader.save(fn)
            self.statusBar().showMessage(f"Saved to {fn}")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Save failed", str(exc))

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
        self.manual_widget.on_accepted()
        self._render_mesh()
        self.clipping_widget.set_enabled_after_accept()
        self.statusBar().showMessage("Tagging accepted. Proceed to clipping.")

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

    def _on_postproc_applied(self) -> None:
        self._render_mesh()
        self._focus_3d()
        self.plotter.reset_camera()

    def _reset_view(self) -> None:
        """Clear the 3D viewport — preserves segmentation slice views in 2×2 mode."""
        self._focus_3d()
        self.plotter.clear()
        self.plotter.set_background("black")
        self.plotter.add_axes()
        if self._segmentation_mode:
            self.plotter.add_text(
                TITLE_FOR["3d"], font_size=9, color="white",
                name=f"_title_{SUBPLOT_3D[0]}_{SUBPLOT_3D[1]}",
            )
        self._mesh_actor = None

    def _render_mesh(self) -> None:
        mesh = self.loader.mesh
        if mesh is None:
            return
        self._focus_3d()
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
                "fmt": "%d",
                "shadow": True,
                # Interactive scalar bars are unsupported on multi-renderer
                # plotters — disable them when in 4-quadrant mode.
                "interactive": not self._segmentation_mode,
            }
        )
        sbar = self.plotter.scalar_bar
        sbar.SetUnconstrainedFontSize(True)
        sbar.SetAnnotationTextScaling(False)

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
            img = self._voxelise_polydata(self.loader.mesh, spacing, flip=flip)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Voxelisation failed", str(exc))
            return
        self._set_segmentation(img)
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
            poly = self._segmentation_to_polydata(
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
        self._replace_mesh(new_mesh)
        self._render_mesh()
        self.act_save.setEnabled(True)
        self.plotter.reset_camera()
        self.plotter.render()
        self.seed_widget.set_start_enabled(True)
        self.seed_widget.set_reset_enabled(True)
        self.seed_widget.set_prompt("Mesh loaded. Click 'Start seed selection'.")
        self.statusBar().showMessage("Segmentation converted and visualised.")

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
        
        self.statusBar().showMessage("Segmentation closed.")

    def _sync_sitk_from_array(self) -> None:
        """Push numpy edits back into the SITK image (preserves geometry)."""
        if self._seg_array is None or self._seg_sitk is None:
            return
        new = sitk.GetImageFromArray(self._seg_array.astype(np.int16))
        new.SetOrigin(self._seg_sitk.GetOrigin())
        new.SetSpacing(self._seg_sitk.GetSpacing())
        new.SetDirection(self._seg_sitk.GetDirection())
        self._seg_sitk = new

    def _binary_mask_image(self) -> sitk.Image:
        """Build a uint8 0/1 mask of the current segmentation.

        Done in numpy to dodge ITK's ``BinaryThreshold`` parameter-range
        checks (which fail when ``upperThreshold`` exceeds the pixel
        type's max — e.g. ``2**31-1`` on a uint8 voxelisation).
        """
        if self._seg_sitk is None:
            raise RuntimeError("No segmentation loaded.")
        arr = (sitk.GetArrayFromImage(self._seg_sitk) > 0).astype(np.uint8)
        out = sitk.GetImageFromArray(arr)
        out.CopyInformation(self._seg_sitk)
        return out

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
            row, col = SUBPLOT_FOR[axis]
            self.plotter.subplot(row, col)
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
                f"{TITLE_FOR[axis]}  idx={idx}",
                font_size=9, color="yellow",
                name=f"_title_{row}_{col}",
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
            row, col = SUBPLOT_FOR[axis]
            self.plotter.subplot(row, col)
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
        cx = ox + 0.5 * nx * sx
        cy = oy + 0.5 * ny * sy
        cz = oz + 0.5 * nz * sz
        ix = int(np.clip(self._seg_idx["sagittal"], 0, nx - 1))
        iy = int(np.clip(self._seg_idx["coronal"], 0, ny - 1))
        iz = int(np.clip(self._seg_idx["axial"], 0, nz - 1))

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
            row, col = SUBPLOT_FOR[axis]
            self.plotter.subplot(row, col)
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
            self.plotter.subplot(*SUBPLOT_FOR[ax])
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
            mask = self._binary_mask_image()
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
            mask = self._binary_mask_image()
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
                polys[lbl] = self._segmentation_to_polydata(
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

    # ==================================================================
    # Segmentation — voxelisation (mesh → image)
    # ==================================================================
    def _voxelise_polydata(self, mesh: pv.PolyData,
                           spacing: Tuple[float, float, float],
                           *, flip: bool) -> sitk.Image:
        """Convert a polydata surface to a binary SITK volume.

        Self-contained stencil-based rasterisation. Foreground fill is
        vectorised (single allocation through numpy_support).
        """
        # Take a writable deep copy of the polydata so optional flips
        # don't mutate the caller's mesh.
        poly = vtk.vtkPolyData()
        poly.DeepCopy(mesh if isinstance(mesh, vtk.vtkPolyData) else mesh)

        if flip:
            self._negate_xy_inplace(poly)

        spacing_arr = np.asarray(spacing, dtype=float)
        if (spacing_arr <= 0).any():
            raise ValueError(f"Spacing must be strictly positive, got {spacing}.")

        white = self._define_image_from_mesh(poly, spacing_arr)

        # Vectorised foreground fill (replaces per-voxel SetTuple1 loop).
        n_pts = white.GetNumberOfPoints()
        ones = np.ones(n_pts, dtype=np.uint8)
        white.GetPointData().SetScalars(
            numpy_support.numpy_to_vtk(ones, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        )

        stencil = vtk.vtkPolyDataToImageStencil()
        stencil.SetInputData(poly)
        stencil.SetOutputOrigin(white.GetOrigin())
        stencil.SetOutputSpacing(white.GetSpacing())
        stencil.SetOutputWholeExtent(white.GetExtent())
        stencil.Update()

        cutter = vtk.vtkImageStencil()
        cutter.SetInputData(white)
        cutter.SetStencilConnection(stencil.GetOutputPort())
        cutter.ReverseStencilOff()
        cutter.SetBackgroundValue(0)
        cutter.Update()

        return self._vtk_image_to_sitk(cutter.GetOutput())

    @staticmethod
    def _define_image_from_mesh(poly: vtk.vtkPolyData,
                                spacing: np.ndarray) -> vtk.vtkImageData:
        """Allocate a vtkImageData covering the mesh bounds at *spacing*."""
        bounds = poly.GetBounds()  # (xmin, xmax, ymin, ymax, zmin, zmax)
        extents_world = np.array([
            bounds[1] - bounds[0],
            bounds[3] - bounds[2],
            bounds[5] - bounds[4],
        ], dtype=float)
        dims = np.maximum(np.ceil(extents_world / spacing).astype(int), 1)

        img = vtk.vtkImageData()
        img.SetSpacing(float(spacing[0]), float(spacing[1]), float(spacing[2]))
        img.SetDimensions(int(dims[0]), int(dims[1]), int(dims[2]))
        # +1 voxel buffer on the upper extent — matches reference behaviour
        # so the stencil never clips a face that lies on the bbox max.
        img.SetExtent(0, int(dims[0]) + 1,
                      0, int(dims[1]) + 1,
                      0, int(dims[2]) + 1)
        img.SetOrigin(float(bounds[0]), float(bounds[2]), float(bounds[4]))
        img.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
        return img

    @staticmethod
    def _vtk_image_to_sitk(vtk_img: vtk.vtkImageData) -> sitk.Image:
        """Convert a vtkImageData to a SimpleITK image (no Python loops)."""
        ext = vtk_img.GetExtent()
        dims = (ext[1] - ext[0] + 1, ext[3] - ext[2] + 1, ext[5] - ext[4] + 1)
        scalars = vtk_img.GetPointData().GetScalars()
        arr = numpy_support.vtk_to_numpy(scalars).reshape(dims[2], dims[1], dims[0])
        out = sitk.GetImageFromArray(arr.astype(np.uint8))
        out.SetSpacing(tuple(float(s) for s in vtk_img.GetSpacing()))
        out.SetOrigin(tuple(float(o) for o in vtk_img.GetOrigin()))
        return out

    # ==================================================================
    # Segmentation — meshing (image → polydata)
    # ==================================================================

    def _label_mask_image(self, label: int) -> sitk.Image:
        """Return a uint8 0/1 mask for the single label value *label*."""
        if self._seg_sitk is None:
            raise RuntimeError("No segmentation loaded.")
        arr = (sitk.GetArrayFromImage(self._seg_sitk) == label).astype(np.uint8)
        out = sitk.GetImageFromArray(arr)
        out.CopyInformation(self._seg_sitk)
        return out

    def _segmentation_to_polydata(self, *, flip: bool,
                                           filt_stdev: list[float],
                                           filt_rfact: list[float],
                                           label: Optional[int] = None,
                                  ) -> vtk.vtkPolyData:
        """Plain marching cubes + smoothing; no preprocessing.

        When *label* is given, only that label value is meshed. Otherwise
        all voxels > 0 form the surface (legacy binary behaviour).
        """
        if self._seg_sitk is None:
            raise RuntimeError("No segmentation loaded.")

        binary = self._label_mask_image(label) if label is not None \
            else self._binary_mask_image()
        vimg = self._sitk_to_vtk_image(binary)

        outside_dist = vtk.vtkImageEuclideanDistance()
        outside_dist.SetInputData(vimg)
        outside_dist.SetConsiderAnisotropy(True)
        outside_dist.SetAlgorithmToSaito()
        outside_dist.Update()
        
        # Flip binary {0,1} → {1,0} so vtkImageEuclideanDistance can compute
        # distances from outside pixels to the nearest inside boundary.
        # SetOperationToInvert would compute 1/x, giving inf for 0-pixels
        # (no background for the distance filter). Use threshold instead.
        thresh = vtk.vtkImageThreshold()
        thresh.SetInputData(vimg)
        thresh.ThresholdByLower(0.5)
        thresh.SetInValue(1.0)
        thresh.SetOutValue(0.0)
        thresh.ReplaceInOn()
        thresh.ReplaceOutOn()
        thresh.Update()
        inside_dist = vtk.vtkImageEuclideanDistance()
        inside_dist.SetInputData(thresh.GetOutput())
        inside_dist.SetConsiderAnisotropy(True)
        inside_dist.SetAlgorithmToSaito()
        inside_dist.Update()

        sdf = vtk.vtkImageMathematics()
        sdf.SetInput1Data(outside_dist.GetOutput())
        sdf.SetInput2Data(inside_dist.GetOutput())
        sdf.SetOperationToSubtract()
        sdf.Update()
        vimg_sdf = sdf.GetOutput()
        
        
        mc = vtk.vtkMarchingCubes()
        if np.any(np.array(filt_stdev)>0.) and np.any(np.array(filt_rfact)>0.): 
            gaussian = vtk.vtkImageGaussianSmooth()
            gaussian.SetStandardDeviations(filt_stdev[0],filt_stdev[1],filt_stdev[2])
            gaussian.SetRadiusFactors(filt_rfact[0],filt_rfact[1],filt_rfact[2])
            gaussian.SetDimensionality(3)
            gaussian.SetInputData(vimg_sdf)
            gaussian.Update()
            mc.SetInputConnection(gaussian.GetOutputPort())
        else:
            mc.SetInputData(vimg_sdf)
        mc.ComputeScalarsOff()
        mc.ComputeNormalsOff()
        mc.ComputeGradientsOff()
        mc.SetValue(0, 0.0)
        mc.Update()

        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(mc.GetOutputPort())
        normals.ComputePointNormalsOn()
        normals.ComputeCellNormalsOff()
        normals.AutoOrientNormalsOn()
        normals.FlipNormalsOn()
        normals.Update()

        tri = vtk.vtkTriangleFilter()
        tri.SetInputConnection(normals.GetOutputPort())
        tri.PassVertsOff()
        tri.PassLinesOff()
        tri.Update()
        out: vtk.vtkPolyData = tri.GetOutput()


        clean = vtk.vtkCleanPolyData()
        clean.SetInputData(out)
        clean.PointMergingOn()
        clean.ConvertLinesToPointsOff()
        clean.ConvertPolysToLinesOff()
        clean.ConvertStripsToPolysOff()
        clean.Update()
        out = clean.GetOutput()
        if flip:
            self._negate_xy_inplace(out)
        return out

    @staticmethod
    def _sitk_to_vtk_image(img: sitk.Image) -> vtk.vtkImageData:
        """Build a vtkImageData mirroring *img*'s geometry (vectorised)."""
        size = list(img.GetSize())
        spacing = list(img.GetSpacing())
        origin = list(img.GetOrigin())
        direction = list(img.GetDirection())

        vimg = vtk.vtkImageData()
        vimg.SetDimensions(int(size[0]), int(size[1]), int(size[2]))
        vimg.SetSpacing(float(spacing[0]), float(spacing[1]), float(spacing[2]))
        vimg.SetOrigin(float(origin[0]), float(origin[1]), float(origin[2]))
        vimg.SetExtent(0, size[0] - 1, 0, size[1] - 1, 0, size[2] - 1)
        if vtk.vtkVersion.GetVTKMajorVersion() >= 9 and len(direction) == 9:
            vimg.SetDirectionMatrix(direction)

        # SITK array shape is (Z, Y, X) C-contiguous; ravel matches VTK
        # linear indexing where X varies fastest.
        arr = sitk.GetArrayFromImage(img)
        flat = np.ascontiguousarray(arr).ravel()
        vimg.GetPointData().SetScalars(numpy_support.numpy_to_vtk(flat, deep=True))
        return vimg

    @staticmethod
    def _smooth_polydata(poly: vtk.vtkPolyData, iterations: int,
                         relaxation: float) -> vtk.vtkPolyData:
        s = vtk.vtkSmoothPolyDataFilter()
        s.SetInputData(poly)
        s.SetNumberOfIterations(int(iterations))
        s.SetFeatureAngle(60.0)
        s.FeatureEdgeSmoothingOff()
        s.SetRelaxationFactor(float(relaxation))
        s.BoundarySmoothingOff()
        s.SetConvergence(0.0)
        s.Update()
        return s.GetOutput()

    @staticmethod
    def _negate_xy_inplace(poly: vtk.vtkPolyData) -> None:
        """Vectorised X/Y flip used by the MIRTK-orientation paths."""
        pts = poly.GetPoints()
        if pts is None or pts.GetNumberOfPoints() == 0:
            return
        arr = numpy_support.vtk_to_numpy(pts.GetData()).copy()
        arr[:, 0] *= -1.0
        arr[:, 1] *= -1.0
        new_data = numpy_support.numpy_to_vtk(arr, deep=True)
        new_pts = vtk.vtkPoints()
        new_pts.SetData(new_data)
        poly.SetPoints(new_pts)


# ---------------------------------------------------------------------------
def _discrete_cmap(hex_colors):
    from matplotlib.colors import ListedColormap
    return ListedColormap(hex_colors)


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
