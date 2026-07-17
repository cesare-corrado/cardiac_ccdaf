"""
PostprocessingWidget
====================

Side-panel widget exposing ``mesh_postprocessor.apply`` to the user:

* three checkboxes (decimate, refine, clean) with per-step parameters
* an "Apply" button that runs the selected steps in the fixed order
  ``decimate -> refine -> clean`` and swaps the working mesh in-place.

The widget is kept intentionally decoupled from the rest of the GUI and
only interacts with the host via two callables: ``mesh_getter`` and
``mesh_setter``.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pyvista as pv
from PyQt5 import QtCore, QtWidgets

from ccdaf.core.mesh_postprocessor import (
    PostprocessOptions, apply as postprocess_apply,
    SMOOTH_TAUBIN, SMOOTH_LAPLACIAN,
)


class PostprocessingWidget(QtWidgets.QGroupBox):

    mesh_changed = QtCore.pyqtSignal()

    def __init__(self,
                 mesh_getter: Callable[[], Optional[pv.PolyData]],
                 mesh_setter: Callable[[pv.PolyData], None],
                 on_status: Optional[Callable[[str], None]] = None,
                 on_surface_moved: Optional[Callable[[object, object], None]] = None,
                 parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__("Mesh post-processing", parent)
        self._get = mesh_getter
        self._set = mesh_setter
        self._status = on_status or (lambda msg: None)
        # Fired with (old_mesh, new_mesh) when smoothing moves the surface,
        # so the app can carry EAM electrodes along with it. The widget
        # itself knows nothing about electrodes.
        self._on_surface_moved = on_surface_moved

        self.setToolTip(
            "Mesh post-processing: decimate, refine and/or clean the mesh. "
            "Steps run in fixed order (decimate → refine → clean)."
        )

        v = QtWidgets.QVBoxLayout(self)

        # -- decimate -------------------------------------------------
        self.chk_decimate = QtWidgets.QCheckBox("Decimate")
        self.chk_decimate.setToolTip(
            "Decimate the number of mesh point by annealing"
        )
        v.addWidget(self.chk_decimate)
        grid = QtWidgets.QGridLayout()
        lbl_target = QtWidgets.QLabel("target points")
        lbl_target.setToolTip("Target number of mesh points")
        grid.addWidget(lbl_target, 0, 0)
        self.spn_target = QtWidgets.QSpinBox()
        self.spn_target.setRange(100, 10_000_000)
        self.spn_target.setValue(5000)
        self.spn_target.setSingleStep(500)
        self.spn_target.setToolTip("Target number of mesh points")
        #grid.addWidget(self.spn_target, 0, 1)
        grid.addWidget(self.spn_target, 1, 0)
        lbl_iters = QtWidgets.QLabel("anneal iters")
        lbl_iters.setToolTip(
            "Maximum number of simulated-annealing iterations used to "
            "redistribute the decimated vertices."
        )
        #grid.addWidget(lbl_iters, 1, 0)
        grid.addWidget(lbl_iters, 0, 1)
        self.spn_iters = QtWidgets.QSpinBox()
        self.spn_iters.setRange(1, 10_000_000)
        self.spn_iters.setValue(200)
        self.spn_iters.setSingleStep(100)
        self.spn_iters.setToolTip(
            "Maximum number of simulated-annealing iterations used to "
            "redistribute the decimated vertices."
        )
        #grid.addWidget(self.spn_iters, 1, 1)
        grid.addWidget(self.spn_iters, 1, 1)
        v.addLayout(grid)

        # -- refine ---------------------------------------------------
        self.chk_refine = QtWidgets.QCheckBox("Refine")
        self.chk_refine.setToolTip(
            "Adaptively subdivide triangles whose longest edge exceeds "
            "the target edge length."
        )
        v.addWidget(self.chk_refine)
        row = QtWidgets.QHBoxLayout()
        lbl_edge = QtWidgets.QLabel("edge len")
        lbl_edge.setToolTip(
            "Target maximum edge length. Any triangle with a longer edge "
            "is split. 0 = use the current median edge length of the mesh."
        )
        row.addWidget(lbl_edge)
        self.spn_edge = QtWidgets.QDoubleSpinBox()
        self.spn_edge.setDecimals(4)
        self.spn_edge.setRange(0.0, 1.0e6)
        self.spn_edge.setValue(0.4)
        self.spn_edge.setSingleStep(0.1)
        self.spn_edge.setToolTip(
            "Target maximum edge length. Any triangle with a longer edge "
            "is split. 0 = use the current median edge length of the mesh."
        )
        row.addWidget(self.spn_edge, 1)
        v.addLayout(row)

        # -- clean ----------------------------------------------------
        self.chk_clean = QtWidgets.QCheckBox("Clean")
        self.chk_clean.setToolTip(
            "Merge duplicate points, drop non-connected points, remove "
            "non-manifold and degenerate cells, orient normals, and "
            "smooth low-quality triangles while preserving listed labels."
        )
        v.addWidget(self.chk_clean)
        
        grid = QtWidgets.QGridLayout()
        lbl_quality = QtWidgets.QLabel("quality threshold")
        lbl_quality.setToolTip(
            "Triangles with radius-ratio quality below this value are "
            "smoothed. 1.0 = equilateral, 0.0 = disables smoothing."
        )
        grid.addWidget(lbl_quality, 0, 0)
        
        self.spn_quality = QtWidgets.QDoubleSpinBox()
        self.spn_quality.setRange(0.0, 1.0)
        self.spn_quality.setSingleStep(0.05)
        self.spn_quality.setValue(0.2)
        self.spn_quality.setToolTip(
            "Triangles with radius-ratio quality below this value are "
            "smoothed. 1.0 = equilateral, 0.0 = disables smoothing."
        )
        grid.addWidget(self.spn_quality, 1, 0)
        
        lbl_smooth = QtWidgets.QLabel("smooth iters")
        lbl_smooth.setToolTip(
            "Maximum Laplacian-smoothing sweeps over bad-triangle "
            "vertices. Loop exits early when no triangle is below the "
            "quality threshold."
        )
        grid.addWidget(lbl_smooth, 0, 1)

        self.spn_smooth = QtWidgets.QSpinBox()
        self.spn_smooth.setRange(0, 1000)
        self.spn_smooth.setValue(20)
        self.spn_smooth.setToolTip(
            "Maximum Laplacian-smoothing sweeps over bad-triangle "
            "vertices. Loop exits early when no triangle is below the "
            "quality threshold."
        )
        grid.addWidget(self.spn_smooth, 1, 1)

        lbl_realx = QtWidgets.QLabel("Relaxation factor")
        lbl_realx.setToolTip(
            "Relaxation factor to move points "
            "fraction of the distance from the mean point."
        )
        grid.addWidget(lbl_realx, 0, 2)

        self.spn_smooth_relax = QtWidgets.QDoubleSpinBox()
        self.spn_smooth_relax.setRange(0.0, 1.0)
        self.spn_smooth_relax.setSingleStep(0.05)
        self.spn_smooth_relax.setDecimals(3)
        self.spn_smooth_relax.setValue(0.1)
        self.spn_smooth_relax.setToolTip(
            "Relaxation factor to move points "
            "fraction of the distance from the mean point."
        )
        grid.addWidget(self.spn_smooth_relax, 1, 2)

        lbl_preserve = QtWidgets.QLabel("preserve labels")
        lbl_preserve.setToolTip(
            "Comma-separated elemTag values whose cells define protected "
            "surfaces. Vertices on these cells are frozen during smoothing."
        )
        grid.addWidget(lbl_preserve, 0, 3)
        self.txt_preserve = QtWidgets.QLineEdit()
        self.txt_preserve.setPlaceholderText("e.g. 11,13,15,17,19")
        self.txt_preserve.setToolTip(
            "Comma-separated elemTag values whose cells define protected "
            "surfaces. Vertices on these cells are frozen during smoothing."
        )
        grid.addWidget(self.txt_preserve, 1, 3)
        v.addLayout(grid)

        # -- fill holes -----------------------------------------------
        _HOLE_TIP = (
            "Close holes whose bounding-sphere radius is ≤ this value. Set "
            "well below the mitral-valve / PV-ostia radius so those "
            "anatomical openings stay open. 0 disables hole filling. "
            "Decimation's own hole-filling pass uses the same value."
        )
        self.chk_fill = QtWidgets.QCheckBox("Fill holes")
        self.chk_fill.setToolTip(
            "Close the remaining holes below 'max hole size'. Runs last: "
            "cleaning drops non-manifold cells and so opens holes of its own."
        )
        v.addWidget(self.chk_fill)

        grid = QtWidgets.QGridLayout()
        lbl_hole = QtWidgets.QLabel("max hole size")
        lbl_hole.setToolTip(_HOLE_TIP)
        grid.addWidget(lbl_hole, 0, 0)
        self.spn_hole = QtWidgets.QDoubleSpinBox()
        self.spn_hole.setDecimals(4)
        self.spn_hole.setRange(0.0, 1.0e6)
        self.spn_hole.setValue(4.0)
        self.spn_hole.setSingleStep(0.5)
        self.spn_hole.setToolTip(_HOLE_TIP)
        grid.addWidget(self.spn_hole, 0, 1)
        grid.setColumnStretch(2, 1)
        v.addLayout(grid)

        # -- smooth ---------------------------------------------------
        self.chk_smooth = QtWidgets.QCheckBox("Smooth")
        self.chk_smooth.setToolTip(
            "Smooth the whole surface to strip acquisition noise. This is a "
            "different job from Clean's smoothing, which only repairs badly-"
            "shaped triangles and leaves the anatomy where it is.\n\n"
            "Runs last, on the final topology. EAM electrodes are carried "
            "along with the surface."
        )
        v.addWidget(self.chk_smooth)

        grid = QtWidgets.QGridLayout()
        _METHOD_TIP = (
            "Taubin preserves the enclosed volume (~+0.1% on a Carto atrium); "
            "Laplacian shrinks the shell progressively with iterations "
            "(~-2.5% at 100), which matters if the shell is later measured."
        )
        lbl_method = QtWidgets.QLabel("method")
        lbl_method.setToolTip(_METHOD_TIP)
        grid.addWidget(lbl_method, 0, 0)
        self.cmb_smooth = QtWidgets.QComboBox()
        self.cmb_smooth.addItem("Taubin", SMOOTH_TAUBIN)
        self.cmb_smooth.addItem("Laplacian", SMOOTH_LAPLACIAN)
        self.cmb_smooth.setToolTip(_METHOD_TIP)
        grid.addWidget(self.cmb_smooth, 1, 0)

        _ITER_TIP = (
            "Number of smoothing sweeps over the surface.\n\n"
            "40 suits a dense surface (a Carto map is ~13k points, where "
            "Taubin holds volume to +0.1%). On a coarser mesh the same count "
            "over-smooths — decimated to 4k it loses ~4% of the volume — so "
            "lower it when smoothing after decimation."
        )
        lbl_iters = QtWidgets.QLabel("iterations")
        lbl_iters.setToolTip(_ITER_TIP)
        grid.addWidget(lbl_iters, 0, 1)
        self.spn_smooth_iters = QtWidgets.QSpinBox()
        self.spn_smooth_iters.setRange(1, 1000)
        self.spn_smooth_iters.setValue(40)
        self.spn_smooth_iters.setToolTip(_ITER_TIP)
        grid.addWidget(self.spn_smooth_iters, 1, 1)

        self.lbl_smooth_param = QtWidgets.QLabel("passband")
        grid.addWidget(self.lbl_smooth_param, 0, 2)
        self.spn_smooth_param = QtWidgets.QDoubleSpinBox()
        self.spn_smooth_param.setDecimals(4)
        self.spn_smooth_param.setRange(0.0001, 1.0)
        self.spn_smooth_param.setSingleStep(0.001)
        self.spn_smooth_param.setValue(0.001)
        grid.addWidget(self.spn_smooth_param, 1, 2)
        grid.setColumnStretch(3, 1)
        v.addLayout(grid)
        self.cmb_smooth.currentIndexChanged.connect(self._sync_smooth_param)
        self._sync_smooth_param()

        # -- apply ----------------------------------------------------
        self.btn_apply = QtWidgets.QPushButton("Apply post-processing")
        self.btn_apply.setToolTip(
            "Run the selected steps on the current mesh in the order "
            "decimate → refine → clean → fill holes."
        )
        self.btn_apply.clicked.connect(self._on_apply)
        v.addWidget(self.btn_apply)

        # Progress bar — shown only during the simulated-annealing
        # outer loop of the decimate step.
        self.progress = QtWidgets.QProgressBar()
        self.progress.setTextVisible(True)
        self.progress.setFormat("annealing %v / %m")
        self.progress.setVisible(False)
        v.addWidget(self.progress)

    # -----------------------------------------------------------------
    def _parse_preserve(self) -> tuple[int, ...]:
        text = self.txt_preserve.text().strip()
        if not text:
            return ()
        out: list[int] = []
        for tok in text.replace(";", ",").split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                out.append(int(tok))
            except ValueError:
                raise ValueError(f"invalid label: {tok!r}")
        return tuple(out)

    def _sync_smooth_param(self, *_args) -> None:
        """The second smoothing knob means different things per method:
        Taubin filters by passband, Laplacian steps by a relaxation factor."""
        taubin = self.cmb_smooth.currentData() == SMOOTH_TAUBIN
        self.lbl_smooth_param.setText("passband" if taubin else "relaxation")
        tip = ("Taubin passband: smaller is smoother (0.001 is a usual choice)."
               if taubin else
               "Laplacian relaxation: how far a vertex moves toward its "
               "neighbours' average each iteration.")
        self.spn_smooth_param.setToolTip(tip)
        self.lbl_smooth_param.setToolTip(tip)
        self.spn_smooth_param.blockSignals(True)
        self.spn_smooth_param.setValue(0.001 if taubin else 0.1)
        self.spn_smooth_param.blockSignals(False)

    def _on_apply(self) -> None:
        mesh = self._get()
        if mesh is None:
            QtWidgets.QMessageBox.information(
                self, "No mesh", "Load a mesh before running post-processing."
            )
            return
        if not (self.chk_decimate.isChecked()
                or self.chk_refine.isChecked()
                or self.chk_clean.isChecked()
                or self.chk_fill.isChecked()
                or self.chk_smooth.isChecked()):
            QtWidgets.QMessageBox.information(
                self, "Nothing to do",
                "Select at least one of decimate / refine / clean / "
                "fill holes / smooth.",
            )
            return

        try:
            preserve = self._parse_preserve()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid labels", str(exc))
            return

        opts = PostprocessOptions(
            do_decimate=self.chk_decimate.isChecked(),
            do_refine=self.chk_refine.isChecked(),
            do_clean=self.chk_clean.isChecked(),
            do_fill_holes=self.chk_fill.isChecked(),
            do_smooth=self.chk_smooth.isChecked(),
            smooth_method=self.cmb_smooth.currentData(),
            smooth_iterations=int(self.spn_smooth_iters.value()),
            smooth_passband=float(self.spn_smooth_param.value()),
            smooth_relaxation=float(self.spn_smooth_param.value()),
            decimate_target_points=int(self.spn_target.value()),
            decimate_iters=int(self.spn_iters.value()),
            max_hole_size=float(self.spn_hole.value()),
            refine_edge_len=float(self.spn_edge.value()),
            clean_quality_threshold=float(self.spn_quality.value()),
            clean_smooth_iterations=int(self.spn_smooth.value()),
            clean_quality_relaxation=float(self.spn_smooth_relax.value()),
            clean_preserve_labels=preserve,
        )

        self._status("Running mesh post-processing…")
        QtWidgets.QApplication.processEvents()
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)

        def _decimate_progress(i: int, n: int) -> None:
            if not self.progress.isVisible():
                self.progress.setRange(0, max(n, 1))
                self.progress.setVisible(True)
            # if the range changed between outer calls (e.g. user edits
            # n_iters mid-run), keep maximum up to date.
            if self.progress.maximum() != n:
                self.progress.setRange(0, max(n, 1))
            self.progress.setValue(i)
            QtWidgets.QApplication.processEvents()

        # How far smoothing actually moved the wall. Reported because the
        # damage is invisible on screen and depends on the mesh: 40 Taubin
        # sweeps barely touch a dense Carto map but round off a decimated
        # one. A number the user can read beats a threshold we guess at.
        moved: dict = {}

        def _surface_moved(old, new) -> None:
            d = np.linalg.norm(
                np.asarray(new.points, dtype=float)
                - np.asarray(old.points, dtype=float), axis=1)
            if d.size:
                moved["mean"] = float(d.mean())
                moved["max"] = float(d.max())
            if self._on_surface_moved is not None:
                self._on_surface_moved(old, new)

        try:
            new_mesh = postprocess_apply(
                mesh, opts, on_decimate_progress=_decimate_progress,
                on_surface_moved=_surface_moved,
            )
        except Exception as exc:
            self.progress.setVisible(False)
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.critical(
                self, "Post-processing failed", str(exc)
            )
            self._status("Post-processing failed.")
            return
        self.progress.setVisible(False)
        QtWidgets.QApplication.restoreOverrideCursor()

        self._set(new_mesh)
        self.mesh_changed.emit()
        msg = (f"Post-processing done: {new_mesh.n_points} points, "
               f"{new_mesh.n_cells} cells.")
        if moved:
            msg += (f" Smoothing moved the surface {moved['mean']:.3g} on "
                    f"average, {moved['max']:.3g} at most.")
        self._status(msg)


__all__ = ["PostprocessingWidget"]
