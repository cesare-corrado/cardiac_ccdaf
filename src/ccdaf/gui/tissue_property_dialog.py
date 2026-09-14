"""
TissuePropertyDialog
====================
**Actions → Assign tissue property…**: the field to read, the tissue to
classify, the criterion and its rows, and the IDs written to ``tissueTag``.

The dialog carries no method of its own. It assembles a
:class:`~ccdaf.core.tissue_property.TissuePropertyOptions` and, on Apply,
runs :func:`~ccdaf.core.tissue_property.assign_tissue_property` — so a
blocking problem is reported while the inputs are still on screen to fix,
and anything worth confirming is asked before the dialog closes. Writing the
result to the mesh is the caller's job; read it back with :meth:`result`.

Threshold rows are linked: each row starts where the previous one ends, so a
gap or an overlap cannot be typed at all.

The rows are rebuilt whenever their number changes, and a rebuilt widget is
unparented **immediately** rather than left to ``deleteLater``: until the
event loop next runs, a merely scheduled widget is still a visible child of
the group, drawn at its default size over everything beneath it.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from PyQt5 import QtCore, QtWidgets

from ccdaf.core.tissue_property import (
    CRITERION_SD, CRITERION_THRESHOLDS, DEFAULT_GROWTH_LIMIT, DEFAULT_K,
    DEFAULT_SEED_PERCENTILE, DEFAULT_TOLERANCE_PERCENT, GROWTH_LIMIT_RANGE,
    MAX_REGIONS, MIN_REGIONS, POINT, SEED_PERCENTILE_RANGE, TISSUE_TAG,
    TissuePropertyOptions, TissueResult, assign_tissue_property,
    default_excluded_id, element_adjacency, element_labels, element_values,
    eligible_fields, threshold_edges,
)

#: Largest ID a spin box offers: ``tissueTag`` is written as int32.
ID_MAX = 2_147_483_647
_DECIMALS = 4
#: Widths that keep the boxes readable without stretching them across the
#: dialog. The trailing stretch column is what absorbs the spare width.
_VALUE_WIDTH = 110
_ID_WIDTH = 90


def _value_spin(minimum: float = -1.0e9) -> QtWidgets.QDoubleSpinBox:
    spin = QtWidgets.QDoubleSpinBox()
    spin.setDecimals(_DECIMALS)
    spin.setRange(minimum, 1.0e9)
    spin.setKeyboardTracking(False)
    spin.setMaximumWidth(_VALUE_WIDTH)
    return spin


class TissuePropertyDialog(QtWidgets.QDialog):

    def __init__(self, dataset, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Assign tissue property")
        self.setSizeGripEnabled(True)
        self._dataset = dataset
        self._adjacency = None
        self._result: Optional[TissueResult] = None
        self._labels = element_labels(dataset)
        self._present = sorted(int(v) for v in np.unique(self._labels))
        self._values: Dict[Tuple[str, str], np.ndarray] = {}
        self._rows: List[dict] = []
        #: Whether the opening size has been set against a real window yet.
        self._fitted = False
        #: True while this class is the one resizing, so a resize the user
        #: made is told apart from one of ours.
        self._fitting = False
        #: Once the user has sized the window, it is theirs: the dialog stops
        #: growing it and lets the scroll area show what no longer fits.
        self._user_sized = False

        # The content scrolls, so ten region rows never make the window
        # taller than the screen (and unmovable with it).
        outer = QtWidgets.QVBoxLayout(self)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        # Without this a scroll area reports a fixed, small size hint, so the
        # window opens shorter than its contents need and the groups are
        # squeezed into one another instead of the view scrolling.
        scroll.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        content = QtWidgets.QWidget()
        scroll.setWidget(content)
        outer.addWidget(scroll, 1)
        layout = QtWidgets.QVBoxLayout(content)
        self._scroll, self._content = scroll, content

        # --- field, tissue, criterion --------------------------------------
        top = QtWidgets.QGridLayout()
        top.addWidget(QtWidgets.QLabel("Field"), 0, 0)
        self.cmb_field = QtWidgets.QComboBox()
        # The field name is the one thing here that must stay readable, so the
        # combo takes the spare width and never shrinks below its contents.
        self.cmb_field.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.cmb_field.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                     QtWidgets.QSizePolicy.Fixed)
        for name, association in eligible_fields(dataset):
            self.cmb_field.addItem(f"{name}  ({association})", (name, association))
        self.cmb_field.setToolTip(
            "Scalar field the regions are read from. A point field is averaged "
            "to each element's centre over its vertices with data.")
        top.addWidget(self.cmb_field, 0, 1)
        self.lbl_range = QtWidgets.QLabel()
        self.lbl_range.setToolTip("Range of the field in the ticked tissue.")
        top.addWidget(self.lbl_range, 0, 2)

        top.addWidget(QtWidgets.QLabel("Tissue"), 1, 0)
        tissue_row = QtWidgets.QHBoxLayout()
        self.chk_tissue: Dict[int, QtWidgets.QCheckBox] = {}
        for value in self._present:
            box = QtWidgets.QCheckBox(f"elemTag {value}")
            box.setChecked(True)
            box.setToolTip(
                "Ticked: classified, and used for the statistics.\n"
                "Unticked: excluded, and written with its own ID below "
                "(a right ventricle with no LGE, for example).")
            box.toggled.connect(self._on_tissue_toggled)
            tissue_row.addWidget(box)
            self.chk_tissue[value] = box
        tissue_row.addStretch(1)
        top.addLayout(tissue_row, 1, 1, 1, 2)

        top.addWidget(QtWidgets.QLabel("Criterion"), 2, 0)
        criterion_row = QtWidgets.QHBoxLayout()
        self.rad_sd = QtWidgets.QRadioButton("Mean + k·SD")
        self.rad_sd.setToolTip(
            "Region i starts at healthy mean + k·SD and runs to the next row; "
            "the top region has no upper limit. Below the first row is healthy.")
        self.rad_thresholds = QtWidgets.QRadioButton("Thresholds")
        self.rad_thresholds.setToolTip(
            "Contiguous value ranges: each row covers from ≤ value < to, and "
            "the last row also includes its 'to'.")
        self.rad_sd.setChecked(True)
        group = QtWidgets.QButtonGroup(self)
        group.addButton(self.rad_sd)
        group.addButton(self.rad_thresholds)
        criterion_row.addWidget(self.rad_sd)
        criterion_row.addWidget(self.rad_thresholds)
        criterion_row.addSpacing(12)
        criterion_row.addWidget(QtWidgets.QLabel("Regions N"))
        self.spn_n = QtWidgets.QSpinBox()
        self.spn_n.setRange(MIN_REGIONS, MAX_REGIONS)
        self.spn_n.setValue(len(DEFAULT_K))
        self.spn_n.setMaximumWidth(_ID_WIDTH)
        self.spn_n.setToolTip("Number of region rows.")
        criterion_row.addWidget(self.spn_n)
        criterion_row.addStretch(1)
        top.addLayout(criterion_row, 2, 1, 1, 2)
        top.setColumnStretch(1, 1)
        layout.addLayout(top)

        # --- healthy reference ---------------------------------------------
        self.grp_sd = QtWidgets.QGroupBox("Healthy tissue")
        # Rows of pairs, not a grid: a grid stretches its columns when the
        # dialog widens, which walks every label away from its own box and
        # up against the next one.
        sd = QtWidgets.QVBoxLayout(self.grp_sd)
        auto_row = QtWidgets.QHBoxLayout()
        value_row = QtWidgets.QHBoxLayout()
        advanced_row = QtWidgets.QHBoxLayout()
        self.chk_auto = QtWidgets.QCheckBox("Auto (estimate healthy tissue)")
        self.chk_auto.setChecked(True)
        self.chk_auto.setToolTip(
            "Ticked: grow the healthy tissue from the lowest values and take "
            "its mean and SD.\nUnticked: type them, on this field's own scale "
            "and per element (from a remote region measured elsewhere, say).")
        auto_row.addWidget(self.chk_auto)
        auto_row.addStretch(1)
        self.btn_estimate = QtWidgets.QPushButton("Estimate")
        self.btn_estimate.setToolTip(
            "Run the healthy-tissue estimate now and show its mean and SD, "
            "without writing anything.")
        auto_row.addWidget(self.btn_estimate)
        sd.addLayout(auto_row)

        value_row.addWidget(QtWidgets.QLabel("mean"))
        self.spn_mean = _value_spin()
        self.spn_mean.setToolTip("Mean of the healthy tissue, per element.")
        value_row.addWidget(self.spn_mean)
        value_row.addSpacing(16)
        value_row.addWidget(QtWidgets.QLabel("SD"))
        self.spn_sd = _value_spin(minimum=0.0)
        self.spn_sd.setToolTip("Standard deviation of the healthy tissue, per element.")
        value_row.addWidget(self.spn_sd)
        value_row.addSpacing(16)
        value_row.addWidget(QtWidgets.QLabel("Healthy ID"))
        self.spn_healthy = QtWidgets.QSpinBox()
        self.spn_healthy.setRange(0, ID_MAX)
        self.spn_healthy.setMaximumWidth(_ID_WIDTH)
        self.spn_healthy.setToolTip(
            "ID of the classified tissue below the first row.")
        value_row.addWidget(self.spn_healthy)
        value_row.addStretch(1)
        sd.addLayout(value_row)

        self.btn_advanced = QtWidgets.QToolButton()
        self.btn_advanced.setText("Advanced")
        self.btn_advanced.setCheckable(True)
        self.btn_advanced.setArrowType(QtCore.Qt.RightArrow)
        self.btn_advanced.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.btn_advanced.setToolTip("Settings of the automatic estimate.")
        advanced_row.addWidget(self.btn_advanced)
        advanced_row.addStretch(1)
        sd.addLayout(advanced_row)

        # A grid, not a row: three labelled boxes side by side are wider than
        # the dialog and would force it to scroll sideways.
        self.wdg_advanced = QtWidgets.QWidget()
        advanced = QtWidgets.QGridLayout(self.wdg_advanced)
        advanced.setContentsMargins(16, 0, 0, 0)
        advanced.addWidget(QtWidgets.QLabel("seeds percentile"), 0, 0)
        self.spn_seed = QtWidgets.QDoubleSpinBox()
        self.spn_seed.setDecimals(1)
        self.spn_seed.setRange(*SEED_PERCENTILE_RANGE)
        self.spn_seed.setValue(DEFAULT_SEED_PERCENTILE)
        self.spn_seed.setMaximumWidth(_VALUE_WIDTH)
        self.spn_seed.setToolTip(
            "Elements at or below this weighted percentile seed the healthy "
            "tissue. It must stay below the healthy share of the tissue.")
        advanced.addWidget(self.spn_seed, 0, 1)
        advanced.addWidget(QtWidgets.QLabel("growth limit m"), 0, 2)
        self.spn_m = QtWidgets.QDoubleSpinBox()
        self.spn_m.setDecimals(2)
        self.spn_m.setSingleStep(0.5)
        self.spn_m.setRange(*GROWTH_LIMIT_RANGE)
        self.spn_m.setValue(DEFAULT_GROWTH_LIMIT)
        self.spn_m.setMaximumWidth(_VALUE_WIDTH)
        self.spn_m.setToolTip(
            "Growth keeps elements at or below mean + m·SD of the region so "
            "far. Below 3 it trims healthy tissue's own tail, and the SD "
            "comes out too small.")
        advanced.addWidget(self.spn_m, 0, 3)
        advanced.addWidget(QtWidgets.QLabel("stop below % change"), 1, 0)
        self.spn_tolerance = QtWidgets.QDoubleSpinBox()
        self.spn_tolerance.setDecimals(3)
        self.spn_tolerance.setSingleStep(0.05)
        self.spn_tolerance.setRange(0.001, 10.0)
        self.spn_tolerance.setValue(DEFAULT_TOLERANCE_PERCENT)
        self.spn_tolerance.setMaximumWidth(_VALUE_WIDTH)
        self.spn_tolerance.setToolTip(
            "Stop when a round changes the region by less than this share of "
            "the tissue.")
        advanced.addWidget(self.spn_tolerance, 1, 1)
        advanced.setColumnStretch(4, 1)
        self.wdg_advanced.setVisible(False)
        sd.addWidget(self.wdg_advanced)

        self.lbl_estimate = QtWidgets.QLabel()
        self.lbl_estimate.setWordWrap(True)
        sd.addWidget(self.lbl_estimate)
        layout.addWidget(self.grp_sd)

        # --- region rows -----------------------------------------------------
        self.grp_rows = QtWidgets.QGroupBox("Regions")
        rows_layout = QtWidgets.QVBoxLayout(self.grp_rows)
        self.grid_rows = QtWidgets.QGridLayout()
        self._headers: Dict[int, QtWidgets.QLabel] = {}
        for column, text in enumerate(("Region", "ID", "k (SD above healthy mean)",
                                       "from", "to")):
            header = QtWidgets.QLabel(f"<b>{text}</b>")
            self.grid_rows.addWidget(header, 0, column)
            self._headers[column] = header
        self.grid_rows.setColumnStretch(5, 1)          # absorbs the spare width
        rows_layout.addLayout(self.grid_rows)
        self.lbl_thresholds = QtWidgets.QLabel(
            "Each row starts where the previous one ends. The first 'from' may "
            "sit below the data and the last 'to' above it; a row with no data "
            "in it simply writes no elements.")
        self.lbl_thresholds.setWordWrap(True)
        rows_layout.addWidget(self.lbl_thresholds)
        layout.addWidget(self.grp_rows)

        # --- excluded tissue -------------------------------------------------
        self.grp_excluded = QtWidgets.QGroupBox("Excluded tissue")
        self.grid_excluded = QtWidgets.QGridLayout(self.grp_excluded)
        self.grid_excluded.setColumnStretch(2, 1)
        self._excluded_spins: Dict[int, QtWidgets.QSpinBox] = {}
        self._excluded_labels: Dict[int, QtWidgets.QLabel] = {}
        layout.addWidget(self.grp_excluded)
        layout.addStretch(1)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self.btn_apply = buttons.button(QtWidgets.QDialogButtonBox.Ok)
        self.btn_apply.setText("Apply")
        self.btn_apply.setToolTip(f"Compute and write the {TISSUE_TAG} cell array.")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

        self.cmb_field.currentIndexChanged.connect(self._on_data_changed)
        self.rad_sd.toggled.connect(self._on_criterion_toggled)
        self.spn_n.valueChanged.connect(self._on_n_changed)
        self.chk_auto.toggled.connect(self._sync_auto)
        self.btn_advanced.toggled.connect(self._on_advanced_toggled)
        self.btn_estimate.clicked.connect(self._estimate)

        self._build_rows(self.spn_n.value())
        self._on_data_changed()
        self._sync_criterion()
        self._sync_auto()
        self._sync_excluded()
        self._fit_to_screen()

    def _fit_to_screen(self) -> None:
        """Grow to the size the contents ask for, capped by the screen.

        The size comes from the *content* widget, plus the chrome measured
        between it and the window. A scroll area's own ``sizeHint`` does not
        follow its widget — it stays at its initial value however much the
        content grows — so sizing the window from it opens the dialog too
        short for what it holds, with the groups behind a scrollbar.

        The cap keeps the window from opening taller than the display, which
        would leave it with no reachable title bar and no way to move or
        resize it; past the cap the scroll area takes over. It only ever
        grows, so a size the user chose is never taken back.
        """
        if self._user_sized:
            return
        self._content.adjustSize()
        layout = self.layout()
        if layout is not None:
            layout.activate()
        viewport = self._scroll.viewport()
        chrome = QtCore.QSize(max(self.width() - viewport.width(), 24),
                              max(self.height() - viewport.height(), 24))
        hint = self._content.sizeHint()
        wanted = QtCore.QSize(hint.width() + chrome.width(),
                              hint.height() + chrome.height())
        wanted = wanted.expandedTo(QtCore.QSize(560, 0)).expandedTo(self.size())
        screen = QtWidgets.QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            wanted = QtCore.QSize(min(wanted.width(), int(available.width() * 0.9)),
                                  min(wanted.height(), int(available.height() * 0.9)))
        self._fitting = True
        try:
            self.resize(wanted)
        finally:
            self._fitting = False
        self._keep_on_screen()

    def _keep_on_screen(self) -> None:
        """Move the whole window back inside the screen after it has grown.

        A window whose title bar has been pushed above the top of the screen
        can still be resized, but there is nothing left to grab to move it.

        Nothing is moved until the window manager has told Qt how big the
        frame is. Before the window is mapped, ``frameGeometry`` equals
        ``geometry`` — margins of zero — and clamping against that would
        place the title bar itself off the top of the screen, causing the
        very problem this guards against.
        """
        screen = QtWidgets.QApplication.primaryScreen()
        if screen is None:
            return
        frame = self.frameGeometry()
        if frame == self.geometry():            # frame margins not known yet
            QtCore.QTimer.singleShot(0, self._keep_on_screen)
            return
        available = screen.availableGeometry()
        x = min(max(frame.x(), available.left()),
                max(available.right() - frame.width() + 1, available.left()))
        y = min(max(frame.y(), available.top()),
                max(available.bottom() - frame.height() + 1, available.top()))
        if (x, y) != (frame.x(), frame.y()):
            self.move(self.pos() + QtCore.QPoint(x - frame.x(), y - frame.y()))

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        """A resize the user made hands the window's size over to them."""
        super().resizeEvent(event)
        if self._fitted and not self._fitting:
            self._user_sized = True

    def showEvent(self, event) -> None:  # type: ignore[override]
        """Fit once more as the dialog appears.

        Only then do the window and its viewport have real sizes, so this is
        the first moment the chrome between them can be measured.
        """
        super().showEvent(event)
        if not self._fitted:
            self._fitted = True
            self._fit_to_screen()

    # -- rows -------------------------------------------------------------
    def _build_rows(self, n: int, edges: Optional[Tuple[float, ...]] = None) -> None:
        """(Re)create *n* rows, keeping the IDs and k values already typed."""
        ids = [row["id"].value() for row in self._rows]
        ks = [row["k"].value() for row in self._rows]
        for row in self._rows:
            for widget in (row["label"], row["id"], row["k"], row["from"], row["to"]):
                self.grid_rows.removeWidget(widget)
                widget.setParent(None)      # off screen now, not at idle
                widget.deleteLater()
        self._rows = []
        if edges is None:
            edges = threshold_edges(0.0, 1.0, n)
        for i in range(n):
            label = QtWidgets.QLabel(str(i + 1))
            spin_id = QtWidgets.QSpinBox()
            spin_id.setRange(0, ID_MAX)
            spin_id.setMaximumWidth(_ID_WIDTH)
            spin_id.setValue(ids[i] if i < len(ids) else i + 1)
            spin_id.setToolTip("ID written to tissueTag for this region.")
            spin_k = QtWidgets.QDoubleSpinBox()
            spin_k.setDecimals(2)
            spin_k.setSingleStep(0.5)
            spin_k.setRange(0.01, 100.0)
            spin_k.setMaximumWidth(_VALUE_WIDTH)
            if i < len(ks):
                spin_k.setValue(ks[i])
            elif i < len(DEFAULT_K):
                spin_k.setValue(DEFAULT_K[i])
            else:
                spin_k.setValue((ks[-1] if ks else DEFAULT_K[-1]) + 1.0 * (i - len(ks) + 1))
            spin_k.setToolTip("The region starts at healthy mean + k·SD.")
            if i == 0:
                frm = _value_spin()
                frm.setValue(edges[0])
                frm.setToolTip("Lower edge of the first row; it may sit below the data.")
            else:
                frm = QtWidgets.QLabel()
                frm.setMinimumWidth(_VALUE_WIDTH)
                frm.setToolTip("Equal to the previous row's 'to'.")
            spin_to = _value_spin()
            spin_to.setValue(edges[i + 1])
            spin_to.setToolTip("Upper edge of this row; the next row starts here.")
            for column, widget in enumerate((label, spin_id, spin_k, frm, spin_to)):
                self.grid_rows.addWidget(widget, i + 1, column)
            self._rows.append({"label": label, "id": spin_id, "k": spin_k,
                               "from": frm, "to": spin_to})
            spin_to.valueChanged.connect(self._sync_links)
        self._sync_links()
        self._sync_criterion()

    def _sync_links(self, *_args) -> None:
        for previous, row in zip(self._rows[:-1], self._rows[1:]):
            row["from"].setText(f"{previous['to'].value():.{_DECIMALS}f}")

    def _edges(self) -> Tuple[float, ...]:
        if not self._rows:
            return ()
        return ((self._rows[0]["from"].value(),)
                + tuple(row["to"].value() for row in self._rows))

    def _on_n_changed(self, n: int) -> None:
        edges = self._edges()
        lo, hi = (edges[0], edges[-1]) if edges else (0.0, 1.0)
        self._build_rows(int(n), threshold_edges(lo, hi, int(n)))
        self._fit_to_screen()

    # -- data ---------------------------------------------------------------
    def _field(self) -> Tuple[str, str]:
        data = self.cmb_field.currentData()
        return (str(data[0]), str(data[1])) if data else ("", POINT)

    def _ticked(self) -> Tuple[int, ...]:
        return tuple(v for v, box in self.chk_tissue.items() if box.isChecked())

    def _field_values(self) -> Optional[np.ndarray]:
        name, association = self._field()
        if not name:
            return None
        key = (name, association)
        if key not in self._values:
            self._values[key] = element_values(self._dataset, name, association)
        return self._values[key]

    def _tissue_range(self) -> Optional[Tuple[float, float]]:
        values = self._field_values()
        if values is None:
            return None
        chosen = np.isin(self._labels, list(self._ticked())) & np.isfinite(values)
        if not chosen.any():
            return None
        return float(values[chosen].min()), float(values[chosen].max())

    def _on_data_changed(self, *_args) -> None:
        span = self._tissue_range()
        if span is None:
            self.lbl_range.setText("no values in the ticked tissue")
            return
        lo, hi = span
        self.lbl_range.setText(f"range {lo:.4g} to {hi:.4g}")
        n = len(self._rows) or self.spn_n.value()
        self._build_rows(n, threshold_edges(lo, hi, n))

    def _on_tissue_toggled(self, *_args) -> None:
        self._sync_excluded()
        self._on_data_changed()
        self._fit_to_screen()

    # -- excluded tissue ---------------------------------------------------
    def _used_ids(self, skip: Optional[int] = None) -> List[int]:
        used = [row["id"].value() for row in self._rows]
        if self.rad_sd.isChecked():
            used.append(self.spn_healthy.value())
        used += [spin.value() for value, spin in self._excluded_spins.items()
                 if value != skip and not self.chk_tissue[value].isChecked()]
        return used

    def _sync_excluded(self) -> None:
        for value, box in self.chk_tissue.items():
            excluded = not box.isChecked()
            if excluded and value not in self._excluded_spins:
                label = QtWidgets.QLabel(f"elemTag {value} → ID")
                spin = QtWidgets.QSpinBox()
                spin.setRange(0, ID_MAX)
                spin.setMaximumWidth(_ID_WIDTH)
                spin.setValue(default_excluded_id(value, self._used_ids(skip=value)))
                spin.setToolTip(
                    "ID written for this excluded tissue. It keeps its elemTag "
                    "number unless that number is already used.")
                row = len(self._excluded_spins)
                self.grid_excluded.addWidget(label, row, 0)
                self.grid_excluded.addWidget(spin, row, 1)
                self._excluded_labels[value] = label
                self._excluded_spins[value] = spin
            if value in self._excluded_spins:
                self._excluded_labels[value].setVisible(excluded)
                self._excluded_spins[value].setVisible(excluded)
        self.grp_excluded.setVisible(
            any(not box.isChecked() for box in self.chk_tissue.values()))

    # -- criterion ---------------------------------------------------------
    def _sync_criterion(self, *_args) -> None:
        sd = self.rad_sd.isChecked()
        self.grp_sd.setVisible(sd)
        self.lbl_thresholds.setVisible(not sd)
        for column, visible in ((2, sd), (3, not sd), (4, not sd)):
            self._headers[column].setVisible(visible)
            key = {2: "k", 3: "from", 4: "to"}[column]
            for row in self._rows:
                row[key].setVisible(visible)

    def _sync_auto(self, *_args) -> None:
        auto = self.chk_auto.isChecked()
        for spin in (self.spn_mean, self.spn_sd):
            spin.setReadOnly(auto)
        self.btn_estimate.setEnabled(auto)
        self.btn_advanced.setEnabled(auto)
        self.wdg_advanced.setVisible(auto and self.btn_advanced.isChecked())

    def _on_criterion_toggled(self, *_args) -> None:
        self._sync_criterion()
        self._fit_to_screen()

    def _on_advanced_toggled(self, on: bool) -> None:
        self.btn_advanced.setArrowType(QtCore.Qt.DownArrow if on else QtCore.Qt.RightArrow)
        self.wdg_advanced.setVisible(bool(on) and self.chk_auto.isChecked())
        self._fit_to_screen()

    # -- options and running ---------------------------------------------
    def options(self) -> TissuePropertyOptions:
        """The dialog's current inputs."""
        name, association = self._field()
        ticked = self._ticked()
        return TissuePropertyOptions(
            field=name, association=association, tissue=ticked,
            excluded_ids={v: spin.value() for v, spin in self._excluded_spins.items()
                          if v not in ticked},
            criterion=CRITERION_SD if self.rad_sd.isChecked() else CRITERION_THRESHOLDS,
            region_ids=tuple(row["id"].value() for row in self._rows),
            ks=tuple(row["k"].value() for row in self._rows),
            edges=self._edges(),
            healthy_id=self.spn_healthy.value(),
            auto=self.chk_auto.isChecked(),
            mean=self.spn_mean.value(), sd=self.spn_sd.value(),
            seed_percentile=self.spn_seed.value(),
            growth_limit=self.spn_m.value(),
            tolerance_percent=self.spn_tolerance.value(),
        )

    def result(self) -> Optional[TissueResult]:
        """What Apply computed, or ``None`` if the dialog was cancelled."""
        return self._result

    def _run(self, options: TissuePropertyOptions) -> TissueResult:
        values = self._field_values()
        tissue = np.isin(self._labels, list(options.tissue))
        needs_graph = ((options.criterion == CRITERION_SD and options.auto)
                       or (values is not None and np.isnan(values[tissue]).any()))
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            if needs_graph and self._adjacency is None:
                self._adjacency = element_adjacency(self._dataset)
            return assign_tissue_property(self._dataset, options,
                                          adjacency=self._adjacency)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _show_estimate(self, result: TissueResult) -> None:
        if result.healthy is None:
            return
        for spin, value in ((self.spn_mean, result.mean), (self.spn_sd, result.sd)):
            spin.blockSignals(True)
            spin.setValue(float(value))
            spin.blockSignals(False)
        settled = "" if result.healthy.settled else ", not settled"
        self.lbl_estimate.setText(
            f"Healthy: {result.healthy_fraction:.1%} of the ticked tissue, "
            f"{result.healthy.rounds} rounds{settled}.")

    def _estimate(self) -> None:
        options = self.options()
        options.criterion = CRITERION_SD
        options.auto = True
        try:
            result = self._run(options)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Cannot estimate", str(exc))
            return
        self._show_estimate(result)

    def accept(self) -> None:  # type: ignore[override]
        options = self.options()
        try:
            options.validate()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Cannot assign tissue property", str(exc))
            return
        if TISSUE_TAG in self._dataset.cell_data:
            reply = QtWidgets.QMessageBox.question(
                self, f"Replace {TISSUE_TAG}?",
                f"The mesh already has a {TISSUE_TAG} array. Replace it?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No)
            if reply != QtWidgets.QMessageBox.Yes:
                return
        try:
            result = self._run(options)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Cannot assign tissue property", str(exc))
            return
        if options.criterion == CRITERION_SD and options.auto:
            self._show_estimate(result)
        if result.warnings:
            text = "\n\n".join(f"• {note}" for note in result.warnings)
            reply = QtWidgets.QMessageBox.question(
                self, "Check before writing",
                f"{text}\n\nWrite {TISSUE_TAG} anyway?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No)
            if reply != QtWidgets.QMessageBox.Yes:
                return
        self._result = result
        super().accept()


__all__ = ["TissuePropertyDialog", "ID_MAX"]
