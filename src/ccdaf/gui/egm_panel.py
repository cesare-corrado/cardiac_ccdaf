"""
EGMPanel
========
The right-hand panel of the ``egm`` view: the electrograms of one catheter
placement, drawn with matplotlib.

Layout, top to bottom:

* a control row — **Pick electrode** (the 3D picker), **Show bipole** (the
  picked bipole's direction arrow in 3D), one tick per trace type
  (Reference, Bipolar, Unipolar, Extra) with its own **−** / **+** gain, and
  **WOI** / **Full** to reset the time window;
* the matplotlib navigation toolbar (zoom, pan);
* the **reference** trace, pinned so it never scrolls away. It carries the
  time axis: ms, with 0 at the reference annotation;
* the other traces in a scroll area, one fixed-height row each: one group per
  bipole (bipolar, then its two unipoles), then the extra channels.

Each trace type has one vertical scale shared by all its rows, so amplitudes
compare by eye (a low-voltage bipole looks small); the gain buttons scale a
whole type. The window of interest is shaded, 0 ms is a dashed line, and a
bipole that is a point carries a tick at its LAT.

The selected bipole's rows get a shaded background, and the panel scrolls to
them. Clicking a row of another bipole emits :attr:`bipole_clicked`, so the
selection can be driven from here or from the 3D view.

The panel holds no selection logic of its own beyond that: the host calls
:meth:`set_egm` once, then :meth:`show_bipole` for each selection.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PyQt5 import QtCore, QtWidgets

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure


# Trace types, in the order of the control row.
REF, BIP, UNI, EXTRA = "Reference", "Bipolar", "Unipolar", "Extra"
TRACE_TYPES = (REF, BIP, UNI, EXTRA)

ROW_PX = 42            # height of one trace row, in pixels
REF_PX = 90            # height of the pinned reference strip (row + time axis)
TIME_AXIS_PX = 34      # of which the time axis labels take this much
LABEL_PX = 110         # room for the row labels on the left
WOI_MARGIN_MS = 100.0  # the default window shows this much either side of the WOI
GAIN_STEP = 1.5        # one press of + or -
# A type's base scale maps this percentile of its |amplitude| to this much of
# a row, so a typical deflection is clearly visible without most rows
# overlapping their neighbours.
SCALE_PERCENTILE = 99.5
SCALE_ROW_FRACTION = 0.45
SCALE_PLACEMENTS = 60      # placements sampled for the scale
SCALE_SAMPLE_STEP = 2      # every 2nd ms: deflections last several ms

# Colours. Two highlight shades: on a dark window background a pale yellow
# band would glare and wash the traces out.
TRACE_COLOR = {REF: "#1f4e9a", BIP: "#1b1b1b", UNI: "#4a6f2a", EXTRA: "#7a4a9a"}
TRACE_COLOR_DARK = {REF: "#8fb4ff", BIP: "#f0f0f0", UNI: "#b8d98a", EXTRA: "#d6b0f0"}
HIGHLIGHT_LIGHT = "#fff1a8"
HIGHLIGHT_DARK = "#3d3d2a"
WOI_COLOR = "#9aa0a6"
LAT_COLOR = "#d62728"


@dataclass
class _Row:
    kind: str                    # BIP, UNI or EXTRA
    column: int                  # trace column
    label: str
    bipole: Optional[int]        # the bipole group the row belongs to


def _is_dark(widget: QtWidgets.QWidget) -> bool:
    return widget.palette().color(widget.backgroundRole()).lightness() < 128


class EGMPanel(QtWidgets.QWidget):
    """Electrograms of one placement; see the module docstring."""

    pick_toggled = QtCore.pyqtSignal(bool)
    bipole_clicked = QtCore.pyqtSignal(int)
    direction_toggled = QtCore.pyqtSignal(bool)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._egm: Optional[dict] = None
        self._lat: Dict[int, float] = {}          # _car.txt row -> LAT (ms)
        self._catheter: Optional[dict] = None
        self._placement: Optional[int] = None
        self._selected: Optional[int] = None
        self._base_scale: Dict[str, float] = {t: 1.0 for t in TRACE_TYPES}
        self._gain: Dict[str, float] = {t: 1.0 for t in TRACE_TYPES}
        self._rows: List[_Row] = []
        self._xlim: Optional[Tuple[float, float]] = None
        self._syncing = False

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(2)

        # --- control row ------------------------------------------------
        ctl = QtWidgets.QHBoxLayout()
        ctl.setSpacing(4)
        self.btn_pick = QtWidgets.QToolButton()
        self.btn_pick.setText("Pick electrode")
        self.btn_pick.setCheckable(True)
        self.btn_pick.setToolTip(
            "Left-click in the 3D view to pick the nearest electrode; its "
            "placement's electrograms are shown here.")
        self.btn_pick.toggled.connect(self.pick_toggled)
        ctl.addWidget(self.btn_pick)
        self.chk_direction = QtWidgets.QCheckBox("Show bipole")
        self.chk_direction.setChecked(True)
        self.chk_direction.setToolTip(
            "Draw an arrow on the picked bipole in the 3D view, from its first "
            "to its second electrode.")
        self.chk_direction.toggled.connect(self.direction_toggled)
        ctl.addWidget(self.chk_direction)
        ctl.addSpacing(8)

        self._ticks: Dict[str, QtWidgets.QCheckBox] = {}
        for kind in TRACE_TYPES:
            chk = QtWidgets.QCheckBox(kind)
            chk.setChecked(True)
            chk.setToolTip(f"Show the {kind.lower()} traces.")
            chk.toggled.connect(self._redraw)
            self._ticks[kind] = chk
            ctl.addWidget(chk)
            for text, factor in (("−", 1.0 / GAIN_STEP), ("+", GAIN_STEP)):
                b = QtWidgets.QToolButton()
                b.setText(text)
                b.setAutoRaise(True)
                b.setToolTip(f"{'Lower' if text == '−' else 'Raise'} the "
                             f"{kind.lower()} gain.")
                b.clicked.connect(lambda _c=False, k=kind, f=factor: self._bump(k, f))
                ctl.addWidget(b)
            ctl.addSpacing(4)
        ctl.addStretch(1)
        for text, tip, slot in (("WOI", "Zoom to the window of interest.", self.zoom_woi),
                                ("Full", "Show the whole recording.", self.zoom_full)):
            b = QtWidgets.QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            ctl.addWidget(b)
        outer.addLayout(ctl)

        # --- figures ------------------------------------------------------
        self._ref_fig = Figure(figsize=(4, 1))
        self._ref_canvas = FigureCanvasQTAgg(self._ref_fig)
        self._ref_canvas.setFixedHeight(REF_PX)
        self._ref_ax = self._ref_fig.add_axes((0, 0, 1, 1))

        self._fig = Figure(figsize=(4, 4))
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._ax = self._fig.add_axes((0, 0, 1, 1))
        self._canvas.mpl_connect("button_press_event", self._on_click)
        self._ax.callbacks.connect("xlim_changed", self._on_xlim)

        self.toolbar = NavigationToolbar2QT(self._canvas, self)
        outer.addWidget(self.toolbar)

        self._scroll = QtWidgets.QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        # Always on, and the same width kept free beside the reference strip,
        # so both canvases are equally wide and share one time axis exactly.
        self._scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self._scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self._scroll.setWidget(self._canvas)

        ref_row = QtWidgets.QHBoxLayout()
        ref_row.setContentsMargins(0, 0, 0, 0)
        ref_row.setSpacing(0)
        ref_row.addWidget(self._ref_canvas, 1)
        ref_row.addSpacing(self._scroll.verticalScrollBar().sizeHint().width())
        outer.addLayout(ref_row)
        outer.addWidget(self._scroll, 1)

        self._message = QtWidgets.QLabel("Pick an electrode to see its electrograms.")
        self._message.setAlignment(QtCore.Qt.AlignCenter)
        outer.addWidget(self._message)
        self._set_empty(True)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    def set_egm(self, egm: Optional[dict], lat_by_row: Optional[Sequence[float]] = None
                ) -> None:
        """Adopt a map's electrograms (``None`` clears the panel).

        *lat_by_row* is the LAT column of ``_car.txt`` (index = row), for the
        LAT ticks; Carto's no-data value (-10000) means no tick.
        """
        self._egm = egm
        self._placement = None
        self._selected = None
        self._lat = {}
        self._catheter = None
        if egm is None:
            self._set_empty(True)
            return
        self._catheter = next(iter(egm["catheters"].values()), None)
        if lat_by_row is not None:
            self._lat = {i: float(v) for i, v in enumerate(lat_by_row)
                         if np.isfinite(v) and abs(v) < 1000.0}
        self._gain = {t: 1.0 for t in TRACE_TYPES}
        self._base_scale = self._type_scales(egm)
        woi = egm.get("woi") or (None, None)
        if None in woi:
            self._xlim = None
        else:
            self._xlim = (woi[0] - WOI_MARGIN_MS, woi[1] + WOI_MARGIN_MS)
        self._set_empty(True)

    def _type_columns(self, egm: dict) -> Dict[str, List[int]]:
        cat = next(iter(egm["catheters"].values()), None)
        bip = list(cat["bipole_column"]) if cat else []
        uni = list(cat["electrode_column"]) if cat else []
        used = {0, *bip, *uni}
        extra = [c for c in range(len(egm["channel_names"])) if c not in used]
        return {REF: [0], BIP: bip, UNI: uni, EXTRA: extra}

    def _type_scales(self, egm: dict) -> Dict[str, float]:
        """Row units per mV for each type, from the whole map's amplitudes."""
        traces, gain, missing = egm["traces"], egm["gain"], egm["missing_value"]
        # A subsample is plenty for a percentile, and keeps a 90-channel map
        # from allocating hundreds of MB here.
        step = max(1, traces.shape[0] // SCALE_PLACEMENTS)
        traces = traces[::step, :, ::SCALE_SAMPLE_STEP]
        out = {}
        for kind, cols in self._type_columns(egm).items():
            if not cols:
                out[kind] = 1.0
                continue
            v = traces[:, cols, :]
            v = np.abs(v[v != missing].astype(np.float32)) * gain
            amp = float(np.percentile(v, SCALE_PERCENTILE)) if v.size else 0.0
            out[kind] = SCALE_ROW_FRACTION / amp if amp > 0 else 1.0
        return out

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------
    @property
    def placement(self) -> Optional[int]:
        return self._placement

    @property
    def selected(self) -> Optional[int]:
        return self._selected

    def show_bipole(self, placement: int, bipole: Optional[int]) -> None:
        """Show *placement*'s traces with *bipole* highlighted."""
        if self._egm is None:
            return
        self._placement = placement
        self._selected = bipole
        self._set_empty(False)
        self._redraw()
        self._scroll_to_selected()

    def show_direction(self) -> bool:
        """True when the picked bipole's direction arrow is to be drawn."""
        return self.chk_direction.isChecked()

    def set_pick_checked(self, on: bool) -> None:
        """Mirror the picker state without re-emitting :attr:`pick_toggled`."""
        self.btn_pick.blockSignals(True)
        self.btn_pick.setChecked(on)
        self.btn_pick.blockSignals(False)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def _set_empty(self, empty: bool) -> None:
        self._message.setVisible(empty)
        self._ref_canvas.setVisible(not empty and self._ticks[REF].isChecked())
        self._scroll.setVisible(not empty)
        self.toolbar.setVisible(not empty)

    def _build_rows(self) -> List[_Row]:
        egm, cat = self._egm, self._catheter
        names = egm["channel_names"]
        show = {k: c.isChecked() for k, c in self._ticks.items()}
        rows: List[_Row] = []
        if cat is not None:
            for b, col in enumerate(cat["bipole_column"]):
                if show[BIP]:
                    rows.append(_Row(BIP, int(col), names[col], b))
                if show[UNI]:
                    for e in cat["bipole_electrodes"][b]:
                        ucol = int(cat["electrode_column"][e])
                        rows.append(_Row(UNI, ucol, names[ucol], b))
        if show[EXTRA]:
            for col in self._type_columns(egm)[EXTRA]:
                rows.append(_Row(EXTRA, col, names[col], None))
        return rows

    def _time_axis(self) -> np.ndarray:
        n = self._egm["traces"].shape[2]
        ref = float(self._egm["reference_annotation"][self._placement])
        return np.arange(n, dtype=float) - ref

    def _trace_mv(self, column: int) -> Optional[np.ndarray]:
        raw = self._egm["traces"][self._placement, column]
        if np.all(raw == self._egm["missing_value"]):
            return None
        return raw.astype(float) * self._egm["gain"]

    def _style_axes(self, ax, dark: bool) -> None:
        fg = "#e0e0e0" if dark else "#202020"
        bg = "#262626" if dark else "#ffffff"
        ax.set_facecolor(bg)
        ax.figure.set_facecolor(bg)
        ax.tick_params(colors=fg, labelsize=8)
        for s in ax.spines.values():
            s.set_visible(False)

    def _draw_woi(self, ax) -> None:
        woi = self._egm.get("woi") or (None, None)
        if None not in woi:
            ax.axvspan(woi[0], woi[1], color=WOI_COLOR, alpha=0.15, lw=0, zorder=0)
        ax.axvline(0.0, color=WOI_COLOR, ls="--", lw=0.8, zorder=1)

    def _redraw(self, *_args) -> None:
        if self._egm is None or self._placement is None:
            return
        dark = _is_dark(self)
        colors = TRACE_COLOR_DARK if dark else TRACE_COLOR
        t = self._time_axis()
        xlim = self._xlim if self._xlim else (t[0], t[-1])

        # Pinned reference strip, carrying the time axis.
        ax = self._ref_ax
        ax.clear()
        self._style_axes(ax, dark)
        self._ref_canvas.setVisible(self._ticks[REF].isChecked())
        y = self._trace_mv(0)
        if y is not None:
            ax.plot(t, y * self._base_scale[REF] * self._gain[REF],
                    color=colors[REF], lw=0.8)
        self._draw_woi(ax)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([0.0])
        ax.set_yticklabels([self._egm["channel_names"][0]])
        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")
        ax.set_xlabel("ms", fontsize=8, labelpad=1,
                      color="#e0e0e0" if dark else "#202020")
        ax.set_xlim(*xlim)
        self._place_axes(ax, self._ref_canvas, top_px=TIME_AXIS_PX)

        # Scrolling rows.
        self._rows = self._build_rows()
        n = max(len(self._rows), 1)
        self._canvas.setMinimumHeight(n * ROW_PX)
        self._canvas.setMaximumHeight(n * ROW_PX)
        ax = self._ax
        ax.clear()
        self._style_axes(ax, dark)
        self._draw_woi(ax)
        link = self._catheter["point_link"][self._placement] if self._catheter else None
        sel_rows = [i for i, r in enumerate(self._rows)
                    if self._selected is not None and r.bipole == self._selected]
        if sel_rows:
            ax.axhspan(-max(sel_rows) - 0.5, -min(sel_rows) + 0.5, lw=0, zorder=0.5,
                       color=HIGHLIGHT_DARK if dark else HIGHLIGHT_LIGHT)
        labels = []
        for i, r in enumerate(self._rows):
            y = self._trace_mv(r.column)
            if y is None:
                labels.append(f"{r.label} (not recorded)")
                continue
            labels.append(r.label)
            ax.plot(t, -i + y * self._base_scale[r.kind] * self._gain[r.kind],
                    color=colors[r.kind], lw=0.8, zorder=2)
            if r.kind == BIP and link is not None and link[r.bipole] >= 0:
                lat = self._lat.get(int(link[r.bipole]))
                if lat is not None:
                    ax.plot([lat, lat], [-i - 0.4, -i + 0.4], color=LAT_COLOR,
                            lw=1.6, zorder=3)
        ax.set_ylim(-n + 0.5, 0.5)
        ax.set_yticks([-i for i in range(len(self._rows))])
        ax.set_yticklabels(labels)
        ax.tick_params(axis="x", labelbottom=False)
        self._syncing = True
        ax.set_xlim(*xlim)
        self._syncing = False
        self._place_axes(ax, self._canvas, top_px=0)
        self._ref_canvas.draw_idle()
        self._canvas.draw_idle()

    @staticmethod
    def _place_axes(ax, canvas, top_px: int) -> None:
        """Leave LABEL_PX on the left for the row names and *top_px* above
        for the time axis; the axes take the rest of the canvas."""
        w = max(canvas.width(), LABEL_PX + 10)
        h = max(canvas.height(), top_px + 10)
        left = LABEL_PX / w
        ax.set_position((left, 0.0, 1.0 - left - 4.0 / w, 1.0 - top_px / h))

    def _scroll_to_selected(self) -> None:
        rows = [i for i, r in enumerate(self._rows) if r.bipole == self._selected]
        if rows:
            y = int((min(rows) + max(rows) + 1) / 2 * ROW_PX)
            self._scroll.ensureVisible(0, y, 0, int(len(rows) * ROW_PX))

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------
    def _on_xlim(self, ax) -> None:
        if self._syncing or self._egm is None:
            return
        self._xlim = tuple(ax.get_xlim())
        self._ref_ax.set_xlim(*self._xlim)
        self._ref_canvas.draw_idle()

    def _bump(self, kind: str, factor: float) -> None:
        self._gain[kind] *= factor
        self._redraw()

    def zoom_woi(self) -> None:
        woi = (self._egm or {}).get("woi") or (None, None)
        if None not in woi:
            self._xlim = (woi[0] - WOI_MARGIN_MS, woi[1] + WOI_MARGIN_MS)
            self._redraw()

    def zoom_full(self) -> None:
        if self._egm is None or self._placement is None:
            return
        t = self._time_axis()
        self._xlim = (float(t[0]), float(t[-1]))
        self._redraw()

    def _on_click(self, event) -> None:
        if self.toolbar.mode or event.inaxes is not self._ax or event.ydata is None:
            return
        i = int(round(-event.ydata))
        if 0 <= i < len(self._rows) and self._rows[i].bipole is not None:
            self.bipole_clicked.emit(int(self._rows[i].bipole))

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt)
        super().resizeEvent(event)
        self._redraw()

    # For tests: what is drawn.
    def row_labels(self) -> List[str]:
        return [r.label for r in self._rows]

    def row_bipoles(self) -> List[Optional[int]]:
        return [r.bipole for r in self._rows]


__all__ = ["EGMPanel", "TRACE_TYPES", "REF", "BIP", "UNI", "EXTRA"]
