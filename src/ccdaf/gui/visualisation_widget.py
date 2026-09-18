"""
VisualisationWidget
===================
Side-panel widget choosing what colours the surface, for any loaded mesh —
an EAM mapping's Carto fields, or just the ``elemTag`` a plain mesh carries.

* **Field**      — any field on the mesh, whether it is stored on the points
                   (an EAM mapping's ``Unipolar``, ``LAT``, …, interpolated
                   across each triangle) or on the cells (``elemTag``, flat
                   over each triangle).
* **Colour map** — the colour map applied to it.
* **Auto / Min / Max** — the colour range. With *auto* ticked the range follows
                   the field's own data range and the spin boxes only report it;
                   unticked, the range is the user's and values outside it are
                   flagged rather than clamped.
* **Iso lines**  — how many discrete colour bands the map is quantised into
                   (minimum 2); the band boundaries are the isolines.
* **Show fibres** — short line segments along a volume's ``fiber`` (or
                   ``sheet``) directions at a sample of elements, over a
                   see-through surface. Greyed until the volume has fibres.

Some fields are *categorical*: their values are labels, not measurements, so a
continuous colour ramp over them would be meaningless. Those are drawn by the
app's region path instead — discrete colours with names — and the scale
controls above are disabled while one is selected. ``elemTag`` is the only one
today; further label fields should join it rather than grow a second paradigm.

Any change emits :attr:`settings_changed`; the app reads the state back through
the accessors and re-renders.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

from PyQt5 import QtCore, QtWidgets


# Colour maps offered for the surface (matplotlib names).
CMAPS: List[str] = [
    "viridis", "jet", "turbo", "rainbow", "coolwarm", "bwr", "seismic",
    "plasma", "inferno", "magma", "hot", "gray",
]
DEFAULT_CMAP     = "viridis"
DEFAULT_ISOLINES = 256      # a continuous-looking map; lower it to band
MIN_ISOLINES     = 2
MAX_ISOLINES     = 256
DEFAULT_SEGMENTS = 20_000   # fibre segments drawn: enough to read, quick to draw

POINT_FIELD = "point"
CELL_FIELD = "cell"


class VisualisationWidget(QtWidgets.QGroupBox):

    settings_changed = QtCore.pyqtSignal()
    # Emitted by the electrode checkbox alone: toggling visibility only needs
    # the electrode actor redrawn, not the whole field re-rendered.
    electrodes_toggled = QtCore.pyqtSignal(bool)
    # Emitted by the fibre controls alone: the glyphs are their own actor,
    # so changing them needs no re-render of the field underneath.
    fibres_changed = QtCore.pyqtSignal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._categorical: set = set()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        grid = QtWidgets.QGridLayout()
        row = 0

        grid.addWidget(QtWidgets.QLabel("Field:"), row, 0)
        self.cmb_field = QtWidgets.QComboBox()
        self.cmb_field.setToolTip(
            "Field used to colour the surface. Point fields are interpolated "
            "across each triangle; cell fields are flat over it."
        )
        grid.addWidget(self.cmb_field, row, 1)
        row += 1

        grid.addWidget(QtWidgets.QLabel("Colour map:"), row, 0)
        self.cmb_cmap = QtWidgets.QComboBox()
        self.cmb_cmap.setToolTip("Colour map used to render the selected field.")
        for name in CMAPS:
            self.cmb_cmap.addItem(name, name)
        self.cmb_cmap.setCurrentIndex(CMAPS.index(DEFAULT_CMAP))
        grid.addWidget(self.cmb_cmap, row, 1)
        row += 1

        self.chk_auto = QtWidgets.QCheckBox("Auto min/max")
        self.chk_auto.setChecked(True)
        self.chk_auto.setToolTip(
            "Take the colour range from the field's own data range."
        )
        grid.addWidget(self.chk_auto, row, 0, 1, 2)
        row += 1

        grid.addWidget(QtWidgets.QLabel("Min:"), row, 0)
        self.spn_min = self._make_range_spin()
        self.spn_min.setToolTip(
            "Lower limit of the colour range (used when 'Auto min/max' is off)."
        )
        grid.addWidget(self.spn_min, row, 1)
        row += 1

        grid.addWidget(QtWidgets.QLabel("Max:"), row, 0)
        self.spn_max = self._make_range_spin()
        self.spn_max.setToolTip(
            "Upper limit of the colour range (used when 'Auto min/max' is off)."
        )
        grid.addWidget(self.spn_max, row, 1)
        row += 1

        grid.addWidget(QtWidgets.QLabel("Iso lines:"), row, 0)
        self.spn_iso = QtWidgets.QSpinBox()
        self.spn_iso.setRange(MIN_ISOLINES, MAX_ISOLINES)
        self.spn_iso.setValue(DEFAULT_ISOLINES)
        self.spn_iso.setToolTip("Number of discrete colour bands (minimum 2).")
        grid.addWidget(self.spn_iso, row, 1)
        row += 1

        self.chk_electrodes = QtWidgets.QCheckBox("Show electrodes")
        self.chk_electrodes.setChecked(True)
        self.chk_electrodes.setEnabled(False)   # grey until a mapping brings some
        self.chk_electrodes.setToolTip(
            "Draw the mapping's electrode positions on the surface. "
            "Greyed until a mapping with electrodes is loaded."
        )
        grid.addWidget(self.chk_electrodes, row, 0, 1, 2)
        row += 1

        self.chk_fibres = QtWidgets.QCheckBox("Show fibres")
        self.chk_fibres.setEnabled(False)       # grey until a volume has fibres
        self.chk_fibres.setToolTip(
            "Draw short line segments along the volume's directions at a "
            "sample of elements, and make the surface see-through so the "
            "wall shows. Greyed until the volume carries 'fiber'.")
        grid.addWidget(self.chk_fibres, row, 0, 1, 2)
        row += 1

        grid.addWidget(QtWidgets.QLabel("Direction:"), row, 0)
        self.cmb_direction = QtWidgets.QComboBox()
        self.cmb_direction.setToolTip("Which direction the segments follow.")
        grid.addWidget(self.cmb_direction, row, 1)
        row += 1

        grid.addWidget(QtWidgets.QLabel("Segments:"), row, 0)
        self.spn_segments = QtWidgets.QSpinBox()
        self.spn_segments.setRange(1_000, 200_000)
        self.spn_segments.setSingleStep(5_000)
        self.spn_segments.setValue(DEFAULT_SEGMENTS)
        self.spn_segments.setKeyboardTracking(False)
        self.spn_segments.setToolTip(
            "How many elements get a segment. More shows more detail and "
            "draws more slowly.")
        grid.addWidget(self.spn_segments, row, 1)

        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)

        self.cmb_field.currentIndexChanged.connect(self._on_field_changed)
        self.cmb_cmap.currentIndexChanged.connect(self._emit)
        self.chk_auto.toggled.connect(self._on_auto_toggled)
        self.spn_min.valueChanged.connect(self._emit)
        self.spn_max.valueChanged.connect(self._emit)
        self.spn_iso.valueChanged.connect(self._emit)
        self.chk_electrodes.toggled.connect(self.electrodes_toggled.emit)
        self.chk_fibres.toggled.connect(self._on_fibres_toggled)
        self.cmb_direction.currentIndexChanged.connect(self._emit_fibres)
        self.spn_segments.valueChanged.connect(self._emit_fibres)
        self._sync_enabled()
        self._sync_fibre_enabled()

    @staticmethod
    def _make_range_spin() -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setDecimals(3)
        spin.setRange(-1.0e9, 1.0e9)
        spin.setKeyboardTracking(False)   # emit on commit, not per keystroke
        return spin

    # -- population -----------------------------------------------------
    def set_fields(self, point_fields: Sequence[str],
                   cell_fields: Sequence[str],
                   categorical: Iterable[str] = ()) -> None:
        """Populate the field drop-down. Does not emit ``settings_changed``.

        ``categorical`` names the fields whose values are labels; they are
        offered like any other but drawn by the region path.
        """
        self._categorical = {str(c) for c in categorical}
        self.cmb_field.blockSignals(True)
        self.cmb_field.clear()
        for names, kind in ((point_fields, POINT_FIELD), (cell_fields, CELL_FIELD)):
            for name in names:
                self.cmb_field.addItem(f"{name}  ({kind})", (str(name), kind))
        self.cmb_field.setCurrentIndex(0 if self.cmb_field.count() else -1)
        self.cmb_field.blockSignals(False)
        self._sync_enabled()

    def select_field(self, name: str) -> bool:
        """Select ``name`` if present. Does not emit ``settings_changed``."""
        for i in range(self.cmb_field.count()):
            data = self.cmb_field.itemData(i)
            if data and data[0] == str(name):
                self.cmb_field.blockSignals(True)
                self.cmb_field.setCurrentIndex(i)
                self.cmb_field.blockSignals(False)
                self._sync_enabled()
                return True
        return False

    def set_range_display(self, lo: float, hi: float) -> None:
        """Report an auto-computed range in the spin boxes without emitting."""
        for spin, value in ((self.spn_min, lo), (self.spn_max, hi)):
            spin.blockSignals(True)
            spin.setValue(float(value))
            spin.blockSignals(False)

    # -- queries --------------------------------------------------------
    def current_field(self) -> Optional[str]:
        data = self.cmb_field.currentData()
        return None if not data else str(data[0])

    def current_association(self) -> Optional[str]:
        data = self.cmb_field.currentData()
        return None if not data else str(data[1])

    def is_categorical(self) -> bool:
        field = self.current_field()
        return field is not None and field in self._categorical

    def current_cmap(self) -> str:
        data = self.cmb_cmap.currentData()
        return DEFAULT_CMAP if data is None else str(data)

    def is_auto(self) -> bool:
        return bool(self.chk_auto.isChecked())

    def clim(self) -> Tuple[float, float]:
        return float(self.spn_min.value()), float(self.spn_max.value())

    def n_isolines(self) -> int:
        return int(self.spn_iso.value())

    def show_electrodes(self) -> bool:
        return bool(self.chk_electrodes.isChecked())

    def set_electrodes_available(self, available: bool) -> None:
        """Grey the electrode checkbox when there are none to show.

        The checked state is the user's choice and survives; only the
        greying follows the data. Does not emit."""
        self.chk_electrodes.setEnabled(bool(available))

    def set_fibre_directions(self, names: Sequence[str]) -> None:
        """Offer these direction arrays; none greys the fibre controls.

        The tick survives, as the electrode one does: a mesh arriving
        without fibres greys it, and the next one with fibres shows them
        again. Does not emit."""
        current = self.fibre_direction()
        self.cmb_direction.blockSignals(True)
        self.cmb_direction.clear()
        for name in names:
            self.cmb_direction.addItem(str(name), str(name))
        if current is not None:
            i = self.cmb_direction.findData(current)
            if i >= 0:
                self.cmb_direction.setCurrentIndex(i)
        self.cmb_direction.blockSignals(False)
        self.chk_fibres.setEnabled(bool(names))
        self._sync_fibre_enabled()

    def show_fibres(self) -> bool:
        return bool(self.chk_fibres.isChecked() and self.chk_fibres.isEnabled())

    def fibre_direction(self) -> Optional[str]:
        data = self.cmb_direction.currentData()
        return None if data is None else str(data)

    def fibre_segments(self) -> int:
        return int(self.spn_segments.value())

    # -- internals ------------------------------------------------------
    def _sync_fibre_enabled(self) -> None:
        on = self.show_fibres()
        self.cmb_direction.setEnabled(on)
        self.spn_segments.setEnabled(on)

    def _on_fibres_toggled(self, *_args) -> None:
        self._sync_fibre_enabled()
        self._emit_fibres()

    def _emit_fibres(self, *_args) -> None:
        self.fibres_changed.emit()

    def _sync_enabled(self) -> None:
        """Scale controls mean nothing for a label field, so switch them off
        rather than let them imply an effect they cannot have."""
        scalar = self.current_field() is not None and not self.is_categorical()
        for w in (self.cmb_cmap, self.chk_auto, self.spn_iso):
            w.setEnabled(scalar)
        manual = scalar and not self.chk_auto.isChecked()
        self.spn_min.setEnabled(manual)
        self.spn_max.setEnabled(manual)

    def _on_field_changed(self, *_args) -> None:
        self._sync_enabled()
        self._emit()

    def _on_auto_toggled(self, *_args) -> None:
        self._sync_enabled()
        self._emit()

    def _emit(self, *_args) -> None:
        self.settings_changed.emit()


__all__ = [
    "VisualisationWidget", "CMAPS", "DEFAULT_CMAP", "DEFAULT_ISOLINES",
    "MIN_ISOLINES", "POINT_FIELD", "CELL_FIELD", "DEFAULT_SEGMENTS",
]
