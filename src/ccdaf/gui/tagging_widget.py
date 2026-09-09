"""
TaggingWidget
=============
Side-panel widget for automatic region tagging controls.

What the panel offers follows the seed type chosen in the Seed selection
panel, pushed in through :meth:`set_profile`: the radius factors are the
profile's ``radius_names``, one spin box each, and a profile with none
(a landmark set, a signpost anatomy) has nothing to tag with, so the
whole panel goes quiet and says which seed type it is following. That
keeps the atrium's five veins out of every other anatomy's panel without
the widget knowing what an atrium is.

The "Run automatic tagging" button needs three things at once: a profile
that tags, seeds complete, and the disable-checkbox unchecked. All three
are tracked internally so the host only calls ``set_profile`` and
``set_seeds_complete``.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

from PyQt5 import QtCore, QtWidgets


#: Starting value of a radius factor, as a multiple of the median edge
#: length. Per-seed rather than global in the tagger, but they start
#: together: the caps are tuned by eye against the ostia, so a common
#: starting point is one fewer thing to explain.
RADIUS_DEFAULT: float = 25.0

#: Range of a radius factor. The floor is above zero because a cap of
#: zero tags nothing, which is a state the box should not be able to
#: offer.
RADIUS_RANGE: Tuple[float, float] = (0.1, 500.0)


class TaggingWidget(QtWidgets.QGroupBox):

    tagging_requested = QtCore.pyqtSignal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._seeds_complete: bool = False
        self._tags: bool = False
        # One row per radius name, kept across seed-type switches so a
        # tuned value is still there when the user comes back to it.
        self.spn_radius: Dict[str, QtWidgets.QDoubleSpinBox] = {}
        self._rows: Dict[str, QtWidgets.QWidget] = {}
        self._active: Tuple[str, ...] = ()

        self.lbl_follows = QtWidgets.QLabel()
        self.lbl_follows.setWordWrap(True)
        self.lbl_follows.setToolTip(
            "Tagging acts on the seed type selected in the Seed selection "
            "panel. Change it there to tag a different set."
        )
        layout.addWidget(self.lbl_follows)

        self._radius_box = QtWidgets.QWidget()
        self._radius_layout = QtWidgets.QVBoxLayout(self._radius_box)
        self._radius_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._radius_box)

        self.chk_disable_tag = QtWidgets.QCheckBox("Disable automatic tagging")
        self.chk_disable_tag.setToolTip(
            "When checked, the 'Run automatic tagging' button is disabled "
            "to prevent accidental re-runs."
        )
        self.chk_disable_tag.toggled.connect(self._update_button_state)
        layout.addWidget(self.chk_disable_tag)

        self.btn_tag = QtWidgets.QPushButton("Run automatic tagging")
        self.btn_tag.setToolTip(
            "Automatically tag the regions of the selected seed type from "
            "its seeds and the radius factors above."
        )
        self.btn_tag.clicked.connect(self.tagging_requested.emit)
        self.btn_tag.setEnabled(False)
        layout.addWidget(self.btn_tag)

    # ------------------------------------------------------------------
    def _make_row(self, name: str) -> QtWidgets.QWidget:
        row = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(QtWidgets.QLabel(f"{name} radius ×"))
        sp = QtWidgets.QDoubleSpinBox()
        sp.setDecimals(1)
        sp.setRange(*RADIUS_RANGE)
        sp.setSingleStep(1.0)
        sp.setValue(RADIUS_DEFAULT)
        sp.setToolTip(
            f"{name}: radius cap = factor × median edge length.\n"
            f"Tune before running automatic tagging."
        )
        lay.addWidget(sp, 1)
        self.spn_radius[name] = sp
        return row

    def set_profile(self, profile) -> None:
        """Show the radii *profile* tags with, and nothing else.

        Rows are re-ordered to the profile's own order rather than the
        order they were first created in, so a set that lists its seeds
        differently reads the way it is picked. Rows for other profiles
        are hidden, not destroyed: their tuned values come back with them.
        """
        names: Sequence[str] = tuple(profile.radius_names)
        self._tags = bool(profile.tags) and bool(names)
        self._active = tuple(names)

        # Detach every row, then re-attach the wanted ones in order.
        while self._radius_layout.count():
            item = self._radius_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
        for name in names:
            row = self._rows.get(name)
            if row is None:
                row = self._make_row(name)
                self._rows[name] = row
            self._radius_layout.addWidget(row)
            row.setVisible(True)

        self._radius_box.setVisible(bool(names))
        self.lbl_follows.setText(
            f"Follows seed type: <b>{profile.label}</b>" if self._tags
            else (f"Follows seed type: <b>{profile.label}</b><br>"
                  f"<i>This seed type has no regions to tag.</i>")
        )
        self._update_button_state()

    # ------------------------------------------------------------------
    def _update_button_state(self) -> None:
        enabled = (self._tags and self._seeds_complete
                   and not self.chk_disable_tag.isChecked())
        self.btn_tag.setEnabled(enabled)

    def set_seeds_complete(self, complete: bool) -> None:
        self._seeds_complete = complete
        self._update_button_state()

    def radius_factors(self) -> Dict[str, float]:
        """The active profile's radius factors, by seed name.

        Only the rows the current profile asked for: a value left over
        from another seed type is still held, but it is not this
        profile's and must not reach its tagger config.
        """
        return {name: float(self.spn_radius[name].value())
                for name in self._active if name in self.spn_radius}


__all__ = ["TaggingWidget", "RADIUS_DEFAULT", "RADIUS_RANGE"]
