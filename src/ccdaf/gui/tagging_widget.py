"""
TaggingWidget
=============
Side-panel widget for automatic region tagging controls.

The "Run automatic tagging" button is only enabled when seeds are
complete AND the disable-checkbox is unchecked.  Both conditions are
managed internally so the host only needs to call ``set_seeds_complete``.
"""
from __future__ import annotations

from typing import Dict, Optional

from PyQt5 import QtCore, QtWidgets


_RADIUS_DEFAULTS = (
    ("LSPV", 25.0),
    ("LIPV", 25.0),
    ("RSPV", 25.0),
    ("RIPV", 25.0),
    ("LAA",  25.0),
)


class TaggingWidget(QtWidgets.QGroupBox):

    tagging_requested = QtCore.pyqtSignal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._seeds_complete: bool = False

        grid_r = QtWidgets.QGridLayout()
        self.spn_radius: Dict[str, QtWidgets.QDoubleSpinBox] = {}
        for row_idx, (name, default) in enumerate(_RADIUS_DEFAULTS):
            grid_r.addWidget(QtWidgets.QLabel(f"{name} radius ×"), row_idx, 0)
            sp = QtWidgets.QDoubleSpinBox()
            sp.setDecimals(1)
            sp.setRange(0.1, 500.0)
            sp.setSingleStep(1.0)
            sp.setValue(default)
            sp.setToolTip(
                f"{name}: radius cap = factor × median edge length.\n"
                f"Tune before running automatic tagging."
            )
            grid_r.addWidget(sp, row_idx, 1)
            self.spn_radius[name] = sp
        layout.addLayout(grid_r)

        self.chk_disable_tag = QtWidgets.QCheckBox("Disable automatic tagging")
        self.chk_disable_tag.setToolTip(
            "When checked, the 'Run automatic tagging' button is disabled "
            "to prevent accidental re-runs."
        )
        self.chk_disable_tag.toggled.connect(self._update_button_state)
        layout.addWidget(self.chk_disable_tag)

        self.btn_tag = QtWidgets.QPushButton("Run automatic tagging")
        self.btn_tag.clicked.connect(self.tagging_requested.emit)
        self.btn_tag.setEnabled(False)
        layout.addWidget(self.btn_tag)

    def _update_button_state(self) -> None:
        enabled = self._seeds_complete and not self.chk_disable_tag.isChecked()
        self.btn_tag.setEnabled(enabled)

    def set_seeds_complete(self, complete: bool) -> None:
        self._seeds_complete = complete
        self._update_button_state()

    def radius_factors(self) -> Dict[str, float]:
        return {name: float(sp.value()) for name, sp in self.spn_radius.items()}


__all__ = ["TaggingWidget"]
