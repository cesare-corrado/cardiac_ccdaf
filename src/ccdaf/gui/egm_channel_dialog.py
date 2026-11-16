"""
EGMChannelDialog
================
Shown after a map is picked with **Load electrograms** ticked. It tells the
user what will always be stored and lets them add more.

* **Mapping catheter** — the connector whose electrodes give the positions.
  Pre-set to the one the import detected; the user can change it.
* **Always stored** — the reference channel, the point bipoles and their
  unipoles. Listed, not editable.
* **Extra channels** — every other channel found in the map's ECG files,
  unticked. A ticked channel is stored as a trace only (no position).

Accepting returns the choice through :meth:`selected_connector` and
:meth:`selected_extra_channels`; rejecting means "load the map without
electrograms".
"""
from __future__ import annotations

from typing import List, Optional

from PyQt5 import QtCore, QtWidgets

from ccdaf.io.carto_egm import EGMScan


def _span(names: List[str]) -> str:
    """'20A_1-2 … 20A_19-20 (10)' — a short summary of a channel list."""
    if not names:
        return "none"
    if len(names) <= 3:
        return ", ".join(names)
    return f"{names[0]} … {names[-1]} ({len(names)})"


class EGMChannelDialog(QtWidgets.QDialog):

    def __init__(self, scan: EGMScan,
                 parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Electrograms")
        self.setMinimumWidth(420)
        layout = QtWidgets.QVBoxLayout(self)

        layout.addWidget(QtWidgets.QLabel(
            f"Map <b>{scan.map_name}</b>: {len(scan.points)} points, "
            f"{len(scan.start_times)} catheter placements."))

        # --- mapping catheter ---------------------------------------------
        form = QtWidgets.QFormLayout()
        self._cmb_connector = QtWidgets.QComboBox()
        self._cmb_connector.setToolTip(
            "The catheter whose electrode positions are stored. Detected from "
            "where the points sit; change it only if the detection is wrong.")
        for conn in scan.connectors:
            label = conn + ("  (detected)" if conn == scan.mapping_connector else "")
            self._cmb_connector.addItem(label, userData=conn)
        if scan.mapping_connector in scan.connectors:
            self._cmb_connector.setCurrentIndex(
                scan.connectors.index(scan.mapping_connector))
        form.addRow("Mapping catheter:", self._cmb_connector)
        layout.addLayout(form)

        # --- always stored ----------------------------------------------------
        box = QtWidgets.QGroupBox("Always stored")
        bl = QtWidgets.QFormLayout(box)
        bl.addRow("Reference:", QtWidgets.QLabel(", ".join(scan.reference_channels)))
        bl.addRow("Bipoles:", QtWidgets.QLabel(_span(scan.bipoles)))
        bl.addRow("Unipoles:", QtWidgets.QLabel(_span(scan.unipoles)))
        layout.addWidget(box)

        # --- extra channels ---------------------------------------------------
        layout.addWidget(QtWidgets.QLabel(
            "Extra channels (stored as traces only):"))
        always = set(scan.default_channels)
        self._list = QtWidgets.QListWidget()
        self._list.setToolTip("Tick the channels to store as well.")
        for name in scan.available_channels:
            if name in always:
                continue
            item = QtWidgets.QListWidgetItem(name)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Unchecked)
            self._list.addItem(item)
        layout.addWidget(self._list, 1)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.button(QtWidgets.QDialogButtonBox.Ok).setText("Load")
        buttons.button(QtWidgets.QDialogButtonBox.Cancel).setToolTip(
            "Load the map without electrograms.")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_connector(self) -> Optional[str]:
        data = self._cmb_connector.currentData()
        return str(data) if data else None

    def selected_extra_channels(self) -> List[str]:
        return [self._list.item(i).text() for i in range(self._list.count())
                if self._list.item(i).checkState() == QtCore.Qt.Checked]


__all__ = ["EGMChannelDialog"]
