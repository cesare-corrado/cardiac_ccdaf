"""
ChamberExtractDialog
====================
Actions > Extract chambers…: choose which chambers' tissue to keep.

One check box per chamber that has tissue in the mesh, as the tag names
say (see :mod:`ccdaf.core.tag_map`), with the tags it would keep. See
:mod:`ccdaf.core.chamber_extract` for what is kept and what is recorded.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence

from PyQt5 import QtWidgets

from ccdaf.core import tag_map

_FULL = {"LV": "Left ventricle", "RV": "Right ventricle",
         "LA": "Left atrium", "RA": "Right atrium"}


class ChamberExtractDialog(QtWidgets.QDialog):
    """Pick chambers; :meth:`chambers` is the choice."""

    def __init__(self, available: Sequence[str],
                 mapping: Mapping[int, object], tags: Sequence[int],
                 parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Extract chambers")
        named = {t: v for t, v in tag_map.normalise(mapping).items()
                 if t in {int(x) for x in tags}}
        intro = QtWidgets.QLabel(
            "Keep the tissue (wall, layer, vein and appendage tags) of the "
            "chambers ticked. Caps, valves, vessels and the other chambers "
            "are removed; where the caps and valves met the tissue is "
            "remembered as the rims. The result replaces the working mesh.")
        intro.setWordWrap(True)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(intro)
        self._boxes: Dict[str, QtWidgets.QCheckBox] = {}
        for chamber in available:
            kept = tag_map.tags_of(named, chamber)
            box = QtWidgets.QCheckBox(
                f"{_FULL.get(chamber, chamber)} (tags "
                f"{', '.join(map(str, kept))})")
            box.setChecked(chamber in ("LA", "RA"))
            box.toggled.connect(self._sync)
            layout.addWidget(box)
            self._boxes[chamber] = box
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self.btn_ok = buttons.button(QtWidgets.QDialogButtonBox.Ok)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._sync()

    def _sync(self, *_args) -> None:
        self.btn_ok.setEnabled(bool(self.chambers()))

    def chambers(self) -> List[str]:
        """The chambers ticked, in the dialog's order."""
        return [c for c, box in self._boxes.items() if box.isChecked()]


__all__ = ["ChamberExtractDialog"]
