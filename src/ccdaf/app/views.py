"""View registry — each purpose the plotter can serve, bound to its layout.

A :class:`ViewSpec` ties a purpose (general viewing, segmentation editing,
…) to the plotter layout that purpose needs: pyvista's own ``shape`` and
``groups`` arguments, a role name for every renderer, and the on-screen
title each role carries. The active spec answers the two questions the old
``_segmentation_mode`` boolean conflated — *which task is the view serving*
(``name``) and *what is the plotter's structure* (everything else) — so a
second multi-view task can never silently inherit answers that were only
ever true of segmentation.

A locator is whatever ``plotter.subplot(*locator)`` accepts: ``(row, col)``
for grid shapes, a single index ``(i,)`` for pyvista's string shapes such
as ``"1|2"`` (one pane left, two stacked right). Adding a multi-view task
is therefore a new entry in :data:`VIEWS`, not a new flag or code path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence, Tuple, Union


@dataclass(frozen=True)
class ViewSpec:
    """One view purpose: its plotter layout and the role of every pane."""

    name: str
    shape: Union[Tuple[int, int], str]   # passed verbatim to the Plotter
    roles: Mapping[str, Tuple[int, ...]]  # role -> subplot locator, all panes
    titles: Mapping[str, str] = field(default_factory=dict)  # role -> title
    groups: Optional[Sequence] = None    # passed verbatim when present

    @property
    def is_multiview(self) -> bool:
        return len(self.roles) > 1


def title_actor_name(locator: Tuple[int, ...]) -> str:
    """Actor name of a pane's title text.

    The build-time title and every later replacement (slice index overlays,
    the 3D pane after a clear) must resolve to the same actor, so the name
    is derived here and nowhere else.
    """
    return "_title_" + "_".join(str(i) for i in locator)


VIEWS: Mapping[str, ViewSpec] = {
    "general": ViewSpec(
        name="general",
        shape=(1, 1),
        roles={"3d": (0, 0)},
    ),
    "segmentation": ViewSpec(
        name="segmentation",
        shape=(2, 2),
        #   (0,0) Top-Left     Axial    XY
        #   (0,1) Top-Right    Sagittal YZ
        #   (1,1) Bottom-Right Coronal  XZ
        #   (1,0) Bottom-Left  3D
        roles={
            "axial": (0, 0),
            "sagittal": (0, 1),
            "3d": (1, 0),
            "coronal": (1, 1),
        },
        titles={
            "axial": "Axial (XY)",
            "sagittal": "Sagittal (YZ)",
            "coronal": "Coronal (XZ)",
            "3d": "3D",
        },
    ),
}
