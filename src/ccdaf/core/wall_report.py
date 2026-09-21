"""
wall_report
===========
Measure what is wrong with a ventricular wall, without changing it.

Every other volumetric tool here either edits the mesh or answers one
narrow question about it. This one composes them into the question a
user actually asks before running anything: **is my wall sound, and if
not, where is it not?**

Why it works on a copy
----------------------
The measurements it needs are only defined on a manifold boundary. The
genus — how many handles the surface has — is undefined otherwise, and
the join finder walks faces across shared edges, which a non-manifold
edge blocks. But the cleaner's default deliberately leaves the boundary
non-manifold, because separating material that only touches breaks the
ventricular surface labelling (see :class:`volume_clean.CleanOptions`).

So this repairs a copy, fully, measures that, and reports it. Your mesh
is not touched. What it tells you is true of the wall, which is a
property of the anatomy and not of which repairs happen to be switched
on.

How many handles a wall should have
-----------------------------------
Not zero. A shell around ``p`` cavities opened by ``o`` valve openings
has ``o - p`` handles when every cavity opens at least once: one cavity
opened twice gives a loop you cannot shrink — in through the mitral
opening, along the cavity, out through the aortic one. The example
ventricle has two pools and three openings, so it should have one
handle, and it measures two. The extra one is a perforation, 11.7 units
around, in a wall that thins to 0.39 mm beside it.

That is the whole report: the count that should be, the count that is,
and where each way through the wall lies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from ccdaf.core.orifice_labels import (
    OrificeOptions, Passage, describe_passages, find_openings, find_passages,
)
from ccdaf.core.volume_clean import CleanOptions, clean
from ccdaf.core.volume_mesh import boundary_faces, tetrahedra
from ccdaf.core.volume_repair import (
    boundary_genus, non_manifold_edges, pinched_vertices,
)


@dataclass
class WallReport:
    """What the wall is, against what the anatomy implies it should be."""

    #: Handles measured on the repaired copy, or ``None`` if even that
    #: could not be made manifold.
    genus: Optional[int]
    #: Handles the openings imply: one per opening beyond the first of
    #: each cavity.
    expected: Optional[int]
    #: Blood pools found, by volume in mL.
    pool_ml: List[float] = field(default_factory=list)
    #: Valve openings found, by area in mm².
    opening_mm2: List[float] = field(default_factory=list)
    #: Every join between the epicardium and a cavity, largest first.
    joins: List[Passage] = field(default_factory=list)
    #: What the repair had to do to the copy to make it measurable.
    repaired: str = ""
    #: Where the copy is still not manifold, when it is. Each entry is a
    #: place the repair could not resolve, which is the only thing worth
    #: saying when the counts cannot be given.
    unresolved: List[np.ndarray] = field(default_factory=list)

    @property
    def extra_handles(self) -> Optional[int]:
        """Handles the anatomy does not account for — holes in the wall."""
        if self.genus is None or self.expected is None:
            return None
        return max(0, self.genus - self.expected)

    @property
    def sound(self) -> bool:
        return self.extra_handles == 0

    def summary(self) -> str:
        if self.genus is None:
            left = len(self.unresolved)
            return ("The wall could not be measured: the boundary is not "
                    "manifold even after a full repair"
                    + (f", at {left} place"
                       + ("" if left == 1 else "s") + "." if left else "."))
        if self.expected is None:
            return (f"The boundary has {self.genus} handle"
                    + ("" if self.genus == 1 else "s")
                    + "; no valve openings were found to compare it with.")
        extra = self.extra_handles
        if extra == 0:
            return (f"The wall is sound: {self.genus} handle"
                    + ("" if self.genus == 1 else "s")
                    + ", which is what "
                    + f"{len(self.opening_mm2)} openings into "
                    + f"{len(self.pool_ml)} cavities imply.")
        return (f"The wall has {extra} hole"
                + ("" if extra == 1 else "s")
                + f" in it: {self.genus} handles measured against the "
                + f"{self.expected} that {len(self.opening_mm2)} openings "
                + f"into {len(self.pool_ml)} cavities imply.")

    def details(self) -> str:
        """The full report, for a dialog rather than a status line."""
        lines = [self.summary(), ""]
        if self.genus is None:
            if self.unresolved:
                lines.append("Still not manifold at:")
                for spot in self.unresolved[:10]:
                    lines.append("  ("
                                 + ", ".join(f"{v:.3g}" for v in spot) + ")")
                if len(self.unresolved) > 10:
                    lines.append(f"  … and {len(self.unresolved) - 10} more")
                lines.append("")
            lines.append(
                "Run this on the mesh as loaded, before cleaning. A weld "
                "that has already fused a contact leaves nothing for the "
                "separating pass, and a contact the weld refused stays "
                "refused — so a cleaned mesh can be harder to measure "
                "than the one it came from.")
            lines.append("")
        if self.pool_ml:
            pools = ", ".join(f"{v:.1f} mL" for v in self.pool_ml)
            areas = ", ".join(f"{v:.0f} mm²" for v in self.opening_mm2)
            lines.append(f"Cavities: {pools}")
            lines.append(f"Openings: {areas}")
        if self.joins:
            lines.append("")
            lines.append(describe_passages(self.joins,
                                           keep=len(self.opening_mm2)))
        if self.repaired:
            lines.append("")
            lines.append(f"Measured on a fully repaired copy; your mesh is "
                         f"unchanged. The copy needed: {self.repaired}")
        return "\n".join(lines)


def check_wall(grid,
               options: Optional[OrificeOptions] = None,
               on_status: Optional[Callable[[str], None]] = None
               ) -> WallReport:
    """Measure *grid*'s wall on a fully repaired copy. *grid* is not modified."""
    options = options or OrificeOptions()
    if on_status is not None:
        on_status("Repairing a copy to measure it…")
    repaired, clean_report = clean(
        grid, CleanOptions(separate_touching=True))

    faces = boundary_faces(tetrahedra(repaired))
    genus = boundary_genus(faces)
    unresolved: List[np.ndarray] = []
    if genus is None:
        points = np.asarray(repaired.points, dtype=float)
        edges = non_manifold_edges(faces)
        unresolved.extend(points[edges].mean(axis=1))
        unresolved.extend(points[pinched_vertices(faces)])
    if on_status is not None:
        on_status("Looking for the openings…")
    found = find_openings(repaired, options)
    expected = (len(found.openings) - len(found.pool_ml)
                if found.openings else None)

    joins: List[Passage] = []
    if genus is not None and found.openings:
        if on_status is not None:
            on_status("Looking for ways through the wall…")
        joins = find_passages(repaired, options, openings=found.openings)

    report = WallReport(
        genus=genus, expected=expected,
        pool_ml=list(found.pool_ml),
        opening_mm2=[o.area_mm2 for o in found.openings],
        joins=joins,
        repaired=clean_report.repair.summary(),
        unresolved=unresolved)
    if on_status is not None:
        on_status(report.summary())
    return report


__all__ = ["WallReport", "check_wall"]
