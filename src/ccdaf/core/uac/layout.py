"""
layout
======
Where each ring goes on the square, as data the flat map reads.

Both atria are cut open the same way and laid out in the same working
frame: ``u`` along the roof, ``v`` across it, the first half (``v`` 0 to
0.5) below the roof and the second half (``v`` 0.5 to 1) above it. Two
vessels sit on the roof line, each a half-ellipse standing on it and
bulging into the second half: one near ``u = 1`` (where the first side line
of the cut ends) and one near ``u = 0`` (where the second side line starts).
Every other structure is a closed ring (circle or ellipse) inside one half.
Open vessels also have an end: a small circle the vessel's open rim lands on.

For the left atrium the working frame is the coordinate frame itself
(``alpha = u``, ``beta = v``). For the right atrium it is transposed (see
:mod:`ccdaf.core.uac.template_ra`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

from ccdaf.core.uac import template as tpl

Point = Tuple[float, float]


@dataclass(frozen=True)
class Closed:
    """A closed ring: an ellipse with centre ``c`` and semi-axes ``ra``
    (along ``u``) and ``rb`` (along ``v``), in half ``side`` (0 or 1)."""

    side: int
    c: Point
    ra: float
    rb: float


@dataclass(frozen=True)
class Layout:
    """The template of one atrium, in the working frame.

    ``roof_far`` and ``roof_near``: the labels of the two vessels on the roof
    (near ``u = 1`` and ``u = 0``), with their centres ``far_c``/``near_c``
    and the semi-axes of their half-ellipses (``sup_ra`` along the roof,
    ``sup_rb`` into the second half). ``closed``: every other ring, by label,
    in the order its targets are set. ``ends``: centre and radius of each
    open vessel's end. ``gap``: the stretch of each side line laid along the
    roof (see :data:`ccdaf.core.uac.template.GAP`)."""

    roof_far: int
    roof_near: int
    far_c: Point
    near_c: Point
    sup_ra: float
    sup_rb: float
    closed: Dict[int, Closed]
    ends: Dict[int, Tuple[Point, float]]
    gap: float = tpl.GAP
    names: Dict[int, str] = field(default_factory=dict)
    #: Start the warp from a map whose cut points are already at their
    #: template places (the rest shape stays the arc-length first map). The
    #: warp then never moves them, so both halves hold every cut point at
    #: exactly the same place. Off for the left atrium, whose results were
    #: validated without it (there the warp reaches the template within
    #: about 1e-6).
    exact_cut: bool = False


#: The left atrium: LSPV on the far end of the roof, RSPV on the near end;
#: the inferior veins in the posterior half, the appendage in the anterior.
LA = Layout(
    roof_far=tpl.LSPV, roof_near=tpl.RSPV,
    far_c=tpl.LSPV_C, near_c=tpl.RSPV_C, sup_ra=tpl.SUP_RA, sup_rb=tpl.SUP_RB,
    closed={tpl.LIPV: Closed(0, tpl.LIPV_C, tpl.INF_R, tpl.INF_R),
            tpl.RIPV: Closed(0, tpl.RIPV_C, tpl.INF_R, tpl.INF_R),
            tpl.LAA: Closed(1, tpl.LAA_C, tpl.LAA_RA, tpl.LAA_RB)},
    ends={L: (c, tpl.END_R) for L, c in tpl.END_C.items()},
    names=tpl.NAMES,
)

__all__ = ["Closed", "Layout", "LA"]
