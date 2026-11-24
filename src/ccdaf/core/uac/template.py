"""
template
========
The fixed square every left atrium is mapped onto.

``alpha`` runs 0 (septal, right veins) to 1 (lateral, left veins); ``beta``
runs 0 (posterior mitral valve) through 0.5 (roof) to 1 (anterior mitral
valve). The posterior half is ``beta`` 0 to 0.5, the anterior half 0.5 to 1.

Every patient lands on the same layout, so a point of the square means the
same place in every atrium. The layout is the fixed-vein ("version 2")
layout of the code that accompanies Roney et al. (2019), with one change: the superior veins sit
:data:`GAP` in from the side edges instead of touching them. A vein that
touches a corner of the square forces a cusp that no fold-free map can
reach; the gap removes it at the cost of a strip 1.5 % of the roof wide.

Vein openings (the "ends") are small circles inside each vein's disc: the
vein's own tissue fills the annulus between its ostium and its end.
"""
from __future__ import annotations

from typing import Dict, Tuple

BODY, LSPV, LIPV, RSPV, RIPV, LAA = 1, 11, 13, 15, 17, 19
VEINS: Tuple[int, ...] = (LSPV, LIPV, RSPV, RIPV)
NAMES: Dict[int, str] = {BODY: "body", LSPV: "LSPV", LIPV: "LIPV",
                         RSPV: "RSPV", RIPV: "RIPV", LAA: "LAA"}

#: Distance of each superior vein from its side edge, as a fraction of the
#: square. Also the length, as a fraction of the roof, of the end of each
#: side line that is laid along the roof.
GAP: float = 0.015

# Inferior veins (posterior half): full circles.
INF_R: float = 0.06
LIPV_C: Tuple[float, float] = (0.87, 0.315)
RIPV_C: Tuple[float, float] = (0.13, 0.315)

# Superior veins (anterior half): half-ellipses standing on the roof line.
SUP_RA: float = 0.115 - GAP        # along the roof
SUP_RB: float = 0.11               # into the anterior half
LSPV_C: Tuple[float, float] = (1 - 0.115, 0.5)
RSPV_C: Tuple[float, float] = (0.115, 0.5)

# Vein ends: small circles; the superior ones just above the roof.
END_R: float = 0.02
END_C: Dict[int, Tuple[float, float]] = {
    LIPV: LIPV_C, RIPV: RIPV_C,
    LSPV: (1 - 0.115, 0.555), RSPV: (0.115, 0.555),
}

# Appendage: an ellipse in the anterior half, lateral side.
LAA_RA: float = 0.069
LAA_RB: float = 0.075
LAA_C: Tuple[float, float] = (1 - (2 * 0.115 - LAA_RA), 0.865)

#: Length of each vein's sleeve, as a fraction of its ostium's
#: circumference, that the template's annulus implies. A shorter sleeve is
#: lengthened with a virtual extension so the annulus is not crushed.
SLEEVE_RATIO: Dict[int, float] = {LSPV: 0.22, RSPV: 0.22, LIPV: 0.175, RIPV: 0.175}

__all__ = [
    "BODY", "LSPV", "LIPV", "RSPV", "RIPV", "LAA", "VEINS", "NAMES", "GAP",
    "INF_R", "LIPV_C", "RIPV_C", "SUP_RA", "SUP_RB", "LSPV_C", "RSPV_C",
    "END_R", "END_C", "LAA_RA", "LAA_RB", "LAA_C", "SLEEVE_RATIO",
]
