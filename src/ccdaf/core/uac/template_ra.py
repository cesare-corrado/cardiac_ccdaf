"""
template_ra
===========
The fixed square every right atrium is mapped onto.

``alpha`` runs 0 (lateral tricuspid valve) through 0.5 (the roof, the line
between the venae cavae) to 1 (septal tricuspid valve); ``beta`` runs 0
(IVC side) to 1 (SVC side). The lateral wall is ``alpha`` 0 to 0.5 and
holds the appendage; the septal wall is ``alpha`` 0.5 to 1 and holds the
coronary sinus. These are the directions of Roney et al. (2019).

The places are those of the published fixed-structure right-atrial
coordinates (Roney et al. 2021), measured on the 16 published right-atrial
surfaces (7 hearts and the atlas, endocardium and epicardium): every vessel
end sits on the same circle in all of them, and every ostium stays inside
the same box.

* SVC and IVC: half-ellipses standing on the roof line and bulging into the
  septal wall, 0.115 along the roof and 0.11 across it, centred 0.115 from
  the top and bottom edges. Ends: circles of radius 0.02 at
  ``(0.555, 0.885)`` and ``(0.555, 0.115)``.
* CS: a circle of radius 0.03 at ``(0.85, 0.38)``; end radius 0.015.
* RAA: an ellipse at ``(0.135, 0.839)``, 0.075 across (alpha) and 0.069
  along (beta) the square.

As on the left atrium, the venae cavae sit :data:`~ccdaf.core.uac.template.GAP`
in from the top and bottom edges, so neither touches a corner.

The map itself is computed in the left atrium's working frame (see
:mod:`ccdaf.core.uac.layout`) and transposed at the end:
``alpha = v``, ``beta = u``. In that frame the RA is the LA with the SVC in
place of the LSPV, the IVC in place of the RSPV, the appendage mirrored
into the first half, and one vessel (the CS) in the second half instead of
two in the first.

Labels (``elemTag``): body 2, SVC 21, IVC 23, CS 25, RAA 27. The tricuspid
valve is an open boundary.
"""
from __future__ import annotations

from typing import Dict, Tuple

from ccdaf.core.uac import template as la
from ccdaf.core.uac.layout import Closed, Layout

from ccdaf.core.atrial_labels import RA as _RA  # noqa: E402

#: The right-atrial labels, from ccdaf.core.atrial_labels.
BODY = _RA.body
SVC, IVC, CS, RAA = (_RA.regions[k] for k in ("SVC", "IVC", "CS", "RAA"))
VESSELS: Tuple[int, ...] = (SVC, IVC, CS)
NAMES: Dict[int, str] = {BODY: "body", SVC: "SVC", IVC: "IVC", CS: "CS", RAA: "RAA"}

# All places below in the square's own coordinates (alpha, beta).
CAVA_ALONG: float = 0.115 - la.GAP       #: half-ellipse semi-axis along the roof (beta)
CAVA_ACROSS: float = 0.11                #: semi-axis into the septal wall (alpha)
SVC_C: Tuple[float, float] = (0.5, 1 - 0.115)
IVC_C: Tuple[float, float] = (0.5, 0.115)
CAVA_END_R: float = 0.02
SVC_END_C: Tuple[float, float] = (0.555, 1 - 0.115)
IVC_END_C: Tuple[float, float] = (0.555, 0.115)

CS_C: Tuple[float, float] = (0.85, 0.38)
CS_R: float = 0.03
CS_END_R: float = 0.015

RAA_C: Tuple[float, float] = (0.135, 0.839)
RAA_R_ALPHA: float = 0.075
RAA_R_BETA: float = 0.069

#: Sleeve length the template's annulus implies, as a fraction of the
#: ostium's circumference (see :mod:`ccdaf.core.uac.extension`). The cavae
#: have the left atrium's superior-vein shape, so they take its value. The
#: CS value scales the left atrium's inferior-vein value (0.175 for an
#: annulus 0.04 wide on a circle of radius 0.06) to the CS annulus
#: (0.015 wide on 0.03); it is a first estimate, not tuned.
SLEEVE_RATIO: Dict[int, float] = {SVC: 0.22, IVC: 0.22, CS: 0.13}


def _w(p: Tuple[float, float]) -> Tuple[float, float]:
    """A point of the square in the working frame ``(u, v) = (beta, alpha)``."""
    return (p[1], p[0])


LAYOUT = Layout(
    roof_far=SVC, roof_near=IVC,
    far_c=_w(SVC_C), near_c=_w(IVC_C), sup_ra=CAVA_ALONG, sup_rb=CAVA_ACROSS,
    closed={RAA: Closed(0, _w(RAA_C), RAA_R_BETA, RAA_R_ALPHA),
            CS: Closed(1, _w(CS_C), CS_R, CS_R)},
    ends={SVC: (_w(SVC_END_C), CAVA_END_R), IVC: (_w(IVC_END_C), CAVA_END_R),
          CS: (_w(CS_C), CS_END_R)},
    names=NAMES,
    # On the right atrium the warp did not always reach the cut's template
    # places within its iterations (up to 3e-3 off, at the IVC end of the
    # roof), and the two halves then disagreed on the cut.
    exact_cut=True,
)

__all__ = ["BODY", "SVC", "IVC", "CS", "RAA", "VESSELS", "NAMES", "CAVA_ALONG", "CAVA_ACROSS",
           "SVC_C", "IVC_C", "CAVA_END_R", "SVC_END_C", "IVC_END_C", "CS_C", "CS_R", "CS_END_R",
           "RAA_C", "RAA_R_ALPHA", "RAA_R_BETA", "SLEEVE_RATIO", "LAYOUT"]
