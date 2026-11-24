"""
right
=====
Universal atrial coordinates for a labelled right-atrial surface.

Input: a triangle surface with ``elemTag`` 2 (body), 21 (SVC), 23 (IVC),
25 (CS) and 27 (RAA), open at the tricuspid valve and at the end of each
vessel (SVC, IVC, CS). Lengths in millimetres. No landmarks.

Output: ``uac_alpha`` from the lateral tricuspid valve (0) over the roof
between the venae cavae (0.5) to the septal tricuspid valve (1), and
``uac_beta`` from the IVC side (0) to the SVC side (1), with every vessel
and the appendage at its fixed place (:mod:`ccdaf.core.uac.template_ra`).

The run is the left atrium's (:mod:`ccdaf.core.uac.pipeline`): the same
repairs, vessel extensions, fold-free first map and template warp, with the
right atrium's cut (:mod:`ccdaf.core.uac.cut_ra`) and template. It is done
in the left atrium's working frame and transposed at the end, so the
halves of the run (``run.attempt.halves``) keep working-frame values; the
point arrays ``run.alpha`` and ``run.beta`` are the right atrium's.

A point on the cut has two values; the point arrays carry the lateral
one. Its septal copy is ``(1 - alpha, beta)`` on the lines from the cavae
to the valve (``beta`` 0 or 1) and the same value on the roof;
:func:`flat_from_values` recovers it.
"""
from __future__ import annotations

import dataclasses
from typing import Callable, Optional, Tuple

import numpy as np
import pyvista as pv

from ccdaf.core.uac import cut_ra
from ccdaf.core.uac import pipeline as pl
from ccdaf.core.uac import template_ra as tr

TAG = pl.TAG
REQUIRED = (tr.BODY,) + tr.VESSELS + (tr.RAA,)

RIGHT = pl.Atrium(layout=tr.LAYOUT, vessels=tr.VESSELS, sleeve_ratio=tr.SLEEVE_RATIO,
                  check_sides=cut_ra.check_sides, anatomy=cut_ra.anatomy,
                  find_cut=cut_ra.find_cut, half_names=("Lateral", "Septal"),
                  roof_first=False)


def availability(dataset) -> Tuple[bool, str]:
    """Whether right-atrial UAC apply to *dataset*, and if not, why."""
    if dataset is None:
        return False, "Load a mesh first."
    if not isinstance(dataset, pv.PolyData):
        return False, "UAC are computed on an atrial surface, not a volume."
    if TAG not in dataset.cell_data:
        return False, "The surface has no elemTag labels."
    present = set(np.unique(np.asarray(dataset.cell_data[TAG]).astype(int)).tolist())
    missing = [tr.NAMES[L] for L in REQUIRED if L not in present]
    if missing:
        return False, ("Label the right atrium first: missing " + ", ".join(missing)
                       + f" (body {tr.BODY}, SVC {tr.SVC}, IVC {tr.IVC}, CS {tr.CS}, RAA {tr.RAA}).")
    return True, ""


def compute(points: np.ndarray, triangles: np.ndarray, tags: np.ndarray,
            options: Optional[pl.Options] = None, *,
            on_status: Optional[Callable[[str], None]] = None,
            should_stop: Optional[Callable[[], bool]] = None) -> pl.UacRun:
    """Right-atrial UAC on a labelled surface. Raises
    :class:`~ccdaf.core.uac.pipeline.UacError`."""
    run = pl.run_atrium(RIGHT, points, triangles, tags, options, on_status=on_status,
                        should_stop=should_stop)
    return dataclasses.replace(run, alpha=run.beta, beta=run.alpha)


def flat_from_values(alpha: np.ndarray, beta: np.ndarray, t: np.ndarray, eps: float = 1e-9):
    """The flat map from the stored right-atrial coordinates; returns what
    :func:`ccdaf.core.uac.pipeline.flat_from_values` returns, with points
    as ``(alpha, beta)``."""
    uv, tri, cells, src = pl.flat_from_values(beta, alpha, t, eps)
    return uv[:, ::-1].copy(), tri, cells, src


__all__ = ["RIGHT", "REQUIRED", "availability", "compute", "flat_from_values"]
