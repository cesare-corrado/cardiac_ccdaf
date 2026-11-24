"""
uac
===
Universal atrial coordinates (UAC) for a labelled atrial surface, left or
right; which one is read from the labels (:mod:`ccdaf.core.atrial_labels`).

Each point of the atrium gets two coordinates on a fixed unit square. Left
atrium: ``alpha`` from septal (0) to lateral (1) and ``beta`` from the
posterior mitral valve (0) over the roof (0.5) to the anterior mitral valve
(1). Right atrium (:mod:`ccdaf.core.uac.right`): ``alpha`` from the lateral
tricuspid valve (0) over the roof (0.5) to the septal tricuspid valve (1)
and ``beta`` from the IVC (0) to the SVC (1). Every vessel and appendage
lands in the same place in every patient, so the square is a common frame
for comparing atria. See :mod:`ccdaf.core.uac.pipeline` for the method and
:mod:`ccdaf.core.uac.template` / :mod:`ccdaf.core.uac.template_ra` for the
layouts. A surface holding both atria is refused: the two together are not
supported yet.

This module holds the parts that touch a pyvista mesh; the computation
itself works on plain arrays.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pyvista as pv

from ccdaf.core import atrial_labels as al
from ccdaf.core.uac import template
from ccdaf.core.uac import pipeline as _pipeline
from ccdaf.core.uac.pipeline import (
    ALPHA, BETA, TAG, Options, UacError, UacRun, flat_from_values, prepare,
)
from ccdaf.core.uac.solvers import available as available_solvers

REQUIRED = (template.BODY,) + template.VEINS + (template.LAA,)


def _triangles(mesh: pv.PolyData) -> np.ndarray:
    faces = np.asarray(mesh.faces)
    if not len(faces) or (faces.reshape(-1, 4)[:, 0] != 3).any():
        raise ValueError("UAC need a surface made of triangles only.")
    return faces.reshape(-1, 4)[:, 1:].astype(np.int64)


BOTH_ATRIA = ("This surface holds labels of both atria. Compute UAC works on one "
              "atrium at a time; the two atria together are not supported yet.")


def which_atrium(tags) -> str:
    """``"LA"``, ``"RA"`` or ``"both"``, from the region labels present: an
    atrium counts when one of its regions is there. ``"LA"`` when neither
    is (an untagged surface is offered the left atrium, as before)."""
    present = set(np.unique(np.asarray(tags).astype(int)).tolist())
    la = bool(present & set(al.LA.regions.values()))
    ra = bool(present & set(al.RA.regions.values()))
    return "both" if la and ra else ("RA" if ra else "LA")


def availability(dataset) -> Tuple[bool, str]:
    """Whether Compute UAC applies to *dataset*, and if not, why."""
    if dataset is None:
        return False, "Load a mesh first."
    if not isinstance(dataset, pv.PolyData):
        return False, "UAC are computed on an atrial surface, not a volume."
    if TAG not in dataset.cell_data:
        return False, "The surface has no elemTag labels."
    side = which_atrium(dataset.cell_data[TAG])
    if side == "both":
        return False, BOTH_ATRIA
    if side == "RA":
        from ccdaf.core.uac import right
        return right.availability(dataset)
    present = set(np.unique(np.asarray(dataset.cell_data[TAG]).astype(int)).tolist())
    missing = [template.NAMES[L] for L in REQUIRED if L not in present]
    if missing:
        return False, ("Label the left atrium first: missing " + ", ".join(missing)
                       + " (body 1, LSPV 11, LIPV 13, RSPV 15, RIPV 17, LAA 19).")
    return True, ""


def inputs(mesh: pv.PolyData) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(points, triangles, tags)`` copied from *mesh*."""
    return (np.asarray(mesh.points, float).copy(), _triangles(mesh),
            np.asarray(mesh.cell_data[TAG]).astype(int).copy())


def compute(points: np.ndarray, triangles: np.ndarray, tags: np.ndarray,
            options: Optional[Options] = None, **kw) -> UacRun:
    """UAC of the atrium the labels describe: the left atrium
    (:func:`ccdaf.core.uac.pipeline.compute`, unchanged) or the right one
    (:func:`ccdaf.core.uac.right.compute`). Raises :class:`UacError`, also
    for a surface holding both atria."""
    side = which_atrium(tags)
    if side == "both":
        raise UacError(BOTH_ATRIA)
    if side == "RA":
        from ccdaf.core.uac import right
        return right.compute(points, triangles, tags, options, **kw)
    return _pipeline.compute(points, triangles, tags, options, **kw)


def write(mesh: pv.PolyData, run: UacRun) -> pv.PolyData:
    """The repaired surface of *run*, carrying every field of *mesh*, with
    ``uac_alpha`` and ``uac_beta`` added. *mesh* itself is not changed.

    *mesh* must be the surface the run was computed from."""
    from ccdaf.core import surface_repair
    out = surface_repair.apply(mesh, run.surface, label=TAG)
    out.point_data[ALPHA] = run.alpha
    out.point_data[BETA] = run.beta
    return out


def written_names() -> Tuple[str, ...]:
    return (ALPHA, BETA)


def has_coordinates(mesh) -> bool:
    """Whether *mesh* carries both coordinate arrays (the flat map can be
    drawn from them)."""
    try:
        names = set(mesh.point_data.keys())
    except AttributeError:
        return False
    return ALPHA in names and BETA in names and isinstance(mesh, pv.PolyData)


def flat_mesh(mesh: pv.PolyData) -> pv.PolyData:
    """The flat map of an atrium that has ``uac_alpha`` and ``uac_beta``, as
    a mesh in the plane ``z = 0`` (see :func:`flat_from_values`).

    Points on the cut appear once per half. Point data ``uac_source`` is the
    index of the atrium's point each flat point comes from, so any field of
    the atrium can be carried across; every other point field of the atrium
    is carried as well, and every cell field (``elemTag`` included).
    """
    a = np.asarray(mesh.point_data[ALPHA], float)
    b = np.asarray(mesh.point_data[BETA], float)
    flat = flat_from_values
    if TAG in mesh.cell_data and which_atrium(mesh.cell_data[TAG]) == "RA":
        from ccdaf.core.uac import right
        flat = right.flat_from_values
    uv, tri, cells, src = flat(a, b, _triangles(mesh))
    m = pv.PolyData(np.c_[uv, np.zeros(len(uv))], np.c_[np.full(len(tri), 3), tri].ravel())
    for name in mesh.point_data.keys():
        if name not in (ALPHA, BETA):
            m.point_data[name] = np.asarray(mesh.point_data[name])[src]
    for name in mesh.cell_data.keys():
        m.cell_data[name] = np.asarray(mesh.cell_data[name])[cells]
    m.point_data["uac_source"] = src
    m.point_data[ALPHA] = uv[:, 0]
    m.point_data[BETA] = uv[:, 1]
    return m


__all__ = ["availability", "which_atrium", "BOTH_ATRIA", "inputs", "compute", "prepare", "write", "written_names",
           "has_coordinates", "flat_mesh", "flat_from_values",
           "Options", "UacRun", "UacError", "available_solvers", "ALPHA", "BETA", "template"]
