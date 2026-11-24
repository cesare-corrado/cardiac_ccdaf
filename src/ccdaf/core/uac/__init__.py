"""
uac
===
Universal atrial coordinates (UAC) for a labelled left-atrial surface.

Each point of the atrium gets two coordinates on a fixed unit square:
``alpha`` from septal (0) to lateral (1) and ``beta`` from the posterior
mitral valve (0) over the roof (0.5) to the anterior mitral valve (1). The
veins and the appendage land in the same place in every patient, so the
square is a common frame for comparing atria. See
:mod:`ccdaf.core.uac.pipeline` for the method and
:mod:`ccdaf.core.uac.template` for the layout.

This module holds the parts that touch a pyvista mesh; the computation
itself works on plain arrays.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pyvista as pv

from ccdaf.core.uac import template
from ccdaf.core.uac.pipeline import (
    ALPHA, BETA, TAG, Options, UacError, UacRun, compute, flat_from_values, prepare,
)
from ccdaf.core.uac.solvers import available as available_solvers

REQUIRED = (template.BODY,) + template.VEINS + (template.LAA,)


def _triangles(mesh: pv.PolyData) -> np.ndarray:
    faces = np.asarray(mesh.faces)
    if not len(faces) or (faces.reshape(-1, 4)[:, 0] != 3).any():
        raise ValueError("UAC need a surface made of triangles only.")
    return faces.reshape(-1, 4)[:, 1:].astype(np.int64)


def availability(dataset) -> Tuple[bool, str]:
    """Whether Compute UAC applies to *dataset*, and if not, why."""
    if dataset is None:
        return False, "Load a mesh first."
    if not isinstance(dataset, pv.PolyData):
        return False, "UAC are computed on an atrial surface, not a volume."
    if TAG not in dataset.cell_data:
        return False, "The surface has no elemTag labels."
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
    uv, tri, cells, src = flat_from_values(a, b, _triangles(mesh))
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


__all__ = ["availability", "inputs", "compute", "prepare", "write", "written_names",
           "has_coordinates", "flat_mesh", "flat_from_values",
           "Options", "UacRun", "UacError", "available_solvers", "ALPHA", "BETA", "template"]
