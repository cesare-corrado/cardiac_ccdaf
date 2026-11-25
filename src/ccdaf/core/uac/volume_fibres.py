"""
volume_fibres
=============
Fibres in the wall of an atrial volume, turning smoothly from the
endocardial atlas fibres to the epicardial ones through the wall.

Needs the volume UAC (:mod:`ccdaf.core.uac.volume`) and the labelled
surfaces.

The steps
---------
1. **Fibres on the two surfaces.** The endocardial atlas is mapped onto the
   volume's endocardium and the epicardial atlas onto its epicardium, by
   the surface mapping (:func:`ccdaf.core.uac.fibres.map_layer`),
   unchanged. Each surface reads its α and β from the volume's nodes,
   stored as the surface UAC store them (a node on the cut keeps its
   posterior value), so the flat map is built the same way.
2. **Into each tetrahedron:** the fibre of the endocardial triangle and of
   the epicardial triangle nearest its centre. This is what the published
   code of Roney et al. 2023 does.
3. **Through the wall:** instead of switching from the endocardial to the
   epicardial fibre at half depth, the fibre turns by an angle that grows
   linearly with the depth d (0 endocardium, 1 epicardium), in the plane
   of the wall. With n the unit gradient of the depth (the direction
   through the wall), both fibres are projected onto the plane normal to
   n, the epicardial one is flipped if needed (a fibre is a line, not an
   arrow), and the endocardial one is turned about n by d times the angle
   between them.
4. **Sheet:** n. The fibre is normal to it by construction. A tetrahedron
   whose four nodes lie on one surface has a flat depth and no n of its
   own: it takes its neighbours' sheet.

Fibres are written as ``fiber`` and ``sheet`` on the tetrahedra, the
names the ventricular fibres use, so the CARP export writes them as they
are.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

from ccdaf.core.chamber_extract import _shared_faces
from ccdaf.core.surface_labels import (
    EPI, LA_ENDO, LABEL_MASK_FIELD, RA_ENDO, face_labels_from_mask,
)
from ccdaf.core.uac import fibres as surface_fibres
from ccdaf.core.uac import volume as vuac
from ccdaf.core.volume_mesh import boundary_surface, is_volume, tetrahedra

FIBRE, SHEET = "fiber", "sheet"


@dataclass
class VolumeFibres:
    """What :func:`compute` produced."""

    fibre: np.ndarray            # (n_tets, 3) unit
    sheet: np.ndarray            # (n_tets, 3) unit
    reports: List[object] = field(default_factory=list)   # FibreReport per surface
    turn_deg: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: Tetrahedra with a flat depth that took their neighbours' sheet.
    filled: int = 0

    def lines(self) -> List[str]:
        out = [ln for r in self.reports for ln in r.lines()]
        if len(self.turn_deg):
            q = np.percentile(self.turn_deg, [50, 90])
            out.append(f"Endocardial to epicardial fibre turn through the wall: "
                       f"median {q[0]:.0f}°, 90% {q[1]:.0f}°.")
        if self.filled:
            out.append(f"{self.filled} tetrahedra lying flat on one surface took "
                       "their neighbours' sheet.")
        return out


def availability(dataset) -> Tuple[bool, str]:
    """Whether volume fibres can be computed on *dataset*, and if not, why."""
    if dataset is None or not is_volume(pv.wrap(dataset)):
        return False, "Load an atrial volume first."
    grid = pv.wrap(dataset)
    if vuac.DEPTH not in grid.point_data or vuac.ALPHA not in grid.point_data:
        return False, "Compute the volume UAC first (Actions > Compute UAC…)."
    if LABEL_MASK_FIELD not in grid.point_data:
        return False, "Label the atrial surfaces first."
    return True, ""


def _surface_with_uac(grid, surface, labels, key) -> Tuple[pv.PolyData, np.ndarray]:
    """The boundary faces labelled *key* as a surface carrying the volume's
    α and β on its points and the tetrahedra's labels on its faces; and
    the face centres."""
    origin = np.asarray(surface.point_data["vtkOriginalPointIds"], dtype=np.int64)
    tri = np.asarray(surface.faces).reshape(-1, 4)[:, 1:]
    faces = np.where(labels == key)[0]
    nodes, local = np.unique(tri[faces], return_inverse=True)
    local = local.reshape(-1, 3)
    out = pv.PolyData(np.asarray(surface.points, dtype=float)[nodes],
                      np.hstack([np.full((len(local), 1), 3), local]).ravel())
    out.point_data[vuac.ALPHA] = np.asarray(grid.point_data[vuac.ALPHA])[origin[nodes]]
    out.point_data[vuac.BETA] = np.asarray(grid.point_data[vuac.BETA])[origin[nodes]]
    out.cell_data["elemTag"] = np.asarray(surface.cell_data["elemTag"])[faces]
    return out, np.asarray(out.points)[local].mean(1)


def _depth_gradient(points: np.ndarray, tets: np.ndarray, depth: np.ndarray
                    ) -> np.ndarray:
    """Per tetrahedron, the gradient of the (linear) depth."""
    e = points[tets[:, 1:]] - points[tets[:, :1]]
    dd = depth[tets[:, 1:]] - depth[tets[:, :1]]
    return np.linalg.solve(e, dd[..., None])[..., 0]


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 1e-12, v / n, np.nan)


def blend(f_endo: np.ndarray, f_epi: np.ndarray, normal: np.ndarray,
          depth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The fibre at *depth*: *f_endo* turned about *normal* towards *f_epi*
    by *depth* times the angle between them, both first projected onto the
    plane normal to *normal*. Returns the fibres and the full angles (deg).
    """
    n = _unit(normal)
    a = _unit(f_endo - np.sum(f_endo * n, 1, keepdims=True) * n)
    b = _unit(f_epi - np.sum(f_epi * n, 1, keepdims=True) * n)
    # A fibre is a line: take the epicardial direction nearer the endocardial.
    flip = np.sum(a * b, 1) < 0
    b[flip] *= -1
    angle = np.arctan2(np.sum(n * np.cross(a, b), 1), np.sum(a * b, 1))
    t = np.clip(depth, 0.0, 1.0) * angle
    f = (np.cos(t)[:, None] * a + np.sin(t)[:, None] * np.cross(n, a))
    return _unit(f), np.degrees(np.abs(angle))


def compute(grid) -> VolumeFibres:
    """Fibres and sheets of an atrial volume with volume UAC.

    Raises ``ValueError`` (or the atlas' ``AtlasError``) saying why not."""
    grid = pv.wrap(grid)
    ok, why = availability(grid)
    if not ok:
        raise ValueError(why)
    surface = boundary_surface(grid)
    origin = np.asarray(surface.point_data["vtkOriginalPointIds"], dtype=np.int64)
    labels = face_labels_from_mask(
        surface, np.asarray(grid.point_data[LABEL_MASK_FIELD])[origin])
    endo_key = LA_ENDO if np.any(labels == LA_ENDO) else RA_ENDO
    if np.any(labels == LA_ENDO) and np.any(labels == RA_ENDO):
        raise ValueError("This volume holds both atria: fibres are computed on "
                         "one atrium at a time.")

    reports = []
    picked = []
    tets = tetrahedra(grid)
    X = np.asarray(grid.points, dtype=float)
    centres = X[tets].mean(1)
    for key, layer in ((endo_key, "endo"), (EPI, "epi")):
        surf, face_centres = _surface_with_uac(grid, surface, labels, key)
        f, report = surface_fibres.map_layer(surf, layer)
        reports.append(report)
        _d, nearest = cKDTree(face_centres).query(centres)
        picked.append(f[nearest])

    depth = np.asarray(grid.point_data[vuac.DEPTH], dtype=float)
    normal = _depth_gradient(X, tets, depth)
    fibre, turn = blend(picked[0], picked[1], normal, depth[tets].mean(1))
    sheet = _unit(normal)
    # A tetrahedron whose four nodes lie on one surface has a flat depth and
    # so no direction through the wall (913 of 223,067 on one atrium): it
    # takes its neighbours' sheet, and its fibre is blended about that.
    flat = ~np.isfinite(sheet).all(1)
    filled = 0
    if flat.any():
        _faces, a, b = _shared_faces(tets)
        filled = _fill_from_neighbours(sheet, a, b)
        left = ~np.isfinite(sheet).all(1)
        if left.any() and (~left).any():
            # No face neighbour with a sheet: the nearest tetrahedron's.
            _d, j = cKDTree(centres[~left]).query(centres[left])
            sheet[left] = sheet[np.where(~left)[0][j]]
            filled += int(left.sum())
        redo = flat & np.isfinite(sheet).all(1)
        fibre[redo], _t = blend(picked[0][redo], picked[1][redo], sheet[redo],
                                depth[tets[redo]].mean(1))
    bad = ~np.isfinite(fibre).all(1)
    if bad.any():
        near_epi = depth[tets].mean(1) > 0.5
        fallback = np.where(near_epi[:, None], picked[1], picked[0])
        fibre[bad] = _unit(fallback[bad])
    sheet[~np.isfinite(sheet).all(1)] = 0.0
    fibre[~np.isfinite(fibre).all(1)] = 0.0
    return VolumeFibres(fibre=fibre, sheet=sheet, reports=reports,
                        turn_deg=turn[np.isfinite(turn)], filled=filled)


def _fill_from_neighbours(v: np.ndarray, a: np.ndarray, b: np.ndarray,
                          passes: int = 20) -> int:
    """Give each row of *v* that is not finite the mean of its finite
    face neighbours' (signs aligned: an axis, not an arrow), from the edge
    of each gap inwards. In place; returns how many were filled."""
    bad = ~np.isfinite(v).all(1)
    n0 = int(bad.sum())
    for _ in range(passes):
        if not bad.any():
            break
        pairs = np.concatenate([np.stack([a, b], 1), np.stack([b, a], 1)])
        pairs = pairs[bad[pairs[:, 0]] & ~bad[pairs[:, 1]]]
        if len(pairs) == 0:
            break
        acc = np.zeros_like(v)
        ref = np.full_like(v, np.nan)
        order = np.argsort(pairs[:, 0], kind="stable")
        pairs = pairs[order]
        first = np.r_[True, pairs[1:, 0] != pairs[:-1, 0]]
        ref[pairs[first, 0]] = v[pairs[first, 1]]
        w = v[pairs[:, 1]]
        sign = np.sign(np.sum(w * ref[pairs[:, 0]], 1))
        sign[sign == 0] = 1
        np.add.at(acc, pairs[:, 0], w * sign[:, None])
        rows = np.unique(pairs[:, 0])
        v[rows] = _unit(acc[rows])
        bad = ~np.isfinite(v).all(1)
    return n0 - int(bad.sum())


def write(grid, result: VolumeFibres):
    """*grid* with ``fiber`` and ``sheet`` on its tetrahedra (in place)."""
    grid.cell_data[FIBRE] = result.fibre
    grid.cell_data[SHEET] = result.sheet
    return grid


__all__ = ["FIBRE", "SHEET", "VolumeFibres", "availability", "blend",
           "compute", "write"]
