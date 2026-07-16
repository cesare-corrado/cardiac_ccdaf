"""
Field transfer
==============

Carry a mesh's fields onto a different mesh of the same anatomy, where the
two share no vertex correspondence — what a segmentation round trip returns,
since marching cubes builds its surface from the voxels and knows nothing of
the Carto vertices it ultimately came from.

Each destination vertex is matched to the closest point on the source
*surface* (not the closest source vertex, which overshoots it by roughly a
third of an edge length), and the field is read there.

Two rules, following the association the field is stored under:

* **point fields are measurements** — Carto's ``LAT``, ``Bipolar``, … — and
  are interpolated linearly within the triangle the closest point landed in;
* **cell fields are labels** — ``elemTag`` — and are copied from the nearest
  cell, never interpolated. Averaging two labels invents a third.

So a categorical quantity belongs on the cells, which is where ``elemTag``
already lives and where anything joining it should go.

No-data
-------
Carto's sentinels arrive as NaN. Interpolating a triangle with one invalid
vertex would spread that NaN across the triangle's whole area, so the weights
are renormalised over the valid vertices instead and a value only goes NaN
when the triangle has nothing valid to offer.

``max_distance`` guards the other direction. Editing a segmentation — a
morphological closing, a painted region, a filled hole — creates surface the
mapping system never measured. Its closest point on the source is the rim of
whatever it grew from, so interpolation would hand it that rim's activation
times: in range, smoothly varying, and indistinguishable from real data.
Beyond ``max_distance`` a point field is NaN instead, which says what is
true — nothing was measured here. A faithful round trip stays well inside
any sane threshold (a 1mm voxelisation reproduces the wall to under 1mm), so
the guard fires only on geometry that was invented.

Labels are exempt: new wall is still part of the body, so inheriting the
nearest ``elemTag`` states a fact rather than fabricating a measurement.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pyvista as pv
import vtk
from scipy.spatial import cKDTree


# Bookkeeping the renderer stamps onto whatever mesh is on screen, for
# picking. It is not the mesh's data and must not be carried anywhere, or it
# resurfaces as a selectable field. mesh_postprocessor._transfer_arrays skips
# it for the same reason.
_INTERNAL_ARRAYS = frozenset({"render_idx"})


def _closest_on_surface(src: pv.PolyData, points: np.ndarray):
    """For each point, the closest point on ``src``, its cell, and the range.

    Returns ``(closest_xyz, cell_ids, distance)``.
    """
    locator = vtk.vtkCellLocator()
    locator.SetDataSet(src)
    locator.BuildLocator()

    n = len(points)
    closest = np.empty((n, 3), dtype=float)
    cells = np.empty(n, dtype=np.int64)
    dist2 = np.empty(n, dtype=float)

    c = [0.0, 0.0, 0.0]
    cid = vtk.mutable(0)
    sub = vtk.mutable(0)
    d2 = vtk.mutable(0.0)
    for i, p in enumerate(points):
        locator.FindClosestPoint(p, c, cid, sub, d2)
        closest[i] = c
        cells[i] = int(cid)
        dist2[i] = float(d2)
    return closest, cells, np.sqrt(dist2)


def _barycentric(closest: np.ndarray, tri_xyz: np.ndarray) -> np.ndarray:
    """Barycentric weights of each point within its triangle.

    ``tri_xyz`` is (n, 3, 3): per point, the three corners. A degenerate
    triangle has no barycentric frame; those fall back to its first corner,
    which is a nearest-vertex copy for that one point.
    """
    p0, p1, p2 = tri_xyz[:, 0], tri_xyz[:, 1], tri_xyz[:, 2]
    v0, v1, v2 = p1 - p0, p2 - p0, closest - p0
    d00 = np.einsum("ij,ij->i", v0, v0)
    d01 = np.einsum("ij,ij->i", v0, v1)
    d11 = np.einsum("ij,ij->i", v1, v1)
    d20 = np.einsum("ij,ij->i", v2, v0)
    d21 = np.einsum("ij,ij->i", v2, v1)
    denom = d00 * d11 - d01 * d01

    ok = np.abs(denom) > 1e-20
    safe = np.where(ok, denom, 1.0)
    v = (d11 * d20 - d01 * d21) / safe
    w = (d00 * d21 - d01 * d20) / safe
    u = 1.0 - v - w

    weights = np.stack([u, v, w], axis=1)
    weights[~ok] = (1.0, 0.0, 0.0)
    # The closest point is on the triangle, so the weights are already in
    # [0, 1] up to rounding; clip rather than let a -1e-16 flip a sign.
    return np.clip(weights, 0.0, 1.0)


def _triangle_vertices(src: pv.PolyData, cells: np.ndarray) -> np.ndarray:
    """The three vertex ids of each named cell, (n, 3)."""
    faces = np.asarray(src.faces).reshape(-1, 4)
    if faces.size and np.any(faces[:, 0] != 3):
        raise ValueError("source mesh must contain triangles only")
    return faces[cells, 1:].astype(np.int64)


def transfer_fields(src: pv.PolyData,
                    dst: pv.PolyData,
                    max_distance: Optional[float] = None,
                    on_status: Optional[Callable[[str], None]] = None) -> None:
    """Copy ``src``'s fields onto ``dst``, in place. ``src`` is not touched.

    ``max_distance`` — beyond which a point field is NaN rather than
    invented. ``None`` disables the guard and lets new surface inherit
    whatever it is nearest to.
    """
    if dst.n_points == 0 or src.n_points == 0 or src.n_cells == 0:
        return

    point_names = [n for n in src.point_data.keys() if n not in _INTERNAL_ARRAYS]
    cell_names = [n for n in src.cell_data.keys() if n not in _INTERNAL_ARRAYS]

    guarded = 0
    if point_names:
        pts = np.asarray(dst.points, dtype=float)
        closest, cells, dist = _closest_on_surface(src, pts)
        vids = _triangle_vertices(src, cells)
        weights = _barycentric(closest, np.asarray(src.points)[vids])

        too_far = (np.zeros(len(pts), dtype=bool) if max_distance is None
                   else dist > float(max_distance))
        guarded = int(too_far.sum())

        for name in point_names:
            arr = np.asarray(src.point_data[name], dtype=float)
            vals = arr[vids]                       # (n, 3) or (n, 3, k)
            w = weights if vals.ndim == 2 else weights[:, :, None]

            valid = np.isfinite(vals)
            wv = np.where(valid, w, 0.0)
            total = wv.sum(axis=1)
            out = np.divide((wv * np.nan_to_num(vals)).sum(axis=1), total,
                            out=np.full_like(total, np.nan),
                            where=total > 0.0)
            out[too_far] = np.nan
            dst.point_data[name] = out

    if cell_names and dst.n_cells:
        src_c = np.asarray(src.cell_centers().points)
        dst_c = np.asarray(dst.cell_centers().points)
        _, cid = cKDTree(src_c).query(dst_c, k=1)
        for name in cell_names:
            arr = np.asarray(src.cell_data[name])
            dst.cell_data[name] = arr[cid]         # dtype, and labels, intact

    if on_status is not None:
        note = ""
        if guarded:
            note = (f"; {guarded} of {dst.n_points} vertices sit further than "
                    f"{max_distance:g} from anything measured and were left "
                    f"as no-data")
        on_status(
            f"Transferred {len(point_names)} point and "
            f"{len(cell_names)} cell fields{note}."
        )


__all__ = ["transfer_fields"]
