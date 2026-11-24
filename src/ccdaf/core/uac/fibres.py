"""
fibres.py
=========
Fibres for a left atrium from a fibre atlas, through the universal atrial
coordinates (UAC).

Every triangle of the atrium is placed on the flat map by its stored
coordinates, and the atlas triangle at the same place gives its fibre:

1. The atlas fibre, a direction in its triangle, is written as a direction
   on the flat map through that triangle's own map from 3D to UAC.
2. That flat direction is turned back into 3D through the atrium
   triangle's map, and normalised. So a fibre keeps its course relative to
   the anatomy (round a vein, along the roof) whatever the shape.

A triangle whose place on the flat map has no atlas triangle (the atlas
veins are short, so the outer part of a long vein) takes the nearest atlas
triangle on the flat map: that one lies further in along the same vein, at
the same position round it, so a fibre running round or along the vein
keeps doing so. The report gives that area per label. A triangle squeezed
to nothing on the flat map has no direction of its own and takes the mean
of its neighbours.

Fibres are axial: ``f`` and ``-f`` are the same fibre.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pyvista as pv
import scipy.sparse as sp

from ccdaf.core.uac import fibre_atlas
from ccdaf.core.uac.pipeline import flat_from_values
from ccdaf.core.uac.template import NAMES

ALPHA, BETA = "uac_alpha", "uac_beta"
FIELDS = {"endo": "fibre_endo", "epi": "fibre_epi"}
LABEL = "elemTag"


@dataclass
class FibreReport:
    layer: str
    triangles: int
    extrapolated: Dict[str, float] = field(default_factory=dict)   # % of each label's area
    filled: int = 0                                                  # took the neighbours' mean

    def lines(self):
        out = [f"{self.layer}: {self.triangles} triangles mapped from the average atlas."]
        ext = {k: v for k, v in self.extrapolated.items() if v >= 0.05}
        if ext:
            out.append("Beyond the atlas veins (nearest atlas fibre used): "
                       + ", ".join(f"{k} {v:.1f}%" for k, v in ext.items()) + " of the area.")
        if self.filled:
            out.append(f"{self.filled} triangles squeezed on the flat map took their "
                       "neighbours' mean.")
        return out


def _faces(m: pv.PolyData) -> np.ndarray:
    return m.faces.reshape(-1, 4)[:, 1:].astype(np.int64)


def _flat(m: pv.PolyData):
    a = np.asarray(m.point_data[ALPHA], float)
    b = np.asarray(m.point_data[BETA], float)
    return flat_from_values(a, b, _faces(m))       # uv, flat triangles, 3D cell, 3D point


def _edges(X: np.ndarray, t: np.ndarray) -> np.ndarray:
    """(n, d, 2): the two edge vectors of each triangle as columns."""
    return np.stack([X[t[:, 1]] - X[t[:, 0]], X[t[:, 2]] - X[t[:, 0]]], axis=2)


def _to_flat(X3: np.ndarray, U: np.ndarray, t3: np.ndarray, tu: np.ndarray,
             f: np.ndarray) -> np.ndarray:
    """Each 3D direction *f* in its triangle, as a direction on the flat map."""
    E = _edges(X3, t3)                                   # 3 x 2
    D = _edges(U, tu)                                    # 2 x 2
    G = np.einsum("nki,nkj->nij", E, E)                  # E^T E
    with np.errstate(all="ignore"):
        c = np.linalg.solve(G, np.einsum("nki,nk->ni", E, f)[..., None])[..., 0]
    return np.einsum("nij,nj->ni", D, c)                 # edge coefficients -> flat


def _to_3d(X3: np.ndarray, U: np.ndarray, t3: np.ndarray, tu: np.ndarray,
           d: np.ndarray) -> np.ndarray:
    """Each flat direction *d* as a unit 3D direction in its triangle (NaN
    where the flat triangle is squeezed to nothing)."""
    E = _edges(X3, t3)
    D = _edges(U, tu)
    det = D[:, 0, 0] * D[:, 1, 1] - D[:, 0, 1] * D[:, 1, 0]
    scale = np.sqrt(np.abs(det)) + 1e-300
    ok = np.abs(det) > 1e-14
    out = np.full((len(t3), 3), np.nan)
    c = np.linalg.solve(D[ok] / scale[ok, None, None], d[ok][..., None] / scale[ok, None, None])[..., 0]
    v = np.einsum("nij,nj->ni", E[ok], c)
    n = np.linalg.norm(v, axis=1)
    good = n > 0
    idx = np.where(ok)[0]
    out[idx[good]] = v[good] / n[good, None]
    return out


def _fill(t: np.ndarray, f: np.ndarray, max_pass: int = 50) -> int:
    """Give each NaN fibre the mean of its neighbours' (sign-aligned), in
    passes from the edge of each gap inwards. Returns how many were filled."""
    bad = ~np.isfinite(f).all(1)
    n0 = int(bad.sum())
    if not n0:
        return 0
    nt, npts = len(t), int(t.max()) + 1
    inc = sp.csr_matrix((np.ones(3 * nt), (np.repeat(np.arange(nt), 3), t.ravel())),
                        shape=(nt, npts))
    adj = (inc @ inc.T).tocsr()
    for _ in range(max_pass):
        if not bad.any():
            break
        rows = np.where(bad)[0]
        new = {}
        for r in rows:
            nb = adj.indices[adj.indptr[r]:adj.indptr[r + 1]]
            v = f[nb[~bad[nb]]]
            if not len(v):
                continue
            v = v * np.sign(v @ v[0])[:, None]          # axial: one sign before the mean
            m = v.mean(0)
            if np.linalg.norm(m) > 0:
                new[r] = m / np.linalg.norm(m)
        if not new:
            break
        for r, m in new.items():
            f[r] = m
            bad[r] = False
    return n0 - int(bad.sum())


def map_layer(mesh: pv.PolyData, layer: str, atlas: Optional[pv.PolyData] = None):
    """``(fibres (n_cells, 3), FibreReport)`` for *layer* (``endo`` or
    ``epi``) on an atrium that carries ``uac_alpha`` and ``uac_beta``."""
    A = atlas if atlas is not None else fibre_atlas.load("atlas", layer)
    # atlas: one flat direction per atlas flat triangle
    auv, atri, acell, asrc = _flat(A)
    AX = np.asarray(A.points, float)
    af = np.asarray(A.cell_data["fibre"], float)[acell]
    a_flat = _to_flat(AX, auv, asrc[atri], atri, af)
    amesh = pv.PolyData(np.c_[auv, np.zeros(len(auv))], np.c_[np.full(len(atri), 3), atri].ravel())

    # atrium: where each triangle's centre falls on the atlas map
    uv, tri, cell, src = _flat(mesh)
    cen = np.c_[uv[tri].mean(1), np.zeros(len(tri))]
    hit = np.asarray(amesh.find_containing_cell(cen))
    out = hit < 0
    if out.any():
        hit[out] = np.asarray(amesh.find_closest_cell(cen[out]))
    X = np.asarray(mesh.points, float)
    fl = _to_3d(X, uv, src[tri], tri, a_flat[hit])
    fib = np.full((mesh.n_cells, 3), np.nan)
    fib[cell] = fl

    # report the area beyond the atlas, per label
    tag = np.asarray(mesh.cell_data[LABEL]) if LABEL in mesh.cell_data else np.ones(mesh.n_cells)
    area = mesh.compute_cell_sizes(length=False, volume=False).cell_data["Area"]
    beyond = np.zeros(mesh.n_cells, bool)
    beyond[cell[out]] = True
    ext = {}
    for L in np.unique(tag):
        s = tag == L
        ext[NAMES.get(int(L), str(int(L)))] = float(100 * area[s & beyond].sum() / max(area[s].sum(), 1e-300))
    filled = _fill(_faces(mesh), fib)
    return fib, FibreReport(layer, mesh.n_cells, ext, filled)


def read_thickness(path: str, n_points: int) -> np.ndarray:
    """One thickness per point from *path* (``.npy``, or text with one value
    per line or comma-separated). Raises ``ValueError`` saying what is wrong."""
    if path.lower().endswith(".npy"):
        v = np.load(path)
    else:
        txt = open(path).read().replace(",", " ").split()
        try:
            v = np.array([float(x) for x in txt])
        except ValueError:
            raise ValueError(f"{path}: not a list of numbers") from None
    v = np.asarray(v, float).ravel()
    if len(v) == n_points + 1 and v[0] == n_points:         # a count header line
        v = v[1:]
    if len(v) != n_points:
        raise ValueError(f"{path} holds {len(v)} values; the mesh has {n_points} points")
    if not np.isfinite(v).all() or (v < 0).any():
        raise ValueError(f"{path}: every thickness must be a finite value of 0 or more")
    return v


def thickness_fields(mesh: pv.PolyData):
    """Names of point fields usable as a thickness (one finite, non-negative
    value per point)."""
    out = []
    for k in mesh.point_data.keys():
        v = np.asarray(mesh.point_data[k])
        if v.ndim == 1 and v.dtype.kind in "fiu" and np.isfinite(v).all() and (v >= 0).all():
            if k not in (ALPHA, BETA):
                out.append(k)
    return out


def apply(mesh: pv.PolyData, layer: str, thickness=None):
    """A copy of *mesh* with its fibres: ``fibre_endo`` (layer ``endo``),
    ``fibre_epi`` (``epi``) or both (``bilayer``, with point field
    ``thickness`` in mm: a number, or one value per point). Returns
    ``(mesh, [FibreReport])``."""
    if layer not in ("endo", "epi", "bilayer"):
        raise ValueError(f"unknown layer {layer!r}")
    if ALPHA not in mesh.point_data or BETA not in mesh.point_data:
        raise ValueError("The atrium has no UAC: compute UAC first.")
    out = mesh.copy()
    reports = []
    for lay in (("endo", "epi") if layer == "bilayer" else (layer,)):
        f, rep = map_layer(mesh, lay)
        out.cell_data[FIELDS[lay]] = f
        reports.append(rep)
    if layer == "bilayer":
        th = np.broadcast_to(np.asarray(0.1 if thickness is None else thickness, float),
                             (mesh.n_points,)).copy()
        if not np.isfinite(th).all() or (th < 0).any():
            raise ValueError("every thickness must be a finite value of 0 or more")
        out.point_data["thickness"] = th
    return out, reports


__all__ = ["map_layer", "apply", "read_thickness", "thickness_fields", "FibreReport", "FIELDS"]
