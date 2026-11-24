"""
flatten
=======
Lay each half of the cut atrium on its half of the square, then warp it
onto the template.

**First map.** The half is a disc with holes (the veins' open ends and the
appendage). Its outer boundary is fixed on the half-square: the lateral
line on ``alpha = 1``, the roof on ``beta = 0.5``, the septal line on
``alpha = 0``, the valve on ``beta = 0`` (posterior) or ``1`` (anterior),
each by arc length. The holes are closed with a temporary centre point and
both coordinates are solved together as a harmonic map with mean-value
weights (Floater, 2003). Those weights are always positive, so with a
convex boundary the map cannot fold (Floater's theorem). Cotangent weights,
the usual finite-element Laplacian, go negative on obtuse triangles and can
fold.

**Template warp.** The first map puts every ring wherever the patient's
shape takes it. Each ring is then given its template target (a circle,
ellipse or half-ellipse, parameterised along the ring as the first map has
it) and the map is warped onto them with :func:`ccdaf.core.uac.slim.slim`,
which never folds a triangle. The square's edges are held; the roof gets
the template's breakpoints, so each superior vein's posterior half lands on
its stretch of the roof.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu

from ccdaf.core.uac.cut import Anatomy, Cut
from ccdaf.core.uac.mesh_ops import order_loops, path_length, unique_edges
from ccdaf.core.uac.slim import slim
from ccdaf.core.uac.solvers import SpdSolver
from ccdaf.core.uac.template import (
    END_C, END_R, GAP, INF_R, LAA, LAA_C, LAA_RA, LAA_RB, LIPV, LIPV_C, LSPV, LSPV_C,
    RIPV, RIPV_C, RSPV, RSPV_C, SUP_RA, SUP_RB,
)

POSTERIOR, ANTERIOR = 0, 1


#: A triangle whose longest side on the flat map is shorter than this (in
#: units of the square; a typical side is about 1e-3) is collapsed to a
#: point. Inside such a patch the sign of a triangle's area is round-off, so
#: it is not counted as folded.
COLLAPSED = 1e-6


def collapsed_patches(U: np.ndarray, tl: np.ndarray, X: np.ndarray, tags: np.ndarray,
                      real: np.ndarray):
    """Tissue triangles that the flat map shrinks to a point, in connected patches.

    A thin spike or horn of the segmentation (a closed finger a few
    millimetres long and a fraction of a millimetre wide) is squeezed to a
    point by any flat map. Its points all get the coordinates of the spike's
    base. Returns ``mask, patches``: which triangles are collapsed, and per
    patch ``{"triangles", "size_mm", "label"}`` (size: the largest extent in
    3D), largest first.
    """
    e = np.stack([np.linalg.norm(U[tl[:, k]] - U[tl[:, (k + 1) % 3]], axis=1) for k in range(3)], 1)
    mask = (e.max(1) < COLLAPSED) & real
    if not mask.any():
        return mask, []
    ids = np.where(mask)[0]
    sub = tl[ids]
    n = len(U)
    A = sp.coo_matrix((np.ones(3 * len(sub)), (sub.ravel(), np.roll(sub, 1, axis=1).ravel())),
                      shape=(n, n))
    _, comp = connected_components(A + A.T, directed=False)
    lab = comp[sub[:, 0]]
    patches = []
    for c in np.unique(lab):
        m = lab == c
        v = np.unique(sub[m])
        patches.append({"triangles": int(m.sum()),
                        "size_mm": round(float(np.ptp(X[v], axis=0).max()), 1),
                        "label": int(np.bincount(tags[ids[m]]).argmax())})
    patches.sort(key=lambda d: -d["size_mm"])
    return mask, patches


class DegenerateFirstMap(RuntimeError):
    """The first map squashed many triangles: retry with roof-first ends."""


def mvc_matrix(X: np.ndarray, t: np.ndarray) -> sp.csr_matrix:
    """Mean-value weights ``w_ij = (tan(a/2) + tan(b/2)) / |x_i - x_j|``,
    ``a, b`` the angles at ``i`` beside edge ``ij``. Always positive."""
    n = len(X)
    rows, cols, vals = [], [], []
    for k in range(3):
        i, j, o = t[:, k], t[:, (k + 1) % 3], t[:, (k + 2) % 3]
        u, v = X[j] - X[i], X[o] - X[i]
        lu, lv = np.linalg.norm(u, axis=1), np.linalg.norm(v, axis=1)
        th = np.tan(0.5 * np.arccos(np.clip((u * v).sum(1) / (lu * lv + 1e-300), -1, 1)))
        rows += [i, i]
        cols += [j, o]
        vals += [th / (lu + 1e-300), th / (lv + 1e-300)]
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(n, n))


def harmonic(W: sp.csr_matrix, fixed: np.ndarray, values: np.ndarray) -> np.ndarray:
    n = W.shape[0]
    L = (sp.diags(np.asarray(W.sum(1)).ravel()) - W).tocsr()
    free = np.setdiff1d(np.arange(n), fixed)
    lu = splu(L[free][:, free].tocsc())
    B = L[free][:, fixed]
    out = np.zeros((n, values.shape[1]))
    out[fixed] = values
    for c in range(values.shape[1]):
        out[free, c] = lu.solve(-(B @ values[:, c]))
    return out


def signed_areas(U: np.ndarray, t: np.ndarray) -> np.ndarray:
    a, b, c = U[t[:, 0]], U[t[:, 1]], U[t[:, 2]]
    return 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))


def _area2(q):
    return 0.5 * (q[:, 0] @ np.roll(q[:, 1], -1) - q[:, 1] @ np.roll(q[:, 0], -1))


@dataclass
class HalfMap:
    side: int
    triangles: np.ndarray            # indices into the working mesh's triangles
    nodes: np.ndarray                # working-mesh points of this half
    tl: np.ndarray                   # triangles in local point numbering
    uv: np.ndarray                   # (len(nodes), 2) final coordinates
    first_map_bad: int               # triangles folded or squashed in the first map
    folded_real: int                 # folded triangles on tissue after the warp
    folded_added: int                # folded triangles on extensions / patches
    ring_miss: Dict[str, float] = field(default_factory=dict)   # worst miss per ring
    edge_miss: float = 0.0
    collapsed: list = field(default_factory=list)                 # see collapsed_patches


def flatten_half(P: np.ndarray, t: np.ndarray, tag: np.ndarray, is_real: np.ndarray,
                 cut: Cut, anat: Anatomy, side: int, *, retry_if_degenerate: bool,
                 solver: Optional[SpdSolver] = None, iters: int = 100,
                 should_stop: Optional[Callable[[], bool]] = None,
                 on_iteration: Optional[Callable[[int], None]] = None) -> HalfMap:
    tri_ids = np.where(cut.piece == side)[0]
    tri = t[tri_ids]
    nodes, loc = np.unique(tri, return_inverse=True)
    tl = loc.reshape(-1, 3)
    X = P[nodes]
    g2l = {int(g): i for i, g in enumerate(nodes)}
    ue, _, cnt = unique_edges(tl)
    loops = order_loops(ue[cnt == 1])
    cutn = cut.nodes
    cut_loc = {g2l[int(v)] for v in cutn if int(v) in g2l}
    outer = [lp for lp in loops if set(lp.tolist()) & cut_loc]
    holes = [lp for lp in loops if not set(lp.tolist()) & cut_loc]
    if len(outer) != 1:
        raise RuntimeError(f"half {side}: {len(outer)} outer boundaries")
    outer = outer[0]
    y0 = 0.0 if side == POSTERIOR else 1.0
    iP1, iP2, iP4, iP3 = (cut.corners[k] for k in ("P1", "P2", "P4", "P3"))
    lenc = path_length(P, cutn)

    # ---- first map: boundary on the half-square by arc length
    fixed, val = [], []
    Llat, Ltop, Lsep = lenc[iP1], lenc[iP3] - lenc[iP1], lenc[-1] - lenc[iP3]
    for i, v in enumerate(cutn):
        if int(v) not in g2l:
            continue
        s = lenc[i]
        if i <= iP1:
            a, b = 1.0, y0 + (0.5 - y0) * s / Llat
        elif i <= iP3:
            a, b = 1.0 - (s - lenc[iP1]) / Ltop, 0.5
        else:
            a, b = 0.0, 0.5 + (y0 - 0.5) * (s - lenc[iP3]) / Lsep
        fixed.append(g2l[int(v)])
        val.append((a, b))
    ol = list(outer)
    lat_end, sep_end = g2l[int(cutn[0])], g2l[int(cutn[-1])]
    i0 = ol.index(sep_end)
    ol = ol[i0:] + ol[:i0]
    if ol[1] in cut_loc and ol[1] != lat_end:
        ol = [ol[0]] + ol[1:][::-1]
    mvpath = np.array(ol[: ol.index(lat_end) + 1])
    sl = path_length(X, mvpath)
    for v, s in zip(mvpath[1:-1], sl[1:-1]):
        fixed.append(int(v))
        val.append((s / sl[-1], y0))
    fixed = np.array(fixed)
    val = np.array(val)
    if len(set(fixed.tolist())) != len(fixed):
        raise RuntimeError(f"half {side}: a boundary point is fixed twice")
    Xf, tf = X.copy(), tl.copy()
    for lp in holes:
        c = len(Xf)
        Xf = np.vstack([Xf, X[lp].mean(0)])
        tf = np.vstack([tf, np.c_[lp, np.roll(lp, -1), np.full(len(lp), c)]])
    U1 = harmonic(mvc_matrix(Xf, tf), fixed, val)[: len(nodes)]
    sa = signed_areas(U1, tl)
    maj = np.sign(np.median(sa))
    bad = int(((np.sign(sa) == -maj) | (np.abs(sa) < 1e-14)).sum())
    if retry_if_degenerate and bad > 20:
        raise DegenerateFirstMap(f"first map squashed {bad} triangles")

    # ---- template targets
    fx: Dict[int, tuple] = {int(g): tuple(v) for g, v in zip(fixed, val)}
    iX = int(np.searchsorted(lenc, lenc[iP1] - GAP * Ltop))
    iY = int(np.searchsorted(lenc, lenc[iP3] + GAP * Ltop))
    iX = min(max(iX, 1), iP1 - 1)
    iY = max(min(iY, len(cutn) - 2), iP3 + 1)
    brk = [(iX, 1.0), (iP1, 1.0 - GAP), (iP2, LSPV_C[0] - SUP_RA), (iP4, RSPV_C[0] + SUP_RA),
           (iP3, GAP), (iY, 0.0)]
    for i, v in enumerate(cutn):
        if int(v) not in g2l:
            continue
        lv = g2l[int(v)]
        if i <= iX:
            fx[lv] = (1.0, y0 + (0.5 - y0) * lenc[i] / lenc[iX])
            continue
        if i >= iY:
            fx[lv] = (0.0, 0.5 + (y0 - 0.5) * (lenc[i] - lenc[iY]) / (lenc[-1] - lenc[iY]))
            continue
        for (j0, a0), (j1, a1) in zip(brk[:-1], brk[1:]):
            if j0 <= i <= j1:
                f = (lenc[i] - lenc[j0]) / max(lenc[j1] - lenc[j0], 1e-12)
                fx[lv] = (a0 + (a1 - a0) * f, 0.5)
    cut_set = set(int(v) for v in cutn)
    groups: Dict[str, np.ndarray] = {}

    def closed(loop_l, cen, ra, rb):
        q = U1[loop_l]
        if _area2(q) < 0:
            loop_l, q = loop_l[::-1], q[::-1]
        th = np.arctan2(q[:, 1] - q[:, 1].mean(), q[:, 0] - q[:, 0].mean())
        k = int(np.argmin(np.abs(th)))
        loop_l, th = np.roll(loop_l, -k), np.roll(th, -k)
        ph = np.unwrap(th)
        ph -= ph[0]
        if not ((np.diff(ph) > 0).all() and ph[-1] < 2 * np.pi):
            q2 = U1[np.r_[loop_l, loop_l[:1]]]
            s2 = np.r_[0, np.cumsum(np.linalg.norm(np.diff(q2, axis=0), axis=1))]
            ph = 2 * np.pi * s2[:-1] / s2[-1]
        for v, f in zip(loop_l, ph):
            if int(v) not in fx:
                fx[int(v)] = (cen[0] + ra * np.cos(f), cen[1] + rb * np.sin(f))

    def upper_arc(L, cen, ra, rb):
        rg = anat.ring[L]
        incut = np.array([int(v) in cut_set for v in rg])
        if incut.all() or not incut.any():
            return
        best_k, best_len = None, -1
        for k in np.where(incut & ~np.roll(incut, -1))[0]:
            ln = int(np.argmax(np.roll(incut, -k)[1:]))
            if ln > best_len:
                best_k, best_len = k, ln
        chain = np.roll(rg, -best_k)[: best_len + 2]
        if not all(int(v) in g2l for v in chain):
            return
        cl = np.array([g2l[int(v)] for v in chain])
        e0 = fx.get(int(cl[0]), (U1[cl[0], 0], 0))[0]
        e1 = fx.get(int(cl[-1]), (U1[cl[-1], 0], 0))[0]
        q = U1[cl]
        s2 = np.r_[0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
        f = s2 / s2[-1]
        start_at_outer = abs(e0 - (cen[0] + ra)) < abs(e1 - (cen[0] + ra))
        ph = np.pi * (f if start_at_outer else 1 - f)
        for v, a in zip(cl[1:-1], ph[1:-1]):
            if int(v) not in fx:
                fx[int(v)] = (cen[0] + ra * np.cos(a), cen[1] + rb * np.sin(a))
        groups[f"arc {L}"] = cl[1:-1]

    tag_half = tag[tri_ids]
    if side == POSTERIOR:
        for L, cen in ((LIPV, LIPV_C), (RIPV, RIPV_C)):
            ids = np.array([g2l[int(v)] for v in anat.ring[L] if int(v) in g2l])
            closed(ids, cen, INF_R, INF_R)
            groups[f"ring {L}"] = ids
    else:
        ids = np.array([g2l[int(v)] for v in anat.ring[LAA] if int(v) in g2l])
        closed(ids, LAA_C, LAA_RA, LAA_RB)
        groups[f"ring {LAA}"] = ids
        for L, cen in ((LSPV, LSPV_C), (RSPV, RSPV_C)):
            upper_arc(L, cen, SUP_RA, SUP_RB)
    ends = {}
    for lp in holes:
        m = np.isin(tl, lp).sum(1) >= 2
        L = int(np.bincount(tag_half[m]).argmax())
        if L in END_C:
            closed(lp, END_C[L], END_R, END_R)
            ends[L] = np.asarray(lp)
            groups[f"end {L}"] = np.asarray(lp)
    keys = np.array(list(fx))
    targets = np.array([fx[k] for k in keys])

    # ---- warp
    soft = ~np.isin(keys, fixed)
    lm, lm_tg = keys[soft], targets[soft]
    pos = {int(v): i for i, v in enumerate(lm)}
    circles = [(np.array([pos[int(v)] for v in lp if int(v) in pos]), np.array(END_C[L]), END_R)
               for L, lp in ends.items()]
    hard_tg = np.array([fx[int(v)] for v in fixed])
    U, hist, _ = slim(U1, tl, lm, lm_tg, fixed, hard_tg, circles=circles, iters=iters,
                      solver=solver, should_stop=should_stop, on_iteration=on_iteration)
    sa = signed_areas(U, tl)
    folded = np.sign(sa) == -np.sign(np.median(sa))
    real = is_real[tri_ids]
    gone, patches = collapsed_patches(U, tl, X, tag[tri_ids], real)
    folded &= ~gone
    miss = {}
    for name, ids in groups.items():
        ids = np.array([v for v in ids if int(v) in fx])
        if not len(ids):
            continue
        if name.startswith("end"):
            c = np.array(END_C[int(name.split()[1])])
            mm = np.abs(np.linalg.norm(U[ids] - c, axis=1) - END_R)
        else:
            mm = np.linalg.norm(U[ids] - np.array([fx[int(v)] for v in ids]), axis=1)
        miss[name] = float(mm.max())
    return HalfMap(side, tri_ids, nodes, tl, U, bad, int((folded & real).sum()),
                   int((folded & ~real).sum()), miss, float(hist[-1][3]) if hist else 0.0,
                   patches)


__all__ = ["POSTERIOR", "ANTERIOR", "COLLAPSED", "DegenerateFirstMap", "HalfMap", "flatten_half",
           "collapsed_patches",
           "mvc_matrix", "harmonic", "signed_areas"]
