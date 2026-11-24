"""
extension
=========
Virtual vein extensions.

In the template each vein fills an annulus between its ostium and its
opening. The annulus's width, relative to the ostium, implies a sleeve
length of about 0.2 times the ostium's circumference; measured clipped
sleeves are usually much shorter. A short sleeve forced into the annulus
is crushed, and the crushing spills onto the body. Each vein is therefore
lengthened at its open end with layers of near-equilateral triangles until
its sleeve has the length the template implies.

The extension exists only for the map: it is not tissue, it is dropped
from the output, and it may cross other anatomy without harm, because only
the mesh's own geometry enters the map.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.sparse.csgraph import dijkstra

from ccdaf.core.uac.mesh_ops import graph


def _next_on_rim(t):
    d = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    key = np.minimum(d[:, 0], d[:, 1]) * (t.max() + 1) + np.maximum(d[:, 0], d[:, 1])
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    return {int(a): int(c) for a, c in d[cnt[inv] == 1]}


def _rim_loops(t):
    nxt = _next_on_rim(t)
    seen, out = set(), []
    for v in list(nxt):
        if v in seen:
            continue
        loop = [v]
        while nxt.get(loop[-1], v) != v and loop[-1] in nxt and len(loop) < 10 ** 6:
            loop.append(nxt[loop[-1]])
        seen.update(loop)
        if nxt.get(loop[-1]) == v:
            out.append(loop)
    return out


def _add_layers(P, t, tag, loop, label, length, away_from):
    """Extrude the rim loop (in rim order) by ``length``."""
    loop = list(loop)
    av = np.mean(np.linalg.norm(P[loop] - P[np.roll(loop, -1)], axis=1))
    step = av * np.sqrt(3) / 2
    nlay = int(np.ceil(length / step)) if length > 0 else 0
    newP, newT = [], []
    n0, count = len(P), 0
    cur = P[loop]
    ids = loop
    for _ in range(nlay):
        mu = cur.mean(0)
        nrm = np.linalg.svd(cur - mu)[2][2]
        if np.dot(nrm, mu - away_from) < 0:
            nrm = -nrm
        nxt = 0.5 * (cur + np.roll(cur, -1, axis=0)) + step * nrm
        nid = list(range(n0 + count, n0 + count + len(nxt)))
        count += len(nxt)
        newP.append(nxt)
        k = len(ids)
        for i in range(k):
            vi, vj = ids[i], ids[(i + 1) % k]
            mi, mj = nid[i], nid[(i + 1) % k]
            newT.append((vj, vi, mi))
            newT.append((vj, mi, mj))
        cur, ids = nxt, nid
    if not nlay:
        return P, t, tag, 0, step
    P = np.vstack([P] + newP)
    t = np.vstack([t, np.array(newT)])
    tag = np.r_[tag, np.full(len(newT), label)]
    return P, t, tag, nlay, step


def extend_veins(P: np.ndarray, t: np.ndarray, tag: np.ndarray,
                 rings: Dict[int, np.ndarray], ratios: Dict[int, float]):
    """Lengthen each vein of ``rings`` (ostium loops by label) to
    ``ratios[label]`` times its ostium's circumference.

    Returns ``P, t, tag, n_added_points, report``; added points and
    triangles come after the existing ones.
    """
    n0 = len(P)
    report = {}
    for L, ring in rings.items():
        end = None
        for lp in _rim_loops(t):
            m = np.isin(t, lp).sum(1) >= 2
            if m.any() and np.mean(tag[m] == L) > 0.5:
                end = lp if end is None or len(lp) > len(end) else end
        if end is None:
            report[L] = {"note": "no open end found"}
            continue
        c = float(np.linalg.norm(np.diff(P[np.r_[ring, ring[:1]]], axis=0), axis=1).sum())
        h = float(np.median(dijkstra(graph(P, t), indices=end, min_only=True)[ring]))
        need = ratios[L] * c - h
        P, t, tag, nlay, step = _add_layers(P, t, tag, end, L, need, P[ring].mean(0))
        report[L] = {"ostium_mm": round(c, 1), "sleeve_mm": round(h, 1),
                     "added_mm": round(max(need, 0.0), 1), "layers": nlay}
    return P, t, tag, len(P) - n0, report


__all__ = ["extend_veins"]
