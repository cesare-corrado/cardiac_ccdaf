"""
mesh_ops
========
Triangle-surface bookkeeping for the atrial coordinates: edges, boundary
loops, graphs and shortest paths, label clean-up, and the repairs a
segmented surface needs before it can be flattened.

All functions work on plain arrays: ``P`` (n, 3) points, ``t`` (m, 3)
triangles, ``tag`` (m,) labels. Repairs never move a point; they change
which points the triangles join, and may add copies of points.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, dijkstra


# ---------------------------------------------------------------- edges
def edge_table(t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Every triangle side as a sorted pair, and the triangle it came from."""
    e = np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
    return e, np.tile(np.arange(len(t)), 3)


def unique_edges(t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unique sorted edges, the index of each triangle side into them
    (side k of triangle j at ``k * len(t) + j``), and how many triangles
    share each edge."""
    e, _ = edge_table(t)
    if not len(e):
        return np.zeros((0, 2), np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64)
    n = np.int64(e.max()) + 1                     # one integer per edge: sorts like the rows
    key, inv, cnt = np.unique(e[:, 0].astype(np.int64) * n + e[:, 1], return_inverse=True,
                              return_counts=True)
    return np.c_[key // n, key % n], inv.ravel(), cnt


def boundary_edges(t: np.ndarray) -> np.ndarray:
    ue, _, cnt = unique_edges(t)
    return ue[cnt == 1]


def order_loops(edges: np.ndarray) -> List[np.ndarray]:
    """Chain edges into node sequences, one per connected loop."""
    adj: Dict[int, List[int]] = defaultdict(list)
    for a, b in edges:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))
    seen: Set[int] = set()
    out = []
    for s in adj:
        if s in seen:
            continue
        loop, prev, cur = [s], None, s
        seen.add(s)
        while True:
            nxt = [x for x in adj[cur] if x != prev and x not in seen]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            seen.add(cur)
            loop.append(cur)
        out.append(np.array(loop))
    return out


def boundary_loops(t: np.ndarray) -> List[np.ndarray]:
    return order_loops(boundary_edges(t))


def components(t: np.ndarray) -> Tuple[int, np.ndarray]:
    """Connected pieces (joined through shared points): their number and
    each triangle's piece."""
    if not len(t):
        return 0, np.zeros(0, int)
    n = int(t.max()) + 1
    A = sp.coo_matrix((np.ones(3 * len(t)), (t.ravel(), np.roll(t, 1, axis=1).ravel())),
                      shape=(n, n))
    _, lab = connected_components(A, directed=False)
    used, comp = np.unique(lab[t[:, 0]], return_inverse=True)
    return len(used), comp.ravel()


def topology(t: np.ndarray) -> Tuple[int, int]:
    """``(genus, open boundaries)`` of a triangle surface, summed over its
    connected pieces.

    The Euler characteristic of the used points, edges and faces gives
    ``genus = (2 * pieces - chi - boundaries) / 2``: 0 for spheres with
    holes, the only shape a flat map can be made of; each tunnel adds 1.
    """
    ue, _, cnt = unique_edges(t)
    nb = len(order_loops(ue[cnt == 1]))
    chi = len(np.unique(t)) - len(ue) + len(t)
    pieces, _ = components(t)
    return int(round((2 * pieces - chi - nb) / 2)), nb


# ---------------------------------------------------------------- graphs
def graph(P: np.ndarray, t: np.ndarray, allowed: np.ndarray | None = None,
          nodeweight: np.ndarray | None = None) -> sp.csr_matrix:
    """Edge graph weighted by length (times the mean of ``nodeweight`` at
    the two ends, when given), over edges whose both ends are allowed."""
    e = np.unique(edge_table(t)[0], axis=0)
    if allowed is not None:
        e = e[allowed[e[:, 0]] & allowed[e[:, 1]]]
    w = np.linalg.norm(P[e[:, 0]] - P[e[:, 1]], axis=1)
    if nodeweight is not None:
        w = w * 0.5 * (nodeweight[e[:, 0]] + nodeweight[e[:, 1]])
    n = len(P)
    return sp.csr_matrix((np.r_[w, w], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
                         shape=(n, n))


class NoPath(RuntimeError):
    """No path joins the two ends through the allowed points."""


def shortest(P: np.ndarray, t: np.ndarray, allowed: np.ndarray, src: int,
             targets: Sequence[int], nodeweight: np.ndarray | None = None) -> np.ndarray:
    """Shortest path from ``src`` to the nearest of ``targets``."""
    d, pred = dijkstra(graph(P, t, allowed, nodeweight), indices=src,
                       return_predecessors=True)
    targets = np.asarray(targets)
    tgt = targets[np.argmin(d[targets])]
    if not np.isfinite(d[tgt]):
        raise NoPath("no path")
    path = [int(tgt)]
    while path[-1] != src:
        path.append(int(pred[path[-1]]))
    return np.array(path[::-1])


def path_length(P: np.ndarray, path: np.ndarray) -> np.ndarray:
    """Cumulative length along a node path, starting at 0."""
    return np.r_[0, np.cumsum(np.linalg.norm(np.diff(P[path], axis=0), axis=1))]


def arcs(loop: np.ndarray, a: int, b: int) -> Tuple[np.ndarray, np.ndarray]:
    """The two arcs of a closed loop from node ``a`` to node ``b``."""
    i, j = int(np.where(loop == a)[0][0]), int(np.where(loop == b)[0][0])
    n = len(loop)
    fwd = loop[[(i + k) % n for k in range((j - i) % n + 1)]]
    bwd = loop[[(i - k) % n for k in range((i - j) % n + 1)]]
    return fwd, bwd


def neighbours(t: np.ndarray) -> Dict[int, Set[int]]:
    nb: Dict[int, Set[int]] = defaultdict(set)
    for a, b in np.unique(edge_table(t)[0], axis=0):
        nb[int(a)].add(int(b))
        nb[int(b)].add(int(a))
    return nb


# ---------------------------------------------------------------- labels
def relabel_islands(t: np.ndarray, tag: np.ndarray) -> Tuple[np.ndarray, int]:
    """Keep the largest edge-connected piece of each label; give every other
    piece the label it shares the most edges with. Repeat until stable."""
    tag = tag.copy()
    e, own = edge_table(t)
    key = e[:, 0] * (t.max() + 1) + e[:, 1]
    order = np.argsort(key, kind="stable")
    ks = key[order]
    d = np.where(ks[1:] == ks[:-1])[0]
    ta, tb = own[order][d], own[order][d + 1]
    moved = 0
    for _ in range(10):
        same = tag[ta] == tag[tb]
        A = sp.csr_matrix((np.ones(same.sum()), (ta[same], tb[same])), shape=(len(t), len(t)))
        _, comp = connected_components(A, directed=False)
        changed = 0
        for L in np.unique(tag):
            cs = np.unique(comp[tag == L])
            if len(cs) < 2:
                continue
            sizes = np.array([(comp == c).sum() for c in cs])
            for c in cs[sizes < sizes.max()]:
                m = comp == c
                nb = np.r_[tag[tb[m[ta] & ~m[tb]]], tag[ta[m[tb] & ~m[ta]]]]
                if len(nb):
                    tag[m] = np.bincount(nb).argmax()
                    changed += int(m.sum())
        moved += changed
        if not changed:
            break
    return tag, moved


# ---------------------------------------------------------------- repairs
def fix_orientation(t: np.ndarray) -> Tuple[np.ndarray, int]:
    """Make the winding consistent: two triangles sharing an edge must run
    along it in opposite directions."""
    t = t.copy()
    T = len(t)
    a = t.ravel()
    b = t[:, [1, 2, 0]].ravel()
    tri = np.repeat(np.arange(T), 3)
    key = np.minimum(a, b) * (t.max() + 1) + np.maximum(a, b)
    order = np.argsort(key, kind="stable")
    ks = key[order]
    pr = np.where(ks[1:] == ks[:-1])[0]
    nb: List[List[int]] = [[] for _ in range(T)]
    for h1, h2 in zip(order[pr], order[pr + 1]):
        nb[tri[h1]].append(tri[h2])
        nb[tri[h2]].append(tri[h1])
    seen = np.zeros(T, bool)
    flipped = 0

    def dedges(x):
        r = t[x]
        return {(r[0], r[1]), (r[1], r[2]), (r[2], r[0])}

    for s0 in range(T):
        if seen[s0]:
            continue
        seen[s0] = True
        q = deque([s0])
        while q:
            x = q.popleft()
            dx = dedges(x)
            for y in nb[x]:
                if seen[y]:
                    continue
                if dx & dedges(y):
                    t[y] = t[y][[0, 2, 1]]
                    flipped += 1
                seen[y] = True
                q.append(y)
    return t, flipped


def ear_mask(t: np.ndarray, edges=None) -> np.ndarray:
    """Triangles with two or more sides on the open boundary ("ears").
    ``edges``: :func:`unique_edges` of ``t``, if already at hand."""
    _, inv, cnt = edges if edges is not None else unique_edges(t)
    return (cnt[inv] == 1).reshape(3, -1).sum(0) >= 2


def remove_ears(t: np.ndarray, tag: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    """Drop triangles with two or more sides on the open boundary. Such an
    "ear" has all three corners on the boundary, so a flat map with that
    boundary fixed squashes it to zero area."""
    removed = 0
    while True:
        ear = ear_mask(t)
        if not ear.any():
            return t, tag, removed
        removed += int(ear.sum())
        t, tag = t[~ear], tag[~ear]


#: An angle above this (radians, 170 degrees) makes a triangle a sliver.
SLIVER = np.radians(170.0)


def duplicate_keep(t: np.ndarray) -> np.ndarray:
    """Which triangles to keep when some are listed more than once.

    A duplicated pair is usually a zero-area flap hanging from the surface by
    one edge, which then belongs to three or more triangles; no flat map can
    lay that out. All copies are dropped. A duplicate that is part of the
    sheet (two or more of its sides shared with other triangles) keeps one
    copy, so no hole opens.
    """
    srt = np.sort(t, axis=1)
    _, first, inv, cnt = np.unique(srt, axis=0, return_index=True, return_inverse=True,
                                   return_counts=True)
    inv = inv.ravel()
    dup = cnt[inv] > 1
    keep = ~dup
    if not dup.any():
        return keep
    rest = unique_edges(t[keep])
    rest_cnt = dict(zip(map(tuple, rest[0].tolist()), rest[2].tolist()))
    for g in np.unique(inv[dup]):
        a, b, c = srt[first[g]]
        if sum(rest_cnt.get(e, 0) >= 1 for e in ((a, b), (b, c), (a, c))) >= 2:
            keep[first[g]] = True                # part of the sheet: keep one copy
    return keep


def remove_duplicates(t: np.ndarray, tag: np.ndarray):
    """Drop triangles listed more than once; see :func:`duplicate_keep`."""
    keep = duplicate_keep(t)
    return t[keep], tag[keep], int((~keep).sum())


def rim_sliver_mask(P: np.ndarray, t: np.ndarray, edges=None) -> np.ndarray:
    """Slivers whose long side lies on an open boundary.

    Such a triangle's far corner sits almost on the boundary; with the
    boundary fixed in a flat map it is squeezed to nothing and can flip. No
    edge flip can fix it (there is no triangle across the boundary), so it is
    removed and its far corner joins the boundary. Skipped when that corner
    is already on the boundary (removing it would pinch the rim).
    ``edges``: :func:`unique_edges` of ``t``, if already at hand.
    """
    ue, inv, cnt = edges if edges is not None else unique_edges(t)
    on_b = np.zeros(len(P), bool)
    on_b[ue[cnt == 1].ravel()] = True
    side_b = (cnt[inv] == 1).reshape(3, -1).T          # side k: (t[k], t[k+1])
    X = P[t]
    drop = np.zeros(len(t), bool)
    for k in range(3):                                   # side k faces corner k+2
        c = (k + 2) % 3
        u, v = X[:, k] - X[:, c], X[:, (k + 1) % 3] - X[:, c]
        ang = np.arccos(np.clip((u * v).sum(1) / (np.linalg.norm(u, axis=1)
                                                   * np.linalg.norm(v, axis=1) + 1e-300), -1, 1))
        drop |= side_b[:, k] & (ang > SLIVER) & ~on_b[t[:, c]]
    return drop


def remove_rim_slivers(P: np.ndarray, t: np.ndarray, tag: np.ndarray):
    """Drop slivers on open boundaries (see :func:`rim_sliver_mask`), up to
    20 passes."""
    removed = 0
    for _ in range(20):
        drop = rim_sliver_mask(P, t)
        if not drop.any():
            break
        removed += int(drop.sum())
        t, tag = t[~drop], tag[~drop]
    return t, tag, removed


def _unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-300)


def delaunay_flips(P: np.ndarray, t: np.ndarray, tag: np.ndarray,
                   max_crease_deg: float = 10.0, max_pass: int = 200) -> Tuple[np.ndarray, int]:
    """Flip edges that break the Delaunay condition (the two angles facing
    the edge sum to more than 180 degrees) between two triangles of the
    same label that are nearly coplanar. Such edges carry negative
    stiffness, and a thin triangle there can collapse in a flat map. Points
    stay where they are; only the diagonals change."""
    t = t.copy()
    n = len(P)
    cos_lim = np.cos(np.radians(max_crease_deg))
    total = 0

    def ang(p, q, r):
        return np.arccos(np.clip((_unit(P[q] - P[p]) * _unit(P[r] - P[p])).sum(1), -1, 1))

    for _ in range(max_pass):
        T = len(t)
        a = t.ravel()
        b = t[:, [1, 2, 0]].ravel()
        o = t[:, [2, 0, 1]].ravel()
        tri = np.repeat(np.arange(T), 3)
        key = np.minimum(a, b) * n + np.maximum(a, b)
        order = np.argsort(key, kind="stable")
        ks = key[order]
        pair = np.where(ks[1:] == ks[:-1])[0]
        h1, h2 = order[pair], order[pair + 1]
        t1, t2 = tri[h1], tri[h2]
        ok = tag[t1] == tag[t2]
        A, B, C, D = a[h1], b[h1], o[h1], o[h2]
        aC, aD = ang(C, A, B), ang(D, A, B)
        viol = aC + aD - np.pi
        ok &= viol > 1e-12
        n1 = _unit(np.cross(P[B] - P[A], P[C] - P[A]))
        n2 = _unit(np.cross(P[A] - P[B], P[D] - P[B]))
        # A sliver (an angle near 180 degrees facing the edge) has no reliable
        # normal, so it cannot fail the crease test; the other triangle's
        # normal is the reference instead.
        sliver1, sliver2 = aC > SLIVER, aD > SLIVER
        ok &= ((n1 * n2).sum(1) >= cos_lim) | sliver1 | sliver2
        nref = np.where(sliver1[:, None], n2, n1)
        nn1 = np.cross(P[D] - P[A], P[C] - P[A])
        nn2 = np.cross(P[B] - P[D], P[C] - P[D])
        ok &= ((nn1 * nref).sum(1) > 0) & ((nn2 * nref).sum(1) > 0)
        nk_all = np.minimum(C, D) * n + np.maximum(C, D)       # the new diagonal must not exist
        pos = np.minimum(np.searchsorted(ks, nk_all), len(ks) - 1)
        ok &= ks[pos] != nk_all
        cand = np.where(ok)[0]
        if not len(cand):
            break
        cand = cand[np.argsort(-viol[cand])]
        used = np.zeros(T, bool)
        chosen = []
        for k in cand:
            if not used[t1[k]] and not used[t2[k]]:
                used[t1[k]] = used[t2[k]] = True
                chosen.append(k)
        chosen = np.array(chosen)
        nk = np.minimum(C[chosen], D[chosen]) * n + np.maximum(C[chosen], D[chosen])
        _, first = np.unique(nk, return_index=True)
        chosen = chosen[np.sort(first)]
        t[t1[chosen]] = np.c_[A[chosen], D[chosen], C[chosen]]
        t[t2[chosen]] = np.c_[D[chosen], B[chosen], C[chosen]]
        total += len(chosen)
    return t, total


def split_pinches(P: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Give each extra fan of triangles at a pinched boundary point its own
    copy of the point, so every open boundary is a simple loop. Returns the
    new points, triangles, the original point of every point, and the count."""
    P = P.copy()
    t = t.copy()
    origin = list(range(len(P)))
    e = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    key = np.minimum(e[:, 0], e[:, 1]) * len(P) + np.maximum(e[:, 0], e[:, 1])
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    b = e[cnt[inv] == 1]
    deg = np.bincount(b.ravel(), minlength=len(P))
    split = 0
    for v in np.where(deg > 2)[0]:
        tris = np.where((t == v).any(1))[0]
        parent = {int(x): int(x) for x in tris}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        others: Dict[int, List[int]] = {}
        for ti in tris:
            for w in t[ti]:
                if w != v:
                    others.setdefault(int(w), []).append(int(ti))
        for ts in others.values():
            for a_ in ts[1:]:
                ra, rb = find(ts[0]), find(a_)
                if ra != rb:
                    parent[ra] = rb
        fans: Dict[int, List[int]] = {}
        for ti in tris:
            fans.setdefault(find(int(ti)), []).append(int(ti))
        for fan in list(fans.values())[1:]:
            P = np.vstack([P, P[v]])
            origin.append(origin[v])
            nv = len(P) - 1
            for ti in fan:
                t[ti][t[ti] == v] = nv
            split += 1
    return P, t, np.array(origin), split


def cotan_negative(P: np.ndarray, t: np.ndarray) -> int:
    """Number of interior edges whose cotangent weight is negative."""
    a = t.ravel()
    b = t[:, [1, 2, 0]].ravel()
    o = t[:, [2, 0, 1]].ravel()
    u, v = P[a] - P[o], P[b] - P[o]
    cot = (u * v).sum(1) / (np.linalg.norm(np.cross(u, v), axis=1) + 1e-300)
    key = np.minimum(a, b) * len(P) + np.maximum(a, b)
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    w = np.bincount(inv, weights=cot)
    return int((w[cnt == 2] < -1e-12).sum())


def repair(P: np.ndarray, t: np.ndarray, tag: np.ndarray, max_crease_deg: float = 10.0):
    """Orientation, ears, Delaunay flips, ears again, pinched points.

    Returns ``P, t, tag, origin, report``; ``origin[i]`` is the input
    point that point ``i`` copies.
    """
    t, tag, dups = remove_duplicates(t, tag)
    rep = {"duplicates_removed": dups, "negative_weights_before": cotan_negative(P, t)}
    t, rep["rewound"] = fix_orientation(t)
    t, tag, ears = remove_ears(t, tag)
    t, rep["flips"] = delaunay_flips(P, t, tag, max_crease_deg)
    t, tag, rep["rim_slivers_removed"] = remove_rim_slivers(P, t, tag)
    t, tag, more = remove_ears(t, tag)
    rep["ears_removed"] = ears + more
    rep["negative_weights_after"] = cotan_negative(P, t)
    P, t, origin, rep["pinches_split"] = split_pinches(P, t)
    return P, t, tag, origin, rep


__all__ = [
    "ear_mask", "duplicate_keep", "rim_sliver_mask",
    "edge_table", "unique_edges", "boundary_edges", "order_loops", "boundary_loops",
    "components", "topology", "graph", "NoPath", "shortest", "path_length", "arcs", "neighbours",
    "relabel_islands", "fix_orientation", "remove_duplicates", "remove_ears", "remove_rim_slivers", "delaunay_flips",
    "split_pinches", "cotan_negative", "repair",
]
