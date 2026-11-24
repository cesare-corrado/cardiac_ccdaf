"""
tunnels
=======
Remove tunnels (handles) from an open triangle surface.

An atrium must be a sphere with holes to be laid flat without overlap. A
segmentation sometimes leaves a thin bridge of tissue, or a narrow channel
between two walls: a tunnel, which raises the genus by one and makes part
of any flat map cover itself twice.

For each tunnel the shortest loop that cannot be shrunk to a point is found
(it goes round the bridge or round the channel's neck), the surface is cut
along it and the two new openings are closed with a fan of triangles. The
patches are not tissue and are reported as such. A tunnel whose shortest
loop is longer than a limit (15 mm by default) is not cut: that is no
longer a thin bridge, and the segmentation should be fixed instead.

Loop search: the boundaries are capped for the search, then for a root
point a shortest-path tree and a maximum spanning co-tree by loop length
leave ``2 * genus`` edges; each closes a loop through the root, and the
shortest is the shortest non-separating loop through that root
(Erickson and Whittlesey, 2005). Several roots are tried and the search
re-roots on the best loop until it stops getting shorter.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra, minimum_spanning_tree

from ccdaf.core.uac.mesh_ops import boundary_loops, topology, unique_edges


class TunnelError(ValueError):
    """A tunnel could not be removed automatically."""


def _close(P, t, loops):
    """Cap every boundary loop with a centre point (search only)."""
    P2, t2, capv = [P], [t], []
    n = len(P)
    for lp in loops:
        c = n + len(capv)
        capv.append(c)
        P2.append(P[lp].mean(0)[None])
        t2.append(np.c_[lp, np.roll(lp, -1), np.full(len(lp), c)])
    return np.vstack(P2), np.vstack(t2), np.array(capv, int)


def _loops_from_root(P, t, ue, inv, root, blocked):
    """Greedy system of loops through ``root``: ``[(length, node cycle)]``,
    shortest first."""
    n = len(P)
    w = np.linalg.norm(P[ue[:, 0]] - P[ue[:, 1]], axis=1)
    w[blocked[ue[:, 0]] | blocked[ue[:, 1]]] = 1e6      # never through a cap
    G = sp.csr_matrix((np.r_[w, w], (np.r_[ue[:, 0], ue[:, 1]], np.r_[ue[:, 1], ue[:, 0]])),
                      shape=(n, n))
    d, pred = dijkstra(G, indices=root, return_predecessors=True)
    tree = np.zeros(len(ue), bool)
    has = pred >= 0
    v, p = np.where(has)[0], pred[has]
    key = ue[:, 0] * n + ue[:, 1]
    tree[np.searchsorted(key, np.minimum(v, p) * n + np.maximum(v, p))] = True
    ell = d[ue[:, 0]] + d[ue[:, 1]] + w
    # dual graph over the non-tree edges, maximum spanning tree by loop length
    nt = len(t)
    eid = inv.reshape(3, nt).T.ravel()
    f = np.repeat(np.arange(nt), 3)
    o = np.argsort(eid, kind="stable")
    eid_s, f_s = eid[o], f[o]
    second = np.r_[False, eid_s[1:] == eid_s[:-1]]
    e2, fa, fb = eid_s[second], f_s[np.where(second)[0] - 1], f_s[second]
    m = ~tree[e2]
    e2, fa, fb = e2[m], fa[m], fb[m]
    big = ell.max() + 1.0
    D = sp.csr_matrix((big - ell[e2] + 1e-9, (fa, fb)), shape=(nt, nt))
    T = minimum_spanning_tree(D).tocoo()
    E = sp.csr_matrix((e2 + 1.0, (fa, fb)), shape=(nt, nt))
    E = E + E.T
    cot = np.zeros(len(ue), bool)
    cot[np.asarray(E[T.row, T.col]).ravel().astype(int) - 1] = True
    out = []
    for ei in np.where(~tree & ~cot)[0]:
        u, v = ue[ei]

        def up(x):
            pth = [int(x)]
            while pth[-1] != root:
                pth.append(int(pred[pth[-1]]))
            return pth

        pu, pv = up(u), up(v)
        su = set(pu)
        lca = next(x for x in pv if x in su)
        cyc = pu[: pu.index(lca) + 1] + pv[: pv.index(lca)][::-1]
        out.append((float(ell[ei]), np.array(cyc)))
    return sorted(out, key=lambda x: x[0])


def _shortest_loop(P, t, roots, rng) -> Tuple[float, np.ndarray]:
    loops = boundary_loops(t)
    Pc, tc, capv = _close(P, t, loops)
    blocked = np.zeros(len(Pc), bool)
    blocked[capv] = True
    if loops:
        blocked[np.concatenate(loops)] = True          # never along the open boundary
    ue, inv, _ = unique_edges(tc)
    used = np.setdiff1d(np.unique(t), np.concatenate(loops) if loops else [])
    cand = [int(x) for x in rng.choice(used, size=min(roots, len(used)), replace=False)]
    best: Tuple[float, np.ndarray] = (np.inf, np.zeros(0, int))
    tried = set()
    while cand:
        r = cand.pop()
        if r in tried:
            continue
        tried.add(r)
        res = _loops_from_root(Pc, tc, ue, inv, r, blocked)
        if res and res[0][0] < best[0] - 1e-9:
            best = res[0]
            cyc = best[1]
            cand += [int(x) for x in cyc[:: max(1, len(cyc) // 8)]]
    return best


def _cut_and_cap(P, t, tag, origin, cyc):
    """Cut along the closed cycle and close both openings with a fan."""
    n = len(P)
    cyc = [int(x) for x in cyc]
    k = len(cyc)
    directed = {(cyc[i], cyc[(i + 1) % k]) for i in range(k)}
    undirected = {(min(a, b), max(a, b)) for a, b in directed}
    on = set(cyc)
    touch = np.where(np.isin(t, cyc).any(1))[0]
    left = {ti for ti in touch
            if any((int(t[ti, j]), int(t[ti, (j + 1) % 3])) in directed for j in range(3))}
    # triangles round the cycle, joined across edges that are not on it
    by_edge: Dict[Tuple[int, int], List[int]] = {}
    for ti in touch:
        for j in range(3):
            a, b = int(t[ti, j]), int(t[ti, (j + 1) % 3])
            key = (min(a, b), max(a, b))
            if key not in undirected:
                by_edge.setdefault(key, []).append(int(ti))
    nb: Dict[int, List[int]] = {int(ti): [] for ti in touch}
    for lst in by_edge.values():
        if len(lst) == 2:
            nb[lst[0]].append(lst[1])
            nb[lst[1]].append(lst[0])
    side = set(left)
    stack = list(left)
    while stack:
        x = stack.pop()
        for y in nb[x]:
            if y not in side:
                side.add(y)
                stack.append(y)
    newid = {v: n + i for i, v in enumerate(cyc)}
    t = t.copy()
    for ti in side:
        for j in range(3):
            if int(t[ti, j]) in on:
                t[ti, j] = newid[int(t[ti, j])]
    P = np.vstack([P, P[cyc]])
    origin = np.r_[origin, origin[cyc]]
    caps = []
    for ring in (np.array(cyc), np.array([newid[v] for v in cyc])):
        c = len(P)
        P = np.vstack([P, P[ring].mean(0)[None]])
        origin = np.r_[origin, -1]
        caps.append(np.c_[ring, np.roll(ring, -1), np.full(len(ring), c)])
    used = set()
    for j in range(3):
        used |= set(zip(t[:, j].tolist(), t[:, (j + 1) % 3].tolist()))
    for s, blk in enumerate(caps):     # wind each cap against its neighbours
        if (int(blk[0, 0]), int(blk[0, 1])) in used:
            caps[s] = blk[:, [1, 0, 2]]
    capt = np.vstack(caps)
    lab = np.bincount(tag[touch]).argmax()
    return P, np.vstack([t, capt]), np.r_[tag, np.full(len(capt), lab)], origin, len(capt)


def remove_tunnels(P: np.ndarray, t: np.ndarray, tag: np.ndarray, origin: np.ndarray,
                   max_mm: float = 15.0, roots: int = 24, seed: int = 0, strict: bool = True):
    """Cut every thin tunnel of a connected surface. Returns ``P, t, tag,
    origin, is_patch, report``.

    ``is_patch`` marks the triangles added to close the cuts; they take the
    label most common round the cut. The existing triangles keep their
    order. ``origin`` maps every point to the input point it copies (-1 for
    a patch centre). When a tunnel's shortest loop is longer than
    ``max_mm`` (in the mesh's length unit, millimetres for the usual atrial
    meshes), ``strict`` raises :class:`TunnelError`; otherwise the search
    stops and the last report entry is ``{"left": tunnels, "loop_mm": ...}``.
    """
    g, _ = topology(t)
    report = []
    is_patch = np.zeros(len(t), bool)
    rng = np.random.default_rng(seed)
    while g > 0:
        length, cyc = _shortest_loop(P, t, roots, rng)
        if not len(cyc) or length > max_mm:
            if not strict:
                report.append({"left": int(g), "loop_mm": round(float(length), 1)})
                break
            raise TunnelError(
                f"The surface has {g} tunnel(s), and the shortest loop round one is "
                f"{length:.1f} mm long (the limit is {max_mm:g} mm). That is not a thin "
                f"bridge, so it is not cut automatically: fix the segmentation.")
        centre = P[cyc].mean(0)
        P, t, tag, origin, ncap = _cut_and_cap(P, t, tag, origin, cyc)
        is_patch = np.r_[is_patch, np.ones(ncap, bool)]
        g2, _ = topology(t)
        if g2 >= g:
            raise TunnelError("Cutting a tunnel did not remove it; fix the segmentation.")
        report.append({"loop_mm": round(length, 1), "centre": np.round(centre, 1).tolist(),
                       "patch_triangles": int(ncap)})
        g = g2
    return P, t, tag, origin, is_patch, report


__all__ = ["TunnelError", "remove_tunnels"]
