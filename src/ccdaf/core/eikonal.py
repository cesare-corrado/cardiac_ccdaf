"""
eikonal
=======
Geodesic distance inside a tetrahedral volume: the solution of
``|grad T| = 1`` with ``T = 0`` on a set of source nodes.

Why not shortest paths along edges
----------------------------------
Paths along mesh edges zig-zag. Their lengths come out about 5% too long,
and, worse for anything that differentiates the result, they bend at
nodes: on a 2.9-million-element ventricle the gradient of an edge-path
distance ratio turned by a median 6.7 degrees from one element to its
neighbours, against 0.4 degrees here.

Method
------
The Fast Iterative Method (Jeong and Whitaker 2008; Fu, Kirby and Whitaker
2011) with the local updates of Tsitsiklis (1995): a node's value is the
smallest arrival time over its elements, each solved exactly for a front
that is planar across the element, falling back to the element's faces and
edges. Nodes whose value still changes stay on an active list; when a node
settles, its neighbours are re-evaluated.

Each iteration runs in four phases: the active nodes' updates in parallel,
then applying them and collecting the neighbours to wake, then the
neighbours' updates in parallel, then applying those. Within a phase every
update reads values from before it, so the result is the same on any
number of threads.

Speed
-----
Compiled with numba on first use (a few seconds), then cached on disk. On a
637,000-node ventricle one distance field takes 1.6 s on 24 threads.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numba
import numpy as np
from numba import njit, prange

INF = np.inf

#: numba options shared by every kernel. ``error_model="numpy"`` makes a
#: division by zero give inf or nan, as in C++, instead of raising: the
#: updates reject those values themselves.
_JIT = dict(cache=True, error_model="numpy")

#: Convergence tolerance, relative to the largest extent of the mesh.
DEFAULT_RELATIVE_TOL: float = 1e-9


# ---------------------------------------------------------------------
# Local updates (scalar arithmetic only: no temporary arrays)
# ---------------------------------------------------------------------
@njit(**_JIT)
def _line(ax, ay, az, bx, by, bz, u):
    dx, dy, dz = bx - ax, by - ay, bz - az
    return u + np.sqrt(dx * dx + dy * dy + dz * dz)


@njit(**_JIT)
def _triangle(x1, y1, z1, x2, y2, z2, x3, y3, z3, u1, u2):
    """Arrival at point 3 from points 1 and 2."""
    res = INF
    a0, a1, a2 = x1 - x2, y1 - y2, z1 - z2
    b0, b1, b2 = x2 - x3, y2 - y3, z2 - z3
    k = u1 - u2
    p11 = a0 * a0 + a1 * a1 + a2 * a2
    p12 = a0 * b0 + a1 * b1 + a2 * b2
    p22 = b0 * b0 + b1 * b1 + b2 * b2
    den = p11 - k * k
    if den > 1e-12 * p11:
        sv = (p11 * p22 - p12 * p12) / den
        if sv >= 0.0:
            rhs = k * np.sqrt(sv)
            for s in range(2):
                al = -(p12 + rhs) / p11 if s == 0 else -(p12 - rhs) / p11
                al = min(max(al, 0.0), 1.0)
                dx = x3 - (al * x1 + (1.0 - al) * x2)
                dy = y3 - (al * y1 + (1.0 - al) * y2)
                dz = z3 - (al * z1 + (1.0 - al) * z2)
                v = al * u1 + (1.0 - al) * u2 + np.sqrt(dx * dx + dy * dy + dz * dz)
                if v < res:
                    res = v
    return min(res, _line(x1, y1, z1, x3, y3, z3, u1),
               _line(x2, y2, z2, x3, y3, z3, u2))


@njit(**_JIT)
def _tetra(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4, u1, u2, u3):
    """Arrival at point 4 from points 1-3, inside the element only."""
    p0, p1, p2 = x3 - x1, y3 - y1, z3 - z1
    q0, q1, q2 = x3 - x2, y3 - y2, z3 - z2
    s0, s1, s2 = x4 - x3, y4 - y3, z4 - z3
    k1 = u1 - u3
    k2 = u2 - u3
    r11 = p0 * p0 + p1 * p1 + p2 * p2
    r12 = p0 * q0 + p1 * q1 + p2 * q2
    r13 = p0 * s0 + p1 * s1 + p2 * s2
    r22 = q0 * q0 + q1 * q1 + q2 * q2
    r23 = q0 * s0 + q1 * s1 + q2 * s2
    A1 = k2 * r11 - k1 * r12
    A2 = k2 * r12 - k1 * r22
    B = k2 * r13 - k1 * r23
    eps = 1e-16
    if abs(A1) < eps and abs(A2) < eps:
        det = r11 * r22 - r12 * r12
        a1 = (r12 * r23 - r13 * r22) / det
        a2 = (r12 * r13 - r11 * r23) / det
    elif abs(A1) < eps:
        a1 = 0.0
        a2 = -B / A2
    elif abs(A2) < eps:
        a1 = -B / A1
        a2 = 0.0
    else:
        c = A1 / A2
        k = k1 - c * k2
        m0, m1, m2 = p0 - c * q0, p1 - c * q1, p2 - c * q2
        d = B / A2
        n0, n1, n2 = s0 - d * q0, s1 - d * q1, s2 - d * q2
        p11 = m0 * m0 + m1 * m1 + m2 * m2
        p12 = m0 * n0 + m1 * n1 + m2 * n2
        p22 = n0 * n0 + n1 * n1 + n2 * n2
        den = p11 - k * k
        if den <= 0.0:
            return INF
        sv = (p11 * p22 - p12 * p12) / den
        if not (sv >= 0.0):
            return INF
        a1 = -(p12 + k * np.sqrt(sv)) / p11
        a2 = -(B + a1 * A1) / A2
    a3 = 1.0 - a1 - a2
    t = 1e-12
    if not (a1 >= -t and a2 >= -t and a3 >= -t
            and a1 <= 1 + t and a2 <= 1 + t and a3 <= 1 + t):
        return INF
    dx = x4 - (a1 * x1 + a2 * x2 + a3 * x3)
    dy = y4 - (a1 * y1 + a2 * y2 + a3 * y3)
    dz = z4 - (a1 * z1 + a2 * z2 + a3 * z3)
    v = a1 * u1 + a2 * u2 + a3 * u3 + np.sqrt(dx * dx + dy * dy + dz * dz)
    return v if v == v else INF


@njit(**_JIT)
def _update(i, P, tets, n2e_ptr, n2e, T):
    """Smallest arrival at node ``i`` over its elements."""
    best = INF
    x4, y4, z4 = P[i, 0], P[i, 1], P[i, 2]
    for q in range(n2e_ptr[i], n2e_ptr[i + 1]):
        e = n2e[q]
        o0 = o1 = o2 = -1
        for j in range(4):
            v = tets[e, j]
            if v != i:
                if o0 < 0:
                    o0 = v
                elif o1 < 0:
                    o1 = v
                else:
                    o2 = v
        u1, u2, u3 = T[o0], T[o1], T[o2]
        if u1 == INF and u2 == INF and u3 == INF:
            continue
        x1, y1, z1 = P[o0, 0], P[o0, 1], P[o0, 2]
        x2, y2, z2 = P[o1, 0], P[o1, 1], P[o1, 2]
        x3, y3, z3 = P[o2, 0], P[o2, 1], P[o2, 2]
        c = _tetra(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4, u1, u2, u3)
        c = min(c, _triangle(x1, y1, z1, x2, y2, z2, x4, y4, z4, u1, u2))
        c = min(c, _triangle(x1, y1, z1, x3, y3, z3, x4, y4, z4, u1, u3))
        c = min(c, _triangle(x2, y2, z2, x3, y3, z3, x4, y4, z4, u2, u3))
        if c < best:
            best = c
    return best


@njit(parallel=True, **_JIT)
def _updates(idx, m, P, tets, n2e_ptr, n2e, T, out):
    for a in prange(m):
        out[a] = _update(idx[a], P, tets, n2e_ptr, n2e, T)


@njit(**_JIT)
def _fim(P, tets, n2e_ptr, n2e, n2n_ptr, n2n, sources, tol, max_iter):
    n = P.shape[0]
    T = np.full(n, INF)
    fixed = np.zeros(n, np.bool_)
    for s in sources:
        T[s] = 0.0
        fixed[s] = True
    active = np.zeros(n, np.bool_)
    waking = np.zeros(n, np.bool_)
    lst = np.empty(n, np.int64)
    nxt = np.empty(n, np.int64)
    cand = np.empty(n, np.int64)
    q = np.empty(n)
    qc = np.empty(n)
    m = 0
    for s in sources:
        for r in range(n2n_ptr[s], n2n_ptr[s + 1]):
            nb = n2n[r]
            if not fixed[nb] and not active[nb]:
                active[nb] = True
                lst[m] = nb
                m += 1
    it = 0
    while m > 0 and it < max_iter:
        _updates(lst, m, P, tets, n2e_ptr, n2e, T, q)
        mm = 0
        mc = 0
        for a in range(m):
            active[lst[a]] = False
        for a in range(m):
            i = lst[a]
            p = T[i]
            if q[a] < p:
                T[i] = q[a]
            if not (q[a] < p - tol):          # settled: wake the neighbours
                for r in range(n2n_ptr[i], n2n_ptr[i + 1]):
                    nb = n2n[r]
                    if not fixed[nb] and not waking[nb]:
                        waking[nb] = True
                        cand[mc] = nb
                        mc += 1
            else:                              # still moving: keep it
                active[i] = True
                nxt[mm] = i
                mm += 1
        _updates(cand, mc, P, tets, n2e_ptr, n2e, T, qc)
        for a in range(mc):
            nb = cand[a]
            waking[nb] = False
            if qc[a] < T[nb] - tol:
                T[nb] = qc[a]
                if not active[nb]:
                    active[nb] = True
                    nxt[mm] = nb
                    mm += 1
        lst, nxt = nxt, lst
        m = mm
        it += 1
    return T, it, m == 0


# ---------------------------------------------------------------------
# Adjacency
# ---------------------------------------------------------------------
@njit(**_JIT)
def _node_to_element(tets, n):
    ptr = np.zeros(n + 1, np.int64)
    for e in range(tets.shape[0]):
        for j in range(4):
            ptr[tets[e, j] + 1] += 1
    for i in range(n):
        ptr[i + 1] += ptr[i]
    fill = ptr[:-1].copy()
    out = np.empty(ptr[n], np.int32)
    for e in range(tets.shape[0]):
        for j in range(4):
            v = tets[e, j]
            out[fill[v]] = e
            fill[v] += 1
    return ptr, out


@njit(**_JIT)
def _node_to_node(tets, n, n2e_ptr, n2e):
    seen = np.full(n, -1, np.int64)
    ptr = np.zeros(n + 1, np.int64)
    for i in range(n):
        c = 0
        for q in range(n2e_ptr[i], n2e_ptr[i + 1]):
            for j in range(4):
                v = tets[n2e[q], j]
                if v != i and seen[v] != i:
                    seen[v] = i
                    c += 1
        ptr[i + 1] = ptr[i] + c
    out = np.empty(ptr[n], np.int32)
    seen[:] = -1
    for i in range(n):
        k = ptr[i]
        for q in range(n2e_ptr[i], n2e_ptr[i + 1]):
            for j in range(4):
                v = tets[n2e[q], j]
                if v != i and seen[v] != i:
                    seen[v] = i
                    out[k] = v
                    k += 1
    return ptr, out


@dataclass
class Adjacency:
    """Which elements and which nodes touch each node, as CSR arrays."""

    n2e_ptr: np.ndarray
    n2e: np.ndarray
    n2n_ptr: np.ndarray
    n2n: np.ndarray


def adjacency(tets: np.ndarray, n_points: int) -> Adjacency:
    """The adjacency :func:`distance` needs. Build it once per mesh and
    pass it to every solve on that mesh."""
    tets = np.ascontiguousarray(tets, dtype=np.int64)
    n2e_ptr, n2e = _node_to_element(tets, int(n_points))
    n2n_ptr, n2n = _node_to_node(tets, int(n_points), n2e_ptr, n2e)
    return Adjacency(n2e_ptr, n2e, n2n_ptr, n2n)


# ---------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------
@dataclass
class EikonalReport:
    """What one solve did."""

    iterations: int
    converged: bool
    unreached: int


def distance(points: np.ndarray, tets: np.ndarray, sources: np.ndarray, *,
             adj: Optional[Adjacency] = None,
             tol: Optional[float] = None,
             max_iter: int = 1_000_000,
             threads: Optional[int] = None):
    """``(T, report)``: geodesic distance from *sources* to every node.

    Nodes the front cannot reach (another connected piece of the mesh)
    are left at ``inf`` and counted in the report. *threads* limits the
    worker threads for this call; by default numba uses every core.
    """
    pts = np.ascontiguousarray(points, dtype=np.float64)
    tets = np.ascontiguousarray(tets, dtype=np.int64)
    src = np.unique(np.asarray(sources, dtype=np.int64))
    if len(src) == 0:
        raise ValueError("a distance needs at least one source node")
    if adj is None:
        adj = adjacency(tets, len(pts))
    if tol is None:
        extent = float(np.ptp(pts, axis=0).max()) if len(pts) else 1.0
        tol = DEFAULT_RELATIVE_TOL * (extent or 1.0)
    before = numba.get_num_threads()
    if threads is not None:
        numba.set_num_threads(max(1, min(int(threads), numba.config.NUMBA_NUM_THREADS)))
    try:
        T, it, done = _fim(pts, tets, adj.n2e_ptr, adj.n2e, adj.n2n_ptr, adj.n2n,
                           src, float(tol), int(max_iter))
    finally:
        numba.set_num_threads(before)
    return T, EikonalReport(iterations=int(it), converged=bool(done),
                            unreached=int(np.sum(~np.isfinite(T))))


def distance_ratio(to_zero: np.ndarray, to_one: np.ndarray) -> np.ndarray:
    """``g0 / (g0 + g1)``: 0 on the first set, 1 on the second.

    Where both distances are 0 (a node in both sets) the ratio is 0.5;
    where either is ``inf`` (unreached) it is ``nan``.
    """
    g0 = np.asarray(to_zero, dtype=float)
    g1 = np.asarray(to_one, dtype=float)
    total = g0 + g1
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(total > 0.0, g0 / np.where(total > 0.0, total, 1.0), 0.5)
    out[~(np.isfinite(g0) & np.isfinite(g1))] = np.nan
    return out


__all__ = ["Adjacency", "adjacency", "distance", "distance_ratio",
           "EikonalReport", "DEFAULT_RELATIVE_TOL"]
