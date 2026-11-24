"""
slim
====
Fold-free warp of a flat triangle map onto target positions.

Scalable Locally Injective Mappings (Rabinovich et al., 2017), for a map
of the plane to itself. The energy is

    sum_t a_t (|J_t|^2 + |J_t^-1|^2) + lam * sum_l |u_l - target_l|^2

where ``J_t`` maps triangle ``t`` of the rest map onto its current shape
and ``a_t`` is its rest area. The first term (symmetric Dirichlet) grows
without bound as a triangle collapses, so a line search that never lets a
triangle flip keeps the map one-to-one at every step. The second term pulls
the landmark points (rings, vein ends) towards their targets with a
stiffness ``lam`` that grows every iteration. Hard points (the square's
edges) are held at their targets in every solve.

Each iteration solves one sparse symmetric positive definite system (the
local-global "global step"), assembled directly on its normal equations.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

from ccdaf.core.uac.solvers import SpdSolver

_E = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])


class Cancelled(RuntimeError):
    """The caller asked the run to stop."""


def _grad_ops(R, tri):
    r0, r1, r2 = R[tri[:, 0]], R[tri[:, 1]], R[tri[:, 2]]
    Dm = np.stack([r1 - r0, r2 - r0], axis=2)
    det = Dm[:, 0, 0] * Dm[:, 1, 1] - Dm[:, 0, 1] * Dm[:, 1, 0]
    with np.errstate(all="ignore"):
        inv = np.stack([np.stack([Dm[:, 1, 1], -Dm[:, 0, 1]], 1),
                        np.stack([-Dm[:, 1, 0], Dm[:, 0, 0]], 1)], 1) / det[:, None, None]
    return np.einsum("kj,tjm->tkm", _E, inv), 0.5 * np.abs(det), np.sign(det)


def _jac(U, tri, G):
    """Per-triangle 2x2 Jacobians as four arrays ``(j00, j01, j10, j11)``."""
    Uk = U[tri]
    return ((Uk[:, :, 0] * G[:, :, 0]).sum(1), (Uk[:, :, 0] * G[:, :, 1]).sum(1),
            (Uk[:, :, 1] * G[:, :, 0]).sum(1), (Uk[:, :, 1] * G[:, :, 1]).sum(1))


def _energy(U, tri, G, a, lm, tg, lam):
    j00, j01, j10, j11 = _jac(U, tri, G)
    det = j00 * j11 - j01 * j10
    if (det <= 0).any():
        return np.inf
    fro = j00 ** 2 + j01 ** 2 + j10 ** 2 + j11 ** 2
    return float((a * (fro + fro / det ** 2)).sum() + lam * ((U[lm] - tg) ** 2).sum())


def _svd2(j00, j01, j10, j11):
    """Closed-form SVD of 2x2 matrices with positive determinant:
    ``J = Rot(phi) diag(s1, s2) Rot(theta)``. Returns ``s1, s2, cos(phi),
    sin(phi)`` and the cosine and sine of ``phi + theta``, the angle of the
    closest rotation to ``J``."""
    E, F = 0.5 * (j00 + j11), 0.5 * (j00 - j11)
    Gq, H = 0.5 * (j10 + j01), 0.5 * (j10 - j01)
    Q, R = np.hypot(E, H), np.hypot(F, Gq)
    a1, a2 = np.arctan2(Gq, F), np.arctan2(H, E)
    phi = 0.5 * (a2 + a1)
    return Q + R, Q - R, np.cos(phi), np.sin(phi), np.cos(a2), np.sin(a2)


def max_step(U: np.ndarray, D: np.ndarray, tri: np.ndarray) -> float:
    """Largest ``s > 0`` with no triangle of ``U + s D`` changing orientation."""
    p0, p1, p2 = U[tri[:, 0]], U[tri[:, 1]], U[tri[:, 2]]
    d0, d1, d2 = D[tri[:, 0]], D[tri[:, 1]], D[tri[:, 2]]
    e1, e2, f1, f2 = p1 - p0, p2 - p0, d1 - d0, d2 - d0

    def cr(x, y):
        return x[:, 0] * y[:, 1] - x[:, 1] * y[:, 0]

    qa, qb, qc = cr(f1, f2), cr(e1, f2) + cr(f1, e2), cr(e1, e2)
    s = np.full(len(tri), np.inf)
    with np.errstate(all="ignore"):
        lin = np.abs(qa) < 1e-30
        r = -qc / qb
        s[lin & (r > 0)] = r[lin & (r > 0)]
        disc = qb * qb - 4 * qa * qc
        sq = np.sqrt(np.maximum(disc, 0))
        for rr in ((-qb - sq) / (2 * qa), (-qb + sq) / (2 * qa)):
            ok = ~lin & (disc >= 0) & (rr > 0)
            s[ok] = np.minimum(s[ok], rr[ok])
    return float(s.min()) if len(s) else np.inf


def slim(R: np.ndarray, tri: np.ndarray, lm: np.ndarray, tg: np.ndarray,
         hard: np.ndarray, hard_tg: np.ndarray, *,
         circles: Optional[Sequence[Tuple[np.ndarray, np.ndarray, float]]] = None,
         iters: int = 100, lam0: float = 1.0, lam_mult: float = 1.2, lam_max: float = 1e7,
         solver: Optional[SpdSolver] = None,
         should_stop: Optional[Callable[[], bool]] = None,
         on_iteration: Optional[Callable[[int], None]] = None):
    """Warp the rest map ``R`` so ``lm`` points reach ``tg`` and ``hard``
    points sit at ``hard_tg``.

    ``circles``: ``(positions in lm, centre, radius)``; those landmarks slide
    on their circle (their target is the nearest point of it).

    Returns ``U, history, kept``: the warped points, per-iteration
    ``(lam, step, worst landmark miss, worst hard miss)``, and which
    triangles had a usable rest shape (a triangle degenerate in ``R`` has
    none; its points are tied to their neighbours by weak springs).
    """
    solver = solver or SpdSolver()
    n = len(R)
    R = R.copy()
    tg = np.asarray(tg, float).copy()
    lm = np.asarray(lm, int)
    hard = np.asarray(hard, int)
    hard_tg = np.asarray(hard_tg, float).copy()
    G, a, sgn = _grad_ops(R, tri)
    mirrored = (sgn < 0).sum() > (sgn > 0).sum()
    flip = np.array([-1.0, 1.0])
    if mirrored:                       # work on a positively oriented copy
        R, tg, hard_tg = R * flip, tg * flip, hard_tg * flip
        if circles:
            circles = [(ix, np.asarray(c) * flip, r) for ix, c, r in circles]
        G, a, sgn = _grad_ops(R, tri)
    keep = (sgn > 0) & (a > 1e-16)
    tri_k, G_k, a_k = tri[keep], G[keep], a[keep]
    N2 = 2 * n
    kk, ll, cc, dd = (x.ravel() for x in np.meshgrid(np.arange(3), np.arange(3), np.arange(2),
                                                     np.arange(2), indexing="ij"))
    rows = (tri_k[:, kk] + cc * n).ravel()
    cols = (tri_k[:, ll] + dd * n).ravel()
    GG = np.einsum("tkj,tlj->tkl", G_k, G_k)
    GGkl = GG[:, kk, ll]                                         # (T, 36)
    free = np.ones(N2, bool)
    hd = np.r_[hard, hard + n]
    free[hd] = False
    xfull = np.zeros(N2)
    xfull[hd] = np.r_[hard_tg[:, 0], hard_tg[:, 1]]
    xh = xfull[~free]
    ex = tri[~keep]
    Lsp = sp.coo_matrix((N2, N2))
    if len(ex):
        e = np.r_[ex[:, [0, 1]], ex[:, [1, 2]], ex[:, [2, 0]]]
        dg = np.bincount(tri_k.ravel(), weights=(a_k[:, None] * GG[:, [0, 1, 2], [0, 1, 2]]).ravel(),
                         minlength=n)
        wgt = 0.25 * float(np.median(dg[dg > 0]))
        A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
        A = ((A + A.T) > 0).astype(float) * wgt
        L1 = sp.diags(np.asarray(A.sum(1)).ravel()) - A
        Lsp = sp.block_diag([L1, L1]).tocoo()
    dlm = np.r_[lm, lm + n]
    # The matrix pattern never changes: map every contribution to its slot
    # once, then each iteration only sums values into the slots.
    keys = np.r_[rows * N2 + cols, Lsp.row.astype(np.int64) * N2 + Lsp.col, dlm * N2 + dlm]
    uniq, slot = np.unique(keys, return_inverse=True)
    ur, uc = uniq // N2, uniq % N2
    fmap = np.full(N2, -1)
    fmap[free] = np.arange(free.sum())
    ff = np.where(free[ur] & free[uc])[0]
    fh = np.where(free[ur] & ~free[uc])[0]
    Mf = sp.csc_matrix((np.arange(1.0, len(ff) + 1), (fmap[ur[ff]], fmap[uc[ff]])),
                       shape=(free.sum(), free.sum()))
    Mf.sort_indices()
    perm = ff[Mf.data.astype(np.int64) - 1]
    fh_row, fh_x = fmap[ur[fh]], xfull[uc[fh]]
    r_idx = (tri_k[:, :, None] + np.arange(2)[None, None, :] * n).ravel()
    U = R.copy()
    lam = lam0
    hist: List[Tuple[float, float, float, float]] = []
    for it in range(iters):
        if should_stop is not None and should_stop():
            raise Cancelled("stopped")
        if circles:
            for ix, c, r in circles:
                dv = U[lm[ix]] - c
                tg[ix] = c + r * dv / np.maximum(np.linalg.norm(dv, axis=1, keepdims=True), 1e-12)
        s1, s2, cp, spn, ca, sa = _svd2(*_jac(U, tri_k, G_k))
        w2 = []
        for s in (s1, s2):
            s = np.clip(s, 1e-8, None)
            with np.errstate(all="ignore"):
                w = (s - s ** -3) / (s - 1.0)
            w2.append(np.where(np.abs(s - 1.0) < 1e-6, 4.0, w))      # w squared
        # W^T W = Rot(phi) diag(w1^2, w2^2) Rot(phi)^T
        q00 = cp * cp * w2[0] + spn * spn * w2[1]
        q11 = spn * spn * w2[0] + cp * cp * w2[1]
        q01 = cp * spn * (w2[0] - w2[1])
        WtW = np.stack([np.stack([q00, q01], 1), np.stack([q01, q11], 1)], 1)   # (T, 2, 2)
        vals = (a_k[:, None] * GGkl * WtW[:, cc, dd]).ravel()
        data = np.bincount(slot, weights=np.r_[vals, Lsp.data, np.full(len(dlm), lam)],
                           minlength=len(uniq))
        Mf.data = data[perm]
        # right-hand side: a_t G_t (W^T W Rot)^T, plus the landmark springs
        m00, m01 = q00 * ca + q01 * sa, -q00 * sa + q01 * ca
        m10, m11 = q01 * ca + q11 * sa, -q01 * sa + q11 * ca
        rk0 = a_k[:, None] * (G_k[:, :, 0] * m00[:, None] + G_k[:, :, 1] * m01[:, None])
        rk1 = a_k[:, None] * (G_k[:, :, 0] * m10[:, None] + G_k[:, :, 1] * m11[:, None])
        r = np.bincount(r_idx, weights=np.stack([rk0, rk1], 2).ravel(), minlength=N2)
        r += np.bincount(dlm, weights=lam * np.r_[tg[:, 0], tg[:, 1]], minlength=N2)
        rf = r[free] - np.bincount(fh_row, weights=data[fh] * fh_x, minlength=len(fmap[free]))
        x = np.empty(N2)
        x[~free] = xh
        x[free] = solver.solve(Mf, rf)
        D = np.c_[x[:n], x[n:]] - U
        step = min(1.0, 0.9 * max_step(U, D, tri_k))
        e0 = _energy(U, tri_k, G_k, a_k, lm, tg, lam)
        while step > 1e-8 and _energy(U + step * D, tri_k, G_k, a_k, lm, tg, lam) > e0:
            step *= 0.5
        U = U + step * D
        miss = np.linalg.norm(U[lm] - tg, axis=1).max() if len(lm) else 0.0
        hm = np.linalg.norm(U[hard] - hard_tg, axis=1).max() if len(hard) else 0.0
        hist.append((lam, step, float(miss), float(hm)))
        if on_iteration is not None:
            on_iteration(it)
        lam = min(lam * lam_mult, lam_max)
    if mirrored:
        U = U * flip
    return U, hist, keep


__all__ = ["slim", "max_step", "Cancelled"]
