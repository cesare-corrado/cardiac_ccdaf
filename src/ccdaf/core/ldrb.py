"""
ldrb
====
Rule-based myocardial fibres from four Laplace fields (Bayer et al., Ann
Biomed Eng 2012, "LDRB").

Given the apicobasal field ``psi_ab`` and the three transmural fields
``phi_epi``, ``phi_lv``, ``phi_rv`` (see :mod:`ccdaf.core.laplace`), every
element gets an orthonormal frame: the fibre ``F``, and two directions
across it, ``S`` and ``T``.

Two modes
---------
``PUBLISHED`` follows the supplement's Algorithm 1, and is meant to agree
element by element with ``GlRuleFibers``, which implements it. It builds a
frame in each region, rotates each by its own angles, and blends the rotated
frames with bislerp. Blending two frames rotated by *different* angles is
what bends the transmural helix profile: in the free wall the result is
close to ``alpha_endo + (alpha_epi - alpha_endo) * d**2`` rather than
linear, and once the endo-to-epi difference passes 90 degrees bislerp's
choice between equivalent frames flips partway through the wall.

``LINEAR`` blends the *unrotated* frames and rotates once, by angles linear
in depth. It reproduces a reference whose profile is linear, which the
published algorithm cannot for any choice of angles.

Conventions
-----------
The frame is ``Q = (e0 e1 e2)`` by columns: ``e1`` along the apicobasal
gradient, ``e2`` the transmural gradient with its ``e1`` component removed,
``e0 = e1 x e2`` circumferential. The supplement's Function 2 subtracts the
projection on ``e0`` before ``e0`` exists; the text says ``e1`` is the
direction preserved, so ``e1`` is what is projected out here.

``orient`` rotates by ``alpha`` about ``e2`` and then by ``beta`` about the
new fibre, as the supplement's Function 3 writes it.

Three conventions the paper leaves ambiguous were settled by measurement,
running this module against ``GlRuleFibers`` on identical Laplace fields
over a 2.9-million-element ventricle until they agreed to 0.01 degrees at
the 99th percentile:

* **the septal sheet angle flips** from the LV side to the RV side, as the
  paper's main text states (``beta_s(0) = -beta_s(1)``), not the constant
  its equation 3 writes. With equation 3 the septal sheets came out 55
  degrees wrong and the RV free-wall fibres 11 degrees wrong, the latter
  through bislerp carrying the septal frame into the wall;
* **bislerp's equivalent frames are ``q * u``**, flips about the frame's
  own axes, not the ``u * q`` the supplement prints (0.03 against 1.7
  degrees median);
* **``GlRuleFibers``' second vector is the frame's third column**, the
  in-sheet direction the paper calls ``T`` and CARP's ``.lon`` calls the
  sheet. It is returned here as ``sheet``; the paper's ``S``, the sheet
  normal, is ``sheet_normal``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp

from ccdaf.core.laplace import basis_gradients

PUBLISHED = "published"
LINEAR = "linear"

#: How the equivalent frames are generated inside bislerp. The supplement
#: prints ``i . q``; with frames stored by columns, flipping a frame about
#: its *own* axes is ``q . i``. ``RIGHT`` is the one ``GlRuleFibers`` uses
#: (0.03 degrees median against 1.7 for ``LEFT``), and the only one that
#: returns the end frames at t = 0 and 1: ``LEFT`` turns the frame about
#: the world axes, so its candidates are other frames, not the same frame
#: with axes flipped. ``LEFT`` is kept only to reproduce the printed form.
LEFT = "left"      # u . q, as printed
RIGHT = "right"    # q . u

_EPS = 1e-12


@dataclass
class FibreAngles:
    """The four angles of Bayer et al. equations 1-4, in degrees.

    Defaults are ``GlRuleFibers``' own.
    """

    alpha_endo: float = 40.0
    alpha_epi: float = -50.0
    beta_endo: float = -65.0
    beta_epi: float = 25.0


# ---------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------
def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n < _EPS, 1.0, n)


def axis(grad_psi: np.ndarray, grad_phi: np.ndarray) -> np.ndarray:
    """Frames ``(e0 e1 e2)`` by columns, shape ``(n, 3, 3)``."""
    e1 = _unit(grad_psi)
    e2 = _unit(grad_phi - np.sum(grad_phi * e1, axis=1, keepdims=True) * e1)
    e0 = np.cross(e1, e2)
    return np.stack([e0, e1, e2], axis=2)


def orient(frames: np.ndarray, alpha_deg: np.ndarray,
           beta_deg: np.ndarray) -> np.ndarray:
    """Rotate each frame by ``alpha`` about ``e2``, then ``beta`` about ``F``."""
    a = np.radians(np.broadcast_to(alpha_deg, frames.shape[:1]))
    b = np.radians(np.broadcast_to(beta_deg, frames.shape[:1]))
    ca, sa, cb, sb = np.cos(a), np.sin(a), np.cos(b), np.sin(b)
    zero, one = np.zeros_like(a), np.ones_like(a)
    rz = np.stack([np.stack([ca, -sa, zero], axis=1),
                   np.stack([sa, ca, zero], axis=1),
                   np.stack([zero, zero, one], axis=1)], axis=1)
    rx = np.stack([np.stack([one, zero, zero], axis=1),
                   np.stack([zero, cb, sb], axis=1),
                   np.stack([zero, -sb, cb], axis=1)], axis=1)
    return frames @ rz @ rx


# ---------------------------------------------------------------------
# Quaternions, (w, x, y, z)
# ---------------------------------------------------------------------
def rot_to_quat(r: np.ndarray) -> np.ndarray:
    """Unit quaternions of proper rotations, by Shepperd's method."""
    n = r.shape[0]
    q = np.empty((n, 4))
    tr = r[:, 0, 0] + r[:, 1, 1] + r[:, 2, 2]
    c0 = tr > 0.0
    c1 = ~c0 & (r[:, 0, 0] >= r[:, 1, 1]) & (r[:, 0, 0] >= r[:, 2, 2])
    c2 = ~c0 & ~c1 & (r[:, 1, 1] >= r[:, 2, 2])
    c3 = ~c0 & ~c1 & ~c2

    s = np.sqrt(np.maximum(tr[c0] + 1.0, 0.0)) * 2.0
    q[c0] = np.stack([0.25 * s,
                      (r[c0, 2, 1] - r[c0, 1, 2]) / s,
                      (r[c0, 0, 2] - r[c0, 2, 0]) / s,
                      (r[c0, 1, 0] - r[c0, 0, 1]) / s], axis=1)
    s = np.sqrt(np.maximum(1.0 + r[c1, 0, 0] - r[c1, 1, 1] - r[c1, 2, 2], 0.0)) * 2.0
    q[c1] = np.stack([(r[c1, 2, 1] - r[c1, 1, 2]) / s, 0.25 * s,
                      (r[c1, 0, 1] + r[c1, 1, 0]) / s,
                      (r[c1, 0, 2] + r[c1, 2, 0]) / s], axis=1)
    s = np.sqrt(np.maximum(1.0 + r[c2, 1, 1] - r[c2, 0, 0] - r[c2, 2, 2], 0.0)) * 2.0
    q[c2] = np.stack([(r[c2, 0, 2] - r[c2, 2, 0]) / s,
                      (r[c2, 0, 1] + r[c2, 1, 0]) / s, 0.25 * s,
                      (r[c2, 1, 2] + r[c2, 2, 1]) / s], axis=1)
    s = np.sqrt(np.maximum(1.0 + r[c3, 2, 2] - r[c3, 0, 0] - r[c3, 1, 1], 0.0)) * 2.0
    q[c3] = np.stack([(r[c3, 1, 0] - r[c3, 0, 1]) / s,
                      (r[c3, 0, 2] + r[c3, 2, 0]) / s,
                      (r[c3, 1, 2] + r[c3, 2, 1]) / s, 0.25 * s], axis=1)
    return _unit(q)


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = _unit(q).T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=1),
    ], axis=1)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product, with ``R(a * b) = R(a) @ R(b)``."""
    w1, x1, y1, z1 = np.moveaxis(np.broadcast_to(a, np.broadcast_shapes(a.shape, b.shape)), -1, 0)
    w2, x2, y2, z2 = np.moveaxis(np.broadcast_to(b, np.broadcast_shapes(a.shape, b.shape)), -1, 0)
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], axis=-1)


_UNITS = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])


def bislerp(qa: np.ndarray, qb: np.ndarray, t: np.ndarray,
            order: str = RIGHT) -> np.ndarray:
    """Interpolate frames that are equal up to a half turn about their axes.

    ``qa`` and ``qb`` are rotation matrices, shape ``(n, 3, 3)``. Among the
    eight quaternions equivalent to ``qa`` (plus or minus ``qa`` times one
    of 1, i, j, k) the one closest to ``qb`` is taken, then slerped to it.
    """
    a = rot_to_quat(qa)
    b = rot_to_quat(qb)
    cands = np.stack([quat_mul(_UNITS[k], a) if order == LEFT
                      else quat_mul(a, _UNITS[k]) for k in range(4)], axis=1)
    dots = np.einsum("nkc,nc->nk", cands, b)
    best = np.argmax(np.abs(dots), axis=1)
    pick = cands[np.arange(len(a)), best]
    pick *= np.sign(dots[np.arange(len(a)), best])[:, None] + (dots[np.arange(len(a)), best] == 0)[:, None]

    t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)[:, None]
    cos = np.clip(np.sum(pick * b, axis=1, keepdims=True), -1.0, 1.0)
    theta = np.arccos(cos)
    sin = np.sin(theta)
    small = sin < 1e-8
    wa = np.where(small, 1.0 - t, np.sin((1.0 - t) * theta) / np.where(small, 1.0, sin))
    wb = np.where(small, t, np.sin(t * theta) / np.where(small, 1.0, sin))
    return quat_to_rot(wa * pick + wb * b)


# ---------------------------------------------------------------------
# The rules
# ---------------------------------------------------------------------
@dataclass
class Fibres:
    """Per-element directions, each shape ``(n, 3)``, unit length.

    ``sheet`` is the in-sheet direction across the fibre (the paper's
    ``T``), which is what CARP's ``.lon`` and ``GlRuleFibers`` write as the
    sheet. ``sheet_normal`` is the paper's ``S``.
    """

    fibre: np.ndarray
    sheet: np.ndarray          # third column of the frame, the paper's T
    sheet_normal: np.ndarray   # second column, the paper's S
    #: Per-element mask: a gradient this element needed was undefined and
    #: was rebuilt from its neighbours. See :func:`repair_gradients`.
    repaired: Optional[np.ndarray] = None
    #: Per-element mask: still undefined after repair, because the whole
    #: neighbourhood was too. Their frame is not meaningful.
    unresolved: Optional[np.ndarray] = None


#: A gradient smaller than this share of the field's median gradient is
#: treated as undefined. The gap it sits in is wide: on a 2.9-million-
#: element ventricle the undefined gradients are ~1e-15 of the median
#: (floating-point residue of an exact zero) and the smallest genuine ones
#: ~1e-5, so anything between works and the value is not critical.
DEGENERATE_RTOL: float = 1e-8

#: Sine of the angle below which the apicobasal and a transmural gradient
#: count as parallel. The case it catches is exact: an element with three
#: nodes on the epicardium at the base rim or the apex, where both fields
#: are held constant on the same face, so both gradients are its normal
#: (sine ~1e-13). Any small value separates that from genuine tissue.
PARALLEL_SINE: float = 1e-6

#: A frame blended in with less weight than this moves the result by at
#: most this share of the angle between frames, under a thousandth of a
#: degree, so an undefined gradient behind it does not matter.
_NEGLIGIBLE: float = 1e-6


def _centroid(values: np.ndarray, tets: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, float)[tets].mean(axis=1), 0.0, 1.0)


def _norm(v: np.ndarray) -> np.ndarray:
    return np.linalg.norm(v, axis=1)


def _scale(grad: np.ndarray) -> float:
    magnitude = _norm(grad)
    positive = magnitude[magnitude > 0.0]
    return float(np.median(positive)) if positive.size else 1.0


def degenerate_elements(tets: np.ndarray, fields: Dict[str, np.ndarray],
                        grads: Dict[str, np.ndarray], *,
                        rtol: float = DEGENERATE_RTOL,
                        parallel_sine: float = PARALLEL_SINE) -> np.ndarray:
    """Elements whose frame cannot be built from their own gradients.

    Two cases, both from elements lying against a labelled surface:

    * a gradient the frame uses is undefined, because the element's four
      nodes all lie on one surface and carry the same value (below ``rtol``
      of the field's median gradient);
    * the apicobasal gradient is parallel to a transmural one the frame
      uses, because three nodes lie on the epicardium where the base or
      apex holds the apicobasal field too. The frame needs two directions
      and has one.

    "Uses" matters: ``phi_lv`` is exactly 0 across most of the RV free wall
    once the fields are written to file, but the LV frame has no weight
    there, so its missing gradient is harmless and is not counted.
    """
    tets = np.asarray(tets, dtype=np.int64)
    epi = _centroid(fields["epi"], tets)
    lv = _centroid(fields["lv"], tets)
    rv = _centroid(fields["rv"], tets)
    ds = rv / np.where(lv + rv < _EPS, 1.0, lv + rv)
    used = {"ab": np.ones(len(tets), dtype=bool),
            "epi": epi > _NEGLIGIBLE,
            "lv": (1.0 - epi) * (1.0 - ds) > _NEGLIGIBLE,
            "rv": (1.0 - epi) * ds > _NEGLIGIBLE}
    bad = np.zeros(len(tets), dtype=bool)
    for k, g in grads.items():
        bad |= used[k] & (_norm(g) < rtol * _scale(g))
    a = _unit(grads["ab"])
    for k in ("epi", "lv", "rv"):
        sine = _norm(np.cross(a, _unit(grads[k])))
        bad |= used[k] & (sine < parallel_sine)
    return bad


def repair_gradients(tets: np.ndarray, grad: np.ndarray, volume: np.ndarray,
                     n_points: int, targets: np.ndarray, *,
                     rtol: float = DEGENERATE_RTOL, passes: int = 5
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Rebuild the gradient of each element in ``targets`` from around it.

    The paper does not say what to do where a frame is undefined (see
    :func:`degenerate_elements`). ``GlRuleFibers`` writes a placeholder,
    the global x axis turned about z by the helix angle, which on a
    2.9-million-element ventricle sits a median 47 degrees from its
    neighbours; that is not a fibre, so it is not copied.

    Instead each target takes the mean, over its four nodes, of the
    volume-weighted average gradient of the elements at that node,
    counting only elements that are not targets themselves. Every field is
    rebuilt this way, not only the one that failed: an element lying flat
    against a surface fixes the gradient's normal component with a very
    short height, so its other gradients are badly conditioned too.

    Averaging needs no sign care: these are gradients of one scalar field,
    and they all point up it. A target whose neighbourhood is all targets
    is retried once its neighbours have been rebuilt, up to ``passes``
    times.

    Returns the gradients (a copy) and the mask of targets still without a
    usable gradient.
    """
    grad = np.array(grad, dtype=float, copy=True)
    targets = np.asarray(targets, dtype=bool)
    floor = rtol * _scale(grad)
    source = ~targets & (_norm(grad) >= floor)
    missing = targets.copy()
    if not missing.any():
        return grad, missing
    rows = np.asarray(tets, dtype=np.int64).ravel()
    cols = np.repeat(np.arange(len(tets)), 4)
    incidence = sp.csr_matrix((np.ones(len(rows)), (rows, cols)),
                              shape=(n_points, len(tets)))
    for _ in range(passes):
        if not missing.any():
            break
        weight = np.where(source, volume, 0.0)
        num = incidence @ (weight[:, None] * grad)
        den = incidence @ weight
        node = num / np.where(den > 0.0, den, 1.0)[:, None]
        idx = np.where(missing)[0]
        candidate = node[tets[idx]].mean(axis=1)
        found = _norm(candidate) >= floor
        grad[idx[found]] = candidate[found]
        missing[idx[found]] = False
        source[idx[found]] = True
    return grad, missing


def fibres(points: np.ndarray, tets: np.ndarray, fields: Dict[str, np.ndarray],
           angles: Optional[FibreAngles] = None, *, mode: str = PUBLISHED,
           order: str = RIGHT, chunk: int = 400_000,
           rtol: float = DEGENERATE_RTOL) -> Fibres:
    """Fibre, sheet and transverse directions for every element.

    ``fields`` carries the nodal Laplace solutions under ``"ab"``,
    ``"epi"``, ``"lv"`` and ``"rv"``. Field values are averaged to each
    element and clipped to [0, 1]: a mesh whose slivers break the discrete
    maximum principle overshoots slightly, and a weight outside [0, 1]
    would extrapolate an angle rather than interpolate it.

    Where an element's frame is undefined (:func:`degenerate_elements`) its
    gradients are rebuilt from its neighbours first
    (:func:`repair_gradients`). That is the one place this deliberately
    differs from ``GlRuleFibers``; everywhere else it matches it.
    """
    angles = angles or FibreAngles()
    if mode not in (PUBLISHED, LINEAR):
        raise ValueError(f"unknown mode {mode!r}")
    tets = np.asarray(tets, dtype=np.int64)
    points = np.asarray(points, dtype=float)

    # All gradients first: an element's replacement comes from its
    # neighbours, which a chunk boundary would cut off.
    basis, volume = basis_gradients(points, tets)
    grads = {k: np.einsum("nij,nj->ni", basis, np.asarray(fields[k], float)[tets])
             for k in ("ab", "epi", "lv", "rv")}
    del basis
    targets = degenerate_elements(tets, fields, grads, rtol=rtol)
    unresolved = np.zeros(len(tets), dtype=bool)
    if targets.any():
        for k in grads:
            grads[k], _ = repair_gradients(tets, grads[k], volume, len(points),
                                           targets, rtol=rtol)
        # A field with no gradient anywhere around matters only where the
        # frame uses it, which is what the recheck decides.
        unresolved = targets & degenerate_elements(tets, fields, grads, rtol=rtol)
    repaired = targets & ~unresolved

    out = [np.empty((len(tets), 3)) for _ in range(3)]
    for start in range(0, len(tets), chunk):
        sl = slice(start, start + chunk)
        tt = tets[sl]
        g = {k: grads[k][sl] for k in grads}
        epi = _centroid(fields["epi"], tt)
        lv = _centroid(fields["lv"], tt)
        rv = _centroid(fields["rv"], tt)
        ds = rv / np.where(lv + rv < _EPS, 1.0, lv + rv)

        # Both septal angles flip from the LV side to the RV side. For beta
        # that is the paper's main text, not its equation 3, and it is what
        # GlRuleFibers does: see the module docstring.
        a_s = angles.alpha_endo * (1.0 - ds) - angles.alpha_endo * ds
        b_s = angles.beta_endo * (1.0 - ds) - angles.beta_endo * ds
        a_w = angles.alpha_endo * (1.0 - epi) + angles.alpha_epi * epi
        b_w = angles.beta_endo * (1.0 - epi) + angles.beta_epi * epi

        f_lv = axis(g["ab"], -g["lv"])
        f_rv = axis(g["ab"], g["rv"])
        f_epi = axis(g["ab"], g["epi"])
        if mode == PUBLISHED:
            q_endo = bislerp(orient(f_lv, a_s, b_s), orient(f_rv, a_s, b_s), ds, order)
            q = bislerp(q_endo, orient(f_epi, a_w, b_w), epi, order)
        else:
            base = bislerp(bislerp(f_lv, f_rv, ds, order), f_epi, epi, order)
            alpha = (1.0 - epi) * a_s + epi * angles.alpha_epi
            beta = (1.0 - epi) * b_s + epi * angles.beta_epi
            q = orient(base, alpha, beta)
        out[0][sl], out[1][sl], out[2][sl] = q[:, :, 0], q[:, :, 1], q[:, :, 2]
    return Fibres(fibre=_unit(out[0]), sheet=_unit(out[2]),
                  sheet_normal=_unit(out[1]),
                  repaired=repaired & ~unresolved, unresolved=unresolved)


__all__ = ["FibreAngles", "Fibres", "fibres", "repair_gradients",
           "degenerate_elements", "DEGENERATE_RTOL", "PARALLEL_SINE", "axis", "orient", "bislerp",
           "rot_to_quat", "quat_to_rot", "quat_mul", "PUBLISHED", "LINEAR",
           "LEFT", "RIGHT"]
