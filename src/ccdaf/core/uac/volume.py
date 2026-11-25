"""
volume
======
Universal atrial coordinates in the wall of one atrium: α and β carried
through the wall from the surface UAC, and a third coordinate, the depth,
from the endocardium (0) to the epicardium (1).

Following Roney et al. 2023 (*Interface Focus* 13: 20230038, §2.5), with
one change the cut of ccdaf's surface UAC needs.

The input
---------
An atrial volume holding one atrium, whose boundary is labelled (rim,
epicardium, endocardium: :mod:`ccdaf.core.atrial_surfaces`) and whose
tetrahedra carry the atrial regions (body, veins, appendage:
:mod:`ccdaf.core.volume_tags`).

The steps
---------
1. **Surface UAC on the endocardium**: :func:`ccdaf.core.uac.compute`,
   unchanged, on the endocardium with the labels of the tetrahedra behind
   it. With ``epicardium="own"`` the epicardium gets its own surface UAC
   too, as in the paper.
2. **Halves through the wall.** The surface UAC open the atrium along a
   cut into a posterior and an anterior half; on the lateral and septal
   lines β jumps from b to 1 − b. Each tetrahedron takes the half of the
   endocardium behind it, by harmonic extension (as the regions do in
   :mod:`ccdaf.core.volume_tags`).
3. **α and β in the wall**, one Laplace problem per half, solved on that
   half's tetrahedra alone: a node on the face between the halves has a
   copy in each, so β keeps its jump instead of being averaged across it.
   Held at the surface UAC of that half on the endocardium (and, with
   ``"own"``, on the epicardium); free (homogeneous Neumann) on the rims,
   on the face between the halves and, with ``"endo"``, on the
   epicardium, so the values run straight out through the wall.
4. **Depth**: Laplace with 0 on the endocardium and 1 on the epicardium,
   free on the rims.

Which epicardium
----------------
Measured on one four-chamber left atrium (wall 2.35 mm, median): the
epicardium's own cut lies a median 2.5 mm from the endocardium's, so the
jump surface runs almost straight through the wall, and the two choices
differ by a median 0.004 in α and 0.002 in β (95%: 0.018, 0.014), more
towards the epicardium. The epicardium's own surface map had 122 folded
triangles (the endocardium's none): ``"own"`` carries those onto the
epicardium, ``"endo"`` does not. ``"endo"`` is the default; it is also what
the paper's published code does.

What is written
---------------
Node fields ``uac_alpha``, ``uac_beta`` (on the face between the halves,
the posterior value, as the surface UAC store it) and ``uac_depth``. Cell
fields ``uac_alpha_tet`` and ``uac_beta_tet`` (each tetrahedron's mean in
its own half, exact on both sides of the cut), ``uac_half`` (0 posterior,
1 anterior) and ``uac_folded`` (1 where the three coordinates turn the
tetrahedron inside out).

Folded tetrahedra
-----------------
Each tetrahedron belongs wholly to one half, so the face between the
halves follows tetrahedron faces in steps rather than the cut itself, and
a few tetrahedra just on the wrong side are turned inside out. Measured on
one four-chamber left atrium with ``"endo"``: 526 folded (0.24%), 377 of
them touching the face between the halves (3.8% of the tetrahedra there),
114 touching a rim. With ``"own"``: 1,390 (0.62%).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

from ccdaf.core import atrial_labels as al
from ccdaf.core.laplace import DirichletSystem, stiffness_matrix
from ccdaf.core.orifice_labels import UNITS, guess_unit
from ccdaf.core.surface_labels import (
    EPI, LA_ENDO, LABEL_MASK_FIELD, RA_ENDO, face_labels_from_mask,
)
from ccdaf.core.uac.flatten import POSTERIOR
from ccdaf.core.volume_mesh import boundary_surface, is_volume, tetrahedra

ALPHA, BETA, DEPTH = "uac_alpha", "uac_beta", "uac_depth"
ALPHA_TET, BETA_TET, HALF = "uac_alpha_tet", "uac_beta_tet", "uac_half"
FOLDED = "uac_folded"
#: Choices for the epicardium.
FROM_ENDO, OWN = "endo", "own"


class VolumeUacError(RuntimeError):
    """The volume cannot be given UAC, with the reason in words."""


@dataclass
class VolumeUac:
    """A finished run."""

    alpha: np.ndarray          # per node
    beta: np.ndarray
    depth: np.ndarray
    alpha_tet: np.ndarray      # per tetrahedron
    beta_tet: np.ndarray
    half: np.ndarray
    atrium: str
    epicardium: str
    endo_run: object           # the surface UacRun
    epi_run: Optional[object] = None
    seconds: float = 0.0
    notes: List[str] = field(default_factory=list)
    #: Tetrahedra the coordinates turn inside out (0 is the normal result).
    folded: int = 0
    folded_mask: Optional[np.ndarray] = None

    def summary(self) -> str:
        return (f"Volume UAC of the {self.atrium} computed in "
                f"{self.seconds:.0f} s (epicardium "
                f"{'from the endocardium' if self.epicardium == FROM_ENDO else 'with its own UAC'}).")

    def details(self) -> str:
        lines = [self.summary(), "", "Endocardium: " + self.endo_run.summary()]
        if self.epi_run is not None:
            lines.append("Epicardium: " + self.epi_run.summary())
        n = np.bincount(self.half, minlength=2)
        lines.append(f"Tetrahedra: {n[0]:,} posterior, {n[1]:,} anterior; "
                     f"{self.folded:,} folded ({100 * self.folded / max(n.sum(), 1):.2f}%).")
        lines += self.notes
        return "\n".join(lines)


def _atrium_of(grid) -> Tuple[str, str]:
    """``(atrium, reason)``: which atrium the volume is, or "" and why not."""
    if LABEL_MASK_FIELD not in grid.point_data:
        return "", "Label the atrial surfaces first (Actions > Label atrial surfaces…)."
    from ccdaf.core.volume_tags import endocardium_present
    atria = endocardium_present(grid)
    if not atria:
        return "", "Label the atrial surfaces first (Actions > Label atrial surfaces…)."
    if len(atria) > 1:
        return "", ("This volume holds both atria. Volume UAC work on one atrium: "
                    "extract each with Actions > Extract chambers…")
    atrium = atria[0]
    own = al.LA if atrium == "LA" else al.RA
    tags = set(np.unique(np.asarray(grid.cell_data["elemTag"]).astype(int)))
    missing = [n for n, v in own.regions.items() if v not in tags]
    if missing:
        return "", (f"The tetrahedra carry no {', '.join(missing)}: tag the "
                    "regions first (Actions > Tag the endocardium…).")
    return atrium, ""


def availability(dataset) -> Tuple[bool, str]:
    """Whether volume UAC can be computed on *dataset*, and if not, why."""
    if dataset is None or not is_volume(pv.wrap(dataset)):
        return False, "Load an atrial volume first."
    atrium, why = _atrium_of(pv.wrap(dataset))
    return bool(atrium), why


def _surface(grid, surface, labels, key):
    """Points (mesh units), triangles, tags and volume node ids of the
    boundary faces labelled *key*."""
    origin = np.asarray(surface.point_data["vtkOriginalPointIds"], dtype=np.int64)
    tri = np.asarray(surface.faces).reshape(-1, 4)[:, 1:]
    faces = np.where(labels == key)[0]
    nodes, local = np.unique(tri[faces], return_inverse=True)
    tags = np.asarray(surface.cell_data["elemTag"]).astype(int)[faces]
    return (np.asarray(surface.points, dtype=float)[nodes], local.reshape(-1, 3),
            tags, origin[nodes])


def _halves(run, points, vol) -> Tuple[Dict[int, Dict[int, Tuple[float, float]]],
                                       np.ndarray]:
    """Per half, volume node -> (α, β) of that half; and each repaired
    triangle's half. The repaired surface may copy a point (a pinch, a
    tunnel's cut): copies are matched back to the volume node they sit on;
    a tunnel patch's new centre point has none and is left out."""
    P = np.asarray(run.surface.P, dtype=float)
    d, idx = cKDTree(points).query(P)
    tol = 1e-6 * max(float(np.ptp(points, axis=0).max()), 1.0)
    to_vol = np.where(d <= tol, vol[idx], -1)
    att = run.attempt
    n_real = len(P)
    values: Dict[int, Dict[int, Tuple[float, float]]] = {0: {}, 1: {}}
    for h in att.halves:
        side = 0 if h.side == POSTERIOR else 1
        ok = h.nodes < n_real
        for node, uv in zip(h.nodes[ok], h.uv[ok]):
            v = to_vol[node]
            if v >= 0:
                values[side][int(v)] = (float(uv[0]), float(uv[1]))
    t = np.asarray(run.surface.t)
    face_half = np.zeros(len(t), dtype=int)
    for h in att.halves:
        tri = h.triangles[h.triangles < len(t)]
        face_half[tri] = 0 if h.side == POSTERIOR else 1
    return values, (to_vol[t], face_half)


def _status(cb, text):
    if cb is not None:
        cb(text)


def compute(grid, epicardium: str = FROM_ENDO, *, options=None,
            on_status: Optional[Callable[[str], None]] = None,
            should_stop: Optional[Callable[[], bool]] = None,
            rtol: float = 1e-8) -> VolumeUac:
    """Volume UAC of the one atrium *grid* holds (see the module notes).
    Raises :class:`VolumeUacError` with the reason when it cannot."""
    from ccdaf.core import uac as uac_core

    t0 = time.time()
    grid = pv.wrap(grid)
    atrium, why = _atrium_of(grid)
    if not atrium:
        raise VolumeUacError(why)
    if epicardium not in (FROM_ENDO, OWN):
        raise ValueError(f"epicardium must be {FROM_ENDO!r} or {OWN!r}")
    mm = UNITS[guess_unit(grid)]
    surface = boundary_surface(grid)
    origin = np.asarray(surface.point_data["vtkOriginalPointIds"], dtype=np.int64)
    labels = face_labels_from_mask(
        surface, np.asarray(grid.point_data[LABEL_MASK_FIELD])[origin])
    endo_key = LA_ENDO if atrium == "LA" else RA_ENDO

    def surface_uac(key, what):
        P, T, tags, vol = _surface(grid, surface, labels, key)
        _status(on_status, f"Surface UAC on the {what}…")
        try:
            run = uac_core.compute(P * mm, T, tags, options, on_status=on_status,
                                   should_stop=should_stop)
        except uac_core.UacError as exc:
            raise VolumeUacError(f"The {what}'s surface UAC failed: {exc}") from None
        return run, *_halves(run, P * mm, vol)

    endo_run, endo_values, (endo_tri, endo_half) = surface_uac(endo_key, "endocardium")
    epi_run, epi_values = None, {0: {}, 1: {}}
    if epicardium == OWN:
        epi_run, epi_values, _ = surface_uac(EPI, "epicardium")

    if should_stop is not None and should_stop():
        raise VolumeUacError("Stopped.")
    _status(on_status, "Carrying the halves through the wall…")
    X = np.asarray(grid.points, dtype=float) * mm
    tets = tetrahedra(grid)
    K = stiffness_matrix(X, tets)
    # Each endocardial node's half: the half most of its triangles are in.
    ok = (endo_tri >= 0).all(1)
    count = np.zeros((grid.n_points, 2))
    for k in range(3):
        np.add.at(count, (endo_tri[ok, k], endo_half[ok]), 1)
    held = np.where(count.sum(1) > 0)[0]
    u, _ = DirichletSystem(K, held).solve(
        (count[held, 1] > count[held, 0]).astype(float), rtol=rtol)
    half = (u[tets].mean(1) > 0.5).astype(np.int32)

    _status(on_status, "Solving the depth…")
    endo_nodes = np.unique(origin[np.asarray(surface.faces).reshape(-1, 4)[:, 1:][labels == endo_key]])
    epi_nodes = np.setdiff1d(np.unique(origin[np.asarray(surface.faces).reshape(-1, 4)[:, 1:][labels == EPI]]),
                             endo_nodes)
    held = np.concatenate([endo_nodes, epi_nodes])
    values = np.r_[np.zeros(len(endo_nodes)), np.ones(len(epi_nodes))]
    order = np.argsort(held)
    depth, _ = DirichletSystem(K, held[order]).solve(values[order], rtol=rtol)
    depth = np.clip(depth, 0.0, 1.0)

    _status(on_status, "Solving α and β in each half…")
    alpha = np.full(grid.n_points, np.nan)
    beta = np.full(grid.n_points, np.nan)
    alpha_tet = np.empty(len(tets))
    beta_tet = np.empty(len(tets))
    jac = np.empty(len(tets))
    for side in (1, 0):                          # posterior written last: it wins
        sel = half == side
        if not sel.any():
            continue
        sub = tets[sel]
        nodes, loc = np.unique(sub, return_inverse=True)
        loc = loc.reshape(-1, 4)
        position = {int(n): i for i, n in enumerate(nodes)}
        fixed: Dict[int, Tuple[float, float]] = {}
        for source in (endo_values[side], epi_values[side]):
            for n, v in source.items():
                if n in position:
                    fixed[position[n]] = v
        if not fixed:
            raise VolumeUacError("A half of the wall has no endocardium.")
        idx = np.array(sorted(fixed))
        vals = np.array([fixed[i] for i in idx])
        system = DirichletSystem(stiffness_matrix(X[nodes], loc), idx)
        a, _ = system.solve(vals[:, 0], rtol=rtol)
        b, _ = system.solve(vals[:, 1], rtol=rtol)
        alpha[nodes], beta[nodes] = a, b
        alpha_tet[sel], beta_tet[sel] = a[loc].mean(1), b[loc].mean(1)
        jac[sel] = _jacobians(X[nodes], loc, a, b, depth[nodes])

    # A tetrahedron is folded when (α, β, depth) turn it inside out: its
    # Jacobian has the sign opposite to most tetrahedra's.
    big = np.abs(jac) > 1e-12 * np.abs(jac).max()
    folded_mask = big & (np.sign(jac) == -np.sign(np.median(jac[big])))
    folded = int(folded_mask.sum())
    notes = []
    if epi_run is not None and epi_run.attempt.folded_real:
        notes.append(f"The epicardium's surface map has {epi_run.attempt.folded_real} "
                     "folded triangles, carried onto the epicardium of the volume.")
    return VolumeUac(alpha=alpha, beta=beta, depth=depth, alpha_tet=alpha_tet,
                     beta_tet=beta_tet, half=half, atrium=atrium,
                     epicardium=epicardium, endo_run=endo_run, epi_run=epi_run,
                     seconds=time.time() - t0, notes=notes, folded=folded,
                     folded_mask=folded_mask)


def _jacobians(X: np.ndarray, tets: np.ndarray, *fields: np.ndarray) -> np.ndarray:
    """Per tetrahedron, det of the gradient of the three *fields*."""
    e = X[tets[:, 1:]] - X[tets[:, :1]]                     # (n, 3, 3) edges
    grads = []
    for f in fields:
        df = f[tets[:, 1:]] - f[tets[:, :1]]                # (n, 3)
        grads.append(np.linalg.solve(e, df[..., None])[..., 0])
    return np.linalg.det(np.stack(grads, axis=1))


def write(grid, result: VolumeUac):
    """*grid* with the volume UAC fields added (in place, and returned)."""
    grid.point_data[ALPHA] = result.alpha
    grid.point_data[BETA] = result.beta
    grid.point_data[DEPTH] = result.depth
    grid.cell_data[ALPHA_TET] = result.alpha_tet
    grid.cell_data[BETA_TET] = result.beta_tet
    grid.cell_data[HALF] = result.half
    grid.cell_data[FOLDED] = (result.folded_mask.astype(np.int32)
                              if result.folded_mask is not None
                              else np.zeros(len(result.half), np.int32))
    return grid


def written_names() -> Tuple[str, ...]:
    return (ALPHA, BETA, DEPTH, ALPHA_TET, BETA_TET, HALF, FOLDED)


__all__ = ["ALPHA", "BETA", "DEPTH", "ALPHA_TET", "BETA_TET", "HALF", "FOLDED",
           "FROM_ENDO", "OWN", "VolumeUacError", "VolumeUac", "availability",
           "compute", "write", "written_names"]
