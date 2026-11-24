"""
pipeline
========
Universal atrial coordinates for a labelled left-atrial surface: the engine
behind Actions > Compute UAC.

Input: a triangle surface with ``elemTag`` 1 (body), 11 (LSPV), 13 (LIPV),
15 (RSPV), 17 (RIPV), 19 (LAA), open at the mitral valve and at each vein.
No landmarks are needed. Lengths are taken to be in millimetres.

Steps:

1. repair the surface (:func:`prepare`): label islands join their
   neighbour, duplicate triangles, winding, ears and slivers on openings,
   Delaunay flips, pinched rims, and tunnels (shortest loop under 15 mm,
   cut and closed with a patch that takes the label round it; a wider one
   is refused). The repairs are kept: the caller writes the repaired
   surface back, so the stored atrium and its flat map match;
2. lengthen short veins with a virtual extension (never written);
3. cut the atrium open (:mod:`ccdaf.core.uac.cut`);
4. map each half to the square and warp it onto the template
   (:mod:`ccdaf.core.uac.flatten`).

Two fallbacks, each validated on 150 atria: if the first map of a half is
degenerate, the roof is re-cut between the closest points of the superior
ostia ("roof-first"); if the cut fails or a ring misses its template target
by more than 0.05, the whole run is repeated without the ring penalty and
the better result kept.

Output: the repaired surface with ``uac_alpha`` and ``uac_beta`` on its
points. A point on the cut has two values, one per half; the point arrays
carry the posterior one, and :func:`flat_from_values` recovers the other,
so the flat map can be drawn from any atrium that has the two arrays.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp

# surface_repair uses this package's mesh helpers, so it is used here as a
# module (its names are looked up when a run starts, not at import).
from ccdaf.core import surface_repair as sr
from ccdaf.core.uac import cut as _cut_la
from ccdaf.core.uac import layout as lay
from ccdaf.core.uac import template as tpl
from ccdaf.core.uac.cut import CutError, label_rings
from ccdaf.core.uac.extension import extend_veins
from ccdaf.core.uac.flatten import (
    ANTERIOR, COLLAPSED, POSTERIOR, DegenerateFirstMap, HalfMap, flatten_half, signed_areas,
)
from ccdaf.core.uac.mesh_ops import unique_edges
from ccdaf.core.uac.slim import Cancelled
from ccdaf.core.uac.solvers import SpdSolver
from ccdaf.core.uac.tunnels import TunnelError

ALPHA, BETA = "uac_alpha", "uac_beta"
TAG = "elemTag"
#: A ring further than this from its template target triggers the retry
#: without the ring penalty.
RETRY_MISS = 0.05
#: An ostium longer than this (mm) is reported: its label probably runs onto
#: the body, which costs accuracy.
LARGE_OSTIUM_MM = 90.0


class UacError(RuntimeError):
    """The coordinates cannot be computed; the message says why."""


@dataclass
class Options:
    tunnel_max_mm: float = 15.0
    ring_penalty: float = 4.0
    penalty_mm: float = 2.0
    iterations: int = 100
    solver: Optional[str] = None      # None: the fastest available


@dataclass(frozen=True)
class Atrium:
    """What differs between the atria: the template (in the working frame,
    see :mod:`ccdaf.core.uac.layout`), the vessels to lengthen, and how the
    labels are checked and the atrium is cut open."""

    layout: lay.Layout
    vessels: Tuple[int, ...]
    sleeve_ratio: Dict[int, float]
    check_sides: Callable
    anatomy: Callable
    find_cut: Callable
    half_names: Tuple[str, str] = ("Posterior", "Anterior")
    #: Whether the cut has a second way of placing the roof to fall back on
    #: when a first map is degenerate.
    roof_first: bool = True

    @property
    def names(self) -> Dict[int, str]:
        return self.layout.names


LEFT = Atrium(layout=lay.LA, vessels=tpl.VEINS, sleeve_ratio=tpl.SLEEVE_RATIO,
              check_sides=_cut_la.check_sides, anatomy=_cut_la.anatomy,
              find_cut=_cut_la.find_cut)


@dataclass
class Attempt:
    penalty: float
    roof_first: bool
    rule: str
    barrier: bool
    halves: Tuple[HalfMap, HalfMap]
    cut_nodes: np.ndarray

    @property
    def folded_real(self) -> int:
        return sum(h.folded_real for h in self.halves)

    @property
    def collapsed(self) -> List[Dict]:
        return [c for h in self.halves for c in h.collapsed]

    @property
    def worst_miss(self) -> float:
        return max([m for h in self.halves for m in h.ring_miss.values()] + [0.0])


@dataclass
class UacRun:
    """A finished run. ``surface`` is the repaired atrium the coordinates
    belong to (see :mod:`ccdaf.core.surface_repair`); ``alpha`` and
    ``beta`` are on its points."""

    alpha: np.ndarray
    beta: np.ndarray
    surface: "sr.Edit"
    solver: str
    seconds: float
    attempt: Attempt
    extension: Dict
    warnings: List[str] = field(default_factory=list)
    atrium: Atrium = LEFT

    @property
    def repair(self) -> Dict:
        return self.surface.report

    @property
    def tunnels(self) -> List[Dict]:
        return list(self.surface.report.get("tunnels", []))

    def summary(self) -> str:
        a = self.attempt
        return (f"UAC computed in {self.seconds:.0f} s: {a.folded_real} folded triangles, "
                f"worst ring {a.worst_miss:.3f} from the template.")

    def details(self) -> str:
        a = self.attempt
        lines = [self.summary(), ""]
        lines.append(f"Solver: {self.solver}.")
        if self.atrium.roof_first:
            lines.append(f"Cut: lateral rule '{a.rule}', barrier {'on' if a.barrier else 'off'}, "
                         f"roof {'between the closest ostium points' if a.roof_first else 'default'}, "
                         f"ring penalty {'on' if a.penalty > 0 else 'off'}.")
        else:
            lines.append(f"Cut: shortest lines, barriers {'on' if a.barrier else 'off'}, "
                         f"ring penalty {'on' if a.penalty > 0 else 'off'}.")
        names = self.atrium.names
        for h in a.halves:
            name = self.atrium.half_names[0 if h.side == POSTERIOR else 1]
            rings = ", ".join(f"{_ring_name(k, names)} {v:.3f}"
                              for k, v in sorted(h.ring_miss.items()))
            lines.append(f"{name}: {h.folded_real} folded on tissue, {h.folded_added} on "
                         f"extensions; worst miss per ring: {rings}.")
        for h in a.halves:
            for c in h.collapsed:
                lines.append(f"Collapsed to a point: a {c['size_mm']} mm patch on the "
                             f"{names.get(c['label'], c['label'])} ({c['triangles']} "
                             f"triangles), most likely a thin spike in the segmentation. Its "
                             f"points take the coordinates of its base.")
        lines.append(repair_words(self.repair))
        ext = [f"{names[L]} +{v['added_mm']} mm" for L, v in self.extension.items()
               if isinstance(v, dict) and v.get("added_mm")]
        if ext:
            lines.append("Virtual extensions (not tissue, not written): "
                         + ", ".join(ext) + ".")
        lines += [f"Warning: {w}" for w in self.warnings]
        lines.append("")
        lines.append("Written: uac_alpha, uac_beta (points on the cut carry their "
                     f"{self.atrium.half_names[0].lower()} value; the flat map has both).")
        return "\n".join(lines)


def repair_words(r: Dict) -> str:
    """What the repair changed on the mesh, in one line."""
    parts = [
        (r.get("island_triangles_relabelled", 0), "triangles of label islands relabelled"),
        (r.get("duplicates_removed", 0), "duplicate triangles removed"),
        (r.get("rewound", 0), "triangles rewound"),
        (r.get("ears_removed", 0), "ears removed"),
        (r.get("rim_slivers_removed", 0), "slivers on openings removed"),
        (r.get("flips", 0), "edges flipped"),
        (r.get("pinches_split", 0), "pinched points split"),
    ]
    done = [f"{n} {what}" for n, what in parts if n]
    for tn in r.get("tunnels", []):
        done.append(f"a tunnel cut (loop {tn['loop_mm']} mm at {tn['centre']}) and closed "
                    f"with {tn['patch_triangles']} triangles")
    if not done:
        return "Mesh: no repair needed."
    return "Mesh repaired (Undo puts it back): " + ", ".join(done) + "."


def _ring_name(key: str, names=None) -> str:
    kind, L = key.split()
    return f"{(names or tpl.NAMES)[int(L)]} {'end' if kind == 'end' else 'ostium'}"


def _status(cb, text):
    if cb is not None:
        cb(text)


def _attempt(P, t, tag, is_real, anat, opts, penalty, solver, on_status, should_stop,
             atrium: Atrium = LEFT):
    roof_first = False
    while True:
        _status(on_status, "Cutting the atrium open…")
        cut = atrium.find_cut(P, t, tag, anat, roof_first=roof_first, ring_penalty=penalty,
                              penalty_mm=opts.penalty_mm)
        halves = []
        try:
            for side, name in ((POSTERIOR, atrium.half_names[0].lower()),
                               (ANTERIOR, atrium.half_names[1].lower())):
                def tick(it, name=name):
                    if it % 10 == 0:
                        _status(on_status, f"Warping the {name} half onto the template: "
                                           f"iteration {it + 1} of {opts.iterations}…")
                _status(on_status, f"Flattening the {name} half…")
                halves.append(flatten_half(P, t, tag, is_real, cut, anat, side,
                                           retry_if_degenerate=atrium.roof_first and not roof_first,
                                           solver=solver, iters=opts.iterations,
                                           should_stop=should_stop, on_iteration=tick,
                                           layout=atrium.layout))
        except DegenerateFirstMap:
            roof_first = True
            continue
        return Attempt(penalty, roof_first, cut.rule, cut.barrier, tuple(halves), cut.nodes)


#: The repairs (steps of :mod:`ccdaf.core.surface_repair`), in order, that
#: make a labelled surface ready to be laid flat. They are written to the
#: mesh, so the stored atrium and its flat map always match; a second run
#: finds nothing left to do.
REPAIRS = ("join_islands", "remove_duplicates", "orient", "trim_rims", "flip_edges",
           "trim_rims", "split_pinches")


def prepare(points: np.ndarray, triangles: np.ndarray, tags: np.ndarray,
            tunnel_max_mm: float = 15.0) -> "sr.Edit":
    """Repair a labelled surface for UAC (see :data:`REPAIRS`), cutting thin
    tunnels; raises :class:`UacError` on a wider one."""
    e = sr.Edit.start(points, triangles, tags)
    for name in REPAIRS:
        e = getattr(sr, name)(e)
    try:
        e = sr.cut_tunnels(e, max_mm=tunnel_max_mm, strict=True)
    except TunnelError as exc:
        raise UacError(str(exc)) from None
    return e


def compute(points: np.ndarray, triangles: np.ndarray, tags: np.ndarray,
            options: Optional[Options] = None, *,
            on_status: Optional[Callable[[str], None]] = None,
            should_stop: Optional[Callable[[], bool]] = None) -> UacRun:
    """Compute UAC on a labelled atrial surface. Raises :class:`UacError`.

    The surface is repaired first (:func:`prepare`); the run's ``surface``
    is that repaired surface, and the coordinates are on its points."""
    return run_atrium(LEFT, points, triangles, tags, options, on_status=on_status,
                      should_stop=should_stop)


def run_atrium(atrium: Atrium, points: np.ndarray, triangles: np.ndarray, tags: np.ndarray,
               options: Optional[Options] = None, *,
               on_status: Optional[Callable[[str], None]] = None,
               should_stop: Optional[Callable[[], bool]] = None) -> UacRun:
    """The run of :func:`compute` for any atrium. The coordinates come out
    in the working frame (``alpha = u``, ``beta = v``, see
    :mod:`ccdaf.core.uac.layout`)."""
    t0 = time.time()
    opts = options or Options()
    solver = SpdSolver(opts.solver)

    _status(on_status, "Repairing the surface…")
    surf = prepare(points, triangles, tags, opts.tunnel_max_mm)
    P, t, tag = surf.P, surf.t, surf.tag
    n0 = len(P)
    try:
        atrium.check_sides(P, t, tag)
        rings0 = label_rings(P, t, tag, atrium.vessels, atrium.names)
    except CutError as exc:
        raise UacError(str(exc)) from None
    _status(on_status, "Extending short veins…")
    n_tri = len(t)
    P, t, tag, n_added, ext = extend_veins(P, t, tag, rings0, atrium.sleeve_ratio)
    is_real = np.r_[np.ones(n_tri, bool), np.zeros(len(t) - n_tri, bool)]
    try:
        anat = atrium.anatomy(P, t, tag)
    except CutError as exc:
        raise UacError(str(exc)) from None

    best, first_error = None, None
    for penalty in ((opts.ring_penalty, 0.0) if opts.ring_penalty > 0 else (0.0,)):
        try:
            att = _attempt(P, t, tag, is_real, anat, opts, penalty, solver, on_status, should_stop,
                           atrium)
        except Cancelled:
            raise UacError("Stopped.") from None
        except CutError as exc:
            first_error = first_error or str(exc)
            continue
        if best is None or (att.folded_real, att.worst_miss) < (best.folded_real, best.worst_miss):
            best = att
        if best.worst_miss <= RETRY_MISS:
            break
        _status(on_status, "A ring is far from its template target: trying again without "
                           "the ring penalty…")
    if best is None:
        raise UacError(first_error or "No valid cut.")

    _match_cut(best)
    alpha, beta = _on_points(best, n0, surf.t)
    # count folds on the map the stored values draw, so none is missed
    for h, k in zip(sorted(best.halves, key=lambda h: h.side), stored_folds(alpha, beta, surf.t)):
        h.folded_real = k
    warnings = []
    if best.folded_real:
        warnings.append(f"{best.folded_real} triangles are folded in the flat map.")
    if best.worst_miss > RETRY_MISS:
        large = [f"{atrium.names[L]} ({v['ostium_mm']:.0f} mm)" for L, v in ext.items()
                 if isinstance(v, dict) and v.get("ostium_mm", 0) > LARGE_OSTIUM_MM]
        hint = (" Very large ostia, whose labels probably run onto the body: "
                + ", ".join(large) + "." if large else "")
        warnings.append(f"A ring is {best.worst_miss:.3f} from its template target: check the "
                        f"vein and appendage labels.{hint}")
    return UacRun(alpha, beta, surf, solver.name, time.time() - t0, best, ext, warnings, atrium)


def stored_folds(alpha: np.ndarray, beta: np.ndarray, t: np.ndarray) -> Tuple[int, int]:
    """Folded triangles of the flat map drawn from the stored values
    (:func:`flat_from_values`), per half ``(posterior, anterior)``: a
    triangle whose signed area has the sign opposite to most triangles'.
    A triangle collapsed to a point (longest flat side under
    :data:`~ccdaf.core.uac.flatten.COLLAPSED`) is not counted."""
    uv, tri, cells, _ = flat_from_values(alpha, beta, t)
    sa = signed_areas(uv, tri)
    e = np.stack([np.linalg.norm(uv[tri[:, k]] - uv[tri[:, (k + 1) % 3]], axis=1)
                  for k in range(3)], 1).max(1)
    ok = e >= COLLAPSED
    if not ok.any():
        return 0, 0
    folded = ok & (np.sign(sa) == -np.sign(np.median(sa[ok])))
    ant = (np.asarray(beta, float)[np.asarray(t)[cells]] > 0.5 + 1e-9).any(1)
    return int((folded & ~ant).sum()), int((folded & ant).sum())


def _match_cut(att: Attempt) -> None:
    """Put each cut point's anterior copy exactly where the stored values
    say: ``(alpha, 1 - beta)`` on the side lines, the same value on the roof.

    The two halves reach the cut to about 1e-6 (a ring point is pulled by a
    stiff spring, not pinned), so this moves points by at most that much,
    against triangle sides of about 1e-3. It makes the flat map drawn from
    the stored arrays (:func:`flat_from_values`) the run's own map exactly.
    """
    post = next(h for h in att.halves if h.side == POSTERIOR)
    ant = next(h for h in att.halves if h.side == ANTERIOR)
    pos = {int(v): i for i, v in enumerate(post.nodes)}
    for j, v in enumerate(ant.nodes):
        i = pos.get(int(v))
        if i is not None:
            a, b = post.uv[i]
            ant.uv[j] = (a, 1.0 - b if b < 0.5 else b)


def _on_points(att: Attempt, n0: int, t: np.ndarray):
    """Values on the repaired surface's points (the first ``n0`` of the
    working mesh; the rest are the extensions'). The posterior half wins on
    the cut."""
    val = np.full((n0, 2), np.nan)
    for h in sorted(att.halves, key=lambda h: -h.side):     # anterior first, posterior overwrites
        ok = h.nodes < n0
        val[h.nodes[ok]] = h.uv[ok]
    missing = np.isnan(val[:, 0])
    if missing.any():                # a point no half reached: average its neighbours
        ue, _, _ = unique_edges(t)
        A = sp.coo_matrix((np.ones(2 * len(ue)), (np.r_[ue[:, 0], ue[:, 1]], np.r_[ue[:, 1], ue[:, 0]])),
                          shape=(n0, n0)).tocsr()
        for _ in range(50):
            known = ~np.isnan(val[:, 0])
            if known.all():
                break
            s = A @ np.where(known[:, None], val, 0.0)
            c = A @ known.astype(float)
            fill = ~known & (c > 0)
            if not fill.any():
                break
            val[fill] = s[fill] / c[fill, None]
    return val[:, 0], val[:, 1]


def flat_from_values(alpha: np.ndarray, beta: np.ndarray, t: np.ndarray, eps: float = 1e-9):
    """The flat map, from the coordinates stored on the atrium.

    A triangle with a point above ``beta = 0.5`` is on the anterior half,
    every other on the posterior half. A point on the cut stores its
    posterior value; its anterior copy is ``(alpha, 1 - beta)`` on the
    lateral and septal lines and the same value on the roof, which is how
    the two halves are laid on the square. Returns ``points (k, 2),
    triangles, triangle index on the atrium, and the atrium point of each
    flat point``.
    """
    alpha = np.asarray(alpha, float)
    beta = np.asarray(beta, float)
    t = np.asarray(t, np.int64)
    ant = (beta[t] > 0.5 + eps).any(1)
    pts, tris, cells, src = [], [], [], []
    off = 0
    for side in (False, True):
        rows = np.where(ant == side)[0]
        used, loc = np.unique(t[rows], return_inverse=True)
        b = beta[used]
        if side:
            b = np.where(b < 0.5, 1.0 - b, b)
        pts.append(np.c_[alpha[used], b])
        tris.append(loc.reshape(-1, 3) + off)
        cells.append(rows)
        src.append(used)
        off += len(used)
    return np.vstack(pts), np.vstack(tris), np.concatenate(cells), np.concatenate(src)


__all__ = ["ALPHA", "BETA", "Options", "UacRun", "UacError", "compute", "prepare", "Atrium",
           "LEFT", "run_atrium", "stored_folds",
           "flat_from_values", "repair_words", "REPAIRS"]
