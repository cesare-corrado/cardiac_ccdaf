"""
cut_ra
======
Open the right atrium into a lateral and a septal half.

The cut follows Roney et al. (2019): a roof line between the venae cavae
splits the lateral from the septal wall, and two lines run from the cavae
down to the tricuspid valve (TV):

    TV -> P1 (SVC) -> part of the SVC ring -> P2 -> roof -> P4
       -> part of the IVC ring -> P3 (IVC) -> TV

The appendage must end on the lateral half; the CS and the sleeves of both
cavae on the septal half. Each line is a shortest path over body tissue:

* **Roof.** The shortest path between the SVC and IVC rings; its ends are
  P2 and P4. On the 16 published right atria this path lies on the
  published roof line (``alpha`` near 0.5).
* **Side lines.** From a point of each cava's ring to the valve. The paper
  takes the shortest such path; here the start is chosen among the ring
  points at least a quarter of the ring away from the roof's end (closest
  to the valve first), so the ring part on the cut is not a sliver.
* **Barriers.** The CS and the appendage each sit between a cava and the
  valve, so a side line can pass them on either side. A barrier from the CS
  ring to the IVC ring and to the valve (and the same for the appendage and
  the SVC) makes each line pass on one side; which side is then set by the
  start point and the ring part on the cut, and every combination is tried
  until the halves come out right. The barriers are dropped if no cut
  works with them.
* **Ring penalty** and **pockets**: as on the left atrium
  (:mod:`ccdaf.core.uac.cut`).

The cut is laid out like the left atrium's: the SVC line plays the left
atrium's lateral line, the IVC line its septal line, the lateral half its
posterior half (piece 0) and the septal half its anterior half (piece 1).
"""
from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import dijkstra

from ccdaf.core.uac import template_ra as tr
from ccdaf.core.uac.cut import Anatomy, Cut, CutError, build_anatomy, label_rings, split_by_cut, tidy_cut
from ccdaf.core.uac.mesh_ops import NoPath, arcs, edge_table, graph, neighbours, shortest

LABELS = tr.VESSELS + (tr.RAA,)


def anatomy(P: np.ndarray, t: np.ndarray, tag: np.ndarray) -> Anatomy:
    missing = [tr.NAMES[L] for L in (tr.BODY,) + LABELS if not (tag == L).any()]
    if missing:
        raise CutError("Labels missing: " + ", ".join(missing) + ". The right atrium needs "
                       "body 1, SVC 21, IVC 23, CS 25 and RAA 27.")
    return build_anatomy(P, t, tag, LABELS, body_label=tr.BODY, names=tr.NAMES,
                         valve="tricuspid valve")


#: The IVC ostium is at least this many times as far from the appendage as
#: the SVC ostium. On the 14 distinct published right atria the smallest
#: ratio was 1.27; swapped cavae give the inverse.
SIDE_RATIO = 1.0


def check_sides(P: np.ndarray, t: np.ndarray, tag: np.ndarray) -> None:
    """Refuse the venae cavae labelled the wrong way round.

    The appendage lies next to the SVC: its ostium centre is nearer the SVC
    ostium centre than the IVC's (by a factor of 1.27 to 3.4 on the
    published atria). Swapped cavae would put the appendage next to the IVC
    on the map, and the warp would drag it across the square."""
    if not all((tag == L).any() for L in (tr.SVC, tr.IVC, tr.RAA)):
        return                                        # anatomy() names what is missing
    rings = label_rings(P, t, tag, (tr.SVC, tr.IVC, tr.RAA), tr.NAMES)
    c = {L: P[r].mean(0) for L, r in rings.items()}
    ds = float(np.linalg.norm(c[tr.SVC] - c[tr.RAA]))
    di = float(np.linalg.norm(c[tr.IVC] - c[tr.RAA]))
    if di < SIDE_RATIO * ds:
        raise CutError("The venae cavae look swapped: the appendage lies next to the SVC, but "
                       f"here it is nearer the IVC (SVC {ds:.0f} mm, IVC {di:.0f} mm from the "
                       "appendage). Check the labels (SVC 21, IVC 23).")


def _path_to_set(G, P, src_nodes, targets):
    """Shortest path in graph ``G`` from any of ``src_nodes`` to any of
    ``targets``, as a node list (source first), or None."""
    d, pred, src = dijkstra(G, indices=src_nodes, min_only=True, return_predecessors=True)
    targets = np.asarray(targets)
    tgt = int(targets[np.argmin(d[targets])])
    if not np.isfinite(d[tgt]):
        return None
    pth = [tgt]
    while pth[-1] != src[tgt]:
        pth.append(int(pred[pth[-1]]))
    return np.array(pth[::-1])


def _starts(ring: np.ndarray, end: int, P: np.ndarray, dTV: np.ndarray, k: int = 3):
    """Up to ``k`` start points on ``ring`` for a side line: ring points at
    least a quarter of the ring (by length) from ``end``, one per sector,
    closest to the valve first."""
    rr = np.roll(ring, -int(np.where(ring == end)[0][0]))
    seg = np.linalg.norm(np.diff(P[np.r_[rr, rr[:1]]], axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)[:-1]] / seg.sum()
    out = []
    for lo, hi in ((0.25, 0.42), (0.42, 0.58), (0.58, 0.75)):
        m = (s >= lo) & (s < hi)
        if m.any():
            cand = rr[m]
            out.append(int(cand[np.argmin(dTV[cand])]))
    return sorted(set(out), key=lambda v: dTV[v])[:k]


def find_cut(P: np.ndarray, t: np.ndarray, tag: np.ndarray, anat: Anatomy, *,
             roof_first: bool = False, ring_penalty: float = 4.0,
             penalty_mm: float = 2.0) -> Cut:
    """The cut, or :class:`CutError` saying what is wrong. ``roof_first`` is
    accepted for the left atrium's interface; the roof here is always laid
    first."""
    n = len(P)
    ring, TV = anat.ring, anat.mv
    edgeset = set(map(tuple, np.unique(edge_table(t)[0], axis=0).tolist()))
    nbrs = neighbours(t)
    body = anat.body
    body_or_ring = body & ~anat.on_boundary
    G0 = graph(P, t)
    Gb = graph(P, t, body)
    dTV = dijkstra(Gb, indices=TV, min_only=True)
    hw = None
    if ring_penalty > 0:
        dR = dijkstra(G0, indices=np.unique(np.concatenate(list(ring.values()))), min_only=True)
        hw = 1.0 + ring_penalty * np.exp(-dR / penalty_mm)

    def path(allowed, src, targets):
        if hw is None:
            return shortest(P, t, allowed, src, targets)
        w = np.nan_to_num(hw * np.ones(n), nan=1e6, posinf=1e6)
        return shortest(P, t, allowed, src, targets, nodeweight=w)

    # roof: the shortest body path between the cava rings
    rp = _path_to_set(Gb, P, ring[tr.SVC], ring[tr.IVC])
    if rp is None:
        raise CutError("No body tissue joins the SVC and IVC ostia, so the roof line has "
                       "nowhere to run.")
    p2, p4 = int(rp[0]), int(rp[-1])

    # barriers: vessel -> cava ring and vessel -> valve, over body tissue
    def barrier_for(L, cava):
        b = np.zeros(n, bool)
        for tgt in (ring[cava], TV):
            pth = _path_to_set(Gb, P, ring[L], tgt)
            if pth is not None:
                b[pth] = True
        b[ring[L]] = False
        return b

    bar_ivc = barrier_for(tr.CS, tr.IVC)
    bar_svc = barrier_for(tr.RAA, tr.SVC)

    def build(p1, p3, half_s, half_i, barrier):
        allowed = body.copy()
        allowed[list(set(half_s.tolist()) | set(half_i.tolist()))] = False
        allowed[[p2, p4]] = True
        roof = path(allowed, p2, [p4])
        used = set(half_s.tolist()) | set(half_i.tolist()) | set(roof.tolist())
        allowed = body.copy()
        allowed[list(used)] = False
        allowed &= ~barrier
        allowed[p1] = True
        allowed[TV] = True
        allowed &= ~barrier
        sl = path(allowed, p1, TV)
        used |= set(sl.tolist())
        allowed = body.copy()
        allowed[list(used)] = False
        allowed &= ~barrier
        allowed[p3] = True
        allowed[TV] = True
        allowed[sl[-1]] = False
        allowed[barrier] = False
        il = path(allowed, p3, TV)
        segs = [sl[::-1], half_s, roof, half_i[::-1], il]
        parts = [segs[0]]
        for s in segs[1:]:
            if s[0] != parts[-1][-1]:
                raise CutError("cut segments do not join")
            parts.append(s[1:])
        cut = tidy_cut(np.concatenate(parts), edgeset, nbrs, body_or_ring)
        idx = {int(v): i for i, v in enumerate(cut)}

        def pos(v):
            return idx[v] if v in idx else int(np.argmin(np.linalg.norm(P[cut] - P[v], axis=1)))

        corners = {"P1": pos(p1), "P2": pos(p2), "P4": pos(p4), "P3": pos(p3)}
        ncomp, comp = split_by_cut(t, n, cut)
        if ncomp < 2:
            raise CutError("the cut does not split the atrium")
        sizes = np.bincount(comp)
        big = np.argsort(sizes)[::-1][:2]
        raa0 = (tag[comp == big[0]] == tr.RAA).sum()
        raa1 = (tag[comp == big[1]] == tr.RAA).sum()
        lat_c, sep_c = (big[0], big[1]) if raa0 > raa1 else (big[1], big[0])
        piece = np.full(len(t), -1)
        piece[comp == lat_c] = 0
        piece[comp == sep_c] = 1
        return cut, piece, corners, int(np.isin(comp, big, invert=True).sum())

    starts_s = _starts(ring[tr.SVC], p2, P, dTV)
    starts_i = _starts(ring[tr.IVC], p4, P, dTV)
    attempts = []
    for use_barrier in (True, False):
        barrier = (bar_ivc | bar_svc) if use_barrier else np.zeros(n, bool)
        if use_barrier:
            barrier[[p2, p4]] = False
        for p1 in starts_s:
            for p3 in starts_i:
                for hs in arcs(ring[tr.SVC], p1, p2):
                    for hi in arcs(ring[tr.IVC], p3, p4):
                        hs_ = hs if hs[0] == p1 else hs[::-1]
                        hi_ = hi if hi[0] == p3 else hi[::-1]
                        b = barrier.copy()
                        b[[p1, p3]] = False
                        try:
                            cut, piece, corners, stray = build(p1, p3, hs_, hi_, b)
                        except (NoPath, CutError) as exc:
                            attempts.append({"error": str(exc)})
                            continue
                        where = {L: int(np.bincount(piece[tag == L] + 1, minlength=3).argmax()) - 1
                                 for L in LABELS}
                        ok = (where[tr.SVC] == where[tr.IVC] == where[tr.CS] == 1
                              and where[tr.RAA] == 0)
                        for pc in (0, 1):
                            nodes_pc = np.unique(t[piece == pc])
                            ok = ok and bool(np.isin([cut[0], cut[-1]], nodes_pc).all())
                        attempts.append({"where": where})
                        if ok:
                            return Cut(cut, piece, corners, "shortest", use_barrier, stray)
    raise CutError(diagnose(attempts))


def diagnose(attempts) -> str:
    """Say, in terms of the labels, why no cut worked."""
    split = [a["where"] for a in attempts if "where" in a]
    if split:
        def score(w):
            return ((w[tr.SVC] == 1) + (w[tr.IVC] == 1) + (w[tr.CS] == 1) + (w[tr.RAA] == 0))
        best = max(split, key=score)
        wrong = [tr.NAMES[L] for L, s in ((tr.SVC, 1), (tr.IVC, 1), (tr.CS, 1), (tr.RAA, 0))
                 if best[L] != s]
        return ("No valid cut. No cut puts the appendage on the lateral half and the CS and "
                "both cavae on the septal half; the closest attempt misplaced "
                f"{', '.join(wrong)}. Check the CS and appendage labels.")
    return ("No valid cut. No path for the cut lines over body tissue; check that body tissue "
            "surrounds both cavae, the CS and the appendage.")


__all__ = ["anatomy", "check_sides", "find_cut", "diagnose", "LABELS", "SIDE_RATIO"]
