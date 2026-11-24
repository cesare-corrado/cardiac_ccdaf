"""
cut
===
Open the atrium into a posterior and an anterior half.

The cut runs from the mitral valve up the lateral wall to the LSPV ostium,
round the posterior half of that ostium, over the roof to the RSPV ostium,
round its posterior half, and down the septal wall back to the valve:

    MV -> P1 (LSPV, facing the LIPV) -> half ring -> P2 -> roof -> P4
       -> half ring -> P3 (RSPV, facing the RIPV) -> MV

Both inferior veins must end on the posterior half; both superior veins
and the appendage on the anterior half. Each line is a shortest path over
body tissue, with these safeguards, all validated on 150 atria:

* **Barrier.** A line from the LIPV to the RIPV ring, and from its middle
  down to the valve, that the side lines may not cross; otherwise a side
  line can slip behind an inferior vein and carry it to the anterior half.
  Tried without it if no cut works with it.
* **Ring penalty.** Paths cost more within a few millimetres of any ring,
  so no line runs alongside a ring; tissue squeezed between a line and a
  ring cannot be laid flat.
* **Lateral rule.** The lateral line follows the line equidistant from the
  LIPV and the appendage (it must pass between them); three simpler rules
  are tried if it fails.
* **Pockets.** Where the cut nearly touches itself, the stretch between is
  replaced by the direct edge, so no strip of tissue is enclosed by cut
  points only.
* **Side check.** The lateral line must run nearer the LIPV than the RIPV,
  the septal line the other way round, so a mirrored cut is rejected.

The half of each superior ostium that goes on the cut is chosen by trying
both. When no combination works, :func:`diagnose` says why in terms of the
labels.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from ccdaf.core.uac.mesh_ops import (
    NoPath, arcs, edge_table, graph, neighbours, order_loops, shortest,
    unique_edges,
)
from ccdaf.core.uac.template import BODY, LAA, LIPV, LSPV, NAMES, RIPV, RSPV, VEINS


class CutError(RuntimeError):
    """No valid cut: the message says what is wrong with the labels."""


# ---------------------------------------------------------------- anatomy
@dataclass
class Anatomy:
    """Rings (label borders), the valve loop and per-point label sets."""

    ring: Dict[int, np.ndarray]
    mv: np.ndarray
    node_tags: Dict[int, Set[int]]
    body: np.ndarray                 # point touches the body
    on_boundary: np.ndarray          # point on an open boundary
    notes: List[str] = field(default_factory=list)


def label_rings(P: np.ndarray, t: np.ndarray, tag: np.ndarray,
                labels=VEINS + (LAA,)) -> Dict[int, np.ndarray]:
    """The border loop of each label (the longest, if there are several)."""
    n = len(P)
    e, own = edge_table(t)
    key = e[:, 0] * n + e[:, 1]
    order = np.argsort(key, kind="stable")
    ks = key[order]
    dup = np.where(ks[1:] == ks[:-1])[0]
    ea, ta, tb = e[order][dup], tag[own[order][dup]], tag[own[order][dup + 1]]
    out = {}
    for L in labels:
        loops = order_loops(ea[(ta == L) != (tb == L)])
        if not loops:
            raise CutError(f"The {NAMES[L]} label has no border with the rest of the atrium.")
        out[L] = max(loops, key=len)
    return out


def check_sides(P: np.ndarray, t: np.ndarray, tag: np.ndarray) -> None:
    """Refuse veins labelled on the wrong side.

    The LAA lies next to the left veins, so each left vein is nearer the
    LAA than the right vein of the same level (superior, inferior). On 226
    labelled atria the right vein was always at least 1.18 times as far
    from the LAA as the farthest left vein; a swapped pair turns this
    round. Left and right swapped would put the LAA next to the right veins
    on the map, which no later step can notice."""
    if not all((tag == L).any() for L in VEINS + (LAA,)):
        return                                        # anatomy() names what is missing
    c = P[t].mean(1)
    laa = c[tag == LAA].mean(0)
    d = {L: float(np.linalg.norm(c[tag == L].mean(0) - laa)) for L in VEINS}
    wrong = [(a, b) for a, b in ((LSPV, RSPV), (LIPV, RIPV)) if d[a] > d[b]]
    if wrong:
        pairs = "; ".join(f"{NAMES[a]} {d[a]:.0f} mm, {NAMES[b]} {d[b]:.0f} mm" for a, b in wrong)
        raise CutError("Left and right veins look swapped: the LAA lies next to the left "
                       f"veins, but here it is nearer the right one ({pairs} from the LAA). "
                       "Check the vein labels (left 11/13, right 15/17).")


def anatomy(P: np.ndarray, t: np.ndarray, tag: np.ndarray) -> Anatomy:
    missing = [NAMES[L] for L in (BODY,) + VEINS + (LAA,) if not (tag == L).any()]
    if missing:
        raise CutError("Labels missing: " + ", ".join(missing) + ". The atrium needs "
                       "body 1, LSPV 11, LIPV 13, RSPV 15, RIPV 17 and LAA 19.")
    n = len(P)
    node_tags: Dict[int, Set[int]] = defaultdict(set)
    for k in range(3):
        for v, g in zip(t[:, k], tag):
            node_tags[int(v)].add(int(g))
    body = np.zeros(n, bool)
    for v, s in node_tags.items():
        body[v] = BODY in s
    ue, inv, cnt = unique_edges(t)
    bnd_e = ue[cnt == 1]
    on_boundary = np.zeros(n, bool)
    on_boundary[bnd_e.ravel()] = True
    # the valve: the longest open boundary next to body triangles
    own = np.tile(np.arange(len(t)), 3)
    side_is_bnd = cnt[inv] == 1
    mv_loops = []
    for lp in order_loops(bnd_e):
        sel = side_is_bnd & np.isin(edge_table(t)[0], lp).all(1)
        if sel.any() and np.bincount(tag[own[sel]]).argmax() == BODY:
            mv_loops.append(lp)
    if not mv_loops:
        raise CutError("No mitral valve opening: the atrium needs an open boundary "
                       "next to body tissue.")
    ring = label_rings(P, t, tag)
    return Anatomy(ring, max(mv_loops, key=len), node_tags, body, on_boundary)


# ---------------------------------------------------------------- path helpers
def shortcut(path: np.ndarray, edgeset) -> np.ndarray:
    """Drop a point when its two neighbours on the path share an edge, so
    no triangle has all three corners on the cut."""
    p = list(path)
    changed = True
    while changed:
        changed = False
        k = 0
        while k + 2 < len(p):
            a, c = p[k], p[k + 2]
            if (min(a, c), max(a, c)) in edgeset:
                del p[k + 1]
                changed = True
            else:
                k += 1
    return np.array(p)


def close_pockets(path: np.ndarray, nbrs, ok_node: np.ndarray, window: int = 30) -> np.ndarray:
    """Where two path points up to ``window`` steps apart are neighbours, or
    share a neighbour off the path, replace the stretch between them by the
    direct edge (or by that neighbour)."""
    p = list(path)
    changed, guard = True, 0
    while changed and guard < 50:
        changed, guard = False, guard + 1
        on = set(p)
        i = 0
        while i < len(p) - 3:
            best = None
            for j in range(min(len(p) - 1, i + window), i + 2, -1):
                if p[j] in nbrs[p[i]]:
                    best = (j, [])
                    break
                if j >= i + 4:
                    com = [x for x in (nbrs[p[i]] & nbrs[p[j]]) if x not in on and ok_node[x]]
                    if com:
                        best = (j, [com[0]])
                        break
            if best:
                j, mid = best
                p = p[: i + 1] + mid + p[j:]
                on = set(p)
                changed = True
            i += 1
    return np.array(p)


# ---------------------------------------------------------------- the cut
@dataclass
class Cut:
    nodes: np.ndarray            # MV (lateral) ... P1 ... P2 ... P4 ... P3 ... MV (septal)
    piece: np.ndarray            # per triangle: 0 posterior, 1 anterior, -1 neither
    corners: Dict[str, int]      # positions of P1, P2, P4, P3 in ``nodes``
    rule: str
    barrier: bool
    stray: int


LATERAL_RULES = ("border", "voronoi", "via_q", "plain")


def find_cut(P: np.ndarray, t: np.ndarray, tag: np.ndarray, anat: Anatomy, *,
             roof_first: bool = False, ring_penalty: float = 4.0,
             penalty_mm: float = 2.0) -> Cut:
    """The cut, or :class:`CutError` with a diagnosis."""
    n = len(P)
    ring, MV = anat.ring, anat.mv
    edgeset = set(map(tuple, np.unique(edge_table(t)[0], axis=0).tolist()))
    nbrs = neighbours(t)
    body_only = anat.body
    body_or_ring = body_only & ~anat.on_boundary
    cen = {L: P[r].mean(0) for L, r in ring.items()}
    ant_dir = cen[LAA] - np.mean([cen[L] for L in VEINS], axis=0)
    ant_dir /= np.linalg.norm(ant_dir)

    G0 = graph(P, t)
    dLIPV = dijkstra(G0, indices=ring[LIPV], min_only=True)
    dRIPV = dijkstra(G0, indices=ring[RIPV], min_only=True)
    dLAA = dijkstra(G0, indices=ring[LAA], min_only=True)
    hw = None
    if ring_penalty > 0:
        dR = dijkstra(G0, indices=np.unique(np.concatenate(list(ring.values()))), min_only=True)
        hw = 1.0 + ring_penalty * np.exp(-dR / penalty_mm)

    def path(allowed, src, targets, extra=None):
        if hw is None and extra is None:
            return shortest(P, t, allowed, src, targets)
        w = (hw if hw is not None else 1.0) * (extra if extra is not None else 1.0)
        w = np.nan_to_num(w * np.ones(n), nan=1e6, posinf=1e6)
        return shortest(P, t, allowed, src, targets, nodeweight=w)

    roof_end = None
    if roof_first:
        d, pred, src = dijkstra(graph(P, t, body_only), indices=ring[RSPV], min_only=True,
                                return_predecessors=True)
        eL = ring[LSPV][np.argmin(d[ring[LSPV]])]
        roof_end = {LSPV: int(eL), RSPV: int(src[eL])}

    def sup_halves(sup, inf, far_from):
        rs, ri = ring[sup], ring[inf]
        d, _ = cKDTree(P[ri]).query(P[rs])
        p1 = int(rs[np.argmin(d)])
        shared = [v for v in rs if v in set(ri.tolist()) and BODY in anat.node_tags[int(v)]]
        if shared:   # the two veins share an ostium: take the end facing away
            fc = np.mean([cen[L] for L in far_from], axis=0)
            p1 = int(max(shared, key=lambda v: np.linalg.norm(P[v] - fc)))
        p2 = int(rs[np.argmax(np.linalg.norm(P[rs] - P[p1], axis=1))])
        if roof_end is not None:
            p2 = roof_end[sup]
        a1, a2 = arcs(rs, p1, p2)
        post, ant = (a1, a2) if (P[a1] @ ant_dir).mean() < (P[a2] @ ant_dir).mean() else (a2, a1)
        return p1, p2, post, ant

    p1, p2, halfL, antL = sup_halves(LSPV, LIPV, (RSPV, RIPV))
    p3, p4, halfR, antR = sup_halves(RSPV, RIPV, (LSPV, LIPV))

    barrier_full = np.zeros(n, bool)
    try:
        d, pred, src = dijkstra(graph(P, t, body_only), indices=ring[LIPV], min_only=True,
                                return_predecessors=True)
        tgt = int(ring[RIPV][np.argmin(d[ring[RIPV]])])
        if np.isfinite(d[tgt]):
            pth = [tgt]
            while pth[-1] != src[tgt]:
                pth.append(int(pred[pth[-1]]))
            down = shortest(P, t, body_only, pth[len(pth) // 2], MV)
            barrier_full[pth] = True
            barrier_full[down] = True
            barrier_full[[p1, p3]] = False
    except NoPath:
        pass
    q = int(ring[LAA][np.argmin(cKDTree(P[ring[LIPV]]).query(P[ring[LAA]])[0])])

    def build(halfL, halfR, rule, barrier):
        allowed = body_only.copy()
        allowed[list(set(halfL.tolist()) | set(halfR.tolist()))] = False
        allowed[[p2, p4]] = True
        roof = path(allowed, p2, [p4])
        used = set(halfL.tolist()) | set(halfR.tolist()) | set(roof.tolist())
        allowed = body_only.copy()
        allowed[list(used)] = False
        allowed &= ~barrier
        if rule == "border":
            allowed[p1] = True
            allowed[MV] = True
            allowed &= ~barrier
            with np.errstate(invalid="ignore"):
                nw = 1.0 + np.abs(dLIPV - dLAA) / 2.0
            lat = path(allowed, p1, MV, nw)
        elif rule == "voronoi":
            allowed &= dLIPV <= dLAA
            allowed[p1] = True
            allowed[MV] = True
            allowed &= ~barrier
            lat = path(allowed, p1, MV)
        elif rule == "via_q":
            a1 = allowed.copy()
            a1[[p1, q]] = True
            la1 = path(a1, p1, [q])
            a2 = allowed.copy()
            a2[list(set(la1[:-1].tolist()))] = False
            a2[ring[LAA]] = False
            a2[q] = True
            a2[MV] = True
            a2 &= ~barrier
            la2 = path(a2, q, MV)
            lat = np.concatenate([la1, la2[1:]])
        else:
            allowed[p1] = True
            allowed[MV] = True
            allowed &= ~barrier
            lat = path(allowed, p1, MV)
        used |= set(lat.tolist())
        allowed = body_only.copy()
        allowed[list(used)] = False
        allowed &= ~barrier
        allowed[p3] = True
        allowed[MV] = True
        allowed[lat[-1]] = False
        allowed[barrier] = False
        sep = path(allowed, p3, MV)

        segs = [lat[::-1], halfL, roof, halfR[::-1], sep]
        parts = [segs[0]]
        for s in segs[1:]:
            if s[0] != parts[-1][-1]:
                raise CutError("cut segments do not join")
            parts.append(s[1:])
        cut = np.concatenate(parts)
        if len(set(cut.tolist())) != len(cut):      # loop erasure
            out_, pos_ = [], {}
            for v in cut:
                if v in pos_:
                    k_ = pos_[v]
                    for w in out_[k_ + 1:]:
                        pos_.pop(w, None)
                    out_ = out_[: k_ + 1]
                else:
                    pos_[v] = len(out_)
                    out_.append(v)
            cut = np.array(out_)
        cut = shortcut(close_pockets(shortcut(cut, edgeset), nbrs, body_or_ring), edgeset)
        idx = {int(v): i for i, v in enumerate(cut)}

        def pos(v):
            return idx[v] if v in idx else int(np.argmin(np.linalg.norm(P[cut] - P[v], axis=1)))

        corners = {"P1": pos(p1), "P2": pos(p2), "P4": pos(p4), "P3": pos(p3)}
        # split the triangles across the cut
        cut_edges = set((min(a, b), max(a, b)) for a, b in zip(cut[:-1].tolist(), cut[1:].tolist()))
        e_all, own = edge_table(t)
        keep = np.array([(a, b) not in cut_edges for a, b in e_all.tolist()])
        key = e_all[keep, 0] * n + e_all[keep, 1]
        o = np.argsort(key, kind="stable")
        ks = key[o]
        d_ = np.where(ks[1:] == ks[:-1])[0]
        ta, tb = own[keep][o][d_], own[keep][o][d_ + 1]
        A = sp.csr_matrix((np.ones(len(ta)), (ta, tb)), shape=(len(t), len(t)))
        ncomp, comp = connected_components(A, directed=False)
        if ncomp < 2:
            raise CutError("the cut does not split the atrium")
        sizes = np.bincount(comp)
        big = np.argsort(sizes)[::-1][:2]
        inf0 = np.isin(tag[comp == big[0]], [LIPV, RIPV]).sum()
        inf1 = np.isin(tag[comp == big[1]], [LIPV, RIPV]).sum()
        post_c, ant_c = (big[0], big[1]) if inf0 > inf1 else (big[1], big[0])
        piece = np.full(len(t), -1)
        piece[comp == post_c] = 0
        piece[comp == ant_c] = 1
        return cut, piece, corners, int(np.isin(comp, big, invert=True).sum())

    attempts = []
    for use_barrier in (True, False):
        if not use_barrier and not barrier_full.any():
            break
        barrier = barrier_full if use_barrier else np.zeros(n, bool)
        for rule in LATERAL_RULES:
            for hL in (halfL, antL):
                for hR in (halfR, antR):
                    try:
                        cut, piece, corners, stray = build(hL, hR, rule, barrier)
                    except (NoPath, CutError) as exc:
                        attempts.append({"error": str(exc)})
                        continue
                    where = {L: int(np.bincount(piece[tag == L] + 1, minlength=3).argmax()) - 1
                             for L in VEINS + (LAA,)}
                    ok = (where[LSPV] == where[RSPV] == where[LAA] == 1
                          and where[LIPV] == where[RIPV] == 0)
                    lat_n, sep_n = cut[: corners["P1"] + 1], cut[corners["P3"]:]
                    sides_ok = (np.median(dLIPV[lat_n]) < np.median(dRIPV[lat_n])
                                and np.median(dRIPV[sep_n]) < np.median(dLIPV[sep_n]))
                    for pc in (0, 1):
                        nodes_pc = np.unique(t[piece == pc])
                        ok = ok and bool(np.isin([cut[0], cut[-1]], nodes_pc).all())
                    attempts.append({"where": where, "sides_ok": sides_ok})
                    if ok and sides_ok:
                        return Cut(cut, piece, corners, rule, use_barrier, stray)
    raise CutError(diagnose(P, t, tag, anat, attempts))


def diagnose(P, t, tag, anat: Anatomy, attempts) -> str:
    """Say, in terms of the labels, why no cut worked."""
    msgs = []
    left = set(anat.ring[LSPV].tolist()) | set(anat.ring[LIPV].tolist())
    right = set(anat.ring[RSPV].tolist()) | set(anat.ring[RIPV].tolist())
    if left & right:
        msgs.append("The left and right vein labels touch, so there is no body tissue "
                    "between them for the roof line.")
    else:
        allowed = anat.body.copy()
        try:
            shortest(P, t, allowed, int(anat.ring[LSPV][0]), anat.ring[RSPV])
        except NoPath:
            msgs.append("No body tissue joins the LSPV and RSPV ostia, so the roof line "
                        "has nowhere to run.")
    split = [a["where"] for a in attempts if "where" in a]
    if split:
        def score(w):
            return (w[LIPV] == 0) + (w[RIPV] == 0) + (w[LSPV] == 1) + (w[RSPV] == 1) + (w[LAA] == 1)
        best = max(split, key=score)
        wrong = [NAMES[L] for L, s in ((LIPV, 0), (RIPV, 0), (LSPV, 1), (RSPV, 1), (LAA, 1))
                 if best[L] != s]
        if wrong:
            msgs.append("No cut puts both inferior veins on the posterior half and both "
                        "superior veins and the appendage on the anterior half; the closest "
                        f"attempt misplaced {', '.join(wrong)}. The vein layout may be twisted, "
                        "or vein labels swapped.")
        elif not any(a.get("sides_ok") for a in attempts if "where" in a):
            msgs.append("Every cut that separates the veins runs its lateral line nearer the "
                        "right veins than the left: the atrium may be mirrored, or the left "
                        "and right labels swapped.")
    if not msgs:
        msgs.append("No path for the cut lines over body tissue; check that body tissue "
                    "surrounds every vein and the appendage.")
    return "No valid cut. " + " ".join(msgs)


__all__ = ["CutError", "Anatomy", "anatomy", "check_sides", "label_rings", "Cut", "find_cut", "diagnose",
           "shortcut", "close_pockets"]
