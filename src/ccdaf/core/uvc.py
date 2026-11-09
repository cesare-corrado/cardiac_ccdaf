"""
uvc
===
Universal ventricular coordinates on a labelled biventricular volume: the
engine behind Actions > Compute UVC.

The four coordinates
--------------------
* ``uvc_ven`` (nu): -1 in the LV, +1 in the RV, from the element tags. A
  node shared by LV and RV elements belongs to the LV.
* ``uvc_z``: apex to base, 0 to 1, computed in each ventricle from that
  ventricle's own apex, so it starts again at 0 in the RV.
* ``uvc_rho``: endocardium to epicardium, 0 to 1. In the LV the outer wall
  includes the septum, which faces the RV.
* ``uvc_phi``: rotation around the long axis, in radians: 0 mid-septum,
  +-pi/2.5 at the septal edges, +-pi/2 at the LV-RV junctions, +-pi on the
  LV free wall; in the RV, 0 mid free wall and +-pi/2 at the junctions.
  Positive is anterior.

Every coordinate is a ratio of two geodesic distances, ``g0 / (g0 + g1)``,
from the two boundaries that should carry its end values, or a linear
interpolation by that ratio. Distances come from :mod:`ccdaf.core.eikonal`.
This follows the distance-based construction of the reference UVC tool
the models this is compared against were built with, which was measured
to agree to about 0.005 in every coordinate. It is not the Laplace-based
construction of Bayer et al. (2018), whose apicobasal coordinate needs a
geodesic remapping to be usable.

Derived surfaces
----------------
From the surface labels and the LV/RV element tags:

* RV septum: the RV-endocardial faces of LV elements;
* RV free-wall endocardium: the RV-endocardial faces of RV elements;
* LV outer wall: the boundary of the LV elements, less the LV endocardium
  and the base. It includes the septum and the LV-RV interface;
* LV-RV interface: faces shared by an LV and an RV element.

Apices
------
Each apex is a short transmural line of nodes, from an endocardial node
to the epicardial node nearest it: for the LV, the LV-endocardial node
farthest from the centroid of the base; for the RV, the RV-endocardial
node farthest from its base rim along the RV endocardium. See
:func:`automatic_apices` for why, and how closely this follows the
reference tool. Either apex can be replaced by a picked one.

Rotational sectors
------------------
An axis frame is built from the apices: origin at the centroid of the LV
endocardium, long axis from the LV apex to that origin, LV-RV axis from
the LV apex towards the RV apex. Each element gets an angle around the
long axis; the septal edges are the angular extremes of the RV septum and
the junctions the extremes of the LV-RV interface. LV elements fall in six
sectors, RV elements in two. Each sector runs between two faces, the
interfaces with its neighbouring sectors, and phi is interpolated across it
by the distance ratio to those faces.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

from ccdaf.core import eikonal
from ccdaf.core.surface_labels import BASE, EPI, LABEL_MASK_FIELD, LV_ENDO, MASK_BITS, RV_ENDO
from ccdaf.core.volume_mesh import face_table, is_volume, tetrahedra

#: Point arrays written, one per coordinate.
UVC_FIELDS: Dict[str, str] = {"ven": "uvc_ven", "z": "uvc_z", "rho": "uvc_rho",
                              "phi": "uvc_phi"}

#: Element arrays tried, in order, for the LV/RV tags.
TAG_FIELDS: Tuple[str, ...] = ("elemTag", "tag", "Tag", "region")

#: Apex nodes lie within this many mean edge lengths of the apex line.
APEX_LINE_EDGES: float = 0.5

#: phi at the sector boundaries.
PI = np.pi
PHI_SEPTAL_EDGE: float = PI / 2.5
PHI_JUNCTION: float = PI / 2

# Sector numbers: LV 1-6, RV 11-12, as the reference tool numbers them.
SEPT_ANT, SEPT_POST, JUNC_ANT, JUNC_POST, FREE_ANT, FREE_POST = 3, 4, 5, 6, 1, 2
RV_ANT, RV_POST = 11, 12

#: Each sector: (face it starts from, phi there, face it ends at, phi there).
SECTORS: Dict[int, Tuple[str, float, str, float]] = {
    SEPT_ANT: ("lvsept", 0.0, "lvseptant", PHI_SEPTAL_EDGE),
    SEPT_POST: ("lvsept", 0.0, "lvseptpost", -PHI_SEPTAL_EDGE),
    JUNC_ANT: ("lvseptant", PHI_SEPTAL_EDGE, "lvjuncant", PHI_JUNCTION),
    JUNC_POST: ("lvseptpost", -PHI_SEPTAL_EDGE, "lvjuncpost", -PHI_JUNCTION),
    FREE_ANT: ("lvjuncant", PHI_JUNCTION, "lvfree", PI),
    FREE_POST: ("lvjuncpost", -PHI_JUNCTION, "lvfree", -PI),
    RV_ANT: ("rvfree", 0.0, "rvjuncant", PHI_JUNCTION),
    RV_POST: ("rvfree", 0.0, "rvjuncpost", -PHI_JUNCTION),
}

#: Which pair of sectors each face lies between.
FACES: Dict[str, Tuple[int, int]] = {
    "lvsept": (SEPT_ANT, SEPT_POST),
    "lvseptant": (SEPT_ANT, JUNC_ANT),
    "lvseptpost": (SEPT_POST, JUNC_POST),
    "lvjuncant": (JUNC_ANT, FREE_ANT),
    "lvjuncpost": (JUNC_POST, FREE_POST),
    "lvfree": (FREE_ANT, FREE_POST),
    "rvfree": (RV_ANT, RV_POST),
}
# rvjuncant / rvjuncpost are the LV-RV interface on each side; see _faces.


# ---------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------
def element_tags(dataset) -> Tuple[Optional[str], Optional[np.ndarray]]:
    """``(name, values)`` of the first integer element array found, or Nones."""
    for name in TAG_FIELDS:
        if name in dataset.cell_data:
            values = np.asarray(dataset.cell_data[name])
            if values.ndim == 1 and values.dtype.kind in "iu":
                return name, values
    return None, None


def lv_rv_tags(tets: np.ndarray, tags: np.ndarray, mask: np.ndarray) -> Tuple[int, int]:
    """``(lv, rv)``: which of exactly two tag values is which.

    The LV tag is the one whose elements touch more LV-endocardial nodes.
    """
    values = np.unique(tags)
    if len(values) != 2:
        raise ValueError(f"UVC need exactly two element tags, LV and RV; "
                         f"this mesh has {len(values)} ({', '.join(map(str, values[:6]))})")
    lv_nodes = (np.asarray(mask, dtype=np.int64) & MASK_BITS[LV_ENDO]) != 0
    touch = [int(np.sum(lv_nodes[np.unique(tets[tags == v])])) for v in values]
    if touch[0] == touch[1]:
        raise ValueError("cannot tell the LV tag from the RV tag: both touch "
                         "the LV endocardium equally")
    lv = int(values[int(np.argmax(touch))])
    return lv, int(values[values != lv][0])


def availability(dataset) -> Tuple[bool, str]:
    """Whether UVC can be computed on *dataset*, and if not, why."""
    if dataset is None:
        return False, "Load a tetrahedral volume first."
    if not is_volume(dataset):
        return False, "UVC need a tetrahedral volume, not a surface."
    if LABEL_MASK_FIELD not in dataset.point_data:
        return False, ("Label the surfaces first "
                       "(Actions > Label ventricular surfaces).")
    name, tags = element_tags(dataset)
    if tags is None:
        return False, ("UVC need LV and RV element tags; no integer element "
                       f"array named {' / '.join(TAG_FIELDS)} was found.")
    try:
        lv_rv_tags(tetrahedra(dataset), tags, dataset.point_data[LABEL_MASK_FIELD])
    except ValueError as exc:
        return False, str(exc)
    return True, ("Compute the four universal ventricular coordinates "
                  "(nu, z, rho, phi) from geodesic distances.")


# ---------------------------------------------------------------------
# Surfaces and apices
# ---------------------------------------------------------------------
@dataclass
class Surfaces:
    """Node sets of every boundary the coordinates are measured from."""

    base: np.ndarray
    epi: np.ndarray
    lv_endo: np.ndarray
    rv_endo: np.ndarray          # septum included
    rv_septum: np.ndarray
    rv_free_endo: np.ndarray     # septum excluded
    lv_outer: np.ndarray
    interface: np.ndarray        # LV-RV element interface
    lv_nodes: np.ndarray         # bool, node of any LV element
    lv_elems: np.ndarray         # bool, per element


def surfaces(tets: np.ndarray, mask: np.ndarray, is_lv: np.ndarray) -> Surfaces:
    """The derived surfaces, from the labels and the LV element flags."""
    tets = np.asarray(tets, dtype=np.int64)
    mask = np.asarray(mask, dtype=np.int64)
    n = len(mask)
    # One face table serves every surface: per unique face, how many
    # elements share it and how many of those are LV. Building it once
    # matters: on 35 million elements each table sorts 140 million faces.
    uniq, inv, counts, owner = face_table(tets)
    lv_count = np.bincount(inv, weights=is_lv[owner].astype(float),
                           minlength=len(uniq)).astype(np.int64)
    one_owner = np.empty(len(uniq), dtype=np.int64)
    one_owner[inv] = owner                       # exact where counts == 1
    del inv, owner

    boundary = counts == 1
    faces = uniq[boundary]
    face_lv = is_lv[one_owner[boundary]]
    fmask = mask[faces[:, 0]] & mask[faces[:, 1]] & mask[faces[:, 2]]

    # A surface is the nodes of its faces, a face belonging to a surface
    # when all three of its nodes do. Taking nodes by their own bits
    # instead adds every seam node that touches the surface without
    # bounding it: on a 637,000-node ventricle that is 5,400 extra
    # epicardial nodes, and the RV transmural coordinate moved by 0.005.
    def nodes_of(bit: int) -> np.ndarray:
        return np.unique(faces[(fmask & MASK_BITS[bit]) != 0])

    rv_face = (fmask & MASK_BITS[RV_ENDO]) != 0
    septum = np.unique(faces[rv_face & face_lv])
    rv_free = np.unique(faces[rv_face & ~face_lv])

    # The LV sub-mesh boundary: faces with exactly one LV element, on the
    # outside of the heart or against an RV element. Less the LV
    # endocardium and the base, it is the LV outer wall.
    lv_boundary = uniq[lv_count == 1]
    lm = mask[lv_boundary[:, 0]] & mask[lv_boundary[:, 1]] & mask[lv_boundary[:, 2]]
    keep = (lm & (MASK_BITS[LV_ENDO] | MASK_BITS[BASE])) == 0
    lv_outer = np.unique(lv_boundary[keep])

    # Faces shared by an LV and an RV element.
    interface = np.unique(uniq[(counts == 2) & (lv_count == 1)])
    del uniq, counts, lv_count, one_owner

    lv_nodes = np.zeros(n, dtype=bool)
    lv_nodes[tets[is_lv].ravel()] = True
    return Surfaces(base=np.where(mask & MASK_BITS[BASE])[0], epi=nodes_of(EPI),
                    lv_endo=nodes_of(LV_ENDO), rv_endo=nodes_of(RV_ENDO),
                    rv_septum=septum, rv_free_endo=rv_free, lv_outer=lv_outer,
                    interface=interface, lv_nodes=lv_nodes, lv_elems=is_lv)


def check_labels(points: np.ndarray, surf: Surfaces) -> None:
    """Refuse surfaces that cannot be what their labels say.

    The epicardium is the outside of the heart, so it must spread farther
    from the centre than either endocardium. A labelling that swapped the
    epicardium and the RV endocardium passes every other check and gives
    coordinates that look plausible and are wrong: three of 32 pipeline
    ventricles were built that way.
    """
    p = np.asarray(points, dtype=float)
    centre = p.mean(axis=0)
    spread = {name: float(np.sqrt(((p[nodes] - centre) ** 2).sum(axis=1).mean()))
              for name, nodes in (("epicardium", surf.epi), ("LV endocardium", surf.lv_endo),
                                  ("RV endocardium", surf.rv_endo)) if len(nodes)}
    inner = [name for name in ("LV endocardium", "RV endocardium")
             if spread.get(name, 0.0) >= spread.get("epicardium", np.inf)]
    if inner:
        raise ValueError(
            f"the {' and the '.join(inner)} spread{'s' if len(inner) == 1 else ''} farther "
            f"from the centre than the epicardium: the surface labels are probably "
            f"swapped. Relabel the surfaces before computing UVC.")


def transmural_sources(surf: "Surfaces") -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """``(sources, released)``: the four surfaces rho is measured from.

    A node on an endocardium and on the outer surface it is paired with is
    where the wall has no thickness: a pinch, or the rim of a hole through
    the wall. Both distances would be 0 there and rho an arbitrary 0.5, so
    such a node is held by neither surface and takes its value from the
    tissue around it, as the fibre solves treat the same nodes. *released*
    lists them.

    Only a node of the ventricle whose rho it would carry is released: an
    LV node for the LV pair, an RV node for the RV pair. Where the RV free
    wall, the septum and the epicardium meet at a junction, the nodes are
    LV nodes, take the LV rho, and are left alone.
    """
    lv_pinch = np.intersect1d(surf.lv_endo, surf.lv_outer)
    lv_pinch = lv_pinch[surf.lv_nodes[lv_pinch]]
    rv_pinch = np.intersect1d(surf.rv_free_endo, surf.epi)
    rv_pinch = rv_pinch[~surf.lv_nodes[rv_pinch]]
    sources = {"lv endo": np.setdiff1d(surf.lv_endo, lv_pinch),
               "lv outer": np.setdiff1d(surf.lv_outer, lv_pinch),
               "rv free endo": np.setdiff1d(surf.rv_free_endo, rv_pinch),
               "epi": np.setdiff1d(surf.epi, rv_pinch)}
    return sources, np.union1d(lv_pinch, rv_pinch)


def base_plane(points: np.ndarray, base: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    p = np.asarray(points, dtype=float)[base]
    origin = p.mean(axis=0)
    _, _, vt = np.linalg.svd(p - origin, full_matrices=False)
    return origin, vt[-1]


def mean_edge(points: np.ndarray, tets: np.ndarray, sample: int = 20_000) -> float:
    t = np.asarray(tets, dtype=np.int64)
    if len(t) > sample:
        t = t[np.random.default_rng(0).choice(len(t), sample, replace=False)]
    p = np.asarray(points, dtype=float)[t]
    pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    return float(np.mean([np.linalg.norm(p[:, i] - p[:, j], axis=1).mean()
                          for i, j in pairs]))


@dataclass
class Apex:
    """A transmural line of nodes, from an endocardial to an epicardial node."""

    endo: int
    epi: int
    nodes: np.ndarray
    automatic: bool = True


def apex_line(points: np.ndarray, endo: int, epi: np.ndarray, edge: float, *,
              automatic: bool = True) -> Apex:
    """The nodes along the segment from *endo* to its nearest node of *epi*."""
    p = np.asarray(points, dtype=float)
    out = int(epi[int(np.argmin(np.linalg.norm(p[epi] - p[endo], axis=1)))])
    a, b = p[endo], p[out]
    ab = b - a
    length = float(np.linalg.norm(ab)) or 1.0
    reach = APEX_LINE_EDGES * edge
    # Candidates near the segment's bounding box first, for speed.
    lo, hi = np.minimum(a, b) - reach, np.maximum(a, b) + reach
    cand = np.where(np.all((p >= lo) & (p <= hi), axis=1))[0]
    s = np.clip((p[cand] - a) @ ab / (length * length), 0.0, 1.0)
    d = np.linalg.norm(p[cand] - (a + s[:, None] * ab), axis=1)
    nodes = np.union1d(cand[d <= reach], [endo, out]).astype(np.int64)
    return Apex(endo=int(endo), epi=out, nodes=nodes, automatic=automatic)


def lv_apex_ends(points: np.ndarray, lv_endo: np.ndarray, epi: np.ndarray,
                 base: np.ndarray) -> Tuple[int, int]:
    """``(endo, epi)``: the two ends of the LV apex, shared by UVC and fibres.

    *endo* is the LV-endocardial node farthest, in a straight line, from
    the centroid of the base nodes; *epi* the epicardial node nearest it.
    Base nodes are never chosen. UVC hold the transmural line between the
    two; the fibres hold an epicardial patch around *epi*. One rule for
    both means "the apex" is the same place in either window.
    """
    p = np.asarray(points, dtype=float)
    endo = np.setdiff1d(lv_endo, base)
    outer = np.setdiff1d(epi, base)
    if len(endo) == 0 or len(outer) == 0:
        raise ValueError("the LV endocardium or the epicardium has no node off the base")
    centre = p[np.asarray(base, dtype=np.int64)].mean(axis=0)
    e = int(endo[int(np.argmax(np.linalg.norm(p[endo] - centre, axis=1)))])
    o = int(outer[int(np.argmin(np.linalg.norm(p[outer] - p[e], axis=1)))])
    return e, o


def _mask_nodes(mask: np.ndarray, bit_key: int, base: np.ndarray) -> np.ndarray:
    return np.setdiff1d(np.where(np.asarray(mask, dtype=np.int64) & MASK_BITS[bit_key])[0], base)


def rv_apex_node(points: np.ndarray, tets: np.ndarray, mask: np.ndarray) -> int:
    """The RV-endocardial node farthest from the RV endocardium's base rim,
    measured along the RV endocardium (septum included).

    Distances are shortest paths along the mesh edges that join two
    RV-endocardial nodes, from the RV-endocardial nodes on the base. Only
    the elements with two or more RV-endocardial nodes are looked at, so
    this is cheap on any mesh.
    """
    p = np.asarray(points, dtype=float)
    t = np.asarray(tets, dtype=np.int64)
    m = np.asarray(mask, dtype=np.int64)
    on = (m & MASK_BITS[RV_ENDO]) != 0
    base = (m & MASK_BITS[BASE]) != 0
    t = t[on[t].sum(axis=1) >= 2]
    pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    e = np.vstack([t[:, [a, b]] for a, b in pairs])
    e = np.unique(np.sort(e[on[e[:, 0]] & on[e[:, 1]]], axis=1), axis=0)
    rim = np.where(on & base)[0]
    if len(e) == 0 or len(rim) == 0:
        raise ValueError("the RV endocardium has no edges or does not reach the base")
    n = len(p)
    w = np.linalg.norm(p[e[:, 0]] - p[e[:, 1]], axis=1)
    graph = sp.coo_matrix((w, (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    d = dijkstra(graph + graph.T, indices=rim, min_only=True)
    cand = np.where(on & ~base & np.isfinite(d))[0]
    if len(cand) == 0:
        raise ValueError("no RV-endocardial node off the base is connected to it")
    return int(cand[np.argmax(d[cand])])


def automatic_apices(points: np.ndarray, tets: np.ndarray, mask: np.ndarray
                     ) -> Tuple[Apex, Apex]:
    """``(lv, rv)``: the automatic apices, from the node label mask.

    * LV: :func:`lv_apex_ends`, the LV-endocardial node farthest from the
      centre of the base; the fibres use the same location.
    * RV: :func:`rv_apex_node`, the RV-endocardial node farthest from its
      base rim along the RV endocardium.

    Each becomes a line across the wall to the nearest epicardial node.
    Cheap, so the window can show both before computing.

    The reference UVC tool chooses its apices by a rule it does not
    document, and no rule tried reproduced it. These were chosen by
    measurement on 29 ventricles with reference coordinates: the LV rule
    puts the LV apex within a few elements of the reference's, and with
    this RV rule the rotational coordinate differs from the reference's by
    a median 0.08 rad in the LV and 0.11 rad in the RV. The RV apex sets
    the LV-RV direction, and with it where phi is 0, so where an exact
    match with a reference model matters, pick its apices.
    """
    m = np.asarray(mask, dtype=np.int64)
    base = np.where(m & MASK_BITS[BASE])[0]
    epi = _mask_nodes(m, EPI, base)
    lv_endo = _mask_nodes(m, LV_ENDO, base)
    edge = mean_edge(points, tets)
    lv_node, _ = lv_apex_ends(points, lv_endo, epi, base)
    rv_node = rv_apex_node(points, tets, m)
    return (apex_line(points, lv_node, epi, edge),
            apex_line(points, rv_node, epi, edge))


def apex_at(points: np.ndarray, tets: np.ndarray, mask: np.ndarray, position,
            ventricle: str) -> Apex:
    """The apex line from the endocardial node of *ventricle* (``"lv"`` or
    ``"rv"``) nearest *position*: what a click becomes.

    Works from the node label mask alone, so a click costs nothing on a
    large mesh. The RV endocardium includes the septum.
    """
    mask = np.asarray(mask, dtype=np.int64)
    base = np.where(mask & MASK_BITS[BASE])[0]
    bit = MASK_BITS[LV_ENDO] if ventricle == "lv" else MASK_BITS[RV_ENDO]
    endo = np.setdiff1d(np.where(mask & bit)[0], base)
    epi = np.setdiff1d(np.where(mask & MASK_BITS[EPI])[0], base)
    if len(endo) == 0 or len(epi) == 0:
        raise ValueError(f"the {ventricle.upper()} endocardium or the epicardium "
                         f"has no node off the base")
    p = np.asarray(points, dtype=float)
    i = int(endo[int(np.argmin(np.linalg.norm(p[endo] - np.asarray(position, float), axis=1)))])
    return apex_line(points, i, epi, mean_edge(points, tets), automatic=False)

# ---------------------------------------------------------------------
# Rotational sectors
# ---------------------------------------------------------------------
@dataclass
class Frame:
    """The axis frame phi is measured in."""

    origin: np.ndarray
    long_axis: np.ndarray        # apex to base
    lv_rv: np.ndarray            # towards the RV, perpendicular to long_axis
    anterior: np.ndarray         # completes the frame; anterior is positive phi

    def angle(self, pts: np.ndarray) -> np.ndarray:
        """Angle around the long axis, degrees, 0 towards the RV, positive anterior."""
        c = np.asarray(pts, dtype=float) - self.origin
        return np.degrees(np.arctan2(c @ self.anterior, c @ self.lv_rv))


def frame(points: np.ndarray, surf: Surfaces, lv: Apex, rv: Apex) -> Frame:
    p = np.asarray(points, dtype=float)
    origin = p[surf.lv_endo].mean(axis=0)
    lv_apex = p[lv.nodes].mean(axis=0)
    long_axis = origin - lv_apex
    long_axis /= np.linalg.norm(long_axis)
    to_rv = p[rv.nodes].mean(axis=0) - lv_apex
    lv_rv = to_rv - (to_rv @ long_axis) * long_axis
    norm = np.linalg.norm(lv_rv)
    if norm == 0.0:
        raise ValueError("the RV apex lies on the long axis; the LV-RV "
                         "direction is undefined")
    lv_rv /= norm
    return Frame(origin=origin, long_axis=long_axis, lv_rv=lv_rv,
                 anterior=np.cross(long_axis, lv_rv))


@dataclass
class Junctions:
    """Angles (degrees) of the septal edges and the LV-RV junctions."""

    septal_ant: float
    septal_post: float
    junction_ant: float
    junction_post: float


def junctions(points: np.ndarray, surf: Surfaces, fr: Frame) -> Junctions:
    p = np.asarray(points, dtype=float)
    if len(surf.rv_septum) == 0 or len(surf.interface) == 0:
        raise ValueError("no RV septum or no LV-RV interface: the LV and RV "
                         "tags do not meet as a biventricular mesh should")
    sa = fr.angle(p[surf.rv_septum])
    ia = fr.angle(p[surf.interface])
    j = Junctions(septal_ant=float(sa.max()), septal_post=float(sa.min()),
                  junction_ant=float(ia.max()), junction_post=float(ia.min()))
    if j.septal_ant - j.septal_post > 300.0:
        raise ValueError("the RV septum surrounds the long axis instead of facing "
                         "the RV: check the surface labels and the LV/RV tags")
    if not (j.septal_post < 0.0 < j.septal_ant):
        raise ValueError(f"the RV septum does not straddle the LV-RV direction "
                         f"(it spans {j.septal_post:.0f} to {j.septal_ant:.0f} "
                         f"degrees); check the apices")
    return j


def sectors(points: np.ndarray, tets: np.ndarray, surf: Surfaces, fr: Frame,
            j: Junctions) -> np.ndarray:
    """Sector number per element (see :data:`SECTORS`)."""
    a = fr.angle(np.asarray(points, dtype=float)[np.asarray(tets, dtype=np.int64)].mean(axis=1))
    lv = surf.lv_elems
    out = np.empty(len(a), dtype=np.int64)
    out[~lv] = np.where(a[~lv] >= 0.0, RV_ANT, RV_POST)
    al = a[lv]
    sec = np.where(al >= 0.0, SEPT_ANT, SEPT_POST)
    sec = np.where(al > j.septal_ant, JUNC_ANT, sec)
    sec = np.where(al < j.septal_post, JUNC_POST, sec)
    sec = np.where(al > max(j.junction_ant, j.septal_ant), FREE_ANT, sec)
    sec = np.where(al < min(j.junction_post, j.septal_post), FREE_POST, sec)
    out[lv] = sec
    return out


def face_nodes(tets: np.ndarray, sector: np.ndarray, n: int, surf: Surfaces
               ) -> Dict[str, np.ndarray]:
    """Nodes of each sector boundary face: nodes touching both sectors."""
    tets = np.asarray(tets, dtype=np.int64)
    touches = {}
    for s in SECTORS:
        flag = np.zeros(n, dtype=bool)
        flag[tets[sector == s].ravel()] = True
        touches[s] = flag
    out = {name: np.where(touches[a] & touches[b])[0] for name, (a, b) in FACES.items()}
    rv_side = {"rvjuncant": RV_ANT, "rvjuncpost": RV_POST}
    for name, s in rv_side.items():
        flag = np.zeros(n, dtype=bool)
        flag[surf.interface] = True
        out[name] = np.where(flag & touches[s])[0]
    return out


# ---------------------------------------------------------------------
# The coordinates
# ---------------------------------------------------------------------
@dataclass
class UvcRun:
    """Everything one run produced."""

    fields: Dict[str, np.ndarray]
    lv_apex: Apex
    rv_apex: Apex
    lv_tag: int
    rv_tag: int
    frame: Frame
    junctions: Junctions
    sector: np.ndarray
    seconds: float = 0.0
    timings: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    #: Nodes on an endocardium and the outer wall at once, held by neither
    #: transmural surface.
    released: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    def summary(self) -> str:
        return f"UVC for {len(self.fields['z']):,} nodes in {self.seconds:.0f} s."

    def details(self) -> str:
        f, j = self.fields, self.junctions
        lines = [self.summary(), "",
                 f"LV tag {self.lv_tag}, RV tag {self.rv_tag}: "
                 f"{int(np.sum(f['ven'] < 0)):,} LV and {int(np.sum(f['ven'] > 0)):,} RV nodes.",
                 f"LV apex: {len(self.lv_apex.nodes)} nodes "
                 f"({'found automatically' if self.lv_apex.automatic else 'picked'}); "
                 f"RV apex: {len(self.rv_apex.nodes)} nodes "
                 f"({'found automatically' if self.rv_apex.automatic else 'picked'}).",
                 f"Septal edges at {j.septal_ant:+.1f} and {j.septal_post:+.1f} degrees, "
                 f"junctions at {j.junction_ant:+.1f} and {j.junction_post:+.1f} degrees "
                 f"around the long axis.",
                 "Distance fields: " + ", ".join(f"{k} {v:.1f} s" for k, v in self.timings.items()) + "."]
        if len(self.released):
            lines.append(f"Released {len(self.released)} node(s) lying on an "
                         f"endocardium and the outer wall at once (the wall has no "
                         f"thickness there): held by neither for rho.")
        lines += [f"Warning: {w}" for w in self.warnings]
        return "\n".join(lines)


def compute(points: np.ndarray, tets: np.ndarray, mask: np.ndarray,
            tags: np.ndarray, *, lv_apex: Optional[Apex] = None,
            rv_apex: Optional[Apex] = None, threads: Optional[int] = None,
            on_status: Optional[Callable[[str], None]] = None) -> UvcRun:
    """The four coordinates. Pure numerics on arrays, safe off the GUI thread."""
    say = on_status or (lambda _m: None)
    start = time.time()
    points = np.asarray(points, dtype=float)
    tets = np.asarray(tets, dtype=np.int64)
    tags = np.asarray(tags)
    n = len(points)
    lv_tag, rv_tag = lv_rv_tags(tets, tags, mask)
    say("Finding the RV septum, the LV outer wall and the LV-RV interface…")
    surf = surfaces(tets, mask, tags == lv_tag)
    if len(surf.rv_septum) == 0:
        raise ValueError("no RV-endocardial face belongs to an LV element: "
                         "the RV septum is empty")
    check_labels(points, surf)
    say("Building the mesh adjacency…")
    adj = eikonal.adjacency(tets, n)
    g, timings, warnings = {}, {}, []

    def solve(name: str, src: np.ndarray) -> None:
        t0 = time.time()
        g[name], rep = eikonal.distance(points, tets, src, adj=adj, threads=threads)
        timings[name] = time.time() - t0
        if rep.unreached:
            warnings.append(f"{rep.unreached} node(s) unreachable from the {name}")

    say("Distance 1: base…")
    solve("base", surf.base)
    if lv_apex is None or rv_apex is None:
        auto_lv, auto_rv = automatic_apices(points, tets, mask)
        lv_apex = lv_apex or auto_lv
        rv_apex = rv_apex or auto_rv
    fr = frame(points, surf, lv_apex, rv_apex)
    j = junctions(points, surf, fr)
    sector = sectors(points, tets, surf, fr, j)
    faces = face_nodes(tets, sector, n, surf)
    empty = [k for k, v in faces.items() if len(v) == 0]
    if empty:
        raise ValueError(f"empty sector boundaries: {', '.join(empty)}")

    transmural, released = transmural_sources(surf)
    sources = {"lv apex": lv_apex.nodes, "rv apex": rv_apex.nodes, **transmural}
    sources.update({f"face {k}": v for k, v in faces.items()})
    total = len(sources) + 1
    for i, (name, src) in enumerate(sources.items(), start=2):
        say(f"Distance {i} of {total}: {name}…")
        solve(name, src)

    say("Combining the distances…")
    ratio = eikonal.distance_ratio
    lv = surf.lv_nodes
    ven = np.where(lv, -1.0, 1.0)
    z = np.where(lv, ratio(g["lv apex"], g["base"]), ratio(g["rv apex"], g["base"]))
    rho = np.where(lv, ratio(g["lv endo"], g["lv outer"]),
                   ratio(g["rv free endo"], g["epi"]))
    phi = _phi(tets, sector, surf, g, n)
    return UvcRun(fields={"ven": ven, "z": z, "rho": rho, "phi": phi},
                  lv_apex=lv_apex, rv_apex=rv_apex, lv_tag=lv_tag, rv_tag=rv_tag,
                  frame=fr, junctions=j, sector=sector, seconds=time.time() - start,
                  timings=timings, warnings=warnings, released=released)


def _phi(tets: np.ndarray, sector: np.ndarray, surf: Surfaces,
         g: Dict[str, np.ndarray], n: int) -> np.ndarray:
    """phi at every node, from the sector its elements lie in.

    A node on a face between two sectors gets the face's value from either
    side. The LV free-wall face, where +pi meets -pi, takes +pi. LV nodes
    take an LV sector and RV nodes an RV one.
    """
    phi = np.full(n, np.nan)
    # Later sectors overwrite earlier ones, so FREE_ANT goes last among the
    # LV sectors (its +pi wins on the cut), and LV nodes are written last.
    order = [RV_POST, RV_ANT, SEPT_POST, SEPT_ANT, JUNC_POST, JUNC_ANT, FREE_POST, FREE_ANT]
    for s in order:
        nodes = np.unique(tets[sector == s].ravel())
        nodes = nodes[surf.lv_nodes[nodes]] if s not in (RV_ANT, RV_POST) else nodes[~surf.lv_nodes[nodes]]
        a, va, b, vb = SECTORS[s]
        r = eikonal.distance_ratio(g[f"face {a}"][nodes], g[f"face {b}"][nodes])
        phi[nodes] = va + (vb - va) * r
    return phi


# ---------------------------------------------------------------------
# On a dataset
# ---------------------------------------------------------------------
def inputs(grid) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(points, tets, mask, tags)`` of a labelled, tagged volume."""
    _name, tags = element_tags(grid)
    return (np.asarray(grid.points, dtype=float), tetrahedra(grid),
            np.asarray(grid.point_data[LABEL_MASK_FIELD], dtype=np.int64), tags)


def write(grid, run: UvcRun) -> None:
    """Put the four coordinates on *grid* as point arrays."""
    for key, name in UVC_FIELDS.items():
        grid.point_data[name] = np.asarray(run.fields[key], dtype=float)


def written_names() -> Tuple[str, ...]:
    return tuple(UVC_FIELDS.values())


__all__ = ["UVC_FIELDS", "TAG_FIELDS", "SECTORS", "FACES", "Surfaces", "Apex",
           "Frame", "Junctions", "UvcRun", "element_tags", "lv_rv_tags",
           "availability", "surfaces", "check_labels", "transmural_sources", "lv_apex_ends", "rv_apex_node", "automatic_apices", "apex_at", "apex_line",
           "frame", "junctions", "sectors", "face_nodes", "compute", "inputs",
           "write", "written_names"]
