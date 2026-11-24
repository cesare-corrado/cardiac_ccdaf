"""
surface_repair
==============
Connectivity repairs of a triangle surface, shared by **Clean**, clipping,
tagging and **Compute UAC**, so each of them changes a mesh the same way.

No repair moves a point. They change which points the triangles join, drop
triangles, and add copies of points:

* :func:`remove_duplicates` - triangles listed twice: a zero-area flap is
  dropped whole; a duplicate that is part of the wall keeps one copy.
* :func:`trim_rims` - "ears" (triangles with two sides on an opening) and
  slivers whose long side lies on an opening, as clipping leaves them on a
  saw-tooth rim; points left with no triangle go too.
* :func:`flip_edges` - Delaunay edge flips between nearly coplanar
  triangles of the same label: a long thin pair is re-cut along its other
  diagonal.
* :func:`split_pinches` - a point where the rim passes twice (a bow tie)
  gets one copy per side.
* :func:`join_islands` - small islands of a label take the label round them.
* :func:`cut_tunnels` - a thin bridge of tissue (a tunnel) is cut along its
  shortest loop and both cut ends are closed with a fan of triangles that
  takes the label round the loop.

Each step works on an :class:`Edit`, which keeps, for every point and every
triangle, the input point or triangle it comes from (-1 for a new one).
:func:`apply` then builds the new surface and carries every point and cell
field across: a copied point or triangle gets its source's values, a new
point the mean of its neighbours (a label: a neighbour's value), a new
triangle a neighbour's values. The label array (``elemTag``) is the
exception: it is set from the edit's own labels.

The ``*_mesh`` wrappers run one step on a :class:`pyvista.PolyData`.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Optional, Tuple

import numpy as np
import pyvista as pv
import scipy.sparse as sp

from ccdaf.core.uac import mesh_ops as mo
from ccdaf.core.uac.tunnels import TunnelError, remove_tunnels

LABEL = "elemTag"

#: Two triangles are flipped only if their planes are within this many
#: degrees, so a flip never changes the shape of the surface.
MAX_CREASE_DEG = 10.0

#: A tunnel whose shortest loop is longer than this (mm) is not a thin
#: bridge, and is left alone.
TUNNEL_MAX_MM = 15.0


@dataclass
class Edit:
    """A surface being repaired, with where each point and triangle came from."""

    P: np.ndarray
    t: np.ndarray
    tag: np.ndarray
    psrc: np.ndarray                 # input point of each point, -1 if new
    csrc: np.ndarray                 # input triangle of each triangle, -1 if new
    report: Dict = field(default_factory=dict)
    frozen: Optional[np.ndarray] = None   # triangles no step may drop or re-cut

    @classmethod
    def start(cls, P, t, tag=None, frozen=None) -> "Edit":
        P = np.asarray(P, float)
        t = np.asarray(t, np.int64)
        tag = np.zeros(len(t), int) if tag is None else np.asarray(tag).astype(int)
        frozen = np.zeros(len(t), bool) if frozen is None else np.asarray(frozen, bool)
        return cls(P, t, tag, np.arange(len(P)), np.arange(len(t)), {}, frozen)

    def note(self, key: str, value) -> None:
        if isinstance(value, (int, np.integer)):
            self.report[key] = int(self.report.get(key, 0)) + int(value)
        else:
            self.report.setdefault(key, []).extend(value)

    @property
    def changed(self) -> bool:
        return any(v for v in self.report.values())


def _keep_cells(e: Edit, keep: np.ndarray) -> Edit:
    """Keep these triangles and drop the points no triangle uses any more."""
    t = e.t[keep]
    used = np.unique(t)
    remap = np.full(len(e.P), -1)
    remap[used] = np.arange(len(used))
    return replace(e, P=e.P[used], t=remap[t], tag=e.tag[keep], psrc=e.psrc[used],
                   csrc=e.csrc[keep], frozen=e.frozen[keep])


# ---------------------------------------------------------------- steps
def remove_duplicates(e: Edit) -> Edit:
    keep = mo.duplicate_keep(e.t) | e.frozen
    n = int((~keep).sum())
    if n:
        e = _keep_cells(e, keep)
    e.note("duplicates_removed", n)
    return e


def trim_rims(e: Edit, max_pass: int = 20) -> Edit:
    """Drop ears and slivers on openings until none is left."""
    ears = slivers = 0
    for _ in range(max_pass):
        edges = mo.unique_edges(e.t)
        ear = mo.ear_mask(e.t, edges) & ~e.frozen
        sl = mo.rim_sliver_mask(e.P, e.t, edges) & ~ear & ~e.frozen
        if not (ear.any() or sl.any()):
            break
        ears += int(ear.sum())
        slivers += int(sl.sum())
        e = _keep_cells(e, ~(ear | sl))
    e.note("ears_removed", ears)
    e.note("rim_slivers_removed", slivers)
    return e


def flip_edges(e: Edit, max_crease_deg: float = MAX_CREASE_DEG) -> Edit:
    """Delaunay flips, decided on a consistently wound copy; only the
    flipped triangles change (and take the copy's winding)."""
    t_or, _ = mo.fix_orientation(e.t)
    # flips join two triangles of one label: a frozen triangle gets a label
    # of its own, so it is never re-cut
    tag = np.where(e.frozen, -1 - np.arange(len(e.t)), e.tag)
    t_fl, n = mo.delaunay_flips(e.P, t_or, tag, max_crease_deg)
    if n:
        moved = (t_fl != t_or).any(1)
        t = e.t.copy()
        t[moved] = t_fl[moved]
        e = replace(e, t=t)
    e.note("flips", n)
    return e


def orient(e: Edit) -> Edit:
    t, n = mo.fix_orientation(e.t)
    e = replace(e, t=t)
    e.note("rewound", n)
    return e


def split_pinches(e: Edit) -> Edit:
    P, t, origin, n = mo.split_pinches(e.P, e.t)
    if n:
        e = replace(e, P=P, t=t, psrc=e.psrc[origin])
    e.note("pinches_split", n)
    return e


def join_islands(e: Edit) -> Edit:
    tag, n = mo.relabel_islands(e.t, e.tag)
    e = replace(e, tag=tag)
    e.note("island_triangles_relabelled", n)
    return e


def cut_tunnels(e: Edit, max_mm: float = TUNNEL_MAX_MM, strict: bool = False) -> Edit:
    """Cut the thin tunnels of every piece of the surface.

    ``strict``: raise :class:`TunnelError` on a tunnel wider than
    ``max_mm`` (UAC cannot go on); otherwise leave it and report it under
    ``tunnels_left``.
    """
    genus, _ = mo.topology(e.t)
    if genus == 0:
        e.note("tunnels", [])
        return e
    npieces, piece = mo.components(e.t)
    done, left = [], 0
    for c in range(npieces):
        rows = np.where(piece == c)[0]
        if mo.topology(e.t[rows])[0] == 0:
            continue
        used = np.unique(e.t[rows])
        remap = np.full(len(e.P), -1)
        remap[used] = np.arange(len(used))
        P2, t2, tag2, org, is_patch, rep = remove_tunnels(
            e.P[used], remap[e.t[rows]], e.tag[rows], np.arange(len(used)),
            max_mm=max_mm, strict=strict)
        if rep and "left" in rep[-1]:
            left += rep.pop()["left"]
        if not rep:
            continue
        # put the piece back: its old points keep their numbers, new points go last
        n0 = len(e.P)
        new_pts = np.where(org < 0, -1, used[np.maximum(org, 0)])
        k = len(used)
        gid = np.r_[used, n0 + np.arange(len(P2) - k)]
        P = np.vstack([e.P, P2[k:]])
        psrc = np.r_[e.psrc, np.where(new_pts[k:] >= 0, e.psrc[np.maximum(new_pts[k:], 0)], -1)]
        t = e.t.copy()
        t[rows] = gid[t2[:len(rows)]]
        caps = gid[t2[len(rows):]]
        tag = e.tag.copy()
        tag[rows] = tag2[:len(rows)]
        e = replace(e, P=P, t=np.vstack([t, caps]), tag=np.r_[tag, tag2[len(rows):]],
                    psrc=psrc, csrc=np.r_[e.csrc, np.full(len(caps), -1)],
                    frozen=np.r_[e.frozen, np.zeros(len(caps), bool)])
        done += rep
        piece = np.r_[piece, np.full(len(caps), c)]
    e.note("tunnels", done)
    e.note("tunnels_left", left)
    return e


# ---------------------------------------------------------------- fields
def _point_neighbours(n: int, t: np.ndarray) -> sp.csr_matrix:
    ue, _, _ = mo.unique_edges(t)
    return sp.coo_matrix((np.ones(2 * len(ue)), (np.r_[ue[:, 0], ue[:, 1]], np.r_[ue[:, 1], ue[:, 0]])),
                         shape=(n, n)).tocsr()


def _cell_neighbours(t: np.ndarray) -> sp.csr_matrix:
    _, inv, _ = mo.unique_edges(t)
    m = len(t)
    side = inv.reshape(3, m)
    own = np.tile(np.arange(m), 3)
    E = sp.coo_matrix((np.ones(3 * m), (side.ravel(), own)), shape=(int(inv.max()) + 1, m)).tocsr()
    A = (E.T @ E).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()
    return A


def _fill(values: np.ndarray, known: np.ndarray, A: sp.csr_matrix, average: bool) -> np.ndarray:
    """Give unknown rows their known neighbours' mean (or one neighbour's value)."""
    values = values.copy()
    known = known.copy()
    for _ in range(100):
        if known.all():
            break
        k = known.astype(float)
        cnt = A @ k
        grow = ~known & (cnt > 0)
        if not grow.any():
            break
        flat = values.reshape(len(values), -1)
        if average and np.issubdtype(values.dtype, np.floating):
            s = A @ np.where(known[:, None], np.nan_to_num(flat), 0.0)
            flat[grow] = s[grow] / cnt[grow, None]
        else:
            rows = np.where(grow)[0]
            for r in rows:
                nb = A.indices[A.indptr[r]:A.indptr[r + 1]]
                nb = nb[known[nb]]
                flat[r] = flat[nb[0]]
        known = known | grow
    return values


def apply(mesh: pv.PolyData, e: Edit, label: Optional[str] = LABEL) -> pv.PolyData:
    """The repaired surface, with every field of *mesh* carried across."""
    faces = np.c_[np.full(len(e.t), 3), e.t].ravel()
    out = pv.PolyData(e.P.copy(), faces)
    pk = e.psrc >= 0
    A_p = _point_neighbours(len(e.P), e.t) if not pk.all() else None
    for name in mesh.point_data.keys():
        arr = np.asarray(mesh.point_data[name])
        new = np.zeros((len(e.P),) + arr.shape[1:], arr.dtype)
        new[pk] = arr[e.psrc[pk]]
        if A_p is not None:
            new = _fill(new, pk, A_p, average=True)
        out.point_data[name] = new
    ck = e.csrc >= 0
    A_c = _cell_neighbours(e.t) if not ck.all() else None
    for name in mesh.cell_data.keys():
        arr = np.asarray(mesh.cell_data[name])
        new = np.zeros((len(e.t),) + arr.shape[1:], arr.dtype)
        new[ck] = arr[e.csrc[ck]]
        if A_c is not None:
            new = _fill(new, ck, A_c, average=False)
        out.cell_data[name] = new
    if label is not None and label in mesh.cell_data.keys():
        out.cell_data[label] = e.tag.astype(np.asarray(mesh.cell_data[label]).dtype)
    for key in mesh.field_data.keys():
        out.field_data[key] = mesh.field_data[key]
    return out


def _start(mesh: pv.PolyData, label: Optional[str] = LABEL, frozen_labels=()) -> Edit:
    faces = np.asarray(mesh.faces).reshape(-1, 4)
    if faces.size and (faces[:, 0] != 3).any():
        raise ValueError("the surface must be made of triangles only")
    tag = (np.asarray(mesh.cell_data[label]) if label is not None and label in mesh.cell_data.keys()
           else None)
    frozen = (np.isin(tag, list(frozen_labels)) if tag is not None and len(frozen_labels)
              else None)
    return Edit.start(mesh.points, faces[:, 1:], tag, frozen)


def run(mesh: pv.PolyData, *steps, label: Optional[str] = LABEL,
        frozen_labels=()) -> Tuple[pv.PolyData, Dict]:
    """Run the steps in order on *mesh*; return the repaired copy and the
    report. The input is not changed; when nothing changed, the copy is a
    plain copy of it.

    ``frozen_labels``: triangles with these labels are never dropped or
    re-cut (their points may still gain a copy where a pinch is split or a
    tunnel is cut; no point moves either way)."""
    e = _start(mesh, label, frozen_labels)
    for step in steps:
        e = step(e)
    return (apply(mesh, e, label) if e.changed else mesh.copy()), e.report


__all__ = [
    "Edit", "LABEL", "MAX_CREASE_DEG", "TUNNEL_MAX_MM", "TunnelError",
    "remove_duplicates", "trim_rims", "flip_edges", "orient", "split_pinches",
    "join_islands", "cut_tunnels", "apply", "run",
]
