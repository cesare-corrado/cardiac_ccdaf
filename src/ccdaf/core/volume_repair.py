"""
volume_repair
=============
Make a tetrahedral volume's boundary manifold: separate what only
touches, close what only just fails to close.

Why this is worth doing
-----------------------
A segmented ventricle arrives with two kinds of defect that no amount of
dropping elements will fix, and both of them show up as a *hole-like*
patch on the wall when the surface is rendered:

* **Touching blocks.** Two pieces of material meet at a single node or
  along a single edge and share no face. The mesh is one piece to the
  eye and two to a solver, and the boundary is non-manifold at the
  contact.
* **Pinholes.** The wall thins to nothing at a point or along an edge,
  so the surface passes through the same node twice. The material is
  continuous *around* the contact: the node's link — the surface made by
  the faces of its tetrahedra opposite it — is an annulus rather than a
  disk, which is the signature of a passage through the wall that closes
  to zero width.

On the 290,474-element example ventricle there are 18 non-manifold
boundary edges and 17 pinched vertices, all of them on the basal rim and
the left/right junction, and every one of them is one of those two
things.

Why the two are repaired differently
------------------------------------
Touching blocks are **split**: each fan of material meeting at the node
gets its own copy of it. That is exact — no element is removed, no
coordinate moves, no tissue is invented — and it is possible precisely
because the fans share no face, so nothing is being torn apart. It can
leave a block attached by nothing at all, which is the right answer: the
caller's component pass then sees it for the stray it is.

Pinholes cannot be split: the tetrahedra around the node form one
face-connected fan, so separating them would tear material that is
genuinely joined. They are **welded** instead, by filling the empty
wedge at the contact with a tetrahedron (at an edge) or by capping the
smaller loop of the annulus and coning it back to the node (at a
vertex). That adds material, which is why every weld is bounded: no
element a weld adds may be larger than the elements already meeting
there, by more than :attr:`RepairOptions.max_weld_volume`. A pinhole is
a sliver of nothing and passes that bound; a cap spanning a genuinely
open passage is many times the local element and does not, so it is
reported rather than filled. Inventing anatomy is the one thing a repair
must not do quietly.

A weld is bounded in shape as well as in volume. A wedge can be wide and
thin at once, which passes a volume test and still produces an element
with no usable shape; such a contact is reported rather than welded,
because an element that flat repairs nothing — it moves the problem from
the topology report to the quality report.

What a weld costs
-----------------
A weld fuses the two sides of the wall at the contact. Where the wall
already had zero thickness that is the honest reading of the geometry,
but it is a real change: a solver that saw two separated surfaces there
now sees one piece of material. The counts and the added volume are
reported for exactly that reason.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

# Ear-clipping a boundary loop is the surface cleaner's problem too, and
# it solved it first: reused rather than written again, so a loop that
# one of them can triangulate cannot be one the other cannot.
from ccdaf.core.mesh_postprocessor import _earclip_triangulate
from ccdaf.core.volume_mesh import (
    boundary_table, face_table, node_cells, orient_positive, shape_measure,
    signed_volumes, sorted_unique_rows, unique_rows,
)

#: How nearly parallel a face and its material direction may be before
#: the wedge test at a non-manifold edge is called undecidable, in
#: radians. Below it the owning tetrahedron is so flat that "which side
#: is material" has no reliable answer, and the edge is reported instead
#: of guessed at.
_ANGLE_EPS = 1e-6

#: A new element's volume must exceed this fraction of the mean volume
#: of the elements at the contact, or it is a sliver that would be
#: dropped by the next degenerate pass anyway — better not to make it.
_MIN_RELATIVE_VOLUME = 1e-6

#: How wide a gap may be and still count as a perforation to plug, as a
#: fraction of the elements around it. Sub-element on purpose: this is
#: what keeps the plug away from anatomy. On the example ventricle the
#: perforation measures 0.38 mm against a 1.58 mm mean edge, while the
#: valve openings it must never touch are 177 to 1226 mm2 across.
_PLUG_GAP: float = 0.5

#: The most distorted element a weld may add, on the shape measure where
#: 0 is regular and 1 is flat.
#:
#: A volume bound alone is not enough, and the mesh that proved it was a
#: 14-million-element one: every weld there passed the volume test and
#: one of them still came out at 0.9999, because a wedge can be wide and
#: thin at the same time. An element that flat repairs nothing — it
#: moves the problem from the topology report to the quality report — so
#: the contact is reported instead. 0.95 refuses only the genuinely
#: degenerate: at the 25 contacts welded on the example ventricle the
#: worst element added measured 0.667.
_MAX_WELD_QUALITY = 0.95


@dataclass
class RepairOptions:
    """What the repair is allowed to do.

    Splitting is exact and is on. Welding adds material and is also on,
    because a pinhole left alone keeps the boundary non-manifold and
    every surface-labelling pass downstream has to cope with it — but it
    is bounded by :attr:`max_weld_volume` and can be switched off.
    """

    #: Give each fan of material meeting at a node its own copy of it.
    split_touching: bool = True
    #: Fill the empty wedge at a pinhole. See the module docstring.
    weld_pinholes: bool = True
    #: The largest element a weld may add, as a multiple of the mean
    #: volume of the elements already meeting at that contact.
    #:
    #: Two, not one, and measured rather than guessed: at the 17 pinched
    #: vertices of the example ventricle the elements already there
    #: range up to 1.7 times their own mean, so a limit of one mean
    #: refuses welds that are the same size as the mesh around them. The
    #: elements a weld actually adds there come out at 0.8 to 1.8 times
    #: that mean. A cap spanning a genuinely open passage is not twice
    #: the local element, it is many times it, so the bound still has
    #: teeth.
    max_weld_volume: float = 2.0
    #: Plug a perforation: a passage through the wall that is open
    #: rather than closed to a point, and narrow enough to be a defect
    #: rather than anatomy. Off by default — unlike a pinhole, this one
    #: adds material across a gap that is genuinely there.
    plug_perforations: bool = False
    #: The widest gap a plug may span, as a fraction of the elements
    #: around it. See :data:`_PLUG_GAP`.
    max_plug_gap: float = _PLUG_GAP
    #: How many split/weld rounds to run. Closing one contact can expose
    #: the next, so the passes repeat; they stop as soon as a round
    #: changes nothing.
    max_passes: int = 6

    def validate(self) -> None:
        if self.max_weld_volume < 0.0:
            raise ValueError("the weld volume limit must not be negative")
        if self.max_passes < 1:
            raise ValueError("the repair needs at least one pass")


@dataclass
class RepairReport:
    """What the repair did, and what it could not do."""

    split_vertices: int = 0
    split_copies: int = 0
    welded_edges: int = 0
    welded_vertices: int = 0
    added_cells: int = 0
    added_volume: float = 0.0
    passes: int = 0
    #: Contacts still non-manifold when the repair stopped: either the
    #: weld was refused (too much material) or the geometry was too
    #: degenerate to decide. Reported, never silently left.
    unrepaired_edges: int = 0
    unrepaired_vertices: int = 0
    #: Perforations plugged, and the genus of the boundary before and
    #: after. The genus is the count of handles — passages you could
    #: thread a loop through — and it is reported rather than driven to
    #: zero, because some of those handles are anatomy: a cavity with
    #: two valve openings has one whatever the mesh does.
    plugged: int = 0
    genus_before: Optional[int] = None
    genus_after: Optional[int] = None
    #: Why plugging was asked for and plugged nothing, or empty. A pass
    #: that was requested and did nothing must say so: on a perforated
    #: ventricle it used to leave a report identical to one with
    #: plugging off, which read as "no perforations" when the truth was
    #: "could not look".
    plug_note: str = ""

    @property
    def changed(self) -> bool:
        return bool(self.split_copies or self.added_cells or self.plugged)

    def summary(self) -> str:
        if not self.changed:
            if self.unrepaired_edges or self.unrepaired_vertices:
                text = (f"{self.unrepaired_edges + self.unrepaired_vertices} "
                        f"non-manifold contacts could not be repaired")
            else:
                text = "the boundary was already manifold"
            return f"{text}, {self.plug_note}" if self.plug_note else text
        bits: List[str] = []
        if self.split_vertices:
            bits.append(f"separated {self.split_vertices} touching contact"
                        + ("" if self.split_vertices == 1 else "s"))
        if self.plugged:
            bits.append(f"plugged {self.plugged} perforation"
                        + ("" if self.plugged == 1 else "s")
                        + f" (handles {self.genus_before} → "
                        + f"{self.genus_after})")
        welded = self.welded_edges + self.welded_vertices
        if welded:
            bits.append(f"welded {welded} pinhole"
                        + ("" if welded == 1 else "s")
                        + f" shut with {self.added_cells} element"
                        + ("" if self.added_cells == 1 else "s")
                        + f" ({self.added_volume:.4g} in volume)")
        left = self.unrepaired_edges + self.unrepaired_vertices
        if left:
            bits.append(f"{left} contact"
                        + ("" if left == 1 else "s")
                        + " left alone")
        if self.plug_note:
            bits.append(self.plug_note)
        return ", ".join(bits)


# ---------------------------------------------------------------------
# What the boundary is doing
# ---------------------------------------------------------------------
def boundary_edge_table(faces: np.ndarray):
    """Undirected edges of a triangle set: ``(uniq, inverse, counts)``."""
    pairs = faces[:, [[0, 1], [1, 2], [0, 2]]].reshape(-1, 2)
    uniq, inv, counts = sorted_unique_rows(pairs)
    return uniq, inv.ravel(), counts


def non_manifold_edges(faces: np.ndarray) -> np.ndarray:
    """Edges of *faces* carried by more than two triangles."""
    if len(faces) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    uniq, _inv, counts = boundary_edge_table(faces)
    return uniq[counts > 2]


def _corner_groups(faces: np.ndarray):
    """Group the corners of *faces* into fans around each vertex.

    A corner is a ``(triangle, slot)`` pair, identified as
    ``3 * triangle + slot``. Two corners at the same vertex are joined
    when their triangles share an edge through that vertex, so a vertex
    ends up with one group per fan of surface passing through it, and a
    vertex with two groups is a place where the surface touches itself.

    Corners are the unit of work rather than triangles, which is exactly
    what keeps the two fans apart at a pinch. Triangles are joined
    across non-manifold edges as well as manifold ones, so that this
    counts *only* surfaces touching at a point: joining across manifold
    edges alone would report every endpoint of a non-manifold edge as a
    pinch too, which on the example ventricle reads 47 rather than the
    17 places where the surface genuinely touches itself.
    """
    uniq, inv, counts = boundary_edge_table(faces)
    face_of_edge = np.repeat(np.arange(len(faces), dtype=np.int64), 3)
    order = np.argsort(inv, kind="stable")
    starts = np.searchsorted(inv[order], np.arange(len(counts)))

    multi = np.where(counts >= 2)[0]
    extra = counts[multi] - 1
    total = int(extra.sum())
    n_corners = 3 * len(faces)
    if total == 0:
        return np.arange(n_corners, dtype=np.int64), faces.reshape(-1)

    base = np.repeat(starts[multi], extra)
    offsets = np.arange(total) - np.repeat(np.cumsum(extra) - extra, extra) + 1
    lead = face_of_edge[order[base]]
    trail = face_of_edge[order[base + offsets]]
    ends = np.repeat(uniq[multi], extra, axis=0)

    rows: List[np.ndarray] = []
    cols: List[np.ndarray] = []
    for end in (0, 1):
        vertex = ends[:, end]
        slot0 = np.argmax(faces[lead] == vertex[:, None], axis=1)
        slot1 = np.argmax(faces[trail] == vertex[:, None], axis=1)
        rows.append(3 * lead + slot0)
        cols.append(3 * trail + slot1)
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    labels = connected_components(
        coo_matrix((np.ones(len(r), dtype=np.int8), (r, c)),
                   shape=(n_corners, n_corners)), directed=False)[1]
    return labels, faces.reshape(-1)


def pinched_vertices(faces: np.ndarray) -> np.ndarray:
    """Vertices where the surface *faces* passes through twice or more."""
    if len(faces) == 0:
        return np.zeros(0, dtype=np.int64)
    labels, corner_vertex = _corner_groups(faces)
    fans = np.unique(np.stack([corner_vertex, labels], axis=1), axis=0)
    vertices, counts = np.unique(fans[:, 0], return_counts=True)
    return vertices[counts > 1]


def boundary_sheets(faces: np.ndarray) -> int:
    """How many separate sheets *faces* forms, joined across manifold edges."""
    if len(faces) == 0:
        return 0
    uniq, inv, counts = boundary_edge_table(faces)
    face_of_edge = np.repeat(np.arange(len(faces), dtype=np.int64), 3)
    order = np.argsort(inv, kind="stable")
    starts = np.searchsorted(inv[order], np.arange(len(counts)))
    shared = np.where(counts == 2)[0]
    if shared.size == 0:
        return 0
    f0 = face_of_edge[order[starts[shared]]]
    f1 = face_of_edge[order[starts[shared] + 1]]
    return int(connected_components(
        coo_matrix((np.ones(len(f0), dtype=np.int8), (f0, f1)),
                   shape=(len(faces),) * 2), directed=False)[0])


# ---------------------------------------------------------------------
# Separating what only touches
# ---------------------------------------------------------------------
def _tet_corner_fans(tets: np.ndarray) -> np.ndarray:
    """Label every ``(tetrahedron, corner)`` by the fan of material it is in.

    Two corners at the same node are joined when their tetrahedra share
    a face through that node. A node whose corners fall into more than
    one group is a node where separate pieces of material touch: the
    pieces share no face, so each can be given its own copy of the node
    without tearing anything.
    """
    n = len(tets)
    if n == 0:
        return np.zeros((0, 4), dtype=np.int64)
    uniq, inv, counts, owner = face_table(tets)
    order = np.argsort(inv, kind="stable")
    starts = np.searchsorted(inv[order], np.arange(len(counts)))

    multi = np.where(counts >= 2)[0]
    extra = counts[multi] - 1
    total = int(extra.sum())
    size = 4 * n
    if total == 0:
        return np.arange(size, dtype=np.int64).reshape(n, 4)

    base = np.repeat(starts[multi], extra)
    offsets = np.arange(total) - np.repeat(np.cumsum(extra) - extra, extra) + 1
    lead = owner[order[base]]
    trail = owner[order[base + offsets]]
    shared_nodes = np.repeat(uniq[multi], extra, axis=0)

    rows: List[np.ndarray] = []
    cols: List[np.ndarray] = []
    for k in range(3):
        node = shared_nodes[:, k]
        slot0 = np.argmax(tets[lead] == node[:, None], axis=1)
        slot1 = np.argmax(tets[trail] == node[:, None], axis=1)
        rows.append(4 * lead + slot0)
        cols.append(4 * trail + slot1)
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    labels = connected_components(
        coo_matrix((np.ones(len(r), dtype=np.int8), (r, c)),
                   shape=(size, size)), directed=False)[1]
    return labels.reshape(n, 4)


def split_touching(points: np.ndarray, tets: np.ndarray):
    """Give each fan of material at a node its own copy of that node.

    Returns ``(points, tets, source_point, split_nodes, copies)``.
    ``source_point`` says, for every point of the result, which point of
    the input it came from, so a caller carrying point data copies the
    original value onto each copy rather than inventing one.

    Nothing is removed and no coordinate changes: the copies sit exactly
    where the original sat.
    """
    if len(tets) == 0:
        return (points, tets, np.arange(len(points), dtype=np.int64), 0, 0)

    labels = _tet_corner_fans(tets)
    pairs = np.stack([tets.ravel(), labels.ravel()], axis=1)
    # Not sorted within each row: here the first column is the node and
    # the second its fan, and swapping them would be nonsense.
    uniq_pairs, inverse, _counts = unique_rows(pairs)
    inverse = inverse.ravel()

    vertex = uniq_pairs[:, 0]
    first = np.ones(len(vertex), dtype=bool)
    first[1:] = vertex[1:] != vertex[:-1]
    copies = np.where(~first)[0]
    if copies.size == 0:
        return (points, tets, np.arange(len(points), dtype=np.int64), 0, 0)

    new_index = np.where(first, vertex, -1)
    new_index[copies] = len(points) + np.arange(len(copies))
    out_tets = new_index[inverse].reshape(-1, 4).astype(np.int64)
    source = np.concatenate(
        [np.arange(len(points), dtype=np.int64), vertex[copies]])
    return (points[source], out_tets, source,
            int(len(np.unique(vertex[copies]))), int(len(copies)))


# ---------------------------------------------------------------------
# Closing what only just fails to close
# ---------------------------------------------------------------------
def _cells_at(starts: np.ndarray, cells: np.ndarray, node: int) -> np.ndarray:
    return cells[starts[node]:starts[node + 1]]


def _spanning(tets: np.ndarray, starts: np.ndarray, cells: np.ndarray,
              nodes: np.ndarray) -> int:
    """How many existing elements hold all of *nodes*.

    Asked of the elements at one of the nodes rather than of a table of
    every face in the mesh: a handful of contacts are being checked, and
    building a lookup over tens of millions of faces to answer that
    would cost more than the question.

    The count matters, not just the fact. A face a weld would add is
    welcome to exist already with **one** owner — that is a boundary
    face, and the new element is exactly what closes it. Two owners is
    an interior face, and a third owner would trade one non-manifold
    place for another.
    """
    total = 0
    for cell in _cells_at(starts, cells, int(nodes[0])):
        if np.isin(nodes, tets[cell]).all():
            total += 1
    return total


def _inside(points: np.ndarray, tet: np.ndarray, probe: np.ndarray) -> bool:
    """Whether *probe* lies strictly inside the tetrahedron *tet*."""
    a, b, c, d = points[tet]
    matrix = np.stack([b - a, c - a, d - a], axis=1)
    try:
        bary = np.linalg.solve(matrix, probe - a)
    except np.linalg.LinAlgError:
        return False
    return bool((bary > 1e-9).all() and bary.sum() < 1.0 - 1e-9)


def _overlaps(points: np.ndarray, tets: np.ndarray, neighbours: np.ndarray,
              candidate: np.ndarray) -> bool:
    """Whether a proposed element sits on top of material already there.

    Two cheap one-sided tests, not an exact intersection: the centroid
    of the candidate must not fall inside a neighbouring element, and no
    neighbouring node may fall inside the candidate. A weld fills a
    wedge of empty space; either test failing means the wedge was not
    empty, which is the case worth refusing.
    """
    centroid = points[candidate].mean(axis=0)
    for cell in neighbours:
        if _inside(points, tets[cell], centroid):
            return True
    nodes = np.setdiff1d(np.unique(tets[neighbours]), candidate)
    for node in nodes:
        if _inside(points, candidate, points[node]):
            return True
    return False


def _fan_angles(points: np.ndarray, a: int, b: int,
                apexes: np.ndarray) -> np.ndarray:
    """Angle of each apex about the axis ``a -> b``, in radians.

    The frame is arbitrary but shared by every apex on the edge, which
    is all the wedge test needs: it compares angles with each other and
    never with anything absolute.
    """
    axis = points[b] - points[a]
    axis = axis / np.linalg.norm(axis)
    reference = np.array([1.0, 0.0, 0.0])
    if abs(float(axis @ reference)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, reference)
    u /= np.linalg.norm(u)
    w = np.cross(axis, u)
    offset = points[np.asarray(apexes)] - points[a]
    offset = offset - np.outer(offset @ axis, axis)
    return np.arctan2(offset @ w, offset @ u)


def _weld_one_edge(points, tets, edge, face_ids, faces, owners, apexes,
                   starts, cells, limit):
    """Propose the elements that make one non-manifold edge manifold.

    An edge carrying ``2k`` boundary faces has ``k`` fans of material
    around it and ``k`` empty wedges between them. Filling ``k - 1`` of
    them — always the narrowest, leaving the widest open — joins the
    fans into one and drops the edge to the two boundary faces a
    manifold edge has.

    Returns a list of ``(nodes, source_cell)``, or ``None`` when the
    wedge structure cannot be read off the geometry.
    """
    a, b = int(edge[0]), int(edge[1])
    tri_apex = faces[face_ids].sum(axis=1) - a - b
    theta = _fan_angles(points, a, b, tri_apex)
    phi = _fan_angles(points, a, b, apexes[face_ids])
    # A face whose element's apex sits at the face's own angle gives no
    # answer to "which side is material": the element is flat against
    # the edge. Half a turn is not ambiguous — it is a fat element and
    # the side is still clear — so only zero, from either direction, is
    # refused.
    delta = np.mod(phi - theta, 2.0 * np.pi)
    if np.any(np.minimum(delta, 2.0 * np.pi - delta) < _ANGLE_EPS):
        return None

    order = np.argsort(theta)
    theta_s = theta[order]
    apex_s = tri_apex[order]
    owner_s = owners[face_ids[order]]
    # Material lies counter-clockwise of a face when its element's apex
    # is within half a turn counter-clockwise of it.
    material_ccw = delta[order] < np.pi
    if not np.all(material_ccw != np.roll(material_ccw, -1)):
        return None

    gaps = np.mod(np.roll(theta_s, -1) - theta_s, 2.0 * np.pi)
    empty = np.where(~material_ccw)[0]
    if len(empty) < 2:
        return None
    keep_open = empty[int(np.argmax(gaps[empty]))]

    proposals: List[Tuple[np.ndarray, int]] = []
    neighbours = np.union1d(_cells_at(starts, cells, a),
                            _cells_at(starts, cells, b))
    for sector in empty:
        if sector == keep_open:
            continue
        nxt = (sector + 1) % len(theta_s)
        nodes = np.array([a, b, int(apex_s[sector]), int(apex_s[nxt])],
                         dtype=np.int64)
        if len(np.unique(nodes)) != 4:
            return None
        # Oriented before it is measured, not after. The node order here
        # comes from the angular sweep and says nothing about winding,
        # and the shape measure reads a negatively wound element as
        # inverted — which would refuse every weld for a fault in the
        # bookkeeping rather than in the geometry.
        nodes = orient_positive(points, nodes[None, :])[0][0]
        volume = abs(float(signed_volumes(points, nodes[None, :])[0]))
        if volume <= limit * _MIN_RELATIVE_VOLUME or volume > limit:
            return None
        if float(shape_measure(points[nodes][None, ...])[0]) > _MAX_WELD_QUALITY:
            return None
        if _spanning(tets, starts, cells, nodes):
            return None
        if (_spanning(tets, starts, cells, nodes[[0, 2, 3]]) > 1
                or _spanning(tets, starts, cells, nodes[[1, 2, 3]]) > 1):
            return None
        if _overlaps(points, tets, neighbours, nodes):
            return None
        proposals.append((nodes, int(owner_s[sector])))
    return proposals


def _loops_of(link_edges: np.ndarray) -> Optional[List[np.ndarray]]:
    """Split the boundary of a vertex's link into ordered cycles.

    Returns ``None`` unless every vertex of the link boundary has
    exactly two neighbours, i.e. the boundary really is a union of
    simple closed curves. Anything else is a contact this repair does
    not claim to understand, and it is reported rather than guessed at.
    """
    nodes = np.unique(link_edges)
    local = np.searchsorted(nodes, link_edges)
    degree = np.bincount(local.ravel(), minlength=len(nodes))
    if np.any(degree != 2):
        return None

    neighbours: List[List[int]] = [[] for _ in nodes]
    for i, j in local:
        neighbours[i].append(j)
        neighbours[j].append(i)

    seen = np.zeros(len(nodes), dtype=bool)
    loops: List[np.ndarray] = []
    for start in range(len(nodes)):
        if seen[start]:
            continue
        loop = [start]
        seen[start] = True
        previous, current = start, neighbours[start][0]
        while current != start:
            seen[current] = True
            loop.append(current)
            first, second = neighbours[current]
            previous, current = current, (second if first == previous
                                          else first)
        loops.append(nodes[np.array(loop, dtype=np.int64)])
    return loops


def _loop_area(points: np.ndarray, loop: np.ndarray) -> float:
    """Area of the fan spanning *loop*, as a size for the opening."""
    centre = points[loop].mean(axis=0)
    spokes = points[loop] - centre
    nxt = np.roll(spokes, -1, axis=0)
    return float(0.5 * np.linalg.norm(np.cross(spokes, nxt), axis=1).sum())


def _weld_one_vertex(points, tets, vertex, face_ids, faces,
                     starts, cells, limit):
    """Propose the elements that make one pinched vertex manifold.

    The boundary faces at the vertex give the boundary of its link. When
    that boundary is two loops the link is an annulus — a passage
    through the wall closing to a point — and capping the smaller loop
    turns it into a disk, which is what a manifold vertex has. The cap's
    triangles are coned back to the vertex, so the elements added fill
    exactly the empty cone the loop bounds.

    Returns a list of ``(nodes, source_cell)``, or ``None``.
    """
    at_vertex = faces[face_ids]
    link_edges = at_vertex[at_vertex != vertex].reshape(-1, 2)
    loops = _loops_of(link_edges)
    if loops is None or len(loops) < 2:
        return None

    areas = [_loop_area(points, loop) for loop in loops]
    loop = loops[int(np.argmin(areas))]
    if len(loop) < 3:
        return None

    # A cap edge that already carries boundary faces would leave that
    # edge with more than two of them, trading one non-manifold place
    # for another. The loop's own edges are fine: each loses the face it
    # had through the vertex and gains the cap's.
    on_loop = {(min(int(p), int(q)), max(int(p), int(q)))
               for p, q in zip(loop, np.roll(loop, -1))}
    forbidden = set()
    for p in loop:
        for q in np.unique(tets[_cells_at(starts, cells, int(p))]):
            key = (min(int(p), int(q)), max(int(p), int(q)))
            if key not in on_loop and key[0] != key[1]:
                if _spanning(tets, starts, cells, np.array(key)):
                    forbidden.add(key)

    cap, residual = _earclip_triangulate(points[loop], loop, forbidden)
    if len(residual) or len(cap) != len(loop) - 2:
        return None

    neighbours = _cells_at(starts, cells, int(vertex))
    total = 0.0
    proposals: List[Tuple[np.ndarray, int]] = []
    for triangle in cap:
        nodes = np.array([int(vertex), *(int(t) for t in triangle)],
                         dtype=np.int64)
        if len(np.unique(nodes)) != 4:
            return None
        # See the edge weld: the cap's triangles come out of ear-clipping
        # with whichever winding the loop had, so orient before judging.
        nodes = orient_positive(points, nodes[None, :])[0][0]
        volume = abs(float(signed_volumes(points, nodes[None, :])[0]))
        total += volume
        if volume <= limit * _MIN_RELATIVE_VOLUME or volume > limit:
            return None
        if float(shape_measure(points[nodes][None, ...])[0]) > _MAX_WELD_QUALITY:
            return None
        if _spanning(tets, starts, cells, nodes):
            return None
        if _spanning(tets, starts, cells, nodes[1:]) > 1:
            return None
        if _overlaps(points, tets, neighbours, nodes):
            return None
        proposals.append((nodes, int(neighbours[0])))
    return proposals


def _weld_pass(points: np.ndarray, tets: np.ndarray, options: RepairOptions,
               table=None):
    """One round of welding. Returns ``(tets, sources, report fragments)``.

    *table* is the boundary table if the caller already has one. It
    always does, because it had to look at the boundary to know there
    was anything to repair, and building it twice on a mesh of millions
    of elements is the kind of waste that turns a repair into a wait.
    """
    faces, owners, apexes = (boundary_table(tets) if table is None
                             else table)
    starts, cells = node_cells(tets, len(points))
    face_starts, face_of_node = node_cells(faces, len(points))
    volumes = np.abs(signed_volumes(points, tets))

    new_nodes: List[np.ndarray] = []
    new_source: List[int] = []
    welded_edges = welded_vertices = 0
    failed_edges = failed_vertices = 0
    added_volume = 0.0

    def local_limit(cell_ids: np.ndarray) -> float:
        """The largest element a weld at this contact may add."""
        return float(volumes[cell_ids].mean()) * options.max_weld_volume

    # --- edges ------------------------------------------------------
    uniq, inv, counts = boundary_edge_table(faces)
    face_of_edge = np.repeat(np.arange(len(faces), dtype=np.int64), 3)
    order = np.argsort(inv, kind="stable")
    edge_starts = np.searchsorted(inv[order], np.arange(len(counts)))
    touched = set()
    for k in np.where(counts > 2)[0]:
        ids = face_of_edge[order[edge_starts[k]:edge_starts[k] + counts[k]]]
        edge = uniq[k]
        if int(edge[0]) in touched or int(edge[1]) in touched:
            failed_edges += 1
            continue
        neighbours = np.union1d(_cells_at(starts, cells, int(edge[0])),
                                _cells_at(starts, cells, int(edge[1])))
        proposals = _weld_one_edge(points, tets, edge, ids, faces, owners,
                                   apexes, starts, cells,
                                   local_limit(neighbours))
        if not proposals:
            failed_edges += 1
            continue
        welded_edges += 1
        touched.update(int(v) for v in np.unique(np.stack(
            [nodes for nodes, _ in proposals])))
        for nodes, source in proposals:
            new_nodes.append(nodes)
            new_source.append(source)
            added_volume += abs(
                float(signed_volumes(points, nodes[None, :])[0]))

    # --- vertices ---------------------------------------------------
    # Only where no edge weld has already changed the neighbourhood:
    # the faces and owners in hand describe the mesh as it was, and a
    # second repair judged against stale tables is a repair judged
    # against a mesh that no longer exists. What is left waits for the
    # next pass.
    for vertex in pinched_vertices(faces):
        vertex = int(vertex)
        if vertex in touched:
            continue
        ids = _cells_at(face_starts, face_of_node, vertex)
        neighbours = _cells_at(starts, cells, vertex)
        proposals = _weld_one_vertex(points, tets, vertex, ids, faces,
                                     starts, cells, local_limit(neighbours))
        if not proposals:
            failed_vertices += 1
            continue
        welded_vertices += 1
        touched.update(int(v) for v in np.unique(np.stack(
            [nodes for nodes, _ in proposals])))
        for nodes, source in proposals:
            new_nodes.append(nodes)
            new_source.append(source)
            added_volume += abs(
                float(signed_volumes(points, nodes[None, :])[0]))

    if not new_nodes:
        return tets, np.zeros(0, dtype=np.int64), (0, 0, 0, 0.0,
                                                   failed_edges,
                                                   failed_vertices)
    # Oriented here rather than left to the caller's orientation pass:
    # a repair that hands back inverted elements is only sound for
    # callers that happen to run one.
    added, _flipped = orient_positive(points, np.stack(new_nodes))
    grown = np.concatenate([tets, added])
    return (grown, np.array(new_source, dtype=np.int64),
            (welded_edges, welded_vertices, len(new_nodes), added_volume,
             failed_edges, failed_vertices))



# ---------------------------------------------------------------------
# Plugging a perforation
# ---------------------------------------------------------------------
def boundary_genus(faces: np.ndarray) -> Optional[int]:
    """Handles on the boundary *faces*, or ``None`` if it is not closed.

    From the Euler characteristic of the whole boundary, ``V - E + F``,
    which for a closed orientable surface is ``2 * sheets - 2 * genus``.
    A handle is a passage you could thread a loop through, and counting
    them is how a plug proves it did something: filling an anatomical
    crevice leaves the genus alone, filling a perforation drops it by
    one.
    """
    if len(faces) == 0:
        return None
    vertices = len(np.unique(faces))
    edges_uniq, _inv, counts = boundary_edge_table(faces)
    if np.any(counts != 2):
        return None                     # not a closed manifold surface
    chi = vertices - len(edges_uniq) + len(faces)
    sheets = boundary_sheets(faces)
    if (2 * sheets - chi) % 2:
        return None                     # non-orientable: not our case
    return int((2 * sheets - chi) // 2)


def _facing_pairs(points: np.ndarray, faces: np.ndarray, apexes: np.ndarray,
                  limit: float):
    """Boundary faces that look at each other across a gap, narrowest first.

    Two tests, and both matter. The normals must oppose, or the faces
    are not two sides of the same passage. And each face's centre must
    lie on the *outward* side of the other, which is what tells a
    passage from a wall: across a wall the far face sits behind this
    one's material, not in front of it.
    """
    from scipy.spatial import cKDTree

    p0, p1, p2 = points[faces[:, 0]], points[faces[:, 1]], points[faces[:, 2]]
    centre = (p0 + p1 + p2) / 3.0
    normal = np.cross(p1 - p0, p2 - p0)
    inward = np.einsum("ij,ij->i", normal, points[apexes] - centre)
    normal[inward > 0] *= -1.0
    normal /= np.maximum(np.linalg.norm(normal, axis=1), 1e-300)[:, None]

    tree = cKDTree(centre)
    rows: List[Tuple[float, int, int]] = []
    for i, others in enumerate(tree.query_ball_point(centre, r=limit)):
        for j in others:
            if j <= i:
                continue
            if float(normal[i] @ normal[j]) > -0.5:
                continue
            offset = centre[j] - centre[i]
            if float(normal[i] @ offset) <= 0.0:
                continue
            if float(normal[j] @ -offset) <= 0.0:
                continue
            rows.append((float(np.linalg.norm(offset)), i, j))
    rows.sort()
    return rows, centre, normal


def _prism(points: np.ndarray, near: np.ndarray,
           far: np.ndarray) -> np.ndarray:
    """Three tetrahedra filling the prism between two triangles.

    The far triangle is rotated so that each of its nodes sits opposite
    the nearest node of the near one; without that the prism is built
    with a twist in it and the three elements it splits into overlap.
    """
    cost = np.linalg.norm(points[near][:, None, :] - points[far][None, :, :],
                          axis=2)
    best, order = None, None
    for shift in range(3):
        for flip in (far, far[::-1]):
            candidate = np.roll(flip, shift)
            total = float(sum(cost[k, list(far).index(candidate[k])]
                              for k in range(3)))
            if best is None or total < best:
                best, order = total, candidate
    a, b, c = near
    d, e, f = order
    return np.array([[a, b, c, d], [b, c, d, e], [c, d, e, f]],
                    dtype=np.int64)


def _plug_pass(points: np.ndarray, tets: np.ndarray, options: RepairOptions,
               on_status: Optional[Callable[[str], None]]):
    """Plug perforations, keeping only the plugs that remove a handle.

    Each candidate is filled, the genus is measured again, and the fill
    is undone unless it fell. That is the whole safety argument: the
    size bound keeps the search away from anatomy, and the genus check
    means a plug has to *prove* it closed a passage rather than merely
    filling a dent.
    """
    faces, owners, apexes = boundary_table(tets)
    genus = boundary_genus(faces)
    if genus is None or genus == 0:
        return tets, np.zeros(0, dtype=np.int64), 0, genus, genus

    volumes = np.abs(signed_volumes(points, tets))
    scale = float(np.cbrt(volumes.mean()) * 6.0 ** (1.0 / 3.0))
    limit = options.max_plug_gap * scale
    rows, _centre, _normal = _facing_pairs(points, faces, apexes, limit)
    if not rows:
        return tets, np.zeros(0, dtype=np.int64), 0, genus, genus

    starts, cells = node_cells(tets, len(points))
    added: List[np.ndarray] = []
    sources: List[int] = []
    plugged = 0
    current = genus
    for _gap, i, j in rows:
        if current == 0:
            break
        candidate = _prism(points, faces[i], faces[j])
        if len(np.unique(candidate)) != 6:
            continue
        candidate, _flipped = orient_positive(points, candidate)
        worst = float(shape_measure(points[candidate]).max())
        smallest = float(np.abs(signed_volumes(points, candidate)).min())
        if worst > _MAX_WELD_QUALITY or smallest <= 0.0:
            continue
        if any(_spanning(tets, starts, cells, quad) for quad in candidate):
            continue
        # The elements around the two faces, which means around their
        # nodes. ``owners`` holds cell indices and this wants nodes, so
        # passing one for the other reads the adjacency table past its
        # end — caught on the example ventricle as node 123632 of 66824,
        # and latent only because no candidate had ever got this far.
        neighbours = np.unique(np.concatenate(
            [_cells_at(starts, cells, int(node))
             for node in np.concatenate([faces[i], faces[j]])]))
        if any(_overlaps(points, tets, neighbours, quad)
               for quad in candidate):
            continue

        trial = np.concatenate([tets, candidate])
        after = boundary_genus(boundary_table(trial)[0])
        if after is None or after >= current:
            continue                    # filled a dent, not a passage
        tets = trial
        starts, cells = node_cells(tets, len(points))
        faces, owners, apexes = boundary_table(tets)
        added.append(candidate)
        sources.extend([int(owners[i])] * 3)
        plugged += 1
        current = after
        if on_status is not None:
            on_status(f"Plugged a perforation {_gap:.3g} across; "
                      f"the boundary now has {after} handle"
                      + ("" if after == 1 else "s") + ".")

    source = (np.array(sources, dtype=np.int64) if sources
              else np.zeros(0, dtype=np.int64))
    return tets, source, plugged, genus, current


def _why_nothing_was_plugged(tets: np.ndarray,
                             genus: Optional[int]) -> str:
    """The reason a requested plugging pass plugged nothing."""
    if genus is None:
        faces = boundary_table(tets)[0]
        left = len(non_manifold_edges(faces)) + len(pinched_vertices(faces))
        return ("no perforation plugged: the boundary is still not "
                f"manifold ({left} contact" + ("" if left == 1 else "s")
                + "), so its handles cannot be counted")
    if genus == 0:
        return "no perforation to plug: the boundary has no handles"
    return (f"no perforation plugged: none of the {genus} handle"
            + ("" if genus == 1 else "s")
            + " is a gap narrow enough")


# ---------------------------------------------------------------------
# The repair
# ---------------------------------------------------------------------
def repair(points: np.ndarray,
           tets: np.ndarray,
           options: Optional[RepairOptions] = None,
           on_status: Optional[Callable[[str], None]] = None):
    """Make the boundary of a tetrahedral volume manifold.

    Returns ``(points, tets, source_point, source_cell, report)``.
    ``source_point`` and ``source_cell`` index the *inputs*, so a caller
    carries every field by selection: a split node's copies take the
    original node's value, and a welded element takes the value of the
    element it plugs against, which is the only element it touches.

    Neither input array is modified.
    """
    options = options or RepairOptions()
    options.validate()

    points = np.asarray(points, dtype=float)
    tets = np.asarray(tets, dtype=np.int64)
    source_point = np.arange(len(points), dtype=np.int64)
    source_cell = np.arange(len(tets), dtype=np.int64)
    report = RepairReport()

    for _ in range(options.max_passes):
        moved = False
        report.passes += 1

        # Ask what is wrong before doing anything about it. A sound
        # boundary is the common case — a mesh straight out of the
        # remesher has one — and this is what keeps the repair from
        # building a corner graph over millions of elements to discover
        # there was nothing to separate.
        table = boundary_table(tets)
        if (len(non_manifold_edges(table[0])) == 0
                and len(pinched_vertices(table[0])) == 0):
            report.unrepaired_edges = report.unrepaired_vertices = 0
            break

        if options.split_touching:
            points, tets, mapping, nodes, copies = split_touching(points, tets)
            if copies:
                source_point = source_point[mapping]
                report.split_vertices += nodes
                report.split_copies += copies
                moved = True
                if on_status is not None:
                    on_status(f"Separated {nodes} touching contact"
                              + ("" if nodes == 1 else "s") + ".")
                # The split renumbered nodes, so the table describes a
                # mesh that no longer exists. What is left to weld waits
                # for the next pass rather than being judged against it.
                table = None

        report.unrepaired_edges = report.unrepaired_vertices = 0
        if options.weld_pinholes and table is not None:
            tets, sources, counts = _weld_pass(points, tets, options, table)
            (edges, vertices, added, volume,
             failed_edges, failed_vertices) = counts
            report.unrepaired_edges = failed_edges
            report.unrepaired_vertices = failed_vertices
            if added:
                source_cell = np.concatenate(
                    [source_cell, source_cell[sources]])
                report.welded_edges += edges
                report.welded_vertices += vertices
                report.added_cells += added
                report.added_volume += volume
                moved = True
                if on_status is not None:
                    on_status(f"Welded {edges + vertices} pinhole"
                              + ("" if edges + vertices == 1 else "s")
                              + f" shut with {added} element"
                              + ("" if added == 1 else "s") + ".")
        if not moved:
            break

    # Plugging comes last, and only once: it asks what the boundary's
    # genus is, and that question has no answer until the boundary is
    # manifold, which is what everything above was for.
    if options.plug_perforations:
        tets, sources, plugged, before_genus, after_genus = _plug_pass(
            points, tets, options, on_status)
        report.genus_before, report.genus_after = before_genus, after_genus
        report.plugged = plugged
        if not plugged:
            report.plug_note = _why_nothing_was_plugged(tets, before_genus)
        if plugged:
            source_cell = np.concatenate([source_cell, source_cell[sources]])
            report.added_cells += len(sources)
            report.added_volume += float(np.abs(
                signed_volumes(points, tets[-len(sources):])).sum())

    return points, tets, source_point, source_cell, report


__all__ = ["RepairOptions", "RepairReport", "repair", "split_touching",
           "pinched_vertices", "non_manifold_edges", "boundary_sheets",
           "boundary_edge_table"]
