"""
volume_quality
==============
Repair the *shape* of a tetrahedral volume's worst elements, in place.

The volume post-processor changes element sizes by remeshing the whole
volume; the cleaner repairs connectivity and moves nothing. Neither
fixes the handful of badly shaped elements a mesh carries — the caps and
slivers a segmentation leaves behind — without disturbing the rest, and
that is what this does.

How shape is measured
---------------------
``q = 1 - sqrt(2) * 6V / rms_edge^3``, where ``rms_edge`` is the root
mean square of the six edge lengths. It is 0 for a regular tetrahedron,
approaches 1 as the element flattens, and is reported as 2 for one that
has turned inside out. **Lower is better**, which is the opposite of the
convention most quality measures use, so every threshold here is an
upper bound.

The scale is worth knowing before choosing a threshold. On the example
ventricle the mean is 0.22 and 48% of the elements are above 0.2, so a
threshold of 0.2 does not mean "the bad ones" — it means almost all of
them. That matters: a repair pass turned loose on a whole mesh shrinks
it. Measured on a 14-million-element remesh, smoothing everything above
0.2 moved all 2.7 million vertices, by 0.125 mm on average, and cost 15%
of the myocardial volume. The default here is 0.8, where the example
ventricle has 889 elements of 290,508 — 0.31% — which is what "the bad
ones" should mean.

What it does, in order
----------------------
1. **Flips.** Connectivity only, no vertex moves: a node whose only four
   elements span five nodes collapses into one element (4-to-1), and
   three elements around an interior edge become two (3-to-2). Each is
   applied only where the worst element involved gets better.
2. **Smoothing.** Quality-guarded Taubin over the bad elements and two
   layers around them: a forward pass toward the neighbour average and a
   slightly stronger backward one, which is what keeps Laplacian
   smoothing from shrinking what it smooths. A node's move is kept only
   if its own worst element does not get worse.
3. **Shifting.** Gradient descent on ``sum(10^(q + 1 - thr))`` over the
   elements at each bad node, with a backtracking step. The power makes
   the worst element dominate the sum, so the step goes where the damage
   is rather than averaging it away.

Why the boundary may move, and how far
--------------------------------------
On the example ventricle 81% of the elements above 0.8 have at least one
node on the wall and 17% have all four, so a pass that froze the
boundary could not touch most of what it exists to fix. Boundary nodes
therefore move, but only in the local tangent plane: the component of
every step along the surface normal is projected out, so the wall stays
where it is to first order and any volume drift is second order. The
report gives the largest distance any boundary node actually travelled,
because "to first order" is an argument, not a measurement.

Two kinds of node never move at all: nodes on a **feature edge**, where
two boundary faces meet at more than ``feature_angle``, and nodes where
the boundary is not manifold — an open or non-manifold edge, or a pinch.
The first is what keeps the rim of a valve opening a rim instead of
letting it round away. The second is because a tangent plane is not
defined there: at a pinch the surface passes through the node twice, and
the average of the two sheets points where neither of them does. Run the
cleaner first and there will be none of the second kind left.

Simultaneous, not sequential
----------------------------
Every node's move is computed from the same starting positions and
applied together, rather than each node seeing the moves already made.
That is what makes the pass fast enough in array form to run on a mesh
of millions of elements, and it is why acceptance is checked twice: once
per node against its own elements, and again for the round as a whole,
which is undone if the count of bad elements did not fall.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np
import pyvista as pv

from ccdaf.core.volume_mesh import (
    EDGE_NODES, TETRA, boundary_table, node_cells, orient_positive,
    shape_measure, signed_volumes, sorted_unique_rows, tetrahedra,
    unique_rows, validate_tetrahedral,
)
# The same test the cleaner reports with, so the panel cannot say a node
# is pinched while this pass treats it as ordinary wall.
from ccdaf.core.volume_repair import pinched_vertices

#: The shape measure's threshold: elements above it are the ones to fix.
#: See the module docstring for why it is not 0.2.
QUALITY_THRESHOLD: float = 0.8

#: Taubin's pair: a forward step toward the neighbour average, then a
#: backward one 2.5% stronger. The imbalance is the whole trick — equal
#: steps would undo the smoothing, a forward step alone shrinks the mesh.
_SMOOTH_FORWARD: float = 0.14
_SMOOTH_BACKWARD: float = -1.025

#: Angle between two boundary faces above which their shared edge is a
#: feature to be held, in degrees. Thirty keeps the rim of a valve
#: opening and lets ordinary curvature through.
_FEATURE_ANGLE: float = 30.0

#: How many times the gradient step is halved before a node gives up.
_BACKTRACK = 12

#: How often the movers re-check which nodes still have a bad element.
#: Every pass would be honest and wasteful; never would leave the whole
#: neighbourhood being polished long after it was sound.
_REFRESH = 4

def quality(points: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """Shape measure of every tetrahedron: 0 is regular, 2 is inverted."""
    if len(tets) == 0:
        return np.zeros(0, dtype=float)
    return _quality_of(np.asarray(points, dtype=float)[tets])


#: The measure itself lives in :mod:`volume_mesh`, so that the boundary
#: repair can refuse to weld in an element this pass would then have to
#: come and fix.
_quality_of = shape_measure


@dataclass
class QualityOptions:
    """What the repair may do, and how hard it may try."""

    #: Elements above this are the ones repaired. Not 0.2 — see the
    #: module docstring.
    threshold: float = QUALITY_THRESHOLD
    #: Apply the connectivity flips.
    flips: bool = True
    #: Taubin iterations per round, over the bad elements and two
    #: layers around them.
    smooth_iterations: int = 20
    #: Gradient-descent iterations per round.
    shift_iterations: int = 20
    #: One gradient step, as a fraction of the shortest edge at the
    #: node. Small because the step is taken by every bad node at once.
    step_fraction: float = 0.05
    #: Let boundary nodes slide in their own tangent plane. Off freezes
    #: the wall vertex for vertex, and then only flips and interior
    #: nodes can improve an element that touches it.
    slide_boundary: bool = True
    #: Angle in degrees above which a boundary edge is a feature whose
    #: nodes are held. Ignored while the boundary is frozen.
    feature_angle: float = _FEATURE_ANGLE
    #: The furthest a node may end up from where it started, as a
    #: fraction of the shortest edge it had there. Without it a node is
    #: free to slide a little on every one of forty iterations, and a
    #: tangent plane is only the wall for small steps.
    max_travel: float = 0.3
    #: Rounds of flip / smooth / shift. A round that does not reduce the
    #: count of bad elements is undone and ends the pass.
    max_rounds: int = 3

    def validate(self) -> None:
        if not 0.0 < self.threshold < 2.0:
            raise ValueError(
                "the quality threshold must be between 0 and 2 "
                "(0 is a regular element, 1 a flat one)")
        if self.step_fraction <= 0.0:
            raise ValueError("the step fraction must be positive")
        if self.max_travel <= 0.0:
            raise ValueError("the travel limit must be positive")
        if min(self.smooth_iterations, self.shift_iterations) < 0:
            raise ValueError("iteration counts must not be negative")
        if self.max_rounds < 1:
            raise ValueError("the repair needs at least one round")


@dataclass
class QualityReport:
    """What the repair achieved, and what it cost."""

    cells_before: int = 0
    cells_after: int = 0
    bad_before: int = 0
    bad_after: int = 0
    worst_before: float = 0.0
    worst_after: float = 0.0
    mean_before: float = 0.0
    mean_after: float = 0.0
    flips_4_to_1: int = 0
    flips_3_to_2: int = 0
    #: Nodes the flips took out of the connectivity and that were
    #: therefore dropped. A 4-to-1 flip removes exactly one.
    removed_nodes: int = 0
    moved_nodes: int = 0
    max_shift: float = 0.0
    max_boundary_shift: float = 0.0
    volume_before: float = 0.0
    volume_after: float = 0.0
    rounds: int = 0
    threshold: float = QUALITY_THRESHOLD

    @property
    def changed(self) -> bool:
        return bool(self.flips_4_to_1 or self.flips_3_to_2
                    or self.moved_nodes)

    @property
    def volume_change(self) -> float:
        """Signed relative change in total volume."""
        if self.volume_before == 0.0:
            return 0.0
        return (self.volume_after - self.volume_before) / self.volume_before

    def summary(self) -> str:
        if not self.changed:
            if self.bad_before == 0:
                return (f"No element is above {self.threshold:g}; "
                        f"nothing to repair.")
            return (f"{self.bad_before} element"
                    + ("" if self.bad_before == 1 else "s")
                    + f" above {self.threshold:g} could not be improved.")
        bits: List[str] = []
        flips = self.flips_4_to_1 + self.flips_3_to_2
        if flips:
            bits.append(f"{flips} flip" + ("" if flips == 1 else "s"))
        if self.removed_nodes:
            bits.append(f"removed {self.removed_nodes} node"
                        + ("" if self.removed_nodes == 1 else "s"))
        if self.moved_nodes:
            bits.append(f"moved {self.moved_nodes} node"
                        + ("" if self.moved_nodes == 1 else "s")
                        + f" (up to {self.max_shift:.3g}, "
                        + f"{self.max_boundary_shift:.3g} on the wall)")
        return (f"Elements above {self.threshold:g}: "
                f"{self.bad_before} → {self.bad_after}; "
                f"worst {self.worst_before:.3f} → {self.worst_after:.3f}, "
                f"mean {self.mean_before:.3f} → {self.mean_after:.3f}; "
                + ", ".join(bits)
                + f"; volume {self.volume_change * 100.0:+.4f}%.")


# ---------------------------------------------------------------------
# Neighbourhoods
# ---------------------------------------------------------------------
def _gather(starts: np.ndarray, items: np.ndarray, nodes: np.ndarray):
    """Every ``items`` entry of every node, plus which node each is for.

    Returns ``(values, counts, segments)``: the flattened adjacency of
    the given nodes, how many each has, and where each node's run
    begins. Built by arithmetic rather than by a loop over nodes,
    because the caller does this on every iteration.
    """
    counts = (starts[nodes + 1] - starts[nodes]).astype(np.int64)
    total = int(counts.sum())
    segments = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    if total == 0:
        return np.zeros(0, dtype=np.int64), counts, segments
    base = np.repeat(starts[nodes], counts)
    offset = np.arange(total) - np.repeat(segments, counts)
    return items[base + offset], counts, segments


def _grow(tets: np.ndarray, starts: np.ndarray, cells: np.ndarray,
          seed_cells: np.ndarray, layers: int) -> np.ndarray:
    """The nodes of *seed_cells*, grown by *layers* rings of elements."""
    nodes = np.unique(tets[seed_cells])
    for _ in range(layers):
        neighbours, _counts, _segments = _gather(starts, cells, nodes)
        nodes = np.unique(tets[np.unique(neighbours)])
    return nodes


def _rings(tets: np.ndarray, starts: np.ndarray, cells: np.ndarray,
           nodes: np.ndarray):
    """Who neighbours each of *nodes*, as ``(neighbours, valence, rings)``.

    Built for the nodes asked about, not for the mesh. The whole-mesh
    version this replaced sorted both directions of every element edge —
    168 million rows on an 18-million-element mesh — to answer a
    question about the few thousand nodes next to a badly shaped
    element. Measured at 2.35 million elements it cost 13.1 s of a
    24.7 s round, for adjacency of 451,484 nodes of which 1,688 were
    used.
    """
    incident, counts, _segments = _gather(starts, cells, nodes)
    if len(incident) == 0:
        return (np.zeros(0, dtype=np.int64), np.zeros(len(nodes), dtype=np.int64),
                np.zeros(len(nodes), dtype=np.int64))
    owner = np.repeat(np.arange(len(nodes), dtype=np.int64), counts * 4)
    neighbour = tets[incident].ravel()
    keep = neighbour != np.repeat(nodes, counts * 4)
    unique_pairs = unique_rows(
        np.stack([owner[keep], neighbour[keep]], axis=1))[0]
    valence = np.bincount(unique_pairs[:, 0],
                          minlength=len(nodes)).astype(np.int64)
    rings = np.zeros(len(nodes), dtype=np.int64)
    np.cumsum(valence[:-1], out=rings[1:])
    return np.ascontiguousarray(unique_pairs[:, 1]), valence, rings


# ---------------------------------------------------------------------
# The boundary: normals, and what must not move
# ---------------------------------------------------------------------
@dataclass
class _Boundary:
    """Which nodes are on the wall, where it faces, and what is held."""

    is_boundary: np.ndarray
    normals: np.ndarray
    held: np.ndarray
    faces: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), int))


def _boundary_frame(points: np.ndarray, tets: np.ndarray,
                    feature_angle: float) -> _Boundary:
    """Outward normals per boundary node, and the nodes that must not move.

    Each boundary face is oriented by its element's apex — the material
    is on the apex side, so the outward normal is the one pointing away
    from it — which avoids needing a consistently wound surface.

    A node is held when it sits on an edge where two faces meet at more
    than *feature_angle*, where the boundary is not manifold, or where
    the normals of its own faces disagree so strongly that no single
    tangent plane represents them.
    """
    n_points = len(points)
    faces, _owners, apexes = boundary_table(tets)
    is_boundary = np.zeros(n_points, dtype=bool)
    held = np.zeros(n_points, dtype=bool)
    normals = np.zeros((n_points, 3), dtype=float)
    if len(faces) == 0:
        return _Boundary(is_boundary, normals, held)

    is_boundary[np.unique(faces)] = True
    p0, p1, p2 = points[faces[:, 0]], points[faces[:, 1]], points[faces[:, 2]]
    face_normal = np.cross(p1 - p0, p2 - p0)
    inward = np.einsum("ij,ij->i",
                       face_normal, points[apexes] - (p0 + p1 + p2) / 3.0)
    face_normal[inward > 0.0] *= -1.0          # point away from the material

    for k in range(3):
        np.add.at(normals, faces[:, k], face_normal)   # area weighted
    lengths = np.linalg.norm(normals, axis=1)
    safe = lengths > 0.0
    normals[safe] /= lengths[safe][:, None]
    held |= is_boundary & ~safe                # no usable tangent plane

    # A pinch has no tangent plane either: the surface passes through
    # the node twice, and averaging the two sheets gives a direction
    # that belongs to neither. Held by the exact test rather than by how
    # much the normals happen to cancel — measured on the example
    # ventricle, its 17 pinches read between 0.10 and 0.42 on that
    # ratio while ordinary sharp corners of the wall go down to 0.085,
    # so no threshold on it separates the two.
    held[pinched_vertices(faces)] = True

    unit = face_normal / np.maximum(
        np.linalg.norm(face_normal, axis=1), 1e-300)[:, None]
    pairs = faces[:, [[0, 1], [1, 2], [0, 2]]].reshape(-1, 2)
    uniq, inv, counts = sorted_unique_rows(pairs)
    inv = inv.ravel()
    face_of_edge = np.repeat(np.arange(len(faces), dtype=np.int64), 3)
    order = np.argsort(inv, kind="stable")
    starts = np.searchsorted(inv[order], np.arange(len(counts)))

    # Non-manifold or open edges: nothing to project onto, so held.
    odd = np.where(counts != 2)[0]
    if odd.size:
        held[np.unique(uniq[odd])] = True

    shared = np.where(counts == 2)[0]
    if shared.size:
        first = face_of_edge[order[starts[shared]]]
        second = face_of_edge[order[starts[shared] + 1]]
        cosine = np.einsum("ij,ij->i", unit[first], unit[second])
        sharp = cosine < np.cos(np.radians(feature_angle))
        if sharp.any():
            held[np.unique(uniq[shared][sharp])] = True

    return _Boundary(is_boundary, normals, held, faces)


def _project(displacement: np.ndarray, nodes: np.ndarray,
             boundary: _Boundary, slide: bool) -> np.ndarray:
    """Remove what a step would do to the wall, and hold what is held."""
    out = displacement.copy()
    on_wall = boundary.is_boundary[nodes]
    if slide:
        normal = boundary.normals[nodes]
        along = np.einsum("ij,ij->i", out, normal)
        out[on_wall] -= (along[:, None] * normal)[on_wall]
    else:
        out[on_wall] = 0.0
    out[boundary.held[nodes]] = 0.0
    return out


# ---------------------------------------------------------------------
# Measuring a move before making it
# ---------------------------------------------------------------------
def _incident(tets: np.ndarray, starts: np.ndarray, cells: np.ndarray,
              nodes: np.ndarray):
    """The ``(node, element)`` pairs at *nodes*, and where each run starts."""
    incident, counts, segments = _gather(starts, cells, nodes)
    slot = np.argmax(tets[incident] == np.repeat(nodes, counts)[:, None],
                     axis=1)
    return incident, slot, counts, segments


def _quality_with(points: np.ndarray, tets: np.ndarray, incident: np.ndarray,
                  slot: np.ndarray, counts: np.ndarray,
                  candidate: np.ndarray) -> np.ndarray:
    """Quality of every incident element, each with its own node moved."""
    corners = points[tets[incident]]
    corners[np.arange(len(incident)), slot] = np.repeat(
        candidate, counts, axis=0)
    return _quality_of(corners)


def _worst_per_node(values: np.ndarray, counts: np.ndarray,
                    segments: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return np.full(len(counts), 2.0)
    return np.maximum.reduceat(values, segments)


def _functional(values: np.ndarray, segments: np.ndarray,
                threshold: float) -> np.ndarray:
    """``sum(10^(q + 1 - thr))``: the worst element dominates the sum."""
    if len(values) == 0:
        return np.zeros(len(segments))
    return np.add.reduceat(
        np.power(10.0, np.clip(values + 1.0 - threshold, None, 30.0)),
        segments)


# ---------------------------------------------------------------------
# Flips: connectivity only
# ---------------------------------------------------------------------
def _spans(tets: np.ndarray, starts: np.ndarray, cells: np.ndarray,
           nodes: np.ndarray) -> bool:
    """Whether some element already holds all of *nodes*."""
    for cell in cells[starts[int(nodes[0])]:starts[int(nodes[0]) + 1]]:
        if np.isin(nodes, tets[cell]).all():
            return True
    return False


def _flip_4_to_1(points: np.ndarray, tets: np.ndarray, measure: np.ndarray,
                 starts: np.ndarray, cells: np.ndarray,
                 on_wall: np.ndarray, threshold: float):
    """Replace the four elements at an interior node with one.

    The node has to be interior. On the wall its four elements do not
    surround it, so removing it would take a corner off the boundary —
    a shape change dressed up as a connectivity change.

    Returns ``(dropped, added, sources)``.
    """
    valence = starts[1:] - starts[:-1]
    candidates = np.unique(tets[measure > threshold])
    candidates = candidates[(valence[candidates] == 4) & ~on_wall[candidates]]

    dropped: set = set()
    added: List[np.ndarray] = []
    sources: List[int] = []
    for node in candidates:
        group = cells[starts[node]:starts[node + 1]]
        if dropped.intersection(group.tolist()):
            continue
        corners = np.unique(tets[group])
        if len(corners) != 5:
            continue
        # Oriented before it is judged: the four remaining nodes come
        # out of np.unique in index order, which says nothing about
        # their winding, and the measure reads a negatively wound
        # element as inverted. That is a fact about the node order, not
        # about the shape.
        quad = corners[corners != node]
        quad, _flipped = orient_positive(points, quad[None, :])
        quad = quad[0]
        replacement = float(_quality_of(points[quad][None, ...])[0])
        if replacement >= float(measure[group].max()) or replacement >= 1.0:
            continue
        if _spans(tets, starts, cells, quad):
            continue
        dropped.update(group.tolist())
        added.append(quad)
        sources.append(int(group[0]))
    return dropped, added, sources


def _cells_with_edge(tets, starts, cells, u: int, v: int) -> np.ndarray:
    at_u = cells[starts[u]:starts[u + 1]]
    return at_u[(tets[at_u] == v).any(axis=1)]


def _flip_3_to_2(points: np.ndarray, tets: np.ndarray, measure: np.ndarray,
                 starts: np.ndarray, cells: np.ndarray, threshold: float):
    """Replace three elements around an interior edge with two.

    The edge must be surrounded: each of the three faces through it is
    shared by two of the three elements, which is what makes the ring
    closed and the two replacements fill exactly the same space. The
    three remaining nodes must also each project onto the edge between
    its ends, or the replacement pair is a worse shape than what it
    replaced however the measure reads.

    Returns ``(dropped, added, sources)``.
    """
    dropped: set = set()
    added: List[np.ndarray] = []
    sources: List[int] = []
    seen: set = set()
    for cell in np.where(measure > threshold)[0]:
        if cell in dropped:
            continue
        for i, j in EDGE_NODES:
            u, v = int(tets[cell, i]), int(tets[cell, j])
            key = (min(u, v), max(u, v))
            if key in seen:
                continue
            seen.add(key)
            ring = _cells_with_edge(tets, starts, cells, u, v)
            if len(ring) != 3 or dropped.intersection(ring.tolist()):
                continue
            corners = np.unique(tets[ring])
            if len(corners) != 5:
                continue
            base = corners[(corners != u) & (corners != v)]

            # Closed ring: every face through the edge is shared.
            through = [tuple(sorted((u, v, int(w))))
                       for c in ring for w in tets[c]
                       if int(w) not in (u, v)]
            if sorted({f: through.count(f) for f in through}.values()) != [2, 2, 2]:
                continue

            axis = points[v] - points[u]
            length = float(np.dot(axis, axis))
            if length <= 0.0:
                continue
            along = (points[base] - points[u]) @ axis / length
            if not np.all((along > 0.0) & (along < 1.0)):
                continue

            pair = np.array([[u, *base.tolist()], [v, *base.tolist()]],
                            dtype=np.int64)
            # One of these two is always wound the other way: u and v
            # lie on opposite sides of the base triangle. Orient first,
            # or every candidate reads as inverted and no flip ever
            # fires — which is exactly what happened before this line.
            pair, _flipped = orient_positive(points, pair)
            replacement = _quality_of(points[pair])
            if (float(replacement.max()) >= float(measure[ring].max())
                    or float(replacement.max()) >= 1.0):
                continue
            if any(_spans(tets, starts, cells, q) for q in pair):
                continue
            dropped.update(ring.tolist())
            added.extend(pair)
            sources.extend([int(ring[0])] * 2)
            break
    return dropped, added, sources


def _apply_flips(tets: np.ndarray, source_cell: np.ndarray, dropped: set,
                 added: List[np.ndarray], sources: List[int]):
    """Remove the flipped elements and append their replacements."""
    if not dropped:
        return tets, source_cell
    keep = np.ones(len(tets), dtype=bool)
    keep[np.fromiter(dropped, dtype=np.int64, count=len(dropped))] = False
    kept_source = source_cell[keep]
    if added:
        return (np.vstack([tets[keep], np.stack(added)]),
                np.concatenate([kept_source,
                                source_cell[np.array(sources,
                                                     dtype=np.int64)]]))
    return tets[keep], kept_source


# ---------------------------------------------------------------------
# Moving nodes
# ---------------------------------------------------------------------

def _shortest_spoke(origin: np.ndarray, nodes: np.ndarray,
                    neighbours: np.ndarray, valence: np.ndarray,
                    rings: np.ndarray) -> np.ndarray:
    """The shortest edge at each node, measured where the nodes started.

    Against *origin* rather than the current positions, so that the
    travel limit means the same thing in the third round as in the
    first: a node may spend its allowance once, not once per call.
    """
    if len(neighbours) == 0:
        return np.zeros(len(nodes))
    spokes = np.linalg.norm(
        origin[neighbours] - np.repeat(origin[nodes], valence, axis=0),
        axis=1)
    shortest = np.minimum.reduceat(spokes, rings)
    return np.where(valence > 0, shortest, 0.0)


def _commit(points: np.ndarray, tets: np.ndarray, nodes: np.ndarray,
            candidate: np.ndarray, cap: np.ndarray, incident: np.ndarray,
            slot: np.ndarray, counts: np.ndarray, segments: np.ndarray,
            anchor: np.ndarray, reach: np.ndarray) -> None:
    """Apply the moves that survive, then take back those that did not.

    Every node's move was judged with only that node moved. Applied
    together, two nodes of the same element can each have been right and
    the element still end up worse, so the result is checked again with
    all the moves in place and the offending nodes are put back. Two
    rounds of that settle it: putting a node back can only move its
    elements toward the shape they already had.

    *anchor* and *reach* bound how far a node may end up from where it
    started, which is what keeps a wall from walking: sliding in the
    tangent plane is exact only to first order, and a node free to slide
    for twenty iterations is not taking a first-order step.
    """
    travel = candidate - anchor
    distance = np.linalg.norm(travel, axis=1)
    far = distance > reach
    if far.any():
        candidate = candidate.copy()
        candidate[far] = anchor[far] + travel[far] * (
            reach[far] / distance[far])[:, None]

    before = points[nodes].copy()
    trial = _quality_with(points, tets, incident, slot, counts, candidate)
    keep = _worst_per_node(trial, counts, segments) <= cap
    points[nodes[keep]] = candidate[keep]

    for _check in range(2):
        if not keep.any():
            break
        now = _quality_with(points, tets, incident, slot, counts,
                            points[nodes])
        undo = keep & (_worst_per_node(now, counts, segments) > cap)
        if not undo.any():
            break
        points[nodes[undo]] = before[undo]
        keep &= ~undo


def _still_bad(points: np.ndarray, tets: np.ndarray, nodes: np.ndarray,
               incident: np.ndarray, slot: np.ndarray, counts: np.ndarray,
               segments: np.ndarray, threshold: float) -> np.ndarray:
    """Which of *nodes* still have an element above the threshold."""
    now = _quality_with(points, tets, incident, slot, counts, points[nodes])
    return _worst_per_node(now, counts, segments) > threshold


def _subset(pair_owner: np.ndarray, keep: np.ndarray, incident: np.ndarray,
            slot: np.ndarray, counts: np.ndarray):
    """The pair arrays restricted to the nodes *keep* selects.

    Masking the flat pair list costs one pass over it; evaluating the
    shape of every element at every node costs twenty flops per pair.
    That is the whole reason this exists: a back-tracking step that
    re-measures the nodes which already succeeded spends most of its
    time on them.
    """
    mask = keep[pair_owner]
    sub_counts = counts[keep]
    sub_segments = np.zeros(len(sub_counts), dtype=np.int64)
    np.cumsum(sub_counts[:-1], out=sub_segments[1:])
    return incident[mask], slot[mask], sub_counts, sub_segments


def _smooth(points: np.ndarray, tets: np.ndarray, nodes: np.ndarray,
            adjacency: Tuple[np.ndarray, np.ndarray],
            boundary: _Boundary, options: QualityOptions,
            origin: np.ndarray) -> None:
    """Quality-guarded Taubin smoothing of *nodes*, in place.

    The guard is per node and one-sided: a move is kept only if the
    node's own worst element does not get worse than it already was, or
    than the threshold if it was better than that. It is deliberately
    stricter than letting well-shaped neighbourhoods smooth freely —
    that is the licence under which an unguarded pass walks a wall
    inwards millimetre by millimetre.
    """
    if len(nodes) == 0 or options.smooth_iterations <= 0:
        return
    cell_starts, cells = adjacency
    incident, slot, counts, segments = _incident(tets, cell_starts, cells,
                                                 nodes)
    start = _quality_with(points, tets, incident, slot, counts, points[nodes])
    cap = np.maximum(_worst_per_node(start, counts, segments),
                     options.threshold)

    neighbours, valence, rings = _rings(tets, cell_starts, cells, nodes)
    anchor = origin[nodes]
    reach = _shortest_spoke(origin, nodes, neighbours, valence,
                            rings) * options.max_travel
    usable = valence > 0
    pair_owner = np.repeat(np.arange(len(nodes), dtype=np.int64), counts)
    reachable = np.zeros(len(points), dtype=bool)
    live = np.ones(len(nodes), dtype=bool)

    for iteration in range(options.smooth_iterations):
        # A node whose elements are all sound, and none of whose
        # neighbours has a bad one either, has nothing to smooth
        # towards. The test reaches one ring further than the shift's,
        # because smoothing a sound node is exactly how its bad
        # neighbour gets room to improve.
        if iteration % _REFRESH == 0:
            bad_here = _still_bad(points, tets, nodes, incident, slot,
                                  counts, segments, options.threshold)
            reachable[:] = False
            reachable[nodes[bad_here]] = True
            near = np.add.reduceat(reachable[neighbours].astype(np.int8),
                                   rings) > 0
            live &= bad_here | near
            if not live.any():
                return
            idx = np.flatnonzero(live)
            sub = _subset(pair_owner, live, incident, slot, counts)

        for factor in (_SMOOTH_FORWARD, _SMOOTH_FORWARD * _SMOOTH_BACKWARD):
            average = np.zeros((len(nodes), 3))
            average[usable] = (np.add.reduceat(points[neighbours], rings)
                               / valence[:, None])[usable]
            step = _project((average - points[nodes]) * factor, nodes,
                            boundary, options.slide_boundary)
            candidate = points[nodes] + step
            _commit(points, tets, nodes[idx], candidate[idx], cap[idx],
                    sub[0], sub[1], sub[2], sub[3], anchor[idx], reach[idx])


def _shift(points: np.ndarray, tets: np.ndarray, nodes: np.ndarray,
           adjacency: Tuple[np.ndarray, np.ndarray],
           boundary: _Boundary, options: QualityOptions,
           origin: np.ndarray) -> None:
    """Gradient descent on the badness at *nodes*, in place.

    The functional is ``sum(10^(q + 1 - thr))`` over the elements at the
    node. The power is what makes this different from smoothing: the
    worst element dominates the sum, so the step is aimed at it rather
    than at the average of the neighbourhood.
    """
    if len(nodes) == 0 or options.shift_iterations <= 0:
        return
    cell_starts, cells = adjacency
    incident, slot, counts, segments = _incident(tets, cell_starts, cells,
                                                 nodes)
    neighbours, valence, rings = _rings(tets, cell_starts, cells, nodes)
    shortest = _shortest_spoke(origin, nodes, neighbours, valence, rings)
    anchor = origin[nodes]
    reach = shortest * options.max_travel
    start = _quality_with(points, tets, incident, slot, counts,
                          points[nodes])
    cap = np.maximum(_worst_per_node(start, counts, segments),
                     options.threshold)

    pair_owner = np.repeat(np.arange(len(nodes), dtype=np.int64), counts)
    live = np.ones(len(nodes), dtype=bool)

    for iteration in range(options.shift_iterations):
        # Nodes whose elements are all below the threshold have nothing
        # left to ask for. They are dropped rather than carried: the
        # functional is smooth and would happily keep improving them for
        # every one of the remaining iterations, at the cost of the
        # nodes that still need the work. Re-checked periodically rather
        # than every pass, because the check is itself a measurement of
        # every element at every live node.
        if iteration % _REFRESH == 0:
            live &= _still_bad(points, tets, nodes, incident, slot, counts,
                               segments, options.threshold)
            if not live.any():
                return
        idx = np.flatnonzero(live)
        sub_incident, sub_slot, sub_counts, sub_segments = _subset(
            pair_owner, live, incident, slot, counts)
        sub_nodes = nodes[idx]
        sub_shortest = shortest[idx]

        def badness(candidate, incident=sub_incident, slot=sub_slot,
                    counts=sub_counts, segments=sub_segments):
            return _functional(
                _quality_with(points, tets, incident, slot, counts,
                              candidate),
                segments, options.threshold)

        base = points[sub_nodes].copy()
        here = badness(base)
        delta = 1e-4 * np.maximum(sub_shortest, 1e-12)
        gradient = np.empty((len(sub_nodes), 3))
        for axis in range(3):
            probe = base.copy()
            probe[:, axis] += delta
            gradient[:, axis] = badness(probe) - here
        length = np.linalg.norm(gradient, axis=1)
        moving = length > 0.0
        direction = np.zeros_like(gradient)
        direction[moving] = -gradient[moving] / length[moving][:, None]

        step = sub_shortest * options.step_fraction
        best = base.copy()
        active = moving.copy()
        sub_owner = np.repeat(np.arange(len(sub_nodes), dtype=np.int64),
                              sub_counts)
        for _attempt in range(_BACKTRACK):
            if not active.any():
                break
            candidate = base + _project(direction * step[:, None], sub_nodes,
                                        boundary, options.slide_boundary)
            hot = np.flatnonzero(active)
            inc, slt, cnt, seg = _subset(sub_owner, active, sub_incident,
                                         sub_slot, sub_counts)
            value = _functional(
                _quality_with(points, tets, inc, slt, cnt, candidate[hot]),
                seg, options.threshold)
            better = value < here[hot]
            best[hot[better]] = candidate[hot[better]]
            active[hot[better]] = False
            step[active] *= 0.5

        _commit(points, tets, sub_nodes, best,
                cap[idx], sub_incident, sub_slot, sub_counts, sub_segments,
                anchor[idx], reach[idx])


# ---------------------------------------------------------------------
# The pass
# ---------------------------------------------------------------------
def improve(points: np.ndarray,
            tets: np.ndarray,
            options: Optional[QualityOptions] = None,
            on_status: Optional[Callable[[str], None]] = None):
    """Repair the shape of the elements above the threshold.

    Returns ``(points, tets, point_source, source_cell, report)``.
    ``source_cell`` indexes the input elements, so a caller carries
    every element field by selection: a flip's replacement takes the
    value of one of the elements it replaced.

    ``point_source`` indexes the input *nodes*, and a caller has to use
    it. The 4-to-1 flip exists to remove a node, so the node count is
    not fixed and a point field cannot simply be handed across. Leaving
    the node in place instead was a real bug: a node no element mentions
    is invisible here and stops a solver later, because a solver numbers
    its nodes from the element list and the count then comes up short.

    Neither input array is modified.
    """
    options = options or QualityOptions()
    options.validate()

    points = np.array(points, dtype=float, copy=True)
    tets = np.array(tets, dtype=np.int64, copy=True)
    original = points.copy()
    source_cell = np.arange(len(tets), dtype=np.int64)

    measure = quality(points, tets)
    volumes = np.abs(signed_volumes(points, tets))
    report = QualityReport(
        cells_before=len(tets), cells_after=len(tets),
        bad_before=int((measure > options.threshold).sum()),
        bad_after=int((measure > options.threshold).sum()),
        worst_before=float(measure.max()) if len(measure) else 0.0,
        worst_after=float(measure.max()) if len(measure) else 0.0,
        mean_before=float(measure.mean()) if len(measure) else 0.0,
        mean_after=float(measure.mean()) if len(measure) else 0.0,
        volume_before=float(volumes.sum()),
        volume_after=float(volumes.sum()),
        threshold=options.threshold)
    if report.bad_before == 0:
        return (points, tets, np.arange(len(points), dtype=np.int64),
                source_cell, report)

    # The boundary's faces do not change: a flip works on an interior
    # edge and a move changes no connectivity, so the table is rebuilt
    # only when a flip has actually fired — it is which element owns a
    # boundary face, and so which way is outward, that a flip can move.
    boundary = None
    for _round in range(options.max_rounds):
        was_points, was_tets = points.copy(), tets.copy()
        was_source, was_bad = source_cell.copy(), report.bad_after
        # Carried from the end of the previous round rather than
        # remeasured: nothing has touched the mesh since.
        measure = quality(points, tets) if report.rounds == 0 else measure
        report.rounds += 1
        if on_status is not None:
            on_status(f"Quality round {report.rounds}: "
                      f"{was_bad} element"
                      + ("" if was_bad == 1 else "s")
                      + f" above {options.threshold:g}…")

        starts, cells = node_cells(tets, len(points))
        if boundary is None:
            boundary = _boundary_frame(points, tets, options.feature_angle)

        if options.flips:
            dropped, added, sources = _flip_4_to_1(
                points, tets, measure, starts, cells,
                boundary.is_boundary, options.threshold)
            report.flips_4_to_1 += len(added)
            tets, source_cell = _apply_flips(tets, source_cell, dropped,
                                             added, sources)
            if dropped:
                starts, cells = node_cells(tets, len(points))
                boundary = None
                measure = quality(points, tets)

            dropped, added, sources = _flip_3_to_2(
                points, tets, measure, starts, cells, options.threshold)
            report.flips_3_to_2 += len(added) // 2
            tets, source_cell = _apply_flips(tets, source_cell, dropped,
                                             added, sources)
            if dropped:
                starts, cells = node_cells(tets, len(points))
                boundary = None
                measure = quality(points, tets)

        if boundary is None:
            boundary = _boundary_frame(points, tets, options.feature_angle)
        seed = np.where(measure > options.threshold)[0]
        if len(seed):
            nodes = _grow(tets, starts, cells, seed, layers=2)
            _smooth(points, tets, nodes, (starts, cells), boundary,
                    options, original)
            nodes = _grow(tets, starts, cells,
                          np.where(quality(points, tets)
                                   > options.threshold)[0], layers=1)
            _shift(points, tets, nodes, (starts, cells), boundary,
                   options, original)

        measure = quality(points, tets)
        now = int((measure > options.threshold).sum())
        if now >= was_bad:
            # The round did not pay for itself. Undone whole rather than
            # kept in part: a half-applied round is a mesh nobody chose.
            points, tets = was_points, was_tets
            source_cell, report.rounds = was_source, report.rounds - 1
            break
        report.bad_after = now

    measure = quality(points, tets)
    volumes = np.abs(signed_volumes(points, tets))
    shift = np.linalg.norm(points - original, axis=1)
    # The frame the last round used already knows which nodes are on the
    # wall, and building another one to ask costs as much as a round.
    if boundary is None:
        boundary = _boundary_frame(points, tets, options.feature_angle)
    on_wall = boundary.is_boundary
    report.cells_after = len(tets)
    report.bad_after = int((measure > options.threshold).sum())
    report.worst_after = float(measure.max()) if len(measure) else 0.0
    report.mean_after = float(measure.mean()) if len(measure) else 0.0
    report.volume_after = float(volumes.sum())
    report.moved_nodes = int((shift > 0.0).sum())
    report.max_shift = float(shift.max()) if len(shift) else 0.0
    report.max_boundary_shift = (float(shift[on_wall].max())
                                 if on_wall.any() else 0.0)
    # The flips are what make this necessary: a 4-to-1 flip replaces a
    # node's four elements with one that does not mention the node, so
    # the node is left in the array with nothing using it. Reported
    # after the shift statistics above, which are over the nodes as they
    # were asked to move.
    point_source = np.unique(tets) if len(tets) else np.zeros(0, np.int64)
    report.removed_nodes = len(points) - len(point_source)
    if report.removed_nodes:
        remap = np.empty(len(points), dtype=np.int64)
        remap[point_source] = np.arange(len(point_source))
        points, tets = points[point_source], remap[tets]

    if on_status is not None:
        on_status(report.summary())
    return points, tets, point_source, source_cell, report


def improve_grid(grid,
                 options: Optional[QualityOptions] = None,
                 on_status: Optional[Callable[[str], None]] = None):
    """:func:`improve`, for a grid rather than arrays.

    ``grid`` is not modified. Both element and point arrays are carried
    by selection, never by interpolation: a flip's replacement inherits
    the label and fibre of an element it replaced, and a surviving node
    keeps its own values. Point arrays are selected rather than copied
    whole because the 4-to-1 flip removes a node, so the result can have
    fewer nodes than the source.

    Raises ``ValueError`` for a non-tetrahedral volume or impossible
    options.
    """
    source = pv.wrap(grid)
    validate_tetrahedral(source)
    points, tets, point_source, source_cell, report = improve(
        np.asarray(source.points, dtype=float), tetrahedra(source),
        options, on_status)

    out = pv.UnstructuredGrid({TETRA: tets.astype(np.int64)}, points)
    for name in source.cell_data.keys():
        out.cell_data[name] = np.asarray(source.cell_data[name])[source_cell]
    for name in source.point_data.keys():
        out.point_data[name] = np.asarray(source.point_data[name])[point_source]
    return out, report


__all__ = ["QualityOptions", "QualityReport", "improve", "improve_grid",
           "quality", "QUALITY_THRESHOLD"]
