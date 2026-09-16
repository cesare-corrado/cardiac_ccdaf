"""
volume_clean
============
Repair a tetrahedral volume's connectivity, and report its topology.

This is not the volume counterpart of the surface cleaner's smoothing and
hole filling. It touches no vertex position and invents no material: every
operation here either removes something that is not part of the mesh's
body, or renumbers what stays. That restraint is the point. A volume is
usually the last stage of a pipeline, already validated by eye, and a
"repair" that quietly moved the wall would invalidate that judgement.

Why a volume needs this at all
------------------------------
A stray element is not a cosmetic problem. Every solver that puts a
Dirichlet condition on named surfaces solves one linear system over the
whole mesh, and a component carrying no condition makes that system
singular: the run fails, or worse, converges to something arbitrary on
that piece. On the 290,474-element example ventricle there are exactly
three such elements, each a single tetrahedron attached to the body by two
or three nodes and by no face at all, sitting *outside* the body rather
than plugging a void, so removing them leaves no hole behind.

What it deliberately cannot fix
-------------------------------
Two defects survive this pass, and both are reported rather than repaired:

* **Tunnels.** A hole through the material is either anatomy (the example
  has one, and the left ventricle alone is simply connected, so it comes
  from how the right ventricular wall joins) or a segmentation artefact.
  Nothing local can tell those apart, and filling one means inventing
  material that was never imaged.
* **Pinches**, where the wall thins to nothing and the boundary touches
  itself. The example has 17, and at every one of them the tetrahedra
  around the node still form a single face-connected fan: the material is
  continuous, and only the surface touches. Splitting such a node would
  tear apart material that is genuinely joined — measured on the example,
  splitting fixed 9 volume-level nodes and left all 17 surface pinches
  exactly as they were.

Reporting them is not a consolation prize. A pinch is where a surface
label can leak from epicardium to endocardium, so a caller that labels
surfaces needs to know the count before it trusts the labelling.

Topology, and when it is trustworthy
------------------------------------
The Euler characteristic ``V - E + F - T`` is exact for any simplicial
complex, manifold or not, so it is always reported. Tunnels and cavities
are not: deriving them needs the number of boundary sheets, and counting
sheets assumes a manifold boundary. On a boundary that pinches, that count
is wrong, and the resulting tunnel figure is wrong with it — which is how
this module's own development produced three different tunnel counts for
one mesh before the assumption was spotted. So they are derived only when
the boundary is manifold, and reported as unknown otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
import pyvista as pv
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

# Shared with the surface cleaner rather than copied: the two panels offer
# the same control and it would be a trap for them to drift apart.
from ccdaf.core.mesh_postprocessor import MIN_COMPONENT_FRACTION
from ccdaf.core.volume_mesh import (
    TETRA, orient_positive, signed_volumes, tetrahedra, validate_tetrahedral,
)

#: The four faces and six edges of a tetrahedron, as node positions.
_FACE_NODES = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
_EDGE_NODES = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])


@dataclass
class CleanOptions:
    """What the clean is allowed to do.

    Every default is the conservative one: weld nothing that is not
    already coincident, drop nothing that could be anatomy, and change no
    coordinate.
    """

    #: Absolute welding distance for coincident points, in mesh units.
    #: ``0.0`` merges only exactly coincident points and so moves no
    #: vertex. A positive value welds near-duplicates, and the survivor
    #: is the first point of each group rather than its centroid, so the
    #: points that remain are still points the mesh had.
    merge_tol: float = 0.0
    #: Drop a tetrahedron whose absolute volume is at or below this.
    #: ``0.0`` drops only the exactly flat ones.
    min_volume: float = 0.0
    #: The smallest share of the elements a face-connected component may
    #: hold and still be kept. The largest component is always kept, so
    #: this can never empty the mesh.
    min_component_fraction: float = MIN_COMPONENT_FRACTION
    #: Reorder the nodes of negatively oriented tetrahedra. Costs
    #: nothing, changes no geometry, and the remesher refuses a mesh
    #: without it.
    fix_inverted: bool = True

    def validate(self) -> None:
        if self.merge_tol < 0.0:
            raise ValueError("the merge tolerance must not be negative")
        if self.min_volume < 0.0:
            raise ValueError("the minimum element volume must not be negative")
        if not 0.0 <= self.min_component_fraction <= 1.0:
            raise ValueError(
                "the minimum component fraction must be between 0 and 1")


@dataclass
class TopologyReport:
    """What the mesh is shaped like, and how much of that is knowable.

    ``tunnels`` and ``cavities`` are ``None`` when the boundary is not
    manifold. See the module docstring: a pinched boundary makes the sheet
    count wrong, and a tunnel figure derived from it is wrong too.
    """

    cells: int
    components: int
    euler_characteristic: int
    boundary_faces: int
    boundary_sheets: int
    open_edges: int
    non_manifold_edges: int
    pinched_vertices: int
    tunnels: Optional[int]
    cavities: Optional[int]

    @property
    def boundary_is_manifold(self) -> bool:
        return (self.open_edges == 0 and self.non_manifold_edges == 0
                and self.pinched_vertices == 0)

    def summary(self) -> str:
        parts = [f"{self.cells} tetrahedra",
                 f"{self.components} component"
                 + ("" if self.components == 1 else "s"),
                 f"χ = {self.euler_characteristic}"]
        if self.tunnels is None:
            parts.append("tunnels and cavities not determined "
                         "(the boundary is not manifold)")
        else:
            parts.append(f"{self.tunnels} tunnel"
                         + ("" if self.tunnels == 1 else "s"))
            parts.append(f"{self.cavities} enclosed cavit"
                         + ("y" if self.cavities == 1 else "ies"))
        flaws = []
        if self.open_edges:
            flaws.append(f"{self.open_edges} open boundary edges")
        if self.non_manifold_edges:
            flaws.append(f"{self.non_manifold_edges} non-manifold edges")
        if self.pinched_vertices:
            flaws.append(f"{self.pinched_vertices} pinch points")
        if flaws:
            parts.append("; ".join(flaws))
        return ", ".join(parts)


@dataclass
class CleanReport:
    """What the clean actually did, in the order it did it."""

    points_before: int
    points_after: int
    cells_before: int
    cells_after: int
    merged_points: int
    dropped_degenerate: int
    dropped_duplicate: int
    dropped_components: int
    dropped_component_cells: int
    flipped: int
    before: TopologyReport
    after: TopologyReport

    @property
    def changed(self) -> bool:
        """Whether anything at all was altered.

        A clean that changed nothing must not be allowed to mark the
        session dirty, or every inspection would look like an edit.
        """
        return bool(self.merged_points or self.dropped_degenerate
                    or self.dropped_duplicate or self.dropped_component_cells
                    or self.flipped)

    def summary(self) -> str:
        if not self.changed:
            return "Nothing to clean; the mesh was already sound."
        bits: List[str] = []
        if self.merged_points:
            bits.append(f"merged {self.merged_points} coincident points")
        if self.dropped_degenerate:
            bits.append(f"dropped {self.dropped_degenerate} degenerate "
                        f"element" + ("" if self.dropped_degenerate == 1
                                      else "s"))
        if self.dropped_duplicate:
            bits.append(f"dropped {self.dropped_duplicate} duplicate "
                        f"element" + ("" if self.dropped_duplicate == 1
                                      else "s"))
        if self.dropped_components:
            bits.append(f"dropped {self.dropped_components} detached "
                        f"component"
                        + ("" if self.dropped_components == 1 else "s")
                        + f" ({self.dropped_component_cells} element"
                        + ("" if self.dropped_component_cells == 1 else "s")
                        + ")")
        if self.flipped:
            bits.append(f"reoriented {self.flipped} inverted element"
                        + ("" if self.flipped == 1 else "s"))
        return (f"{self.cells_before} → {self.cells_after} tetrahedra: "
                + ", ".join(bits) + ".")

    def details(self) -> str:
        """The full report, for a dialog rather than a status line."""
        return (f"{self.summary()}\n\n"
                f"Before: {self.before.summary()}\n"
                f"After:  {self.after.summary()}")


# ---------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------
def _unique_rows(rows: np.ndarray):
    """``np.unique`` over sorted rows, with the inverse and the counts."""
    return np.unique(np.sort(rows, axis=1), axis=0,
                     return_inverse=True, return_counts=True)


def _face_table(tets: np.ndarray):
    """Unique faces of *tets*, which tetrahedron each came from, and counts.

    Vectorised on purpose: the obvious dictionary loop costs seconds on a
    290,000-element mesh, and this runs on every clean and every report.
    """
    faces = tets[:, _FACE_NODES].reshape(-1, 3)
    uniq, inv, counts = _unique_rows(faces)
    owner = np.repeat(np.arange(len(tets), dtype=np.int64), 4)
    return uniq, inv.ravel(), counts, owner


def _cell_components(tets: np.ndarray) -> Tuple[int, np.ndarray]:
    """Label the tetrahedra by face-connected component.

    Elements that touch only at a node or along an edge are *not*
    connected: a solver sees no coupling across a point contact either.
    """
    if len(tets) == 0:
        return 0, np.zeros(0, dtype=np.int64)
    _uniq, inv, counts, owner = _face_table(tets)
    order = np.argsort(inv, kind="stable")
    starts = np.searchsorted(inv[order], np.arange(len(counts)))
    shared = np.where(counts == 2)[0]
    if shared.size:
        a = owner[order[starts[shared]]]
        b = owner[order[starts[shared] + 1]]
    else:
        a = b = np.zeros(0, dtype=np.int64)
    graph = coo_matrix((np.ones(len(a)), (a, b)),
                       shape=(len(tets), len(tets)))
    return connected_components(graph, directed=False)


def _boundary_faces(tets: np.ndarray) -> np.ndarray:
    """The faces of *tets* that belong to exactly one tetrahedron."""
    uniq, _inv, counts, _owner = _face_table(tets)
    return uniq[counts == 1]


def _boundary_flaws(boundary: np.ndarray) -> Tuple[int, int, int, int]:
    """``(open_edges, non_manifold_edges, pinched_vertices, sheets)``.

    A pinch is a vertex whose incident boundary faces form more than one
    fan: the surface passes through the point twice. It is found by
    joining, at each endpoint of every shared edge, the faces meeting
    there, then counting how many groups each vertex ends up with.
    Corners are the unit of work, not faces, which is what keeps the two
    fans apart at the pinch itself.

    Faces are joined across non-manifold edges as well as manifold ones,
    so that this counts *only* surfaces touching at a point. Joining
    across manifold edges alone would report every endpoint of a
    non-manifold edge as a pinch too, inflating the count by counting
    one defect twice: on the example ventricle that reads 47 rather than
    the 17 places where the surface actually touches itself.
    """
    if len(boundary) == 0:
        return 0, 0, 0, 0
    pairs = boundary[:, [[0, 1], [1, 2], [0, 2]]].reshape(-1, 2)
    uniq, inv, counts = _unique_rows(pairs)
    open_edges = int((counts == 1).sum())
    non_manifold = int((counts > 2).sum())

    inv = inv.ravel()
    face_of_edge = np.repeat(np.arange(len(boundary), dtype=np.int64), 3)
    order = np.argsort(inv, kind="stable")
    starts = np.searchsorted(inv[order], np.arange(len(counts)))
    shared = np.where(counts == 2)[0]
    if shared.size == 0:
        return open_edges, non_manifold, 0, 0
    f0 = face_of_edge[order[starts[shared]]]
    f1 = face_of_edge[order[starts[shared] + 1]]

    # Sheets: faces joined across manifold edges.
    sheets = connected_components(
        coo_matrix((np.ones(len(f0)), (f0, f1)),
                   shape=(len(boundary),) * 2), directed=False)[0]

    # Corner ids: 3 * face + slot. At every edge with two or more faces,
    # join the corners sitting at the same vertex, pairing each face with
    # the first one there.
    multi = np.where(counts >= 2)[0]
    extra = counts[multi] - 1
    total = int(extra.sum())
    base = np.repeat(starts[multi], extra)
    offsets = np.arange(total) - np.repeat(np.cumsum(extra) - extra, extra) + 1
    lead = face_of_edge[order[base]]
    trail = face_of_edge[order[base + offsets]]
    ends = np.repeat(uniq[multi], extra, axis=0)

    rows: List[np.ndarray] = []
    cols: List[np.ndarray] = []
    for end in (0, 1):
        vertex = ends[:, end]
        slot0 = np.argmax(boundary[lead] == vertex[:, None], axis=1)
        slot1 = np.argmax(boundary[trail] == vertex[:, None], axis=1)
        rows.append(3 * lead + slot0)
        cols.append(3 * trail + slot1)
    corner_rows = np.concatenate(rows)
    corner_cols = np.concatenate(cols)
    n_corners = 3 * len(boundary)
    labels = connected_components(
        coo_matrix((np.ones(len(corner_rows)), (corner_rows, corner_cols)),
                   shape=(n_corners, n_corners)), directed=False)[1]
    corner_vertex = boundary.reshape(-1)
    fans = np.unique(np.stack([corner_vertex, labels], axis=1), axis=0)
    _vertices, fan_counts = np.unique(fans[:, 0], return_counts=True)
    pinched = int((fan_counts > 1).sum())
    return open_edges, non_manifold, pinched, int(sheets)


def topology(grid) -> TopologyReport:
    """Measure *grid*: components, Euler characteristic, boundary flaws.

    ``grid`` is not modified. See the module docstring for why tunnels and
    cavities are sometimes ``None``.
    """
    source = pv.wrap(grid)
    validate_tetrahedral(source)
    tets = tetrahedra(source)
    if len(tets) == 0:
        return TopologyReport(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    n_components, _labels = _cell_components(tets)
    n_vertices = len(np.unique(tets))
    n_edges = len(np.unique(np.sort(tets[:, _EDGE_NODES].reshape(-1, 2),
                                    axis=1), axis=0))
    uniq_faces, _inv, counts, _owner = _face_table(tets)
    chi = n_vertices - n_edges + len(uniq_faces) - len(tets)

    boundary = uniq_faces[counts == 1]
    open_edges, non_manifold, pinched, sheets = _boundary_flaws(boundary)

    manifold = open_edges == 0 and non_manifold == 0 and pinched == 0
    if manifold and sheets >= n_components:
        cavities = sheets - n_components
        tunnels = n_components + cavities - chi
    else:
        cavities = tunnels = None

    return TopologyReport(
        cells=len(tets), components=int(n_components),
        euler_characteristic=int(chi), boundary_faces=len(boundary),
        boundary_sheets=int(sheets), open_edges=open_edges,
        non_manifold_edges=non_manifold, pinched_vertices=pinched,
        tunnels=tunnels, cavities=cavities)


# ---------------------------------------------------------------------
# The clean
# ---------------------------------------------------------------------
def _merge_points(points: np.ndarray,
                  tol: float) -> Tuple[np.ndarray, np.ndarray, int]:
    """Group coincident points; return ``(groups, first_of_group, merged)``.

    The survivor of a group is its first point, never a centroid: welding
    must not move geometry that the caller has already accepted.
    """
    if len(points) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), 0
    if tol > 0.0:
        pairs = cKDTree(points).query_pairs(tol, output_type="ndarray")
        graph = coo_matrix(
            (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
            shape=(len(points),) * 2)
        _n, groups = connected_components(graph, directed=False)
    else:
        _uniq, groups = np.unique(points, axis=0, return_inverse=True)
        groups = groups.ravel()
    n_groups = int(groups.max()) + 1 if len(groups) else 0
    first = np.zeros(n_groups, dtype=np.int64)
    # The first original index in each group, which is the one that stays.
    order = np.argsort(groups, kind="stable")
    seen = np.searchsorted(groups[order], np.arange(n_groups))
    first[:] = order[seen]
    return groups, first, len(points) - n_groups


def clean(grid,
          options: Optional[CleanOptions] = None,
          on_status: Optional[Callable[[str], None]] = None
          ) -> Tuple[pv.UnstructuredGrid, CleanReport]:
    """Repair the connectivity of *grid* and report what was done.

    ``grid`` is not modified. Every point and cell array it carries is
    kept, with the values the surviving elements already had: nothing here
    interpolates, so nothing here can invent a value.

    Raises ``ValueError`` for a non-tetrahedral volume or impossible
    options.
    """
    options = options or CleanOptions()
    options.validate()

    source = pv.wrap(grid)
    validate_tetrahedral(source)
    points = np.asarray(source.points, dtype=float)
    tets = tetrahedra(source)
    before = topology(source)
    if on_status is not None:
        on_status(f"Cleaning {len(tets)} tetrahedra…")

    kept = np.arange(len(tets), dtype=np.int64)

    # 1. Coincident points. Done first: welding is what turns a sliver
    #    into a degenerate element, and step 2 is what removes it.
    groups, first_of_group, merged = _merge_points(points, options.merge_tol)
    if len(tets):
        tets = groups[tets]
    new_points = points[first_of_group] if len(points) else points

    # 2. Degenerate elements: a repeated node, or no volume at all.
    if len(tets):
        ordered = np.sort(tets, axis=1)
        repeated = (ordered[:, 1:] == ordered[:, :-1]).any(axis=1)
        flat = np.abs(signed_volumes(new_points, tets)) <= options.min_volume
        bad = repeated | flat
        dropped_degenerate = int(bad.sum())
        tets, kept = tets[~bad], kept[~bad]
    else:
        dropped_degenerate = 0

    # 3. Duplicate elements: the same four nodes twice over. Harmless to
    #    look at and fatal to assemble, since the pair shares all four
    #    faces and so reads as a closed shell with no interior.
    if len(tets):
        ordered = np.sort(tets, axis=1)
        _uniq, keep_index = np.unique(ordered, axis=0, return_index=True)
        keep_index = np.sort(keep_index)
        dropped_duplicate = len(tets) - len(keep_index)
        tets, kept = tets[keep_index], kept[keep_index]
    else:
        dropped_duplicate = 0

    # 4. Detached components. The largest is always kept, so this cannot
    #    empty the mesh however the fraction is set.
    dropped_components = dropped_cells = 0
    if len(tets):
        n_components, labels = _cell_components(tets)
        if n_components > 1:
            sizes = np.bincount(labels, minlength=n_components)
            keep_mask = sizes >= options.min_component_fraction * len(tets)
            keep_mask[int(sizes.argmax())] = True
            dropped_components = int((~keep_mask).sum())
            survive = keep_mask[labels]
            dropped_cells = int((~survive).sum())
            tets, kept = tets[survive], kept[survive]

    # 5. Orientation. Last, so it only pays for the elements that stay.
    flipped = 0
    if options.fix_inverted and len(tets):
        tets, flipped = orient_positive(new_points, tets)

    # 6. Renumber, dropping points no surviving element uses.
    used = np.unique(tets) if len(tets) else np.zeros(0, dtype=np.int64)
    remap = np.full(len(new_points), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    final_points = new_points[used]
    final_tets = remap[tets] if len(tets) else tets.reshape(0, 4)

    out = pv.UnstructuredGrid({TETRA: final_tets.astype(np.int64)},
                              final_points)
    _carry_arrays(source, out, kept, first_of_group[used])

    after = topology(out)
    report = CleanReport(
        points_before=len(points), points_after=out.n_points,
        cells_before=len(np.asarray(tetrahedra(source))),
        cells_after=out.n_cells, merged_points=merged,
        dropped_degenerate=dropped_degenerate,
        dropped_duplicate=dropped_duplicate,
        dropped_components=dropped_components,
        dropped_component_cells=dropped_cells, flipped=int(flipped),
        before=before, after=after)
    if on_status is not None:
        on_status(report.summary())
    return out, report


def _carry_arrays(source: pv.UnstructuredGrid, out: pv.UnstructuredGrid,
                  kept_cells: np.ndarray, source_points: np.ndarray) -> None:
    """Copy every array across, by selection rather than by interpolation.

    A cleaned element is one of the source's own elements, and a surviving
    point is one of its own points, so each value is carried verbatim. An
    integer label stays an integer label, which averaging would not
    guarantee.
    """
    for name in source.cell_data.keys():
        values = np.asarray(source.cell_data[name])
        out.cell_data[name] = values[kept_cells]
    for name in source.point_data.keys():
        values = np.asarray(source.point_data[name])
        out.point_data[name] = values[source_points]


__all__ = ["CleanOptions", "CleanReport", "TopologyReport", "clean",
           "topology", "MIN_COMPONENT_FRACTION"]
