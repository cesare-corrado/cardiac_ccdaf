"""
volume_tags
===========
Atrial region labels (body, veins, appendage) for the tetrahedra of an
atrial volume, from labels drawn on its endocardium.

Why the endocardium
-------------------
Every tagging tool (seeds and region growing, painting, the snake, hole
filling, smoothing) works on triangles and their neighbours. The
endocardium of a labelled atrial volume (see
:mod:`ccdaf.core.atrial_surfaces`) is an open triangle surface, the same
kind the surface UAC take, so the tools work on it unchanged:
:func:`endocardium` cuts it out, and :func:`apply_to_volume` carries what
was drawn on it into the wall.

Through the wall: harmonic extension
------------------------------------
Copying the nearest endocardial label fails once the wall is several
tetrahedra thick: a mid-wall tetrahedron can lie nearest to the wrong
place, and the border between two regions then zig-zags inside the wall
however clean it looks on the endocardium. Instead, for each label one
Laplace problem is solved over the volume, held at 1 on the endocardial
nodes with that label and 0 on the other endocardial nodes, free
(homogeneous Neumann) on the epicardium and the rims. Each tetrahedron
takes the label whose solution is highest at its centre. The borders
between regions are then level sets of smooth functions: they run
through the wall without concavities, and a vein on the endocardium is
the same vein on the epicardium. All the problems hold the same nodes,
so they share one matrix and one preconditioner.

In a volume holding both atria, the other atrium's endocardium is held at
1 in a problem of its own, so its wall keeps its labels.

Finally each label keeps only its largest connected piece of tetrahedra;
smaller islands join the neighbouring label they touch most.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional

import numpy as np
import pyvista as pv
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, dijkstra

from ccdaf.core import atrial_labels as al
from ccdaf.core import tag_map
from ccdaf.core.atrial_surfaces import ENDO_OF
from ccdaf.core.chamber_extract import OPENING_FIELD, _shared_faces
from ccdaf.core.laplace import DirichletSystem, stiffness_matrix
from ccdaf.core.opening_band import boundary_loops, loop_band
from ccdaf.core.surface_labels import (
    LABEL_MASK_FIELD, face_labels_from_mask,
)
from ccdaf.core.volume_mesh import boundary_surface, tetrahedra

#: Point field on the endocardium surface: each node's id in the volume.
VOLUME_NODE: str = "volumeNode"
#: Cell field on the endocardium surface: each face's tetrahedron.
VOLUME_CELL: str = "volumeCell"

#: Pseudo-label of the other atrium in a biatrial volume.
_OTHER = -1


def endocardium_present(grid) -> List[str]:
    """The atria whose endocardium is labelled on *grid* ("LA", "RA")."""
    grid = pv.wrap(grid)
    if LABEL_MASK_FIELD not in grid.point_data:
        return []
    mask = np.asarray(grid.point_data[LABEL_MASK_FIELD], dtype=np.int64)
    from ccdaf.core.surface_labels import MASK_BITS
    return [a for a, key in ENDO_OF.items() if np.any(mask & MASK_BITS[key])]


def _boundary_labels(grid):
    surface = boundary_surface(grid)
    origin = np.asarray(surface.point_data["vtkOriginalPointIds"], dtype=np.int64)
    mask = np.asarray(grid.point_data[LABEL_MASK_FIELD], dtype=np.int64)
    return surface, face_labels_from_mask(surface, mask[origin])


def endocardium(grid, atrium: str = "LA") -> pv.PolyData:
    """*atrium*'s endocardium as a triangle surface, ready to tag.

    It carries ``elemTag`` (the tetrahedra's labels where they are already
    in ccdaf's atrial scheme, the atrium's body elsewhere) and the maps
    back to the volume, :data:`VOLUME_NODE` and :data:`VOLUME_CELL`.
    Raises ``ValueError`` when that endocardium is not labelled.
    """
    grid = pv.wrap(grid)
    if atrium not in endocardium_present(grid):
        raise ValueError(f"the {atrium} endocardium is not labelled: run "
                         "Label atrial surfaces first")
    surface, labels = _boundary_labels(grid)
    faces = np.where(labels == ENDO_OF[atrium])[0]
    tri = np.asarray(surface.faces).reshape(-1, 4)[:, 1:][faces]
    nodes, local = np.unique(tri, return_inverse=True)
    local = local.reshape(-1, 3)
    out = pv.PolyData(np.asarray(surface.points)[nodes],
                      np.hstack([np.full((len(local), 1), 3), local]).ravel())
    origin = np.asarray(surface.point_data["vtkOriginalPointIds"], dtype=np.int64)
    cells = np.asarray(surface.cell_data["vtkOriginalCellIds"], dtype=np.int64)
    out.point_data[VOLUME_NODE] = origin[nodes]
    out.cell_data[VOLUME_CELL] = cells[faces]
    a = al.LA if atrium == "LA" else al.RA
    tags = np.asarray(grid.cell_data["elemTag"]).astype(int)[cells[faces]]
    out.cell_data["elemTag"] = np.where(np.isin(tags, a.values), tags,
                                        a.body).astype(np.int32)
    if OPENING_FIELD in grid.point_data:
        out.point_data[OPENING_FIELD] = np.asarray(
            grid.point_data[OPENING_FIELD])[origin[nodes]]
    return out


#: How far inside an opening a seed is placed, as a share of the
#: opening's radius: inside the vein, clear of its edge.
_SEED_DEPTH: float = 0.5


def opening_loops(endo: pv.PolyData,
                  mapping: Optional[Mapping[int, object]]
                  ) -> Dict[str, np.ndarray]:
    """The edge loop of each opening the tag names explain, by seed name
    ("LSPV", "LAA", "MV", …): node ids of *endo*. Region growing uses a
    vein's loop as an extra source (see :meth:`RegionTagger.tag`)."""
    if OPENING_FIELD not in endo.point_data:
        return {}
    named = tag_map.normalise(mapping)
    opening = np.asarray(endo.point_data[OPENING_FIELD]).astype(int)
    out: Dict[str, np.ndarray] = {}
    for loop in boundary_loops(endo):
        values, counts = np.unique(opening[loop], return_counts=True)
        counts[values == 0] = 0
        if counts.max() == 0:
            continue
        e = named.get(int(values[np.argmax(counts)]))
        if e is None:
            continue
        if e.role == tag_map.VALVE:
            name = "MV" if "LA" in e.chambers else "TV"
        elif e.name.endswith(" cap"):
            name = e.name[:-4]
        else:
            continue
        out[name] = loop
    return out


#: Default width of the band tagged at each known opening, in mm: the
#: band that matched the segmentation's own vein rings on a four-chamber
#: mesh (every ring tetrahedron took its vein's label).
BAND_MM: float = 3.0


def band_openings(endo: pv.PolyData, mapping: Optional[Mapping[int, object]],
                  atrium: str, width: float) -> Dict[str, int]:
    """Tag a band of *width* (mesh units) at each known vein and appendage
    opening of *endo*, in place; every other face is the atrium's body.

    For veins clipped at or near the ostium, which leave too little stub to
    grow a region from a seed. Returns the faces tagged per region.
    """
    own = al.LA if atrium == "LA" else al.RA
    tags = np.full(endo.n_cells, own.body, dtype=np.int32)
    loops = {n: lp for n, lp in opening_loops(endo, mapping).items()
             if n in own.regions}                 # the valve is no region
    if not loops:
        endo.cell_data["elemTag"] = tags
        return {}
    # Each face goes to the nearest opening, so two openings closer than
    # the band width split the faces between them rather than one band
    # running into the other's edge.
    tri = np.asarray(endo.faces).reshape(-1, 4)[:, 1:]
    names = list(loops)
    dist = np.full((len(names), endo.n_cells), np.inf)
    for i, name in enumerate(names):
        faces = loop_band(endo, loops[name], width)
        dist[i, faces] = _face_distance(endo, tri, loops[name])[faces]
    near = dist.argmin(0)
    inside = np.isfinite(dist.min(0))
    counts: Dict[str, int] = {}
    for i, name in enumerate(names):
        mine = inside & (near == i)
        tags[mine] = own.regions[name]
        counts[name] = int(mine.sum())
    endo.cell_data["elemTag"] = tags
    return counts


def _face_distance(endo: pv.PolyData, tri: np.ndarray, loop: np.ndarray
                   ) -> np.ndarray:
    """Each face's distance from *loop* along the surface: the mean of its
    nodes'."""
    pts = np.asarray(endo.points, dtype=float)
    edges = np.unique(np.sort(np.concatenate(
        [tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [0, 2]]]), axis=1), axis=0)
    w = np.maximum(np.linalg.norm(pts[edges[:, 1]] - pts[edges[:, 0]], axis=1),
                   1e-12)
    graph = sp.coo_matrix((w, (edges[:, 0], edges[:, 1])),
                          shape=(endo.n_points,) * 2).tocsr()
    d = dijkstra(graph, directed=False, indices=loop, min_only=True)
    return d[tri].mean(1)


def seeds_from_openings(endo: pv.PolyData,
                        mapping: Optional[Mapping[int, object]]
                        ) -> Dict[str, np.ndarray]:
    """Seed positions ccdaf can tell from the recorded openings.

    Each opening of *endo* is a loop whose nodes carry, in ``openingTag``,
    the cap or valve that closed it. A cap named after a vein or the
    appendage ("LSPV cap", "LAA cap") gives that seed; a valve gives "MV"
    or "TV". The seed is the node nearest the loop's centre among those
    about half the opening's radius inside it, measured along the surface:
    inside the vein, clear of its edge. Openings the names do not explain
    give nothing, and those seeds are left to the user.
    """
    pts = np.asarray(endo.points, dtype=float)
    tri = np.asarray(endo.faces).reshape(-1, 4)[:, 1:]
    edges = np.unique(np.sort(np.concatenate(
        [tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [0, 2]]]), axis=1), axis=0)
    w = np.maximum(np.linalg.norm(pts[edges[:, 1]] - pts[edges[:, 0]], axis=1),
                   1e-12)
    graph = sp.coo_matrix((w, (edges[:, 0], edges[:, 1])),
                          shape=(endo.n_points,) * 2).tocsr()
    seeds: Dict[str, np.ndarray] = {}
    for name, loop in opening_loops(endo, mapping).items():
        centre = pts[loop].mean(0)
        radius = float(np.median(np.linalg.norm(pts[loop] - centre, axis=1)))
        dist = dijkstra(graph, directed=False, indices=loop, min_only=True,
                        limit=radius * 2)
        ring = np.where(np.abs(dist - _SEED_DEPTH * radius)
                        < 0.25 * radius)[0]
        if name in ("MV", "TV") or len(ring) == 0:
            ring = loop
        best = ring[np.argmin(np.linalg.norm(pts[ring] - centre, axis=1))]
        seeds[name] = pts[best]
    return seeds


@dataclass
class VolumeTags:
    """What :func:`apply_to_volume` produced."""

    #: The new ``elemTag`` of every tetrahedron.
    tags: np.ndarray
    #: Labels carried into the wall, ascending.
    labels: List[int]
    #: Share of the endocardium's area whose tetrahedron got the face's label.
    agreement: float
    #: Tetrahedra moved out of islands, per label.
    islands: Dict[int, int] = field(default_factory=dict)

    def summary(self) -> str:
        names = ", ".join(al.NAMES.get(t, str(t)) for t in self.labels)
        text = (f"Labels carried into the wall: {names}. The tetrahedra "
                f"behind the endocardium agree with it on "
                f"{100 * self.agreement:.1f}% of its area.")
        moved = {t: n for t, n in self.islands.items() if n}
        if moved:
            text += " Islands joined to their neighbours: " + ", ".join(
                f"{al.NAMES.get(t, t)} {n}" for t, n in moved.items()) + "."
        return text


def _node_labels(n_nodes: int, tri: np.ndarray, tags: np.ndarray
                 ) -> np.ndarray:
    """Each node's label: the one most of its faces carry (-2: none)."""
    labels = np.unique(tags)
    count = np.zeros((n_nodes, len(labels)), dtype=np.int32)
    col = np.searchsorted(labels, tags)
    for k in range(3):
        np.add.at(count, (tri[:, k], col), 1)
    out = np.full(n_nodes, -2, dtype=np.int64)
    has = count.sum(1) > 0
    out[has] = labels[count[has].argmax(1)]
    return out


def _drop_islands(tags: np.ndarray, a: np.ndarray, b: np.ndarray,
                  movable: np.ndarray) -> Dict[int, int]:
    """Keep each label's largest piece; move every other piece to the label
    of the main pieces it shares most faces with. Works in place.

    Only main pieces are candidates, so two small islands touching each
    other cannot swap labels back and forth; an island touching no main
    piece waits for a neighbour to be resolved first.
    """
    moved: Dict[int, int] = {}
    n = len(tags)
    pairs = np.concatenate([np.stack([a, b], 1), np.stack([b, a], 1)])
    for _round in range(20):
        same = tags[a] == tags[b]
        graph = sp.coo_matrix((np.ones(int(same.sum())), (a[same], b[same])),
                              shape=(n, n))
        _k, comp = connected_components(graph, directed=False)
        main = np.zeros(n, dtype=bool)
        for label in np.unique(tags[movable]):
            mine = movable & (tags == label)
            sizes = np.bincount(comp[mine])
            main |= mine & (comp == int(np.argmax(sizes)))
        main |= ~movable
        stray = ~main
        if not stray.any():
            break
        edge = pairs[stray[pairs[:, 0]] & main[pairs[:, 1]]]
        if len(edge) == 0:
            break
        changed = False
        owner_comp = comp[edge[:, 0]]
        for c in np.unique(owner_comp):
            hit_pairs = edge[owner_comp == c]
            vals, cnt = np.unique(tags[hit_pairs[:, 1]], return_counts=True)
            new = int(vals[np.argmax(cnt)])
            hit = stray & (comp == c)
            old = int(tags[np.argmax(hit)])
            if new != old:
                tags[hit] = new
                moved[old] = moved.get(old, 0) + int(hit.sum())
                changed = True
        if not changed:
            break
    return moved


def apply_to_volume(grid, endo: pv.PolyData, atrium: str = "LA", *,
                    rtol: float = 1e-8) -> VolumeTags:
    """Carry the labels drawn on *endo* (from :func:`endocardium`) into
    *grid*'s tetrahedra by harmonic extension. *grid* is not modified."""
    grid = pv.wrap(grid)
    tets = tetrahedra(grid)
    points = np.asarray(grid.points, dtype=float)
    tri_local = np.asarray(endo.faces).reshape(-1, 4)[:, 1:]
    vnode = np.asarray(endo.point_data[VOLUME_NODE], dtype=np.int64)
    tri = vnode[tri_local]
    face_tags = np.asarray(endo.cell_data["elemTag"]).astype(int)
    # Triangles left unassigned (tagging not accepted yet) or carrying a
    # label of no region of this atrium are body, as Accept tagging makes
    # them: an "unassigned" region must not be carried into the wall.
    own = al.LA if atrium == "LA" else al.RA
    face_tags = np.where(np.isin(face_tags, own.values), face_tags, own.body)

    node_label = _node_labels(grid.n_points, tri, face_tags)
    held = np.unique(tri)
    values = {int(t): node_label[held] == t for t in np.unique(face_tags)}

    # The other atrium's endocardium, held at 1 in its own problem so its
    # wall keeps what it has.
    other = [a for a in endocardium_present(grid) if a != atrium]
    if other:
        surface, labels = _boundary_labels(grid)
        origin = np.asarray(surface.point_data["vtkOriginalPointIds"],
                            dtype=np.int64)
        stri = np.asarray(surface.faces).reshape(-1, 4)[:, 1:]
        their = np.setdiff1d(np.unique(origin[stri[labels == ENDO_OF[other[0]]]]),
                             held)
        held = np.concatenate([held, their])
        values = {t: np.concatenate([v, np.zeros(len(their), bool)])
                  for t, v in values.items()}
        values[_OTHER] = np.concatenate([np.zeros(len(held) - len(their), bool),
                                         np.ones(len(their), bool)])

    order = np.argsort(held)
    held = held[order]
    system = DirichletSystem(stiffness_matrix(points, tets), held)
    keys = sorted(values)
    score = np.empty((len(keys), len(tets)))
    for i, key in enumerate(keys):
        u, _report = system.solve(values[key][order].astype(float), rtol=rtol)
        score[i] = u[tets].mean(1)
    winner = np.asarray(keys)[score.argmax(0)]

    old = np.asarray(grid.cell_data["elemTag"]).astype(int)
    if other:
        # The other atrium keeps its labels, or takes its body when what
        # it has is not in its own scheme (an untagged volume's default).
        theirs = al.LA if other[0] == "LA" else al.RA
        old = np.where(np.isin(old, theirs.values), old, theirs.body)
    new = np.where(winner == _OTHER, old, winner).astype(np.int32)
    movable = winner != _OTHER
    _faces, a, b = _shared_faces(tets)
    islands = _drop_islands(new, a, b, movable)

    cells = np.asarray(endo.cell_data[VOLUME_CELL], dtype=np.int64)
    area = np.asarray(endo.compute_cell_sizes(length=False, volume=False)
                      .cell_data["Area"], dtype=float)
    agree = float(area[new[cells] == face_tags].sum() / (area.sum() or 1.0))
    return VolumeTags(tags=new, labels=[k for k in keys if k != _OTHER],
                      agreement=agree, islands=islands)


__all__ = ["VOLUME_NODE", "VOLUME_CELL", "endocardium_present", "endocardium",
           "BAND_MM", "band_openings", "opening_loops", "seeds_from_openings", "VolumeTags", "apply_to_volume"]
