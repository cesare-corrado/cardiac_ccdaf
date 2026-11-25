"""
chamber_extract
===============
Keep the tissue of some chambers of a tagged volume, and remember where
its openings were.

A four-chamber mesh holds the walls of all four chambers, the valve
planes, the discs that close the veins and the appendage, and the great
vessels, each under its own tag. An atrial workflow needs the atrial wall
alone. The tag map (:mod:`ccdaf.core.tag_map`) says which tags are tissue
of which chamber, so the wall is every tag whose chamber is one of those
asked for and whose role is tissue (wall, layer, vein, appendage). Tissue
on two chambers (the interatrial bundle, "LA+RA") comes with either.

Tissue on two chambers
----------------------
A tag on two chambers (the interatrial bundle, "LA+RA") is split when only
one of them is asked for. In the four-chamber cohort that tag is
full-thickness wall over the roof, the septum and round the right veins,
and lines about a quarter of the left atrial endocardium, so each of its
tetrahedra goes to the atrium **whose blood it lines**: both atria are
labelled together first (:func:`ccdaf.core.atrial_surfaces.label_atria`),
the tag's tetrahedra on each endocardium are the seeds, and every other
one goes to the seed nearer along the tag (shortest path between
tetrahedron centres). The cut then runs inside the wall, like the septum
between the two bodies, and is epicardium of each atrium. Measured on one
four-chamber mesh: each atrium's endocardium is the same, face for face,
extracted alone or with the other.

When the chambers cannot be labelled together (they are not atria, or the
labelling fails), the seeds are the tag's tetrahedra touching tissue of one
chamber alone. A tetrahedron no path reaches stays with the chamber asked
for.

Remembering the openings
------------------------
The caps and valves are dropped, but where they sat is exactly where the
endocardium meets the epicardium: the rim of each opening. That is lost
once they are gone, so each node of a face the kept tissue shared with a
dropped cap or valve gets that cap's tag in the node field
:data:`OPENING_FIELD` (0 elsewhere). A boundary face whose three nodes
carry it is a rim face, and the rims then split the boundary into
endocardium and epicardium without any geometry (see
:mod:`ccdaf.core.atrial_surfaces`). Measured on a four-chamber mesh of each
of two cohorts, for each atrium alone and for both: the rims leave exactly
one epicardium and one endocardium per atrium, plus at most three stray
pieces of 4 faces.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pyvista as pv
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

from ccdaf.core import tag_map
from ccdaf.core.volume_mesh import tetrahedra

#: Node field holding, for each node on a dropped cap or valve, its tag.
OPENING_FIELD: str = "openingTag"

#: Chambers in the order the dialog lists them.
CHAMBER_ORDER: Tuple[str, ...] = ("LV", "RV", "LA", "RA")

#: Arrays that describe the old mesh's boundary and stop being true once
#: tissue is removed.
_STALE_POINT_FIELDS: Tuple[str, ...] = ("surfaceLabelMask", "surfaceLabel",
                                        OPENING_FIELD)


@dataclass
class Extraction:
    """What :func:`extract_chambers` kept."""

    grid: pv.UnstructuredGrid
    chambers: Tuple[str, ...]
    #: Tags kept as tissue, ascending.
    tissue_tags: List[int]
    #: Dropped cap and valve tags that left a rim on the kept tissue.
    opening_tags: List[int]
    #: How many tetrahedra were kept, and how many the input had.
    kept: int
    total: int
    #: For each tag shared with a chamber not kept: (kept, dropped) tets.
    split: Dict[int, Tuple[int, int]] = None

    def __post_init__(self) -> None:
        if self.split is None:
            self.split = {}

    def summary(self) -> str:
        text = (f"Kept {', '.join(self.chambers)}: tags "
                f"{', '.join(map(str, self.tissue_tags))}, {self.kept:,} of "
                f"{self.total:,} tetrahedra. Rims recorded at tags "
                f"{', '.join(map(str, self.opening_tags)) or 'none'}.")
        for tag, (kept, dropped) in sorted(self.split.items()):
            text += (f" Shared tag {tag} split: {kept:,} tetrahedra kept, "
                     f"{dropped:,} left to the other chamber.")
        return text


def available_chambers(mapping: Mapping[int, object] | None,
                       tags: Iterable[int]) -> List[str]:
    """The chambers with tissue among *tags*, in :data:`CHAMBER_ORDER`."""
    present = {int(t) for t in tags}
    named = {t: v for t, v in tag_map.normalise(mapping).items() if t in present}
    return [c for c in CHAMBER_ORDER if tag_map.tags_of(named, c)]


def _shared_faces(tets: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every face two tetrahedra share: ``(face nodes, tet a, tet b)``."""
    faces = np.sort(np.concatenate([tets[:, [0, 1, 2]], tets[:, [0, 1, 3]],
                                    tets[:, [0, 2, 3]], tets[:, [1, 2, 3]]]), 1)
    owner = np.tile(np.arange(len(tets)), 4)
    order = np.lexsort(faces.T[::-1])
    faces, owner = faces[order], owner[order]
    same = np.all(faces[1:] == faces[:-1], axis=1)
    return faces[:-1][same], owner[:-1][same], owner[1:][same]


def extract_chambers(grid, mapping: Mapping[int, object],
                     chambers: Sequence[str], *,
                     _keep_ids: bool = False) -> Extraction:
    """The tissue of *chambers* in *grid*, with its rims recorded.

    *grid* is a tetrahedral volume with ``elemTag``; it is not modified.
    The result keeps every cell and point array of the kept tissue, and
    the tag map. Raises ``ValueError`` when nothing would be kept.
    """
    grid = pv.wrap(grid)
    if "elemTag" not in grid.cell_data:
        raise ValueError("the mesh has no element tags (elemTag)")
    named = tag_map.normalise(mapping)
    chambers = tuple(c for c in CHAMBER_ORDER if c in set(chambers))
    if not chambers:
        raise ValueError("no chamber was chosen")
    tissue = sorted({t for c in chambers for t in tag_map.tags_of(named, c)})
    openings = sorted({t for c in chambers for t in
                       tag_map.tags_of(named, c, (tag_map.CAP, tag_map.VALVE))}
                      - set(tissue))
    tags = np.asarray(grid.cell_data["elemTag"]).astype(int)
    keep = np.isin(tags, tissue)
    if not keep.any():
        raise ValueError(f"no tetrahedra carry the tissue tags of "
                         f"{', '.join(chambers)}")

    tets = tetrahedra(grid)
    faces, a, b = _shared_faces(tets)
    split: Dict[int, Tuple[int, int]] = {}
    shared = [t for t in tissue
              if set(named[t].chambers) - set(chambers)]
    if shared:
        centres = np.asarray(grid.cell_centers().points, dtype=float)
        lining = _lining(grid, named, chambers, shared)
        for t in shared:
            owner = _split_shared(t, named, tags, a, b, centres, lining)
            mine = tags == t
            drop = mine & ~np.isin(owner, chambers + ("",))
            keep &= ~drop
            split[t] = (int((mine & keep).sum()), int(drop.sum()))

    node_opening = np.zeros(grid.n_points, dtype=np.int32)
    left: List[int] = []
    if openings:
        is_open = np.isin(tags, openings)
        for x, y in ((a, b), (b, a)):
            hit = keep[x] & is_open[y]
            if hit.any():
                node_opening[faces[hit].ravel()] = np.repeat(tags[y[hit]], 3)
        left = sorted({int(v) for v in np.unique(node_opening) if v})

    sub = grid.extract_cells(np.where(keep)[0])
    origin = np.asarray(sub.point_data["vtkOriginalPointIds"], dtype=np.int64)
    for name in ("vtkOriginalPointIds",):
        del sub.point_data[name]
    if "vtkOriginalCellIds" in sub.cell_data:
        if _keep_ids:
            sub.cell_data[_SOURCE_CELL] = sub.cell_data["vtkOriginalCellIds"]
        del sub.cell_data["vtkOriginalCellIds"]
    for name in _STALE_POINT_FIELDS:
        if name in sub.point_data:
            del sub.point_data[name]
    sub.point_data[OPENING_FIELD] = node_opening[origin]
    tag_map.write(sub, named)
    return Extraction(grid=sub, chambers=chambers, tissue_tags=tissue,
                      opening_tags=left, kept=int(keep.sum()),
                      total=int(len(tags)), split=split)


_SOURCE_CELL = "_sourceCell"


def _lining(grid, named: tag_map.TagMap, chambers: Sequence[str],
            shared: Sequence[int]) -> Dict[str, np.ndarray]:
    """For atria, the tets (ids in *grid*) lining each atrium's
    endocardium; empty when that cannot be told.

    Found by labelling every chamber the shared tags touch at once, which
    needs no split. A tet whose face lines an atrium's blood is that
    atrium's, whatever its tag.
    """
    from ccdaf.core.atrial_surfaces import ENDO_OF, label_atria
    from ccdaf.core.surface_labels import _as_boundary

    union = sorted(set(chambers) | {c for t in shared
                                    for c in named[t].chambers})
    if not set(union) <= set(ENDO_OF):
        return {}
    try:
        whole = extract_chambers(grid, named, union, _keep_ids=True)
        labels = label_atria(whole.grid, named)
    except ValueError:
        return {}
    surface = _as_boundary(whole.grid)
    source = np.asarray(whole.grid.cell_data[_SOURCE_CELL], dtype=np.int64)
    owner = source[np.asarray(surface.cell_data["vtkOriginalCellIds"],
                              dtype=np.int64)]
    return {c: np.unique(owner[labels.face_labels == key])
            for c, key in ENDO_OF.items()
            if np.any(labels.face_labels == key)}


def _split_shared(tag: int, named: tag_map.TagMap, tags: np.ndarray,
                  a: np.ndarray, b: np.ndarray, centres: np.ndarray,
                  lining: Mapping[str, np.ndarray] | None = None
                  ) -> np.ndarray:
    """Per tetrahedron, the chamber each tet of *tag* belongs to ("" for
    the others, and for a tet of *tag* no path reaches).

    Seeds are the tets of *tag* lining a chamber's endocardium, when
    *lining* gives them for every chamber of *tag*; otherwise the tets of
    *tag* touching tissue of one chamber alone. The distance is the
    shortest path between tet centres inside *tag*.
    """
    options = named[tag].chambers
    own = {c: [t for t in tag_map.tags_of(named, c)
               if named[t].chambers == (c,)] for c in options}
    index = np.where(tags == tag)[0]
    pos = np.full(len(tags), -1)
    pos[index] = np.arange(len(index))
    inner = (tags[a] == tag) & (tags[b] == tag)
    w = np.linalg.norm(centres[a[inner]] - centres[b[inner]], axis=1)
    w = np.maximum(w, 1e-12)
    graph = sp.coo_matrix((w, (pos[a[inner]], pos[b[inner]])),
                          shape=(len(index),) * 2).tocsr()
    dist = np.full((len(options), len(index)), np.inf)
    use_lining = bool(lining) and all(
        c in lining and np.any(tags[lining[c]] == tag) for c in options)
    for i, c in enumerate(options):
        if use_lining:
            ids = lining[c]
            seeds = np.unique(pos[ids[tags[ids] == tag]])
        else:
            seeds = np.unique(np.concatenate(
                [pos[x[(tags[x] == tag) & np.isin(tags[y], own[c])]]
                 for x, y in ((a, b), (b, a))]))
        if len(seeds):
            dist[i] = dijkstra(graph, directed=False, indices=seeds,
                               min_only=True)
    out = np.full(len(tags), "", dtype=object)
    reached = np.isfinite(dist).any(0)
    out[index[reached]] = np.asarray(options, dtype=object)[
        dist[:, reached].argmin(0)]
    return out


__all__ = ["OPENING_FIELD", "CHAMBER_ORDER", "Extraction",
           "available_chambers", "extract_chambers"]
