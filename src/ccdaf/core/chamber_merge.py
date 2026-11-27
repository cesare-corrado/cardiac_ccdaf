"""
chamber_merge
=============
Put what was computed on extracted chambers back into the mesh they came
from.

An atrium is extracted from a four-chamber mesh
(:func:`ccdaf.core.chamber_extract.extract_chambers`), its regions tagged
and its UAC and fibres computed. :func:`merge_into` writes those results
onto the same elements of the four-chamber mesh. The atrium's elements
must be the source mesh's elements, unchanged: a remeshed or cleaned
atrium is refused.

Finding the elements
--------------------
Extraction stores, for each point and element, its id in the source mesh
(:data:`SOURCE_POINT`, :data:`SOURCE_CELL`). The merge uses them when they
are there and still right (each point at its source point's place, each
element on the same nodes). Otherwise, as for an atrium extracted before
the ids were stored, points are found by their coordinates and elements
by their nodes.

The fields
----------
Every field of the atrium is written on its rows. A field the source mesh
already has under that name (one ``fiber`` array for the whole heart) keeps
its values elsewhere; a new field reads "not computed" elsewhere: NaN for
decimals, :data:`NOT_COMPUTED` for integers. On a node the atrium shares
with tissue outside it (the interface), a value already there is kept: so
merging the left atrium and then the right, the nodes they share keep the
left atrium's α. Fields describing the atrium alone (its rims, its surface
labels, its source tags) are not written.

The tags
--------
The atrial regions (body, veins, appendage) are the same entities as the
source mesh's tags, numbered differently, and the region growing moves
their borders into the body. So the regions are written into the source
``elemTag`` in its own numbering, by name: an element of the atrium's body
tag that the tagging put in a vein takes the tag of that vein's tissue.
Any other element keeps its tag, including a layer such as the
interatrial bundle where it lies in a vein's region; so does a region
with no tag in the source mesh.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import pyvista as pv

from ccdaf.core import atrial_labels, tag_map
from ccdaf.core.chamber_extract import SOURCE_CELL, SOURCE_POINT
from ccdaf.core.volume_mesh import tetrahedra

#: Cell field and field-data entry the volume tagging keeps the source
#: tags in.
SOURCE_TAG: str = "sourceTag"
SOURCE_TAG_MAP: str = "sourceTagMap"
#: Integer fields read this where the atrium did not compute them.
NOT_COMPUTED: int = -32000

#: Fields that describe the extracted chambers alone.
_SKIP = frozenset({
    SOURCE_POINT, SOURCE_CELL, SOURCE_TAG, "elemTag", "openingTag",
    "surfaceLabelMask", "surfaceLabel", "volumeNode", "volumeCell",
    "vtkOriginalPointIds", "vtkOriginalCellIds", "render_idx",
})


class MergeError(ValueError):
    """The chambers cannot be merged into this mesh."""


@dataclass
class Merge:
    """What :func:`merge_into` produced."""

    grid: pv.UnstructuredGrid
    #: "ids" or "coordinates".
    method: str
    n_points: int
    n_cells: int
    point_fields: List[str] = field(default_factory=list)
    cell_fields: List[str] = field(default_factory=list)
    #: Interface nodes that kept a value already there.
    kept_nodes: int = 0
    #: Body elements given a vein or appendage tag: {new tag: count}.
    retagged: Dict[int, int] = field(default_factory=dict)
    #: Elements of other tags whose region differs, which kept their tag.
    kept_tags: int = 0
    #: Region names with no tag in the source mesh: {name: elements}.
    no_tag: Dict[str, int] = field(default_factory=dict)
    #: Elements whose source tag no longer matches the mesh's.
    tag_mismatch: int = 0

    def summary(self) -> str:
        how = ("their stored ids" if self.method == "ids"
               else "their coordinates")
        lines = [f"Merged {self.n_cells:,} elements and {self.n_points:,} "
                 f"points, found by {how}."]
        names = self.point_fields + self.cell_fields
        lines.append("Fields written: " + (", ".join(names) or "none") + ".")
        if self.kept_nodes:
            lines.append(f"{self.kept_nodes:,} nodes shared with tissue "
                         "already holding values kept them.")
        if self.retagged:
            lines.append("Body elements retagged: " + ", ".join(
                f"{n:,} to {t}" for t, n in sorted(self.retagged.items())) + ".")
        if self.kept_tags:
            lines.append(f"{self.kept_tags:,} elements of other tags lie in "
                         "another region and kept their tag.")
        for name, n in sorted(self.no_tag.items()):
            lines.append(f"Region {name} has no tag in this mesh: its "
                         f"{n:,} elements kept theirs.")
        if self.tag_mismatch:
            lines.append(f"Warning: {self.tag_mismatch:,} elements carry a "
                         "different tag in this mesh than when extracted.")
        return "\n".join(lines)


def _tolerance(points: np.ndarray) -> float:
    span = np.ptp(points, axis=0) if len(points) else np.zeros(3)
    return 1e-6 * max(float(np.linalg.norm(span)), 1e-12)


def _by_ids(source: pv.UnstructuredGrid, part: pv.UnstructuredGrid,
            src_tets: np.ndarray, part_tets: np.ndarray
            ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """The stored ids, if present and still right."""
    if SOURCE_POINT not in part.point_data or SOURCE_CELL not in part.cell_data:
        return None
    pid = np.asarray(part.point_data[SOURCE_POINT]).astype(np.int64)
    cid = np.asarray(part.cell_data[SOURCE_CELL]).astype(np.int64)
    if (pid.min(initial=0) < 0 or pid.max(initial=0) >= source.n_points
            or cid.min(initial=0) < 0 or cid.max(initial=0) >= len(src_tets)
            or len(np.unique(pid)) != len(pid)):
        return None
    S = np.asarray(source.points, dtype=float)
    P = np.asarray(part.points, dtype=float)
    if np.max(np.linalg.norm(S[pid] - P, axis=1), initial=0) > _tolerance(S):
        return None
    if not np.array_equal(np.sort(src_tets[cid], 1),
                          np.sort(pid[part_tets], 1)):
        return None
    return pid, cid


def _by_coordinates(source: pv.UnstructuredGrid, part: pv.UnstructuredGrid,
                    src_tets: np.ndarray, part_tets: np.ndarray
                    ) -> Tuple[np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree

    S = np.asarray(source.points, dtype=float)
    P = np.asarray(part.points, dtype=float)
    dist, pid = cKDTree(S).query(P)
    lost = int(np.sum(dist > _tolerance(S)))
    if lost:
        raise MergeError(f"{lost:,} of {len(P):,} points are not in this mesh: "
                         "it is not the mesh the chambers came from, or they "
                         "were remeshed or cleaned since.")
    inside = np.zeros(source.n_points, dtype=bool)
    inside[pid] = True
    cand = np.where(inside[src_tets].all(1))[0]
    keys = np.concatenate([np.sort(src_tets[cand], 1), np.sort(pid[part_tets], 1)])
    _u, cls = np.unique(keys, axis=0, return_inverse=True)
    cls = cls.ravel()
    owner = np.full(cls.max() + 1 if len(cls) else 0, -1, dtype=np.int64)
    owner[cls[:len(cand)]] = cand
    cid = owner[cls[len(cand):]]
    lost = int(np.sum(cid < 0))
    if lost:
        raise MergeError(f"{lost:,} of {len(part_tets):,} elements are not in "
                         "this mesh: the chambers were remeshed or cleaned "
                         "since they were extracted.")
    return pid, cid


def _target(attr, n: int, name: str, values: np.ndarray) -> Optional[np.ndarray]:
    """The array *name* will be written into: the mesh's own when it has
    one of that shape, else a new one reading "not computed"."""
    if values.dtype.kind not in "biuf":
        return None
    shape = (n,) + values.shape[1:]
    if name in attr:
        old = np.asarray(attr[name])
        if old.shape == shape and old.dtype.kind in "biuf":
            return old.astype(np.result_type(old.dtype, values.dtype), copy=True)
    if values.dtype.kind == "f":
        return np.full(shape, np.nan, dtype=values.dtype)
    return np.full(shape, NOT_COMPUTED,
                   dtype=np.result_type(values.dtype, np.int32))


def _empty(values: np.ndarray) -> np.ndarray:
    """Per row, whether *values* reads "not computed"."""
    if values.dtype.kind == "f":
        bad = ~np.isfinite(values)
    else:
        bad = values == NOT_COMPUTED
    return bad.reshape(len(values), -1).all(1)


def _source_map(source, part) -> Optional[tag_map.TagMap]:
    named = tag_map.read(source)
    if named:
        return named
    if SOURCE_TAG_MAP in part.field_data:
        try:
            return tag_map.decode(str(np.asarray(
                part.field_data[SOURCE_TAG_MAP]).ravel()[0]))
        except (ValueError, TypeError, IndexError):
            return None
    return None


def _retag(merge: Merge, tags: np.ndarray, cid: np.ndarray,
           region: np.ndarray, source_tag: np.ndarray,
           named: Mapping[int, object]) -> None:
    """Write the regions into *tags* (the source elemTag), in place."""
    merge.tag_mismatch = int(np.sum(tags[cid] != source_tag))
    of_tag = {int(t): tag_map.atrial_region(v) for t, v in named.items()}
    values, inverse = np.unique(source_tag, return_inverse=True)
    own_region = np.array([of_tag.get(int(t), "") for t in values],
                          dtype=object)[inverse.ravel()]
    for atrium in atrial_labels.ATRIA:
        body_name = atrial_labels.NAMES[atrium.body]
        bodies = [t for t, v in named.items()
                  if tag_map.entry(v).role == tag_map.WALL
                  and tag_map.entry(v).chambers == (atrium.name,)]
        for name, label in atrium.regions.items():
            here = region == label
            if not here.any():
                continue
            tissue = [t for t, r in of_tag.items()
                      if r == name and tag_map.entry(named[t]).role
                      in (tag_map.VEIN, tag_map.APPENDAGE)]
            on_body = here & np.isin(source_tag, bodies)
            others = here & ~on_body & (own_region != name)
            merge.kept_tags += int(others.sum())
            if not on_body.any():
                continue
            if not tissue:
                merge.no_tag[name] = int(on_body.sum())
                continue
            # Two tags for one vein: the one most of the region already has.
            pick = max(tissue, key=lambda t: int(np.sum(here & (source_tag == t))))
            tags[cid[on_body]] = pick
            merge.retagged[pick] = merge.retagged.get(pick, 0) + int(on_body.sum())
        # Elements of other tags the tagging put in the body keep theirs too.
        here = region == atrium.body
        merge.kept_tags += int(np.sum(here & ~np.isin(source_tag, bodies)
                                      & ~np.isin(own_region, ["", body_name])))


def merge_into(source, part) -> Merge:
    """*source* with the fields and regions of *part* written onto it.

    *source* is the mesh *part* was extracted from; neither is modified.
    Raises :class:`MergeError` when *part*'s elements are not in *source*.
    """
    source, part = pv.wrap(source), pv.wrap(part)
    for name, g in (("mesh", source), ("chambers", part)):
        if not isinstance(g, pv.UnstructuredGrid) or g.n_cells == 0:
            raise MergeError(f"the {name} is not a tetrahedral volume")
    try:
        src_tets, part_tets = tetrahedra(source), tetrahedra(part)
    except ValueError as exc:
        raise MergeError(str(exc)) from exc
    if len(src_tets) != source.n_cells or len(part_tets) != part.n_cells:
        raise MergeError("both meshes must hold tetrahedra only")
    if len(part_tets) > len(src_tets):
        raise MergeError("the chambers have more elements than the mesh")

    found = _by_ids(source, part, src_tets, part_tets)
    method = "ids"
    if found is None:
        found = _by_coordinates(source, part, src_tets, part_tets)
        method = "coordinates"
    pid, cid = found
    if len(np.unique(cid)) != len(cid):
        raise MergeError("two elements of the chambers fall on one element "
                         "of the mesh")

    grid = source.copy(deep=True)
    merge = Merge(grid=grid, method=method, n_points=len(pid), n_cells=len(cid))

    outside = np.ones(len(src_tets), dtype=bool)
    outside[cid] = False
    shared = np.zeros(source.n_points, dtype=bool)
    shared[src_tets[outside].ravel()] = True
    interface = shared[pid]

    kept = np.zeros(len(pid), dtype=bool)
    for name in part.point_data.keys():
        if name in _SKIP:
            continue
        values = np.asarray(part.point_data[name])
        out = _target(grid.point_data, grid.n_points, name, values)
        if out is None:
            continue
        keep = interface & ~_empty(out[pid])
        kept |= keep
        out[pid[~keep]] = values[~keep]
        grid.point_data[name] = out
        merge.point_fields.append(name)
    merge.kept_nodes = int(kept.sum())

    for name in part.cell_data.keys():
        if name in _SKIP:
            continue
        values = np.asarray(part.cell_data[name])
        out = _target(grid.cell_data, grid.n_cells, name, values)
        if out is None:
            continue
        out[cid] = values
        grid.cell_data[name] = out
        merge.cell_fields.append(name)

    if "elemTag" in part.cell_data and "elemTag" in grid.cell_data:
        tags = np.asarray(grid.cell_data["elemTag"]).copy()
        region = np.asarray(part.cell_data["elemTag"]).astype(int)
        named = _source_map(source, part)
        if SOURCE_TAG in part.cell_data and named:
            source_tag = np.asarray(part.cell_data[SOURCE_TAG]).astype(int)
            _retag(merge, tags, cid, region, source_tag, named)
            if not tag_map.read(grid):
                tag_map.write(grid, named)
        else:
            # Not region-tagged: the tags are the source mesh's own.
            merge.tag_mismatch = int(np.sum(tags[cid] != region))
            if not tag_map.read(grid) and tag_map.read(part):
                tag_map.write(grid, tag_map.read(part))
        grid.cell_data["elemTag"] = tags
        grid.set_active_scalars("elemTag", preference="cell")
    return merge


__all__ = ["SOURCE_POINT", "SOURCE_CELL", "NOT_COMPUTED", "Merge",
           "MergeError", "merge_into"]
