"""
atrial_surfaces
===============
Name the surfaces of an atrial volume: the rim, the epicardium, and the
endocardium of each atrium present (left, right, or both).

An atrium is a ventricle turned inside out in one respect only: its
openings are many (the mitral or tricuspid valve, the veins, a clipped
appendage), and at each one the endocardium meets the epicardium across a
crown of faces one wall thick. That crown is the **rim** (stored as
:data:`~ccdaf.core.surface_labels.BASE`). Removing it leaves one
epicardium and one endocardium per atrium.

Two ways to find the rim
------------------------
**From the tags.** A volume taken out of a tagged mesh by
:func:`ccdaf.core.chamber_extract.extract_chambers` records, in the node
field ``openingTag``, the nodes of every face its tissue shared with a
dropped cap or valve. A boundary face whose three nodes carry it is a rim
face. No geometry is involved. Measured on a four-chamber mesh of each of
two cohorts, for the left atrium, the right and both: one epicardium, one
endocardium per atrium, and at most three stray pieces of 4 faces, which
join the rim.

**From the shape**, for a volume without that field: the openings are
found on voxels as for the ventricles
(:mod:`ccdaf.core.orifice_labels`): close the solid with a large ball so
each blood pool fills, keep where a pool meets the outside air, then cut
the shortest ring round each opening inside a narrow band. With N pools
the ring leaves N endocardia and one epicardium. The defaults
(:class:`AtrialOptions`) are finer than the ventricles': an atrial wall
is 1 to 3 mm thick and a vein opening 10 to 20 mm wide.

Naming
------
The epicardium is the piece nearest the convex hull. Each endocardium is
named from the element tags behind it, read through the tag map (or
ccdaf's own atrial scheme, when the mesh holds more than one tag: a mesh
loaded without tags has a single default tag, which proves nothing). Without either, it is a guess: the cavity
with more openings is the left atrium (four veins and the mitral valve,
against two caval veins and the tricuspid valve). A guess is reported as
uncertain, and :func:`swap_atria` exchanges the two names.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

from ccdaf.core import tag_map
from ccdaf.core.chamber_extract import OPENING_FIELD
from ccdaf.core.orifice_labels import (
    OrificeOptions, _close_ring, _components, _manifold_adjacency, _min_cut, _openings_in,
    _significant, _voxel_model,
)
from ccdaf.core.surface_labels import (
    ATRIAL_KEYS, BASE, EPI, LA_ENDO, RA_ENDO, UNLABELLED, SurfaceLabels,
    _as_boundary, _assemble, _faces, _areas, _hull_depth,
)

#: Endocardium label of each atrium.
ENDO_OF: Dict[str, int] = {"LA": LA_ENDO, "RA": RA_ENDO}

#: The share of the voted area that must lie on one atrium's tissue for
#: the name to count as certain.
_NAMING_PURITY: float = 0.9

#: The least share of an endocardium's area whose tags must name one
#: atrium alone for the vote to count.
_VOTED_SHARE: float = 0.5

#: An atrium with at least this many openings is guessed to be the left.
_LA_OPENINGS: int = 4


@dataclass
class AtrialOptions(OrificeOptions):
    """:class:`~ccdaf.core.orifice_labels.OrificeOptions` with atrial
    defaults. Used only when the volume has no recorded rims."""

    #: Half the thinnest atrial wall, so the wall stays solid on voxels.
    voxel_mm: float = 0.5
    #: An atrium holds 50 to 150 mL; anything under 5 mL is a pocket.
    min_pool_ml: float = 5.0
    #: A vein opening is 50 mm² or more; 5 mm² keeps the smallest.
    min_opening_mm2: float = 5.0
    #: Each endocardium must hold this share of the boundary area, or its
    #: pool was not filled and a larger closing ball is tried.
    min_endo_share: float = 0.12


def has_recorded_rims(dataset) -> bool:
    """Whether *dataset* carries rims from a chamber extraction."""
    data = pv.wrap(dataset)
    return (OPENING_FIELD in data.point_data
            and bool(np.any(np.asarray(data.point_data[OPENING_FIELD]))))


# ---------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------
def _atrium_of_tags(surface: pv.PolyData, mapping) -> Optional[np.ndarray]:
    """Per face, "LA", "RA" or "" from the tag behind it, or ``None``."""
    if "elemTag" not in surface.cell_data:
        return None
    tags = np.asarray(surface.cell_data["elemTag"]).astype(int)
    present = np.unique(tags)
    named = tag_map.normalise(mapping)
    if not named and len(present) > 1:
        # ccdaf's own atrial scheme, but only for a real tagging: a mesh
        # loaded without tags gets one default tag, the LA body's number,
        # which says nothing about which atrium it is.
        named = tag_map.propose(present)
    if not named:
        return None
    lookup = {}
    for t in np.unique(tags):
        e = named.get(int(t))
        atria = [c for c in (e.chambers if e else ()) if c in ENDO_OF]
        lookup[int(t)] = atria[0] if len(atria) == 1 else ""
    return np.array([lookup[int(t)] for t in tags], dtype=object)


def _name_endocardia(surface: pv.PolyData, pieces: List[np.ndarray],
                     area: np.ndarray, mapping, opening_counts: List[int]
                     ) -> Tuple[Dict[int, np.ndarray], bool, str]:
    """Which endocardium each piece is, whether that is certain, and why not."""
    atrium = _atrium_of_tags(surface, mapping)
    names: List[str] = []
    confident = True
    notes: List[str] = []
    for i, faces in enumerate(pieces):
        name = ""
        if atrium is not None:
            # Tissue shared by both atria (the interatrial bundle) has no
            # vote; what is left must be one atrium's, and be most of it.
            share = {a: float(area[faces][atrium[faces] == a].sum())
                     for a in ENDO_OF}
            voted = sum(share.values())
            best = max(share, key=share.get)
            if (voted >= _VOTED_SHARE * float(area[faces].sum())
                    and share[best] >= _NAMING_PURITY * voted):
                name = best
        names.append(name)
    if any(not n for n in names) or len(set(names)) != len(names):
        confident = False
        if len(pieces) == 2:
            first_left = opening_counts[0] >= opening_counts[1]
            names = ["LA", "RA"] if first_left else ["RA", "LA"]
        else:
            names = ["LA" if opening_counts[0] >= _LA_OPENINGS else "RA"]
        notes.append(
            "The element tags do not say which atrium each endocardium "
            "lines, so this is a guess from the number of openings (the "
            "left atrium has more). Swap the names if it is wrong.")
    return ({ENDO_OF[n]: p for n, p in zip(names, pieces)}, confident,
            " ".join(notes))


def swap_atria(labels: SurfaceLabels) -> SurfaceLabels:
    """*labels* with the left and right endocardia exchanged (or the one
    endocardium renamed), for a naming that was guessed wrong."""
    swap = {LA_ENDO: RA_ENDO, RA_ENDO: LA_ENDO}
    face = labels.face_labels.copy()
    for a, b in swap.items():
        face[labels.face_labels == a] = b
    node_sets = {swap.get(k, k): v for k, v in labels.node_sets.items()}
    areas = {k: labels.areas.get(swap.get(k, k), 0.0) for k in ATRIAL_KEYS}
    return replace(labels, face_labels=face, node_sets=node_sets, areas=areas,
                   naming_confident=True,
                   note=(labels.note + "\nThe atria were swapped by hand."
                         ).strip())


# ---------------------------------------------------------------------
# The two methods
# ---------------------------------------------------------------------
def _epicardia(surface, tri, fa, fb, pieces: np.ndarray,
               significant: np.ndarray) -> Tuple[List[int], int]:
    """The epicardial piece of each closed boundary, and how many there are.

    One per boundary: two atria that do not touch are two solids, each
    with its own epicardium. Each is the piece nearest that boundary's own
    convex hull.
    """
    _count, solid = _components(np.ones(len(tri), dtype=bool), fa, fb)
    points = np.asarray(surface.points, dtype=float)
    centres = points[tri].mean(1)
    epi: List[int] = []
    solids = sorted({int(solid[pieces == k][0]) for k in significant})
    for c in solids:
        mine = [k for k in significant if solid[pieces == k][0] == c]
        nodes = np.unique(tri[solid == c])
        depth = _hull_depth(pv.PolyData(points[nodes]), centres, 0)
        epi.append(min(mine, key=lambda k: np.median(depth[pieces == k])))
    return epi, len(solids)


def _finish(surface, tri, area, fa, fb, rim: np.ndarray, mapping,
            opening_counts_of, method: str, openings, note: str,
            expected: Optional[int]) -> SurfaceLabels:
    """Pieces left by *rim*, named; strays join the rim."""
    significant, pieces = _significant(~rim, fa, fb, area)
    epi, solids = (_epicardia(surface, tri, fa, fb, pieces, significant)
                   if len(significant) else ([], 0))
    n_endo = len(significant) - len(epi)
    if n_endo < 1 or (expected is not None and len(significant) != expected):
        want = (f"{expected}" if expected is not None
                else "an epicardium and at least one endocardium")
        raise ValueError(
            f"the rims leave {len(significant)} surface(s), not {want}: an "
            "opening is probably missing, or the wall has a hole.")
    if n_endo > 2:
        raise ValueError(f"the rims leave {n_endo} endocardia; at most two "
                         "atria are expected.")
    stray = (pieces >= 0) & ~np.isin(pieces, significant)
    rim = _close_ring(tri, rim | stray)

    epi_faces = np.where(np.isin(pieces, epi) & ~rim)[0]
    endo = [np.where((pieces == k) & ~rim)[0] for k in significant
            if k not in epi]
    named, confident, naming_note = _name_endocardia(
        surface, endo, area, mapping, opening_counts_of(endo))

    face_labels = np.full(len(tri), UNLABELLED, dtype=np.int32)
    face_labels[epi_faces] = EPI
    for key, faces in named.items():
        face_labels[faces] = key
    face_labels[rim] = BASE
    if stray.any():
        note = (note + f" {int(stray.sum())} stray face(s) beside a rim were "
                "added to it.").strip()
    note = (note + " " + naming_note).strip()
    return _assemble(surface, tri, face_labels, area, keys=ATRIAL_KEYS,
                     plane=None, naming_confident=confident, note=note,
                     dropped_patches=0, method=method, openings=openings)


def _label_from_tags(surface, tri, area, fa, fb, mapping) -> SurfaceLabels:
    marked = np.asarray(surface.point_data[OPENING_FIELD]) != 0
    rim = marked[tri].all(1)

    opening = np.asarray(surface.point_data[OPENING_FIELD]).astype(int)

    def counts(endo: List[np.ndarray]) -> List[int]:
        # Distinct openings touching each endocardium.
        out = []
        for faces in endo:
            near = np.unique(opening[tri[faces].ravel()])
            out.append(int((near != 0).sum()))
        return out

    return _finish(surface, tri, area, fa, fb, rim, mapping, counts,
                   method="tags", openings=[], note="", expected=None)


def _label_from_shape(dataset, surface, tri, area, fa, fb, mapping,
                      options: AtrialOptions) -> SurfaceLabels:
    model = _voxel_model(surface, tri, options)
    openings = _openings_in(model, options)
    if not openings:
        raise ValueError("no openings were found")
    points = np.asarray(surface.points, dtype=float)
    centres = points[tri].mean(1)
    width = options.band_mm / options.mm_per_unit
    distance = np.full(len(tri), np.inf)
    for opening in openings:
        near, _ = cKDTree(opening.points).query(centres,
                                                distance_upper_bound=width)
        distance = np.minimum(distance, near)
    band = distance < width
    pools = len(model.pool_ml)
    significant, pieces = _significant(~band, fa, fb, area)
    epi, solids = (_epicardia(surface, tri, fa, fb, pieces, significant)
                   if len(significant) else ([], 0))
    if len(significant) != pools + solids:
        raise ValueError(
            f"{pools} blood pool(s) and their openings leave "
            f"{len(significant)} surface(s), not {pools + solids}: an opening "
            "is probably missing, or the wall has a hole.")
    # The ring: the cheapest loop between the epicardium and the cavities,
    # found inside the band only.
    source = np.isin(pieces, epi)
    sink = np.isin(pieces, significant) & ~source
    free = ~(source | sink)
    _fa, _fb, shared = _manifold_adjacency(tri, len(points))
    lengths = np.linalg.norm(points[shared[:, 1]] - points[shared[:, 0]],
                             axis=1)
    capacity = np.maximum(1, np.round(lengths / lengths.mean() * 1000))
    epi_side = _min_cut(len(tri), fa, fb, capacity, free, source, sink)
    on_cut = np.zeros(len(points), dtype=bool)
    on_cut[shared[epi_side[fa] != epi_side[fb]].ravel()] = True
    rim = on_cut[tri].any(1)

    opening_centres = np.array([o.centre for o in openings], dtype=float)

    def counts(endo: List[np.ndarray]) -> List[int]:
        # Each opening counts for the endocardium nearest its centre.
        if not endo:
            return []
        d = np.stack([cKDTree(centres[f]).query(opening_centres)[0]
                      for f in endo])
        owner = d.argmin(0)
        return [int((owner == i).sum()) for i in range(len(endo))]

    return _finish(surface, tri, area, fa, fb, rim, mapping, counts,
                   method="openings", openings=list(openings), note="",
                   expected=pools + solids)


def label_atria(dataset, mapping: Optional[Mapping[int, object]] = None,
                options: Optional[AtrialOptions] = None) -> SurfaceLabels:
    """Label an atrial volume's boundary: rim, epicardium, LA and/or RA
    endocardium.

    Rims recorded by a chamber extraction are used when present; otherwise
    the openings are found on voxels, with a larger closing ball tried
    when an endocardium comes out too small (an opening not bridged).
    *mapping* names the element tags (a tag map); without it ccdaf's own
    atrial scheme is tried. *dataset* is not modified.
    """
    options = options or AtrialOptions()
    options.validate()
    surface = _as_boundary(dataset)
    tri = _faces(surface)
    if len(tri) == 0:
        raise ValueError("this mesh has no boundary faces to label")
    area = _areas(surface)
    fa, fb, _shared = _manifold_adjacency(tri, surface.n_points)
    if has_recorded_rims(surface):
        return _label_from_tags(surface, tri, area, fa, fb, mapping)

    failures: List[str] = []
    for radius in (options.closing_mm, *options.closing_retry_mm):
        attempt = replace(options, closing_mm=radius)
        try:
            labels = _label_from_shape(dataset, surface, tri, area, fa, fb,
                                       mapping, attempt)
        except ValueError as exc:
            failures.append(f"With a {radius:g} mm closing ball, {exc}")
            continue
        total = sum(labels.areas.values()) or 1.0
        small = [k for k in (LA_ENDO, RA_ENDO) if 0 < labels.areas.get(k, 0)
                 < options.min_endo_share * total]
        if small:
            failures.append(
                f"With a {radius:g} mm closing ball, an endocardium holds "
                f"under {100 * options.min_endo_share:g}% of the boundary "
                f"({labels.summary()}).")
            continue
        if failures:
            labels.note = ("\n".join(failures) + f"\nLabelled with a "
                           f"{radius:g} mm closing ball instead.\n\n"
                           + labels.note).strip()
        return labels
    raise ValueError("\n".join(failures))


__all__ = ["AtrialOptions", "ENDO_OF", "has_recorded_rims", "label_atria",
           "swap_atria"]
