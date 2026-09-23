"""
surface_labels
==============
Name the surfaces of a truncated ventricular volume: base, epicardium, LV
endocardium, RV endocardium.

Why a truncated mesh is handled here
------------------------------------
On a mesh that still has its valve orifices, the epicardium and both
endocardia are **one connected surface**, joined around each orifice. Six
local rules failed to split it on the example ventricle: normals against
an axis, flat-patch growing, global plane binning, crease splitting at
30/45/60 degrees, clipping at a guessed base, and sealing the orifices.
An orifice is as wide as the chamber it opens, so no local rule separates
the two sides. Sealing alone fills the whole base as one region; it takes
a second step, opening the filled region, to find each opening. That
method lives in :mod:`ccdaf.core.orifice_labels`, which
:func:`~ccdaf.core.orifice_labels.label_ventricles` falls back to when
this module's flat cut fails.

What works is removing the orifice region first. Once the mesh is
truncated, the boundary falls apart into exactly three pieces, and on the
example those pieces are 48%, 30% and 21% of the area. That is why this
module takes a *plane* and not a set of seeds: the plane is the
information the geometry does not contain.

The rule
--------
A boundary face belongs to the base when all three of its nodes lie within
``tol_edges`` mean edge lengths of the plane **and** its normal is within
``cos_min`` of parallel to the plane normal. Measured against a known cut:
that pair scores precision and recall 1.000, while the angle test alone
scores 0.960 and scatters the selection across 28 patches instead of 1,
pinning nodes to the wrong places. The distance test is what makes it
exact; the angle test guards the case where the surface runs tangent to
the plane.

Acceptance
----------
Removing the base must leave exactly three connected surfaces. This is not
a formality: it is the check that catches a plane in the wrong place, and
on a mesh that was never truncated it is what says so instead of letting a
meaningless Laplace solve proceed. It was also the check that caught a
*correct* base being scored against a mesh that still held an orifice.

Naming
------
The epicardium is the piece lying on the convex hull: on the example its
median depth beneath the hull is 0.05 against 3.55 and 4.26 for the two
cavities. The cavities are told apart by the tags of the tetrahedra behind
them, and the signature is anatomical rather than incidental: the LV cavity
is enclosed entirely by LV and septal tissue, which carry one tag, while
the RV cavity is bounded by the RV free wall *and* by the septum, so it
comes back mixed (measured 1.000 pure against 0.74/0.26 mixed). Note that
the obvious shortcut is wrong: the LV endocardium is the *smaller* of the
two on the example, 21% against 30%, so picking by area mislabels them.

Where the tags cannot decide, the naming is reported as unconfident rather
than guessed, and the caller is expected to ask.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyvista as pv
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull

from ccdaf.core.volume_mesh import boundary_surface, is_volume

#: Face and node labels. ``UNLABELLED`` exists so a partially labelled
#: mesh is representable; a complete labelling uses none of it.
UNLABELLED: int = 0
BASE: int = 1
EPI: int = 2
LV_ENDO: int = 3
RV_ENDO: int = 4

#: The array written to the mesh, on faces for display and on nodes for
#: the solve. One name for both so a user colouring by it sees the thing
#: the solver will read.
LABEL_FIELD: str = "surfaceLabel"

#: Per-node **membership** of each surface, as bit flags, written beside
#: :data:`LABEL_FIELD`. A node on a seam carries more than one bit: 3 is
#: base and epicardium at once.
#:
#: It exists because one label per node cannot describe a seam. Measured
#: on the truncated example, 1,162 nodes belong to two surfaces, so the
#: single-label form has to choose, and 1,775 of 45,138 faces then have
#: disagreeing nodes and come back unlabelled. Membership does not
#: choose, and recovers all 45,138 exactly.
#:
#: Deliberately a **separate name**: the value 3 means "LV endocardium"
#: as a label and "base + epicardium" as a mask, so reusing the name
#: would make an older file silently wrong.
LABEL_MASK_FIELD: str = "surfaceLabelMask"

#: The bit each surface owns in :data:`LABEL_MASK_FIELD`.
MASK_BITS: Dict[int, int] = {BASE: 1, EPI: 2, LV_ENDO: 4, RV_ENDO: 8}

NAMES: Dict[int, str] = {
    UNLABELLED: "unlabelled",
    BASE: "base",
    EPI: "epicardium",
    LV_ENDO: "LV endocardium",
    RV_ENDO: "RV endocardium",
}

#: How much of the best coplanar area a face must hold to count as a real
#: cut when snapping. It exists to keep a sliver from winning on proximity
#: alone, while still letting a smaller genuine cut beat a larger one that
#: the user was not pointing at.
_SNAP_AREA_SHARE: float = 0.5

#: Points used to build the convex hull the epicardium is recognised by.
#: The hull is only needed to say which piece is outermost, so a subsample
#: is ample and keeps a 6,000-facet hull from being built on every call.
_HULL_SAMPLE: int = 4000


@dataclass
class Plane:
    """An infinite plane, as a point on it and a unit normal."""

    origin: np.ndarray
    normal: np.ndarray

    def __post_init__(self) -> None:
        self.origin = np.asarray(self.origin, dtype=float).reshape(3)
        n = np.asarray(self.normal, dtype=float).reshape(3)
        length = float(np.linalg.norm(n))
        if length < 1e-12:
            raise ValueError("the plane normal has no length")
        self.normal = n / length

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        return (np.asarray(points, dtype=float) - self.origin) @ self.normal

    def describe(self) -> str:
        o = ", ".join(f"{v:.3f}" for v in self.origin)
        n = ", ".join(f"{v:.3f}" for v in self.normal)
        return f"point ({o}), normal ({n})"


@dataclass
class LabelOptions:
    """How the base is selected, and how strictly the result is judged."""

    #: Distance band, in multiples of the mean boundary edge length. A
    #: planar cut is exactly planar, so the value barely matters: 0.5 and
    #: 1.0 gave identical selections on the example.
    tol_edges: float = 0.5
    #: Minimum ``|n_plane . n_face|``. A cut face is parallel by
    #: construction, so this is generous.
    cos_min: float = 0.95
    #: Selected patches smaller than this share of the selected area are
    #: dropped. Without it the angle test drags in small parallel patches
    #: elsewhere on the surface.
    min_patch_fraction: float = 0.01
    #: Candidate planes tried when detecting the cut.
    samples: int = 600
    #: Seed for the detector's sampling, so a run is reproducible.
    seed: int = 0
    #: How far, in mean edge lengths, a snap looks for a real cut around a
    #: plane that selected nothing. Measured on the truncated example: a
    #: plane slid 1, 3 or 6 units off the cut snaps back onto it exactly,
    #: and at 10 the search honestly finds nothing rather than guessing.
    #: ``0`` turns snapping off.
    snap_edges: float = 8.0
    #: How far from parallel a face may be and still be considered when
    #: snapping. 45 degrees recovers the cut from a drag tilted by up to
    #: 40 degrees; 30 fails at 40.
    snap_cone_deg: float = 45.0

    def validate(self) -> None:
        if self.tol_edges <= 0.0:
            raise ValueError("the distance tolerance must be positive")
        if not 0.0 < self.cos_min <= 1.0:
            raise ValueError("cos_min must lie in (0, 1]")
        if not 0.0 <= self.min_patch_fraction < 1.0:
            raise ValueError("the patch fraction must lie in [0, 1)")
        if self.samples < 1:
            raise ValueError("at least one candidate plane must be tried")
        if self.snap_edges < 0.0:
            raise ValueError("the snap distance must not be negative")
        if not 0.0 < self.snap_cone_deg <= 90.0:
            raise ValueError("the snap cone must lie in (0, 90] degrees")


@dataclass
class SurfaceLabels:
    """The labelling, and everything needed to judge it."""

    #: One label per boundary face, in the boundary surface's own order.
    face_labels: np.ndarray
    #: Node ids (into the volume, or the surface when given one) per label.
    node_sets: Dict[int, np.ndarray]
    #: The plane the base came from. ``None`` when the base was found at
    #: the valve openings instead, which have no single plane.
    plane: Optional[Plane]
    #: Area held by each label.
    areas: Dict[int, float]
    #: False when the tags could not tell the two cavities apart, in which
    #: case LV_ENDO and RV_ENDO are a guess and the caller should ask.
    naming_confident: bool
    #: What made the naming uncertain, when it is.
    note: str = ""
    #: Faces dropped as small stray patches by ``min_patch_fraction``.
    dropped_patches: int = 0
    #: How the base was found: ``"plane"`` (a truncated mesh's flat cut)
    #: or ``"openings"`` (rings at the valve openings).
    method: str = "plane"
    #: The openings the rings were built at, for the ``"openings"`` method.
    #: Items are :class:`ccdaf.core.orifice_labels.Opening`.
    openings: List[Any] = field(default_factory=list)
    #: Where the labels were cut through holes in the wall, for the
    #: ``"openings"`` method. Items are
    #: :class:`ccdaf.core.orifice_labels.Passage`. Each ring's nodes
    #: belong to the epicardium and to a cavity at once.
    cuts: List[Any] = field(default_factory=list)

    def summary(self) -> str:
        total = sum(self.areas.values()) or 1.0
        parts = [f"{NAMES[k]} {100 * self.areas.get(k, 0.0) / total:.1f}%"
                 for k in (BASE, EPI, LV_ENDO, RV_ENDO)]
        text = "; ".join(parts)
        if not self.naming_confident:
            text += " (naming uncertain)"
        return text

    def details(self) -> str:
        if self.plane is not None:
            lines = [f"Plane: {self.plane.describe()}", ""]
        else:
            lines = [f"Base: rings at {len(self.openings)} valve opening(s)", ""]
            for i, opening in enumerate(self.openings, 1):
                lines.append(f"  opening {i}: {opening.describe()}")
            lines.append("")
        total = sum(self.areas.values()) or 1.0
        for key in (BASE, EPI, LV_ENDO, RV_ENDO):
            faces = int((self.face_labels == key).sum())
            nodes = len(self.node_sets.get(key, ()))
            lines.append(f"{NAMES[key]:>15}: {faces:6d} faces, {nodes:6d} nodes, "
                         f"{100 * self.areas.get(key, 0.0) / total:5.1f}% of area")
        if self.dropped_patches:
            lines.append("")
            lines.append(f"{self.dropped_patches} stray patch(es) dropped from "
                         f"the base selection.")
        if self.note:
            lines.append("")
            lines.append(self.note)
        return "\n".join(lines)


# ---------------------------------------------------------------------
# Boundary helpers
# ---------------------------------------------------------------------
def _as_boundary(dataset) -> pv.PolyData:
    """The boundary surface of a volume, or the surface itself.

    Taken through :func:`boundary_surface` so the parent tetrahedron's
    cell data rides onto each face: that is where the ``elemTag`` used to
    tell the two cavities apart comes from.
    """
    data = pv.wrap(dataset)
    if is_volume(data):
        return boundary_surface(data)
    return data.triangulate()


def _origin_ids(surface: pv.PolyData) -> Optional[np.ndarray]:
    """Map from boundary node ids back to the parent volume's, if any.

    ``boundary_surface`` asks for it, so a volume's boundary carries it and
    a surface given directly does not. Without this map the node sets would
    be indices into a different mesh: they would not raise, they would
    silently name the wrong nodes.
    """
    if "vtkOriginalPointIds" in surface.point_data:
        return np.asarray(surface.point_data["vtkOriginalPointIds"],
                          dtype=np.int64)
    return None


def _faces(surface: pv.PolyData) -> np.ndarray:
    faces = np.asarray(surface.faces).reshape(-1, 4)
    return faces[:, 1:].astype(np.int64)


def _areas(surface: pv.PolyData) -> np.ndarray:
    sized = surface.compute_cell_sizes(length=False, volume=False)
    return np.asarray(sized.cell_data["Area"], dtype=float)


def _normals(surface: pv.PolyData) -> np.ndarray:
    if "Normals" in surface.cell_data:
        return np.asarray(surface.cell_data["Normals"], dtype=float)
    with_normals = surface.compute_normals(
        cell_normals=True, point_normals=False,
        consistent_normals=True, auto_orient_normals=True, inplace=False)
    return np.asarray(with_normals.cell_data["Normals"], dtype=float)


def mean_edge_length(surface: pv.PolyData) -> float:
    """Mean edge length of a triangular surface, in mesh units."""
    tri = _faces(surface)
    if len(tri) == 0:
        return 0.0
    pts = np.asarray(surface.points, dtype=float)
    lengths = [np.linalg.norm(pts[tri[:, b]] - pts[tri[:, a]], axis=1)
               for a, b in ((0, 1), (1, 2), (2, 0))]
    return float(np.concatenate(lengths).mean())


def _face_components(tri: np.ndarray, subset: np.ndarray) -> Tuple[int, np.ndarray]:
    """Connected components of *subset*, joined across shared edges.

    Edge adjacency, not node adjacency: two patches touching at a single
    point are two patches, which is the same rule the volume cleaner uses
    and the reason a pinch point cannot leak a label.
    """
    if len(subset) == 0:
        return 0, np.zeros(0, dtype=np.int64)
    position = {int(f): i for i, f in enumerate(subset)}
    edges: Dict[Tuple[int, int], List[int]] = {}
    for f in subset:
        a, b, c = (int(x) for x in tri[f])
        for u, v in ((a, b), (b, c), (a, c)):
            edges.setdefault((min(u, v), max(u, v)), []).append(int(f))
    rows: List[int] = []
    cols: List[int] = []
    for shared in edges.values():
        if len(shared) == 2:
            rows.append(position[shared[0]])
            cols.append(position[shared[1]])
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)),
                       shape=(len(subset),) * 2)
    return connected_components(graph, directed=False)


# ---------------------------------------------------------------------
# Detecting the cut
# ---------------------------------------------------------------------
def detect_base_plane(dataset,
                      options: Optional[LabelOptions] = None
                      ) -> Optional[Plane]:
    """Find the plane of a truncated mesh's basal cut, or ``None``.

    Samples candidate planes from the faces themselves, weighted by area,
    and keeps the one capturing the most area within the distance and
    angle tests. On a mesh truncated by an exact plane this recovers it to
    0.00 degrees and zero offset, selecting precisely the cut.

    ``None`` means no candidate stood out, which is the honest answer for
    a mesh that was never truncated.
    """
    options = options or LabelOptions()
    options.validate()
    surface = _as_boundary(dataset)
    tri = _faces(surface)
    if len(tri) == 0:
        return None

    normals = _normals(surface)
    centres = np.asarray(surface.cell_centers().points, dtype=float)
    area = _areas(surface)
    tol = options.tol_edges * mean_edge_length(surface)
    if tol <= 0.0:
        return None

    rng = np.random.default_rng(options.seed)
    count = min(options.samples, len(tri))
    weights = area / area.sum() if area.sum() > 0 else None
    sample = rng.choice(len(tri), count, replace=False, p=weights)

    best_score = -1.0
    best: Optional[Plane] = None
    for i in sample:
        normal = normals[i]
        origin = centres[i]
        near = np.abs((centres - origin) @ normal) < tol
        parallel = np.abs(normals @ normal) > options.cos_min
        score = float(area[near & parallel].sum())
        if score > best_score:
            best_score = score
            best = Plane(origin=origin, normal=normal)
    return best


# ---------------------------------------------------------------------
# Labelling
# ---------------------------------------------------------------------
def _select_base(surface: pv.PolyData, plane: Plane,
                 options: LabelOptions) -> Tuple[np.ndarray, int]:
    """Faces of the base, and how many stray patches were dropped."""
    tri = _faces(surface)
    pts = np.asarray(surface.points, dtype=float)
    normals = _normals(surface)
    area = _areas(surface)
    tol = options.tol_edges * mean_edge_length(surface)

    node_distance = np.abs(plane.signed_distance(pts))
    on_plane = (node_distance < tol)[tri].all(axis=1)
    parallel = np.abs(normals @ plane.normal) > options.cos_min
    mask = on_plane & parallel

    dropped = 0
    chosen = np.where(mask)[0]
    if len(chosen):
        count, labels = _face_components(tri, chosen)
        if count > 1:
            per_patch = np.bincount(labels, weights=area[chosen])
            keep = per_patch >= options.min_patch_fraction * per_patch.sum()
            keep[int(per_patch.argmax())] = True
            dropped = int((~keep).sum())
            mask = np.zeros(len(tri), dtype=bool)
            mask[chosen[keep[labels]]] = True
    return mask, dropped


def _hull_depth(surface: pv.PolyData, centres: np.ndarray,
                seed: int) -> np.ndarray:
    """How far each face centre lies beneath the convex hull."""
    pts = np.asarray(surface.points, dtype=float)
    rng = np.random.default_rng(seed)
    if len(pts) > _HULL_SAMPLE:
        pts = pts[rng.choice(len(pts), _HULL_SAMPLE, replace=False)]
    equations = ConvexHull(pts).equations.astype(np.float64)
    depth = np.empty(len(centres))
    chunk = 200_000
    for start in range(0, len(centres), chunk):
        block = centres[start:start + chunk]
        depth[start:start + chunk] = -(
            block @ equations[:, :3].T + equations[:, 3]).max(axis=1)
    return depth


def _name_pieces(surface: pv.PolyData, tri: np.ndarray, pieces: List[np.ndarray],
                 options: LabelOptions) -> Tuple[Dict[int, np.ndarray], bool, str]:
    """Decide which piece is epicardium, LV endocardium, RV endocardium."""
    centres = np.asarray(surface.cell_centers().points, dtype=float)
    depth = _hull_depth(surface, centres, options.seed)
    medians = [float(np.median(depth[p])) for p in pieces]
    outermost = int(np.argmin(medians))

    rest = [i for i in range(3) if i != outermost]
    note = ""
    confident = True

    tags = (np.asarray(surface.cell_data["elemTag"])
            if "elemTag" in surface.cell_data else None)
    lv_index = rv_index = None
    if tags is not None:
        purity = []
        for i in rest:
            values, counts = np.unique(tags[pieces[i]], return_counts=True)
            purity.append(float(counts.max() / counts.sum()))
        # The LV cavity is enclosed by one tag; the RV cavity also sees the
        # septum, which carries the LV's tag, so it comes back mixed.
        if abs(purity[0] - purity[1]) > 0.05:
            lv_index = rest[int(np.argmax(purity))]
            rv_index = rest[int(np.argmin(purity))]
        else:
            note = ("The two cavities have equally mixed element tags, so "
                    "which is the LV could not be decided from them.")
    else:
        note = ("The mesh carries no elemTag, so which cavity is the LV "
                "could not be decided.")

    if lv_index is None:
        # Fall back to area, and say so: on the example this is the wrong
        # way round, which is exactly why it is not the primary rule.
        area = _areas(surface)
        sizes = [float(area[pieces[i]].sum()) for i in rest]
        lv_index = rest[int(np.argmax(sizes))]
        rv_index = rest[int(np.argmin(sizes))]
        confident = False
        note += (" Falling back to area, which is not reliable: on the "
                 "example ventricle the LV endocardium is the smaller of "
                 "the two.")

    return ({EPI: pieces[outermost], LV_ENDO: pieces[lv_index],
             RV_ENDO: pieces[rv_index]}, confident, note.strip())


def label_boundary(dataset,
                   plane: Optional[Plane] = None,
                   options: Optional[LabelOptions] = None) -> SurfaceLabels:
    """Label the boundary of *dataset* as base, epi, LV endo and RV endo.

    ``plane`` is the basal cut. Left out, it is detected. ``dataset`` is
    not modified.

    Raises ``ValueError`` when the base does not separate the boundary
    into exactly three pieces, which is what a plane in the wrong place,
    or a mesh that was never truncated, looks like.
    """
    options = options or LabelOptions()
    options.validate()
    surface = _as_boundary(dataset)
    tri = _faces(surface)
    if len(tri) == 0:
        raise ValueError("this mesh has no boundary faces to label")

    if plane is None:
        plane = detect_base_plane(surface, options)
        if plane is None:
            raise ValueError("no basal cut plane could be detected")

    base_mask, dropped = _select_base(surface, plane, options)
    if not base_mask.any():
        raise ValueError(
            "no faces lie on that plane: the mesh is not cut there. "
            f"Plane: {plane.describe()}")

    rest = np.where(~base_mask)[0]
    count, piece_labels = _face_components(tri, rest)
    area = _areas(surface)
    if count < 1:
        raise ValueError("removing the base left no surface at all")
    per_piece = np.bincount(piece_labels, weights=area[rest])
    significant = np.where(per_piece >= 0.01 * per_piece.sum())[0]
    if len(significant) != 3:
        shares = ", ".join(f"{100 * s / per_piece.sum():.1f}%"
                           for s in np.sort(per_piece)[::-1][:5])
        raise ValueError(
            f"removing the base left {len(significant)} surface(s), not 3 "
            f"(shares {shares}). Either the plane is in the wrong place, or "
            f"the mesh still has its valve orifices, which join the "
            f"epicardium to the endocardium whatever is cut.")

    pieces = [rest[piece_labels == k] for k in significant]
    named, confident, note = _name_pieces(surface, tri, pieces, options)

    face_labels = np.full(len(tri), UNLABELLED, dtype=np.int32)
    for key, faces in named.items():
        face_labels[faces] = key
    face_labels[_close_ring(tri, base_mask)] = BASE

    return _assemble(surface, tri, face_labels, area, plane=plane,
                     naming_confident=confident, note=note,
                     dropped_patches=dropped)


def _close_ring(tri: np.ndarray, base: np.ndarray) -> np.ndarray:
    """*base* plus every face whose three nodes all lie on it.

    Such a face sits inside the base, and the saved form cannot tell it
    apart anyway: labels are stored as per-node membership, and a face
    whose nodes are all base members reads back as base. Measured before
    this rule, with the valve-opening method: 48 of 77,720 faces (open
    example) and 246 of 193,210 (a reference mesh) changed on a save and
    reload, every one of them this kind. A flat cut rarely makes such a
    face: the truncated example has none that would change, so there the
    rule is a guarantee rather than a repair. It adds no base node, so one
    pass is enough.
    """
    on_base = np.zeros(int(tri.max()) + 1, dtype=bool)
    on_base[tri[base].ravel()] = True
    return base | on_base[tri].all(1)


def _assemble(surface: pv.PolyData, tri: np.ndarray, face_labels: np.ndarray,
              area: np.ndarray, **extra) -> SurfaceLabels:
    """Node sets and areas from per-face labels, packed as the result."""
    # Node sets are returned in the caller's own numbering, so a volume's
    # sets index the volume and not its derived boundary.
    origin = _origin_ids(surface)
    node_sets: Dict[int, np.ndarray] = {}
    areas: Dict[int, float] = {}
    for key in (BASE, EPI, LV_ENDO, RV_ENDO):
        owned = np.where(face_labels == key)[0]
        nodes = np.unique(tri[owned]) if len(owned) else np.zeros(
            0, dtype=np.int64)
        node_sets[key] = origin[nodes] if origin is not None else nodes
        areas[key] = float(area[owned].sum())
    return SurfaceLabels(face_labels=face_labels, node_sets=node_sets,
                         areas=areas, **extra)


def snap_to_cut(dataset,
                plane: Plane,
                options: Optional[LabelOptions] = None) -> Optional[Plane]:
    """The real cut nearest *plane*, or ``None`` if there is none nearby.

    This is what makes a draggable plane usable at all. The tool selects
    faces on a cut the mesh already has, so a plane dragged by hand lands
    *beside* the answer rather than on it: nothing lies on it, and the
    labelling fails through no fault of the user. Snapping lets the drag
    mean "that one" instead of "exactly here".

    The search is the detector restricted to the plane's neighbourhood:
    faces within ``snap_edges`` mean edges of it and within
    ``snap_cone_deg`` of parallel. It deliberately refuses to reach
    further, so pointing at nothing returns nothing.
    """
    options = options or LabelOptions()
    options.validate()
    if options.snap_edges <= 0.0:
        return None

    surface = _as_boundary(dataset)
    tri = _faces(surface)
    if len(tri) == 0:
        return None
    edge = mean_edge_length(surface)
    if edge <= 0.0:
        return None

    normals = _normals(surface)
    centres = np.asarray(surface.cell_centers().points, dtype=float)
    area = _areas(surface)

    near = ((np.abs((centres - plane.origin) @ plane.normal)
             < options.snap_edges * edge)
            & (np.abs(normals @ plane.normal)
               > np.cos(np.radians(options.snap_cone_deg))))
    candidates = np.where(near)[0]
    if len(candidates) == 0:
        return None

    rng = np.random.default_rng(options.seed)
    weights = area[candidates]
    picked = rng.choice(candidates, min(options.samples, len(candidates)),
                        replace=False, p=weights / weights.sum())
    tol = options.tol_edges * edge
    found: List[Tuple[float, float, Plane]] = []
    for i in picked:
        normal, origin = normals[i], centres[i]
        on = np.abs((centres - origin) @ normal) < tol
        parallel = np.abs(normals @ normal) > options.cos_min
        score = float(area[on & parallel].sum())
        distance = abs(float((origin - plane.origin) @ plane.normal))
        found.append((score, distance, Plane(origin=origin, normal=normal)))
    if not found:
        return None

    # Nearest, not largest. A drag means "use that cut", and a mesh flat at
    # both ends has two candidates in range: scoring by area alone would
    # answer with the bigger face even when the user is standing on the
    # other one. Slivers are still excluded, by keeping only faces holding
    # a real share of the best area found.
    best_area = max(score for score, _d, _p in found)
    serious = [(d, p) for score, d, p in found
               if score >= _SNAP_AREA_SHARE * best_area]
    return min(serious, key=lambda item: item[0])[1]


def label_boundary_or_snap(dataset,
                           plane: Plane,
                           options: Optional[LabelOptions] = None
                           ) -> Tuple[SurfaceLabels, bool]:
    """Label with *plane*; failing that, snap to the nearest cut and retry.

    Returns the labelling and whether it came from a snapped plane. The
    plane actually used is on the result, so a caller can show it: a snap
    that happened invisibly would be worse than no snap at all.

    Raises ``ValueError`` when the plane fails and no nearby cut rescues
    it, with the original reason kept, because that reason is the useful
    one (the mesh is not cut there, or the cut separates nothing).
    """
    options = options or LabelOptions()
    try:
        return label_boundary(dataset, plane, options), False
    except ValueError as first:
        snapped = snap_to_cut(dataset, plane, options)
        if snapped is None:
            raise ValueError(
                f"{first} No flat cut was found within "
                f"{options.snap_edges:g} mean edge lengths of it either."
            ) from first
        try:
            return label_boundary(dataset, snapped, options), True
        except ValueError as second:
            raise ValueError(
                f"{first} The nearest flat face, at {snapped.describe()}, "
                f"does not work either: {second}"
            ) from second


def node_label_mask(dataset, labels: SurfaceLabels) -> np.ndarray:
    """Per-node membership of every surface, as bit flags.

    Unlike :func:`node_labels` this loses nothing: a node on the rim
    between base and epicardium carries both bits instead of having to be
    called one or the other.
    """
    data = pv.wrap(dataset)
    out = np.zeros(data.n_points, dtype=np.int32)
    for key, bit in MASK_BITS.items():
        nodes = labels.node_sets.get(key)
        if nodes is not None and len(nodes):
            out[nodes] |= bit
    return out


def face_labels_from_mask(surface, mask) -> np.ndarray:
    """Per-face labels from per-node membership. Exact, where agreement is not.

    A face belongs to the surface all three of its nodes are members of,
    which is what the bitwise AND of their masks says. Every face of the
    truncated example is recovered this way, against 1,775 lost by
    :func:`face_labels_from_nodes`.

    Where the AND leaves more than one bit — one face in 45,138 on that
    mesh, a strip one face wide belonging to two surfaces at once — the
    lowest bit wins, which is the same base-before-epicardium precedence
    the node labels use. Dropping such a face instead would trade a rare
    wrong answer for a rare hole, which is not a better bargain.
    """
    poly = pv.wrap(surface)
    tri = _faces(poly)
    if len(tri) == 0:
        return np.zeros(0, dtype=np.int32)
    per_node = np.asarray(mask, dtype=np.int64)[tri]
    common = per_node[:, 0] & per_node[:, 1] & per_node[:, 2]
    lowest = common & (-common)          # isolate the lowest set bit
    out = np.zeros(len(tri), dtype=np.int32)
    for key, bit in MASK_BITS.items():
        out[lowest == bit] = key
    return out


def face_labels_from_nodes(surface, values) -> np.ndarray:
    """Per-face labels from per-node labels: what all three nodes agree on.

    Needed because only the node labels survive a save: the per-face form
    lives on the boundary, which is rebuilt from the volume every time it
    changes. Reading a saved labelling back therefore means deriving the
    faces again.

    A face whose nodes disagree is ``UNLABELLED`` rather than an average.
    The average of *base* and *epicardium* is a label that does not exist,
    and it would draw as a third colour along every seam. Disagreement
    happens exactly on the one-face-wide seam where two surfaces meet, so
    it reads as a thin unlabelled line, which is the truth.
    """
    poly = pv.wrap(surface)
    tri = _faces(poly)
    if len(tri) == 0:
        return np.zeros(0, dtype=np.int32)
    per_node = np.asarray(values)[tri]
    agree = ((per_node[:, 0] == per_node[:, 1])
             & (per_node[:, 1] == per_node[:, 2]))
    return np.where(agree, per_node[:, 0], UNLABELLED).astype(np.int32)


def node_labels(dataset, labels: SurfaceLabels) -> np.ndarray:
    """Per-node labels for the whole mesh, for the solve and for saving.

    The sets already index *dataset*, whether that is a volume or a bare
    surface, so this only scatters them.

    A node shared by two surfaces keeps the one written last in the order
    RV, LV, epicardium, base, so the base wins: it is the set every solve
    treats specially. Callers needing the exact per-face truth use
    ``face_labels``, which is why that stays the primary form.
    """
    data = pv.wrap(dataset)
    out = np.full(data.n_points, UNLABELLED, dtype=np.int32)
    for key in (RV_ENDO, LV_ENDO, EPI, BASE):
        nodes = labels.node_sets.get(key)
        if nodes is not None and len(nodes):
            out[nodes] = key
    return out


__all__ = ["Plane", "LabelOptions", "SurfaceLabels", "detect_base_plane",
           "label_boundary", "label_boundary_or_snap", "snap_to_cut",
           "node_labels", "node_label_mask", "face_labels_from_nodes",
           "face_labels_from_mask", "mean_edge_length", "LABEL_MASK_FIELD",
           "MASK_BITS",
           "UNLABELLED", "BASE", "EPI", "LV_ENDO", "RV_ENDO", "LABEL_FIELD",
           "NAMES"]
