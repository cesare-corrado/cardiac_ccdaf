"""
orifice_labels
==============
Name the surfaces of a ventricular volume that still has its valve
openings: base, epicardium, LV endocardium, RV endocardium.

The problem
-----------
On such a mesh the epicardium and both endocardia are one connected
surface, joined through each opening. The base is the band where they
meet: the *throat* of each opening, a short tube lining the inside of the
hole. It is not flat and not planar (a mitral annulus is saddle-shaped),
so the truncated method in :mod:`ccdaf.core.surface_labels`, which selects
the faces on one plane, cannot find it.

Two geometric shortcuts were measured against a reference mesh whose base
is known exactly, and both fail. A shortest-curve cut over the whole
surface hugs the cavities instead of the openings, because a cavity's
cross-section is shorter than the path round its valves. A flat plane per
valve cuts into the epicardium at a saddle-shaped annulus.

The method
----------
1. **Find the blood pools.** Voxelise the solid, close it with a ball
   wider than any opening (``closing_mm``) so the pools fill, then open
   the filled region with a smaller ball (``opening_mm``). The opening
   removes the thin layer the closing laid over the base, and what is left
   is exactly the two pools, each touching only its own endocardium.
2. **Find the openings.** An opening is where a pool meets the outside
   air. Air counts as outside only if it reaches the edge of the grid
   through gaps wider than ``film_voxels``, which excludes thin pockets
   left between a pool and its wall.
3. **Build the rings.** Faces within ``band_mm`` of an opening form a
   search band. Inside it, the shortest loop separating the epicardial
   side from the endocardial side is found as a minimum cut. The throat is
   by definition the narrowest section of the opening, which is what a
   shortest loop finds; the band keeps the cut from drifting into a
   cavity. The base is the faces touching that loop.
4. **Cut through holes, if the rings are not enough.** A hole through
   the wall narrower than a voxel is closed wall to the voxel model, but
   it still joins the epicardium to a cavity on the mesh, and then the
   rings leave two surfaces instead of three. Only then, each face is
   classed by what the voxel model says it faces, outside or a pool, and
   the cheapest cut between the two is taken outside the bands. Near
   each join the cut is then solved again with a price on every face put
   on the side it does not face, so that it runs through each hole
   rather than round a cluster of them. The labels are split along the
   cut: the mesh itself is not changed. A cut where the voxel model
   sees a pool open to the outside is refused, because that is an
   opening missing from the list, not a hole.

Measured on three meshes (a reference with gold labels, a second one with
labels from another pipeline, and the example ventricle):

* every valve opening found, each once, except that a mitral and an
  aortic valve lying side by side come back as one opening. That is
  harmless: the base there is one ring instead of two;
* on the gold mesh, 93% of the loop's nodes lie on the gold base, and the
  epicardium, LV and RV agree with gold on 99.2%, 99.3% and 99.9% of their
  area;
* the band must stay narrow: at 8 mm the loop slips into the LV cavity
  and the LV agreement falls to 96.9%;
* step 4 made no cut on any of them. On a fourth heart, whose wall has
  a cluster of holes joining the epicardium to the LV where it thins to
  under 1 mm, it made five cuts of 8 to 13 mm, one per hole, and no
  outward-facing face near them was labelled LV. Without the second
  solve it made one 42 mm loop round the cluster, and 55 mm² of outer
  wall inside it was labelled LV.

Units
-----
Every parameter is an anatomical length in millimetres, and meshes arrive
in millimetres, centimetres or micrometres. :func:`guess_unit` guesses
from the mesh's RMS radius about its barycentre, which is about 45 mm on
all three measured hearts (44.8, 44.7, 43.7). The candidate units differ
by a factor of ten or more, so the nearest one is never in doubt. The
parameters are deliberately *not* scaled with heart size: wall volume
varied twofold between those hearts at the same radius, and the voxels
must resolve the thinnest wall in absolute terms.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyvista as pv
import scipy.sparse as sp
import vtk
from scipy import ndimage
from scipy.sparse.csgraph import (breadth_first_order, connected_components,
                                  maximum_flow)
from scipy.spatial import cKDTree
from vtk.util.numpy_support import vtk_to_numpy

from ccdaf.core.surface_labels import (
    BASE, MASK_BITS, UNLABELLED, LabelOptions, Plane, SurfaceLabels, _as_boundary,
    _areas, _assemble, _close_ring, _faces, _hull_depth, _name_pieces,
    _normals, label_boundary,
)
from ccdaf.core.volume_mesh import is_volume, tetrahedra

#: Millimetres per mesh unit, for each unit the dialog offers.
UNITS: Dict[str, float] = {"mm": 1.0, "cm": 10.0, "µm": 1e-3}

#: RMS radius of a ventricular volume about its barycentre, in mm.
#: Measured 44.8, 44.7 and 43.7 on three hearts.
TYPICAL_RMS_RADIUS_MM: float = 45.0

#: Largest voxel grid the detection will build. A mesh in micrometres
#: read as millimetres would ask for about 10^15 voxels; this turns that
#: into an error message instead of an exhausted machine.
MAX_VOXELS: int = 200_000_000

#: Pieces smaller than this share of the area are strays, not surfaces.
#: The same threshold the truncated method uses.
_SIGNIFICANT_SHARE: float = 0.01


# ---------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------
def rms_radius(dataset) -> float:
    """RMS distance of the mesh from its barycentre, in mesh units.

    A volume is weighted by tetrahedron volume, so it measures the solid
    and not how finely it was meshed; a surface falls back to its points.
    It is the square root of the sum of the principal moments, and less
    sensitive than a bounding box to one long protrusion.
    """
    data = pv.wrap(dataset)
    if is_volume(data):
        sized = data.compute_cell_sizes(length=False, area=False, volume=True)
        weights = np.abs(np.asarray(sized.cell_data["Volume"], dtype=float))
        points = np.asarray(data.cell_centers().points, dtype=float)
    else:
        points = np.asarray(data.points, dtype=float)
        weights = np.ones(len(points))
    if len(points) == 0 or weights.sum() <= 0.0:
        return 0.0
    centre = (points * weights[:, None]).sum(0) / weights.sum()
    spread = ((points - centre) ** 2).sum(1)
    return float(np.sqrt((spread * weights).sum() / weights.sum()))


def guess_unit(dataset) -> str:
    """The unit (a key of :data:`UNITS`) that makes the mesh heart-sized."""
    radius = rms_radius(dataset)
    if radius <= 0.0:
        return "mm"
    return min(UNITS, key=lambda u: abs(np.log(
        radius * UNITS[u] / TYPICAL_RMS_RADIUS_MM)))


# ---------------------------------------------------------------------
# Options and results
# ---------------------------------------------------------------------
@dataclass
class OrificeOptions:
    """How openings are found and rings built. Lengths are in millimetres."""

    #: Millimetres per mesh unit; see :data:`UNITS` and :func:`guess_unit`.
    mm_per_unit: float = 1.0
    #: Voxel edge. It must resolve the thinnest wall.
    voxel_mm: float = 1.0
    #: Radius of the ball that fills the pools. It must exceed the radius
    #: of the widest opening, or that opening is not bridged.
    closing_mm: float = 30.0
    #: Radius of the ball that separates the pools from the layer over the
    #: base. Results were identical from 5 to 8 mm.
    opening_mm: float = 6.0
    #: Width of the band the ring is searched in. Narrow on purpose: wider
    #: bands let the ring slip into a cavity.
    band_mm: float = 3.0
    #: Air narrower than this many voxels does not count as outside.
    film_voxels: float = 1.5
    #: Filled regions smaller than this are not pools.
    min_pool_ml: float = 20.0
    #: Openings smaller than this are noise.
    min_opening_mm2: float = 30.0

    def validate(self) -> None:
        if self.mm_per_unit <= 0.0:
            raise ValueError("the unit scale must be positive")
        if self.voxel_mm <= 0.0:
            raise ValueError("the voxel size must be positive")
        if not self.voxel_mm < self.opening_mm < self.closing_mm:
            raise ValueError("the sizes must satisfy voxel < opening < closing")
        if self.band_mm <= 0.0:
            raise ValueError("the band width must be positive")
        if self.film_voxels < 0.0:
            raise ValueError("the film width must not be negative")
        if self.min_pool_ml < 0.0 or self.min_opening_mm2 < 0.0:
            raise ValueError("the size thresholds must not be negative")


@dataclass
class Opening:
    """One valve opening (or several side by side), in mesh units."""

    #: Which pool it opens, an index into :attr:`OpeningSearch.pool_ml`.
    pool: int
    #: Centre of the opening's voxels.
    centre: np.ndarray
    #: Normal of the plane that fits it best.
    normal: np.ndarray
    #: Area of the opening's voxel face, in mm^2.
    area_mm2: float
    #: The opening's voxel centres. The ring is searched within
    #: ``band_mm`` of these.
    points: np.ndarray = field(repr=False)

    @property
    def plane(self) -> Plane:
        return Plane(origin=self.centre, normal=self.normal)

    def describe(self) -> str:
        c = ", ".join(f"{v:.3g}" for v in self.centre)
        return f"pool {self.pool + 1}, {self.area_mm2:.0f} mm², centre ({c})"


@dataclass
class OpeningSearch:
    """What :func:`find_openings` found."""

    #: Volume of each pool, in mL.
    pool_ml: List[float]
    openings: List[Opening]


# ---------------------------------------------------------------------
# Finding the openings
# ---------------------------------------------------------------------
def _voxelise(points_mm: np.ndarray, tri: np.ndarray, voxel: float,
              pad: float) -> Tuple[np.ndarray, np.ndarray]:
    """Inside/outside of a closed triangle surface on a voxel grid.

    Returns the mask indexed ``[x, y, z]`` and the grid origin.
    """
    low = points_mm.min(0) - pad
    high = points_mm.max(0) + pad
    dims = np.ceil((high - low) / voxel).astype(int) + 1
    if int(np.prod(dims.astype(np.int64))) > MAX_VOXELS:
        raise ValueError(
            f"the voxel grid would be {dims[0]} x {dims[1]} x {dims[2]}, "
            "too large. Check the mesh units: a mesh in micrometres read "
            "as millimetres looks a thousand times too big.")
    cells = np.hstack([np.full((len(tri), 1), 3), tri]).ravel()
    poly = pv.PolyData(points_mm, cells)

    image = vtk.vtkImageData()
    image.SetSpacing(voxel, voxel, voxel)
    image.SetOrigin(*low)
    image.SetDimensions(*(int(d) for d in dims))
    image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    image.GetPointData().GetScalars().Fill(1)

    stencil = vtk.vtkPolyDataToImageStencil()
    stencil.SetInputData(poly)
    stencil.SetOutputOrigin(*low)
    stencil.SetOutputSpacing(voxel, voxel, voxel)
    stencil.SetOutputWholeExtent(image.GetExtent())
    stencil.Update()
    apply = vtk.vtkImageStencil()
    apply.SetInputData(image)
    apply.SetStencilConnection(stencil.GetOutputPort())
    apply.ReverseStencilOff()
    apply.SetBackgroundValue(0)
    apply.Update()
    flat = vtk_to_numpy(apply.GetOutput().GetPointData().GetScalars())
    return flat.reshape(dims[::-1]).transpose(2, 1, 0).astype(bool), low


#: Region codes of :class:`_VoxelModel`. Pool ``i`` is ``_POOL + i``.
_WALL, _OUTSIDE, _POOL = 0, 1, 2
#: A voxel that is neither wall, outside nor pool: a crevice, a pocket,
#: or the inside of a channel too narrow to count as air.
_OTHER = -1


@dataclass
class _VoxelModel:
    """The solid, its blood pools and the outside air, on one voxel grid.

    Everything here is in millimetres, whatever the mesh units.
    """

    #: One region code per voxel, indexed ``[x, y, z]``.
    region: np.ndarray
    #: Position of voxel ``[0, 0, 0]``.
    low: np.ndarray
    #: Voxel edge.
    voxel: float
    #: Volume of each pool, in mL, largest first.
    pool_ml: List[float]

    def rims(self) -> List[Tuple[int, np.ndarray]]:
        """``(pool, voxel indices)`` of every place a pool meets outside.

        Each place is one 26-connected patch. No size filter: that is
        the caller's decision.
        """
        touching = ndimage.binary_dilation(self.region == _OUTSIDE)
        out: List[Tuple[int, np.ndarray]] = []
        for index in range(len(self.pool_ml)):
            rim = (self.region == _POOL + index) & touching
            parts, count = ndimage.label(rim, structure=np.ones((3, 3, 3)))
            for j in range(1, count + 1):
                out.append((index, np.argwhere(parts == j)))
        return out

    def position(self, indices: np.ndarray) -> np.ndarray:
        """Voxel indices as positions, in mm."""
        return indices * self.voxel + self.low

    def at(self, points_mm: np.ndarray) -> np.ndarray:
        """The region code of the voxel nearest each point."""
        index = np.round((points_mm - self.low) / self.voxel).astype(int)
        index = np.clip(index, 0, np.array(self.region.shape) - 1)
        return self.region[index[:, 0], index[:, 1], index[:, 2]]


def _voxel_model(surface: pv.PolyData, tri: np.ndarray,
                 options: OrificeOptions) -> _VoxelModel:
    """Voxelise the solid and find its pools and the outside air."""
    h = options.voxel_mm
    points = np.asarray(surface.points, dtype=float) * options.mm_per_unit

    solid, low = _voxelise(points, tri, h, pad=options.closing_mm + 2 * h)
    edt = ndimage.distance_transform_edt
    # Closing: dilate by the radius, then erode by it, as exact distances.
    dilated = edt(~solid, sampling=h) <= options.closing_mm
    closed = (edt(dilated, sampling=h) > options.closing_mm) | solid
    filled = closed & ~solid
    del dilated, closed
    # Opening of the filled region: erode, then dilate back.
    eroded = edt(filled, sampling=h) > options.opening_mm
    core = (edt(~eroded, sampling=h) <= options.opening_mm) & filled
    del eroded, filled

    labels, count = ndimage.label(core)
    ml = np.bincount(labels.ravel())[1:] * h ** 3 / 1e3
    pool_ids = [k + 1 for k in np.argsort(ml)[::-1]
                if ml[k] >= options.min_pool_ml]
    pools = np.isin(labels, pool_ids)

    air = ~solid & ~pools & (edt(~solid, sampling=h) > options.film_voxels * h)
    air_labels, _ = ndimage.label(air)
    outside = air_labels == air_labels[0, 0, 0]
    del air, air_labels

    region = np.full(solid.shape, _OTHER, dtype=np.int8)
    region[solid] = _WALL
    region[outside] = _OUTSIDE
    for index, k in enumerate(pool_ids):
        region[labels == k] = _POOL + index
    return _VoxelModel(region=region, low=low, voxel=h,
                       pool_ml=[float(ml[k - 1]) for k in pool_ids])


def _openings_in(model: _VoxelModel, options: OrificeOptions) -> List[Opening]:
    """The rims large enough to be valve openings, largest first per pool."""
    scale = options.mm_per_unit
    h = model.voxel
    openings: List[Opening] = []
    by_pool: Dict[int, List[np.ndarray]] = {}
    for pool, indices in model.rims():
        by_pool.setdefault(pool, []).append(indices)
    for pool in sorted(by_pool):
        parts = sorted(by_pool[pool], key=len, reverse=True)
        for indices in parts:
            area = len(indices) * h ** 2
            if area < options.min_opening_mm2:
                break
            vox = model.position(indices)
            centre = vox.mean(0)
            if len(vox) >= 3:
                _w, v = np.linalg.eigh(np.cov((vox - centre).T))
                normal = v[:, 0]
            else:
                normal = np.array([0.0, 0.0, 1.0])
            openings.append(Opening(pool=pool, centre=centre / scale,
                                    normal=normal, area_mm2=float(area),
                                    points=vox / scale))
    return openings


def find_openings(dataset,
                  options: Optional[OrificeOptions] = None) -> OpeningSearch:
    """Find the blood pools and where each opens to the outside.

    ``dataset`` is a volume or its closed boundary surface; it is not
    modified. Openings are returned largest first within each pool.
    """
    options = options or OrificeOptions()
    options.validate()
    surface = _as_boundary(dataset)
    tri = _faces(surface)
    if len(tri) == 0:
        raise ValueError("this mesh has no boundary faces")
    model = _voxel_model(surface, tri, options)
    return OpeningSearch(pool_ml=list(model.pool_ml),
                         openings=_openings_in(model, options))


# ---------------------------------------------------------------------
# Building the rings
# ---------------------------------------------------------------------
def _manifold_adjacency(tri: np.ndarray, n_points: int
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Face pairs sharing an edge of exactly two faces, and that edge.

    Edges shared by more than two faces are left out on purpose. On the
    example ventricle 18 such edges join the epicardium straight to an
    endocardium, and counting them as connections keeps the surface in one
    piece whatever is cut. It is the same rule the truncated method's
    component search follows.
    """
    edges = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]],
                                    tri[:, [0, 2]]]), axis=1)
    owner = np.tile(np.arange(len(tri)), 3)
    key = edges[:, 0].astype(np.int64) * n_points + edges[:, 1]
    order = np.argsort(key, kind="stable")
    key, owner, edges = key[order], owner[order], edges[order]
    _unique, start, count = np.unique(key, return_index=True,
                                      return_counts=True)
    first = start[count == 2]
    return owner[first], owner[first + 1], edges[first]


def _components(mask: np.ndarray, fa: np.ndarray, fb: np.ndarray
                ) -> Tuple[int, np.ndarray]:
    """Components of the faces in *mask*; -1 for faces outside it."""
    index = np.where(mask)[0]
    position = np.full(len(mask), -1)
    position[index] = np.arange(len(index))
    both = mask[fa] & mask[fb]
    graph = sp.coo_matrix((np.ones(int(both.sum())),
                           (position[fa[both]], position[fb[both]])),
                          shape=(len(index),) * 2)
    count, labels = connected_components(graph, directed=False)
    out = np.full(len(mask), -1)
    out[index] = labels
    return count, out


def _min_cut(n_faces: int, fa: np.ndarray, fb: np.ndarray,
             capacity: np.ndarray, free: np.ndarray,
             source: np.ndarray, sink: np.ndarray) -> np.ndarray:
    """Faces on the source side of the cheapest cut through *free* faces."""
    keep = free[fa] | free[fb]
    return _st_cut(n_faces, fa[keep], fb[keep], capacity[keep],
                   np.where(source, _HELD, 0), np.where(sink, _HELD, 0))


#: Terminal capacity that holds a face on its side whatever the cut costs.
_HELD = 2 ** 30


def _st_cut(n_faces: int, fa: np.ndarray, fb: np.ndarray,
            capacity: np.ndarray, to_source: np.ndarray,
            to_sink: np.ndarray) -> np.ndarray:
    """Faces on the source side of the cheapest cut.

    Cutting the edge between faces ``fa[i]`` and ``fb[i]`` costs
    ``capacity[i]``; putting face ``k`` on the sink side costs
    ``to_source[k]``, and on the source side ``to_sink[k]``. All integer.
    """
    s, t = n_faces, n_faces + 1
    src = np.where(to_source > 0)[0]
    snk = np.where(to_sink > 0)[0]
    rows = np.concatenate([fa, fb, np.full(len(src), s), snk])
    cols = np.concatenate([fb, fa, src, np.full(len(snk), t)])
    caps = np.concatenate([capacity, capacity, to_source[src], to_sink[snk]])
    graph = sp.csr_matrix((caps.astype(np.int32), (rows, cols)),
                          shape=(n_faces + 2,) * 2)
    flow = maximum_flow(graph, s, t)
    residual = (graph - flow.flow).tocsr()
    residual.data[residual.data <= 0] = 0
    residual.eliminate_zeros()
    reach = breadth_first_order(residual, s, directed=True,
                                return_predecessors=False)
    side = np.zeros(n_faces + 2, dtype=bool)
    side[reach] = True
    return side[:n_faces]


def label_open_boundary(dataset,
                        openings: Optional[List[Opening]] = None,
                        options: Optional[OrificeOptions] = None,
                        label_options: Optional[LabelOptions] = None
                        ) -> SurfaceLabels:
    """Label a mesh with valve openings as base, epi, LV endo and RV endo.

    ``openings`` defaults to :func:`find_openings`; pass them to label with
    an edited set. ``label_options`` feeds the naming shared with the
    truncated method. ``dataset`` is not modified.

    Raises ``ValueError`` when the rings do not separate the boundary into
    exactly three surfaces, which is what a missed opening looks like.
    """
    options = options or OrificeOptions()
    options.validate()
    label_options = label_options or LabelOptions()
    surface = _as_boundary(dataset)
    tri = _faces(surface)
    if len(tri) == 0:
        raise ValueError("this mesh has no boundary faces to label")
    model = None
    if openings is None:
        model = _voxel_model(surface, tri, options)
        openings = _openings_in(model, options)
    if not openings:
        raise ValueError("no valve openings were found")

    points = np.asarray(surface.points, dtype=float)
    centres = points[tri].mean(1)
    area = _areas(surface)
    fa, fb, shared = _manifold_adjacency(tri, len(points))

    # The search band: faces near any opening.
    band_width = options.band_mm / options.mm_per_unit
    distance = np.full(len(tri), np.inf)
    for opening in openings:
        near, _ = cKDTree(opening.points).query(
            centres, distance_upper_bound=band_width)
        distance = np.minimum(distance, near)
    band = distance < band_width

    significant, pieces = _significant(~band, fa, fb, area)
    cuts: List[Passage] = []
    if len(significant) < 3:
        # Joined where there is no opening: look for holes in the wall
        # that the voxel model cannot see, and cut the labels there.
        if model is None:
            model = _voxel_model(surface, tri, options)
        try:
            keep, cuts = _cut_through_wall(dataset, surface, tri, area, fa,
                                           fb, shared, band, openings, model,
                                           options)
        except ValueError as exc:
            raise ValueError(
                f"the openings separate the surface into {len(significant)} "
                f"piece(s), not 3, and {exc}") from exc
        if cuts:
            fa, fb, shared = fa[keep], fb[keep], shared[keep]
            significant, pieces = _significant(~band, fa, fb, area)
    if len(significant) != 3:
        raise ValueError(
            f"the openings separate the surface into {len(significant)} "
            "piece(s), not 3: an opening is probably missing or misplaced.")

    # The outermost piece is the epicardium; the ring is the cheapest loop
    # between it and the two cavities, found inside the band only.
    depth = _hull_depth(surface, centres, label_options.seed)
    epi_piece = significant[int(np.argmin(
        [np.median(depth[pieces == k]) for k in significant]))]
    source = pieces == epi_piece
    sink = np.isin(pieces, significant) & ~source
    free = ~(source | sink)
    lengths = np.linalg.norm(points[shared[:, 1]] - points[shared[:, 0]],
                             axis=1)
    capacity = np.maximum(1, np.round(lengths / lengths.mean() * 1000))
    epi_side = _min_cut(len(tri), fa, fb, capacity, free, source, sink)

    on_cut = np.zeros(len(points), dtype=bool)
    on_cut[shared[epi_side[fa] != epi_side[fb]].ravel()] = True
    base = on_cut[tri].any(1)

    significant, pieces = _significant(~base, fa, fb, area)
    if len(significant) != 3:
        raise ValueError(
            f"the rings leave {len(significant)} surface(s), not 3.")
    # Stray pieces (a few faces cut off between the ring and a wall
    # defect) join the base: they lie on the rim, not on any surface.
    stray = (pieces >= 0) & ~np.isin(pieces, significant)
    base |= stray
    base = _close_ring(tri, base)

    named, confident, note = _name_pieces(
        surface, tri, [np.where(pieces == k)[0] for k in significant],
        label_options)
    face_labels = np.full(len(tri), UNLABELLED, dtype=np.int32)
    for key, faces in named.items():
        face_labels[faces] = key
    face_labels[base] = BASE
    if cuts:
        face_labels = _as_read_back(tri, face_labels)
    if stray.any():
        note = (note + f" {int(stray.sum())} stray face(s) next to a ring "
                "were added to the base.").strip()
    if cuts:
        note = (note + "\n" + describe_cuts(cuts, options.mm_per_unit)
                ).strip()
    return _assemble(surface, tri, face_labels, area, plane=None,
                     naming_confident=confident, note=note,
                     dropped_patches=0, method="openings",
                     openings=list(openings), cuts=cuts)


def _significant(mask: np.ndarray, fa: np.ndarray, fb: np.ndarray,
                 area: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The pieces of *mask* large enough to be surfaces, and every piece."""
    count, pieces = _components(mask, fa, fb)
    sizes = np.bincount(pieces[pieces >= 0], weights=area[pieces >= 0],
                        minlength=count)
    return np.where(sizes >= _SIGNIFICANT_SHARE * sizes.sum())[0], pieces


def label_ventricles(dataset,
                     options: Optional[OrificeOptions] = None,
                     label_options: Optional[LabelOptions] = None
                     ) -> SurfaceLabels:
    """Label a ventricular mesh, truncated or not, choosing the method.

    The flat-cut method goes first: it is fast and exact on a truncated
    mesh, and on a mesh with openings its three-piece check fails, which is
    the signal to look for openings instead. The method used is on the
    result's ``method``.
    """
    try:
        return label_boundary(dataset, None, label_options)
    except ValueError as flat:
        try:
            return label_open_boundary(dataset, None, options, label_options)
        except ValueError as open_:
            raise ValueError(
                f"As a truncated mesh: {flat} With valve openings: {open_}"
            ) from open_



# ---------------------------------------------------------------------
# Where two surfaces that should be separate have been joined
# ---------------------------------------------------------------------
#: The share of the boundary that counts as its outer shell, by depth
#: beneath the convex hull, and by symmetry the share that counts as
#: deepest. A piece holding none of the first does not reach the outside
#: of the wall; one holding none of the second holds no cavity.
_OUTER_SHARE: float = 0.05


@dataclass
class Passage:
    """One route from the cavity side of the wall to the outside.

    A valve opening the band did not fully separate is one of these. So
    is a hole through the wall. They differ in size and in nothing else,
    which is why this reports them together and sorted.
    """

    #: The ring of boundary edges the route is narrowest at.
    ring: np.ndarray = field(repr=False)
    #: Length of that ring, in mesh units. The route's size.
    circumference: float
    #: Where it is.
    centre: np.ndarray

    def describe(self) -> str:
        c = ", ".join(f"{v:.3g}" for v in self.centre)
        return f"{self.circumference:.1f} around, centre ({c})"


def find_passages(dataset,
                  options: Optional[OrificeOptions] = None,
                  label_options: Optional[LabelOptions] = None,
                  openings: Optional[List[Opening]] = None) -> List[Passage]:
    """Where the epicardium and a cavity are joined, largest ring first.

    This answers one question: when the labelling reports that the
    openings leave two surface pieces where it needs three, *what* joins
    them and *where*? Every route from the outside of the wall to the
    inside of a piece must cross the cheapest cut between the two, so
    the cut lands on the narrowest ring of each join. On the example
    ventricle it returns the three valve rims at 225, 179 and 91 units
    around, and the perforation at 11.7 — told apart by size alone.

    **It reports joins between surfaces, not every hole in the mesh.** A
    piece holding one surface is skipped, because nothing passes through
    it, so a hole that does not happen to join two pieces is not found
    here. Counting *all* the handles is a different question, and
    :func:`volume_repair.boundary_genus` answers it without saying where
    they are.

    Two approaches that do not work, both measured, in case they look
    tempting. Run over the whole surface rather than one piece at a
    time, the cut takes the cheapest single loop it can find: on the
    example, one 278.9 unit contour around the entire base, which
    separates inside from outside perfectly and says nothing about how
    many ways through there are. And the voxel model cannot be asked
    instead: it is a 1 mm approximation, and lowering its opening-size
    filter from 30 mm² to 2 still finds the same three valves and not
    the 3.7 mm perforation, because a channel that narrow never carries
    outside air through the wall at that resolution.
    """
    options = options or OrificeOptions()
    label_options = label_options or LabelOptions()
    surface = _as_boundary(dataset)
    tri = _faces(surface)
    if len(tri) == 0:
        return []
    points = np.asarray(surface.points, dtype=float)
    fa, fb, shared = _manifold_adjacency(tri, len(points))
    if len(fa) == 0:
        return []
    if openings is None:
        openings = find_openings(surface, options).openings
    if not openings:
        return []

    centres = points[tri].mean(1)
    area = _areas(surface)
    width = options.band_mm / options.mm_per_unit
    distance = np.full(len(tri), np.inf)
    for opening in openings:
        near, _ = cKDTree(opening.points).query(
            centres, distance_upper_bound=width)
        distance = np.minimum(distance, near)
    count, pieces = _components(~(distance < width), fa, fb)
    sizes = np.bincount(pieces[pieces >= 0], weights=area[pieces >= 0],
                        minlength=count)
    significant = np.where(sizes >= _SIGNIFICANT_SHARE * sizes.sum())[0]

    depth = _hull_depth(surface, centres, label_options.seed)
    lengths = np.linalg.norm(points[shared[:, 1]] - points[shared[:, 0]],
                             axis=1)
    capacity = np.maximum(1, np.round(lengths / lengths.mean() * 1000))
    # Outer and inner are measured against the whole boundary, never
    # against each piece. Against the piece they mean nothing: a cavity
    # surface also runs from its shallowest face to its deepest, so
    # splitting it at its own median reports a ring that is not there.
    # Inclusive bounds. On a real mesh the depths are continuous and it
    # makes no difference; on a mesh whose faces all sit at one of two
    # depths — any box — a strict bound selects nothing at all, and the
    # question then has no source and no sink to ask about.
    outermost = depth <= np.percentile(depth, _OUTER_SHARE * 100.0)
    innermost = depth >= np.percentile(depth, 100.0 - _OUTER_SHARE * 100.0)

    passages: List[Passage] = []
    for piece in significant:
        inside = pieces == piece
        source, sink = inside & outermost, inside & innermost
        if not (source.any() and sink.any()):
            continue                     # one surface: nothing crosses it
        side = _min_cut(len(tri), fa, fb, capacity,
                        inside & ~(source | sink), source, sink)
        passages.extend(_rings(shared[side[fa] != side[fb]], points))
    passages.sort(key=lambda item: -item.circumference)
    return passages


def _rings(cut: np.ndarray, points: np.ndarray) -> List[Passage]:
    """The edges of a cut, as one :class:`Passage` per ring.

    Rings are grouped by the edges that share an end: two routes never
    share a node, where clustering by distance would only usually
    separate them.
    """
    if len(cut) == 0:
        return []
    nodes = np.unique(cut)
    local = np.searchsorted(nodes, cut)
    groups = connected_components(
        sp.coo_matrix((np.ones(len(local), dtype=np.int8),
                       (local[:, 0], local[:, 1])),
                      shape=(len(nodes),) * 2), directed=False)[1]
    out: List[Passage] = []
    for k in np.unique(groups[local[:, 0]]):
        ring = cut[groups[local[:, 0]] == k]
        out.append(Passage(
            ring=ring,
            circumference=float(np.linalg.norm(
                points[ring[:, 1]] - points[ring[:, 0]], axis=1).sum()),
            centre=points[np.unique(ring)].mean(axis=0)))
    return out


def describe_passages(passages: List[Passage], keep: int = 0) -> str:
    """The passages as a report, naming which are too small to be valves.

    *keep* is how many the anatomy is expected to have; the rest are
    named as defects. Zero lists them without judgement, because how
    many valve openings a mesh should have is the caller's question.
    """
    if not passages:
        return "Nothing joins the epicardium to a cavity."
    lines = [f"{len(passages)} join(s) between the epicardium and a cavity, "
             f"largest first:"]
    for index, passage in enumerate(passages):
        tag = "   <- too small to be a valve opening" if (
            keep and index >= keep) else ""
        lines.append(f"  {passage.describe()}{tag}")
    return "\n".join(lines)


# ---------------------------------------------------------------------
# Labelling through holes in the wall
# ---------------------------------------------------------------------
#: How far a face looks along its outward normal for the region it
#: faces, in mm. Past the thickest trabecular crevice the voxel model
#: keeps, short of the far wall of a cavity.
_LOOK_MM: float = 4.0

#: Rings of faces taken off the edge of each seed, so that every seam
#: between two classes, and every hole, lies in faces the cut may choose.
#: Three rings is about 4.5 mm on a 1.5 mm mesh: wider than the holes
#: measured, narrower than the thinnest surface.
_SEED_RINGS: int = 3

#: A cut closer than this to a place where a pool meets the outside, at
#: no opening in use, is refused, in mm. Such a place is an opening the
#: voxel model can see, so a join there is a missing opening rather than
#: a hole. Only places of ``min_opening_mm2`` or more count, the same
#: size :func:`find_openings` reports: the uncleaned perforated heart
#: shows two of 3 and 4 mm² inside its cluster of holes, which are
#: holes. Measured: on the perforated heart the cut lay 26 mm or more
#: from every opening; on a block with one opening left out, the cut
#: lay at the one left out.
_UNEXPLAINED_MM: float = 10.0


#: What a face costs, per mm² of its area, on the side it does not face,
#: against 1 per mm of cut. See :func:`_refine`.
_DATA_WEIGHT: float = 1.0

#: How far from a join the cut may move, in mm.
_REFINE_MM: float = 10.0

#: Integer cut capacities per mm (and per mm², for the data term).
_PER_MM: float = 1000.0


def _outward_normals(dataset, surface: pv.PolyData,
                     tri: np.ndarray) -> np.ndarray:
    """Unit normal of each boundary face, pointing away from the material.

    From a volume this is exact: the tetrahedron behind each face says
    which side is solid. A bare surface falls back to normals oriented
    as one consistent, outward-facing set, which is the best a surface
    alone can say.
    """
    points = np.asarray(surface.points, dtype=float)
    a, b, c = points[tri[:, 0]], points[tri[:, 1]], points[tri[:, 2]]
    normal = np.cross(b - a, c - a)
    data = pv.wrap(dataset)
    if (is_volume(data) and "vtkOriginalCellIds" in surface.cell_data
            and "vtkOriginalPointIds" in surface.point_data):
        origin = np.asarray(surface.point_data["vtkOriginalPointIds"],
                            dtype=np.int64)
        owner = np.asarray(surface.cell_data["vtkOriginalCellIds"],
                           dtype=np.int64)
        tets = tetrahedra(data)
        apex = tets[owner].sum(1) - origin[tri].sum(1)
        inward = np.einsum("ij,ij->i", normal,
                           np.asarray(data.points, dtype=float)[apex]
                           - (a + b + c) / 3.0)
        normal[inward > 0] *= -1.0
    else:
        oriented = _normals(surface)
        normal[np.einsum("ij,ij->i", normal, oriented) < 0] *= -1.0
    return normal / np.maximum(np.linalg.norm(normal, axis=1),
                               1e-300)[:, None]


def _face_classes(centres_mm: np.ndarray, normals: np.ndarray,
                  model: _VoxelModel) -> np.ndarray:
    """What each face looks at: outside, a pool, or nothing (``_WALL``).

    Each face steps out along its normal and takes the first outside or
    pool voxel it meets. Meeting wall again first means it faces another
    wall across a gap, as the lining of a hole does, and it is left
    undecided.
    """
    out = np.full(len(centres_mm), _WALL, dtype=np.int8)
    open_ = np.ones(len(centres_mm), dtype=bool)
    step = model.voxel / 4.0
    for t in np.arange(step, _LOOK_MM + step / 2, step):
        region = model.at(centres_mm + normals * t)
        found = open_ & (region >= _OUTSIDE)
        out[found] = region[found]
        open_ &= ~found
        # The face's own voxel is usually wall: stop only past it.
        if t > 1.5 * model.voxel:
            open_ &= region != _WALL
        if not open_.any():
            break
    return out


def _seeds(classes: np.ndarray, band: np.ndarray, area: np.ndarray,
           fa: np.ndarray, fb: np.ndarray) -> np.ndarray:
    """The faces sure enough of their class to anchor the cut.

    Large patches of one class only, so a few faces misread where the
    wall is thinner than a voxel cannot pull the cut round them, and
    trimmed back from every seam.
    """
    seed = np.zeros(len(classes), dtype=np.int8)
    total = area.sum()
    for code in np.unique(classes[classes >= _OUTSIDE]):
        count, pieces = _components((classes == code) & ~band, fa, fb)
        sizes = np.bincount(pieces[pieces >= 0], weights=area[pieces >= 0],
                            minlength=count)
        seed[np.isin(pieces, np.where(sizes >= _SIGNIFICANT_SHARE * total)[0])
             ] = code
    for _ in range(_SEED_RINGS):
        seam = seed[fa] != seed[fb]
        seed[fa[seam]] = seed[fb[seam]] = _WALL
    return seed


def _cut_through_wall(dataset, surface: pv.PolyData, tri: np.ndarray,
                      area: np.ndarray, fa: np.ndarray, fb: np.ndarray,
                      shared: np.ndarray, band: np.ndarray,
                      openings: List[Opening], model: _VoxelModel,
                      options: OrificeOptions
                      ) -> Tuple[np.ndarray, List[Passage]]:
    """Where to cut the labels so holes through the wall stop joining them.

    Returns which adjacencies to keep and the cuts, one per ring. A hole
    too small for the voxel model, which sees closed wall there, still
    joins the epicardium to a cavity on the mesh, and no ring at the
    openings can separate them. So each face is classed by what the
    voxel model says it faces, and the cheapest cut between the faces
    sure to face outside and those sure to face a pool is taken. That
    finds the joins; :func:`_refine` then places the cut in each of
    them, through the hole.

    Raises ``ValueError`` when a cut lies at a place the voxel model
    sees open, which is an opening missing from *openings*.
    """
    points = np.asarray(surface.points, dtype=float)
    scale = options.mm_per_unit
    centres = points[tri].mean(1)
    classes = _face_classes(centres * scale,
                            _outward_normals(dataset, surface, tri), model)
    seed = _seeds(classes, band, area, fa, fb)
    source = seed == _OUTSIDE
    sink = seed >= _POOL
    keep = np.ones(len(fa), dtype=bool)
    if not (source.any() and sink.any()):
        return keep, []

    # The band belongs to the rings at the openings: the cut may not use it.
    off_band = ~band[fa] & ~band[fb]
    lengths = np.linalg.norm(points[shared[:, 1]] - points[shared[:, 0]],
                             axis=1)
    capacity = np.maximum(1, np.round(lengths / lengths.mean() * 1000))
    side = _min_cut(len(tri), fa[off_band], fb[off_band], capacity[off_band],
                    ~band & ~source & ~sink, source, sink)
    joins = shared[off_band & (side[fa] != side[fb])]
    if len(joins):
        side = _refine(side, classes, band, area * scale ** 2,
                       fa[off_band], fb[off_band], lengths[off_band] * scale,
                       centres * scale, points[np.unique(joins)] * scale)
    keep = ~(off_band & (side[fa] != side[fb]))
    cuts = _rings(shared[~keep], points)
    cuts.sort(key=lambda item: -item.circumference)

    unexplained = _unexplained_rims(model, openings, options)
    if len(unexplained) and cuts:
        tree = cKDTree(unexplained)
        for cut in cuts:
            near, index = tree.query(points[np.unique(cut.ring)] * scale)
            if near.min() < _UNEXPLAINED_MM:
                where = ", ".join(f"{v:.3g}" for v in
                                  unexplained[index[np.argmin(near)]] / scale)
                raise ValueError(
                    f"a blood pool meets the outside near ({where}), where "
                    "no opening was given: an opening is probably missing.")
    return keep, cuts


def _refine(side: np.ndarray, classes: np.ndarray, band: np.ndarray,
            area_mm2: np.ndarray, fa: np.ndarray, fb: np.ndarray,
            length_mm: np.ndarray, centres_mm: np.ndarray,
            joins_mm: np.ndarray) -> np.ndarray:
    """Move the cut near each join so it also respects what faces face.

    The first cut only asks for the shortest loop, and round a cluster of
    holes the shortest loop can run over the outer surface, taking the
    patch inside it to the cavity side. Here each face near a join also
    costs :data:`_DATA_WEIGHT` per mm² to put on the side it does not
    face, so the cut follows the hole linings instead. Faces farther
    than :data:`_REFINE_MM` from a join keep their side: away from the
    joins nothing needs cutting, and a data term there would cut round
    every patch the voxel model misreads.
    """
    near, _ = cKDTree(joins_mm).query(centres_mm,
                                      distance_upper_bound=_REFINE_MM)
    near = (near < _REFINE_MM) & ~band
    held = ~near & ~band
    price = np.round(_DATA_WEIGHT * area_mm2 * _PER_MM).astype(np.int64)
    to_source = np.where(held & side, _HELD, 0)
    to_source[near & (classes == _OUTSIDE)] = price[
        near & (classes == _OUTSIDE)]
    to_sink = np.where(held & ~side, _HELD, 0)
    to_sink[near & (classes >= _POOL)] = price[near & (classes >= _POOL)]
    capacity = np.maximum(1, np.round(length_mm * _PER_MM)).astype(np.int64)
    return _st_cut(len(side), fa, fb, capacity, to_source, to_sink)


def _unexplained_rims(model: _VoxelModel, openings: List[Opening],
                      options: OrificeOptions) -> np.ndarray:
    """Voxels, in mm, where a pool opens as widely as an opening does,
    at no opening in use."""
    used = [o.points * options.mm_per_unit for o in openings]
    tree = cKDTree(np.vstack(used)) if used else None
    out = [np.zeros((0, 3))]
    for _pool, indices in model.rims():
        if len(indices) * model.voxel ** 2 < options.min_opening_mm2:
            continue
        vox = model.position(indices)
        if tree is not None and tree.query(vox)[0].min() < model.voxel / 2:
            continue
        out.append(vox)
    return np.vstack(out)


def _as_read_back(tri: np.ndarray, face_labels: np.ndarray) -> np.ndarray:
    """Each face as the saved form will read it back, once that is stable.

    Labels are saved per node, and a face reads back as the lowest
    surface all three of its nodes belong to. Every node on a cut belongs
    to the epicardium and to a cavity, so a cavity face whose other
    nodes also touch the epicardium reads back as epicardium: the lining
    of a small hole, or a face beside a cut whose third node is a pinch,
    where epicardium meets it at that node alone. Such a face is given
    that label now, which is what :func:`surface_labels._close_ring`
    does at the base. A face only ever moves to a lower label, so this
    stops. Measured on the perforated heart: 5 of 77,576 faces cleaned,
    34 of 77,782 uncleaned, where pinches are many.
    """
    out = face_labels.copy()
    n_nodes = int(tri.max()) + 1
    while True:
        bits = np.zeros(n_nodes, dtype=np.int64)
        for key, bit in MASK_BITS.items():
            np.bitwise_or.at(bits, tri[out == key].ravel(), bit)
        common = bits[tri[:, 0]] & bits[tri[:, 1]] & bits[tri[:, 2]]
        lowest = common & (-common)
        read = out.copy()
        for key, bit in MASK_BITS.items():
            read[lowest == bit] = key
        if np.array_equal(read, out):
            return out
        out = read


def describe_cuts(cuts: List[Passage], scale: float = 1.0) -> str:
    """The cuts made through holes in the wall, for a report."""
    if not cuts:
        return ""
    lines = [f"The wall has {len(cuts)} hole(s) joining the epicardium to a cavity, too small to be valve "
             "openings. The labels were cut there, each ring "
             "belonging to both the epicardium and the cavity:"]
    for cut in cuts:
        c = ", ".join(f"{v:.3g}" for v in cut.centre)
        lines.append(f"  {cut.circumference * scale:.1f} mm around, "
                     f"centre ({c})")
    return "\n".join(lines)


__all__ = ["UNITS", "TYPICAL_RMS_RADIUS_MM", "MAX_VOXELS", "OrificeOptions",
           "Opening", "OpeningSearch", "rms_radius", "guess_unit",
           "find_openings", "label_open_boundary", "label_ventricles",
           "Passage", "find_passages", "describe_passages", "describe_cuts"]
