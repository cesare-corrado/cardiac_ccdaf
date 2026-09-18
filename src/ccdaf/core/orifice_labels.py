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

Measured on three meshes (a reference with gold labels, a second one with
labels from another pipeline, and the example ventricle):

* every valve opening found, each once, except that a mitral and an
  aortic valve lying side by side come back as one opening. That is
  harmless: the base there is one ring instead of two;
* on the gold mesh, 93% of the loop's nodes lie on the gold base, and the
  epicardium, LV and RV agree with gold on 99.2%, 99.3% and 99.9% of their
  area;
* the band must stay narrow: at 8 mm the loop slips into the LV cavity
  and the LV agreement falls to 96.9%.

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
    BASE, UNLABELLED, LabelOptions, Plane, SurfaceLabels, _as_boundary,
    _areas, _assemble, _close_ring, _faces, _hull_depth, _name_pieces,
    label_boundary,
)
from ccdaf.core.volume_mesh import is_volume

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
    scale = options.mm_per_unit
    h = options.voxel_mm
    points = np.asarray(surface.points, dtype=float) * scale

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
    touching = ndimage.binary_dilation(outside)

    openings: List[Opening] = []
    for index, k in enumerate(pool_ids):
        rim = (labels == k) & touching
        parts, n_parts = ndimage.label(rim, structure=np.ones((3, 3, 3)))
        sizes = np.bincount(parts.ravel())[1:] * h ** 2
        for j in np.argsort(sizes)[::-1]:
            if sizes[j] < options.min_opening_mm2:
                break
            vox = np.argwhere(parts == j + 1) * h + low
            centre = vox.mean(0)
            if len(vox) >= 3:
                _w, v = np.linalg.eigh(np.cov((vox - centre).T))
                normal = v[:, 0]
            else:
                normal = np.array([0.0, 0.0, 1.0])
            openings.append(Opening(pool=index, centre=centre / scale,
                                    normal=normal, area_mm2=float(sizes[j]),
                                    points=vox / scale))
    return OpeningSearch(pool_ml=[float(ml[k - 1]) for k in pool_ids],
                         openings=openings)


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
    s, t = n_faces, n_faces + 1
    big = np.int32(2 ** 30)
    keep = free[fa] | free[fb]
    src = np.where(source)[0]
    snk = np.where(sink)[0]
    rows = np.concatenate([fa[keep], fb[keep], np.full(len(src), s), snk])
    cols = np.concatenate([fb[keep], fa[keep], src, np.full(len(snk), t)])
    caps = np.concatenate([capacity[keep], capacity[keep],
                           np.full(len(src), big), np.full(len(snk), big)])
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
    if openings is None:
        openings = find_openings(surface, options).openings
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

    count, pieces = _components(~band, fa, fb)
    sizes = np.bincount(pieces[pieces >= 0], weights=area[pieces >= 0],
                        minlength=count)
    significant = np.where(sizes >= _SIGNIFICANT_SHARE * sizes.sum())[0]
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

    count, pieces = _components(~base, fa, fb)
    sizes = np.bincount(pieces[pieces >= 0], weights=area[pieces >= 0],
                        minlength=count)
    significant = np.where(sizes >= _SIGNIFICANT_SHARE * sizes.sum())[0]
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
    if stray.any():
        note = (note + f" {int(stray.sum())} stray face(s) next to a ring "
                "were added to the base.").strip()
    return _assemble(surface, tri, face_labels, area, plane=None,
                     naming_confident=confident, note=note,
                     dropped_patches=0, method="openings",
                     openings=list(openings))


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


__all__ = ["UNITS", "TYPICAL_RMS_RADIUS_MM", "MAX_VOXELS", "OrificeOptions",
           "Opening", "OpeningSearch", "rms_radius", "guess_unit",
           "find_openings", "label_open_boundary", "label_ventricles"]
