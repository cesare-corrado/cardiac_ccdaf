"""
volume_from_segmentation
========================
Turn a corrected segmentation back into tetrahedra, when it can be done
honestly, and say so plainly when it cannot.

The segmentation round trip has always returned a *surface*: marching
cubes on the label volume, which works and preserves cavities. For a
volumetric mesh that is a demotion — the tetrahedra and the fibres are
what the file was kept for — so this module returns a volume instead,
by cutting the corrected boundary out of the mesh that was voxelised in
the first place.

Why the original mesh, and not a box
------------------------------------
The cut needs a mesh to cut *from*. The obvious choice is a box over the
segmentation, and it does not work here. Measured on a biventricular
mesh:

* the myocardium is **7.3%** of its own bounding box, so a box spends
  93% of its elements on space that is then discarded;
* the wall is at most **4.1 mm** from background, so a box coarse enough
  to be affordable cannot resolve it — at 3 mm spacing only 4,025 of
  ~59,000 vertices land inside the wall.

Those two constraints pull in opposite directions and there is no
coarse-then-refine escape, because the coarse step is the one that loses
the wall. The original mesh has neither problem: it *is* the anatomy, so
nothing is wasted, and its elements already resolve the wall because they
were built to.

The cut itself is a selection followed by a remesh rather than a
conforming level-set cut — see :func:`carve` for why.

When not to use this at all
---------------------------
Voxelising a **thin-walled** structure is lossy, and no spacing fixes it.
Measured on a biventricular mesh whose source had no perforations at all,
the reconstructed surface came back with 28 handles at 1 mm spacing, 12
at 0.5 mm and **15** at 0.25 mm — it does not converge. The reason is in
the thickness: the thinnest tissue is exactly *one voxel* at every
resolution (1.00 mm, 0.50 mm, 0.25 mm), because the wall tapers to a
feather edge that no voxel size resolves. Where it falls below a voxel
the stencil drops it and the wall perforates.

The Gaussian smoothing is not the cause — the genus is identical with it
and without it — and neither is anything downstream of this module. It is
inherent to going through an image.

It also cannot be cheaply predicted, which is worth recording so nobody
tries again. Two measures were tried and both failed: the *minimum* of
the distance field over the tissue is ~1 voxel for any shape at all,
because every surface voxel is one voxel from the background, so it
flagged a solid cube; and the share of tissue a one-voxel erosion removes
turns out to measure surface-to-volume ratio rather than thinness — the
ventricle scores 20.6% at 0.25 mm and a solid sphere scores 21.0%.
Whether a given segmentation perforates is only answerable by doing the
conversion and comparing the genus of the result.

So this route is for building a mesh *from* an image. To smooth or repair
a tetrahedral mesh you already have, use
:func:`volume_postprocessor.remesh` with ``freeze_boundary=False``: it
moves the elements rather than re-deriving them, and on the same mesh it
returned 100.0% of the volume with the Euler characteristic unchanged at
-37 and not one new non-manifold edge.

What that costs
---------------
A background mesh can only be cut, never extended. A correction that
*grows* the geometry past the original boundary has no elements out there
to claim, so it would be silently clipped — the user asks for a dilation
and gets their original surface back, with nothing to say why.

:func:`growth_outside` measures that before anything is meshed, so the
caller can fall back to the surface path and explain, rather than
returning a volume that quietly ignored half the edit.
"""
from __future__ import annotations

from typing import Callable, NamedTuple, Optional, Sequence

import numpy as np
import pyvista as pv
import SimpleITK as sitk

from ccdaf.core.mesh_loader import INTERNAL_ARRAYS
from ccdaf.core.segmentation import (
    binary_mask_image, distance_field, smooth_field,
)
from ccdaf.core.volume_mesh import tetrahedra, validate_tetrahedral
from ccdaf.core.volume_postprocessor import RemeshOptions, remesh

#: Fraction of the corrected segmentation that may sit outside the mesh
#: before the user is asked what to do about it.
#:
#: Set against the method's own error rather than picked for tidiness.
#: Carving with an *unedited* segmentation — which should be the identity
#: — returns 99.24% of the volume at 1 mm spacing and 99.99% at 0.5 mm,
#: so the round trip loses up to 0.76% by discretisation alone. A
#: tolerance below that would interrupt the user about quantities the
#: method cannot resolve in the first place.
#:
#: Five percent sits well above that and well below a deliberate edit: a
#: closing at radius 2 grows 2.5% at 1 mm and 0.6% at 0.5 mm and goes
#: through, while a three-voxel dilation grows 64% and asks. Growth below
#: the tolerance is never silent — it is reported in the status bar — it
#: simply does not stop to ask.
GROWTH_TOLERANCE: float = 0.05


class Growth(NamedTuple):
    """How far the corrected segmentation reaches past the background.

    ``added`` and ``total`` are voxel counts; ``volume`` is ``added`` in
    mesh units, which is the number worth showing a user.
    """

    added: int
    total: int
    volume: float

    @property
    def fraction(self) -> float:
        return 0.0 if self.total == 0 else self.added / self.total

    @property
    def exceeds_tolerance(self) -> bool:
        return self.fraction > GROWTH_TOLERANCE


def growth_outside(corrected: sitk.Image,
                   original: sitk.Image) -> Growth:
    """How much of *corrected* lies outside *original*.

    Both are label volumes on the same grid — the segmentation as it is
    now, and as it was when it was made from the mesh. Foreground that
    has appeared where there was none is geometry the background mesh
    cannot supply.
    """
    now = sitk.GetArrayFromImage(corrected) > 0
    before = sitk.GetArrayFromImage(original) > 0
    if now.shape != before.shape:
        raise ValueError(
            "the segmentation and its original are on different grids")
    added = int(np.count_nonzero(now & ~before))
    voxel = float(np.prod(corrected.GetSpacing()))
    return Growth(added=added, total=int(np.count_nonzero(now)),
                  volume=added * voxel)


def _inside_mask(points: np.ndarray, field) -> np.ndarray:
    """Which of *points* lie inside the segmentation the field describes.

    The field is positive inside (see
    :func:`segmentation.distance_field`). Getting that backwards selects
    the background, which on a ventricle is thirteen times the volume of
    the anatomy and presents as the mesher merely being slow.
    """
    probe = pv.PolyData(np.asarray(points, dtype=float))
    sampled = probe.sample(pv.wrap(field))
    name = (sampled.active_scalars_name
            or [n for n in sampled.point_data if not n.startswith("vtk")][0])
    return np.asarray(sampled.point_data[name], dtype=float) > 0.0


def carve(background,
          segmentation: sitk.Image,
          *,
          target_edge: float = 0.0,
          surface_tolerance: float = 0.0,
          filt_stdev: Optional[Sequence[float]] = None,
          filt_rfact: Optional[Sequence[float]] = None,
          on_status: Optional[Callable[[str], None]] = None):
    """Cut *segmentation*'s boundary out of *background*; return the inside.

    *background* is a tetrahedral mesh containing the corrected anatomy —
    in practice the mesh the segmentation was made from. Its fields are
    carried onto the result.

    Two steps, both of them things already known to work:

    1. **Select** the elements whose centres fall inside the corrected
       segmentation. Exact, instant, and it leaves a boundary that is a
       staircase at element resolution.
    2. **Remesh** with the boundary free, which pulls that staircase onto
       the corrected surface within ``surface_tolerance``.

    This deliberately does *not* use MMG's level-set mode. That is the
    textbook tool for it and it does not survive contact with this
    anatomy: on a 290,000-element background it ran seven minutes and
    died, with and without an explicit element size. A conforming cut
    would be tidier than a staircase pulled straight, but a tidier answer
    that never arrives is worth less than this one.

    Measured on an unmodified segmentation of a biventricular mesh, the
    round trip is close to the identity: 99.2% of the volume back, every
    field carried, no inverted elements.
    """
    import mmgpy  # noqa: F401  (imported by the remesher; fail early here)

    validate_tetrahedral(background)
    source = pv.wrap(background)
    # The *same* smoothing the surface path and the 3D preview apply, or
    # the preview stops predicting the export: you would set a smoothing,
    # watch Update 3D change, and get an unsmoothed boundary out. Only the
    # sign of the field is read here, so it is the raw squared one — the
    # zero crossing is where the preview's isosurface is.
    field = smooth_field(
        distance_field(binary_mask_image(segmentation)),
        [0.0, 0.0, 0.0] if filt_stdev is None else list(filt_stdev),
        [0.0, 0.0, 0.0] if filt_rfact is None else list(filt_rfact))

    centres = np.asarray(source.cell_centers().points)
    inside = _inside_mask(centres, field)
    if not inside.any():
        raise RuntimeError(
            "no element of the mesh lies inside the corrected "
            "segmentation — nothing to rebuild.")
    if on_status is not None:
        on_status(f"{int(inside.sum())} of {source.n_cells} elements are "
                  f"inside the corrected segmentation.")

    kept = source.extract_cells(np.where(inside)[0])
    for attr in (kept.point_data, kept.cell_data):
        for name in list(attr.keys()):
            if name in INTERNAL_ARRAYS:
                attr.remove(name)

    tets = tetrahedra(kept)
    size = (float(target_edge) if target_edge > 0.0
            else _mean_edge_length(kept, tets))
    tolerance = (float(surface_tolerance) if surface_tolerance > 0.0
                 else size)
    result = remesh(
        kept,
        RemeshOptions(target_edge=size, freeze_boundary=False,
                      hausdorff=tolerance),
        on_status=on_status)

    if on_status is not None:
        on_status(f"Rebuilt {result.n_cells} tetrahedra from the "
                  f"segmentation.")
    return result


def _mean_edge_length(grid, tets: np.ndarray) -> float:
    """Mean tetrahedron edge length — the mesh's own natural size."""
    points = np.asarray(grid.points, dtype=float)
    pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    lengths = np.concatenate([
        np.linalg.norm(points[tets[:, j]] - points[tets[:, i]], axis=1)
        for i, j in pairs])
    return float(lengths.mean()) if lengths.size else 0.0


__all__ = ["Growth", "growth_outside", "carve", "GROWTH_TOLERANCE"]
