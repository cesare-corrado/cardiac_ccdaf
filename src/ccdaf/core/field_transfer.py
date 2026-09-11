"""
Field transfer
==============

Carry a mesh's fields onto a different mesh of the same anatomy, where the
two share no vertex correspondence — what a segmentation round trip returns,
since marching cubes builds its surface from the voxels and knows nothing of
the Carto vertices it ultimately came from.

Each destination vertex is matched to the closest point on the source
*surface* (not the closest source vertex, which overshoots it by roughly a
third of an edge length), and the field is read there.

Two rules, following the association the field is stored under:

* **point fields are measurements** — Carto's ``LAT``, ``Bipolar``, … — and
  are interpolated linearly within the triangle the closest point landed in;
* **cell fields are labels** — ``elemTag`` — and are copied from the nearest
  cell, never interpolated. Averaging two labels invents a third.

So a categorical quantity belongs on the cells, which is where ``elemTag``
already lives and where anything joining it should go.

Volumes
-------
:func:`transfer_volume_fields` does the same job between two tetrahedral
meshes, which is what a remesh needs: MMG returns bare geometry, so every
field is carried across by us. The rules are the same with one addition.

A **direction** field — a fibre orientation, one unit vector per element —
is neither a measurement nor a label. It is *axial*: ``f`` and ``-f`` are
the same direction, and which of the two a file happens to store is
arbitrary. Averaging them as vectors is therefore wrong, and not subtly:
two neighbouring elements whose stored vectors point opposite ways average
to nothing at all, and the result is a direction pointing nowhere that
looks like data. :func:`average_axial` averages the outer products
``f·fᵀ`` and takes the dominant eigenvector instead, which is sign-free by
construction and gives the right answer whichever way each contributor was
written down.

No-data
-------
Carto's sentinels arrive as NaN. Interpolating a triangle with one invalid
vertex would spread that NaN across the triangle's whole area, so the weights
are renormalised over the valid vertices instead and a value only goes NaN
when the triangle has nothing valid to offer.

``max_distance`` guards the other direction. Editing a segmentation — a
morphological closing, a painted region, a filled hole — creates surface the
mapping system never measured. Its closest point on the source is the rim of
whatever it grew from, so interpolation would hand it that rim's activation
times: in range, smoothly varying, and indistinguishable from real data.
Beyond ``max_distance`` a point field is NaN instead, which says what is
true — nothing was measured here.

Choosing it is :func:`guard_distance`'s job, and getting it wrong is not
symmetric: too wide and invented wall quietly reads as measurement, too tight
and real measurement is thrown away.

Labels are exempt: new wall is still part of the body, so inheriting the
nearest ``elemTag`` states a fact rather than fabricating a measurement.
"""
from __future__ import annotations

from typing import Callable, Iterable, Optional

import numpy as np
import pyvista as pv
import vtk
from scipy.spatial import cKDTree


# Bookkeeping stamped on by the renderer (for picking) and by taking a
# volume's boundary (the map back to the parent tetrahedron). Not the
# mesh's data, and it must not be carried anywhere — the indices name
# cells of a mesh the destination is not. mesh_postprocessor skips the
# same set for the same reason; it is defined once, in mesh_loader.
from ccdaf.core.mesh_loader import INTERNAL_ARRAYS as _INTERNAL_ARRAYS
from ccdaf.core.volume_mesh import tetrahedra

#: Cell fields whose vectors are *axial* — a direction with no sign, such
#: as a fibre orientation. Named rather than guessed from the component
#: count: a displacement is also three numbers per element and averaging
#: it as an axis would be just as wrong the other way.
AXIAL_CELL_FIELDS: frozenset = frozenset({"fiber", "fibre", "sheet",
                                          "sheet_normal"})


def _median_edge_length(mesh: pv.PolyData) -> float:
    """Median triangle edge — how far apart this mesh's measurements sit."""
    pts = np.asarray(mesh.points, dtype=float)
    faces = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]
    if not len(faces):
        return 0.0
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    return float(np.median(np.linalg.norm(pts[e[:, 0]] - pts[e[:, 1]], axis=1)))


def guard_distance(src: pv.PolyData, spacing) -> float:
    """How far off ``src`` a rebuilt vertex may sit and still be believed.

    A rebuilt vertex is legitimately off the source wall for two independent
    reasons, and the guard has to clear both or it discards real data:

    * **the voxels** — marching cubes can only place the wall to within the
      grid it was rasterised on;
    * **the measurements** — the source is itself a sampling of the anatomy,
      and a point closer to it than one triangle's edge is inside the reach of
      the interpolation, not beyond it.

    Scaling on the voxels alone — which this did — ties the guard to the wrong
    quantity, because what actually moves the wall is the smoothing and the
    morphology, in millimetres. Refine the voxels and the drift stays put while
    the threshold shrinks past it, so the *better* reconstruction throws away
    more data. At 0.5mm it drops under the map's own sampling (~0.9-1.3mm here)
    and starts calling points no-data that sit within a single source triangle,
    1.3mm from a real measurement.

    Taking the larger of the two is what the guard meant all along. At a 1mm
    voxelisation of a Carto shell the voxels win and this is the old rule
    exactly; it only opens up where the source is coarser than the grid.
    """
    scale = max(float(max(spacing)), _median_edge_length(src))
    return 2.0 * scale


def _closest_on_surface(src: pv.PolyData, points: np.ndarray):
    """For each point, the closest point on ``src``, its cell, and the range.

    Returns ``(closest_xyz, cell_ids, distance)``.
    """
    locator = vtk.vtkCellLocator()
    locator.SetDataSet(src)
    locator.BuildLocator()

    n = len(points)
    closest = np.empty((n, 3), dtype=float)
    cells = np.empty(n, dtype=np.int64)
    dist2 = np.empty(n, dtype=float)

    c = [0.0, 0.0, 0.0]
    cid = vtk.mutable(0)
    sub = vtk.mutable(0)
    d2 = vtk.mutable(0.0)
    for i, p in enumerate(points):
        locator.FindClosestPoint(p, c, cid, sub, d2)
        closest[i] = c
        cells[i] = int(cid)
        dist2[i] = float(d2)
    return closest, cells, np.sqrt(dist2)


def _barycentric(closest: np.ndarray, tri_xyz: np.ndarray) -> np.ndarray:
    """Barycentric weights of each point within its triangle.

    ``tri_xyz`` is (n, 3, 3): per point, the three corners. A degenerate
    triangle has no barycentric frame; those fall back to its first corner,
    which is a nearest-vertex copy for that one point.
    """
    p0, p1, p2 = tri_xyz[:, 0], tri_xyz[:, 1], tri_xyz[:, 2]
    v0, v1, v2 = p1 - p0, p2 - p0, closest - p0
    d00 = np.einsum("ij,ij->i", v0, v0)
    d01 = np.einsum("ij,ij->i", v0, v1)
    d11 = np.einsum("ij,ij->i", v1, v1)
    d20 = np.einsum("ij,ij->i", v2, v0)
    d21 = np.einsum("ij,ij->i", v2, v1)
    denom = d00 * d11 - d01 * d01

    ok = np.abs(denom) > 1e-20
    safe = np.where(ok, denom, 1.0)
    v = (d11 * d20 - d01 * d21) / safe
    w = (d00 * d21 - d01 * d20) / safe
    u = 1.0 - v - w

    weights = np.stack([u, v, w], axis=1)
    weights[~ok] = (1.0, 0.0, 0.0)
    # The closest point is on the triangle, so the weights are already in
    # [0, 1] up to rounding; clip rather than let a -1e-16 flip a sign.
    return np.clip(weights, 0.0, 1.0)


def _triangle_vertices(src: pv.PolyData, cells: np.ndarray) -> np.ndarray:
    """The three vertex ids of each named cell, (n, 3)."""
    faces = np.asarray(src.faces).reshape(-1, 4)
    if faces.size and np.any(faces[:, 0] != 3):
        raise ValueError("source mesh must contain triangles only")
    return faces[cells, 1:].astype(np.int64)


def transfer_fields(src: pv.PolyData,
                    dst: pv.PolyData,
                    max_distance: Optional[float] = None,
                    on_status: Optional[Callable[[str], None]] = None) -> None:
    """Copy ``src``'s fields onto ``dst``, in place. ``src`` is not touched.

    ``max_distance`` — beyond which a point field is NaN rather than
    invented. ``None`` disables the guard and lets new surface inherit
    whatever it is nearest to.
    """
    if dst.n_points == 0 or src.n_points == 0 or src.n_cells == 0:
        return

    point_names = [n for n in src.point_data.keys() if n not in _INTERNAL_ARRAYS]
    cell_names = [n for n in src.cell_data.keys() if n not in _INTERNAL_ARRAYS]

    guarded = 0
    if point_names:
        pts = np.asarray(dst.points, dtype=float)
        closest, cells, dist = _closest_on_surface(src, pts)
        vids = _triangle_vertices(src, cells)
        weights = _barycentric(closest, np.asarray(src.points)[vids])

        too_far = (np.zeros(len(pts), dtype=bool) if max_distance is None
                   else dist > float(max_distance))
        guarded = int(too_far.sum())

        for name in point_names:
            arr = np.asarray(src.point_data[name], dtype=float)
            vals = arr[vids]                       # (n, 3) or (n, 3, k)
            w = weights if vals.ndim == 2 else weights[:, :, None]

            valid = np.isfinite(vals)
            wv = np.where(valid, w, 0.0)
            total = wv.sum(axis=1)
            out = np.divide((wv * np.nan_to_num(vals)).sum(axis=1), total,
                            out=np.full_like(total, np.nan),
                            where=total > 0.0)
            out[too_far] = np.nan
            dst.point_data[name] = out

    if cell_names and dst.n_cells:
        src_c = np.asarray(src.cell_centers().points)
        dst_c = np.asarray(dst.cell_centers().points)
        _, cid = cKDTree(src_c).query(dst_c, k=1)
        for name in cell_names:
            arr = np.asarray(src.cell_data[name])
            dst.cell_data[name] = arr[cid]         # dtype, and labels, intact

    if on_status is not None:
        note = ""
        if guarded:
            note = (f"; {guarded} of {dst.n_points} vertices sit further than "
                    f"{max_distance:g} from anything measured and were left "
                    f"as no-data")
        on_status(
            f"Transferred {len(point_names)} point and "
            f"{len(cell_names)} cell fields{note}."
        )


__all__ = ["guard_distance", "transfer_fields"]


# ---------------------------------------------------------------------
# Volumes
# ---------------------------------------------------------------------
def average_axial(vectors: np.ndarray,
                  weights: Optional[np.ndarray] = None) -> np.ndarray:
    """The mean *direction* of ``vectors``, ignoring their signs.

    ``vectors`` is ``(n, 3)``. The mean of an axial quantity is the
    dominant eigenvector of ``Σ w·f·fᵀ``: the sum is unchanged by flipping
    any contributor, which is exactly the invariance an axis has and a
    vector does not.

    Returns a unit vector. Its own sign is arbitrary — that is the point —
    so it is fixed to a deterministic convention (largest component
    positive) rather than left to the eigensolver, or two runs on the same
    input could return opposite arrows.
    """
    v = np.asarray(vectors, dtype=float).reshape(-1, 3)
    if v.shape[0] == 0:
        return np.zeros(3)
    norm = np.linalg.norm(v, axis=1)
    good = norm > 0.0
    if not good.any():
        return np.zeros(3)
    unit = v[good] / norm[good, None]
    w = (np.ones(len(unit)) if weights is None
         else np.asarray(weights, dtype=float).reshape(-1)[good])
    tensor = np.einsum("i,ij,ik->jk", w, unit, unit)
    eigenvalues, eigenvectors = np.linalg.eigh(tensor)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    # Deterministic sign: an axis has none, so pick one and always pick it.
    lead = int(np.argmax(np.abs(axis)))
    if axis[lead] < 0.0:
        axis = -axis
    return axis


def average_axial_grouped(vectors: np.ndarray,
                          owner: np.ndarray,
                          member: np.ndarray,
                          n_groups: int) -> np.ndarray:
    """:func:`average_axial` for many groups at once.

    ``owner[k]`` is the group that ``vectors[member[k]]`` belongs to. The
    same answer as calling :func:`average_axial` per group, and the same
    sign convention, but as a handful of array operations rather than a
    Python loop: on a 279,000-element remesh the loop was 12 of the 14
    seconds the whole field transfer took, and this is a fraction of a
    second. That is the difference between a remesh dominated by MMG and
    one dominated by us.

    The structure tensor of each group is accumulated component by
    component with ``bincount`` (nine of them, and it is symmetric so six
    suffice), then ``eigh`` runs batched over the whole stack.
    """
    out = np.zeros((n_groups, 3), dtype=float)
    v = np.asarray(vectors, dtype=float)[member]
    norm = np.linalg.norm(v, axis=1)
    good = norm > 0.0
    if not good.any():
        return out
    unit = v[good] / norm[good, None]
    grp = np.asarray(owner)[good]

    tensors = np.zeros((n_groups, 3, 3), dtype=float)
    for a in range(3):
        for b in range(a, 3):
            total = np.bincount(grp, weights=unit[:, a] * unit[:, b],
                                minlength=n_groups)
            tensors[:, a, b] = total
            tensors[:, b, a] = total

    # A group that caught nothing has a zero tensor, whose eigenvectors
    # are arbitrary. Left as the zero vector rather than given a
    # direction out of numerical noise.
    populated = np.einsum("nii->n", tensors) > 0.0
    if not populated.any():
        return out

    _values, vectors_out = np.linalg.eigh(tensors[populated])
    axis = vectors_out[:, :, -1]                  # eigh sorts ascending
    lead = np.argmax(np.abs(axis), axis=1)
    sign = np.sign(axis[np.arange(len(axis)), lead])
    sign[sign == 0.0] = 1.0
    out[populated] = axis * sign[:, None]
    return out


def _flatten_neighbourhoods(neighbourhoods) -> "tuple":
    """``(owner, member)`` index arrays for a list of member lists."""
    counts = np.fromiter((len(n) for n in neighbourhoods), dtype=np.int64,
                         count=len(neighbourhoods))
    owner = np.repeat(np.arange(len(neighbourhoods), dtype=np.int64), counts)
    member = (np.concatenate([np.asarray(n, dtype=np.int64)
                              for n in neighbourhoods])
              if counts.sum() else np.zeros(0, dtype=np.int64))
    return owner, member


def _cell_neighbourhoods(src_centres: np.ndarray,
                         dst_centres: np.ndarray,
                         radii: np.ndarray) -> "list":
    """For each destination cell, the source cells near enough to average.

    A ball around the destination centroid, sized by that cell itself, so
    refining picks up the one cell it came from and coarsening picks up
    everything it swallowed. Never empty: a cell that catches nothing
    falls back to its nearest neighbour, because a direction inherited
    from next door beats no direction at all.
    """
    tree = cKDTree(src_centres)
    found = tree.query_ball_point(dst_centres, radii, workers=-1)
    empty = [i for i, f in enumerate(found) if not f]
    if empty:
        _, nearest = tree.query(dst_centres[empty], k=1, workers=-1)
        for i, n in zip(empty, np.atleast_1d(nearest)):
            found[i] = [int(n)]
    return found


def _tet_centres_and_sizes(grid):
    """Centroids and a characteristic radius for every tetrahedron."""
    tets = tetrahedra(grid)
    pts = np.asarray(grid.points, dtype=float)
    corners = pts[tets]                       # (n, 4, 3)
    centres = corners.mean(axis=1)
    # Distance from the centroid to the furthest node: the ball that
    # contains the element, which is what "the cells this one covers"
    # means when the mesh has been coarsened.
    radii = np.linalg.norm(corners - centres[:, None, :], axis=2).max(axis=1)
    return centres, radii


def transfer_volume_fields(src, dst,
                           axial_fields: Optional[Iterable[str]] = None,
                           exclude: Optional[Iterable[str]] = None,
                           on_status: Optional[Callable[[str], None]] = None
                           ) -> None:
    """Copy ``src``'s fields onto the tetrahedral mesh ``dst``, in place.

    Needed because MMG hands back geometry and nothing else: labels,
    fibres and every point field are ours to carry.

    * **point fields** are interpolated within the source tetrahedron the
      destination vertex falls in, and taken from the nearest one for a
      vertex that falls outside the source (which happens at the surface,
      where a remeshed boundary can sit fractionally outside the old one).
    * **cell fields** are copied from the source cell containing the
      destination centroid — labels are never averaged.
    * **axial cell fields** (see :data:`AXIAL_CELL_FIELDS`) are averaged
      over the source cells the destination cell covers, by
      :func:`average_axial`, and renormalised.

    ``exclude`` names fields the caller has already carried across by a
    better route — the remesher brings ``elemTag`` through as an MMG
    element reference, which is exact, and re-deriving it here by
    proximity would only make it worse.

    ``src`` is not modified.
    """
    if dst.n_points == 0 or src.n_points == 0 or src.n_cells == 0:
        return
    axial = frozenset(AXIAL_CELL_FIELDS if axial_fields is None
                      else {str(a) for a in axial_fields})
    skip = _INTERNAL_ARRAYS | frozenset(
        () if exclude is None else {str(e) for e in exclude})

    point_names = [n for n in src.point_data.keys() if n not in skip]
    cell_names = [n for n in src.cell_data.keys() if n not in skip]

    # -- point fields: VTK's probe does the containing-cell interpolation.
    outside = 0
    if point_names:
        sampled = dst.sample(src, pass_cell_data=False,
                             pass_point_data=True, categorical=False)
        valid = np.asarray(
            sampled.point_data.get("vtkValidPointMask",
                                   np.ones(dst.n_points))).astype(bool)
        outside = int((~valid).sum())
        # Anything the probe could not place takes its nearest source
        # vertex. Leaving it at the probe's zero would write a plausible
        # number that was never measured — the same mistake the surface
        # path guards against with max_distance.
        if outside:
            _, near = cKDTree(np.asarray(src.points)).query(
                np.asarray(dst.points)[~valid], k=1, workers=-1)
        for name in point_names:
            arr = np.asarray(sampled.point_data[name])
            if outside:
                arr = np.array(arr, copy=True)
                arr[~valid] = np.asarray(src.point_data[name])[near]
            dst.point_data[name] = arr

    # -- cell fields: the containing source cell, or the nearest.
    if cell_names and dst.n_cells:
        src_centres, _ = _tet_centres_and_sizes(src)
        dst_centres, dst_radii = _tet_centres_and_sizes(dst)
        _, containing = cKDTree(src_centres).query(dst_centres, k=1,
                                                   workers=-1)

        groups = None
        for name in cell_names:
            arr = np.asarray(src.cell_data[name])
            if name in axial and arr.ndim == 2 and arr.shape[1] == 3:
                if groups is None:
                    groups = _flatten_neighbourhoods(_cell_neighbourhoods(
                        src_centres, dst_centres, dst_radii))
                dst.cell_data[name] = average_axial_grouped(
                    arr, groups[0], groups[1], dst.n_cells)
            else:
                dst.cell_data[name] = arr[containing]   # dtype, labels intact

    if on_status is not None:
        note = (f"; {outside} of {dst.n_points} vertices fell outside the "
                f"previous mesh and took their nearest value"
                if outside else "")
        on_status(f"Transferred {len(point_names)} point and "
                  f"{len(cell_names)} cell fields{note}.")
