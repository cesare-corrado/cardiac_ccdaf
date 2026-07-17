"""
EAM loader
==========

Turn a Carto electroanatomical mapping (EAM) into the objects the GUI works
with:

* :func:`carto_mesh_to_polydata` — a ``read_carto_mesh_file`` dict becomes a
  ``pyvista.PolyData`` carrying each Carto vertex-colour column as a named
  point-data field (e.g. ``Unipolar``, ``Bipolar``, ``LAT``) and an
  ``elemTag`` cell array initialised to the body label (1).
* :func:`load_carto_mapping` — given a directory and a *map name*, read the
  ``<map>.mesh`` geometry/fields and the ``<map>_car.txt`` electrodes and
  bundle them in an :class:`EAMData`.
* :func:`displace_electrodes` — when the surface is smoothed, carry the
  electrodes along with it.

Carto flags missing vertex-colour data with a fill value; ``-10000`` is the
documented invalid flag and ``+10000`` is seen as a no-data fill on real
exports (e.g. the whole ``Impedance`` column). Both are converted to ``NaN``
so they drop out of colour scales instead of flattening them.
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pyvista as pv

from ccdaf.io.carto_functions import read_carto_mesh_file, load_carto_electrodes


# Carto vertex-colour sentinel magnitude: |value| >= this ⇒ no data.
CARTO_NODATA: float = 10000.0

# --- electrode displacement (see displace_electrodes) ------------------
# Kernel: a modelling choice, not a tunable. It decays with distance and takes
# no polynomial tail, so the warp fades to "no motion" away from the surface.
# thin_plate_spline is the obvious alternative and is wrong here: scipy forces
# degree >= 1 on it, so its far field GROWS, and electrodes metres off the wall
# get thrown further than any vertex moved.
EAM_RBF_KERNEL: str = "inverse_multiquadric"
# Centres are chosen to cover the anatomy, so this scales with the geometry,
# not with how finely it happens to be tessellated. 3000 reaches ~0.07mm on a
# Carto atrium; more costs sharply and starts hurting the far field.
EAM_RBF_CENTRES: int = 3000
# Candidate length scales for the search, as multiples of the centre spacing
# (the fill distance of the chosen centres). Relative rather than absolute so
# the search space follows the data: a mesh exported in cm, or a different
# centre count, rescales the candidates instead of invalidating them. Below
# ~1x the spacing the kernel fades between neighbouring centres and the fit
# falls apart; far above it the system goes ill-conditioned.
EAM_RBF_SCALE_MULTIPLES: tuple = (0.5, 1.0, 2.0, 3.0, 5.0, 8.0)
# Regularisations. Dimensionless, so these need no scaling.
EAM_RBF_SMOOTHINGS: tuple = (0.0, 1e-4, 1e-3, 1e-2)

# --- electrode displacement without correspondence ---------------------
# Newton steps taken by displace_electrodes_by_distance. The field's gradient
# is exactly unit length, so a step lands on the target distance to first
# order and two are already past convergence for a wall that moved a
# fraction of a millimetre; the third is margin, and costs ~0.3s on a full
# catheter load.
EAM_SDF_ITERATIONS: int = 3



@dataclass
class EAMData:
    """Everything the GUI needs to display one Carto mapping."""
    mesh: pv.PolyData                       # surface + point fields + elemTag
    field_names: List[str]                  # point-data field names, in file order
    electrode_points: np.ndarray            # (N, 3) electrode coordinates
    map_name: str
    electrodes: Optional[dict] = field(default=None)   # raw load_carto_electrodes() output


def carto_mesh_to_polydata(mesh0: dict) -> pv.PolyData:
    """Build a ``pyvista.PolyData`` from a ``read_carto_mesh_file`` dict.

    Each ``VertexColors`` column named in ``ColorsNames`` becomes a point-data
    field (no-data sentinels → NaN). ``elemTag`` cell data is initialised to
    the body label (1), matching the rest of the pipeline.
    """
    X = np.asarray(mesh0["X"], dtype=float)
    Tri = np.asarray(mesh0["Tri"], dtype=np.int64)
    if X.size == 0 or Tri.size == 0:
        raise ValueError("Carto mesh has no geometry (empty X or Tri).")

    faces = np.empty((Tri.shape[0], 4), dtype=np.int64)
    faces[:, 0] = 3
    faces[:, 1:] = Tri
    poly = pv.PolyData(X, faces.ravel())

    vc = mesh0.get("VertexColors")
    ids = mesh0.get("ColorsIDs")
    names = mesh0.get("ColorsNames")
    if vc is not None and ids is not None and names is not None:
        vc = np.asarray(vc, dtype=float)
        for cid, cname in zip(np.asarray(ids).ravel(), np.asarray(names).ravel()):
            col = np.array(vc[:, int(cid)], dtype=float)
            col[np.abs(col) >= CARTO_NODATA] = np.nan
            poly.point_data[str(cname)] = col

    poly.cell_data["elemTag"] = np.ones(poly.n_cells, dtype=np.int32)
    return poly


def read_bundle(path: str
                ) -> Tuple[pv.PolyData, Optional[Dict[str, np.ndarray]],
                           Optional[dict]]:
    """Read a File → Save pickle bundle back into working objects.

    The inverse of :func:`eam_export.export_binary` used with the bundle
    keys: the ``"surface"`` dict becomes a PolyData through
    :func:`carto_mesh_to_polydata`, the ``"elemTag"`` key (if present)
    restores the cell tags the Carto surface dict cannot carry, and the
    ``"seeds"`` / ``"electrodes"`` keys come back as they were saved.

    Returns ``(mesh, seeds, electrodes)`` — ``seeds`` a name → xyz mapping
    or ``None``, ``electrodes`` the raw record or ``None``.
    """
    with open(path, "rb") as fh:
        payload = pickle.load(fh)
    if not isinstance(payload, dict) or "surface" not in payload:
        raise ValueError(f"{os.path.basename(path)} is not a mesh bundle "
                         "(no 'surface' key).")

    mesh = carto_mesh_to_polydata(payload["surface"])
    if "elemTag" in payload:
        tags = np.asarray(payload["elemTag"], dtype=np.int32).ravel()
        if tags.shape[0] == mesh.n_cells:
            mesh.cell_data["elemTag"] = tags

    seeds: Optional[Dict[str, np.ndarray]] = None
    if payload.get("seeds"):
        seeds = {}
        for name, xyz in payload["seeds"].items():
            arr = np.asarray(xyz, dtype=float).reshape(-1)
            if arr.shape == (3,) and np.isfinite(arr).all():
                seeds[str(name)] = arr

    electrodes = payload.get("electrodes")
    return mesh, seeds, electrodes


def load_carto_mapping(directory: str, map_name: str) -> EAMData:
    """Load ``<map_name>.mesh`` (+ optional ``<map_name>_car.txt``).

    Raises if the mesh file is missing or unreadable (e.g. a Carto export
    lacking a vertex-colours section); the caller is expected to surface that.
    """
    mesh_path = os.path.join(directory, f"{map_name}.mesh")
    if not os.path.isfile(mesh_path):
        raise FileNotFoundError(f"mesh file not found: {mesh_path}")

    mesh0 = read_carto_mesh_file(mesh_path)
    poly = carto_mesh_to_polydata(mesh0)
    field_names = list(poly.point_data.keys())

    car_path = os.path.join(directory, f"{map_name}_car.txt")
    elec_points = np.empty((0, 3), dtype=float)
    electrodes: Optional[dict] = None
    if os.path.isfile(car_path):
        electrodes = load_carto_electrodes(car_path)
        data = np.asarray(electrodes.get("data"), dtype=float)
        if data.ndim == 2 and data.shape[0] > 0 and data.shape[1] >= 4:
            elec_points = data[:, 1:4]

    return EAMData(
        mesh=poly,
        field_names=field_names,
        electrode_points=elec_points,
        map_name=map_name,
        electrodes=electrodes,
    )


def _even_centres(pts: np.ndarray, n: int) -> np.ndarray:
    """Indices of ``n`` points spread evenly over ``pts`` (farthest-point
    sampling). Even coverage is what keeps the RBF accurate everywhere;
    clustering the centres — near the electrodes, or on the largest
    displacements — starves the rest of the surface and the fit degrades.
    """
    n = int(min(n, len(pts)))
    chosen = np.empty(n, dtype=np.int64)
    chosen[0] = 0
    dist = np.linalg.norm(pts - pts[0], axis=1)
    for i in range(1, n):
        chosen[i] = int(np.argmax(dist))
        dist = np.minimum(dist, np.linalg.norm(pts - pts[chosen[i]], axis=1))
    return chosen


def displace_electrodes(electrodes: np.ndarray,
                        old_points: np.ndarray,
                        new_points: np.ndarray,
                        n_centres: int = EAM_RBF_CENTRES,
                        length_scale: Optional[float] = None,
                        smoothing: Optional[float] = None,
                        on_status: Optional[Callable[[str], None]] = None
                        ) -> np.ndarray:
    """Carry electrodes along with a smoothed surface.

    The mesh gives a displacement sampled at its vertices; the electrodes sit
    *off* the surface (a couple of mm out, typically), so the field has to be
    extended into the surrounding space. Fitting a radial basis function to
    the vertex displacements gives that: a smooth deformation of space, which
    the electrodes then simply ride —  ``el + f(el)``.

    ``old_points`` and ``new_points`` must be the same vertices before and
    after, in the same order (see :func:`mesh_postprocessor.smooth`).

    ``length_scale`` is how far a vertex's motion still reaches, in mesh
    units, and ``smoothing`` how loosely the fit tracks the data. Left as
    ``None`` they are chosen by search: candidates are scored on held-out
    vertices, but only among those that keep every electrode within the
    surface's own maximum displacement. That constraint is load-bearing —
    scoring on held-out vertices alone measures the fit *on* the surface,
    which is not what we evaluate, and it prefers wide kernels that fit
    marginally better there while throwing distant electrodes far past
    anything the wall did.

    The candidates are multiples of the centre spacing rather than fixed
    lengths, so the search space follows the mesh's own scale and units.

    ``on_status`` receives a one-line summary of what was chosen.

    Returns the displaced electrode coordinates; an empty input returns empty.
    """
    from scipy.interpolate import RBFInterpolator
    from scipy.spatial import cKDTree

    electrodes = np.asarray(electrodes, dtype=float)
    old_points = np.asarray(old_points, dtype=float)
    new_points = np.asarray(new_points, dtype=float)
    if electrodes.size == 0 or old_points.size == 0:
        return electrodes
    if old_points.shape != new_points.shape:
        raise ValueError("old_points and new_points must correspond 1:1")

    disp = new_points - old_points
    field_max = float(np.linalg.norm(disp, axis=1).max())
    if field_max == 0.0:            # nothing moved
        return electrodes

    # Leave vertices over to score on. Without this a mesh smaller than
    # n_centres makes every vertex a centre, the held-out error comes back
    # NaN for every candidate, and the search silently returns the electrodes
    # unmoved. No effect on a Carto surface, which has multiples of n_centres.
    centres = _even_centres(old_points, min(n_centres,
                                            max(1, int(0.8 * len(old_points)))))
    holdout = np.setdiff1d(np.arange(len(old_points)), centres)
    if holdout.size > 2000:         # scoring a sample is enough, and cheaper
        holdout = holdout[np.linspace(0, holdout.size - 1, 2000).astype(int)]

    def fit(scale: float, smooth_val: float):
        return RBFInterpolator(old_points[centres], disp[centres],
                               kernel=EAM_RBF_KERNEL, epsilon=1.0 / scale,
                               smoothing=smooth_val)

    if length_scale is not None and smoothing is not None:
        return electrodes + fit(length_scale, smoothing)(electrodes)

    # Candidate scales follow the centre spacing, so they carry the mesh's
    # own units rather than assuming millimetres.
    spacing = float(cKDTree(old_points[centres]).query(old_points, k=1)[0].max())
    scales = ((length_scale,) if length_scale is not None
              else tuple(mult * spacing for mult in EAM_RBF_SCALE_MULTIPLES))
    smooths = (smoothing,) if smoothing is not None else EAM_RBF_SMOOTHINGS
    # Rank by (electrodes over the limit, held-out error) and take the first.
    # With any clean candidate that is "best fit among those that violate
    # nothing"; with none — a field the kernels cannot extend without
    # overshooting somewhere — it degrades to the least-violating candidate
    # rather than to whichever merely fits the surface best.
    best = None
    for scale in scales:
        for smooth_val in smooths:
            try:
                f = fit(scale, smooth_val)
                moved = f(electrodes)
                err = float(np.linalg.norm(f(old_points[holdout])
                                           - disp[holdout], axis=1).mean())
            except Exception:       # singular system for this candidate
                continue
            if not np.isfinite(err):
                continue
            over = int((np.linalg.norm(moved, axis=1) > field_max).sum())
            rank = (over, err)
            if best is None or rank < best[0]:
                best = (rank, moved, scale, smooth_val)
    if best is None:
        return electrodes

    (over, err), moved, scale, smooth_val = best
    if on_status is not None:
        note = "" if over == 0 else (
            f" — no candidate kept every electrode inside it; this one "
            f"exceeds it for {over} of {len(electrodes)}")
        on_status(
            f"Electrodes follow the surface: length scale {scale:.2f} "
            f"({scale/spacing:.1f}x centre spacing), smoothing {smooth_val:g}, "
            f"fit {err:.3f}, moved <= {np.linalg.norm(moved, axis=1).max():.2f} "
            f"of the surface's {field_max:.2f}{note}."
        )
    return electrodes + moved


def _outward(mesh: "pv.PolyData") -> "pv.PolyData":
    """A copy of ``mesh`` wound so that its outside reads as outside.

    The sign of a signed distance field comes from the triangle winding, and
    the two surfaces handed to :func:`displace_electrodes_by_distance` need
    not agree on it: a Carto mesh arrives wound outward, while the surface
    marching cubes returns here is deliberately flipped
    (``vtkPolyDataNormals.FlipNormalsOn``) for rendering. Left alone, one
    field reads +3mm where the other reads -3mm, and every electrode is
    driven clean through the wall to the far side.

    Cheap, and a no-op on a mesh already wound correctly.
    """
    return mesh.compute_normals(
        cell_normals=True, point_normals=False, split_vertices=False,
        consistent_normals=True, auto_orient_normals=True, inplace=False)


def _sdf_values(field, pts: np.ndarray) -> np.ndarray:
    return np.array([field.EvaluateFunction(p) for p in pts], dtype=float)


def _sdf_gradients(field, pts: np.ndarray) -> np.ndarray:
    out = np.empty((len(pts), 3), dtype=float)
    g = [0.0, 0.0, 0.0]
    for i, p in enumerate(pts):
        field.EvaluateGradient(p, g)
        out[i] = g
    return out


def displace_electrodes_by_distance(electrodes: np.ndarray,
                                    old_mesh: "pv.PolyData",
                                    new_mesh: "pv.PolyData",
                                    iterations: int = EAM_SDF_ITERATIONS,
                                    on_status: Optional[Callable[[str], None]] = None
                                    ) -> np.ndarray:
    """Carry electrodes across a remesh, keeping each one's distance to the wall.

    Unlike :func:`displace_electrodes`, this needs no correspondence between
    the two surfaces: it reads them as *sets*, through their signed distance
    fields. That is what makes it usable after a segmentation round trip,
    where marching cubes returns a point set with no relation to the Carto
    vertices it came from.

    The invariant is the one that matters for a mapping — an electrode 2mm
    off the wall stays 2mm off the wall, on the same side, since it is the
    distance to tissue that governs the signal. Each electrode moves along
    the new field's gradient (i.e. normal to the new wall) until the new wall
    is as far away as the old one was::

        x <- x + (d0 - d1(x)) * grad d1(x)

    which is a Newton step, exact to first order because the gradient of a
    distance field is unit length. Motion is purely normal: no tangential
    component is recovered, because between two surfaces-as-sets there is
    none to recover.

    The step taken is exactly ``d0(x) - d1(x)``: how much the wall moved, as
    seen from ``x``. Two consequences worth being precise about, because they
    are not "the correction decays with distance" — it does not:

    * it is **bounded** by the largest difference between the two surfaces,
      everywhere and for free. No electrode can be thrown further than the
      wall itself shifted. That is the guarantee :func:`displace_electrodes`
      has to buy with a decaying kernel plus a rejection constraint, and
      which a thin-plate spline cannot provide at all;
    * it is zero where the *nearest* piece of wall did not move, however
      close or far that electrode is. Locality here is by closest point, not
      by distance from the anatomy: an electrode far out along the axis of a
      bump still tracks that bump, because the bump is what it is nearest to.

    So a wall that moved everywhere — a uniform reconstruction offset —
    moves every electrode with it. That is the invariant being honoured, not
    a failure of it.

    Neither mesh is touched. Returns the displaced coordinates; an empty
    input returns empty.
    """
    import time
    import vtk

    electrodes = np.asarray(electrodes, dtype=float)
    if electrodes.size == 0:
        return electrodes

    t0 = time.perf_counter()
    # Both fields must agree on which side is which; see _outward.
    d0_field = vtk.vtkImplicitPolyDataDistance()
    d0_field.SetInput(_outward(old_mesh))
    d1_field = vtk.vtkImplicitPolyDataDistance()
    d1_field.SetInput(_outward(new_mesh))

    d0 = _sdf_values(d0_field, electrodes)
    moved = electrodes.copy()
    for _ in range(max(1, int(iterations))):
        d1 = _sdf_values(d1_field, moved)
        grad = _sdf_gradients(d1_field, moved)
        norm = np.linalg.norm(grad, axis=1, keepdims=True)
        # The gradient is unit length wherever the field is differentiable;
        # guard only against the zero it can return on the medial axis, where
        # leaving the electrode put is the right answer anyway.
        np.divide(grad, norm, out=grad, where=norm > 1e-12)
        moved += (d0 - d1)[:, None] * grad

    if on_status is not None:
        residual = float(np.abs(d0 - _sdf_values(d1_field, moved)).max())
        step = np.linalg.norm(moved - electrodes, axis=1)
        on_status(
            f"Electrodes follow the wall: moved {step.mean():.3f} on average, "
            f"{step.max():.3f} at most; distance to the wall held to "
            f"{residual:.2g} over {len(electrodes)} electrodes "
            f"in {time.perf_counter() - t0:.2f}s."
        )
    return moved


def displace_electrodes_for(electrodes: np.ndarray,
                            old_mesh: "pv.PolyData",
                            new_mesh: "pv.PolyData",
                            on_status: Optional[Callable[[str], None]] = None
                            ) -> np.ndarray:
    """Carry electrodes across a surface change. The entry point to use.

    :func:`displace_electrodes_by_distance` does the work. It is the default
    on every path — including smoothing, where the vertex correspondence the
    RBF wants does exist — because on a Carto map more than half of what
    smoothing does to the vertices is tangential: the mesh evening out its own
    tessellation, not the wall moving. An electrode has no identity on the
    surface to slide with, so following that field means following an
    artefact. Measured on ``1-TACHY 335MS``: 54% tangential, and the RBF puts
    15 of 6720 electrodes on the wrong side of the wall, where this leaves
    none. It is also ~13x faster and has nothing to tune.

    :func:`displace_electrodes` survives as a fallback for the case the
    distance field cannot serve. That is not "the surface has holes" — the
    field degrades gracefully, and 40% of a Carto mesh can be cut away before
    even 4 electrodes in 2215 take the wrong sign — but a mesh degenerate
    enough to fault. The fallback needs the 1:1 vertex correspondence, so it
    is only available when the point count survived.
    """
    electrodes = np.asarray(electrodes, dtype=float)
    if electrodes.size == 0:
        return electrodes
    try:
        return displace_electrodes_by_distance(
            electrodes, old_mesh, new_mesh, on_status=on_status)
    except Exception as exc:
        corresponds = old_mesh.n_points == new_mesh.n_points
        if not corresponds:
            raise
        if on_status is not None:
            on_status(f"Distance field unusable ({exc}); fell back to the RBF.")
        return displace_electrodes(
            electrodes,
            np.asarray(old_mesh.points, dtype=float),
            np.asarray(new_mesh.points, dtype=float),
            on_status=on_status)


__all__ = [
    "EAMData",
    "carto_mesh_to_polydata",
    "load_carto_mapping",
    "displace_electrodes",
    "displace_electrodes_by_distance",
    "displace_electrodes_for",
    "CARTO_NODATA",
    "EAM_RBF_KERNEL",
    "EAM_RBF_CENTRES",
    "EAM_RBF_SCALE_MULTIPLES",
    "EAM_SDF_ITERATIONS",
]
