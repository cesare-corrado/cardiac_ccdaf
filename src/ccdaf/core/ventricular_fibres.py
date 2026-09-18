"""
ventricular_fibres
==================
Fibres for a labelled ventricular volume: the whole pipeline behind
Actions > Generate fibres.

It joins three pieces that exist on their own:

* the **surface labels** (:mod:`ccdaf.core.surface_labels`), stored on the
  volume as a per-node bit mask, give the base, the epicardium and the two
  endocardia;
* the **Laplace solves** (:mod:`ccdaf.core.laplace`) turn them, with an
  apex, into the four fields of Bayer et al.;
* the **rules** (:mod:`ccdaf.core.ldrb`) turn the fields into a fibre and a
  sheet per element.

The apex
--------
The labels do not name an apex, and the paper only says it is "the point
lying closest to the ventricular apex". Here it is found as the
epicardial node farthest from the base, measured across a plane fitted to
the base nodes, and widened to the epicardial nodes within one mean edge
length of it. A single held node would put a point singularity in the
apicobasal field; a small patch does not. The caller may replace the node,
and the patch is grown around the replacement the same way.

What is written
---------------
``fiber`` and ``sheet`` on the cells, the names the CARP export reads;
the four fields on the points (``ldrb_ab``, ``ldrb_epi``, ``ldrb_lv``,
``ldrb_rv``), so a rerun with new angles skips the solves; and, in the
field data, a digest of the geometry and labels plus the apex nodes used.

The digest is how a result is told to still describe its mesh. When the
mesh is replaced, :func:`reconcile` compares and decides:

* **kept**: same geometry, same labels. Nothing to do.
* **carried**: the geometry changed (a remesh or a clean). The remesh
  carries fibres across on purpose, averaging them as axes, and a clean
  keeps each surviving element's own, so they stay. They are marked as
  carried and lose the digest, so the next run solves afresh instead of
  reusing interpolated fields as if they were solutions.
* **removed**: the same geometry with different labels. The fibres were
  built on surfaces that no longer exist, and nothing can carry them
  across, so everything this module wrote goes.

Fibres the mesh arrived with, which carry no digest, are left alone: they
are not this module's to judge.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

from ccdaf.core.laplace import DEFAULT_RTOL, SolveReport, laplace, stiffness_matrix
from ccdaf.core.ldrb import FibreAngles, Fibres, fibres
from ccdaf.core.surface_labels import (
    BASE, EPI, LABEL_MASK_FIELD, LV_ENDO, MASK_BITS, NAMES, RV_ENDO,
)
from ccdaf.core.volume_mesh import is_volume, tetrahedra

#: Cell arrays written, named as :mod:`ccdaf.core.carp_export` reads them.
FIBRE_FIELD: str = "fiber"
SHEET_FIELD: str = "sheet"

#: Point arrays holding the four Laplace solutions, by field key.
FIELD_NAMES: Dict[str, str] = {"ab": "ldrb_ab", "epi": "ldrb_epi",
                               "lv": "ldrb_lv", "rv": "ldrb_rv"}

#: Field-data entries: the digest of what the fields were solved on, and
#: the apex nodes they were solved with.
DIGEST_KEY: str = "ldrb_digest"
APEX_KEY: str = "ldrb_apex"
#: The geometry alone, to tell a changed mesh from changed labels.
GEOMETRY_KEY: str = "ldrb_geometry"
#: Set when the fibres were carried from a previous mesh, not solved on
#: this one.
CARRIED_KEY: str = "ldrb_carried"

#: What :func:`reconcile` did.
KEPT: str = "kept"
CARRIED: str = "carried"
REMOVED: str = "removed"

#: The apex patch reaches this many mean edge lengths from the apex node.
APEX_RADIUS_EDGES: float = 1.0

#: Tetrahedra sampled to estimate the mean edge length. The estimate only
#: sizes the apex patch, so a sample is ample.
_EDGE_SAMPLE: int = 20_000


# ---------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------
def surface_sets(mask: np.ndarray) -> Dict[int, np.ndarray]:
    """Node indices of each labelled surface, from the per-node bit mask.

    A node on a seam belongs to both surfaces it joins, which is right for
    the solves: a rim node between base and epicardium is held at 1 in the
    apicobasal problem and at 1 in the epicardial one.
    """
    mask = np.asarray(mask, dtype=np.int64)
    return {key: np.where(mask & bit)[0] for key, bit in MASK_BITS.items()}


def availability(dataset) -> Tuple[bool, str]:
    """Whether fibres can be generated on *dataset*, and if not, why."""
    if dataset is None:
        return False, "Load a tetrahedral volume first."
    if not is_volume(dataset):
        return False, "Fibres need a tetrahedral volume, not a surface."
    if LABEL_MASK_FIELD not in dataset.point_data:
        return False, ("Label the surfaces first "
                       "(Actions > Label ventricular surfaces).")
    sets = surface_sets(dataset.point_data[LABEL_MASK_FIELD])
    missing = [NAMES[k] for k in (BASE, EPI, LV_ENDO, RV_ENDO) if len(sets[k]) == 0]
    if missing:
        return False, f"The surface labels have no {', '.join(missing)}."
    return True, ("Solve the four Laplace problems and write fibre and sheet "
                  "directions per element.")


def mean_edge_length(points: np.ndarray, tets: np.ndarray) -> float:
    """Mean tetrahedron edge length, estimated from a sample."""
    tets = np.asarray(tets, dtype=np.int64)
    if len(tets) > _EDGE_SAMPLE:
        tets = tets[np.random.default_rng(0).choice(len(tets), _EDGE_SAMPLE,
                                                    replace=False)]
    p = np.asarray(points, dtype=float)[tets]
    pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    return float(np.mean([np.linalg.norm(p[:, i] - p[:, j], axis=1).mean()
                          for i, j in pairs]))


@dataclass
class Apex:
    """The nodes held at 0 in the apicobasal problem."""

    #: The node the patch is centred on: found, or picked.
    node: int
    #: Every held node, the centre included.
    nodes: np.ndarray
    #: True when the node was found rather than picked.
    automatic: bool = True

    def describe(self) -> str:
        how = "found automatically" if self.automatic else "picked"
        return f"apex node {self.node} ({how}), {len(self.nodes)} node(s) held"


def base_plane(points: np.ndarray, base_nodes: np.ndarray
               ) -> Tuple[np.ndarray, np.ndarray]:
    """``(origin, unit normal)`` of the plane best fitting the base nodes.

    A least-squares fit rather than the plane the labelling used, which is
    not stored: the base is the thing labelled, so its own nodes say where
    it is, and a cut that is not quite planar is still averaged fairly.
    """
    p = np.asarray(points, dtype=float)[np.asarray(base_nodes, dtype=np.int64)]
    origin = p.mean(axis=0)
    _, _, vt = np.linalg.svd(p - origin, full_matrices=False)
    return origin, vt[-1]


def _patch(points: np.ndarray, epi: np.ndarray, base: np.ndarray,
           centre: int, radius: float) -> np.ndarray:
    """Epicardial nodes within *radius* of *centre*, never a base node.

    A base node is held at 1 in the same problem, so it cannot also be
    held at 0.
    """
    p = np.asarray(points, dtype=float)
    near = epi[np.linalg.norm(p[epi] - p[centre], axis=1) <= radius]
    near = np.setdiff1d(near, base)
    return np.union1d(near, [centre]).astype(np.int64)


def find_apex(points: np.ndarray, tets: np.ndarray, mask: np.ndarray, *,
              radius_edges: float = APEX_RADIUS_EDGES) -> Apex:
    """The epicardial node farthest from the base, and its patch."""
    sets = surface_sets(mask)
    epi = np.setdiff1d(sets[EPI], sets[BASE])
    if len(epi) == 0:
        raise ValueError("The epicardium has no node off the base.")
    origin, normal = base_plane(points, sets[BASE])
    depth = np.abs((np.asarray(points, dtype=float)[epi] - origin) @ normal)
    centre = int(epi[int(np.argmax(depth))])
    radius = radius_edges * mean_edge_length(points, tets)
    return Apex(node=centre, nodes=_patch(points, epi, sets[BASE], centre, radius))


def apex_at(points: np.ndarray, tets: np.ndarray, mask: np.ndarray,
            position, *, radius_edges: float = APEX_RADIUS_EDGES) -> Apex:
    """The apex patch around the epicardial node nearest *position*.

    What a click on the surface becomes: the click lands on a triangle,
    the apex has to be a node, and only an epicardial one will do.
    """
    sets = surface_sets(mask)
    epi = np.setdiff1d(sets[EPI], sets[BASE])
    if len(epi) == 0:
        raise ValueError("The epicardium has no node off the base.")
    p = np.asarray(points, dtype=float)
    _, i = cKDTree(p[epi]).query(np.asarray(position, dtype=float).reshape(3))
    centre = int(epi[int(i)])
    radius = radius_edges * mean_edge_length(points, tets)
    return Apex(node=centre, nodes=_patch(points, epi, sets[BASE], centre, radius),
                automatic=False)


def transmural_sets(sets: Dict[int, np.ndarray]
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(epi, lv, rv, released)``: the held sets of the transmural solves.

    A node on two of these surfaces is where the wall has no thickness: a
    pinch point, one node touching the epicardium and an endocardium. The
    two solves would hold it at 1 and at 0, which no solve can do, so it is
    held by none of them and takes the value its neighbours give it. The
    three fields still sum to 1, because every node that *is* held has
    values summing to 1. Base seams are not affected: the base is not a
    transmural surface.
    """
    epi, lv, rv = sets[EPI], sets[LV_ENDO], sets[RV_ENDO]
    count = np.zeros(max((int(a.max()) + 1 for a in (epi, lv, rv) if len(a)),
                         default=0), dtype=np.int64)
    for nodes in (epi, lv, rv):
        count[nodes] += 1
    released = np.where(count > 1)[0]
    keep = [np.setdiff1d(nodes, released) for nodes in (epi, lv, rv)]
    return keep[0], keep[1], keep[2], released


# ---------------------------------------------------------------------
# Digest and reuse
# ---------------------------------------------------------------------
def _hash(*arrays) -> np.ndarray:
    h = hashlib.sha1()
    for a in arrays:
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    # A small integer array rather than a string: field data of that kind
    # survives a legacy VTK round trip everywhere.
    return np.frombuffer(h.digest(), dtype=np.uint8).astype(np.int32)


def digest(points: np.ndarray, tets: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """A fingerprint of the geometry and labels, as 20 bytes."""
    return _hash(np.ascontiguousarray(points, dtype=np.float64),
                 np.ascontiguousarray(tets, dtype=np.int64),
                 np.ascontiguousarray(mask, dtype=np.int64))


def geometry_digest(points: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """A fingerprint of the geometry alone."""
    return _hash(np.ascontiguousarray(points, dtype=np.float64),
                 np.ascontiguousarray(tets, dtype=np.int64))


def _grid_digest(grid) -> Optional[np.ndarray]:
    if LABEL_MASK_FIELD not in grid.point_data or not is_volume(grid):
        return None
    return digest(grid.points, tetrahedra(grid), grid.point_data[LABEL_MASK_FIELD])


def _grid_geometry(grid) -> Optional[np.ndarray]:
    if not is_volume(grid):
        return None
    return geometry_digest(grid.points, tetrahedra(grid))


def _stored(grid, key: str) -> Optional[np.ndarray]:
    if key not in grid.field_data:
        return None
    return np.asarray(grid.field_data[key]).ravel()


def written_names() -> Tuple[str, ...]:
    """Every point and cell array this module writes."""
    return (FIBRE_FIELD, SHEET_FIELD, *FIELD_NAMES.values())


def _forget(grid) -> None:
    for key in (DIGEST_KEY, APEX_KEY, GEOMETRY_KEY, CARRIED_KEY):
        if key in grid.field_data:
            del grid.field_data[key]


def drop(grid) -> None:
    """Remove everything this module wrote to *grid*."""
    for name in (FIBRE_FIELD, SHEET_FIELD):
        if name in grid.cell_data:
            del grid.cell_data[name]
    for name in FIELD_NAMES.values():
        if name in grid.point_data:
            del grid.point_data[name]
    _forget(grid)


def stamp(grid) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """``(digest, geometry digest)`` *grid* carries, if this module wrote
    to it; what :func:`reconcile` needs to know about the grid replaced."""
    full, geometry = _stored(grid, DIGEST_KEY), _stored(grid, GEOMETRY_KEY)
    if full is None or geometry is None:
        return None
    return full, geometry


def reconcile(grid, previous: Optional[Tuple[np.ndarray, np.ndarray]] = None
              ) -> Optional[str]:
    """Bring this module's results on *grid* in line with the mesh.

    *previous* is the :func:`stamp` of the grid being replaced, for an
    operation that builds a new grid without carrying field data across
    (a remesh or a clean keeps arrays but not the stamp). Returns
    :data:`KEPT`, :data:`CARRIED` or :data:`REMOVED`, or ``None`` when
    there was nothing of this module's to judge.
    """
    stored = stamp(grid) or previous
    if stored is None:
        return None
    if not any(n in grid.cell_data or n in grid.point_data for n in written_names()):
        _forget(grid)
        return None
    full, geometry = stored
    current = _grid_digest(grid)
    if current is not None and np.array_equal(current, full):
        return KEPT
    here = _grid_geometry(grid)
    if here is None or not np.array_equal(here, geometry):
        _forget(grid)
        grid.field_data[CARRIED_KEY] = np.array([1], dtype=np.int32)
        return CARRIED
    drop(grid)
    return REMOVED


#: What :func:`provenance` can say about a mesh's fibres.
NO_FIBRES: str = "none"
GENERATED: str = "generated"
FOREIGN: str = "foreign"


def provenance(grid) -> str:
    """Where *grid*'s fibres came from: :data:`GENERATED` here for this
    mesh and labels, :data:`CARRIED` from a previous mesh, :data:`FOREIGN`
    (arrived with the file), or :data:`NO_FIBRES`."""
    if FIBRE_FIELD not in grid.cell_data:
        return NO_FIBRES
    if CARRIED_KEY in grid.field_data:
        return CARRIED
    full = _stored(grid, DIGEST_KEY)
    if full is not None:
        current = _grid_digest(grid)
        if current is not None and np.array_equal(current, full):
            return GENERATED
    return FOREIGN


def describe_provenance(grid) -> str:
    """One sentence on where the mesh's fibres came from, or ``""``."""
    return {
        NO_FIBRES: "",
        GENERATED: "The fibres on this mesh were generated here, on this mesh "
                   "and these labels.",
        CARRIED: "The fibres on this mesh were carried across from a previous "
                 "mesh by interpolation, not solved on this one. Generating "
                 "replaces them.",
        FOREIGN: "This mesh already carries fibres from elsewhere. Generating "
                 "replaces them.",
    }[provenance(grid)]


def cached_fields(grid, apex: Apex) -> Optional[Dict[str, np.ndarray]]:
    """The stored Laplace fields, if they were solved on this mesh, these
    labels and this apex; otherwise ``None``."""
    stored = _stored(grid, DIGEST_KEY)
    held = _stored(grid, APEX_KEY)
    if stored is None or held is None:
        return None
    if not all(name in grid.point_data for name in FIELD_NAMES.values()):
        return None
    current = _grid_digest(grid)
    if current is None or not np.array_equal(current, stored):
        return None
    if not np.array_equal(np.sort(held), np.sort(apex.nodes)):
        return None
    return {k: np.asarray(grid.point_data[name], dtype=float)
            for k, name in FIELD_NAMES.items()}


# ---------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------
@dataclass
class FibreRun:
    """Everything one run produced, and what it did to get there."""

    fields: Dict[str, np.ndarray]
    fibres: Fibres
    apex: Apex
    angles: FibreAngles
    digest: np.ndarray
    reports: Dict[str, SolveReport] = field(default_factory=dict)
    reused: bool = False
    #: Pinch nodes on two transmural surfaces, held by no solve.
    released: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    seconds: float = 0.0

    def conservation_error(self) -> float:
        """Largest ``|phi_epi + phi_lv + phi_rv - 1|``; Bayer et al. eq. 5."""
        f = self.fields
        return float(np.max(np.abs(f["epi"] + f["lv"] + f["rv"] - 1.0)))

    def summary(self) -> str:
        n = len(self.fibres.fibre)
        repaired = int(np.sum(self.fibres.repaired)) if self.fibres.repaired is not None else 0
        return (f"Fibres for {n:,} elements in {self.seconds:.0f} s"
                + (" (Laplace fields reused)" if self.reused else "")
                + f"; {repaired} rebuilt from neighbours.")

    def details(self) -> str:
        a = self.angles
        lines = [self.summary(), "",
                 f"Angles: alpha endo {a.alpha_endo:g}, alpha epi {a.alpha_epi:g}, "
                 f"beta endo {a.beta_endo:g}, beta epi {a.beta_epi:g} (degrees).",
                 f"Apex: {self.apex.describe()}."]
        if self.reused:
            lines.append("Laplace fields: reused, solved earlier on this mesh, "
                         "these labels and this apex.")
        else:
            lines.append("Laplace solves (conjugate gradients):")
            for key, rep in self.reports.items():
                state = "converged" if rep.converged else "NOT converged"
                lines.append(f"  {key:4s} {rep.iterations:5d} iterations, "
                             f"residual {rep.relative_residual:.1e}, {state}")
        if len(self.released):
            lines.append(f"Released {len(self.released)} node(s) lying on the "
                         f"epicardium and an endocardium at once (the wall has "
                         f"no thickness there): held by no transmural solve.")
        lines.append(f"Transmural fields sum to 1 within "
                     f"{self.conservation_error():.1e} (should be near the "
                     f"solver tolerance).")
        f = self.fibres
        repaired = int(np.sum(f.repaired)) if f.repaired is not None else 0
        unresolved = int(np.sum(f.unresolved)) if f.unresolved is not None else 0
        lines.append(f"Elements whose frame was undefined and was rebuilt from "
                     f"their neighbours: {repaired}.")
        if unresolved:
            lines.append(f"Elements still undefined after that: {unresolved}; "
                         f"their directions are not meaningful.")
        if not all(r.converged for r in self.reports.values()):
            lines.append("\nA solve did not converge: treat these fibres with care.")
        return "\n".join(lines)


def generate(points: np.ndarray, tets: np.ndarray, mask: np.ndarray,
             apex: Apex, angles: Optional[FibreAngles] = None, *,
             cached: Optional[Dict[str, np.ndarray]] = None,
             rtol: float = DEFAULT_RTOL,
             on_status: Optional[Callable[[str], None]] = None) -> FibreRun:
    """Solve (or reuse) the four fields, then build the fibres.

    Pure numerics on arrays: nothing here touches a VTK object, so it runs
    safely off the GUI thread.
    """
    say = on_status or (lambda _m: None)
    angles = angles or FibreAngles()
    points = np.asarray(points, dtype=float)
    tets = np.asarray(tets, dtype=np.int64)
    start = time.time()
    sets = surface_sets(mask)
    reports: Dict[str, SolveReport] = {}

    if cached is None:
        say("Assembling the stiffness matrix…")
        k = stiffness_matrix(points, tets)
        epi, lv, rv, released = transmural_sets(sets)
        problems = (
            ("epi", epi, np.concatenate([lv, rv]), "epicardium to endocardia"),
            ("lv", lv, np.concatenate([epi, rv]), "LV endocardium"),
            ("rv", rv, np.concatenate([epi, lv]), "RV endocardium"),
            ("ab", sets[BASE], apex.nodes, "apex to base"),
        )
        fields = {}
        for i, (key, ones, zeros, what) in enumerate(problems, start=1):
            say(f"Laplace solve {i} of 4: {what}…")
            fields[key], reports[key] = laplace(points, tets, ones, zeros,
                                                matrix=k, rtol=rtol)
        del k
    else:
        fields = cached
        released = transmural_sets(sets)[3]

    say("Building fibre and sheet directions…")
    result = fibres(points, tets, fields, angles)
    return FibreRun(fields=fields, fibres=result, apex=apex, angles=angles,
                    digest=digest(points, tets, mask), reports=reports,
                    reused=cached is not None, released=released,
                    seconds=time.time() - start)


def write(grid, run: FibreRun) -> None:
    """Put a run's results on *grid*, replacing any fibres it had."""
    grid.cell_data[FIBRE_FIELD] = run.fibres.fibre
    grid.cell_data[SHEET_FIELD] = run.fibres.sheet
    for key, name in FIELD_NAMES.items():
        grid.point_data[name] = np.asarray(run.fields[key], dtype=float)
    _forget(grid)
    grid.field_data[DIGEST_KEY] = run.digest
    grid.field_data[GEOMETRY_KEY] = geometry_digest(grid.points, tetrahedra(grid))
    grid.field_data[APEX_KEY] = np.asarray(run.apex.nodes, dtype=np.int64)


#: Segments are this share of the spacing between sampled elements long,
#: so neighbouring ones rarely overlap.
_SEGMENT_SHARE: float = 0.8


def segments(grid, name: str = FIBRE_FIELD, count: int = 20_000,
             seed: int = 0) -> pv.PolyData:
    """Line segments along *grid*'s per-element directions, for display.

    One segment through the centre of each of *count* randomly sampled
    elements. Lines rather than arrows because a fibre has no sign. Their
    length follows the spacing of the sample, ``(V / count) ** (1/3)`` for
    myocardial volume ``V``, so a denser sample draws shorter segments
    instead of a thicket.
    """
    grid = pv.wrap(grid)
    tets = tetrahedra(grid)
    directions = np.asarray(grid.cell_data[name], dtype=float)
    n = len(tets)
    k = int(min(max(count, 1), n))
    pick = np.sort(np.random.default_rng(seed).choice(n, k, replace=False))
    p = np.asarray(grid.points, dtype=float)[tets[pick]]
    centre = p.mean(axis=1)
    volume = np.abs(np.linalg.det(p[:, 1:] - p[:, :1])) / 6.0
    spacing = (float(volume.mean()) * n / k) ** (1.0 / 3.0)
    u = directions[pick]
    u = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-300)
    half = 0.5 * _SEGMENT_SHARE * spacing * u
    pts = np.empty((2 * k, 3))
    pts[0::2] = centre - half
    pts[1::2] = centre + half
    ids = np.arange(k, dtype=np.int64)
    lines = np.column_stack([np.full(k, 2), 2 * ids, 2 * ids + 1]).ravel()
    return pv.PolyData(pts, lines=lines)


def inputs(grid) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(points, tets, mask)`` of a labelled volume, as plain arrays."""
    grid = pv.wrap(grid)
    return (np.asarray(grid.points, dtype=float), tetrahedra(grid),
            np.asarray(grid.point_data[LABEL_MASK_FIELD], dtype=np.int64))


__all__ = [
    "FIBRE_FIELD", "SHEET_FIELD", "FIELD_NAMES", "DIGEST_KEY", "APEX_KEY",
    "GEOMETRY_KEY", "CARRIED_KEY", "KEPT", "CARRIED", "REMOVED", "NO_FIBRES",
    "GENERATED", "FOREIGN", "stamp", "reconcile", "provenance",
    "describe_provenance", "geometry_digest",
    "APEX_RADIUS_EDGES", "Apex", "FibreRun", "availability", "surface_sets",
    "transmural_sets",
    "mean_edge_length", "base_plane", "find_apex", "apex_at", "digest",
    "drop", "cached_fields", "generate",
    "write", "inputs", "written_names", "segments",
]
