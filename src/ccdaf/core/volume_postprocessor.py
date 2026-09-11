"""
volume_postprocessor
====================
Adapt a tetrahedral volume: change element size, improve quality, keep
the anatomy.

The surface post-processor rebuilds a surface and hands it back. That is
the wrong shape for a volume — the tetrahedra are the mesh, not a
by-product of it — so this is a separate module rather than a mode of
that one. It drives MMG3D, which adapts the mesh it is given rather than
meshing from scratch, so an element in the result is a subdivision or a
merge of elements that were there before. That is what makes carrying a
fibre field across defensible at all.

Indices
-------
``mmgpy`` takes and returns **0-based** connectivity and converts to
MMG's own 1-based storage itself. Passing 1-based indices does not raise
where the mistake is made: MMG accepts the array and then reports
``tetrahedron N has volume null`` from deep inside its analysis phase,
which reads as a complaint about the mesh rather than about the caller.
That cost a feature once — this module was first built around temporary
files to avoid an in-memory API that had never actually failed — so the
convention is stated here and asserted in :func:`_from_mmg` rather than
trusted.

Labels ride as references
-------------------------
MMG carries an integer *reference* per element through the adaptation, so
``elemTag`` goes across as one and comes back exact. That is better than
re-deriving it afterwards by proximity, which is what the other fields
have to settle for. References are positive integers and a labelling is
not required to be, so tags are encoded to a dense ``1..n`` range on the
way in and decoded on the way out; a labelling that starts at 0, or skips
values, survives unchanged.

Everything else is ours to carry: MMG returns geometry and references,
not fibres and not point fields. Every call here therefore ends in
:func:`field_transfer.transfer_volume_fields`, and a caller cannot forget
it because it is not a separate step.

Boundary
--------
``freeze_boundary`` (MMG's ``nosurf``) is the default and is exact: on a
66,819-vertex ventricle every one of the 38,823 boundary vertices came
back at machine zero and the surface area was unchanged. Letting the
surface adapt is a genuinely different operation — the same mesh moved by
up to 1.5 mm, half a millimetre on average, and lost a third of its
boundary vertices — so it is opt-in, and the caller is expected to say so.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
import pyvista as pv

from ccdaf.core.field_transfer import transfer_volume_fields
from ccdaf.core.volume_mesh import (
    TETRA, boundary_surface, orient_positive, tetrahedra, validate_tetrahedral,
)

#: The cell array carried across as MMG element references.
TAG_FIELD = "elemTag"

#: Boundary tolerance, as a fraction of the element size, used when the
#: caller adapts the boundary without naming one.
#:
#: Not left to MMG. Its own default is 0.01 *mesh units* — a sensible
#: figure for unit-scale geometry and absurd for a heart in millimetres,
#: where it demands the surface be approximated to 10 um. Measured on a
#: 290,000-element ventricle at a 1.5 mm target, the cost roughly doubles
#: each time the tolerance halves: 23 s at 0.3, 27 s at 0.1, 56 s at
#: 0.05. MMG's default is five times tighter again and does not finish in
#: any usable time — it simply looks like the application has hung.
#:
#: A fifth of the element size keeps the surface well inside the
#: discretisation it is being meshed at, and is what every validated run
#: used.
HAUSDORFF_FRACTION: float = 0.2


@dataclass
class RemeshOptions:
    """What to ask MMG3D for.

    All lengths are in mesh units. Zero means "do not pass this to MMG",
    which leaves its own default — deliberately not a number of ours,
    because a guess baked in here would be wrong on a mesh in centimetres.

    ``target_edge`` (MMG's ``hsiz``) asks for one uniform size and is the
    simple case. ``min_edge`` / ``max_edge`` (``hmin`` / ``hmax``) give a
    band instead; MMG rejects being given both, so :meth:`validate` says
    so before the call rather than letting it fail deep inside.
    """

    target_edge: float = 0.0
    min_edge: float = 0.0
    max_edge: float = 0.0
    #: Hausdorff distance: how far the adapted boundary may stray from
    #: the original. Ignored while the boundary is frozen.
    hausdorff: float = 0.0
    #: Gradation: the largest ratio allowed between two adjacent edges'
    #: lengths. It limits how fast size may change, not what the size is,
    #: so it only bites where the size varies — which on this path means
    #: while the boundary is adapting and MMG is deriving a size from
    #: surface curvature. With a frozen boundary it is measurably inert.
    gradation: float = 0.0
    #: Freeze the boundary, adapting only the interior. See the module
    #: docstring for why this is the default.
    freeze_boundary: bool = True

    def validate(self) -> None:
        if any(v < 0.0 for v in (self.target_edge, self.min_edge,
                                 self.max_edge, self.hausdorff,
                                 self.gradation)):
            raise ValueError("remesh sizes must not be negative")
        if self.target_edge > 0.0 and (self.min_edge > 0.0
                                       or self.max_edge > 0.0):
            raise ValueError(
                "give either a target edge length or a min/max band, "
                "not both")
        if (self.min_edge > 0.0 and self.max_edge > 0.0
                and self.min_edge > self.max_edge):
            raise ValueError("min edge length exceeds max")

    def as_mmg_options(self) -> dict:
        """The subset MMG is actually told about."""
        options = {"verbose": -1}
        for key, value in (("hsiz", self.target_edge),
                           ("hmin", self.min_edge),
                           ("hmax", self.max_edge),
                           ("hausd", self.hausdorff),
                           ("hgrad", self.gradation)):
            if value > 0.0:
                options[key] = float(value)
        if self.freeze_boundary:
            options["nosurf"] = True
        return options


# ---------------------------------------------------------------------
# References
# ---------------------------------------------------------------------
def encode_tags(tags: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Tags as dense ``1..n`` references, plus the values to decode with.

    MMG's references are positive integers; a labelling need not be. A
    mesh tagged 0 and 7 is perfectly ordinary and would lose its 0
    outright, so the values are encoded rather than passed through.
    """
    values = np.unique(np.asarray(tags))
    codes = (np.searchsorted(values, tags) + 1).astype(np.int32)
    return codes, values


def decode_tags(refs: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Invert :func:`encode_tags`, keeping the original values and dtype.

    A reference outside the encoded range would mean MMG invented one,
    which would silently mislabel elements of a mesh someone is about to
    simulate. Raised rather than clamped.
    """
    refs = np.asarray(refs, dtype=np.int64)
    if refs.size and (refs.min() < 1 or refs.max() > len(values)):
        raise RuntimeError(
            f"MMG returned element references outside the range it was "
            f"given (1..{len(values)}); the labelling cannot be restored.")
    return values[refs - 1]


# ---------------------------------------------------------------------
# To and from MMG
# ---------------------------------------------------------------------
def _mean_edge(grid) -> float:
    """Mean tetrahedron edge length of *grid*, in mesh units."""
    tets = tetrahedra(grid)
    if tets.size == 0:
        return 0.0
    points = np.asarray(grid.points, dtype=float)
    pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    lengths = np.concatenate([
        np.linalg.norm(points[tets[:, j]] - points[tets[:, i]], axis=1)
        for i, j in pairs])
    return float(lengths.mean())


def _to_mmg(grid):
    """Build an ``MmgMesh3D`` from *grid*; return it and the tag values.

    The boundary triangles are supplied explicitly. MMG needs them to
    know which faces are the surface, which is what ``nosurf`` freezes;
    without them a frozen boundary has nothing to freeze.

    Orientation is normalised here rather than at load, so what is read
    from disk is what is written back. MMG refuses a tetrahedron whose
    signed volume is not positive.
    """
    import mmgpy

    validate_tetrahedral(grid)
    points = np.asarray(grid.points, dtype=float)
    tets, _flipped = orient_positive(points, tetrahedra(grid))

    surface = boundary_surface(grid)
    origin_ids = np.asarray(surface.point_data["vtkOriginalPointIds"],
                            dtype=np.int64)
    faces = np.asarray(surface.faces).reshape(-1, 4)[:, 1:]
    triangles = origin_ids[faces].astype(np.int32)

    if TAG_FIELD in grid.cell_data:
        refs, values = encode_tags(np.asarray(grid.cell_data[TAG_FIELD]))
    else:
        refs = np.ones(len(tets), dtype=np.int32)
        values = None

    mesh = mmgpy.MmgMesh3D()
    mesh.set_mesh_size(len(points), len(tets), 0, len(triangles), 0, 0)
    mesh.set_vertices(points, np.zeros(len(points), dtype=np.int32))
    mesh.set_tetrahedra(tets.astype(np.int32), refs)
    mesh.set_triangles(triangles, np.ones(len(triangles), dtype=np.int32))
    return mesh, values


def _from_mmg(mesh, values: Optional[np.ndarray]) -> pv.UnstructuredGrid:
    """Read an ``MmgMesh3D`` back as a grid, restoring the tags."""
    points = np.asarray(mesh.get_vertices_with_refs()[0], dtype=float)
    tets, refs = mesh.get_tetrahedra_with_refs()
    tets = np.asarray(tets, dtype=np.int64)

    # The 0-based convention, asserted rather than trusted: a silent
    # off-by-one here produces a mesh that looks plausible and encloses
    # the wrong volume. See the module docstring.
    if tets.size and (tets.min() < 0 or tets.max() >= len(points)):
        raise RuntimeError(
            "MMG returned connectivity outside the vertex range — the "
            "0-based index convention no longer holds.")

    out = pv.UnstructuredGrid({TETRA: tets}, points)
    if values is not None:
        out.cell_data[TAG_FIELD] = decode_tags(np.asarray(refs), values)
    return out


def remesh(grid,
           options: Optional[RemeshOptions] = None,
           on_status: Optional[Callable[[str], None]] = None):
    """Adapt ``grid`` with MMG3D and return the result, fields and all.

    ``grid`` is not modified.

    Raises ``RuntimeError`` if MMG cannot remesh the mesh, and
    ``ValueError`` for options it would reject.
    """
    options = options or RemeshOptions()
    options.validate()

    source = pv.wrap(grid)
    if on_status is not None:
        on_status(f"Remeshing {source.n_cells} tetrahedra…")

    mesh, values = _to_mmg(source)
    sent = options.as_mmg_options()
    if not options.freeze_boundary and "hausd" not in sent:
        # Never left to MMG — see HAUSDORFF_FRACTION.
        size = options.target_edge or options.max_edge or _mean_edge(source)
        if size > 0.0:
            sent["hausd"] = HAUSDORFF_FRACTION * size
            if on_status is not None:
                on_status(f"Boundary tolerance not set; using "
                          f"{sent['hausd']:.3g} (a fifth of the element "
                          f"size).")
    try:
        report = mesh.remesh(**sent)
    except Exception as exc:
        raise RuntimeError(
            f"MMG3D could not remesh this volume: {exc}") from exc

    result = _from_mmg(mesh, values)
    if result.n_cells == 0:
        raise RuntimeError("MMG3D returned a mesh with no tetrahedra.")

    # elemTag came back exactly, as references. Everything else has to be
    # carried, and re-deriving the labels by proximity would only make
    # them worse.
    transfer_volume_fields(source, result, exclude={TAG_FIELD},
                           on_status=on_status)

    if on_status is not None:
        quality = ""
        if isinstance(report, dict) and "quality_mean_after" in report:
            quality = (f", mean quality "
                       f"{report['quality_mean_before']:.3f} → "
                       f"{report['quality_mean_after']:.3f}")
        on_status(f"Remeshed {source.n_cells} → {result.n_cells} "
                  f"tetrahedra{quality}.")
    return result


__all__ = ["RemeshOptions", "remesh", "encode_tags", "decode_tags",
           "TAG_FIELD", "HAUSDORFF_FRACTION"]
