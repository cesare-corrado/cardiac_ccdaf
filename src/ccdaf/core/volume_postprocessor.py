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

Two things about MMG that shape everything here.

**It is driven through files.** ``mmgpy`` also exposes an in-memory API,
and on the meshes this was built against that API fails inside MMG's
analysis phase for every combination of inputs tried — with and without
boundary triangles, one material and several — while the file path works
first time. So the mesh goes out to a temporary directory and comes back.
Nothing is written beside the user's data.

**It returns geometry and nothing else.** No ``elemTag``, no fibres, no
point fields: measured, not assumed. Every call here therefore ends in
:func:`field_transfer.transfer_volume_fields`, and a caller cannot forget
to do it because it is not a separate step.

Boundary
--------
``freeze_boundary`` (MMG's ``nosurf``) is the default and is exact: on a
66,819-vertex ventricle every one of the 38,823 boundary vertices came
back at distance 0.0 and the surface area was unchanged to two decimal
places. Letting the surface adapt is a genuinely different operation —
the same mesh moved by up to 1.5 mm, half a millimetre on average, and
lost a third of its boundary vertices — so it is opt-in, and the caller
is expected to say so.
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pyvista as pv

from ccdaf.core.field_transfer import transfer_volume_fields
from ccdaf.core.mesh_loader import INTERNAL_ARRAYS
from ccdaf.core.volume_mesh import (
    TETRA, orient_positive, tetrahedra, validate_tetrahedral,
)

#: Prefix of the temporary directory each call works in. Named so that
#: anything left behind by a hard crash inside the C library — which no
#: context manager can clean up — is identifiable as ours rather than
#: mysterious.
WORKDIR_PREFIX = "ccdaf-mmg-"


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


def _geometry_only(grid) -> pv.UnstructuredGrid:
    """The tetrahedra and their points, positively oriented, no arrays.

    Two reasons to strip the arrays rather than let them ride. MMG
    discards them anyway, so writing them is time and disk spent on
    nothing — about a third of the file on a mesh carrying fibres. And a
    field that made a partial round trip would be worse than one that
    made none: it would look transferred.

    Orientation is normalised here rather than at load, so what is read
    from disk is what is written back. MMG refuses a tetrahedron whose
    signed volume is not positive.
    """
    validate_tetrahedral(grid)
    points = np.asarray(grid.points, dtype=float)
    tets, _flipped = orient_positive(points, tetrahedra(grid))
    return pv.UnstructuredGrid({TETRA: tets}, points)


def _read_tetrahedra(path: Path) -> pv.UnstructuredGrid:
    """The tetrahedral part of what MMG wrote.

    MMG emits the boundary triangles alongside the tetrahedra in one
    dataset, so the result has mixed cell types and has to be narrowed
    before it is a volume again. Its own ``refs`` array goes with them:
    it is MMG's bookkeeping, not a field of this mesh.
    """
    out = pv.read(path)
    types = np.asarray(out.celltypes, dtype=int)
    if (types != TETRA).any():
        out = out.extract_cells(np.where(types == TETRA)[0])
    for attr in (out.point_data, out.cell_data):
        for name in list(attr.keys()):
            if name in INTERNAL_ARRAYS or name == "refs":
                attr.remove(name)
    return out


def remesh(grid,
           options: Optional[RemeshOptions] = None,
           workdir: Optional[str] = None,
           on_status: Optional[Callable[[str], None]] = None):
    """Adapt ``grid`` with MMG3D and return the result, fields and all.

    ``grid`` is not modified. ``workdir`` overrides the temporary
    directory, which exists so a test can look at exactly what went in
    and came out; leave it ``None`` in the application, where the
    directory and everything in it is removed on the way out of this
    function — including when it raises.

    Raises ``RuntimeError`` if MMG cannot remesh the mesh, and
    ``ValueError`` for options it would reject.
    """
    options = options or RemeshOptions()
    options.validate()

    source = pv.wrap(grid)
    geometry = _geometry_only(source)
    if on_status is not None:
        on_status(f"Remeshing {geometry.n_cells} tetrahedra…")

    if workdir is not None:
        return _remesh_in(Path(workdir), geometry, source, options, on_status)
    with tempfile.TemporaryDirectory(prefix=WORKDIR_PREFIX) as tmp:
        return _remesh_in(Path(tmp), geometry, source, options, on_status)


def _remesh_in(workdir: Path, geometry, source, options: RemeshOptions,
               on_status: Optional[Callable[[str], None]]):
    """One MMG round trip inside an existing directory."""
    in_path = workdir / "in.vtk"
    out_path = workdir / "out.vtk"
    geometry.save(in_path, binary=True)

    try:
        ok = mmg_remesh(in_path, out_path, options.as_mmg_options())
    except Exception as exc:                       # MMG failed internally
        raise RuntimeError(f"MMG3D could not remesh this volume: {exc}") from exc
    if not ok or not out_path.exists():
        raise RuntimeError("MMG3D could not remesh this volume.")

    result = _read_tetrahedra(out_path)
    if result.n_cells == 0:
        raise RuntimeError("MMG3D returned a mesh with no tetrahedra.")

    transfer_volume_fields(source, result, on_status=on_status)
    if on_status is not None:
        on_status(f"Remeshed {source.n_cells} → {result.n_cells} tetrahedra.")
    return result


def mmg_remesh(in_path: Path, out_path: Path, options: dict) -> bool:
    """Call MMG3D on two files. Separated so a test can stand in for it."""
    import mmgpy
    return bool(mmgpy.mmg3d.remesh(str(in_path), str(out_path),
                                   options=options))


__all__ = ["RemeshOptions", "remesh", "mmg_remesh", "WORKDIR_PREFIX"]
