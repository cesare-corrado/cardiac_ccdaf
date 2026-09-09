"""
MeshLoader
==========

Wrapper around the ``vtkfunctions`` I/O routines. Holds whichever kind of
mesh the file contained and guarantees the presence of the ``elemTag``
cell-data scalar (initialized to 1 everywhere, per spec).

Surface or volume
-----------------
``kind`` says which, decided by the cells rather than by the file's
dataset type — a triangular surface exported as an unstructured grid is a
surface. For a volume:

* ``grid`` is the tetrahedral mesh, exactly as read;
* ``mesh`` is its boundary surface, derived.

For a surface ``grid`` is ``None`` and ``mesh`` is the mesh, which is
what every caller has always read.

The volume is the master and the surface is a view of it. Anything that
changes geometry changes the volume and re-derives the surface, never the
other way round: re-deriving is cheap and always correct, while pushing a
changed surface back into a volume is not defined. What is read from disk
is what is written back, unless something asked for a change — the
loader normalises nothing on the way in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Union

import numpy as np
import pyvista as pv

from ccdaf.io.vtkfunctions import read_dataset, writevtk
from ccdaf.core.eam_loader import CARTO_NODATA
from ccdaf.core.volume_mesh import (
    SURFACE, VOLUME, boundary_surface, kind_of, validate_tetrahedral,
)


BODY_LABEL: int = 1
UNASSIGNED: int = -1

# Fields written when the caller names none. ``Normals`` joins ``elemTag``
# because the downstream project format expects both; a mesh that carries
# neither still gets ``Normals``, computed from its geometry on the way out.
DEFAULT_SAVE_FIELDS: tuple = ("elemTag", "Normals")

#: Arrays that are bookkeeping rather than data. They must never be
#: offered as a field to colour by, written to a file, or carried through
#: a filter onto a mesh where their indices no longer mean anything.
#:
#: ``render_idx`` is stamped on by the renderer for picking. The two
#: ``vtkOriginal*`` arrays come from taking a volume's boundary and are
#: the map from a boundary triangle back to its parent tetrahedron —
#: real information, and exactly as stale as the mesh they indexed the
#: moment anything re-tessellates it.
INTERNAL_ARRAYS: frozenset = frozenset({
    "render_idx", "vtkOriginalPointIds", "vtkOriginalCellIds",
})


def compute_normals(mesh: pv.PolyData) -> pv.PolyData:
    """Return ``mesh`` with a ``Normals`` cell array computed from geometry.

    Cell normals only: that is what the project format carries, and what a
    surface read back from disk here already has. Vertices are not split, so
    point correspondence — and with it every other field — survives.
    """
    return mesh.compute_normals(
        cell_normals=True,
        point_normals=False,
        split_vertices=False,
        consistent_normals=True,
        auto_orient_normals=True,
        inplace=False,
    )

# --------------------------------------------------------------------------
# No-data across ASCII VTK
# --------------------------------------------------------------------------
# VTK's legacy *ASCII* reader cannot parse a ``nan`` token: the first one trips
# the stream's failbit, every value after it misreads, the tuple count drifts,
# and a leftover ``nan`` is taken as the next array's *name* — so every field
# past the first no-data field is lost and a phantom ``nan`` field appears.
# ParaView uses that same reader, so the file opens nowhere. (Binary carries
# NaN as raw IEEE-754 bits and is unaffected; XML .vtp parses ``nan`` fine.)
#
# So no-data is encoded on the way to ASCII as the Carto sentinel and folded
# back to NaN on the way in. The sentinel is purely an ASCII-transport detail:
# in memory, and in binary, no-data is always NaN. Real Carto data is
# ``|v| < CARTO_NODATA`` by the format's own convention (the loader masks
# ``|v| >= CARTO_NODATA`` to NaN), which is what makes the fold unambiguous;
# elemTag labels and unit Normals never approach it, so they are untouched.


def nodata_to_sentinel(mesh: pv.PolyData) -> None:
    """In place: set non-finite float field values to ``CARTO_NODATA``.

    For writing ASCII VTK only — binary carries NaN natively and must be left
    alone. Mutates ``mesh``; callers writing a live mesh pass a copy.
    """
    for attr in (mesh.point_data, mesh.cell_data):
        for name in list(attr.keys()):
            a = np.asarray(attr[name])
            if np.issubdtype(a.dtype, np.floating) and not np.isfinite(a).all():
                a = a.copy()
                a[~np.isfinite(a)] = CARTO_NODATA
                attr[name] = a


def sentinel_to_nodata(mesh: pv.PolyData) -> None:
    """In place: fold the Carto no-data sentinel back to NaN.

    The inverse of :func:`nodata_to_sentinel` for a mesh read from an ASCII
    VTK. Applied only to ASCII legacy ``.vtk`` (see :func:`_is_ascii_legacy_vtk`);
    binary and XML files carry NaN natively and are read untouched.
    """
    for attr in (mesh.point_data, mesh.cell_data):
        for name in list(attr.keys()):
            a = np.asarray(attr[name])
            if np.issubdtype(a.dtype, np.floating):
                mask = np.abs(a) >= CARTO_NODATA
                if mask.any():
                    a = a.copy()
                    a[mask] = np.nan
                    attr[name] = a


def _is_ascii_legacy_vtk(filename: Union[str, Path]) -> bool:
    """True only for a legacy ``.vtk`` whose header declares ``ASCII``.

    Line 3 of the legacy header is ``ASCII`` or ``BINARY``. Any other
    extension (``.vtp``, ``.ply``, …) or an unreadable header returns False,
    so the sentinel fold never touches a format that carries NaN natively.
    """
    if not str(filename).lower().endswith(".vtk"):
        return False
    try:
        with open(filename, "rb") as fh:
            line = b""
            for _ in range(3):
                line = fh.readline()
        return line.strip().upper() == b"ASCII"
    except OSError:
        return False


class MeshLoader:
    """Load / save cardiac meshes and manage the ``elemTag`` array."""

    def __init__(self) -> None:
        self.path: Union[str, None] = None
        self.mesh: Union[pv.PolyData, None] = None
        #: The tetrahedral volume, when the file held one; else ``None``.
        self.grid: Union[pv.UnstructuredGrid, None] = None
        self.kind: str = SURFACE

    # ------------------------------------------------------------------
    @property
    def dataset(self):
        """The working mesh itself: the volume when there is one.

        What saving writes and what a geometry-changing operation acts
        on. ``mesh`` stays the thing to render and pick, whichever kind
        this is.
        """
        return self.grid if self.kind == VOLUME else self.mesh

    # ------------------------------------------------------------------
    def load(self, filename: Union[str, Path]):
        """Read a mesh file; return the volume, or the surface if flat.

        A file holding tetrahedra becomes ``grid``, with ``mesh`` its
        boundary; anything else becomes ``mesh`` and leaves ``grid``
        ``None``. Either way ``elemTag`` is present afterwards.
        """
        filename = str(filename)
        dataset = read_dataset(filename)
        if dataset is None:
            raise ValueError(f"no reader for {filename}")

        kind = kind_of(dataset)
        if kind == VOLUME:
            validate_tetrahedral(dataset)
            grid = pv.wrap(dataset)
            self._ensure_elem_tag(grid)
            if _is_ascii_legacy_vtk(filename):
                sentinel_to_nodata(grid)
            mesh = boundary_surface(grid)
            self._validate_triangles(mesh)
            self.grid = grid
        else:
            mesh = pv.wrap(dataset)
            if not isinstance(mesh, pv.PolyData):
                # A grid of 2-D cells: same surface, wrong container.
                mesh = boundary_surface(mesh)
            self._validate_triangles(mesh)
            self._ensure_elem_tag(mesh)
            if _is_ascii_legacy_vtk(filename):
                sentinel_to_nodata(mesh)   # ASCII stores no-data as CARTO_NODATA
            self.grid = None

        self.kind = kind
        self.path = filename
        self.mesh = mesh
        return self.dataset

    # ------------------------------------------------------------------
    def set_surface(self, mesh: pv.PolyData) -> None:
        """Adopt *mesh* as a surface working mesh, dropping any volume."""
        self.mesh = mesh
        self.grid = None
        self.kind = SURFACE

    def set_volume(self, grid: "pv.UnstructuredGrid") -> None:
        """Adopt *grid* as the working volume and re-derive the surface."""
        validate_tetrahedral(grid)
        self._ensure_elem_tag(grid)
        self.grid = grid
        self.mesh = boundary_surface(grid)
        self.kind = VOLUME

    # ------------------------------------------------------------------
    def save(self, filename: Union[str, Path],
             fields: Union[Iterable[str], None] = None,
             binary: bool = False) -> None:
        """Write the current mesh via ``writevtk``, keeping only ``fields``.

        ``fields`` names the point / cell arrays to write, by name — the
        association is looked up on the mesh, so callers need not know
        whether a field lives on the points (an EAM mapping's Carto fields)
        or the cells (``elemTag``). Anything unnamed is dropped.

        ``None`` keeps :data:`DEFAULT_SAVE_FIELDS`, which is what the
        downstream project format expects; note that means an EAM mapping's
        measured fields are dropped unless asked for.

        Asking for ``Normals`` on a mesh that has none — a Carto mapping,
        which arrives as bare geometry — computes them. ``binary`` selects
        the VTK encoding; ASCII is the default the project format is read
        with.
        """
        if self.mesh is None:
            raise RuntimeError("no mesh loaded")
        if self.kind == VOLUME:
            self._save_volume(filename, fields, binary)
            return

        keep = (set(DEFAULT_SAVE_FIELDS) if fields is None
                else {str(f) for f in fields})
        mesh0 = self.mesh.copy(deep=True)
        if ("Normals" in keep and "Normals" not in mesh0.point_data
                and "Normals" not in mesh0.cell_data):
            mesh0 = compute_normals(mesh0)
        mesh0.points = mesh0.points.astype(np.float32)

        for key in list(mesh0.point_data.keys()):
            if key not in keep:
                mesh0.point_data.remove(key)
            else:
                mesh0.point_data[key] = np.asarray(
                    mesh0.point_data[key]).astype(np.float32)

        for key in list(mesh0.cell_data.keys()):
            if key not in keep:
                mesh0.cell_data.remove(key)
            else:
                mesh0.cell_data[key] = np.copy(mesh0.cell_data[key]).astype(np.float32)

        # Re-add elemTag so it ends up last and active, as the downstream
        # reader expects.
        if 'elemTag' in mesh0.cell_data.keys():
            elem = np.asarray(mesh0.cell_data['elemTag'], dtype=np.float32)
            mesh0.cell_data.remove('elemTag')
            mesh0.cell_data['elemTag'] = elem
            mesh0.set_active_scalars('elemTag', preference='cell')

        # ASCII cannot carry NaN; encode no-data as the sentinel. mesh0 is a
        # private copy, so mutate it directly. Binary keeps NaN untouched.
        if not binary:
            nodata_to_sentinel(mesh0)
        writevtk(mesh0, str(filename), binary=binary)

    # ------------------------------------------------------------------
    def _save_volume(self, filename: Union[str, Path],
                     fields: Union[Iterable[str], None],
                     binary: bool) -> None:
        """Write the tetrahedral volume, keeping only ``fields``.

        Deliberately unlike the surface path in two ways. There are no
        ``Normals``: they are a property of a surface, and computing them
        for a volume would write a field that means nothing. And nothing
        is cast to ``float32`` — the surface path does that because the
        downstream project format expects it, whereas a volume has no
        such reader, and casting would quietly turn an integer label or a
        double-precision fibre into something else.

        ``fields`` of ``None`` keeps every array, since the surface
        default (``elemTag`` + ``Normals``) is a surface contract and
        dropping a volume's fibres to honour it would be silent data
        loss.
        """
        grid = self.grid.copy(deep=True)
        if fields is not None:
            keep = {str(f) for f in fields}
            for attr in (grid.point_data, grid.cell_data):
                for name in list(attr.keys()):
                    if name not in keep:
                        attr.remove(name)

        # elemTag last and active, matching the surface path: a reader
        # that takes the active scalars finds the labels either way.
        if "elemTag" in grid.cell_data.keys():
            elem = np.copy(grid.cell_data["elemTag"])
            grid.cell_data.remove("elemTag")
            grid.cell_data["elemTag"] = elem
            grid.set_active_scalars("elemTag", preference="cell")

        if not binary:
            nodata_to_sentinel(grid)
        writevtk(grid, str(filename), binary=binary)

    # ------------------------------------------------------------------
    @staticmethod
    def field_names(mesh: pv.PolyData) -> "list[str]":
        """Every field on ``mesh``, point arrays first then cell arrays.

        Bookkeeping arrays are left out — see :data:`INTERNAL_ARRAYS`.
        """
        return [n for n in (list(mesh.point_data.keys())
                            + list(mesh.cell_data.keys()))
                if n not in INTERNAL_ARRAYS]



    # ------------------------------------------------------------------
    @staticmethod
    def _validate_triangles(mesh: pv.PolyData) -> None:
        faces = np.asarray(mesh.faces)
        if faces.size == 0 or faces.size % 4 != 0 or np.any(faces[::4] != 3):
            raise ValueError("mesh must contain triangles only")

    @staticmethod
    def _ensure_elem_tag(mesh: pv.PolyData) -> None:
        """Create or reset ``elemTag`` to body label if missing."""
        if "elemTag" not in mesh.cell_data:
            mesh.cell_data["elemTag"] = np.full(
                mesh.n_cells, BODY_LABEL, dtype=np.int32
            )


__all__ = ["MeshLoader", "BODY_LABEL", "DEFAULT_SAVE_FIELDS", "INTERNAL_ARRAYS",
           "compute_normals", "nodata_to_sentinel", "sentinel_to_nodata"]
