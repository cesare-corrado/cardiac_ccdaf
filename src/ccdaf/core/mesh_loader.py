"""
MeshLoader
==========

Thin wrapper around the provided ``vtkfunctions`` I/O routines. Returns a
``pyvista.PolyData`` view of the input mesh and guarantees the presence of
the ``elemTag`` cell-data scalar (initialized to 1 everywhere, per spec).

The ``vtkfunctions`` module is imported as-is; it is NOT re-implemented and
NOT inspected beyond calling ``readvtk`` / ``writevtk``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Union

import numpy as np
import pyvista as pv

from ccdaf.io.vtkfunctions import readvtk, writevtk


BODY_LABEL: int = 1
UNASSIGNED: int = -1

# Fields written when the caller names none. ``Normals`` joins ``elemTag``
# because the downstream project format expects both; a mesh that carries
# neither still gets ``Normals``, computed from its geometry on the way out.
DEFAULT_SAVE_FIELDS: tuple = ("elemTag", "Normals")


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

class MeshLoader:
    """Load / save atrial surface meshes and manage the ``elemTag`` array."""

    def __init__(self) -> None:
        self.path: Union[str, None] = None
        self.mesh: Union[pv.PolyData, None] = None

    # ------------------------------------------------------------------
    def load(self, filename: Union[str, Path]) -> pv.PolyData:
        """Read a VTK file and return a validated PolyData."""
        filename = str(filename)
        vtk_poly = readvtk(filename)           # mandatory entry point
        mesh = pv.wrap(vtk_poly)

        if not isinstance(mesh, pv.PolyData):
            raise TypeError(f"{filename} is not a surface polydata")

        self._validate_triangles(mesh)
        self._ensure_elem_tag(mesh)

        self.path = filename
        self.mesh = mesh
        return mesh

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

        writevtk(mesh0, str(filename), binary=binary)

    # ------------------------------------------------------------------
    @staticmethod
    def field_names(mesh: pv.PolyData) -> "list[str]":
        """Every field on ``mesh``, point arrays first then cell arrays."""
        return list(mesh.point_data.keys()) + list(mesh.cell_data.keys())



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


__all__ = ["MeshLoader", "BODY_LABEL", "DEFAULT_SAVE_FIELDS", "compute_normals"]
