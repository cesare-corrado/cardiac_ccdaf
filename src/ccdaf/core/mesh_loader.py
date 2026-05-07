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
from typing import Union

import numpy as np
import pyvista as pv

from ccdaf.io.vtkfunctions import readvtk, writevtk


BODY_LABEL: int = 1
UNASSIGNED: int = -1

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
    def save(self, filename: Union[str, Path]) -> None:
        """Write the current mesh (with ``elemTag``) via ``writevtk``."""
        if self.mesh is None:
            raise RuntimeError("no mesh loaded")
        
        mesh0 = self.mesh.copy(deep=True)
        mesh0.points = mesh0.points.astype(np.float32)

        for key,arrayvalues in mesh0.point_data.items():
            mesh0.point_data.remove(key)


        for key,arrayvalues in mesh0.cell_data.items():
            if not key in['elemTag','Normals']:
                mesh0.cell_data.remove(key)
            else:
                mesh0.cell_data[key] = np.copy(mesh0.cell_data[key]).astype(np.float32)

        if 'elemTag' in mesh0.cell_data.keys():
            elem = np.asarray(mesh0.cell_data['elemTag'], dtype=np.float32)
            mesh0.cell_data.remove('elemTag')
            mesh0.cell_data['elemTag'] = elem
            mesh0.set_active_scalars('elemTag', preference='cell')

        
        writevtk(mesh0, str(filename))



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


__all__ = ["MeshLoader", "BODY_LABEL"]
