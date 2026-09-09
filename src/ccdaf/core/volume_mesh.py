"""
volume_mesh
===========
Tetrahedral volumes: how to recognise one, measure it, and take its
boundary.

The working mesh has been a surface for as long as there has been one.
A volume changes that, and the first question every caller asks is
"which am I holding?" — asked here once, of the cells, rather than of
the file. A surface stored as an unstructured grid of triangles is a
surface, and a file's dataset type says nothing about it.

Nothing here imports Qt, MMG or anything but VTK and numpy: it is the
vocabulary the loader, the mesh-info panel and the volumetric
post-processor all speak.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pyvista as pv
import vtk

#: The working mesh is either a surface or a volume; ``MeshLoader.kind``
#: carries one of these.
SURFACE = "surface"
VOLUME = "volume"

#: The one 3-D cell type supported. MMG adapts tetrahedra and nothing
#: else, and every field rule downstream (a fibre per element, a label
#: per element) is written for them, so a hexahedral or mixed volume is
#: refused at the door rather than half-handled.
TETRA = int(vtk.VTK_TETRA)


def distinct_cell_types(dataset) -> np.ndarray:
    """The distinct VTK cell types present in *dataset*.

    Asked of the dataset rather than of a per-cell array: VTK keeps this
    summary itself, so it costs nothing on a mesh of millions of cells,
    and it works for a ``vtkPolyData`` too — which has no ``celltypes``
    array at all, being stored as separate vert/line/poly/strip lists.
    """
    if dataset is None:
        return np.zeros(0, dtype=int)
    types = vtk.vtkCellTypes()
    dataset.GetCellTypes(types)
    return np.array([int(types.GetCellType(i))
                     for i in range(types.GetNumberOfTypes())], dtype=int)


#: Where VTK keeps "what dimension is this cell type". It moved in 9.6:
#: ``vtkCellTypes.GetDimension`` still answers but warns on every call,
#: and will go. Resolved once, at import, so neither the deprecation nor
#: the fallback is paid per lookup.
_DIMENSION_OF = getattr(
    getattr(vtk, "vtkCellTypeUtilities", vtk.vtkCellTypes), "GetDimension")


def cell_dimension(cell_type: int) -> int:
    """The topological dimension of one VTK cell type."""
    return int(_DIMENSION_OF(int(cell_type)))


def solid_cell_types(dataset) -> np.ndarray:
    """The distinct 3-D cell types in *dataset*, if any."""
    types = distinct_cell_types(dataset)
    if types.size == 0:
        return types
    return np.array([t for t in types if cell_dimension(t) == 3], dtype=int)


def is_volume(dataset) -> bool:
    """Whether *dataset* carries any 3-D cell.

    The test is the cells, not the container. A tetrahedral mesh read
    from a legacy ``.vtk`` arrives as an unstructured grid, and so does a
    plain triangular surface someone exported that way; only the first is
    a volume.
    """
    return bool(solid_cell_types(dataset).size)


def kind_of(dataset) -> str:
    """``VOLUME`` when *dataset* has 3-D cells, else ``SURFACE``."""
    return VOLUME if is_volume(dataset) else SURFACE


def validate_tetrahedral(dataset) -> None:
    """Raise unless every 3-D cell of *dataset* is a tetrahedron.

    A volume of hexahedra or a mixed volume is refused rather than
    partly handled: the remesher takes tetrahedra, and a fibre or a
    label per element means one element shape.
    """
    solid = solid_cell_types(dataset)
    bad = solid[solid != TETRA]
    if bad.size:
        names = ", ".join(str(int(b)) for b in bad)
        raise ValueError(
            f"only tetrahedral volumes are supported; this mesh also "
            f"contains VTK cell type(s) {names}")


def tetrahedra(dataset) -> np.ndarray:
    """The ``(n, 4)`` point-index array of *dataset*'s tetrahedra."""
    grid = pv.wrap(dataset)
    cells = grid.cells_dict.get(TETRA)
    if cells is None:
        return np.zeros((0, 4), dtype=np.int64)
    return np.asarray(cells, dtype=np.int64)


def signed_volumes(points: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """Signed volume of every tetrahedron.

    Negative means the four nodes wind the other way. That is not wrong
    in itself — VTK does not care — but the remesher rejects a
    non-positive tetrahedron outright, so the sign has to be known
    before one is handed over.
    """
    if tets.size == 0:
        return np.zeros(0, dtype=float)
    p = np.asarray(points, dtype=float)
    a, b, c, d = p[tets[:, 0]], p[tets[:, 1]], p[tets[:, 2]], p[tets[:, 3]]
    return np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a)) / 6.0


def inverted_count(dataset) -> int:
    """How many tetrahedra have non-positive signed volume.

    Reported rather than silently repaired: a handful means a node
    ordering convention, and a lot means the mesh is damaged. Both are
    worth seeing before a remesh rejects them.
    """
    grid = pv.wrap(dataset)
    tets = tetrahedra(grid)
    if tets.size == 0:
        return 0
    return int((signed_volumes(np.asarray(grid.points), tets) <= 0.0).sum())


def orient_positive(points: np.ndarray,
                    tets: np.ndarray) -> Tuple[np.ndarray, int]:
    """Return *tets* with every tetrahedron positively oriented.

    Swapping two nodes flips the sign and changes nothing else: the same
    four points, the same shape, the same cell data. Applied where a mesh
    is handed to the remesher rather than at load, so what is read from
    disk is what is written back unless something asked for a change.

    Returns the array and how many were flipped.
    """
    tets = np.array(tets, dtype=np.int64, copy=True)
    if tets.size == 0:
        return tets, 0
    negative = signed_volumes(points, tets) < 0.0
    swapped = tets[negative][:, [0, 1, 3, 2]]
    tets[negative] = swapped
    return tets, int(negative.sum())


def boundary_surface(dataset) -> pv.PolyData:
    """The boundary of *dataset* as triangles, carrying the parent data.

    ``vtkGeometryFilter`` gives each boundary triangle the cell data of
    the tetrahedron behind it and each vertex its point data, so the
    surface is a view of the volume rather than a copy that has lost
    everything. The bookkeeping arrays it adds are kept: they are the
    map back to the volume, which is what lets a boundary pick name a
    tetrahedron.
    """
    grid = pv.wrap(dataset)
    surface = grid.extract_surface(
        pass_pointid=True, pass_cellid=True,
        algorithm="dataset_surface") if hasattr(grid, "extract_surface") \
        else pv.wrap(dataset)
    return surface.triangulate()


__all__ = [
    "SURFACE", "VOLUME", "TETRA",
    "distinct_cell_types", "cell_dimension", "solid_cell_types",
    "is_volume", "kind_of", "validate_tetrahedral",
    "tetrahedra", "signed_volumes", "inverted_count", "orient_positive",
    "boundary_surface",
]
