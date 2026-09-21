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
    """Raise unless **every** cell of *dataset* is a tetrahedron.

    A volume of hexahedra or a mixed volume is refused rather than
    partly handled: the remesher takes tetrahedra, and a fibre or a
    label per element means one element shape.

    Every cell, not merely every 3-D cell. Checking only the solid ones
    let a file holding tetrahedra *and* its boundary triangles through
    unexamined, because triangles are 2-D: it loaded as a volume whose
    ``n_cells`` counted 193,293 against 148,155 tetrahedra, so every
    per-element array was misaligned. The tools that ask for one value
    per element raised, which was the good case; the volume cleaner
    instead reported "nothing to clean" and returned a mesh with the
    45,138 triangles quietly dropped. A mixed mesh has no honest
    interpretation here, so it is refused at the door.
    """
    present = distinct_cell_types(dataset)
    bad = present[present != TETRA]
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


def element_connectivity(dataset) -> np.ndarray:
    """The ``(n, 3)`` triangles of a surface, or ``(n, 4)`` tetrahedra of a volume.

    One array whichever kind the working mesh is, for everything written per
    element: a value, a size, a neighbour across a shared facet, a row in an
    exported element file. Anything else raises, because each of those rules
    is written for one element shape.
    """
    if isinstance(dataset, pv.PolyData):
        faces = np.asarray(dataset.faces)
        if (faces.size == 0 or faces.size % 4 or np.any(faces[::4] != 3)
                or faces.size // 4 != dataset.n_cells):
            raise ValueError("the surface must contain triangles only")
        return faces.reshape(-1, 4)[:, 1:].astype(np.int64)
    tets = tetrahedra(dataset)
    if len(tets) == 0 or len(tets) != dataset.n_cells:
        raise ValueError("the volume must contain tetrahedra only")
    return tets


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
    "tetrahedra", "element_connectivity",
    "signed_volumes", "inverted_count", "orient_positive",
    "boundary_surface",
    "FACE_NODES", "EDGE_NODES", "sorted_unique_rows", "unique_rows",
    "face_table", "boundary_faces", "boundary_table", "node_cells",
    "shape_measure",
]


#: The four faces and the six edges of a tetrahedron, as node positions.
#: Shared rather than restated: the cleaner, the repair and the topology
#: report all decompose a tetrahedron the same way, and two orderings
#: that drifted apart would make their face tables disagree.
FACE_NODES = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
EDGE_NODES = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])


#: Node ids above this cannot be packed into a 32-bit key, so the
#: packed path falls back. A mesh with two billion nodes is not one this
#: application will see, but the check costs one pass and the failure it
#: prevents would be silent.
_KEY_LIMIT = np.iinfo(np.int32).max


def _unique_by_key(rows: np.ndarray):
    """``np.unique(axis=0)``, via one packed key per row.

    ``np.unique(axis=0)`` lexsorts the columns one after another, which
    on the face table of a large mesh is the single most expensive thing
    the cleaner does. Packing each row into one opaque key turns that
    into a single sort of a flat array: byte for byte the same answer,
    measured three times faster on a 12-million-row face table.

    The keys are **big-endian** on purpose. The comparison is a
    ``memcmp``, so only with the most significant byte first does it
    order the rows the way comparing the numbers would — little-endian
    keys give the same set of unique rows in a different order, which is
    a bug that hides until something downstream assumes rows sharing a
    first column are adjacent. Negative or oversized ids cannot be
    packed this way at all, so they take the original path.
    """
    rows = np.asarray(rows)
    if rows.size == 0:
        return (rows.reshape(0, rows.shape[1]).astype(np.int64),
                np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
    if rows.min() < 0 or rows.max() > _KEY_LIMIT:
        return np.unique(rows, axis=0, return_inverse=True,
                         return_counts=True)
    width = rows.shape[1]
    packed = np.ascontiguousarray(rows.astype(">i4"))
    keys = packed.view(np.dtype((np.void, 4 * width))).ravel()
    uniq, inverse, counts = np.unique(keys, return_inverse=True,
                                      return_counts=True)
    return (uniq.view(">i4").reshape(-1, width).astype(np.int64),
            inverse, counts)


def sorted_unique_rows(rows: np.ndarray):
    """``np.unique`` over rows sorted within themselves.

    Returns ``(uniq, inverse, counts)``. Sorting each row first is what
    makes a face or an edge undirected: ``(7, 3, 1)`` and ``(1, 3, 7)``
    are the same face and have to land in the same bucket.
    """
    return _unique_by_key(np.sort(rows, axis=1))


def unique_rows(rows: np.ndarray):
    """``np.unique(axis=0)``, for rows whose order carries meaning."""
    return _unique_by_key(rows)


def face_table(tets: np.ndarray, table=None):
    """Unique faces of *tets*, plus the inverse, counts and owners.

    *table* returns an already-built one untouched, so a caller that
    needs the faces twice builds them once. That is not a micro-saving:
    a topology report used to build this table four times over, in the
    report itself, in the components, in the boundary and in the edges.

    Returns ``(uniq, inverse, counts, owner)``, where ``owner[k]`` is the
    tetrahedron that contributed the k-th face of the flattened
    ``4 * n`` list. A face with ``counts == 1`` is on the boundary, with
    ``2`` is interior, and with more the mesh is non-manifold at it.

    Vectorised on purpose: the obvious dictionary loop costs seconds on
    a 290,000-element mesh, and this runs on every clean and report.
    """
    if table is not None:
        return table
    faces = tets[:, FACE_NODES].reshape(-1, 3)
    uniq, inv, counts = sorted_unique_rows(faces)
    owner = np.repeat(np.arange(len(tets), dtype=np.int64), 4)
    return uniq, inv.ravel(), counts, owner


def boundary_faces(tets: np.ndarray, table=None) -> np.ndarray:
    """The faces of *tets* that belong to exactly one tetrahedron."""
    uniq, _inv, counts, _owner = face_table(tets, table)
    return uniq[counts == 1]


def boundary_table(tets: np.ndarray, table=None):
    """The boundary faces, whose tetrahedron each is, and its apex.

    Returns ``(faces, owner, apex)``. The apex is the owning
    tetrahedron's fourth node, the one *not* on the face, and it is what
    says which side of a boundary face the material lies on — the
    question every repair at a non-manifold junction has to answer.

    It is recovered by subtracting the face's node sum from the
    tetrahedron's rather than by searching: the four node ids are
    distinct, so the difference is the missing one.
    """
    uniq, inv, counts, owner = face_table(tets, table)
    order = np.argsort(inv, kind="stable")
    starts = np.searchsorted(inv[order], np.arange(len(counts)))
    single = np.where(counts == 1)[0]
    faces = uniq[single]
    if single.size == 0:
        return faces, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    owners = owner[order[starts[single]]]
    apex = tets[owners].sum(axis=1) - faces.sum(axis=1)
    return faces, owners, apex


def shape_measure(corners: np.ndarray) -> np.ndarray:
    """How distorted each tetrahedron is, from its ``(n, 4, 3)`` corners.

    ``1 - sqrt(2) * 6V / rms_edge^3``: 0 for a regular tetrahedron,
    approaching 1 as it flattens, and 2 for one with no volume left or
    turned inside out. **Lower is better**, so every threshold written
    against it is an upper bound.

    It lives here, with the other things that are true of a tetrahedron
    whoever is asking, because both the repair and the quality pass need
    it and neither may import the other.
    """
    a = corners[:, 1] - corners[:, 0]
    b = corners[:, 2] - corners[:, 0]
    c = corners[:, 3] - corners[:, 0]
    volume = np.einsum("ij,ij->i", np.cross(a, b), c) / 6.0
    squared = np.zeros(len(corners))
    for i, j in EDGE_NODES:
        squared += np.sum((corners[:, j] - corners[:, i]) ** 2, axis=1)
    rms = np.sqrt(squared / 6.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        measure = 1.0 - np.sqrt(2.0) * 6.0 * volume / rms ** 3
    return np.where((volume <= 0.0) | ~np.isfinite(measure), 2.0, measure)


def node_cells(elements: np.ndarray, n_points: int):
    """Which elements meet at each node, as ``(starts, cells)``.

    A compressed adjacency built by one sort, because the repair asks
    "what meets here?" of a few hundred nodes and building a dictionary
    of lists for millions of them to answer that would cost far more
    than the question is worth. ``cells[starts[v]:starts[v + 1]]`` are
    the elements holding node ``v``.

    The width is taken from the array, so this answers the same question
    of the boundary triangles as of the tetrahedra. Asking it of the
    triangles matters: without it, finding the faces at one vertex means
    scanning every face in the mesh, once per vertex repaired.
    """
    flat = elements.ravel()
    order = np.argsort(flat, kind="stable")
    cells = (order // elements.shape[1]).astype(np.int64)
    counts = np.bincount(flat, minlength=n_points)
    starts = np.zeros(n_points + 1, dtype=np.int64)
    np.cumsum(counts, out=starts[1:])
    return starts, cells
