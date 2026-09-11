"""
vtkfunctions
============
Read and write the mesh formats the project uses.

Two ways to read, because a file may hold either kind of mesh:

* :func:`read_dataset` returns what the file actually contains — a
  ``vtkPolyData`` for a surface, a ``vtkUnstructuredGrid`` for a volume.
* :func:`readvtk` returns a surface whatever the file held, reducing a
  volume to its boundary. This is the older entry point and its contract
  is unchanged.

A volume must be read through the first of those or its tetrahedra are
gone before anything can decide what to do with them. Which of the two a
caller wants is a real choice, not an implementation detail, hence two
names rather than a flag.

:func:`writevtk` picks its writer from the dataset it is handed, so the
same call writes a surface or a volume.
"""
import os

import vtk


#: Legacy-VTK dataset keywords that mean "not a surface".
_UNSTRUCTURED = "UNSTRUCTURED_GRID"


def _legacy_dataset_type(filename: str) -> str:
    """The ``DATASET`` keyword of a legacy ``.vtk`` header, or ``""``.

    Line 4 of the legacy header is ``DATASET <type>``. Read as bytes: the
    file may be binary from line 5 on, and the header itself is ASCII.
    Anything unreadable returns ``""``, which routes to the polydata
    reader exactly as before.
    """
    try:
        with open(filename, "rb") as fh:
            line = b""
            for _ in range(4):
                line = fh.readline()
        return line.strip().split()[-1].decode("ascii")
    except (OSError, IndexError, UnicodeDecodeError):
        return ""


def _configure(reader, filename: str, extension: str):
    if extension == ".g":
        reader.SetGeometryFileName(filename)
    else:
        reader.SetFileName(filename)
    # Not every reader has these; the legacy ones do and need them, or
    # fields past the first are silently dropped.
    for turn_on in ("ReadAllScalarsOn", "ReadAllVectorsOn"):
        if hasattr(reader, turn_on):
            getattr(reader, turn_on)()
    reader.Update()
    return reader.GetOutput()


def read_dataset(filename: str):
    """Read *filename* and return the dataset it holds, unreduced.

    A surface comes back as ``vtkPolyData``; a legacy ``.vtk`` holding an
    unstructured grid comes back as ``vtkUnstructuredGrid``, tetrahedra
    and all. Every other extension is a surface format by definition.

    Returns ``None`` for an extension nothing here reads, matching the
    older behaviour rather than raising: the caller checks.
    """
    _, extension = os.path.splitext(filename)
    extension = extension.lower()

    if extension == ".vtk":
        if _legacy_dataset_type(filename) == _UNSTRUCTURED:
            return _configure(vtk.vtkUnstructuredGridReader(),
                              filename, extension)
        return _configure(vtk.vtkPolyDataReader(), filename, extension)

    reader = {
        ".ply": vtk.vtkPLYReader,
        ".vtp": vtk.vtkXMLPolyDataReader,
        ".obj": vtk.vtkOBJReader,
        ".stl": vtk.vtkSTLReader,
        ".g":   vtk.vtkBYUReader,
    }.get(extension)
    if reader is None:
        return None
    return _configure(reader(), filename, extension)


def surface_of(dataset):
    """The boundary surface of *dataset* as ``vtkPolyData``.

    A ``vtkPolyData`` is returned as it is. Anything else goes through
    ``vtkGeometryFilter``, which for a tetrahedral mesh yields the
    triangles of the boundary, carrying the parent cells' data.
    """
    if dataset is None or dataset.IsA("vtkPolyData"):
        return dataset
    geometry_filter = vtk.vtkGeometryFilter()
    geometry_filter.SetInputData(dataset)
    geometry_filter.Update()
    return geometry_filter.GetOutput()


def readvtk(filename: str) -> vtk.vtkPolyData:
    ''' readvtk(filename : str) -> vtk.vtkPolyData
    reads a vtk file and returns a vtkPolyData object

    A file holding a volume is reduced to its boundary surface. Use
    :func:`read_dataset` to keep the volume.
    '''
    return surface_of(read_dataset(filename))


def writevtk(dataset, filename: str, binary: bool = False):
    """Write *dataset*, choosing the writer from what it is.

    A ``vtkPolyData`` is written by the polydata writer, as it always
    was; an unstructured grid by the unstructured-grid writer, so a
    volume keeps its tetrahedra and every array on them. File version 42
    on VTK 9.2+ for both, because the downstream reader expects the
    legacy layout rather than the 5.1 header.
    """
    if dataset is not None and not dataset.IsA("vtkPolyData"):
        writer = vtk.vtkUnstructuredGridWriter()
    else:
        writer = vtk.vtkPolyDataWriter()
    if ((vtk.VTK_MAJOR_VERSION > 9)
            or (vtk.VTK_MAJOR_VERSION == 9 and vtk.VTK_MINOR_VERSION >= 2)):
        writer.SetFileVersion(42)
    writer.SetInputData(dataset)
    if binary:
        writer.SetFileTypeToBinary()
    else:
        writer.SetFileTypeToASCII()
    writer.SetFileName(filename)
    writer.Write()


__all__ = ["read_dataset", "surface_of", "readvtk", "writevtk"]
