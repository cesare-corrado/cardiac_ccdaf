"""
test_volume_mesh.py
===================
Holding a tetrahedral volume without losing it.

Before this, a ``.vtk`` carrying tetrahedra was reduced to its boundary
the moment it was read, and saving wrote that boundary back. The volume,
the fibres and every point field on the interior were gone, silently and
with no way to tell from the result. The loader now keeps the volume and
derives the surface from it.

The contract:

* what a mesh *is* comes from its cells, not from the file's dataset
  type — a triangular surface exported as an unstructured grid is a
  surface, and treating it as a volume would switch off every tool;
* a volume round-trips through save and load unchanged: same points,
  same tetrahedra, same arrays, same dtypes. Not "close enough": the
  point of keeping it is that it is the same mesh;
* the surface is derived from the volume and carries its data, so
  rendering and picking need no special case;
* a volume of anything but tetrahedra is refused with a message, rather
  than half-handled;
* bookkeeping arrays never reach a field list or a file;
* orientation is measured, not silently repaired — the loader returns
  what the file held;
* ``load`` returns the *working mesh* — the volume, for a volumetric
  file — while ``mesh`` is always the surface. The mesh-side tools take
  a surface, so what they are handed has to come from ``mesh``.

No display, no Qt.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pytest
import pyvista as pv

from ccdaf.core import volume_mesh as vm
from ccdaf.core.mesh_loader import INTERNAL_ARRAYS, MeshLoader
from ccdaf.core.region_tagger import RegionTagger


def _tet_grid() -> pv.UnstructuredGrid:
    """A small tetrahedral block carrying the arrays a real one does."""
    grid = pv.Box(bounds=(0, 4, 0, 4, 0, 4),
                  level=2).triangulate().delaunay_3d()
    grid = grid.extract_cells(np.arange(grid.n_cells))   # drop filter arrays
    for name in list(grid.cell_data.keys()) + list(grid.point_data.keys()):
        if name.startswith("vtk"):
            (grid.cell_data if name in grid.cell_data
             else grid.point_data).remove(name)
    rng = np.random.default_rng(0)
    grid.cell_data["elemTag"] = np.where(
        np.arange(grid.n_cells) % 3 == 0, 2, 1).astype(np.int32)
    grid.cell_data["fiber"] = rng.normal(size=(grid.n_cells, 3))
    grid.point_data["scar_probability"] = np.linspace(0, 1, grid.n_points)
    return grid


# ------------------------------------------------------- what is a volume
def test_a_surface_is_a_surface_however_it_is_stored():
    """The cells decide, not the container.

    A triangular surface saved as an unstructured grid is legal and
    common. Calling it a volume would disable every tool on a perfectly
    ordinary mesh.
    """
    surface = pv.Sphere(theta_resolution=8, phi_resolution=8).triangulate()
    assert vm.kind_of(surface) == vm.SURFACE
    assert vm.kind_of(surface.cast_to_unstructured_grid()) == vm.SURFACE
    assert not vm.is_volume(surface)


def test_a_tetrahedral_grid_is_a_volume():
    grid = _tet_grid()
    assert vm.kind_of(grid) == vm.VOLUME
    assert vm.tetrahedra(grid).shape == (grid.n_cells, 4)


def test_non_tetrahedral_volumes_are_refused():
    """A hexahedral volume is rejected with a message, not half-handled."""
    hexa = pv.ImageData(dimensions=(3, 3, 3)).cast_to_unstructured_grid()
    assert vm.kind_of(hexa) == vm.VOLUME
    with pytest.raises(ValueError, match="only tetrahedral"):
        vm.validate_tetrahedral(hexa)
    # A tetrahedral one passes.
    vm.validate_tetrahedral(_tet_grid())


# ------------------------------------------------------------ orientation
def test_orientation_is_measured_and_can_be_normalised():
    grid = _tet_grid()
    points = np.asarray(grid.points)
    tets = vm.tetrahedra(grid)

    flipped = tets.copy()
    flipped[::2] = flipped[::2][:, [0, 1, 3, 2]]      # invert every other one
    assert (vm.signed_volumes(points, flipped) < 0).sum() == len(tets[::2])

    fixed, n = vm.orient_positive(points, flipped)
    assert n == len(tets[::2])
    assert (vm.signed_volumes(points, fixed) > 0).all()
    # Same tetrahedra, different node order: the point sets are untouched.
    assert [set(map(int, row)) for row in fixed] == \
           [set(map(int, row)) for row in flipped]


# ------------------------------------------------------------ the loader
def test_loading_a_volume_keeps_it(tmp_path):
    path = tmp_path / "vol.vtk"
    _tet_grid().save(path, binary=True)

    loader = MeshLoader()
    loader.load(path)
    assert loader.kind == vm.VOLUME
    assert loader.grid is not None
    assert loader.dataset is loader.grid
    # The surface is derived, and is smaller than the volume it came from.
    assert loader.mesh.n_cells < loader.grid.n_cells
    assert "elemTag" in loader.mesh.cell_data       # carries the parent's data


def test_loading_a_surface_is_unchanged(tmp_path):
    path = tmp_path / "surf.vtk"
    mesh = pv.Sphere(theta_resolution=8, phi_resolution=8).triangulate()
    mesh.save(path, binary=True)

    loader = MeshLoader()
    loader.load(path)
    assert loader.kind == vm.SURFACE
    assert loader.grid is None
    assert loader.dataset is loader.mesh
    assert "elemTag" in loader.mesh.cell_data       # seeded by the loader


def test_a_volume_round_trips_unchanged(tmp_path):
    """The whole point: what went in comes back out.

    Points, connectivity, every array and every dtype. A label cast to
    float or a fibre truncated to single precision would each be a quiet
    change to someone's simulation input.
    """
    src = tmp_path / "in.vtk"
    original = _tet_grid()
    original.save(src, binary=True)

    loader = MeshLoader()
    loader.load(src)
    out = tmp_path / "out.vtk"
    loader.save(out, fields=None, binary=True)

    back = pv.read(out)
    assert back.n_points == original.n_points
    assert back.n_cells == original.n_cells
    assert np.allclose(back.points, original.points)
    assert np.array_equal(np.asarray(back.cells), np.asarray(original.cells))
    assert np.array_equal(np.asarray(back.cell_data["elemTag"]),
                          np.asarray(original.cell_data["elemTag"]))
    assert np.asarray(back.cell_data["elemTag"]).dtype == np.int32
    assert np.allclose(np.asarray(back.cell_data["fiber"]),
                       np.asarray(original.cell_data["fiber"]))
    assert np.allclose(np.asarray(back.point_data["scar_probability"]),
                       np.asarray(original.point_data["scar_probability"]))


def test_saving_a_volume_writes_no_normals(tmp_path):
    """Normals describe a surface; on a volume they would mean nothing."""
    src = tmp_path / "in.vtk"
    _tet_grid().save(src, binary=True)
    loader = MeshLoader()
    loader.load(src)
    out = tmp_path / "out.vtk"
    loader.save(out, fields=None, binary=True)
    back = pv.read(out)
    assert "Normals" not in back.cell_data
    assert "Normals" not in back.point_data


def test_field_selection_applies_to_a_volume(tmp_path):
    src = tmp_path / "in.vtk"
    _tet_grid().save(src, binary=True)
    loader = MeshLoader()
    loader.load(src)
    out = tmp_path / "out.vtk"
    loader.save(out, fields=["elemTag"], binary=True)
    back = pv.read(out)
    assert list(back.cell_data.keys()) == ["elemTag"]
    assert list(back.point_data.keys()) == []


# --------------------------------------------------------- internal arrays
def test_boundary_bookkeeping_never_reaches_a_field_list(tmp_path):
    """The surface's map back to the volume is not one of its fields.

    ``vtkOriginalPointIds`` / ``vtkOriginalCellIds`` are real information
    and are kept on the surface, but they index the volume, so offering
    them as something to colour by or to write out would be offering
    someone else's row numbers.
    """
    path = tmp_path / "vol.vtk"
    _tet_grid().save(path, binary=True)
    loader = MeshLoader()
    loader.load(path)

    present = set(loader.mesh.point_data.keys()) | set(
        loader.mesh.cell_data.keys())
    assert present & INTERNAL_ARRAYS          # the map is there ...
    offered = set(MeshLoader.field_names(loader.mesh))
    assert not (offered & INTERNAL_ARRAYS)    # ... and is not offered


def test_replacing_with_a_surface_drops_the_volume(tmp_path):
    """A clip or a post-process hands back a surface, and that is the mesh.

    Leaving the old grid in place would make ``kind`` say volume while
    the working mesh is a surface, and the next save would write the
    stale tetrahedra.
    """
    path = tmp_path / "vol.vtk"
    _tet_grid().save(path, binary=True)
    loader = MeshLoader()
    loader.load(path)
    assert loader.kind == vm.VOLUME

    loader.set_surface(pv.Sphere(theta_resolution=8,
                                 phi_resolution=8).triangulate())
    assert loader.kind == vm.SURFACE
    assert loader.grid is None
    assert loader.dataset is loader.mesh


# ------------------------------------------- what the mesh tools are given
@pytest.mark.parametrize("volumetric", [True, False])
def test_the_surface_is_what_the_mesh_tools_get(tmp_path, volumetric):
    """``mesh`` is a surface for every kind of file, and ``load`` is not.

    The bug this pins: ``load`` returns the working mesh, which for a
    volumetric file is the grid. Adopting a mesh by passing on whatever
    ``load`` returned therefore handed the tagger, the editor and the
    clipper a ``vtkUnstructuredGrid``, and opening any tetrahedral file
    raised ``TypeError: mesh must be a pyvista.PolyData`` out of a Qt
    slot — which does not unwind, so the application aborted.
    """
    path = tmp_path / "mesh.vtk"
    if volumetric:
        _tet_grid().save(path, binary=True)
    else:
        pv.Sphere(theta_resolution=8, phi_resolution=8).triangulate().save(
            path, binary=True)

    loader = MeshLoader()
    returned = loader.load(path)

    # The surface is always a PolyData, whatever the file held ...
    assert isinstance(loader.mesh, pv.PolyData)
    # ... and it is what the mesh-side tools accept.
    RegionTagger(loader.mesh)

    if volumetric:
        # ... whereas the return value is the volume, and is not.
        assert returned is loader.grid
        assert not isinstance(returned, pv.PolyData)
        with pytest.raises(TypeError):
            RegionTagger(returned)
    else:
        assert returned is loader.mesh
