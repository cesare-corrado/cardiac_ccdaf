"""
test_carp_export.py
===================
Export → Carp: the writer in ``ccdaf.core.carp_export``.

The contract:

* ``.pts`` holds a count and the points scaled to micrometres; ``.elem`` holds
  a count and one row per element, typed ``Tt`` for a tetrahedron and ``Tr``
  for a triangle, with the region tag last when one is written; ``.lon`` holds
  1 or 2 and one direction per element;
* region tags must be whole and non-negative, and a tag above 255 is reported
  because older CARP could not hold it;
* fibres are normalised in the file and never in the mesh, a zero-length
  vector stops the export, and a mesh with no fibres is written with the
  placeholder and said to be;
* a wrong-looking scale, a missing region tag and a crooked sheet are reported
  rather than refused;
* a dry run checks everything and writes nothing.

Synthetic meshes; no display, no Qt.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pytest
import pyvista as pv

from ccdaf.core import carp_export as ce


def _tets() -> pv.UnstructuredGrid:
    grid = pv.ImageData(dimensions=(4, 4, 4), spacing=(1.0, 1.0, 1.0)).triangulate()
    grid.cell_data["elemTag"] = np.ones(grid.n_cells, dtype=np.int32)
    grid.cell_data["fiber"] = np.tile([0.0, 2.0, 0.0], (grid.n_cells, 1))   # not unit
    return grid


def _triangles() -> pv.PolyData:
    mesh = pv.Plane(i_resolution=3, j_resolution=3).triangulate()
    mesh.cell_data["elemTag"] = np.ones(mesh.n_cells, dtype=np.int32)
    return mesh


def _options(tmp_path, **changes) -> ce.CarpExportOptions:
    base = dict(directory=str(tmp_path), name="mesh", scale=ce.DEFAULT_SCALE)
    base.update(changes)
    return ce.CarpExportOptions(**base)


def _lines(path: Path):
    return path.read_text().splitlines()


def test_the_three_files_carry_counts_points_elements_and_fibres(tmp_path):
    grid = _tets()
    result = ce.write_carp(grid, _options(tmp_path, region_field="elemTag",
                                          fibre_field="fiber"))
    points, elements, fibres = (_lines(p) for p in result.paths)

    assert points[0] == str(grid.n_points)
    assert elements[0] == str(grid.n_cells)
    assert fibres[0] == "1"
    assert len(points) == grid.n_points + 1
    assert len(elements) == grid.n_cells + 1
    assert len(fibres) == grid.n_cells + 1

    first = [float(v) for v in points[1].split()]
    assert first == pytest.approx(np.asarray(grid.points[0]) * 1000.0, abs=1e-3)

    row = elements[1].split()
    assert row[0] == "Tt" and len(row) == 6          # type, four nodes, tag
    assert [int(v) for v in row[1:5]] == list(grid.cells_dict[10][0])
    assert result.element_code == "Tt"
    assert result.tags == {1: grid.n_cells}


def test_a_surface_is_written_as_triangles(tmp_path):
    mesh = _triangles()
    result = ce.write_carp(mesh, _options(tmp_path))
    elements = _lines(result.paths[1])
    assert result.element_code == "Tr"
    assert elements[1].split()[0] == "Tr"
    assert len(elements[1].split()) == 5             # type, three nodes, tag
    assert result.tags == {ce.DEFAULT_TAG: mesh.n_cells}


def test_fibres_are_normalised_in_the_file_and_not_in_the_mesh(tmp_path):
    grid = _tets()
    before = np.array(grid.cell_data["fiber"], copy=True)
    result = ce.write_carp(grid, _options(tmp_path, fibre_field="fiber"))
    written = np.array([[float(v) for v in line.split()]
                        for line in _lines(result.paths[2])[1:]])
    assert np.allclose(np.linalg.norm(written, axis=1), 1.0)
    assert result.normalised == grid.n_cells
    assert np.array_equal(np.asarray(grid.cell_data["fiber"]), before)


def test_no_fibre_field_writes_the_placeholder(tmp_path):
    mesh = _triangles()
    result = ce.write_carp(mesh, _options(tmp_path))
    written = np.array([[float(v) for v in line.split()]
                        for line in _lines(result.paths[2])[1:]])
    assert result.placeholder_fibres
    assert np.allclose(written, ce.PLACEHOLDER_FIBRE)
    assert any("isotropic" in note for note in result.warnings)


def test_a_sheet_makes_two_axes_and_crooked_sheets_are_reported(tmp_path):
    grid = _tets()
    grid.cell_data["fiber"] = np.tile([1.0, 0.0, 0.0], (grid.n_cells, 1))
    grid.cell_data["sheet"] = np.tile([0.0, 1.0, 0.0], (grid.n_cells, 1))
    result = ce.write_carp(grid, _options(tmp_path, fibre_field="fiber",
                                          sheet_field="sheet"))
    assert result.fibre_axes == 2
    assert _lines(result.paths[2])[0] == "2"
    assert len(_lines(result.paths[2])[1].split()) == 6
    assert not any("perpendicular" in note for note in result.warnings)

    grid.cell_data["sheet"] = np.tile([1.0, 0.0, 0.0], (grid.n_cells, 1))   # parallel
    crooked = ce.write_carp(grid, _options(tmp_path, name="crooked",
                                           fibre_field="fiber", sheet_field="sheet"))
    assert any("perpendicular" in note for note in crooked.warnings)


def test_a_zero_length_direction_stops_the_export(tmp_path):
    grid = _tets()
    grid.cell_data["fiber"][0] = [0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="no length"):
        ce.write_carp(grid, _options(tmp_path, fibre_field="fiber"))


def test_a_constant_fibre_field_is_reported_as_a_placeholder(tmp_path):
    grid = _tets()
    grid.cell_data["fiber"] = np.tile([1.0, 0.0, 0.0], (grid.n_cells, 1))
    result = ce.write_carp(grid, _options(tmp_path, fibre_field="fiber"))
    assert any("same direction on every element" in note for note in result.warnings)


def test_region_tags_must_be_whole_and_not_negative(tmp_path):
    grid = _tets()
    grid.cell_data["fractional"] = np.full(grid.n_cells, 1.5)
    grid.cell_data["negative"] = np.full(grid.n_cells, -1, dtype=np.int32)
    with pytest.raises(ValueError, match="whole numbers"):
        ce.write_carp(grid, _options(tmp_path, region_field="fractional"))
    with pytest.raises(ValueError, match="negative"):
        ce.write_carp(grid, _options(tmp_path, region_field="negative"))


def test_a_tag_above_255_is_reported(tmp_path):
    grid = _tets()
    grid.cell_data["big"] = np.full(grid.n_cells, 300, dtype=np.int32)
    result = ce.write_carp(grid, _options(tmp_path, region_field="big"))
    assert any("255" in note for note in result.warnings)
    assert result.tags == {300: grid.n_cells}


def test_with_no_region_array_every_element_is_written_as_the_default_tag(tmp_path):
    grid = _tets()
    result = ce.write_carp(grid, _options(tmp_path))
    row = _lines(result.paths[1])[1].split()
    assert len(row) == 6                                     # type, four nodes, tag
    assert int(row[-1]) == ce.DEFAULT_TAG
    assert result.tags == {ce.DEFAULT_TAG: grid.n_cells}
    assert any(f"written with tag {ce.DEFAULT_TAG}" in note for note in result.warnings)


def test_an_implausible_scale_is_reported(tmp_path):
    # 90 units across: a heart in millimetres, so x1000 is a plausible 90 mm
    # in micrometres and x1 is not.
    grid = pv.ImageData(dimensions=(4, 4, 4), spacing=(30.0, 30.0, 30.0)).triangulate()
    grid.cell_data["elemTag"] = np.ones(grid.n_cells, dtype=np.int32)
    assert not any("scale factor" in note
                   for note in ce.write_carp(grid, _options(tmp_path)).warnings)
    unscaled = ce.write_carp(grid, _options(tmp_path, name="raw", scale=1.0))
    assert any("scale factor" in note for note in unscaled.warnings)


def test_a_dry_run_writes_nothing(tmp_path):
    result = ce.write_carp(_tets(), _options(tmp_path, region_field="elemTag"),
                           dry_run=True)
    assert result.tags and result.n_elements
    assert not any(path.exists() for path in result.paths)


@pytest.mark.parametrize("changes, message", [
    (dict(name=""), "Give the files a name"),
    (dict(name="mesh.pts"), "Leave the suffix off"),
    (dict(directory="/no/such/place"), "No such directory"),
    (dict(scale=0.0), "greater than 0"),
    (dict(sheet_field="sheet"), "needs a fibre direction"),
])
def test_options_that_must_stop_the_export(tmp_path, changes, message):
    with pytest.raises(ValueError, match=message):
        _options(tmp_path, **changes).validate()


def test_what_the_dialog_may_offer(tmp_path):
    grid = _tets()
    grid.cell_data["float_tags"] = np.ones(grid.n_cells)          # whole numbers
    grid.cell_data["measurement"] = np.linspace(0.0, 1.5, grid.n_cells)
    assert set(ce.region_fields(grid)) == {"elemTag", "float_tags"}
    assert ce.direction_fields(grid, ce.FIBRE_FIELDS) == ["fiber"]
    assert ce.direction_fields(grid, ce.SHEET_FIELDS) == []
    assert ce.availability(grid)[0] is True
    assert ce.availability(None)[0] is False


def test_the_files_share_one_prefix(tmp_path):
    options = _options(tmp_path, name="run1")
    assert [p.name for p in options.paths()] == ["run1.pts", "run1.elem", "run1.lon"]
    assert options.existing() == []
    ce.write_carp(_tets(), options)
    assert len(options.existing()) == 3
