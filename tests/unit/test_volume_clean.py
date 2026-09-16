"""
test_volume_clean.py
====================
Repairing a tetrahedral volume's connectivity, and measuring its topology.

The contract:

* a detached element is removed, because a component carrying no Dirichlet
  condition makes a Laplace system singular — that is the defect this
  exists for, and it is exactly what the example ventricle has;
* the clean **moves no vertex**. Welding at the default tolerance merges
  only exactly coincident points, and nothing else here touches a
  coordinate, so a mesh someone has already accepted comes back with the
  same geometry;
* fields are carried by *selection*, never interpolation: a surviving
  element keeps the label it had, so a mesh of 1s and 2s cannot come back
  with a 1.5;
* the largest component always survives, whatever the fraction is set to,
  so the clean cannot empty a mesh;
* a clean that changes nothing says so, because a no-op that marked the
  session dirty would make every inspection look like an edit;
* the Euler characteristic is always reported, and tunnels and cavities
  only when the boundary is manifold. A pinched boundary makes the sheet
  count wrong, and a tunnel figure derived from it wrong with it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pytest
import pyvista as pv
from PyQt5 import QtWidgets

from ccdaf.core import volume_mesh as vm
from ccdaf.core.volume_clean import (
    CleanOptions, clean, topology,
)
from ccdaf.gui.volume_postprocessing_widget import VolumePostprocessingWidget


def _block(n: int = 6, size: float = 6.0) -> pv.UnstructuredGrid:
    """A tetrahedralised box carrying one of each kind of field."""
    grid = pv.ImageData(
        dimensions=(n, n, n),
        spacing=(size / (n - 1),) * 3).cast_to_unstructured_grid()
    grid = grid.triangulate()
    centres = np.asarray(grid.cell_centers().points)
    grid.cell_data["elemTag"] = np.where(
        centres[:, 0] < size / 2.0, 1, 2).astype(np.int32)
    grid.cell_data["fiber"] = np.tile(
        np.array([1.0, 0.0, 0.0]), (grid.n_cells, 1))
    grid.point_data["scar_probability"] = np.linspace(
        0.0, 1.0, grid.n_points)
    return grid


def _with_floater(grid: pv.UnstructuredGrid) -> pv.UnstructuredGrid:
    """Add one tetrahedron joined to the body by an edge and no face.

    This is the shape of the real defect: on the example ventricle the
    three stray elements share two or three nodes with the body and no
    face at all, so nothing couples them to it.

    It must have a real volume, or the degenerate pass removes it first
    and the test passes while measuring the wrong thing — which is what
    a flat first version of this fixture did.
    """
    points = np.vstack([np.asarray(grid.points),
                        [[20.0, 0.0, 0.5], [21.0, 1.0, 2.0]]])
    n = grid.n_points
    tets = np.vstack([vm.tetrahedra(grid), [[0, 1, n, n + 1]]])
    out = pv.UnstructuredGrid({vm.TETRA: tets}, points)
    assert abs(float(vm.signed_volumes(points, tets[-1:])[0])) > 1e-9
    out.cell_data["elemTag"] = np.append(
        np.asarray(grid.cell_data["elemTag"]), 9).astype(np.int32)
    out.cell_data["fiber"] = np.vstack(
        [np.asarray(grid.cell_data["fiber"]), [0.0, 1.0, 0.0]])
    out.point_data["scar_probability"] = np.append(
        np.asarray(grid.point_data["scar_probability"]), [0.0, 0.0])
    return out


def _block_with_tunnel(n: int = 7, size: float = 6.0) -> pv.UnstructuredGrid:
    """A block with one square hole drilled through it.

    Whole cubes are removed, so the result stays manifold and its only
    feature is the tunnel: b1 = 1, and therefore chi = 0.
    """
    grid = _block(n=n, size=size)
    step = size / (n - 1)
    centres = np.asarray(grid.cell_centers().points)
    ix = np.floor(centres[:, 0] / step).astype(int)
    iy = np.floor(centres[:, 1] / step).astype(int)
    mid = (n - 1) // 2
    keep = np.where(~((ix == mid) & (iy == mid)))[0]
    return grid.extract_cells(keep)


# --------------------------------------------------------------- topology
def test_a_solid_block_has_no_tunnels_and_no_cavities():
    report = topology(_block())
    assert report.components == 1
    assert report.euler_characteristic == 1      # contractible
    assert report.boundary_is_manifold
    assert report.tunnels == 0
    assert report.cavities == 0


def test_a_drilled_block_reports_exactly_one_tunnel():
    """The measurement that matters, on a mesh whose answer is known."""
    report = topology(_block_with_tunnel())
    assert report.components == 1
    assert report.euler_characteristic == 0
    assert report.boundary_is_manifold
    assert report.tunnels == 1
    assert report.cavities == 0


def test_a_detached_element_is_counted_as_its_own_component():
    report = topology(_with_floater(_block()))
    assert report.components == 2


def test_tunnels_are_not_guessed_when_the_boundary_is_not_manifold():
    """Two blocks meeting at one node: the sheet count cannot be trusted.

    Reporting a number here is worse than reporting none — it is how this
    module's own development produced three different tunnel counts for
    one mesh.
    """
    a = _block(n=4, size=3.0)
    b = _block(n=4, size=3.0)
    # Two cubes sharing exactly the corner (3, 3, 3). Welded by hand:
    # pyvista's clean has moved its signature more than once, and this
    # merge has to be exact for the pinch to exist at all.
    points = np.vstack([np.asarray(a.points),
                        np.asarray(b.points) + np.array([3.0, 3.0, 3.0])])
    tets = np.vstack([vm.tetrahedra(a), vm.tetrahedra(b) + a.n_points])
    uniq, inverse = np.unique(points.round(9), axis=0, return_inverse=True)
    merged = pv.UnstructuredGrid({vm.TETRA: inverse.ravel()[tets]}, uniq)
    report = topology(merged)
    assert report.pinched_vertices >= 1
    assert report.tunnels is None and report.cavities is None
    assert "not determined" in report.summary()


# ------------------------------------------------------------------ clean
def test_a_detached_element_is_dropped():
    source = _with_floater(_block())
    out, report = clean(source)
    assert report.dropped_components == 1
    assert report.dropped_component_cells == 1
    assert out.n_cells == source.n_cells - 1
    assert topology(out).components == 1
    # The floater's own label went with it.
    assert 9 not in np.unique(np.asarray(out.cell_data["elemTag"])).tolist()


def test_the_largest_component_always_survives():
    """Even at a fraction that would otherwise drop everything."""
    source = _with_floater(_block())
    out, _ = clean(source, CleanOptions(min_component_fraction=1.0))
    assert out.n_cells == source.n_cells - 1
    assert out.n_cells > 0


def test_a_zero_fraction_keeps_every_piece():
    """0 disables the pass rather than meaning "drop everything".

    Someone who wants the other four passes and no component trimming has
    to be able to say so, and the obvious way to say it is 0.
    """
    source = _with_floater(_block())
    out, report = clean(source, CleanOptions(min_component_fraction=0.0))
    assert report.dropped_components == 0
    assert report.dropped_component_cells == 0
    assert out.n_cells == source.n_cells
    assert topology(out).components == 2       # the floater is still there


def test_two_real_bodies_are_both_kept():
    """A separately meshed pair is not a defect, and must not be trimmed."""
    a = _block(n=5, size=4.0)
    b = _block(n=5, size=4.0)
    b.points = np.asarray(b.points) + np.array([10.0, 0.0, 0.0])
    both = a + b
    out, report = clean(both)
    assert report.dropped_components == 0
    assert out.n_cells == both.n_cells


def test_a_clean_mesh_is_reported_as_unchanged():
    source = _block()
    out, report = clean(source)
    assert not report.changed
    assert report.summary().startswith("Nothing to clean")
    assert out.n_cells == source.n_cells
    assert out.n_points == source.n_points


def test_no_vertex_moves():
    """The whole reason this is safe to run on an accepted mesh."""
    source = _with_floater(_block())
    out, _ = clean(source)
    from scipy.spatial import cKDTree
    d, _i = cKDTree(np.asarray(source.points)).query(np.asarray(out.points))
    assert float(d.max()) == 0.0


def test_duplicate_elements_are_dropped():
    source = _block(n=4, size=3.0)
    tets = vm.tetrahedra(source)
    doubled = pv.UnstructuredGrid(
        {vm.TETRA: np.vstack([tets, tets[:3]])}, np.asarray(source.points))
    out, report = clean(doubled)
    assert report.dropped_duplicate == 3
    assert out.n_cells == len(tets)


def test_degenerate_elements_are_dropped():
    """A repeated node has no volume and no business in a solve."""
    source = _block(n=4, size=3.0)
    tets = vm.tetrahedra(source)
    flat = np.array([[tets[0][0], tets[0][1], tets[0][2], tets[0][2]]])
    broken = pv.UnstructuredGrid(
        {vm.TETRA: np.vstack([tets, flat])}, np.asarray(source.points))
    out, report = clean(broken)
    assert report.dropped_degenerate == 1
    assert out.n_cells == len(tets)


def test_coincident_points_are_merged():
    source = _block(n=4, size=3.0)
    points = np.vstack([np.asarray(source.points),
                        np.asarray(source.points)[:5]])
    tets = vm.tetrahedra(source)
    grid = pv.UnstructuredGrid({vm.TETRA: tets}, points)
    out, report = clean(grid)
    # The duplicates were unused as well, so both passes account for them.
    assert report.merged_points == 5
    assert out.n_points == source.n_points


def test_inverted_elements_are_reoriented_not_dropped():
    source = _block(n=4, size=3.0)
    tets = np.array(vm.tetrahedra(source), copy=True)
    tets[:4] = tets[:4][:, [0, 1, 3, 2]]          # flip the first four
    grid = pv.UnstructuredGrid({vm.TETRA: tets}, np.asarray(source.points))
    assert vm.inverted_count(grid) > 0

    out, report = clean(grid)
    assert report.flipped >= 4
    assert out.n_cells == grid.n_cells            # nothing was removed
    assert vm.inverted_count(out) == 0


def test_fields_are_carried_by_selection():
    """Labels stay labels; a fibre stays the fibre that element had."""
    source = _with_floater(_block())
    out, _ = clean(source)
    tags = np.asarray(out.cell_data["elemTag"])
    assert set(np.unique(tags).tolist()) == {1, 2}
    assert np.asarray(out.cell_data["fiber"]).shape == (out.n_cells, 3)
    assert "scar_probability" in out.point_data
    got = np.asarray(out.point_data["scar_probability"])
    assert np.isfinite(got).all()
    assert got.min() >= 0.0 and got.max() <= 1.0


def test_the_source_is_not_modified():
    source = _with_floater(_block())
    cells, points = source.n_cells, source.n_points
    tags = np.array(source.cell_data["elemTag"], copy=True)
    clean(source)
    assert source.n_cells == cells and source.n_points == points
    assert np.array_equal(np.asarray(source.cell_data["elemTag"]), tags)


def test_the_result_is_a_valid_volume():
    out, _ = clean(_with_floater(_block()))
    vm.validate_tetrahedral(out)
    assert set(np.unique(out.celltypes).tolist()) == {vm.TETRA}
    assert vm.inverted_count(out) == 0


# ---------------------------------------------------------------- options
@pytest.mark.parametrize("options", [
    CleanOptions(merge_tol=-1.0),
    CleanOptions(min_volume=-1.0),
    CleanOptions(min_component_fraction=-0.1),
    CleanOptions(min_component_fraction=1.5),
])
def test_impossible_options_are_refused(options):
    with pytest.raises(ValueError):
        clean(_block(n=4, size=3.0), options)


def test_a_non_tetrahedral_volume_is_refused():
    grid = pv.ImageData(dimensions=(3, 3, 3)).cast_to_unstructured_grid()
    with pytest.raises(ValueError):
        clean(grid)


def test_a_volume_carrying_surface_triangles_is_refused():
    """Mixed meshes are refused at the door, not half-handled.

    Checking only the *solid* cell types let a file holding tetrahedra and
    its own boundary triangles through, because triangles are 2-D. It
    loaded as a volume whose cell count included the triangles, so every
    per-element array was misaligned, and this very function then reported
    "nothing to clean" while dropping those triangles without a word.
    """
    block = _block(n=4, size=3.0)
    tets = vm.tetrahedra(block)
    surface = block.extract_surface(algorithm="dataset_surface").triangulate()
    tri = np.asarray(surface.faces).reshape(-1, 4)[:, 1:]
    origin = np.asarray(surface.point_data["vtkOriginalPointIds"],
                        dtype=np.int64)
    mixed = pv.UnstructuredGrid(
        {vm.TETRA: tets, int(pv.CellType.TRIANGLE): origin[tri]},
        np.asarray(block.points))
    assert mixed.n_cells > len(tets)

    with pytest.raises(ValueError, match="only tetrahedral volumes"):
        vm.validate_tetrahedral(mixed)
    with pytest.raises(ValueError, match="only tetrahedral volumes"):
        clean(mixed)


# ------------------------------------------------------------- the panel
@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def test_panel_defaults_match_the_options(qapp):
    """A default in the panel that disagreed with the core would mean the
    documented behaviour was never what the button did."""
    panel = VolumePostprocessingWidget()
    defaults, got = CleanOptions(), panel.clean_options()
    assert got.merge_tol == pytest.approx(defaults.merge_tol)
    assert got.min_component_fraction == pytest.approx(
        defaults.min_component_fraction)
    assert got.fix_inverted is defaults.fix_inverted


def test_clean_controls_have_tooltips(qapp):
    panel = VolumePostprocessingWidget()
    missing = [name for name in ("spn_merge_tol", "spn_min_component",
                                 "chk_fix_inverted", "btn_clean")
               if not getattr(panel, name).toolTip().strip()]
    assert not missing, "controls missing a tooltip: " + ", ".join(missing)


def test_both_actions_are_held_while_one_runs(qapp):
    """They act on the same working volume, so neither may start twice."""
    panel = VolumePostprocessingWidget()
    panel.set_busy(True, cleaning=True)
    assert not panel.btn_apply.isEnabled()
    assert not panel.btn_clean.isEnabled()
    assert panel.btn_clean.text() == "Cleaning…"
    panel.set_busy(False)
    assert panel.btn_apply.isEnabled() and panel.btn_clean.isEnabled()
    assert panel.btn_clean.text() == "Clean volume"
    assert panel.btn_apply.text() == "Remesh volume"


def test_the_button_asks_for_a_clean(qapp):
    panel = VolumePostprocessingWidget()
    seen = []
    panel.clean_requested.connect(lambda: seen.append(True))
    panel.btn_clean.click()
    assert seen == [True]
