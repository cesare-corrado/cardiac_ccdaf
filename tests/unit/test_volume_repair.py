"""
test_volume_repair.py
=====================
Making a tetrahedral volume's boundary manifold.

The contract:

* **material that only touches is separated, and nothing else changes.**
  Every element survives, every coordinate stays where it was, and the
  copies of a split node sit exactly on top of the original — the only
  thing that changes is which element refers to which node;
* **a pinhole is welded shut**, by filling the empty wedge at the
  contact. That adds material, so it is bounded: an element a weld adds
  may not be larger than the elements already at the contact, and a
  contact whose repair would cost more than that is reported rather than
  filled;
* **the boundary comes back manifold**, which is the whole point: a
  pinch is where a surface label leaks from epicardium to endocardium,
  and a non-manifold boundary makes the tunnel and cavity counts
  unknowable;
* **every field is carried by selection.** A split node's copies take
  the original node's value; a welded element takes the value of an
  element it plugs against. Nothing is interpolated, so a mesh of 1s and
  2s cannot come back with a 1.5;
* **what cannot be repaired is counted, never silently left.**
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pytest
import pyvista as pv

from ccdaf.core import volume_mesh as vm
from ccdaf.core import volume_repair as vr
from ccdaf.core.volume_clean import CleanOptions, clean, topology


def _grid(points, tets) -> pv.UnstructuredGrid:
    return pv.UnstructuredGrid({vm.TETRA: np.asarray(tets, dtype=np.int64)},
                               np.asarray(points, dtype=float))


def _with_fields(grid: pv.UnstructuredGrid) -> pv.UnstructuredGrid:
    """One of each kind of field, with values that make their source obvious."""
    grid.cell_data["elemTag"] = (np.arange(grid.n_cells) + 1).astype(np.int32)
    grid.cell_data["fiber"] = np.tile([1.0, 0.0, 0.0], (grid.n_cells, 1))
    grid.point_data["scar_probability"] = np.arange(
        grid.n_points, dtype=float)
    return grid


def _touching_at_a_node() -> pv.UnstructuredGrid:
    """Two tetrahedra sharing node 0 and nothing else."""
    points = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
              (-1, 0, 0), (0, -1, 0), (0, 0, -1)]
    return _with_fields(_grid(points, [(0, 1, 2, 3), (0, 4, 5, 6)]))


def _touching_along_an_edge() -> pv.UnstructuredGrid:
    """Two tetrahedra sharing edge 0-1 and no face."""
    points = [(0, 0, 0), (0, 0, 2),
              (1, 0, 0), (1, 1, 0),
              (-1, 0, 0), (-1, -1, 0)]
    return _with_fields(_grid(points, [(0, 1, 2, 3), (0, 1, 4, 5)]))


def _pinhole_at_a_node(inner: float = 0.2) -> pv.UnstructuredGrid:
    """A ring of tetrahedra around node 0: a passage closing to a point.

    The elements form one face-connected fan, so nothing here can be
    split — the material really is continuous around the contact. The
    surface still passes through node 0 twice, because the link of that
    node is an annulus rather than a disk. That is the defect a weld
    exists for, and it is exactly the shape of the 17 pinches on the
    example ventricle.
    """
    angle = np.arange(4) * np.pi / 2.0
    lower = np.stack([np.cos(angle), np.sin(angle), np.zeros(4)], axis=1)
    upper = np.stack([inner * np.cos(angle), inner * np.sin(angle),
                      np.ones(4)], axis=1)
    points = np.vstack([[0.0, 0.0, 0.0], lower, upper])
    tets = []
    for i in range(4):
        j = (i + 1) % 4
        tets.append((0, 1 + i, 1 + j, 5 + i))
        tets.append((0, 1 + j, 5 + j, 5 + i))
    return _with_fields(_grid(points, tets))


def _volume_of(grid: pv.UnstructuredGrid) -> float:
    return float(np.abs(
        grid.compute_cell_sizes().cell_data["Volume"]).sum())


# ------------------------------------------------------------- detection
def test_a_sound_mesh_is_left_alone():
    """The repair must be a no-op on a mesh that has no defect.

    A repair that always found something to do would make every
    inspection look like an edit.
    """
    grid = _grid([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
                 [(0, 1, 2, 3)])
    points, tets, _sp, _sc, report = vr.repair(
        np.asarray(grid.points), vm.tetrahedra(grid))
    assert not report.changed
    assert len(tets) == 1 and len(points) == 4
    assert report.summary() == "the boundary was already manifold"


def test_a_pinch_is_found_at_the_node_where_the_surface_touches_itself():
    grid = _touching_at_a_node()
    faces = vm.boundary_faces(vm.tetrahedra(grid))
    assert vr.pinched_vertices(faces).tolist() == [0]


def test_a_non_manifold_edge_is_found_by_its_four_faces():
    grid = _touching_along_an_edge()
    faces = vm.boundary_faces(vm.tetrahedra(grid))
    edges = vr.non_manifold_edges(faces)
    assert edges.shape == (1, 2)
    assert sorted(edges[0].tolist()) == [0, 1]


# ----------------------------------------------------------- separating
_SEPARATING = CleanOptions(separate_touching=True)


def test_material_touching_at_a_node_is_separated_losslessly():
    grid = _touching_at_a_node()
    before = _volume_of(grid)
    out, report = clean(grid, _SEPARATING)

    assert report.repair.split_vertices == 1
    assert out.n_cells == grid.n_cells          # nothing removed
    assert out.n_points == grid.n_points + 1    # one node copied
    assert topology(out).boundary_is_manifold
    assert _volume_of(out) == pytest.approx(before)
    assert report.repair.added_cells == 0       # no material invented


def test_the_copies_of_a_split_node_sit_on_the_original():
    """A split changes connectivity, never geometry."""
    grid = _touching_at_a_node()
    points, tets, source, _nodes, copies = vr.split_touching(
        np.asarray(grid.points, dtype=float), vm.tetrahedra(grid))
    assert copies == 1
    assert np.allclose(points[-1], np.asarray(grid.points)[source[-1]])
    assert np.allclose(points[:grid.n_points], np.asarray(grid.points))


def test_material_touching_along_an_edge_is_separated():
    grid = _touching_along_an_edge()
    out, report = clean(grid, _SEPARATING)
    assert out.n_cells == grid.n_cells
    assert out.n_points == grid.n_points + 2    # both ends of the edge
    assert topology(out).boundary_is_manifold
    assert report.repair.added_cells == 0


def test_a_split_node_carries_its_point_data_to_every_copy():
    grid = _touching_at_a_node()
    out, _report = clean(grid, _SEPARATING)
    values = np.asarray(out.point_data["scar_probability"])
    # Node 0 was the contact: its copy must hold node 0's value, which
    # an interpolating carry would have averaged away.
    assert (values == 0.0).sum() == 2


# -------------------------------------------------------------- welding
def test_a_pinhole_at_a_node_is_welded_shut():
    grid = _pinhole_at_a_node()
    before = _volume_of(grid)
    out, report = clean(grid)

    assert report.repair.welded_vertices == 1
    assert report.repair.added_cells == 2       # a four-node loop caps with two
    assert out.n_cells == grid.n_cells + 2
    assert out.n_points == grid.n_points        # a weld adds no node
    assert topology(out).boundary_is_manifold
    assert _volume_of(out) > before             # material was added
    assert report.repair.added_volume == pytest.approx(
        _volume_of(out) - before, rel=1e-6)


def test_a_weld_adds_only_positively_oriented_elements():
    """Sound whatever the caller's orientation pass does, or does not do."""
    grid = _pinhole_at_a_node()
    points, tets, _sp, _sc, report = vr.repair(
        np.asarray(grid.points, dtype=float), vm.tetrahedra(grid))
    assert report.added_cells == 2
    assert (vm.signed_volumes(points, tets) > 0.0).all()


def test_a_welded_element_inherits_a_neighbour_s_labels():
    """Carried by selection: the element it plugs against, never a mean."""
    grid = _pinhole_at_a_node()
    out, report = clean(grid)
    tags = np.asarray(out.cell_data["elemTag"])
    assert tags.dtype == np.int32
    assert set(np.unique(tags)) <= set(
        np.unique(np.asarray(grid.cell_data["elemTag"])))
    fibres = np.asarray(out.cell_data["fiber"])
    assert np.allclose(fibres[-report.repair.added_cells:], [1.0, 0.0, 0.0])


def test_a_weld_is_bounded_by_the_elements_already_there():
    """The guard that keeps a repair from filling a real passage.

    Same mesh, same defect, and the only difference is how much material
    a weld may add: at zero it is refused and counted, and at the
    default it is filled.
    """
    grid = _pinhole_at_a_node()
    strict = clean(grid, CleanOptions(max_weld_volume=1e-9))[1]
    assert strict.repair.added_cells == 0
    assert strict.repair.unrepaired_vertices == 1
    assert not topology(clean(grid, CleanOptions(
        max_weld_volume=1e-9))[0]).boundary_is_manifold

    generous = clean(grid)[1]
    assert generous.repair.added_cells == 2
    assert generous.repair.unrepaired_vertices == 0


def test_a_pinhole_cannot_be_split_apart():
    """Splitting is refused where the material is genuinely continuous.

    The tetrahedra of the ring form one face-connected fan, so there is
    no way to give the node two copies without tearing material apart.
    Welding is the only honest repair, which is why it exists.
    """
    grid = _pinhole_at_a_node()
    _points, _tets, _source, nodes, copies = vr.split_touching(
        np.asarray(grid.points, dtype=float), vm.tetrahedra(grid))
    assert (nodes, copies) == (0, 0)


def test_an_edge_pinhole_is_welded_with_one_element():
    """With splitting off, the edge repair has to stand on its own."""
    grid = _touching_along_an_edge()
    points, tets, _sp, sources, report = vr.repair(
        np.asarray(grid.points, dtype=float), vm.tetrahedra(grid),
        vr.RepairOptions(split_touching=False))
    assert report.welded_edges == 1
    assert report.added_cells == 1
    # One source per element of the result, the new one included: that
    # is what lets the caller carry a field without knowing which
    # elements were added.
    assert len(sources) == len(tets) == 3
    assert sources[:2].tolist() == [0, 1] and sources[2] in (0, 1)
    assert topology(_grid(points, tets)).boundary_is_manifold


# ---------------------------------------------------------------- gating
def test_separating_is_off_unless_it_is_asked_for():
    """The default must not change what a working pipeline sees.

    A non-manifold edge is skipped by anything that walks faces across
    shared edges, so before the split those edges act as accidental cuts
    in the boundary. The ventricular surface labelling depends on that:
    measured on the example ventricle, with the split it finds two
    surface pieces where it needs three. The split is still right and
    still available; it is not the default.
    """
    grid = _touching_at_a_node()
    out, report = clean(grid)
    assert report.repair.split_vertices == 0
    assert out.n_points == grid.n_points
    assert CleanOptions().separate_touching is False


def test_the_repair_can_be_switched_off():
    """The previous behaviour stays reachable: report, repair nothing."""
    grid = _touching_at_a_node()
    out, report = clean(grid, CleanOptions(repair_boundary=False))
    assert not report.repair.changed
    assert out.n_points == grid.n_points
    assert not topology(out).boundary_is_manifold
    assert report.before.pinched_vertices == 1


def test_a_repair_alone_counts_as_a_change():
    """Or the session would not know the mesh on screen had been edited."""
    grid = _touching_at_a_node()
    _out, report = clean(grid)
    assert report.merged_points == 0
    assert report.dropped_degenerate == 0
    assert report.changed


def test_an_impossible_weld_limit_is_refused():
    with pytest.raises(ValueError):
        CleanOptions(max_weld_volume=-1.0).validate()
    with pytest.raises(ValueError):
        vr.RepairOptions(max_passes=0).validate()


def test_a_weld_that_would_add_a_flat_element_is_refused():
    """A wedge can be wide and thin at once.

    Such a weld passes every volume test and still produces an element
    with no usable shape, which repairs nothing: it moves the defect
    from the topology report to the quality report. The same pinhole,
    narrowed until capping it would need a needle, has to be reported
    instead of filled.
    """
    filled = clean(_pinhole_at_a_node(inner=0.4))[1]
    assert filled.repair.welded_vertices == 1

    needle = clean(_pinhole_at_a_node(inner=0.02))[1]
    assert needle.repair.welded_vertices == 0
    assert needle.repair.unrepaired_vertices == 1


def test_a_weld_candidate_is_measured_after_it_is_oriented():
    """The winding trap, pinned where it actually bit.

    A cap triangle comes out of ear-clipping with whichever winding the
    loop had, so coning it back to the vertex gives a node order that
    may be negatively wound. Measured as written it reads 2.0, inverted,
    and every weld is refused for a fault in the bookkeeping rather than
    in the geometry.
    """
    grid = _pinhole_at_a_node(inner=0.4)
    points = np.asarray(grid.points, dtype=float)
    cap = np.array([[0, 5, 8, 7]], dtype=np.int64)
    assert vm.shape_measure(points[cap]) == 2.0      # as written
    oriented, _flipped = vm.orient_positive(points, cap)
    assert float(vm.shape_measure(points[oriented])[0]) < 0.95


def test_a_plug_looks_up_its_neighbours_by_node_not_by_element():
    """The lookup that would have crashed the moment a plug fired.

    The elements around a candidate are found from the nodes of its two
    faces. Passing the faces' *owning elements* instead reads the node
    adjacency past its end, which nothing noticed because no candidate
    had ever survived far enough to ask.
    """
    grid = _pinhole_at_a_node(inner=0.4)
    points = np.asarray(grid.points, dtype=float)
    tets = vm.tetrahedra(grid)
    starts, cells = vm.node_cells(tets, len(points))
    faces, owners, _apexes = vm.boundary_table(tets)

    # An owner is a cell index, and cell indices run past the node count
    # on any mesh with more elements than nodes — which is most of them.
    assert owners.max() >= len(points) or len(tets) < len(points)
    for node in np.concatenate([faces[0], faces[1]]):
        assert vr._cells_at(starts, cells, int(node)).size


def test_plugging_a_pinhole_mesh_does_not_raise():
    """The whole pass, with plugging on, over a mesh that has a pinhole."""
    grid = _pinhole_at_a_node(inner=0.4)
    points, tets, _sp, _sc, report = vr.repair(
        np.asarray(grid.points, dtype=float), vm.tetrahedra(grid),
        vr.RepairOptions(plug_perforations=True))
    assert len(tets) >= grid.n_cells
    assert report.genus_after is None or report.genus_after >= 0


# ------------------------------------------- saying why nothing was plugged
def _ring() -> pv.UnstructuredGrid:
    """A 3 x 3 x 1 slab of cubes with the middle one taken out.

    One handle, and the hole is a whole element wide: far wider than
    any gap a plug may span.
    """
    grid = pv.ImageData(dimensions=(4, 4, 2)).cast_to_unstructured_grid()
    centres = np.asarray(grid.cell_centers().points)
    middle = (np.abs(centres[:, 0] - 1.5) < 0.1) & (
        np.abs(centres[:, 1] - 1.5) < 0.1)
    ring = grid.extract_cells(np.where(~middle)[0]).triangulate()
    return pv.UnstructuredGrid(ring.cells, ring.celltypes,
                               np.asarray(ring.points))


def _plug_only(grid, **options):
    return vr.repair(np.asarray(grid.points, dtype=float), vm.tetrahedra(grid),
                     vr.RepairOptions(plug_perforations=True, **options))[4]


def test_plugging_says_when_the_boundary_is_too_broken_to_look():
    # It used to return silently here: the report read exactly as if
    # plugging had been off.
    report = _plug_only(_touching_along_an_edge(), split_touching=False,
                        weld_pinholes=False)
    assert report.plugged == 0 and report.genus_before is None
    assert "not manifold (1 contact)" in report.plug_note
    assert report.plug_note in report.summary()


def test_plugging_says_when_there_is_nothing_to_plug():
    grid = _grid([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
                 [(0, 1, 2, 3)])
    report = _plug_only(grid)
    assert "no handles" in report.plug_note
    assert report.summary() == ("the boundary was already manifold, "
                                + report.plug_note)


def test_plugging_says_when_no_handle_is_narrow_enough():
    report = _plug_only(_ring())
    assert report.genus_before == 1 and report.plugged == 0
    assert "none of the 1 handle is a gap narrow enough" in report.plug_note


def test_plugging_off_says_nothing_about_plugging():
    report = vr.repair(*(lambda g: (np.asarray(g.points, dtype=float),
                                    vm.tetrahedra(g)))(_ring()))[4]
    assert report.plug_note == ""


def test_the_clean_report_carries_the_reason():
    grid = _ring()
    plain = clean(grid, CleanOptions())[1]
    asked = clean(grid, CleanOptions(plug_perforations=True))[1]
    assert plain.summary() == "Nothing to clean; the mesh was already sound."
    assert "narrow enough" in asked.summary()
    assert not asked.changed
