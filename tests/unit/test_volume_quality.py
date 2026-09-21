"""
test_volume_quality.py
======================
Repairing the shape of a tetrahedral volume's worst elements.

The contract:

* **the measure is the one the panel claims**: 0 for a regular
  tetrahedron, rising as it flattens, 2 for one turned inside out, so
  every threshold is an upper bound;
* **only the bad elements are touched.** A mesh with nothing above the
  threshold comes back untouched, and a node far from any bad element
  does not move;
* **the wall is kept.** A boundary node may slide in its own tangent
  plane but the set of boundary faces is unchanged, the surface area is
  unchanged to five figures, and a node on a sharp feature never moves
  at all;
* **a flip is judged after it is oriented.** The two elements of a
  3-to-2 flip sit on opposite sides of the base, so one is always wound
  the other way; judged unoriented, every flip reads as inverted and
  none ever fires;
* **a round that does not pay for itself is undone**, whole rather than
  in part;
* **fields are carried by selection**: a flip's replacement takes the
  label of an element it replaced, and point data stays with its node.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pytest
import pyvista as pv

from ccdaf.core import volume_mesh as vm
from ccdaf.core import volume_quality as vq


def _block(n: int = 6, size: float = 6.0) -> pv.UnstructuredGrid:
    """A tetrahedralised box carrying one of each kind of field."""
    grid = pv.ImageData(dimensions=(n, n, n),
                        spacing=(size / (n - 1),) * 3)
    grid = grid.cast_to_unstructured_grid().triangulate()
    grid.cell_data["elemTag"] = np.arange(grid.n_cells, dtype=np.int32) + 1
    grid.cell_data["fiber"] = np.tile([1.0, 0.0, 0.0], (grid.n_cells, 1))
    grid.point_data["scar_probability"] = np.linspace(0.0, 1.0,
                                                      grid.n_points)
    return grid


def _sliver(grid: pv.UnstructuredGrid, depth: float = 0.02):
    """Push one interior node almost onto a face it sits opposite.

    That is what a bad element in a segmented mesh looks like: the
    connectivity is sound and a handful of elements around one node are
    nearly flat. Moving the node toward the average of its neighbours
    would do nothing at all on a regular block — the average is where it
    already is — which is a fixture that tests nothing.
    """
    points = np.asarray(grid.points, dtype=float).copy()
    tets = vm.tetrahedra(grid)
    on_wall = np.zeros(len(points), dtype=bool)
    on_wall[np.unique(vm.boundary_faces(tets))] = True
    interior = np.where(~on_wall)[0]
    victim = int(interior[len(interior) // 2])

    host = tets[(tets == victim).any(axis=1)][0]
    face = host[host != victim]
    p0, p1, p2 = points[face]
    normal = np.cross(p1 - p0, p2 - p0)
    normal /= np.linalg.norm(normal)
    height = float(np.dot(points[victim] - p0, normal))
    points[victim] -= normal * height * (1.0 - depth)

    out = pv.UnstructuredGrid({vm.TETRA: tets}, points)
    for name in grid.cell_data:
        out.cell_data[name] = np.asarray(grid.cell_data[name])
    for name in grid.point_data:
        out.point_data[name] = np.asarray(grid.point_data[name])
    return out, victim


# ------------------------------------------------------------- measure
_REGULAR = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
                    dtype=float)


def test_a_regular_tetrahedron_measures_zero():
    assert vq.quality(_REGULAR, np.array([[0, 2, 1, 3]])) == pytest.approx(
        0.0, abs=1e-12)


def test_a_flattening_tetrahedron_approaches_one():
    a = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.4, 0.4, 1e-4]],
                 dtype=float)
    assert float(vq.quality(a, np.array([[0, 1, 2, 3]]))[0]) > 0.999


def test_a_degenerate_or_inverted_tetrahedron_is_reported_as_two():
    """The sentinel the flips exist to avoid producing.

    Zero volume counts as inverted, not as "flat": an element with no
    volume has no shape to rank, and calling it the worst possible is
    what keeps it out of every comparison a flip makes.
    """
    assert vq.quality(_REGULAR, np.array([[0, 1, 2, 3]])) == 2.0
    flat = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
                    dtype=float)
    assert vq.quality(flat, np.array([[0, 1, 2, 3]])) == 2.0


# --------------------------------------------------------- restraint
def test_a_sound_mesh_is_left_alone():
    grid = _block()
    before = np.asarray(grid.points).copy()
    out, report = vq.improve_grid(grid)
    assert not report.changed
    assert report.bad_before == 0
    assert np.array_equal(np.asarray(out.points), before)
    assert out.n_cells == grid.n_cells
    assert "nothing to repair" in report.summary()


def test_only_the_bad_neighbourhood_moves():
    grid, victim = _sliver(_block())
    before = np.asarray(grid.points).copy()
    out, report = vq.improve_grid(grid)

    assert report.bad_before > 0
    assert report.bad_after < report.bad_before
    moved = np.linalg.norm(np.asarray(out.points) - before, axis=1) > 0.0
    assert moved.any()
    assert not moved.all()          # the far side of the block is untouched
    # Whatever moved is next to the element that was wrong.
    tets = vm.tetrahedra(grid)
    near = np.unique(tets[(tets == victim).any(axis=1)])
    assert moved[near].any()


# ------------------------------------------------------------- the wall
def test_the_set_of_boundary_faces_is_unchanged():
    """A flip only ever reconnects material around an interior edge."""
    grid, _victim = _sliver(_block())
    before = vm.boundary_faces(vm.tetrahedra(grid))
    out, _report = vq.improve_grid(grid)
    after = vm.boundary_faces(vm.tetrahedra(out))
    assert before.shape == after.shape
    assert np.array_equal(before, after)


def test_a_frozen_wall_does_not_move_a_single_boundary_node():
    grid, _victim = _sliver(_block())
    before = np.asarray(grid.points).copy()
    out, report = vq.improve_grid(
        grid, vq.QualityOptions(slide_boundary=False))
    on_wall = np.zeros(len(before), dtype=bool)
    on_wall[np.unique(vm.boundary_faces(vm.tetrahedra(grid)))] = True
    assert np.array_equal(np.asarray(out.points)[on_wall], before[on_wall])
    assert report.max_boundary_shift == 0.0


def test_a_node_on_a_sharp_feature_is_held():
    """The rim of a valve opening must stay a rim.

    Every node of a cube's edges sits on a 90 degree feature, so with
    the wall sliding the corners and edges still may not move.
    """
    grid, _victim = _sliver(_block())
    points = np.asarray(grid.points)
    frame = vq._boundary_frame(points, vm.tetrahedra(grid),
                               vq.QualityOptions().feature_angle)
    on_edge = (np.abs(points - points.min(axis=0)) < 1e-9).sum(axis=1)
    on_edge += (np.abs(points - points.max(axis=0)) < 1e-9).sum(axis=1)
    assert frame.held[on_edge >= 2].all()

    out, _report = vq.improve_grid(grid)
    assert np.array_equal(np.asarray(out.points)[on_edge >= 2],
                          points[on_edge >= 2])


def test_a_pinched_node_is_held_even_with_the_wall_sliding():
    """Where the surface passes through a node twice there is no plane.

    Averaging the two sheets gives a direction belonging to neither, so
    sliding along it would take the node off the wall. The cleaner
    removes these; this pass must not depend on having been run after it.
    """
    angle = np.arange(4) * np.pi / 2.0
    lower = np.stack([np.cos(angle), np.sin(angle), np.zeros(4)], axis=1)
    upper = np.stack([0.2 * np.cos(angle), 0.2 * np.sin(angle),
                      np.ones(4)], axis=1)
    points = np.vstack([[0.0, 0.0, 0.0], lower, upper])
    tets = []
    for i in range(4):
        j = (i + 1) % 4
        tets.append((0, 1 + i, 1 + j, 5 + i))
        tets.append((0, 1 + j, 5 + j, 5 + i))
    grid = pv.UnstructuredGrid({vm.TETRA: np.array(tets, dtype=np.int64)},
                               points)
    frame = vq._boundary_frame(points, vm.tetrahedra(grid),
                               vq.QualityOptions().feature_angle)
    assert frame.held[0]

    out, _report = vq.improve_grid(grid)
    assert np.array_equal(np.asarray(out.points)[0], points[0])


def test_a_node_may_not_travel_further_than_its_limit():
    grid, _victim = _sliver(_block())
    before = np.asarray(grid.points).copy()
    options = vq.QualityOptions(max_travel=0.05)
    out, report = vq.improve_grid(grid, options)
    spacing = 6.0 / 5.0                      # the block's edge length
    assert report.max_shift <= options.max_travel * spacing * 1.001
    assert np.linalg.norm(np.asarray(out.points) - before,
                          axis=1).max() == pytest.approx(report.max_shift)


# -------------------------------------------------------------- flips
def test_a_flip_is_judged_after_it_is_oriented():
    """The bug that made every flip look inverted, pinned down.

    The two elements of a 3-to-2 flip lie on opposite sides of their
    shared base, so one of them always comes out wound the other way.
    Measured as written, it reads 2.0 — inverted — and the flip is
    rejected. Measured after orienting, it reads its true shape.
    """
    base = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]])
    points = np.vstack([base, [0.5, 0.4, 1.0], [0.5, 0.4, -1.0]])
    pair = np.array([[3, 0, 1, 2], [4, 0, 1, 2]])
    raw = vq.quality(points, pair)
    assert 2.0 in raw                       # one of the two reads inverted
    oriented, _flipped = vm.orient_positive(points, pair)
    assert (vq.quality(points, oriented) < 1.0).all()


def test_flips_can_be_switched_off():
    grid, _victim = _sliver(_block())
    _out, report = vq.improve_grid(grid, vq.QualityOptions(flips=False))
    assert report.flips_3_to_2 == 0
    assert report.flips_4_to_1 == 0


# ------------------------------------------------------------- carrying
def test_an_element_keeps_a_label_that_was_really_there():
    grid, _victim = _sliver(_block())
    out, _report = vq.improve_grid(grid)
    tags = np.asarray(out.cell_data["elemTag"])
    assert tags.dtype == np.int32
    assert set(tags.tolist()) <= set(
        np.asarray(grid.cell_data["elemTag"]).tolist())
    assert np.asarray(out.cell_data["fiber"]).shape == (out.n_cells, 3)


def test_point_data_stays_with_its_node():
    """No node is added or removed, so nothing is interpolated at all."""
    grid, _victim = _sliver(_block())
    out, _report = vq.improve_grid(grid)
    assert out.n_points == grid.n_points
    assert np.array_equal(np.asarray(out.point_data["scar_probability"]),
                          np.asarray(grid.point_data["scar_probability"]))


# -------------------------------------------------------------- options
def test_an_impossible_threshold_is_refused():
    for bad in (0.0, 2.0, -1.0):
        with pytest.raises(ValueError):
            vq.QualityOptions(threshold=bad).validate()
    with pytest.raises(ValueError):
        vq.QualityOptions(max_travel=0.0).validate()
    with pytest.raises(ValueError):
        vq.QualityOptions(max_rounds=0).validate()


def test_a_surface_is_refused():
    with pytest.raises(ValueError):
        vq.improve_grid(pv.Sphere().cast_to_unstructured_grid())
