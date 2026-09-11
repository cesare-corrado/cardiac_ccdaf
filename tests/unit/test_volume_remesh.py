"""
test_volume_remesh.py
=====================
Adapting a tetrahedral volume, and carrying its fields onto the result.

MMG3D hands back geometry and nothing else — no labels, no fibres, no
point fields. That is measured, not assumed, and it is why every remesh
ends in a transfer. If the transfer were ever skipped the result would
look like a mesh and be one with every field silently gone.

The contract:

* the frozen boundary is *exact*, not approximate — the whole reason it
  is the default is that the anatomy does not move;
* labels are copied, never averaged: a mesh of 1s and 2s comes back with
  1s and 2s and nothing in between;
* a fibre direction is **axial**, so ``f`` and ``-f`` are one direction.
  Averaging them as vectors cancels them to nothing; averaging the outer
  products does not. This is the case that makes the difference visible;
* the result is a valid volume — tetrahedra only, none inverted;
* options MMG would reject are refused before the call, with a message;
* ``elemTag`` rides as an MMG element *reference* and comes back exact,
  including a labelling that does not start at 1 — references are
  positive integers and a labelling need not be.

The MMG call itself is real: a stub would test the plumbing and not the
thing that can actually be wrong. The meshes are small enough to keep it
quick.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pytest
import pyvista as pv

from ccdaf.core import volume_mesh as vm
from ccdaf.core.field_transfer import (
    average_axial, average_axial_grouped, transfer_volume_fields,
)
from ccdaf.core.volume_postprocessor import (
    RemeshOptions, decode_tags, encode_tags, remesh,
)


def _block(n: int = 6, size: float = 6.0) -> pv.UnstructuredGrid:
    """A tetrahedralised box, labelled in two halves along x."""
    grid = pv.ImageData(
        dimensions=(n, n, n),
        spacing=(size / (n - 1),) * 3).cast_to_unstructured_grid()
    grid = grid.triangulate()                     # hexahedra -> tetrahedra
    centres = np.asarray(grid.cell_centers().points)
    grid.cell_data["elemTag"] = np.where(
        centres[:, 0] < size / 2.0, 1, 2).astype(np.int32)
    # A fibre along x, but with the sign of every other element flipped.
    # As a direction they all agree; as vectors they cancel. Any averaging
    # that is not sign-free fails on this.
    fibre = np.tile(np.array([1.0, 0.0, 0.0]), (grid.n_cells, 1))
    fibre[::2] *= -1.0
    grid.cell_data["fiber"] = fibre
    grid.point_data["scar_probability"] = np.linspace(0.0, 1.0, grid.n_points)
    return grid


# ------------------------------------------------------- axial averaging
def test_averaging_opposite_directions_keeps_the_direction():
    """The bug this exists to prevent, in one assertion.

    Two elements storing the same fibre direction with opposite signs
    average, as vectors, to the zero vector: a direction pointing
    nowhere, of unit-ish magnitude nowhere in the data, that renders and
    exports like a real one.
    """
    opposed = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    assert np.allclose(opposed.mean(axis=0), 0.0)          # the naive way

    axis = average_axial(opposed)
    assert np.isclose(np.linalg.norm(axis), 1.0)
    assert np.isclose(abs(float(axis[0])), 1.0)


def test_axial_average_is_a_unit_vector_with_a_stable_sign():
    rng = np.random.default_rng(0)
    v = np.array([1.0, 2.0, -0.5])
    v /= np.linalg.norm(v)
    signs = rng.choice([-1.0, 1.0], size=(20, 1))
    axis = average_axial(v[None, :] * signs)
    assert np.isclose(np.linalg.norm(axis), 1.0)
    assert np.isclose(abs(float(np.dot(axis, v))), 1.0)
    # Deterministic: the same input twice gives the same arrow, not its
    # opposite, or a diff of two runs would be full of sign flips.
    assert np.allclose(axis, average_axial(v[None, :] * signs))


def test_axial_average_of_nothing_is_not_a_crash():
    assert np.allclose(average_axial(np.zeros((0, 3))), 0.0)
    assert np.allclose(average_axial(np.zeros((3, 3))), 0.0)


def test_the_batched_average_agrees_with_the_single_one():
    """The batched form exists for speed and must mean the same thing.

    It replaced a per-cell Python loop that was 12 of the 14 seconds a
    field transfer took; a fast version that quietly disagreed would be
    the worst of both.
    """
    rng = np.random.default_rng(1)
    vectors = rng.normal(size=(60, 3))
    groups = [rng.choice(60, size=rng.integers(1, 8), replace=False)
              for _ in range(25)]
    owner = np.repeat(np.arange(len(groups)),
                      [len(g) for g in groups])
    member = np.concatenate(groups)

    batched = average_axial_grouped(vectors, owner, member, len(groups))
    for i, g in enumerate(groups):
        assert np.allclose(batched[i], average_axial(vectors[g]), atol=1e-10)


def test_a_group_that_caught_nothing_gets_no_direction():
    """A zero structure tensor has arbitrary eigenvectors.

    Inventing a direction from numerical noise would be worse than
    admitting there is none.
    """
    vectors = np.array([[1.0, 0.0, 0.0]])
    out = average_axial_grouped(vectors, np.array([0]), np.array([0]), 3)
    assert np.isclose(np.linalg.norm(out[0]), 1.0)
    assert np.allclose(out[1], 0.0) and np.allclose(out[2], 0.0)


# ------------------------------------------------------------- the remesh
@pytest.fixture(scope="module")
def source() -> pv.UnstructuredGrid:
    return _block()


def test_a_frozen_boundary_does_not_move(source):
    """Exact, not approximate. This is why it is the default."""
    before = source.extract_surface(algorithm="dataset_surface")
    out = remesh(source, RemeshOptions(target_edge=0.6, freeze_boundary=True))
    after = out.extract_surface(algorithm="dataset_surface")

    from scipy.spatial import cKDTree
    d, _ = cKDTree(np.asarray(after.points)).query(np.asarray(before.points))
    assert d.max() < 1e-9
    assert after.area == pytest.approx(before.area, rel=1e-6)


def test_the_result_is_a_valid_volume(source):
    out = remesh(source, RemeshOptions(target_edge=0.6))
    assert vm.kind_of(out) == vm.VOLUME
    vm.validate_tetrahedral(out)
    assert set(np.unique(out.celltypes).tolist()) == {vm.TETRA}
    assert vm.inverted_count(out) == 0
    assert out.n_cells > 0


def test_every_field_comes_back(source):
    out = remesh(source, RemeshOptions(target_edge=0.6))
    assert set(out.cell_data.keys()) >= {"elemTag", "fiber"}
    assert "scar_probability" in out.point_data
    assert np.asarray(out.cell_data["fiber"]).shape == (out.n_cells, 3)


def test_labels_are_copied_not_averaged(source):
    """1 and 2 must not produce a 1.5."""
    out = remesh(source, RemeshOptions(target_edge=0.6))
    assert set(np.unique(out.cell_data["elemTag"]).tolist()) == {1, 2}


def test_fibres_stay_unit_and_keep_their_direction(source):
    """The sign-flipped fixture: every element still points along x."""
    out = remesh(source, RemeshOptions(target_edge=0.6))
    fibre = np.asarray(out.cell_data["fiber"])
    assert np.allclose(np.linalg.norm(fibre, axis=1), 1.0, atol=1e-6)
    # |x| component 1 means the direction survived; the naive average
    # would have produced zeros here.
    assert np.allclose(np.abs(fibre[:, 0]), 1.0, atol=1e-6)


def test_point_fields_stay_in_range(source):
    """Interpolation, so no new extremes and no invented no-data."""
    out = remesh(source, RemeshOptions(target_edge=0.6))
    got = np.asarray(out.point_data["scar_probability"])
    assert np.isfinite(got).all()
    assert got.min() >= -1e-6
    assert got.max() <= 1.0 + 1e-6


def test_the_source_is_not_modified(source):
    cells_before = source.n_cells
    tags_before = np.array(source.cell_data["elemTag"], copy=True)
    remesh(source, RemeshOptions(target_edge=0.6))
    assert source.n_cells == cells_before
    assert np.array_equal(np.asarray(source.cell_data["elemTag"]), tags_before)


# ------------------------------------------------------------- the options
@pytest.mark.parametrize("options", [
    RemeshOptions(target_edge=1.0, min_edge=0.5),      # MMG refuses both
    RemeshOptions(target_edge=1.0, max_edge=2.0),
    RemeshOptions(min_edge=2.0, max_edge=1.0),         # inverted band
    RemeshOptions(target_edge=-1.0),
])
def test_impossible_options_are_refused_before_the_call(source, options):
    with pytest.raises(ValueError):
        remesh(source, options)


def test_only_what_was_asked_for_reaches_mmg():
    """Zero means "leave it to MMG", not "set it to zero"."""
    sent = RemeshOptions(target_edge=1.5).as_mmg_options()
    assert sent["hsiz"] == 1.5
    assert sent["nosurf"] is True
    assert "hmin" not in sent and "hmax" not in sent
    assert "hausd" not in sent and "hgrad" not in sent

    adapting = RemeshOptions(freeze_boundary=False, hausdorff=0.4)
    assert "nosurf" not in adapting.as_mmg_options()
    assert adapting.as_mmg_options()["hausd"] == 0.4


# ------------------------------------------------------------ references
def test_tags_survive_a_labelling_that_does_not_start_at_one():
    """MMG references are positive integers; a labelling need not be.

    A mesh tagged 0 and 7 is ordinary, and passing those straight through
    would lose the 0 outright.
    """
    tags = np.array([0, 7, 0, 7, 7], dtype=np.int32)
    codes, values = encode_tags(tags)
    assert codes.min() >= 1
    assert set(codes.tolist()) == {1, 2}
    assert np.array_equal(decode_tags(codes, values), tags)
    assert decode_tags(codes, values).dtype == tags.dtype


def test_an_invented_reference_is_refused_not_guessed():
    """Silently mislabelling a simulation input is the worst outcome."""
    _, values = encode_tags(np.array([1, 2]))
    with pytest.raises(RuntimeError, match="outside the range"):
        decode_tags(np.array([1, 2, 3]), values)


def test_elemtag_comes_back_exact_on_an_odd_labelling(source):
    """The whole point of carrying it as a reference."""
    odd = source.copy(deep=True)
    tags = np.asarray(odd.cell_data["elemTag"])
    odd.cell_data["elemTag"] = np.where(tags == 1, 0, 7).astype(np.int32)

    out = remesh(odd, RemeshOptions(target_edge=0.6))
    assert set(np.unique(out.cell_data["elemTag"]).tolist()) == {0, 7}


# --------------------------------------------------- transfer on its own
def test_transfer_leaves_the_destination_a_volume(source):
    """The transfer writes fields and touches no geometry."""
    dst = _block(n=5, size=6.0)
    for name in list(dst.cell_data.keys()):
        dst.cell_data.remove(name)
    for name in list(dst.point_data.keys()):
        dst.point_data.remove(name)
    cells, points = dst.n_cells, dst.n_points

    transfer_volume_fields(source, dst)
    assert dst.n_cells == cells and dst.n_points == points
    assert set(np.unique(dst.cell_data["elemTag"]).tolist()) <= {1, 2}
    assert np.isfinite(np.asarray(dst.point_data["scar_probability"])).all()
