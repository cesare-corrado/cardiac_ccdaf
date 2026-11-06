"""
test_laplace.py
===============
Laplace's equation with linear tetrahedra and symmetric Dirichlet conditions.

The contract:

* linear elements reproduce a linear solution **exactly**, so a slab held
  at 0 on one face and 1 on the opposite face, free elsewhere, must come
  back as x / L to solver tolerance, and its gradient as (1/L, 0, 0) on
  every element;
* the Dirichlet modification keeps the matrix **symmetric**, which is what
  makes conjugate gradients legitimate, and reproduces every held value
  exactly;
* a node asked to be 1 and 0 in the same solve is refused, because on a
  labelled boundary that means two surfaces touch there;
* the three transmural fields sum to 1 everywhere (Bayer et al., equation
  5), a check that needs no reference solution at all;
* assembly in chunks gives the same matrix as assembly in one go, and the
  one-pass Dirichlet modification the same matrix as zeroing rows and
  columns with sparse products;
* the multigrid and Jacobi preconditioners give the same solution, the
  report names the one used, and an unknown one is refused;
* problems holding the same nodes share one matrix and preconditioner,
  and give what separate solves give; a family holding different nodes is
  refused.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pytest
import pyvista as pv

from ccdaf.core import volume_mesh as vm
import scipy.sparse as sp

from ccdaf.core.laplace import (
    AMG, JACOBI, DirichletSystem, element_gradients, held_matrix,
    impose_dirichlet, laplace, laplace_family, stiffness_matrix, transmural,
)

L = 4.0


def _slab(n=9):
    """A tetrahedralised box, L long in x."""
    grid = pv.ImageData(dimensions=(n, 5, 5),
                        spacing=(L / (n - 1), 0.5, 0.5)).cast_to_unstructured_grid()
    grid = grid.triangulate()
    return np.asarray(grid.points), vm.tetrahedra(grid)


def test_the_stiffness_matrix_is_symmetric_with_zero_row_sums():
    """Row sums vanish because a constant field has no gradient."""
    pts, tets = _slab()
    k = stiffness_matrix(pts, tets)
    assert abs(k - k.T).max() < 1e-12
    assert np.allclose(np.asarray(k.sum(axis=1)).ravel(), 0.0, atol=1e-10)


def test_dirichlet_keeps_the_matrix_symmetric_and_the_values_exact():
    pts, tets = _slab()
    k = stiffness_matrix(pts, tets)
    held = np.array([0, 5, 17, 40])
    values = np.array([0.0, 1.0, 0.25, 0.75])
    a, b = impose_dirichlet(k, np.zeros(k.shape[0]), held, values)
    assert abs(a - a.T).max() < 1e-12
    # Held rows are the identity scaled by the diagonal.
    for i, v in zip(held, values):
        row = a.getrow(i)
        assert row.nnz == 1 and row.indices[0] == i
        assert np.isclose(b[i], a[i, i] * v)


def test_a_slab_reproduces_the_exact_linear_solution():
    """The strongest test available: no discretisation error at all."""
    pts, tets = _slab()
    x = pts[:, 0]
    zeros = np.where(np.isclose(x, 0.0))[0]
    ones = np.where(np.isclose(x, L))[0]
    phi, report = laplace(pts, tets, ones, zeros)
    assert report.converged
    assert np.max(np.abs(phi - x / L)) < 1e-6

    grad = element_gradients(pts, tets, phi)
    assert np.allclose(grad, [1.0 / L, 0.0, 0.0], atol=1e-5)


def test_held_values_come_back_exactly():
    pts, tets = _slab()
    x = pts[:, 0]
    zeros = np.where(np.isclose(x, 0.0))[0]
    ones = np.where(np.isclose(x, L))[0]
    phi, _ = laplace(pts, tets, ones, zeros)
    assert np.array_equal(phi[ones], np.ones(len(ones)))
    assert np.array_equal(phi[zeros], np.zeros(len(zeros)))


def test_a_node_held_at_both_values_is_refused():
    pts, tets = _slab()
    with pytest.raises(ValueError, match="held at 1 and at 0"):
        laplace(pts, tets, np.array([0, 1]), np.array([1, 2]))


def test_an_empty_set_is_refused():
    pts, tets = _slab()
    with pytest.raises(ValueError, match="must hold nodes"):
        laplace(pts, tets, np.array([], dtype=int), np.array([0]))


def test_the_transmural_fields_sum_to_one():
    """Bayer et al. equation 5, on three faces of a slab standing in for
    epicardium, LV and RV endocardium."""
    pts, tets = _slab()
    x, y = pts[:, 0], pts[:, 1]
    epi = np.where(np.isclose(x, 0.0))[0]
    lv = np.where(np.isclose(x, L))[0]
    rv = np.where(np.isclose(y, 0.0) & (x > 0.0) & (x < L))[0]
    fields = transmural(pts, tets, epi, lv, rv)
    assert all(r.converged for r in fields.reports.values())
    assert fields.conservation_error() < 1e-5
    for phi in (fields.epi, fields.lv, fields.rv):
        assert phi.min() > -1e-6 and phi.max() < 1.0 + 1e-6


def test_a_degenerate_element_is_refused():
    pts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=float)
    with pytest.raises(ValueError, match="degenerate"):
        stiffness_matrix(pts, np.array([[0, 1, 2, 3]]))


# ---------------------------------------------------------------------
# Assembly, Dirichlet modification, preconditioners
# ---------------------------------------------------------------------
def _faces():
    pts, tets = _slab()
    x = pts[:, 0]
    return pts, tets, np.where(np.isclose(x, L))[0], np.where(np.isclose(x, 0.0))[0]


@pytest.mark.parametrize("chunk", [1, 7, 100])
def test_chunked_assembly_matches_one_go(chunk):
    pts, tets = _slab()
    whole = stiffness_matrix(pts, tets, chunk=len(tets))
    part = stiffness_matrix(pts, tets, chunk=chunk)
    assert part.nnz == whole.nnz
    assert abs(part - whole).max() < 1e-12


def test_one_pass_dirichlet_matches_sparse_products():
    """The reference is the old construction: zero the held rows and
    columns with diagonal-matrix products, then put the diagonal back."""
    pts, tets = _slab()
    k = stiffness_matrix(pts, tets)
    held = np.array([0, 5, 17, 40, 41])
    mask = np.zeros(k.shape[0], dtype=bool)
    mask[held] = True
    keep = sp.diags((~mask).astype(float))
    ref = keep @ k @ keep + sp.diags(np.where(mask, k.diagonal(), 0.0))
    before = k.copy()
    a = held_matrix(k, held)
    assert abs(a - ref).max() == 0.0
    assert a.nnz == sp.csr_matrix(ref).count_nonzero()
    assert abs(k - before).max() == 0.0          # the input is untouched


@pytest.mark.parametrize("kind", [AMG, JACOBI])
def test_both_preconditioners_reproduce_the_linear_solution(kind):
    pts, tets, ones, zeros = _faces()
    phi, report = laplace(pts, tets, ones, zeros, preconditioner=kind)
    assert report.converged and report.preconditioner == kind
    assert report.held_nodes == len(ones) + len(zeros)
    assert np.allclose(phi, pts[:, 0] / L, atol=1e-6)


def test_an_unknown_preconditioner_is_refused():
    pts, tets, ones, zeros = _faces()
    with pytest.raises(ValueError, match="unknown preconditioner"):
        laplace(pts, tets, ones, zeros, preconditioner="ilu")


def test_a_family_matches_separate_solves_and_builds_once():
    pts, tets = _slab()
    x, y = pts[:, 0], pts[:, 1]
    a = np.where(np.isclose(x, 0.0))[0]
    b = np.where(np.isclose(x, L))[0]
    c = np.where(np.isclose(y, 0.0) & (x > 0.0) & (x < L))[0]
    k = stiffness_matrix(pts, tets)
    problems = {"a": (a, np.concatenate([b, c])), "b": (b, np.concatenate([a, c]))}
    started = []
    fields, reports = laplace_family(k, problems, on_start=started.append)
    assert started == ["a", "b"]
    for key, (ones, zeros) in problems.items():
        alone, _ = laplace(pts, tets, ones, zeros, matrix=k)
        assert np.allclose(fields[key], alone, atol=1e-6)
    # The preconditioner is built once, and its cost charged to the first.
    assert reports["b"].setup_seconds == 0.0


def test_a_family_holding_different_nodes_is_refused():
    pts, tets, ones, zeros = _faces()
    k = stiffness_matrix(pts, tets)
    with pytest.raises(ValueError, match="same nodes"):
        laplace_family(k, {"a": (ones, zeros), "b": (ones, zeros[:-1])})


def test_a_system_needs_one_value_per_held_node():
    pts, tets, ones, zeros = _faces()
    system = DirichletSystem(stiffness_matrix(pts, tets), np.concatenate([ones, zeros]))
    with pytest.raises(ValueError, match="one value"):
        system.solve(np.ones(3))
