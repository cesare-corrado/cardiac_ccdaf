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
  5), a check that needs no reference solution at all.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pytest
import pyvista as pv

from ccdaf.core import volume_mesh as vm
from ccdaf.core.laplace import (
    element_gradients, impose_dirichlet, laplace, stiffness_matrix,
    transmural,
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
