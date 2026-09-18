"""
laplace
=======
Laplace's equation on a tetrahedral volume, with linear elements.

This is the first stage of the Laplace-Dirichlet rule-based fibre method
(Bayer et al., Ann Biomed Eng 2012). Four problems are solved on the
myocardium, each with the value 1 on one surface and 0 on others:

* ``phi_epi``: 1 on the epicardium, 0 on both endocardia;
* ``phi_lv``: 1 on the LV endocardium, 0 on the epicardium and RV endocardium;
* ``phi_rv``: 1 on the RV endocardium, 0 on the epicardium and LV endocardium;
* ``psi_ab``: 1 on the base, 0 at the apex.

Everywhere else the boundary is left free, which in the weak form is the
zero-flux condition and costs no code at all.

Elements
--------
Linear tetrahedra. With ``J`` the matrix whose rows are the three edge
vectors from node 0, a linear function's gradient is
``inv(J) @ L @ u`` for ``L`` the fixed 3x4 difference operator, so the
element stiffness is ``V * G^T G`` with ``G = inv(J) @ L``. The gradient is
constant over each element, which is why the fibre stage can work per
element without any recovery step.

Dirichlet conditions
--------------------
Imposed symmetrically on the assembled system, as the user specified: for a
node ``i`` held at ``phi_i``, every other entry of row ``i`` and column
``i`` is set to zero, the right-hand side at ``i`` becomes ``a_ii * phi_i``,
and each ``b_j`` loses ``a_ji * phi_i``. The matrix stays symmetric and
positive definite, so conjugate gradients applies, and the held values come
back exactly.

This is a true Dirichlet condition on both the 0 and the 1 sets. openCARP,
which produced the reference fields this is validated against, imposes the
value 1 as a forcing term instead; the difference is confined near those
surfaces and is measured at the fibre level rather than assumed away.

A consequence worth using: the three transmural problems hold the *same*
nodes (epicardium, LV and RV endocardium), so after the modification they
share one matrix and differ only in the right-hand side.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, cg

#: The 3x4 operator taking nodal values to differences from node 0.
_LOCAL_GRAD = np.array([[-1.0, 1.0, 0.0, 0.0],
                        [-1.0, 0.0, 1.0, 0.0],
                        [-1.0, 0.0, 0.0, 1.0]])

#: Relative residual at which conjugate gradients stops. Bayer et al.
#: report 1e-7 as working well for these problems.
DEFAULT_RTOL: float = 1e-7


# ---------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------
def basis_gradients(points: np.ndarray, tets: np.ndarray
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """``(G, volume)``: per-element basis gradients and volumes.

    ``G`` has shape ``(n, 3, 4)``; column ``k`` is the gradient of the
    basis function of the element's ``k``-th node.

    Raises ``ValueError`` for a degenerate element, since its gradient
    does not exist and the clean step is the place to have removed it.
    """
    p = np.asarray(points, dtype=float)[np.asarray(tets, dtype=np.int64)]
    jac = p[:, 1:, :] - p[:, :1, :]                 # rows are edge vectors
    det = np.linalg.det(jac)
    if np.any(np.abs(det) < 1e-300):
        bad = int(np.sum(np.abs(det) < 1e-300))
        raise ValueError(f"{bad} degenerate tetrahedra have no gradient; "
                         f"run Clean volume first")
    grads = np.linalg.inv(jac) @ _LOCAL_GRAD
    return grads, np.abs(det) / 6.0


def stiffness_matrix(points: np.ndarray, tets: np.ndarray) -> sp.csr_matrix:
    """The assembled stiffness matrix for isotropic unit conductivity.

    Unit conductivity because every problem here is a pure Laplace
    problem; the reference pipeline uses conductivity 1 in all directions
    for the same reason.
    """
    tets = np.asarray(tets, dtype=np.int64)
    grads, volume = basis_gradients(points, tets)
    local = volume[:, None, None] * np.einsum("nki,nkj->nij", grads, grads)
    rows = np.repeat(tets, 4, axis=1).ravel()
    cols = np.tile(tets, (1, 4)).ravel()
    n = int(np.asarray(points).shape[0])
    return sp.coo_matrix((local.ravel(), (rows, cols)), shape=(n, n)).tocsr()


def impose_dirichlet(matrix: sp.csr_matrix, rhs: np.ndarray,
                     nodes: np.ndarray, values: np.ndarray
                     ) -> Tuple[sp.csr_matrix, np.ndarray]:
    """Hold ``nodes`` at ``values``, keeping the system symmetric.

    Row and column ``i`` are zeroed except the diagonal; ``b_i`` becomes
    ``a_ii * phi_i``; every other ``b_j`` loses ``a_ji * phi_i``. Neither
    input is modified.
    """
    nodes = np.asarray(nodes, dtype=np.int64)
    values = np.asarray(values, dtype=float)
    n = matrix.shape[0]
    held = np.zeros(n, dtype=bool)
    held[nodes] = True
    lifted = np.zeros(n)
    lifted[nodes] = values

    diagonal = matrix.diagonal()
    if np.any(diagonal[nodes] <= 0.0):
        raise ValueError("a held node has no positive diagonal entry; it is "
                         "not connected to any element")

    # Move the held columns to the right-hand side, then fix the held rows.
    b = np.asarray(rhs, dtype=float) - matrix @ lifted
    b[held] = diagonal[held] * lifted[held]

    keep = sp.diags((~held).astype(float))
    a = (keep @ matrix @ keep).tocsr()
    a = (a + sp.diags(np.where(held, diagonal, 0.0))).tocsr()
    return a, b


# ---------------------------------------------------------------------
# Solving
# ---------------------------------------------------------------------
@dataclass
class SolveReport:
    """What the linear solve did, for the record."""

    iterations: int
    converged: bool
    relative_residual: float
    held_nodes: int
    free_nodes: int


def solve(matrix: sp.csr_matrix, rhs: np.ndarray, *,
          rtol: float = DEFAULT_RTOL, maxiter: Optional[int] = None,
          x0: Optional[np.ndarray] = None
          ) -> Tuple[np.ndarray, SolveReport]:
    """Conjugate gradients with a Jacobi preconditioner.

    Jacobi rather than something stronger because it needs nothing but the
    diagonal, works on any mesh, and is enough for a Laplace problem with a
    tolerance of 1e-7: the cost is iterations, not correctness.
    """
    diagonal = matrix.diagonal()
    inverse = np.where(diagonal > 0.0, 1.0 / np.where(diagonal > 0.0, diagonal, 1.0), 0.0)
    preconditioner = LinearOperator(matrix.shape, matvec=lambda v: inverse * v)

    count = {"n": 0}

    def _tick(_x):
        count["n"] += 1

    x, info = cg(matrix, rhs, x0=x0, rtol=rtol, maxiter=maxiter,
                 M=preconditioner, callback=_tick)
    norm_b = float(np.linalg.norm(rhs)) or 1.0
    residual = float(np.linalg.norm(rhs - matrix @ x)) / norm_b
    held = int(np.sum(matrix.getnnz(axis=1) == 1))
    return x, SolveReport(iterations=count["n"], converged=(info == 0),
                          relative_residual=residual, held_nodes=held,
                          free_nodes=matrix.shape[0] - held)


def laplace(points: np.ndarray, tets: np.ndarray,
            ones: np.ndarray, zeros: np.ndarray, *,
            matrix: Optional[sp.csr_matrix] = None,
            rtol: float = DEFAULT_RTOL
            ) -> Tuple[np.ndarray, SolveReport]:
    """Solve Laplace's equation with 1 on ``ones`` and 0 on ``zeros``.

    ``matrix`` may be passed to reuse an assembly. Raises ``ValueError``
    when a node is asked to be both 1 and 0, which on a labelled boundary
    means two surfaces touch at that node and one solve cannot honour both.
    """
    ones = np.unique(np.asarray(ones, dtype=np.int64))
    zeros = np.unique(np.asarray(zeros, dtype=np.int64))
    if len(ones) == 0 or len(zeros) == 0:
        raise ValueError("both the 1 set and the 0 set must hold nodes, or "
                         "the problem is not determined")
    clash = np.intersect1d(ones, zeros)
    if len(clash):
        raise ValueError(
            f"{len(clash)} node(s) are held at 1 and at 0 in the same solve; "
            f"the surfaces touch there")
    k = matrix if matrix is not None else stiffness_matrix(points, tets)
    nodes = np.concatenate([ones, zeros])
    values = np.concatenate([np.ones(len(ones)), np.zeros(len(zeros))])
    a, b = impose_dirichlet(k, np.zeros(k.shape[0]), nodes, values)
    phi, report = solve(a, b, rtol=rtol)
    # The held rows are decoupled, so their exact solution is known; CG
    # only approaches it, because its step length is shared by every
    # component. Writing the values back makes a held value exact rather
    # than exact to the solver tolerance.
    phi[nodes] = values
    return phi, report


def element_gradients(points: np.ndarray, tets: np.ndarray,
                      values: np.ndarray) -> np.ndarray:
    """The gradient of a nodal field on every element, shape ``(n, 3)``.

    Exact for linear elements: the field is linear inside each one.
    """
    grads, _volume = basis_gradients(points, tets)
    u = np.asarray(values, dtype=float)[np.asarray(tets, dtype=np.int64)]
    return np.einsum("nij,nj->ni", grads, u)


@dataclass
class TransmuralFields:
    """The three transmural solutions, sharing one matrix."""

    epi: np.ndarray
    lv: np.ndarray
    rv: np.ndarray
    reports: Dict[str, SolveReport] = field(default_factory=dict)

    def conservation_error(self) -> float:
        """Largest ``|phi_epi + phi_lv + phi_rv - 1|`` over the nodes.

        The three sum to exactly 1 in the continuous problem (Bayer et al.,
        equation 5), so this is a correctness check that needs no
        reference solution: it should be of the order of the solver
        tolerance.
        """
        return float(np.max(np.abs(self.epi + self.lv + self.rv - 1.0)))


def transmural(points: np.ndarray, tets: np.ndarray, epi: np.ndarray,
               lv_endo: np.ndarray, rv_endo: np.ndarray, *,
               rtol: float = DEFAULT_RTOL) -> TransmuralFields:
    """``phi_epi``, ``phi_lv`` and ``phi_rv`` from one assembled matrix.

    The three problems hold the same nodes, so after the Dirichlet
    modification the matrix is the same for all three; only the values
    differ.
    """
    k = stiffness_matrix(points, tets)
    epi = np.unique(np.asarray(epi, dtype=np.int64))
    lv = np.unique(np.asarray(lv_endo, dtype=np.int64))
    rv = np.unique(np.asarray(rv_endo, dtype=np.int64))
    out = {}
    reports = {}
    for name, one, zero in (("epi", epi, np.concatenate([lv, rv])),
                            ("lv", lv, np.concatenate([epi, rv])),
                            ("rv", rv, np.concatenate([epi, lv]))):
        out[name], reports[name] = laplace(points, tets, one, zero,
                                           matrix=k, rtol=rtol)
    return TransmuralFields(epi=out["epi"], lv=out["lv"], rv=out["rv"],
                            reports=reports)


__all__ = ["stiffness_matrix", "impose_dirichlet", "solve", "laplace",
           "element_gradients", "basis_gradients", "transmural",
           "TransmuralFields", "SolveReport", "DEFAULT_RTOL"]
