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

This is a true Dirichlet condition on both the 0 and the 1 sets. The
openCARP pipeline that produced the reference fields does the same: its
Laplace mode holds the 1 set with an extracellular-voltage stimulus and the
0 set with an extracellular ground, and openCARP imposes both as Dirichlet
conditions on the matrix.

A consequence worth using: the three transmural problems hold the *same*
nodes (epicardium, LV and RV endocardium), so after the modification they
share one matrix and differ only in the right-hand side.
:class:`DirichletSystem` builds that matrix and its preconditioner once.

Solver
------
Conjugate gradients, preconditioned by algebraic multigrid (smoothed
aggregation, from ``pyamg``) by default. The apicobasal problem holds a
small apex patch against a large base, which makes it badly conditioned:
on a 6.2-million-node ventricle, Jacobi-preconditioned CG took 2885
iterations and 45 minutes, multigrid 36 iterations and under 5. Jacobi is
kept as an option, for comparison.

Memory
------
Assembly works through the elements in chunks of :data:`ASSEMBLY_CHUNK`.
Done all at once, it holds 16 entries per element before the duplicates
are summed: 28 GB of temporary arrays for a 1.1 GB matrix on 35 million
elements. The Dirichlet modification is one pass over the stored entries,
not a product of sparse matrices: on the same mesh that took 34 s and
2.5 GB of temporaries.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import time

import numpy as np
import pyamg
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, cg

#: The 3x4 operator taking nodal values to differences from node 0.
_LOCAL_GRAD = np.array([[-1.0, 1.0, 0.0, 0.0],
                        [-1.0, 0.0, 1.0, 0.0],
                        [-1.0, 0.0, 0.0, 1.0]])

#: Relative residual at which conjugate gradients stops.
#:
#: Tighter than the 1e-7 Bayer et al. report, because the residual is
#: measured against a right-hand side dominated by the held rows, so it is
#: loose on the free nodes. On a 2.9-million-element ventricle, against a
#: solution converged to 1e-12: at 1e-7, fields were off by up to 1e-5 and
#: 100 to 200 fibres by more than 1 degree, in elements whose frame is
#: close to undefined; at 1e-9, fields were off by 3e-7 and no fibre by
#: more than 0.1 degree. With multigrid, 1e-9 costs about 10% more time.
DEFAULT_RTOL: float = 1e-9

#: Preconditioners :func:`solve` accepts.
AMG: str = "amg"
JACOBI: str = "jacobi"
PRECONDITIONERS: Tuple[str, ...] = (AMG, JACOBI)
DEFAULT_PRECONDITIONER: str = AMG

#: Elements assembled at a time. Large enough that the per-chunk overhead
#: is negligible, small enough that a chunk's temporaries (about 200 bytes
#: an element) stay well under the size of the matrix being built.
ASSEMBLY_CHUNK: int = 1_000_000


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


def stiffness_matrix(points: np.ndarray, tets: np.ndarray, *,
                     chunk: int = ASSEMBLY_CHUNK) -> sp.csr_matrix:
    """The assembled stiffness matrix for isotropic unit conductivity.

    Unit conductivity because every problem here is a pure Laplace
    problem; the reference pipeline uses conductivity 1 in all directions
    for the same reason.

    The elements are assembled *chunk* at a time, each chunk summed into
    the matrix before the next is built (see the module docstring).
    """
    tets = np.asarray(tets, dtype=np.int64)
    n = int(np.asarray(points).shape[0])
    index = np.int32 if n < np.iinfo(np.int32).max else np.int64
    total: Optional[sp.csr_matrix] = None
    for start in range(0, max(len(tets), 1), max(int(chunk), 1)):
        part_tets = tets[start:start + chunk]
        grads, volume = basis_gradients(points, part_tets)
        local = volume[:, None, None] * np.einsum("nki,nkj->nij", grads, grads)
        del grads, volume
        rows = np.repeat(part_tets, 4, axis=1).ravel().astype(index)
        cols = np.tile(part_tets, (1, 4)).ravel().astype(index)
        part = sp.csr_matrix((local.ravel(), (rows, cols)), shape=(n, n))
        del local, rows, cols
        total = part if total is None else total + part
    return total.tocsr()


def impose_dirichlet(matrix: sp.csr_matrix, rhs: np.ndarray,
                     nodes: np.ndarray, values: np.ndarray
                     ) -> Tuple[sp.csr_matrix, np.ndarray]:
    """Hold ``nodes`` at ``values``, keeping the system symmetric.

    Row and column ``i`` are zeroed except the diagonal; ``b_i`` becomes
    ``a_ii * phi_i``; every other ``b_j`` loses ``a_ji * phi_i``. Neither
    input is modified.
    """
    a = held_matrix(matrix, nodes)
    return a, held_rhs(matrix, rhs, nodes, values)


def _held_mask(n: int, nodes: np.ndarray) -> np.ndarray:
    held = np.zeros(n, dtype=bool)
    held[np.asarray(nodes, dtype=np.int64)] = True
    return held


def held_matrix(matrix: sp.csr_matrix, nodes: np.ndarray) -> sp.csr_matrix:
    """*matrix* with the rows and columns of *nodes* zeroed but the diagonal.

    One pass over the stored entries, then the zeroed ones are dropped, so
    a held row stores its diagonal alone. *matrix* is not modified, beyond
    being put in canonical form (sorted, no duplicates) if it was not.
    """
    m = matrix if sp.isspmatrix_csr(matrix) else sp.csr_matrix(matrix)
    m.sum_duplicates()
    n = m.shape[0]
    held = _held_mask(n, nodes)
    diagonal = m.diagonal()
    if np.any(diagonal[held] <= 0.0):
        raise ValueError("a held node has no positive diagonal entry; it is "
                         "not connected to any element")
    row = np.repeat(np.arange(n, dtype=m.indices.dtype), np.diff(m.indptr))
    hit = (held[row] | held[m.indices]) & (row != m.indices)
    del row
    data = m.data.copy()
    data[hit] = 0.0
    del hit
    out = sp.csr_matrix((data, m.indices.copy(), m.indptr.copy()), shape=m.shape,
                        copy=False)
    out.eliminate_zeros()
    return out


def held_rhs(matrix: sp.csr_matrix, rhs: np.ndarray, nodes: np.ndarray,
             values: np.ndarray) -> np.ndarray:
    """The right-hand side that goes with :func:`held_matrix`.

    *matrix* is the **unmodified** one: the held columns it still has are
    what moves to the right-hand side.
    """
    nodes = np.asarray(nodes, dtype=np.int64)
    lifted = np.zeros(matrix.shape[0])
    lifted[nodes] = np.asarray(values, dtype=float)
    b = np.asarray(rhs, dtype=float) - matrix @ lifted
    b[nodes] = matrix.diagonal()[nodes] * lifted[nodes]
    return b


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
    #: The preconditioner used, one of :data:`PRECONDITIONERS`.
    preconditioner: str = JACOBI
    #: Seconds spent building the preconditioner. A preconditioner shared
    #: by several solves is charged to the first.
    setup_seconds: float = 0.0
    #: Seconds spent in conjugate gradients.
    solve_seconds: float = 0.0


def preconditioner_for(matrix: sp.csr_matrix,
                       kind: str = DEFAULT_PRECONDITIONER) -> LinearOperator:
    """The preconditioner *kind* for *matrix*, as a linear operator."""
    if kind == AMG:
        hierarchy = pyamg.smoothed_aggregation_solver(
            matrix, symmetry="symmetric", max_coarse=500)
        return hierarchy.aspreconditioner(cycle="V")
    if kind == JACOBI:
        diagonal = matrix.diagonal()
        inverse = np.where(diagonal > 0.0,
                           1.0 / np.where(diagonal > 0.0, diagonal, 1.0), 0.0)
        return LinearOperator(matrix.shape, matvec=lambda v: inverse * v)
    raise ValueError(f"unknown preconditioner {kind!r}; "
                     f"expected one of {', '.join(PRECONDITIONERS)}")


def solve(matrix: sp.csr_matrix, rhs: np.ndarray, *,
          rtol: float = DEFAULT_RTOL, maxiter: Optional[int] = None,
          x0: Optional[np.ndarray] = None,
          preconditioner: str = DEFAULT_PRECONDITIONER,
          operator: Optional[LinearOperator] = None
          ) -> Tuple[np.ndarray, SolveReport]:
    """Preconditioned conjugate gradients.

    *operator* is a preconditioner already built for *matrix* (see
    :func:`preconditioner_for`); given, *preconditioner* only names it in
    the report and no setup is done.
    """
    setup = 0.0
    if operator is None:
        t0 = time.perf_counter()
        operator = preconditioner_for(matrix, preconditioner)
        setup = time.perf_counter() - t0

    count = {"n": 0}

    def _tick(_x):
        count["n"] += 1

    t0 = time.perf_counter()
    x, info = cg(matrix, rhs, x0=x0, rtol=rtol, maxiter=maxiter,
                 M=operator, callback=_tick)
    seconds = time.perf_counter() - t0
    norm_b = float(np.linalg.norm(rhs)) or 1.0
    residual = float(np.linalg.norm(rhs - matrix @ x)) / norm_b
    held = int(np.sum(matrix.getnnz(axis=1) == 1))
    return x, SolveReport(iterations=count["n"], converged=(info == 0),
                          relative_residual=residual, held_nodes=held,
                          free_nodes=matrix.shape[0] - held,
                          preconditioner=preconditioner, setup_seconds=setup,
                          solve_seconds=seconds)


class DirichletSystem:
    """One stiffness matrix with one set of nodes held, ready to solve.

    Problems that hold the same nodes, at whatever values, share the
    modified matrix and its preconditioner: they are built here once, and
    :meth:`solve` only forms a right-hand side and runs CG. The three
    transmural problems are such a family.
    """

    def __init__(self, matrix: sp.csr_matrix, nodes: np.ndarray, *,
                 preconditioner: str = DEFAULT_PRECONDITIONER):
        self.nodes = np.unique(np.asarray(nodes, dtype=np.int64))
        self.matrix = matrix
        self.held = held_matrix(matrix, self.nodes)
        self.preconditioner = preconditioner
        t0 = time.perf_counter()
        self._operator = preconditioner_for(self.held, preconditioner)
        self._setup = time.perf_counter() - t0
        self._held_count = len(self.nodes)

    def solve(self, values: np.ndarray, *, rtol: float = DEFAULT_RTOL
              ) -> Tuple[np.ndarray, SolveReport]:
        """The solution with :attr:`nodes` held at *values* (same order)."""
        values = np.asarray(values, dtype=float)
        if values.shape != self.nodes.shape:
            raise ValueError("one value is needed per held node")
        b = held_rhs(self.matrix, np.zeros(self.matrix.shape[0]),
                     self.nodes, values)
        phi, report = solve(self.held, b, rtol=rtol,
                            preconditioner=self.preconditioner,
                            operator=self._operator)
        report.setup_seconds, self._setup = self._setup, 0.0
        report.held_nodes = self._held_count
        report.free_nodes = self.matrix.shape[0] - self._held_count
        # The held rows are decoupled, so their exact solution is known; CG
        # only approaches it, because its step length is shared by every
        # component. Writing the values back makes a held value exact
        # rather than exact to the solver tolerance.
        phi[self.nodes] = values
        return phi, report


def _check_sets(ones: np.ndarray, zeros: np.ndarray
                ) -> Tuple[np.ndarray, np.ndarray]:
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
    return ones, zeros


def _values(nodes: np.ndarray, ones: np.ndarray) -> np.ndarray:
    """1 on the nodes of *ones*, 0 on the others, in the order of *nodes*."""
    return np.isin(nodes, ones).astype(float)


def laplace(points: np.ndarray, tets: np.ndarray,
            ones: np.ndarray, zeros: np.ndarray, *,
            matrix: Optional[sp.csr_matrix] = None,
            rtol: float = DEFAULT_RTOL,
            preconditioner: str = DEFAULT_PRECONDITIONER
            ) -> Tuple[np.ndarray, SolveReport]:
    """Solve Laplace's equation with 1 on ``ones`` and 0 on ``zeros``.

    ``matrix`` may be passed to reuse an assembly. Raises ``ValueError``
    when a node is asked to be both 1 and 0, which on a labelled boundary
    means two surfaces touch at that node and one solve cannot honour both.
    """
    ones, zeros = _check_sets(ones, zeros)
    k = matrix if matrix is not None else stiffness_matrix(points, tets)
    system = DirichletSystem(k, np.concatenate([ones, zeros]),
                             preconditioner=preconditioner)
    return system.solve(_values(system.nodes, ones), rtol=rtol)


def laplace_family(matrix: sp.csr_matrix,
                   problems: Dict[str, Tuple[np.ndarray, np.ndarray]], *,
                   rtol: float = DEFAULT_RTOL,
                   preconditioner: str = DEFAULT_PRECONDITIONER,
                   on_start: Optional[Callable[[str], None]] = None
                   ) -> Tuple[Dict[str, np.ndarray], Dict[str, SolveReport]]:
    """Solve several problems ``{key: (ones, zeros)}`` that hold the same
    nodes, sharing one modified matrix and one preconditioner.

    Raises ``ValueError`` if the problems do not hold the same nodes, or
    one of them holds a node at 1 and at 0. *on_start(key)* is called
    before each solve.
    """
    checked = {k: _check_sets(o, z) for k, (o, z) in problems.items()}
    held = [np.union1d(o, z) for o, z in checked.values()]
    if any(not np.array_equal(held[0], h) for h in held[1:]):
        raise ValueError("the problems of a family must hold the same nodes")
    system = DirichletSystem(matrix, held[0], preconditioner=preconditioner)
    fields, reports = {}, {}
    for key, (ones, _zeros) in checked.items():
        if on_start is not None:
            on_start(key)
        fields[key], reports[key] = system.solve(_values(system.nodes, ones),
                                                 rtol=rtol)
    return fields, reports


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
    out, reports = laplace_family(k, {
        "epi": (epi, np.concatenate([lv, rv])),
        "lv": (lv, np.concatenate([epi, rv])),
        "rv": (rv, np.concatenate([epi, lv]))}, rtol=rtol)
    return TransmuralFields(epi=out["epi"], lv=out["lv"], rv=out["rv"],
                            reports=reports)


__all__ = ["stiffness_matrix", "impose_dirichlet", "held_matrix", "held_rhs",
           "preconditioner_for", "solve", "DirichletSystem", "laplace",
           "laplace_family", "element_gradients", "basis_gradients",
           "transmural", "TransmuralFields", "SolveReport", "DEFAULT_RTOL",
           "AMG", "JACOBI", "PRECONDITIONERS", "DEFAULT_PRECONDITIONER",
           "ASSEMBLY_CHUNK"]
