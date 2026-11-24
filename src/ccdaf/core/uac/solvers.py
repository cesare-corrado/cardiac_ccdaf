"""
solvers
=======
Direct solvers for the symmetric positive definite systems of the atrial
coordinates, picked automatically.

The template warp solves one sparse system per iteration, about 200 per
atrium, so the solver sets the run time. In order of preference:

1. **CHOLMOD** (``scikit-sparse``): a Cholesky factorisation, made for
   exactly this kind of matrix. On a 143 000-unknown system it took 0.3 s,
   0.17 s when the symbolic analysis of an earlier system with the same
   pattern is reused.
2. **PARDISO** (``pypardiso``, Intel MKL): 0.45 s on the same system;
   Intel/AMD processors only.
3. **SciPy** (SuperLU): always present; 1 to 4 s depending on how SciPy
   was built.

The first one that imports is used, and its name is reported, so a run
never depends on the user choosing. All three give the same answer to
round-off (measured differences below 1e-12).

CHOLMOD does its dense work in the BLAS library, which by default starts
one thread per logical core. On these systems that is ruinous: with 24
threads a solve took 5.2 s instead of 0.2 s, because the dense blocks are
too small to share out. More than one thread gains almost nothing (0.216 s
with 1, 0.201 s with 8), so the BLAS runs single-threaded around every
CHOLMOD call (:data:`BLAS_THREADS`), set with ``threadpoolctl``; that also
keeps several runs in parallel from fighting over the cores. Without
``threadpoolctl`` the cap cannot be set, and CHOLMOD is skipped.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
import scipy.sparse as sp


#: BLAS threads during a CHOLMOD factorisation. On a 143 000-unknown
#: system: 1 thread 0.216 s, 2 0.210 s, 4 0.203 s, 8 0.201 s, 24 (one per
#: logical core, the default) 5.2 s.
BLAS_THREADS = 1


def _cholmod() -> Optional[Callable]:
    try:
        import threadpoolctl  # noqa: F401  (needed to cap the BLAS threads)
        from sksparse.cholmod import cho_factor
    except Exception:
        return None
    return cho_factor


def _blas_cap():
    from threadpoolctl import threadpool_limits
    return threadpool_limits(limits=BLAS_THREADS, user_api="blas")


def _pardiso():
    try:
        import pypardiso
    except Exception:
        return None
    return pypardiso


class SpdSolver:
    """Solve ``A x = b`` for a sequence of SPD matrices.

    With CHOLMOD, the fill-reducing ordering and symbolic factorisation are
    kept and reused while the sparsity pattern stays the same, so only the
    numeric factorisation is repeated.
    """

    def __init__(self, prefer: Optional[str] = None) -> None:
        order = ("cholmod", "pardiso", "scipy")
        if prefer is not None:
            order = (prefer,) + tuple(x for x in order if x != prefer)
        self.name = "scipy"
        self._factor = None
        self._pattern: Optional[Tuple[int, int, int]] = None
        for name in order:
            if name == "cholmod" and _cholmod() is not None:
                self.name = name
                break
            if name == "pardiso" and _pardiso() is not None:
                self.name = name
                break
            if name == "scipy":
                self.name = name
                break

    def solve(self, A: sp.spmatrix, b: np.ndarray) -> np.ndarray:
        if self.name == "cholmod":
            return self._solve_cholmod(A, b)
        if self.name == "pardiso":
            return _pardiso().spsolve(sp.csr_matrix(A), b)
        from scipy.sparse.linalg import spsolve
        return spsolve(sp.csc_matrix(A), b)

    def _solve_cholmod(self, A, b):
        A = sp.csc_matrix(A)
        A.sort_indices()
        pattern = (A.shape[0], A.nnz, hash(A.indices.tobytes()) ^ hash(A.indptr.tobytes()))
        with _blas_cap():
            if self._factor is not None and pattern == self._pattern:
                self._factor.factorize(A)
            else:
                self._factor = _cholmod()(A)
                self._pattern = pattern
            return self._factor.solve(b)


def available() -> Tuple[str, ...]:
    """Names of the solvers that can be imported, fastest first."""
    out = []
    if _cholmod() is not None:
        out.append("cholmod")
    if _pardiso() is not None:
        out.append("pardiso")
    out.append("scipy")
    return tuple(out)


__all__ = ["SpdSolver", "available"]
