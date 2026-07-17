"""CCDAF — Cardiac Clinical Data Analysis Framework."""
import os

__version__ = "0.1.0"


def _disable_vtk_accelerated_filters() -> None:
    """Stop VTK offloading a handful of filters to Viskores (ex VTK-m).

    VTK 9.6 ships accelerated overrides for a few filters and reaches for
    CUDA when it sees a GPU. Where the toolchain the PTX was compiled with
    does not match the installed driver, every such call fails
    (``cudaErrorUnsupportedPtxVersion``), prints a screenful of errors, and
    then silently redoes the work on the CPU. The answer is correct either
    way — it is the attempt that is wasted, and it is not cheap:
    cell-to-point on a 42k-cell Carto shell takes 0.427s with the overrides
    against 0.002s without, for the identical result.

    Nothing here is big enough to want a GPU anyway: 2ms of CPU work does not
    survive the round trip. So the CPU path is what has actually been running
    all along, and this only stops us paying to discover that. Set
    ``CCDAF_VTK_ACCEL=1`` to leave the overrides on.

    Not every VTK build has Viskores, hence the guard.
    """
    if os.environ.get("CCDAF_VTK_ACCEL") == "1":
        return
    try:
        import vtk

        vtk.vtkmFilterOverrides.SetEnabled(False)
    except Exception:
        pass


_disable_vtk_accelerated_filters()
