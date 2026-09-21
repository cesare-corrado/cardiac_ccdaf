"""
test_postprocessing_panel.py
============================
GUI-level guards for the mesh post-processing panel:

* its controls carry tooltips;
* the panel's defaults agree with ``PostprocessOptions``, so what the user
  sees is what a script gets;
* the long steps (resampling, quality repair) report progress, so the panel
  is not silent for minutes on a dense mesh.

Runs headless: needs a QApplication with the offscreen platform (the
project's standard pytest invocation sets ``QT_QPA_PLATFORM=offscreen``).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pytest
import pyvista as pv
from PyQt5 import QtWidgets

from ccdaf.core.mesh_postprocessor import PostprocessOptions
from ccdaf.gui.postprocessing_widget import PostprocessingWidget


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def _sphere(theta=20, phi=20):
    s = pv.Sphere(radius=10.0, theta_resolution=theta,
                  phi_resolution=phi).triangulate()
    m = pv.PolyData(np.asarray(s.points), np.asarray(s.faces))
    m.cell_data["elemTag"] = np.ones(m.n_cells, dtype=np.int32)
    return m


def _panel(mesh=None, on_status=None):
    box = {"m": mesh if mesh is not None else _sphere()}
    return PostprocessingWidget(lambda: box["m"],
                                lambda m: box.__setitem__("m", m),
                                on_status=on_status), box


CONTROLS = ["chk_decimate", "spn_target", "spn_iters",
            "chk_refine", "cmb_refine_mode", "spn_edge",
            "chk_clean", "spn_quality", "spn_smooth", "spn_smooth_relax",
            "chk_smooth", "cmb_smooth"]


def test_controls_have_tooltips(qapp):
    panel, _ = _panel()
    missing = [n for n in CONTROLS if not getattr(panel, n).toolTip().strip()]
    assert not missing, "controls missing a tooltip: " + ", ".join(missing)


def test_panel_defaults_match_the_options(qapp):
    panel, _ = _panel()
    opts = PostprocessOptions()
    assert panel.cmb_refine_mode.currentData() == opts.refine_mode
    assert panel.spn_edge.value() == pytest.approx(opts.refine_edge_len)
    assert panel.spn_quality.value() == pytest.approx(
        opts.clean_quality_threshold)
    assert panel.spn_smooth_relax.value() == pytest.approx(
        opts.clean_quality_relaxation)
    assert panel.spn_smooth.value() == opts.clean_smooth_iterations


def test_both_refine_modes_are_offered(qapp):
    panel, _ = _panel()
    offered = [panel.cmb_refine_mode.itemData(i)
               for i in range(panel.cmb_refine_mode.count())]
    assert set(offered) == {"resample", "adaptive"}


def test_long_steps_report_progress(qapp):
    seen: list[str] = []
    panel, box = _panel(on_status=seen.append)
    panel.chk_refine.setChecked(True)
    panel.chk_clean.setChecked(True)
    for name in ("chk_decimate", "chk_fill", "chk_smooth"):
        getattr(panel, name).setChecked(False)
    panel.spn_edge.setValue(0.8)
    panel._on_apply()

    assert any("resample" in m for m in seen), (
        "resampling ran without reporting progress")
    assert any("quality repair" in m for m in seen), (
        "quality repair ran without reporting progress")
    assert not panel.progress.isVisible(), "progress bar left on screen"
    assert box["m"].n_points != _sphere().n_points, "apply did nothing"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


# --------------------------------------------- the volumetric panel order
def test_the_volume_panel_lists_its_actions_in_working_order():
    """Clean, then remesh, then improve: the order they must be run in.

    Not a style preference. A remesh carries every topological defect
    straight through, so cleaning afterwards repairs the same defects on
    a mesh tens of times larger — measured, 11 s against 9 minutes.
    """
    from ccdaf.gui.volume_postprocessing_widget import (
        VolumePostprocessingWidget)

    widget = VolumePostprocessingWidget()
    buttons = [b.text() for b in widget.findChildren(QtWidgets.QPushButton)]
    # "Check the wall" sits with the clean because it measures rather
    # than changes: it belongs beside the repair, not in the sequence.
    assert buttons == ["Clean volume", "Check the wall",
                       "Remesh volume", "Improve quality"]


def test_the_panel_never_asks_to_separate_touching_material():
    """The one repair kept off the panel, and why.

    Separating material that only touches is exact and the mesh it makes
    is better described. Its only measured effect on a real workflow was
    to break Actions → Label ventricular surfaces, which relies on
    non-manifold edges acting as accidental cuts in the boundary. It
    stays reachable from CleanOptions, and off whatever the panel does.
    """
    from ccdaf.core.volume_clean import CleanOptions
    from ccdaf.gui.volume_postprocessing_widget import (
        VolumePostprocessingWidget)

    widget = VolumePostprocessingWidget()
    assert not hasattr(widget, "chk_separate")
    assert not widget.clean_options().separate_touching
    # Still reachable, and still working, for anything that wants it.
    assert CleanOptions(separate_touching=True).repair_options().split_touching
