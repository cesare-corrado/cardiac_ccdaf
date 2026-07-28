"""
test_picker_hardware.py
=======================
Guards the picker-selection contract for the interactive tools.

The three tools that resolve a click to a mesh **vertex** — seed picking, the
PV-clip snake, and the geodesic-tag snake — must request VTK's hardware
(z-buffer) picker via ``enable_point_picking(picker="hardware", ...)``. That
picker returns the front-most *visible* surface point, so a pick can never
bleed through to an occluded back-wall vertex (the symptom of the old
``vtkPointPicker`` default, which snapped to the nearest vertex to the pick
*ray*, hidden vertices included).

The single-triangle **cell** picker is deliberately left on its own path
(``vtkCellPicker`` re-pick inside the callback), so it must NOT request the
hardware picker — this test pins that boundary too.

Headless-safe: the plotter is a stub that records the kwargs passed to
``enable_point_picking``; no rendering or real picking happens here (a z-buffer
pick needs a live GL context, which is exercised in the GUI, not in CI).
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pyvista as pv

from ccdaf.core.region_tagger import RegionTagger
from ccdaf.interaction.seed_selector import SeedSelector
from ccdaf.interaction.manual_editor import ManualEditor
from ccdaf.interaction.clipping_tool import ClippingTool


def _sphere():
    return pv.Sphere(theta_resolution=24, phi_resolution=24).triangulate()


def _tagged_sphere(tag: int = 11):
    mesh = _sphere()
    mesh.cell_data["elemTag"] = np.full(mesh.n_cells, tag, dtype=np.int32)
    return mesh


def _last_picker_kwarg(plotter) -> str:
    """The ``picker`` kwarg of the most recent enable_point_picking call."""
    return plotter.enable_point_picking.call_args.kwargs.get("picker")


# ---------------------------------------------------------------------------
# Vertex-resolving tools -> hardware picker
# ---------------------------------------------------------------------------
def test_seed_selector_uses_hardware_picker():
    plotter = MagicMock()
    selector = SeedSelector(mesh=_sphere(), plotter=plotter)
    selector.start()
    assert plotter.enable_point_picking.called
    assert _last_picker_kwarg(plotter) == "hardware"


def test_pv_clip_snake_uses_hardware_picker():
    mesh = _tagged_sphere(11)
    holder = {"mesh": mesh}
    plotter = MagicMock()
    tool = ClippingTool(
        mesh_getter=lambda: holder["mesh"],
        mesh_setter=lambda m: holder.__setitem__("mesh", m),
        plotter=plotter,
    )
    tool.start_pv_contour(11)
    assert plotter.enable_point_picking.called
    assert _last_picker_kwarg(plotter) == "hardware"


def test_tag_snake_uses_hardware_picker():
    mesh = _tagged_sphere(11)
    plotter = MagicMock()
    editor = ManualEditor(mesh=mesh, plotter=plotter)
    editor.start_snake(RegionTagger(mesh))
    assert plotter.enable_point_picking.called
    assert _last_picker_kwarg(plotter) == "hardware"


# ---------------------------------------------------------------------------
# Triangle/cell picker -> intentionally NOT hardware (kept on vtkCellPicker)
# ---------------------------------------------------------------------------
def test_cell_picker_is_not_hardware():
    mesh = _tagged_sphere(11)
    plotter = MagicMock()
    editor = ManualEditor(mesh=mesh, plotter=plotter)
    editor.activate()  # enters SELECTING -> _enable_cell_picking
    assert plotter.enable_point_picking.called
    assert _last_picker_kwarg(plotter) != "hardware"
