"""
test_picker_hardware.py
=======================
Guards the picker-selection contract for the interactive tools.

All interactive picking now goes through VTK's hardware (z-buffer) picker via
``enable_point_picking(picker="hardware", ...)``, which returns the front-most
*visible* surface hit point. This resolves occlusion (no bleed-through to
back-wall geometry) and uses correct device-pixel coordinates.

* The three **vertex**-resolving tools — seed picking, the PV-clip snake, and
  the geodesic-tag snake — snap that hit point to the nearest vertex.
* The single-triangle **cell** picker (cell-tagging mode) maps that hit point
  to the containing triangle with ``mesh.find_closest_cell`` — the hit point
  lies on a triangle *face*, so it resolves to exactly one cell (no
  vertex-sharing ambiguity).

Headless-safe: the plotter is a stub that records the kwargs passed to
``enable_point_picking``; no rendering or real z-buffer picking happens here
(that needs a live GL context, exercised in the GUI). The cell-mapping test
feeds ``add_cell_at_point`` a hand-computed face point directly, so it exercises
the point->triangle mapping without a real pick.
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
# Cell (triangle-tagging) picker -> hardware picker + point->triangle mapping
# ---------------------------------------------------------------------------
def test_cell_picker_uses_hardware_picker():
    mesh = _tagged_sphere(11)
    plotter = MagicMock()
    editor = ManualEditor(mesh=mesh, plotter=plotter)
    editor.activate()  # enters SELECTING -> _enable_cell_picking
    assert plotter.enable_point_picking.called
    assert _last_picker_kwarg(plotter) == "hardware"


def test_cell_pick_maps_face_point_to_containing_triangle():
    mesh = _tagged_sphere(11)
    plotter = MagicMock()
    editor = ManualEditor(mesh=mesh, plotter=plotter)
    editor.activate()

    # A point in the interior of triangle k's face (biased off-centre so it is
    # nowhere near a vertex) must resolve to exactly triangle k.
    tris = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]
    k = 500
    a, b, c = mesh.points[tris[k]]
    hit = 0.5 * a + 0.3 * b + 0.2 * c

    editor.add_cell_at_point(tuple(float(v) for v in hit))
    assert editor._pending == {k}
