"""
test_clip_pick_validation.py
============================
Guards which picks the PV-clip snake accepts.

Two ways a pick used to be taken when it should not have been:

* **The wrong vein.** The pick was judged by the vertex nearest to it. Where
  two tagged regions come close — the right-vein carina above all — the vertex
  nearest to a point on an RIPV triangle can be an RSPV vertex, and the pick
  was accepted for RSPV even though the surface clicked belongs to RIPV. The
  triangle under the pick decides now, so the region clicked is the region
  judged.

* **A miss.** ``vtkHardwarePicker`` reports a world position even when the ray
  hits no geometry (a point on the focal plane), and the vertex nearest to
  *that* is an arbitrary point of the mesh — accepted or rejected by chance.
  The picker holds no dataset on a miss, which is what pyvista's own observer
  tests before forwarding an event; the ``(0, 0, 0)`` test never caught it.
  The same raw-pick path backs the manual-correction snake, so it is guarded
  in both places.

A rejected pick must leave the snake untouched and say why.

Synthetic meshes and a stub plotter — no display, no real z-buffer pick (that
needs a live GL context and is exercised in the GUI).
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pyvista as pv

from ccdaf.core.region_tagger import RegionTagger
from ccdaf.interaction.clipping_tool import ClipMode, ClippingTool
from ccdaf.interaction.manual_editor import ManualEditor

RSPV, RIPV = 15, 17


def _carina_mesh() -> pv.PolyData:
    """Two veins meeting along a shared rim.

    Cell 0 is RSPV, cell 1 is a large RIPV triangle. Vertex 3 is an interior
    RSPV vertex sitting just across the rim, so a point well inside the RIPV
    triangle has an RSPV vertex as its nearest — the carina case.
    """
    pts = np.array([[0.0, 0.0, 0.0],      # 0 shared rim
                    [1.0, 0.0, 0.0],      # 1 shared rim
                    [0.5, 2.0, 0.0],      # 2 RIPV apex
                    [0.5, -0.05, 0.0]])   # 3 RSPV interior
    faces = np.hstack([[3, 0, 1, 3], [3, 0, 2, 1]])
    mesh = pv.PolyData(pts, faces)
    mesh.cell_data["elemTag"] = np.array([RSPV, RIPV], dtype=np.int32)
    return mesh


def _patchwork_mesh() -> pv.PolyData:
    """A sphere with an RSPV cap, an RIPV cap and body in between."""
    mesh = pv.Sphere(theta_resolution=40, phi_resolution=40).triangulate()
    centers = mesh.cell_centers().points
    tags = np.ones(mesh.n_cells, dtype=np.int32)
    tags[centers[:, 2] > 0.30] = RSPV
    tags[centers[:, 2] < -0.30] = RIPV
    mesh.cell_data["elemTag"] = tags
    return mesh


def _clipper(mesh: pv.PolyData, messages: list, plotter=None) -> ClippingTool:
    return ClippingTool(
        mesh_getter=lambda: mesh,
        mesh_setter=lambda m: None,
        plotter=plotter if plotter is not None else MagicMock(),
        on_status=messages.append,
    )


def _cell_point(mesh: pv.PolyData, cell_id: int) -> np.ndarray:
    """A point on the face of ``cell_id`` — what a surface pick returns."""
    return np.asarray(mesh.extract_cells(cell_id).points, dtype=float).mean(axis=0)


# ---------------------------------------------------------------------------
# The wrong vein
# ---------------------------------------------------------------------------
def test_pick_on_another_veins_triangle_is_rejected():
    mesh = _carina_mesh()
    msgs: list = []
    clip = _clipper(mesh, msgs)
    clip.start_pv_contour(pv_label=RSPV)

    probe = np.array([0.5, 0.10, 0.0])          # inside the RIPV triangle
    # Precondition: the nearest vertex really is a valid RSPV one, which is
    # what used to get this pick accepted.
    nearest = int(mesh.find_closest_point(probe))
    assert int(clip._point_tag[nearest]) == RSPV and not clip._boundary[nearest]

    msgs.clear()
    clip._on_contour_pick(tuple(probe))
    assert clip._pick_count == 0
    assert clip._path == []
    assert any("tagged 17" in m for m in msgs)


def test_pick_on_the_target_triangle_is_accepted():
    mesh = _carina_mesh()
    msgs: list = []
    clip = _clipper(mesh, msgs)
    clip.start_pv_contour(pv_label=RSPV)
    clip._on_contour_pick(tuple(_cell_point(mesh, 0)))
    assert clip._pick_count == 1


def test_accepted_pick_lands_on_a_vertex_of_the_picked_triangle():
    mesh = _patchwork_mesh()
    rspv_cell = int(np.where(
        np.asarray(mesh.cell_data["elemTag"]) == RSPV)[0][0])
    clip = _clipper(mesh, [])
    clip.start_pv_contour(pv_label=RSPV)
    clip._on_contour_pick(tuple(_cell_point(mesh, rspv_cell)))
    tri = np.asarray(mesh.faces).reshape(-1, 4)[rspv_cell, 1:]
    assert clip._path == [clip._head]
    assert clip._head in tri.tolist()


def test_pick_on_the_body_is_still_rejected():
    mesh = _patchwork_mesh()
    body_cell = int(np.where(np.asarray(mesh.cell_data["elemTag"]) == 1)[0][0])
    msgs: list = []
    clip = _clipper(mesh, msgs)
    clip.start_pv_contour(pv_label=RSPV)
    msgs.clear()
    clip._on_contour_pick(tuple(_cell_point(mesh, body_cell)))
    assert clip._pick_count == 0
    assert any("not the target tag" in m for m in msgs)


def test_pick_on_a_border_triangle_is_rejected():
    """A triangle of the target tag whose every vertex is a tag boundary: on
    the rim of the region, with nothing for the snake to travel on."""
    pts = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0],     # 0,1,2 the rim
        [-1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.5, -1.0, 0.0],   # 3,4,5 RIPV
        [5.0, 0.0, 0.0], [6.0, 0.0, 0.0], [5.5, 1.0, 0.0],     # 6,7,8 a second
        [5.5, 0.35, 0.0],                                      # 9  RSPV patch
    ])
    faces = np.hstack([
        [3, 0, 1, 2],                                   # RSPV, rim triangle
        [3, 0, 3, 2], [3, 1, 2, 4], [3, 0, 5, 1],       # RIPV all around it
        [3, 6, 7, 9], [3, 7, 8, 9], [3, 8, 6, 9],       # RSPV with interior v9
    ])
    mesh = pv.PolyData(pts, faces)
    mesh.cell_data["elemTag"] = np.array(
        [RSPV, RIPV, RIPV, RIPV, RSPV, RSPV, RSPV], dtype=np.int32)

    msgs: list = []
    clip = _clipper(mesh, msgs)
    clip.start_pv_contour(pv_label=RSPV)
    assert clip.mode is ClipMode.PV_CONTOUR         # v9 keeps the session alive
    assert not clip._allowed[[0, 1, 2]].any()       # the rim triangle is all border

    msgs.clear()
    clip._on_contour_pick(tuple(_cell_point(mesh, 0)))
    assert clip._pick_count == 0
    assert any("region border" in m for m in msgs)


# ---------------------------------------------------------------------------
# A miss
# ---------------------------------------------------------------------------
def _plotter_with_pick(dataset, position=(0.0, 1.0, 0.0)) -> MagicMock:
    """Stub plotter whose picker reports ``dataset`` and ``position``."""
    plotter = MagicMock()
    plotter.iren.interactor.GetEventPosition.return_value = (100, 100)
    plotter.picker.GetDataSet.return_value = dataset
    plotter.picker.GetPickPosition.return_value = position
    return plotter


def test_missed_pick_places_no_clip_point():
    mesh = _patchwork_mesh()
    rspv_cell = int(np.where(
        np.asarray(mesh.cell_data["elemTag"]) == RSPV)[0][0])
    on_surface = tuple(_cell_point(mesh, rspv_cell))
    msgs: list = []
    # The picker reports a position on the target region, but holds no
    # dataset: the ray hit nothing and that position is off the focal plane.
    clip = _clipper(mesh, msgs, plotter=_plotter_with_pick(None, on_surface))
    clip.start_pv_contour(pv_label=RSPV)
    msgs.clear()
    clip.pick_at_cursor()
    assert clip._pick_count == 0
    assert any("nothing under the cursor" in m for m in msgs)


def test_hit_pick_places_a_clip_point():
    mesh = _patchwork_mesh()
    rspv_cell = int(np.where(
        np.asarray(mesh.cell_data["elemTag"]) == RSPV)[0][0])
    on_surface = tuple(_cell_point(mesh, rspv_cell))
    clip = _clipper(mesh, [], plotter=_plotter_with_pick(mesh, on_surface))
    clip.start_pv_contour(pv_label=RSPV)
    clip.pick_at_cursor()
    assert clip._pick_count == 1


def test_missed_pick_drops_no_snake_anchor():
    """The manual-correction snake shares the raw-pick path."""
    mesh = _patchwork_mesh()
    editor = ManualEditor(mesh=mesh, plotter=_plotter_with_pick(None))
    editor.start_snake(RegionTagger(mesh))
    assert editor.snake_pick_at_cursor() == 0
    assert editor.snake_point_count == 0


def test_hit_pick_drops_a_snake_anchor():
    mesh = _patchwork_mesh()
    point = tuple(np.asarray(mesh.points[0], dtype=float))
    editor = ManualEditor(mesh=mesh, plotter=_plotter_with_pick(mesh, point))
    editor.start_snake(RegionTagger(mesh))
    assert editor.snake_pick_at_cursor() == 1
    assert editor.snake_point_count == 1
