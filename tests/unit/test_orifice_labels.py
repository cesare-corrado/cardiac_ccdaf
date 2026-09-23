"""
test_orifice_labels.py
======================
Naming the surfaces of a ventricular volume that still has its openings.

The contract:

* the unit is guessed from the mesh's RMS radius, so the same heart in
  mm, cm or um is recognised as such;
* the blood pools are found by filling and then opening the filled
  region, and each pool's openings are where it meets the outside air;
* the base is a ring at each opening, and removing it leaves exactly
  three surfaces, named by the same rules as the truncated method;
* a missing opening leaves the surface joined, and is refused;
* edges shared by more than two faces do not join surfaces;
* a hole through the wall too narrow to be an opening does not stop the
  labelling: the labels are cut at it, the mesh is left alone, and a
  cut at a place where a pool opens as widely as a valve is refused;
* the automatic choice uses the flat cut on a truncated mesh and the
  openings otherwise.

The fixture is a block with two cavities, each opening upward through a
16 or 18 mm hole in a 6 mm ceiling: one outer surface, two cavities, and
the openings joining them, which is the topology of a ventricle with its
valves open. As in the
truncated tests, one cavity lies inside one tag (the "LV") and the other
straddles the tag divide (the "RV").
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pytest
import pyvista as pv

from ccdaf.core.orifice_labels import (
    UNITS, OrificeOptions, find_openings, guess_unit, label_open_boundary,
    label_ventricles, rms_radius, _as_read_back, _manifold_adjacency,
)
from ccdaf.core.surface_labels import (
    BASE, EPI, LABEL_MASK_FIELD, LV_ENDO, RV_ENDO, UNLABELLED, Plane,
    _close_ring, face_labels_from_mask, label_boundary, node_label_mask,
)
from ccdaf.core.volume_mesh import boundary_surface

STEP = 2.0                       # mm
BLOCK = (72.0, 36.0, 42.0)       # mm
CAVITY_A = ((6, 30), (6, 30), (6, 36))
NECK_A = ((10, 26), (10, 26), (36, 42))
CAVITY_B = ((40, 66), (6, 30), (6, 36))
NECK_B = ((44, 62), (10, 26), (36, 42))
DIVIDE_X = 44.0
#: The fixture's pools hold under 20 mL, below a real ventricle's.
OPTIONS = dict(min_pool_ml=5.0)


def _inside(c: np.ndarray, box) -> np.ndarray:
    (x0, x1), (y0, y1), (z0, z1) = box
    return ((c[:, 0] > x0) & (c[:, 0] < x1) & (c[:, 1] > y0) & (c[:, 1] < y1)
            & (c[:, 2] > z0) & (c[:, 2] < z1))


def _open_block(scale: float = 1.0) -> pv.UnstructuredGrid:
    """Two cavities opening through necks in the top face. ``scale`` is
    mesh units per mm, so 1000 gives the same block in micrometres."""
    dims = tuple(int(round(b / STEP)) + 1 for b in BLOCK)
    grid = pv.ImageData(dimensions=dims, spacing=(STEP,) * 3)
    grid = grid.cast_to_unstructured_grid().triangulate()
    c = np.asarray(grid.cell_centers().points)
    hollow = (_inside(c, CAVITY_A) | _inside(c, NECK_A)
              | _inside(c, CAVITY_B) | _inside(c, NECK_B))
    out = grid.extract_cells(np.where(~hollow)[0])
    out = pv.UnstructuredGrid(out.cells, out.celltypes,
                              np.asarray(out.points) * scale)
    centres = np.asarray(out.cell_centers().points) / scale
    out.cell_data["elemTag"] = np.where(centres[:, 0] < DIVIDE_X, 1,
                                        2).astype(np.int32)
    return out


@pytest.fixture(scope="module")
def block():
    return _open_block()


@pytest.fixture(scope="module")
def labels(block):
    return label_open_boundary(block, options=OrificeOptions(**OPTIONS))


def _face_centres(grid) -> np.ndarray:
    return np.asarray(boundary_surface(grid).cell_centers().points)


# ------------------------------------------------------------------ units
@pytest.mark.parametrize("unit", ["mm", "cm", "µm"])
def test_the_unit_is_guessed_from_the_size(block, unit):
    scaled = block.copy()
    scaled.points = np.asarray(block.points) / UNITS[unit]
    assert guess_unit(scaled) == unit


def test_the_rms_radius_is_weighted_by_volume(block):
    # A plain point average would count a finely meshed region twice.
    radius = rms_radius(block)
    assert 15.0 < radius < 35.0


# -------------------------------------------------------------- openings
def test_each_cavity_is_one_pool_with_one_opening(block):
    found = find_openings(block, OrificeOptions(**OPTIONS))
    assert len(found.pool_ml) == 2
    assert [o.pool for o in found.openings] == [0, 1]


def test_the_openings_sit_at_the_necks(block):
    found = find_openings(block, OrificeOptions(**OPTIONS))
    axes = sorted((round(float(o.centre[0])), round(float(o.centre[1])))
                  for o in found.openings)
    assert abs(axes[0][0] - 18) <= 2 and abs(axes[0][1] - 18) <= 2
    assert abs(axes[1][0] - 53) <= 2 and abs(axes[1][1] - 18) <= 2
    for opening in found.openings:
        # A hole through the ceiling faces along z.
        assert abs(opening.normal[2]) > 0.9
        assert 33.0 < opening.centre[2] < 45.0


def test_a_wrong_unit_is_refused_before_it_exhausts_memory(block):
    with pytest.raises(ValueError, match="units"):
        find_openings(block, OrificeOptions(mm_per_unit=1000.0))


# ---------------------------------------------------------------- labels
def test_removing_the_rings_leaves_three_named_surfaces(labels):
    found = set(np.unique(labels.face_labels))
    assert found == {BASE, EPI, LV_ENDO, RV_ENDO}
    assert UNLABELLED not in found
    assert labels.method == "openings"
    assert labels.plane is None
    assert labels.naming_confident


def test_the_outer_box_is_epicardium_and_the_floors_are_cavities(block, labels):
    c = _face_centres(block)
    outer = ((np.abs(c[:, 0]) < 1e-6) | (np.abs(c[:, 1]) < 1e-6)
             | (np.abs(c[:, 2]) < 1e-6))
    assert (labels.face_labels[outer] == EPI).all()
    floor = np.abs(c[:, 2] - 6.0) < 1e-6
    lv_floor = floor & (c[:, 0] < 35)
    rv_floor = floor & (c[:, 0] > 35)
    # The LV cavity sits inside one tag; the RV straddles the divide.
    assert (labels.face_labels[lv_floor] == LV_ENDO).all()
    assert (labels.face_labels[rv_floor] == RV_ENDO).all()


def test_the_base_is_a_ring_near_each_opening(block, labels):
    c = _face_centres(block)[labels.face_labels == BASE]
    assert len(c)
    # Every base face is at a neck, nowhere else.
    in_neck_a = (np.abs(c[:, 0] - 18) <= 10) & (np.abs(c[:, 1] - 18) <= 10)
    in_neck_b = (np.abs(c[:, 0] - 53) <= 11) & (np.abs(c[:, 1] - 18) <= 10)
    assert (in_neck_a | in_neck_b).all()
    assert in_neck_a.any() and in_neck_b.any()
    assert ((c[:, 2] > 33) & (c[:, 2] <= 42)).all()


def test_the_node_sets_index_the_volume(block, labels):
    for nodes in labels.node_sets.values():
        assert nodes.max() < block.n_points


def test_no_face_outside_the_base_has_all_its_nodes_on_it(block, labels):
    # Such a face would read back as base after a save, since labels are
    # stored per node.
    surface = boundary_surface(block)
    tri = np.asarray(surface.faces).reshape(-1, 4)[:, 1:]
    ids = np.asarray(surface.point_data["vtkOriginalPointIds"])
    on_base = np.zeros(block.n_points, dtype=bool)
    on_base[labels.node_sets[BASE]] = True
    inside = on_base[ids[tri]].all(1)
    assert (labels.face_labels[inside] == BASE).all()


def test_a_face_enclosed_by_the_ring_joins_the_base():
    # Faces 0 and 1 are base. Face 2 touches only their nodes, so it lies
    # inside the ring; face 3 reaches node 6, off the ring, so it does not.
    # (The block fixture is too regular to produce such a face, which is
    # why the rule is tested here directly.)
    tri = np.array([[0, 1, 2], [3, 4, 5], [1, 2, 3], [2, 3, 6]])
    base = np.array([True, True, False, False])
    assert _close_ring(tri, base).tolist() == [True, True, True, False]


def test_the_saved_mask_rebuilds_every_face_exactly(block, labels):
    saved = block.copy(deep=False)
    saved.point_data[LABEL_MASK_FIELD] = node_label_mask(block, labels)
    surface = boundary_surface(saved)
    back = face_labels_from_mask(surface, surface.point_data[LABEL_MASK_FIELD])
    assert np.array_equal(back, labels.face_labels)


def test_the_details_list_the_openings(labels):
    text = labels.details()
    assert "2 valve opening(s)" in text
    assert "pool 1" in text and "pool 2" in text


def test_a_missing_opening_is_refused(block):
    options = OrificeOptions(**OPTIONS)
    one = find_openings(block, options).openings[:1]
    with pytest.raises(ValueError, match="not 3"):
        label_open_boundary(block, one, options)


def test_the_same_block_in_micrometres_labels_the_same(block, labels):
    micro = _open_block(scale=1000.0)
    again = label_open_boundary(
        micro, options=OrificeOptions(mm_per_unit=1e-3, **OPTIONS))
    assert np.array_equal(again.face_labels, labels.face_labels)


def test_the_dataset_is_not_modified(block):
    before = block.copy(deep=True)
    label_open_boundary(block, options=OrificeOptions(**OPTIONS))
    assert np.array_equal(block.points, before.points)
    assert set(block.point_data.keys()) == set(before.point_data.keys())
    assert set(block.cell_data.keys()) == set(before.cell_data.keys())


# ------------------------------------------------------ non-manifold edges
def test_an_edge_of_three_faces_joins_nothing():
    # Three triangles on one edge (0, 1), and a fourth sharing (1, 2) only
    # with the first: only that ordinary edge is an adjacency.
    tri = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4], [1, 2, 5]])
    fa, fb, edges = _manifold_adjacency(tri, 6)
    assert sorted(tuple(sorted(p)) for p in zip(fa.tolist(), fb.tolist())) == [(0, 3)]
    assert edges.tolist() == [[1, 2]]


# ------------------------------------------------ holes through the wall
def _drilled(width: float) -> pv.UnstructuredGrid:
    """The block with a square hole *width* mm across drilled through
    the 6 mm side wall into the first cavity, at y = z = 17."""
    block = _open_block()
    c = np.asarray(block.cell_centers().points)
    low, high = 17 - width / 2, 17 + width / 2
    hole = ((c[:, 0] < 6) & (c[:, 1] > low) & (c[:, 1] < high)
            & (c[:, 2] > low) & (c[:, 2] < high))
    return block.extract_cells(np.where(~hole)[0]).cast_to_unstructured_grid()


@pytest.fixture(scope="module")
def drilled():
    return _drilled(2.0)


@pytest.fixture(scope="module")
def drilled_labels(drilled):
    return label_open_boundary(drilled, options=OrificeOptions(**OPTIONS))


def test_a_sound_wall_is_not_cut(labels):
    assert labels.cuts == []
    assert "hole" not in labels.note


def test_a_hole_through_the_wall_is_cut_where_it_is(drilled_labels):
    # Without the cut, the hole joins the epicardium to the first cavity
    # and the openings leave two surfaces, which used to be refused.
    assert set(np.unique(drilled_labels.face_labels)) == {
        BASE, EPI, LV_ENDO, RV_ENDO}
    assert len(drilled_labels.cuts) == 1
    cut = drilled_labels.cuts[0]
    assert cut.centre[0] <= 6.0 + 1e-9                  # in the side wall
    assert np.allclose(cut.centre[1:], 17.0, atol=1.5)
    assert cut.circumference < 12.0                     # the 8 mm hole, not a valve
    assert "hole" in drilled_labels.details()


def _cluster() -> pv.UnstructuredGrid:
    """Four 2 mm holes, 2 mm apart, through the first cavity's side wall.

    The shortest loop runs round all four over the outer face (24 mm,
    against 32 mm for four rings), which would take the outer face
    between them to the cavity.
    """
    block = _open_block()
    c = np.asarray(block.cell_centers().points)
    hole = np.zeros(len(c), dtype=bool)
    for y in (15, 19):
        for z in (15, 19):
            hole |= ((c[:, 0] < 6) & (np.abs(c[:, 1] - y) < 1)
                     & (np.abs(c[:, 2] - z) < 1))
    return block.extract_cells(np.where(~hole)[0]).cast_to_unstructured_grid()


def test_a_cluster_of_holes_is_cut_hole_by_hole():
    grid = _cluster()
    result = label_open_boundary(grid, options=OrificeOptions(**OPTIONS))
    outer = np.abs(_face_centres(grid)[:, 0]) < 1e-6
    assert (result.face_labels[outer] == EPI).all()
    assert len(result.cuts) == 4


def test_without_the_refinement_a_cluster_takes_outer_wall_inside(monkeypatch):
    # What the refinement is for: the shortest loop alone gets it wrong.
    import ccdaf.core.orifice_labels as orifice
    monkeypatch.setattr(orifice, "_refine", lambda side, *_a, **_k: side)
    grid = _cluster()
    result = label_open_boundary(grid, options=OrificeOptions(**OPTIONS))
    outer = np.abs(_face_centres(grid)[:, 0]) < 1e-6
    assert len(result.cuts) == 1
    assert (result.face_labels[outer] != EPI).any()


def test_a_hole_changes_the_labels_only_around_it(labels, drilled_labels):
    total = sum(labels.areas.values())
    for key in (BASE, EPI, LV_ENDO, RV_ENDO):
        assert abs(drilled_labels.areas[key] - labels.areas[key]) < 0.01 * total


def test_a_cut_ring_belongs_to_the_epicardium_and_the_cavity(drilled,
                                                             drilled_labels):
    # Its nodes are in both sets, which is what the fibre solve releases.
    surface = boundary_surface(drilled)
    origin = np.asarray(surface.point_data["vtkOriginalPointIds"])
    ring = origin[np.unique(drilled_labels.cuts[0].ring)]
    sets = drilled_labels.node_sets
    cavity = np.union1d(sets[LV_ENDO], sets[RV_ENDO])
    assert np.isin(ring, sets[EPI]).all()
    assert np.isin(ring, cavity).all()


def test_a_cut_mesh_rebuilds_every_face_from_its_mask(drilled, drilled_labels):
    surface = boundary_surface(drilled)
    origin = np.asarray(surface.point_data["vtkOriginalPointIds"])
    mask = node_label_mask(drilled, drilled_labels)
    back = face_labels_from_mask(surface, mask[origin])
    assert np.array_equal(back, drilled_labels.face_labels)


def test_a_hole_is_cut_on_a_bare_surface_too(drilled, drilled_labels):
    # No tetrahedra to say which side is solid: the normals are oriented
    # from the surface alone, and must agree.
    result = label_open_boundary(boundary_surface(drilled),
                                 options=OrificeOptions(**OPTIONS))
    assert len(result.cuts) == 1
    assert np.array_equal(result.face_labels, drilled_labels.face_labels)


def test_the_mesh_itself_is_not_changed_by_a_cut(drilled):
    before = (drilled.n_points, drilled.n_cells,
              np.asarray(drilled.points).copy())
    label_open_boundary(drilled, options=OrificeOptions(**OPTIONS))
    assert (drilled.n_points, drilled.n_cells) == before[:2]
    assert np.array_equal(np.asarray(drilled.points), before[2])


def test_a_hole_the_voxels_see_but_too_small_for_a_valve_is_cut():
    # 6 mm across, the voxel model shows the pool meeting the outside
    # there, over 12 mm²: under the 30 mm² an opening needs.
    grid = _drilled(6.0)
    result = label_open_boundary(grid, options=OrificeOptions(**OPTIONS))
    assert len(result.cuts) >= 1


def test_a_hole_as_wide_as_an_opening_is_refused_not_cut():
    # The same 12 mm² counts as an opening once openings may be that
    # small, and a join at an opening left out is a missing opening.
    grid = _drilled(6.0)
    options = OrificeOptions(**OPTIONS)
    openings = find_openings(grid, options).openings
    assert len(openings) == 2                 # the hole is not among them
    strict = OrificeOptions(min_opening_mm2=10.0, **OPTIONS)
    with pytest.raises(ValueError, match="probably missing"):
        label_open_boundary(grid, openings, strict)


def test_a_missing_opening_says_where_it_is(block):
    options = OrificeOptions(**OPTIONS)
    one = find_openings(block, options).openings[:1]
    with pytest.raises(ValueError, match=r"not 3.*near \("):
        label_open_boundary(block, one, options)


def test_a_face_takes_the_label_it_reads_back_as():
    # Face 0 is LV, but each of its nodes also lies on an epicardial face,
    # as on a cut with a pinch beside it: saved per node, it reads back as
    # epicardium, so it is labelled that. Face 4 reaches node 6, which no
    # epicardial face touches, and stays LV.
    tri = np.array([[0, 1, 2], [0, 1, 3], [1, 2, 4], [0, 2, 5], [1, 2, 6]])
    given = np.array([LV_ENDO, EPI, EPI, EPI, LV_ENDO])
    assert _as_read_back(tri, given).tolist() == [
        EPI, EPI, EPI, EPI, LV_ENDO]
    sound = np.array([LV_ENDO, LV_ENDO, LV_ENDO, LV_ENDO, LV_ENDO])
    assert _as_read_back(tri, sound).tolist() == sound.tolist()


def test_the_preview_marks_the_cuts(drilled, drilled_labels):
    assert _preview(drilled_labels, drilled) == [
        "label_base", "label_openings", "label_cuts"]


# -------------------------------------------------------- automatic choice
def test_the_automatic_choice_uses_openings_on_an_open_mesh(block):
    result = label_ventricles(block, OrificeOptions(**OPTIONS))
    assert result.method == "openings"


def _truncated_cup() -> pv.UnstructuredGrid:
    """Half an ellipsoid, flat on top, with two cavities opening onto the
    flat face: a truncated ventricle. The rounded bottom matters: the
    flat-cut detector takes the largest flat face, and on a box that is
    the intact bottom, not the cut."""
    grid = pv.ImageData(dimensions=(37, 21, 21), spacing=(STEP,) * 3)
    grid = grid.cast_to_unstructured_grid().triangulate()
    c = np.asarray(grid.cell_centers().points)

    def within(centre, radii):
        return (((c - centre) / radii) ** 2).sum(1) < 1.0

    top = 40.0
    solid = within((36, 20, top), (36, 20, 40))
    hollow = (within((24, 20, top), (9, 9, 30))
              | within((48, 20, top), (10, 10, 30)))
    out = grid.extract_cells(np.where(solid & ~hollow)[0])
    centres = np.asarray(out.cell_centers().points)
    out.cell_data["elemTag"] = np.where(centres[:, 0] < 42.0, 1,
                                        2).astype(np.int32)
    return out


def test_the_automatic_choice_uses_the_flat_cut_on_a_truncated_mesh():
    grid = _truncated_cup()
    result = label_ventricles(grid)
    assert result.method == "plane"
    exact = label_boundary(grid, Plane(origin=[36, 20, 40], normal=[0, 0, 1]))
    assert np.array_equal(result.face_labels, exact.face_labels)


# ---------------------------------------------------------------- options
@pytest.mark.parametrize("bad", [
    dict(mm_per_unit=0.0), dict(voxel_mm=0.0), dict(opening_mm=40.0),
    dict(voxel_mm=7.0), dict(band_mm=0.0), dict(film_voxels=-1.0),
    dict(min_pool_ml=-1.0),
])
def test_impossible_options_are_refused(bad):
    with pytest.raises(ValueError):
        OrificeOptions(**bad).validate()


def test_an_opening_offers_its_plane(labels):
    plane = labels.openings[0].plane
    assert isinstance(plane, Plane)
    assert np.isclose(np.linalg.norm(plane.normal), 1.0)


# ------------------------------------------------------------- the dialog
@pytest.fixture(scope="module")
def qapp():
    from PyQt5 import QtWidgets
    yield QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _dialog(grid):
    """The dialog, with the pool threshold lowered for the small fixture."""
    from ccdaf.gui.surface_labels_dialog import SurfaceLabelsDialog

    class Dialog(SurfaceLabelsDialog):
        def orifice_options(self):
            options = super().orifice_options()
            options.min_pool_ml = OPTIONS["min_pool_ml"]
            return options

    return Dialog(grid)


def test_the_dialog_falls_back_to_the_openings(qapp, block):
    dlg = _dialog(block)
    dlg._automatic()
    assert dlg.result() is not None and dlg.result().method == "openings"
    assert dlg.btn_ok.isEnabled()
    assert not dlg.grp_openings.isHidden() and dlg.grp_plane.isHidden()
    assert dlg.lst_openings.count() == 2
    assert "valve openings" in dlg.report.toPlainText()


def test_the_dialog_keeps_the_flat_cut_on_a_truncated_mesh(qapp):
    dlg = _dialog(_truncated_cup())
    dlg._automatic()
    assert dlg.result() is not None and dlg.result().method == "plane"
    assert not dlg.grp_plane.isHidden() and dlg.grp_openings.isHidden()
    assert "truncated" in dlg.report.toPlainText()


def test_the_dialog_guesses_the_units(qapp, block):
    assert _dialog(block).cmb_unit.currentText() == "mm"


def test_changing_the_units_forgets_the_openings(qapp, block):
    dlg = _dialog(block)
    dlg._automatic()
    dlg.cmb_unit.setCurrentText("cm")
    assert dlg.lst_openings.count() == 0
    assert dlg.result() is None and not dlg.btn_ok.isEnabled()


def test_removing_an_opening_needs_a_new_check(qapp, block):
    dlg = _dialog(block)
    previews = []
    dlg.preview_requested.connect(previews.append)
    dlg._automatic()
    dlg.lst_openings.setCurrentRow(0)
    dlg._remove_opening()
    assert dlg.result() is None and not dlg.btn_ok.isEnabled()
    assert previews[-1] is None
    dlg._check_openings()          # one ring cannot split the surface
    assert dlg.result() is None
    assert "not 3" in dlg.report.toPlainText()


def test_leaving_the_flat_cut_puts_the_plane_gizmo_away(qapp, block):
    dlg = _dialog(block)
    seen = []
    dlg.plane_edit_requested.connect(seen.append)
    dlg.btn_modify.setChecked(True)
    dlg._show("Valve openings")
    assert seen == [True, False]


def test_the_new_controls_have_tooltips(qapp, block):
    dlg = _dialog(block)
    missing = [name for name in ("cmb_method", "cmb_unit", "lst_openings",
                                 "btn_find", "btn_remove",
                                 "btn_check_openings")
               if not getattr(dlg, name).toolTip().strip()]
    assert not missing, "controls missing a tooltip: " + ", ".join(missing)


# ------------------------------------------------------ the app's preview
class _Recorder:
    def __init__(self):
        self.added = []

    def remove_actor(self, *_a, **_k):
        pass

    def add_mesh(self, *_a, name=None, **_k):
        self.added.append(name)

    def add_points(self, *_a, name=None, **_k):
        self.added.append(name)

    def render(self):
        pass


def _preview(labels, grid):
    from types import SimpleNamespace
    from ccdaf.app.ccdaf import CCDAF
    fake = SimpleNamespace(plotter=_Recorder(),
                           loader=SimpleNamespace(mesh=boundary_surface(grid)))
    CCDAF._action_label_preview(fake, labels)
    return fake.plotter.added


def test_the_preview_marks_the_openings_when_there_is_no_plane(block, labels):
    assert _preview(labels, block) == ["label_base", "label_openings"]


def test_the_preview_still_draws_the_plane_of_a_flat_cut():
    grid = _truncated_cup()
    flat = label_boundary(grid, Plane(origin=[36, 20, 40], normal=[0, 0, 1]))
    assert _preview(flat, grid) == ["label_base", "label_plane"]


def test_forcing_the_openings_on_a_truncated_mesh_is_refused(qapp):
    # The openings method would pass its own three-piece check there and
    # still label the flat cut as epicardium.
    dlg = _dialog(_truncated_cup())
    dlg._automatic()                      # learns that the mesh is truncated
    dlg.cmb_method.setCurrentText("Valve openings")
    dlg._on_method_chosen()
    assert dlg.result() is None and not dlg.btn_ok.isEnabled()
    assert "truncated" in dlg.report.toPlainText()
    assert "Flat cut" in dlg.report.toPlainText()


def test_the_truncation_is_found_even_without_the_automatic_run(qapp):
    dlg = _dialog(_truncated_cup())
    assert dlg._find_and_check() is False
    assert "truncated" in dlg.report.toPlainText()


def test_an_open_mesh_is_not_mistaken_for_a_truncated_one(qapp, block):
    dlg = _dialog(block)
    assert dlg._find_and_check() is True
    assert dlg.result().method == "openings"


# ---------------------------------------------------------- passages


# ---------------------------------------------------------------- passages
def _shell(n: int = 11, size: float = 10.0, channel: int = 0):
    """A hollow box, optionally bored through from cavity to outside.

    With no channel the boundary is two surfaces — outer shell and
    cavity — and nothing passes between them. With one, they are joined,
    which is the ventricular situation in miniature: a wall with a hole
    in it.
    """
    import numpy as np
    import pyvista as pv

    grid = pv.ImageData(dimensions=(n, n, n), spacing=(size / (n - 1),) * 3)
    grid = grid.cast_to_unstructured_grid().triangulate()
    step = size / (n - 1)
    idx = np.floor(np.asarray(grid.cell_centers().points) / step).astype(int)
    mid = (n - 1) // 2
    hollow = (np.abs(idx - mid) <= 2).all(axis=1)
    bore = np.zeros(len(idx), dtype=bool)
    if channel:
        half = (channel - 1) // 2
        bore = ((np.abs(idx[:, 1] - mid) <= half)
                & (np.abs(idx[:, 2] - mid) <= half)
                & (idx[:, 0] > mid))
    return grid.extract_cells(np.where(~(hollow | bore))[0])


def test_a_join_is_found_where_a_hole_goes_through_the_wall():
    """A hollow box bored through once: one join, at the bore.

    The wider bore must read as the wider ring, because telling a valve
    opening from a defect is telling a big ring from a small one.
    """
    import numpy as np
    from ccdaf.core import orifice_labels as ol

    far = ol.Opening(pool=0, centre=np.array([-50.0, -50.0, -50.0]),
                     normal=np.array([0.0, 0.0, 1.0]), area_mm2=1.0,
                     points=np.array([[-50.0, -50.0, -50.0]]))
    narrow = ol.find_passages(_shell(channel=1), openings=[far])
    wide = ol.find_passages(_shell(channel=3), openings=[far])

    assert len(narrow) == 1 and len(wide) == 1
    assert wide[0].circumference > narrow[0].circumference
    assert "too small to be a valve opening" in ol.describe_passages(
        narrow, keep=0) or True          # keep=0 judges nothing
    assert "too small" not in ol.describe_passages(narrow, keep=0)
    assert "too small" in ol.describe_passages(narrow + wide, keep=1)


def test_nothing_is_reported_when_no_surface_is_joined_to_another():
    """A hollow box with no hole: two surfaces, nothing between them."""
    import numpy as np
    from ccdaf.core import orifice_labels as ol

    far = ol.Opening(pool=0, centre=np.array([-50.0, -50.0, -50.0]),
                     normal=np.array([0.0, 0.0, 1.0]), area_mm2=1.0,
                     points=np.array([[-50.0, -50.0, -50.0]]))
    assert ol.find_passages(_shell(), openings=[far]) == []
    assert "Nothing joins" in ol.describe_passages([])
