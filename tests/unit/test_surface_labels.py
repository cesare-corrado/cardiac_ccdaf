"""
test_surface_labels.py
======================
Naming the surfaces of a truncated ventricular volume.

The contract:

* the base is exactly the faces lying **on** the plane and parallel to it.
  Distance is what makes it exact; the angle test guards the tangent case.
  Measured on a real cut, the pair scores precision and recall 1.000 while
  the angle test alone scores 0.960 and scatters 28 patches;
* removing the base must leave exactly three surfaces. A mesh that still
  has its valve orifices leaves one, and must be **refused**, because an
  orifice joins epicardium to endocardium whatever is cut;
* the epicardium is the piece on the convex hull;
* the two cavities are told apart by the tags behind them, not by size.
  The LV cavity is enclosed by one tag; the RV cavity also sees the
  septum, which carries the LV's tag, so it comes back mixed. Size is the
  wrong signal: on the real ventricle the LV endocardium is the *smaller*
  of the two;
* where the tags cannot decide, the naming says so instead of guessing;
* node sets index the mesh the caller passed in, not the derived boundary.
  Getting that wrong would not raise, it would silently name other nodes.

The fixture is a cube with two blind square holes drilled from the top,
which is the same topology as a truncated ventricle: one outer surface,
two cavities, and a flat base joining them.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pytest
import pyvista as pv
from PyQt5 import QtWidgets

from ccdaf.gui.surface_labels_dialog import SurfaceLabelsDialog
from ccdaf.core.mesh_loader import MeshLoader
from ccdaf.core.surface_labels import (
    BASE, EPI, LABEL_FIELD, LABEL_MASK_FIELD, LV_ENDO, MASK_BITS, RV_ENDO,
    LabelOptions, Plane, detect_base_plane, face_labels_from_mask,
    face_labels_from_nodes, label_boundary, label_boundary_or_snap,
    mean_edge_length, node_label_mask, node_labels, snap_to_cut,
)
from ccdaf.core.volume_mesh import boundary_surface

N = 9          # points per side
SIZE = 8.0     # mesh units


def _drilled(tagged: bool = True) -> pv.UnstructuredGrid:
    """A cube with two blind holes drilled from the top.

    When ``tagged``, the block is split into two materials along x, and the
    holes are placed so one is enclosed by a single tag (the "LV") while
    the other straddles the divide (the "RV", which also sees the
    "septum"). That is the signature the naming relies on.
    """
    grid = pv.ImageData(dimensions=(N, N, N),
                        spacing=(SIZE / (N - 1),) * 3).cast_to_unstructured_grid()
    grid = grid.triangulate()
    step = SIZE / (N - 1)
    centres = np.asarray(grid.cell_centers().points)
    ix = np.floor(centres[:, 0] / step).astype(int)
    iy = np.floor(centres[:, 1] / step).astype(int)
    iz = np.floor(centres[:, 2] / step).astype(int)
    # Two blind holes: open at the top (high z), floored at iz == 0.
    hole_a = (ix == 1) & (iy == 3) & (iz >= 1)          # inside the left half
    hole_b = (ix == 4) & (iy == 3) & (iz >= 1)          # straddles the divide
    keep = np.where(~(hole_a | hole_b))[0]
    out = grid.extract_cells(keep)
    if tagged:
        c = np.asarray(out.cell_centers().points)
        out.cell_data["elemTag"] = np.where(
            c[:, 0] < 4 * step, 1, 2).astype(np.int32)
    else:
        out.cell_data["elemTag"] = np.ones(out.n_cells, dtype=np.int32)
    return out


def _top_plane() -> Plane:
    return Plane(origin=[SIZE / 2, SIZE / 2, SIZE], normal=[0.0, 0.0, 1.0])


def _slab() -> pv.UnstructuredGrid:
    """A wide, thin slab with two blind holes drilled from the top.

    Unlike the cube, one flat face is uniquely the largest: the intact
    bottom at 100 square units, against 98 for the drilled top and 20 for
    each side. That makes the detector's answer unambiguous, which the
    cube's five-way tie at 64 does not.
    """
    wide, thick = 10.0, 2.0
    nx, nz = 11, 3
    grid = pv.ImageData(dimensions=(nx, nx, nz),
                        spacing=(wide / (nx - 1), wide / (nx - 1),
                                 thick / (nz - 1))).cast_to_unstructured_grid()
    grid = grid.triangulate()
    step = wide / (nx - 1)
    c = np.asarray(grid.cell_centers().points)
    ix = np.floor(c[:, 0] / step).astype(int)
    iy = np.floor(c[:, 1] / step).astype(int)
    iz = np.floor(c[:, 2] / (thick / (nz - 1))).astype(int)
    holes = (((ix == 2) & (iy == 5)) | ((ix == 7) & (iy == 5))) & (iz >= 1)
    out = grid.extract_cells(np.where(~holes)[0])
    out.cell_data["elemTag"] = np.ones(out.n_cells, dtype=np.int32)
    return out


# ------------------------------------------------------------- detection
def test_detection_picks_the_largest_flat_face():
    """It maximises coplanar area, so the fixture must have one winner.

    On a real truncated ventricle the basal cut is the only flat face and
    the detector recovers it to 0.00 degrees and zero offset.
    """
    found = detect_base_plane(_slab())
    assert found is not None
    assert abs(float(found.normal[2])) > 0.99          # a z face
    assert abs(float(found.origin[2])) < 1e-6          # the intact bottom


def test_detection_can_pick_the_wrong_end_and_the_check_catches_it():
    """The honest limitation, pinned down.

    The detector knows about flatness, not about anatomy, so on a mesh
    flat at both ends it can return the end the cavities do *not* open
    onto. Nothing downstream should paper over that: the acceptance test
    refuses, because removing that face separates nothing.
    """
    slab = _slab()
    found = detect_base_plane(slab)
    assert abs(float(found.origin[2])) < 1e-6          # it chose the bottom
    with pytest.raises(ValueError, match="not 3|not cut there"):
        label_boundary(slab, found)


def test_detection_is_reproducible():
    a = detect_base_plane(_drilled())
    b = detect_base_plane(_drilled())
    assert np.allclose(a.normal, b.normal)
    assert np.allclose(a.origin, b.origin)


# ------------------------------------------------------------- labelling
def test_the_base_is_the_faces_on_the_plane():
    grid = _drilled()
    labels = label_boundary(grid, _top_plane())
    surface = grid.extract_surface(algorithm="dataset_surface").triangulate()
    tri = np.asarray(surface.faces).reshape(-1, 4)[:, 1:]
    pts = np.asarray(surface.points)
    tol = 0.5 * mean_edge_length(surface)
    on_plane = (np.abs(pts[:, 2] - SIZE) < tol)[tri].all(axis=1)
    assert int((labels.face_labels == BASE).sum()) == int(on_plane.sum())
    assert int((labels.face_labels == BASE).sum()) > 0


def test_removing_the_base_leaves_three_surfaces():
    labels = label_boundary(_drilled(), _top_plane())
    for key in (EPI, LV_ENDO, RV_ENDO):
        assert int((labels.face_labels == key).sum()) > 0
    assert labels.areas[EPI] > labels.areas[LV_ENDO]


def test_the_epicardium_is_the_outermost_piece():
    grid = _drilled()
    labels = label_boundary(grid, _top_plane())
    surface = grid.extract_surface(algorithm="dataset_surface").triangulate()
    centres = np.asarray(surface.cell_centers().points)
    epi = centres[labels.face_labels == EPI]
    cavity = centres[labels.face_labels == LV_ENDO]
    # The cavities sit inside the block; the outer surface does not.
    spread = epi.max(axis=0) - epi.min(axis=0)
    assert spread[0] > 0.9 * SIZE and spread[1] > 0.9 * SIZE
    assert float(np.ptp(cavity[:, 0])) < 0.5 * SIZE


def test_the_tag_signature_names_the_cavities():
    """Pure tag is the LV; mixed is the RV, because the septum is LV tissue."""
    grid = _drilled(tagged=True)
    labels = label_boundary(grid, _top_plane())
    assert labels.naming_confident
    surface = grid.extract_surface(algorithm="dataset_surface").triangulate()
    tags = np.asarray(surface.cell_data["elemTag"])
    lv = tags[labels.face_labels == LV_ENDO]
    rv = tags[labels.face_labels == RV_ENDO]
    assert len(np.unique(lv)) == 1
    assert len(np.unique(rv)) == 2


def test_naming_says_so_when_the_tags_cannot_decide():
    """One material everywhere: the rule has no signal and must admit it."""
    labels = label_boundary(_drilled(tagged=False), _top_plane())
    assert not labels.naming_confident
    assert "area" in labels.note.lower()
    assert "uncertain" in labels.summary()


# --------------------------------------------------------------- snapping
def test_a_plane_beside_the_cut_snaps_onto_it():
    """A dragged plane means "use that cut", not "cut exactly here".

    Selecting faces needs the plane to coincide with a cut the mesh already
    has, which no hand-placed plane does, so without snapping the gizmo
    could only ever move the user off the answer.
    """
    grid = _drilled()
    beside = Plane(origin=[SIZE / 2, SIZE / 2, SIZE - 1.5],
                   normal=[0.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        label_boundary(grid, beside)          # nothing lies on it

    labels, snapped = label_boundary_or_snap(grid, beside)
    assert snapped
    assert abs(float(labels.plane.origin[2]) - SIZE) < 1e-6
    exact = label_boundary(grid, _top_plane())
    assert int((labels.face_labels == BASE).sum()) == \
        int((exact.face_labels == BASE).sum())


def test_snapping_picks_the_nearest_cut_not_the_largest():
    """The cube is flat at both ends, and the bottom is the larger face.

    Scoring by area alone would answer with the bottom wherever the user
    pointed, which is the opposite of what a drag means.
    """
    grid = _drilled()
    near_top = Plane(origin=[SIZE / 2, SIZE / 2, SIZE - 1.5],
                     normal=[0.0, 0.0, 1.0])
    got = snap_to_cut(grid, near_top)
    assert got is not None
    assert abs(float(got.origin[2]) - SIZE) < 1e-6


def test_a_plane_already_on_the_cut_is_not_moved():
    labels, snapped = label_boundary_or_snap(_drilled(), _top_plane())
    assert not snapped
    assert abs(float(labels.plane.origin[2]) - SIZE) < 1e-6


def test_snapping_refuses_to_reach_across_the_mesh():
    """Pointing at nothing returns nothing, rather than a distant guess."""
    grid = _drilled()
    far = Plane(origin=[SIZE / 2, SIZE / 2, -50.0], normal=[0.0, 0.0, 1.0])
    assert snap_to_cut(grid, far) is None
    with pytest.raises(ValueError, match="No flat cut was found"):
        label_boundary_or_snap(grid, far)


def test_snapping_can_be_turned_off():
    grid = _drilled()
    beside = Plane(origin=[SIZE / 2, SIZE / 2, SIZE - 1.5],
                   normal=[0.0, 0.0, 1.0])
    assert snap_to_cut(grid, beside, LabelOptions(snap_edges=0.0)) is None


# ------------------------------------------------------------- refusals
def test_a_mesh_with_no_cut_at_that_plane_is_refused():
    grid = _drilled()
    away = Plane(origin=[SIZE / 2, SIZE / 2, SIZE * 0.5], normal=[0.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="not cut there|not 3"):
        label_boundary(grid, away)


def test_a_mesh_whose_cavities_stay_joined_is_refused():
    """A through hole is an orifice: the base cannot separate anything.

    This is the untruncated ventricle in miniature, and the error has to
    name the cause rather than produce a labelling nobody can use.
    """
    grid = pv.ImageData(dimensions=(N, N, N),
                        spacing=(SIZE / (N - 1),) * 3).cast_to_unstructured_grid()
    grid = grid.triangulate()
    step = SIZE / (N - 1)
    c = np.asarray(grid.cell_centers().points)
    through = (np.floor(c[:, 0] / step).astype(int) == 3) & \
              (np.floor(c[:, 1] / step).astype(int) == 3)
    holed = grid.extract_cells(np.where(~through)[0])
    holed.cell_data["elemTag"] = np.ones(holed.n_cells, dtype=np.int32)
    with pytest.raises(ValueError, match="not 3"):
        label_boundary(holed, _top_plane())


# ------------------------------------------------------------- bookkeeping
def test_node_sets_index_the_volume_not_the_boundary():
    grid = _drilled()
    labels = label_boundary(grid, _top_plane())
    pts = np.asarray(grid.points)
    for key in (BASE, EPI, LV_ENDO, RV_ENDO):
        ids = labels.node_sets[key]
        assert ids.max() < grid.n_points
    # Every base node really is on the plane.
    assert np.allclose(pts[labels.node_sets[BASE]][:, 2], SIZE)


def test_node_labels_cover_the_mesh():
    grid = _drilled()
    labels = label_boundary(grid, _top_plane())
    per_node = node_labels(grid, labels)
    assert per_node.shape == (grid.n_points,)
    assert set(np.unique(per_node).tolist()) <= {0, BASE, EPI, LV_ENDO, RV_ENDO}
    assert int((per_node == BASE).sum()) == len(labels.node_sets[BASE])


def test_the_boundary_offers_one_label_field_not_two():
    """The bookkeeping the app depends on, pinned down here.

    Taking a volume's boundary copies its point arrays across, so the node
    labels land on the boundary as well as the face labels. Left alone the
    field list carries ``surfaceLabel`` twice, once per association, and
    the point copy would draw a label as a ramp between nodes.
    """
    grid = _drilled()
    labels = label_boundary(grid, _top_plane())
    grid.point_data[LABEL_FIELD] = node_labels(grid, labels)

    surface = boundary_surface(grid)
    assert LABEL_FIELD in surface.point_data          # inherited, as the app finds it

    del surface.point_data[LABEL_FIELD]
    surface.cell_data[LABEL_FIELD] = labels.face_labels
    names = MeshLoader.field_names(surface)
    assert names.count(LABEL_FIELD) == 1
    # The volume keeps the node labels: they are what a solve reads.
    assert LABEL_FIELD in grid.point_data


def test_faces_take_the_label_their_nodes_agree_on():
    grid = _drilled()
    labels = label_boundary(grid, _top_plane())
    surface = boundary_surface(grid)
    per_node = node_labels(grid, labels)[
        np.asarray(surface.point_data["vtkOriginalPointIds"], dtype=np.int64)]

    derived = face_labels_from_nodes(surface, per_node)
    assert derived.shape == (surface.n_cells,)
    # Every base face is recovered: its three nodes are all base nodes.
    base_faces = labels.face_labels == BASE
    assert (derived[base_faces] == BASE).all()
    # Nothing is invented: a derived label is either a real one or nothing.
    assert set(np.unique(derived).tolist()) <= {0, BASE, EPI, LV_ENDO, RV_ENDO}


def test_a_disagreeing_face_is_unlabelled_not_averaged():
    """The seam between two surfaces must not become a third label."""
    grid = _drilled()
    surface = boundary_surface(grid)
    per_node = np.full(surface.n_points, EPI, dtype=np.int32)
    tri = np.asarray(surface.faces).reshape(-1, 4)[:, 1:]
    per_node[tri[0][0]] = BASE                      # one node of face 0 differs
    derived = face_labels_from_nodes(surface, per_node)
    assert derived[0] == 0
    assert (derived[1:] != 0).sum() > 0


def test_a_saved_labelling_can_be_drawn_again(tmp_path):
    """The reload path: only node labels survive, and they must still draw.

    The categorical render reads cell values, so a labelling read back from
    a file has to be turned back into face labels. Without that this raises
    KeyError the moment the field is selected.
    """
    grid = _drilled()
    labels = label_boundary(grid, _top_plane())
    grid.point_data[LABEL_FIELD] = node_labels(grid, labels)

    path = tmp_path / "labelled.vtk"
    loader = MeshLoader()
    loader.set_volume(grid)
    loader.save(str(path), fields=None)

    back = MeshLoader()
    back.load(str(path))
    mesh = back.mesh
    assert LABEL_FIELD in mesh.point_data      # how it comes back
    assert LABEL_FIELD not in mesh.cell_data   # what the render path wants

    derived = face_labels_from_nodes(mesh, mesh.point_data[LABEL_FIELD])
    assert derived.shape == (mesh.n_cells,)
    assert int((derived == BASE).sum()) > 0


def test_the_mask_rebuilds_every_face_exactly():
    """What one label per node cannot do, and why the mask exists.

    A node on a seam belongs to two surfaces. A single label has to call
    it one of them, so the faces around it disagree and come back
    unlabelled; membership keeps both, so every face is recovered.
    """
    grid = _drilled()
    labels = label_boundary(grid, _top_plane())
    surface = boundary_surface(grid)
    origin = np.asarray(surface.point_data["vtkOriginalPointIds"],
                        dtype=np.int64)

    mask = node_label_mask(grid, labels)
    from_mask = face_labels_from_mask(surface, mask[origin])
    assert np.array_equal(from_mask, labels.face_labels)

    # The lossy form, for contrast: same mesh, some faces given up.
    from_nodes = face_labels_from_nodes(
        surface, node_labels(grid, labels)[origin])
    assert int((from_nodes == 0).sum()) > 0
    assert int((from_mask == 0).sum()) == 0


def test_a_seam_node_belongs_to_two_surfaces():
    grid = _drilled()
    labels = label_boundary(grid, _top_plane())
    mask = node_label_mask(grid, labels)
    shared = mask[(mask != 0) & ((mask & (mask - 1)) != 0)]
    assert len(shared) > 0, "the rim nodes must carry more than one bit"
    # Base and epicardium meet at the rim: 1 | 2.
    assert (MASK_BITS[BASE] | MASK_BITS[EPI]) in set(shared.tolist())


def test_the_load_sequence_rebuilds_labels_before_listing_fields():
    """The order, not the helper. This is what broke in the application.

    A labelling read back from a file arrives as the node mask alone, and
    the faces are rebuilt from it. Rebuilt *after* the field list is
    built, the labelling is in memory and invisible: nothing offers it and
    nothing draws it, which is what "no trace of surface labels" looked
    like. Testing the helper alone missed it twice, so this test reads the
    sequence itself.
    """
    import inspect
    import re

    from ccdaf.app.ccdaf import CCDAF

    source = inspect.getsource(CCDAF._adopt_mesh)
    order = re.findall(
        r"_sync_surface_labels|_populate_fields|_render_field", source)
    assert "_sync_surface_labels" in order, \
        "_adopt_mesh must rebuild surface labels at all"
    assert order.index("_sync_surface_labels") < order.index("_populate_fields"), \
        "labels must be rebuilt before the field list is built"
    assert order.index("_sync_surface_labels") < order.index("_render_field"), \
        "labels must be rebuilt before the first render"


def test_only_the_mask_is_stored(tmp_path):
    """One array, not two. The plain label is the mask's lowest set bit.

    Storing both would store the same thing twice, and only the mask can
    describe a node lying on a seam.
    """
    from ccdaf.app.ccdaf import HIDDEN_COLOUR_FIELDS

    grid = _drilled()
    labels = label_boundary(grid, _top_plane())
    grid.point_data[LABEL_MASK_FIELD] = node_label_mask(grid, labels)

    path = tmp_path / "mask_only.vtk"
    loader = MeshLoader()
    loader.set_volume(grid)
    loader.save(str(path), fields=None)

    back = MeshLoader()
    back.load(str(path))
    assert LABEL_MASK_FIELD in back.grid.point_data
    assert LABEL_FIELD not in back.grid.point_data

    # The mask is saved but never offered as something to colour by; the
    # labels derived from it are what the user picks.
    assert LABEL_MASK_FIELD in HIDDEN_COLOUR_FIELDS
    faces = face_labels_from_mask(
        back.mesh, back.mesh.point_data[LABEL_MASK_FIELD])
    assert np.array_equal(faces, labels.face_labels)


def test_the_renderer_prefers_the_mask_after_a_reload(tmp_path):
    """Selecting surfaceLabel on a reloaded mesh must show no seam."""
    from ccdaf.app.ccdaf import _label_values

    grid = _drilled()
    labels = label_boundary(grid, _top_plane())
    grid.point_data[LABEL_FIELD] = node_labels(grid, labels)
    grid.point_data[LABEL_MASK_FIELD] = node_label_mask(grid, labels)

    path = tmp_path / "masked.vtk"
    loader = MeshLoader()
    loader.set_volume(grid)
    loader.save(str(path), fields=None)

    back = MeshLoader()
    back.load(str(path))
    mesh = back.mesh
    assert LABEL_MASK_FIELD in mesh.point_data
    drawn = _label_values(mesh, LABEL_FIELD)
    assert drawn is not None
    assert int((drawn == 0).sum()) == 0, "a seam survived the round trip"
    assert np.array_equal(drawn, labels.face_labels)


def test_a_point_only_label_field_is_not_swapped_for_elemtag():
    """The defect behind "after reloading it shows only the ventricles".

    The renderer used to fall back to ``elemTag`` whenever the chosen field
    was absent from ``cell_data``, which a reloaded labelling always is:
    only the node labels are stored. On a ventricular mesh that fallback
    draws the LV and RV walls, which looks like a plausible answer and is
    not the field that was asked for.
    """
    from ccdaf.app.ccdaf import _label_values

    grid = _drilled()
    labels = label_boundary(grid, _top_plane())
    grid.point_data[LABEL_FIELD] = node_labels(grid, labels)
    surface = boundary_surface(grid)
    assert LABEL_FIELD in surface.point_data
    assert LABEL_FIELD not in surface.cell_data

    values = _label_values(surface, LABEL_FIELD)
    assert values is not None
    assert set(np.unique(values).tolist()) <= {0, BASE, EPI, LV_ENDO, RV_ENDO}
    for key in (BASE, EPI, LV_ENDO, RV_ENDO):
        assert int((values == key).sum()) > 0, f"{key} missing after reload"
    assert _label_values(surface, "not a field") is None


def test_the_dataset_is_not_modified():
    grid = _drilled()
    cells, points = grid.n_cells, grid.n_points
    arrays = set(grid.cell_data.keys())
    label_boundary(grid, _top_plane())
    assert grid.n_cells == cells and grid.n_points == points
    assert set(grid.cell_data.keys()) == arrays


@pytest.mark.parametrize("options", [
    LabelOptions(tol_edges=0.0),
    LabelOptions(cos_min=0.0),
    LabelOptions(cos_min=1.5),
    LabelOptions(min_patch_fraction=1.0),
    LabelOptions(samples=0),
])
def test_impossible_options_are_refused(options):
    with pytest.raises(ValueError):
        label_boundary(_drilled(), _top_plane(), options)


def test_a_plane_needs_a_real_normal():
    with pytest.raises(ValueError):
        Plane(origin=[0, 0, 0], normal=[0, 0, 0])


# ------------------------------------------------------------- the dialog
@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def _dialog(grid) -> "SurfaceLabelsDialog":
    return SurfaceLabelsDialog(grid)


def _type_plane(dlg, plane: Plane) -> None:
    for box, value in zip(dlg._origin, plane.origin):
        box.setValue(float(value))
    for box, value in zip(dlg._normal, plane.normal):
        box.setValue(float(value))


def test_the_dialog_opens_with_a_verdict(qapp):
    """It proposes rather than presenting a blank form.

    On this fixture the proposal is a *refusal*, because the cube is flat
    on every side and the detector cannot know which end the holes open
    onto. That is the honest outcome, and it must be visible.
    """
    dlg = _dialog(_drilled())
    dlg._detect_and_check()
    assert dlg.report.toPlainText().strip()


def test_checking_a_good_plane_enables_ok(qapp):
    dlg = _dialog(_drilled())
    _type_plane(dlg, _top_plane())
    dlg._check()
    assert dlg.btn_ok.isEnabled()
    labels = dlg.result()
    assert labels is not None
    assert int((labels.face_labels == BASE).sum()) > 0
    for key in (EPI, LV_ENDO, RV_ENDO):
        assert int((labels.face_labels == key).sum()) > 0


def test_editing_the_plane_invalidates_the_answer(qapp):
    """A stale result must not be writable against an edited plane."""
    dlg = _dialog(_drilled())
    _type_plane(dlg, _top_plane())
    dlg._check()
    assert dlg.result() is not None

    dlg._normal[0].setValue(0.4)
    assert dlg.result() is None
    assert not dlg.btn_ok.isEnabled()


def test_a_bad_plane_is_reported_and_ok_stays_disabled(qapp):
    """Far outside the mesh, where snapping cannot rescue it either.

    A plane merely *beside* a cut is no longer a failure: Check snaps it
    onto the cut, which is the whole point of the draggable plane.
    """
    dlg = _dialog(_drilled())
    _type_plane(dlg, Plane(origin=[SIZE / 2, SIZE / 2, -50.0],
                           normal=[0.0, 0.0, 1.0]))
    dlg._check()
    assert not dlg.btn_ok.isEnabled()
    assert dlg.result() is None
    text = dlg.report.toPlainText()
    assert "not cut there" in text or "No flat cut was found" in text


def test_a_zero_normal_is_reported_not_raised(qapp):
    dlg = _dialog(_drilled())
    for box in dlg._normal:
        box.setValue(0.0)
    dlg._check()
    assert not dlg.btn_ok.isEnabled()
    assert "no length" in dlg.report.toPlainText()


def test_the_modify_button_asks_for_the_plane_gizmo(qapp):
    """The window owns no VTK: it asks, and the application answers."""
    dlg = _dialog(_drilled())
    seen = []
    dlg.plane_edit_requested.connect(seen.append)
    dlg.btn_modify.setChecked(True)
    dlg.btn_modify.setChecked(False)
    assert seen == [True, False]


def test_a_good_check_offers_a_preview(qapp):
    dlg = _dialog(_drilled())
    previews = []
    dlg.preview_requested.connect(previews.append)
    _type_plane(dlg, _top_plane())
    dlg._check()
    assert previews and previews[-1] is not None
    assert previews[-1] is dlg.result()


def test_a_refused_plane_clears_the_preview(qapp):
    """A stale highlight would contradict the report beside it."""
    dlg = _dialog(_drilled())
    previews = []
    dlg.preview_requested.connect(previews.append)
    _type_plane(dlg, _top_plane())
    dlg._check()                       # good: something to show
    _type_plane(dlg, Plane(origin=[SIZE / 2, SIZE / 2, -50.0],
                           normal=[0.0, 0.0, 1.0]))
    dlg._check()                       # out of reach: nothing to show
    assert previews[-1] is None


def test_a_snapped_check_moves_the_boxes_onto_the_plane_it_used(qapp):
    """The numbers must describe what the report describes."""
    dlg = _dialog(_drilled())
    _type_plane(dlg, Plane(origin=[SIZE / 2, SIZE / 2, SIZE - 1.5],
                           normal=[0.0, 0.0, 1.0]))
    dlg._check()
    assert dlg.result() is not None
    assert "snapped" in dlg.report.toPlainText()
    assert abs(float(dlg.result().plane.origin[2]) - SIZE) < 1e-6
    assert abs(float(dlg._origin[2].value()) - SIZE) < 1e-3


def test_the_gizmo_writes_the_boxes_and_invalidates_the_answer(qapp):
    """Dragging the plane cannot leave a result from the old one behind."""
    dlg = _dialog(_drilled())
    _type_plane(dlg, _top_plane())
    dlg._check()
    assert dlg.result() is not None

    dlg.set_plane(Plane(origin=[1.0, 2.0, 3.0], normal=[0.0, 0.0, 1.0]))
    assert [b.value() for b in dlg._origin] == [1.0, 2.0, 3.0]
    assert dlg.result() is None
    assert not dlg.btn_ok.isEnabled()


def test_dialog_controls_have_tooltips(qapp):
    dlg = _dialog(_drilled())
    missing = [name for name in ("btn_detect", "btn_modify", "btn_check",
                                 "spn_tol", "spn_cos", "btn_ok")
               if not getattr(dlg, name).toolTip().strip()]
    assert not missing, "controls missing a tooltip: " + ", ".join(missing)
