"""
test_boundary_smoothing.py
=========================
Morphological smoothing of a PV label's boundary —
`RegionTagger.dilate_label` / `erode_label` and the undoable
`ManualEditor.smooth_label` that drives them.

A per-triangle boundary zigzags because alternating up/down triangles fall on
opposite sides of the tagging threshold. The contract:

* dilate turns a background cell with >=2 edge-neighbours of the label into the
  label — filling the jagged fringe — but never a cell that also borders a
  *different* PV label, so two regions cannot merge across a body seam;
* erode is the inverse: a label cell with >=2 background neighbours reverts to
  body, shaving spikes;
* **body smooths like any other label** — it is the background, so it has no
  boundary of its own: its boundary *is* the PV boundaries seen from the other
  side. Growing body is every PV label eroding, shrinking it is every PV label
  dilating, and the checkboxes keep their usual meaning;
* the all-label pass reads its neighbour counts from one snapshot, so it is
  order-free — running the single-label passes in sequence feeds each label the
  previous one's output and gives a different, LABELS-order-dependent answer;
* neither op mutates its input;
* `smooth_label` snapshots for undo only when something actually changes.

Uses synthetic meshes (no real EAM data required).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pytest
import pyvista as pv

from ccdaf.core import region_tagger
from ccdaf.core.mesh_loader import BODY_LABEL
from ccdaf.core.region_tagger import LABELS, RegionTagger
from ccdaf.interaction.manual_editor import ManualEditor

PVS = sorted(LABELS.values())


def _mesh_tagger():
    mesh = pv.Sphere(theta_resolution=30, phi_resolution=30).triangulate()
    return mesh, RegionTagger(mesh)


def _interior_cell_with_three_neighbours(tagger):
    for c in range(tagger._tri_adj.shape[0]):
        nb = tagger._tri_adj[c].indices
        if len(nb) == 3:
            return c, list(nb)
    raise AssertionError("no 3-neighbour cell")


def test_dilate_fills_a_two_neighbour_gap():
    _, tagger = _mesh_tagger()
    c, nb = _interior_cell_with_three_neighbours(tagger)
    tags = np.full(tagger._tri_adj.shape[0], BODY_LABEL, dtype=np.int32)
    tags[nb[0]] = tags[nb[1]] = 11          # two of c's neighbours are LSPV
    out = tagger.dilate_label(tags, 11)
    assert out[c] == 11                     # the gap fills


def test_dilate_leaves_a_one_neighbour_cell():
    _, tagger = _mesh_tagger()
    c, nb = _interior_cell_with_three_neighbours(tagger)
    tags = np.full(tagger._tri_adj.shape[0], BODY_LABEL, dtype=np.int32)
    tags[nb[0]] = 11                        # only one neighbour is LSPV
    out = tagger.dilate_label(tags, 11)
    assert out[c] == BODY_LABEL             # flat boundary does not grow


def test_dilate_will_not_bridge_into_another_pv():
    _, tagger = _mesh_tagger()
    c, nb = _interior_cell_with_three_neighbours(tagger)
    tags = np.full(tagger._tri_adj.shape[0], BODY_LABEL, dtype=np.int32)
    tags[nb[0]] = tags[nb[1]] = 11          # two LSPV neighbours ...
    tags[nb[2]] = 13                        # ... but also a LIPV neighbour
    out = tagger.dilate_label(tags, 11)
    assert out[c] == BODY_LABEL             # seam preserved, no merge


def test_erode_removes_a_spike():
    _, tagger = _mesh_tagger()
    c, _ = _interior_cell_with_three_neighbours(tagger)
    tags = np.full(tagger._tri_adj.shape[0], BODY_LABEL, dtype=np.int32)
    tags[c] = 11                            # lone LSPV cell, all neighbours body
    out = tagger.erode_label(tags, 11)
    assert out[c] == BODY_LABEL


def test_the_input_is_left_alone():
    _, tagger = _mesh_tagger()
    c, nb = _interior_cell_with_three_neighbours(tagger)
    tags = np.full(tagger._tri_adj.shape[0], BODY_LABEL, dtype=np.int32)
    tags[nb[0]] = tags[nb[1]] = 11
    before = tags.copy()
    for label in (11, BODY_LABEL):
        tagger.dilate_label(tags, label)     # must not mutate its argument
        tagger.erode_label(tags, label)
        assert np.array_equal(tags, before)


def test_an_unknown_label_is_a_noop():
    _, tagger = _mesh_tagger()
    tags = np.full(tagger._tri_adj.shape[0], BODY_LABEL, dtype=np.int32)
    tags[0] = 11
    assert np.array_equal(tagger.dilate_label(tags, 42), tags)
    assert np.array_equal(tagger.erode_label(tags, 42), tags)


# ---------------------------------------------------------------------------
# Body: the same two ops, read from the other side
# ---------------------------------------------------------------------------
def _speckled(tagger, seed: int = 0) -> np.ndarray:
    """A tagging where regions touch each other, not just body — which is what
    tells the simultaneous pass apart from the sequential one."""
    rng = np.random.default_rng(seed)
    return rng.choice([BODY_LABEL, 11, 13, 15, 17, 19],
                      size=tagger._tri_adj.shape[0],
                      p=[0.5, 0.1, 0.1, 0.1, 0.1, 0.1]).astype(np.int64)


def test_growing_body_erodes_every_pv_label():
    """Body grows exactly where the PV labels give way — the same rule as
    erode_label, applied to all five against one snapshot."""
    _, tagger = _mesh_tagger()
    tags = _speckled(tagger)
    grown = tagger.dilate_label(tags, BODY_LABEL)

    expected = tags.copy()
    for label in PVS:                       # each read from the same snapshot
        eroded = tagger.erode_label(tags, label)
        expected = np.where(eroded != tags, BODY_LABEL, expected)
    assert np.array_equal(grown, expected)
    assert np.count_nonzero(grown != tags) > 0
    assert not np.isin(grown[grown != tags], PVS).any()   # only ever to body


def test_shrinking_body_dilates_every_pv_label():
    _, tagger = _mesh_tagger()
    tags = _speckled(tagger)
    shrunk = tagger.erode_label(tags, BODY_LABEL)

    expected = tags.copy()
    for label in PVS:
        grown = tagger.dilate_label(tags, label)
        expected = np.where(grown != tags, label, expected)
    assert np.array_equal(shrunk, expected)
    assert np.count_nonzero(shrunk != tags) > 0
    # Only background cells changed, and only into a PV label.
    assert (tags[shrunk != tags] == BODY_LABEL).all()
    assert np.isin(shrunk[shrunk != tags], PVS).all()


@pytest.mark.parametrize("op", ["dilate_label", "erode_label"])
def test_the_body_pass_does_not_depend_on_label_order(op, monkeypatch):
    """The bug this forecloses: doing the labels one after another feeds each
    the previous one's output, so the answer depends on the order LABELS
    happens to be written in."""
    _, tagger = _mesh_tagger()
    tags = _speckled(tagger)
    first = getattr(tagger, op)(tags, BODY_LABEL)

    monkeypatch.setattr(region_tagger, "LABELS",
                        dict(reversed(list(region_tagger.LABELS.items()))))
    assert np.array_equal(getattr(tagger, op)(tags, BODY_LABEL), first)


@pytest.mark.parametrize("op,single", [("dilate_label", "erode_label"),
                                       ("erode_label", "dilate_label")])
def test_the_sequential_form_would_have_differed(op, single):
    """Why the snapshot is not a detail: the same passes run in sequence give
    a different answer once two regions touch."""
    _, tagger = _mesh_tagger()
    tags = _speckled(tagger)
    sequential = tags.copy()
    for label in PVS:
        sequential = getattr(tagger, single)(sequential, label)
    assert not np.array_equal(getattr(tagger, op)(tags, BODY_LABEL), sequential)


def test_shrinking_body_keeps_the_seam_between_two_regions():
    _, tagger = _mesh_tagger()
    c, nb = _interior_cell_with_three_neighbours(tagger)
    tags = np.full(tagger._tri_adj.shape[0], BODY_LABEL, dtype=np.int32)
    tags[nb[0]] = tags[nb[1]] = 11          # two LSPV neighbours ...
    tags[nb[2]] = 13                        # ... but also a LIPV neighbour
    assert tagger.erode_label(tags, BODY_LABEL)[c] == BODY_LABEL


def test_smooth_label_smooths_body_undoably():
    mesh, tagger = _mesh_tagger()
    tags = _speckled(tagger).astype(np.int32)
    mesh.cell_data["elemTag"] = tags.copy()
    editor = ManualEditor(mesh, plotter=None)

    changed = editor.smooth_label(tagger, BODY_LABEL, dilate=True, erode=False)
    assert changed > 0
    assert np.array_equal(np.asarray(mesh.cell_data["elemTag"]),
                          tagger.dilate_label(tags, BODY_LABEL))
    assert editor.can_undo
    editor.undo()
    assert np.array_equal(np.asarray(mesh.cell_data["elemTag"]), tags)


def test_smooth_label_is_undoable():
    mesh, tagger = _mesh_tagger()
    c, nb = _interior_cell_with_three_neighbours(tagger)
    tags = np.full(mesh.n_cells, BODY_LABEL, dtype=np.int32)
    tags[nb[0]] = tags[nb[1]] = 11
    mesh.cell_data["elemTag"] = tags
    editor = ManualEditor(mesh, plotter=None)

    changed = editor.smooth_label(tagger, 11, dilate=True, erode=False)
    assert changed >= 1
    assert mesh.cell_data["elemTag"][c] == 11
    assert editor.can_undo
    editor.undo()
    assert mesh.cell_data["elemTag"][c] == BODY_LABEL   # restored
