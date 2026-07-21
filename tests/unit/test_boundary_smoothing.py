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
* body is not a smoothing target, and neither op mutates its input;
* `smooth_label` snapshots for undo only when something actually changes.

Uses synthetic meshes (no real EAM data required).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pyvista as pv

from ccdaf.core.mesh_loader import BODY_LABEL
from ccdaf.core.region_tagger import RegionTagger
from ccdaf.interaction.manual_editor import ManualEditor


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


def test_body_and_input_are_left_alone():
    _, tagger = _mesh_tagger()
    c, nb = _interior_cell_with_three_neighbours(tagger)
    tags = np.full(tagger._tri_adj.shape[0], BODY_LABEL, dtype=np.int32)
    tags[nb[0]] = tags[nb[1]] = 11
    before = tags.copy()
    # body is not a PV label -> no-op
    assert np.array_equal(tagger.dilate_label(tags, BODY_LABEL), before)
    tagger.dilate_label(tags, 11)           # must not mutate its argument
    assert np.array_equal(tags, before)


def test_smooth_label_is_undoable_and_body_is_a_noop():
    mesh, tagger = _mesh_tagger()
    c, nb = _interior_cell_with_three_neighbours(tagger)
    tags = np.full(mesh.n_cells, BODY_LABEL, dtype=np.int32)
    tags[nb[0]] = tags[nb[1]] = 11
    mesh.cell_data["elemTag"] = tags
    editor = ManualEditor(mesh, plotter=None)

    # body: nothing happens, no undo snapshot
    assert editor.smooth_label(tagger, BODY_LABEL, dilate=True) == 0
    assert not editor.can_undo

    changed = editor.smooth_label(tagger, 11, dilate=True, erode=False)
    assert changed >= 1
    assert mesh.cell_data["elemTag"][c] == 11
    assert editor.can_undo
    editor.undo()
    assert mesh.cell_data["elemTag"][c] == BODY_LABEL   # restored
