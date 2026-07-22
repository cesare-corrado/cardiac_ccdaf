"""
test_snake_geodesic.py
======================
The manual-tagging "snake": an open geodesic drawn through user-picked
surface points whose incident triangles are tagged with the active label.

Geometry lives on ``RegionTagger`` (reusing the segmentation vertex graph):

* ``geodesic_path(a, b)`` — a shortest vertex path over the mesh 1-skeleton;
  ``[a]`` when ``a == b``; ``[]`` when the two are disconnected;
* ``triangles_incident_to(vertex_ids)`` — cells with >=1 vertex on the path.

The path is grown *bidirectionally* on ``ManualEditor`` (``_snake_extend``):
each new anchor extends whichever endpoint — head or tail — reaches it by the
shorter geodesic, exactly like the PV clip snake. The undoable apply lives on
``ManualEditor.commit_snake``: it tags the incident triangles and is a no-op
for body / fewer than two anchors.

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


def _sphere_tagger():
    mesh = pv.Sphere(theta_resolution=24, phi_resolution=24).triangulate()
    return mesh, RegionTagger(mesh)


def _two_disjoint_triangles():
    """A PolyData with two triangles that share no vertex (disconnected)."""
    pts = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0],          # triangle 0: verts 0,1,2
         [5, 0, 0], [6, 0, 0], [5, 1, 0]],         # triangle 1: verts 3,4,5
        dtype=float,
    )
    faces = np.hstack([[3, 0, 1, 2], [3, 3, 4, 5]])
    return pv.PolyData(pts, faces)


def _edge_exists(graph, u, v) -> bool:
    return graph[int(u), int(v)] != 0.0


# ---------------------------------------------------------------------------
# geodesic_path
# ---------------------------------------------------------------------------
def test_geodesic_path_endpoints_and_adjacency():
    _, tagger = _sphere_tagger()
    a, b = 0, tagger._points.shape[0] - 1
    path = tagger.geodesic_path(a, b)
    assert len(path) >= 2
    assert path[0] == a and path[-1] == b
    # Every step is a real mesh edge.
    for u, v in zip(path[:-1], path[1:]):
        assert _edge_exists(tagger._graph, u, v)


def test_geodesic_path_same_vertex_is_single():
    _, tagger = _sphere_tagger()
    assert tagger.geodesic_path(7, 7) == [7]


def test_geodesic_path_disconnected_is_empty():
    tagger = RegionTagger(_two_disjoint_triangles())
    assert tagger.geodesic_path(0, 3) == []


# ---------------------------------------------------------------------------
# bidirectional growth (_snake_extend)
# ---------------------------------------------------------------------------
def test_snake_extends_the_nearer_endpoint():
    mesh, tagger = _sphere_tagger()
    editor = ManualEditor(mesh, plotter=None)
    editor.set_active_label(11)

    a = 0
    far = mesh.n_points - 1                       # roughly antipodal to vertex 0
    near_a = int(tagger._graph[a].indices[0])     # an immediate neighbour of a

    assert editor._snake_extend(tagger, a) == 1
    assert editor._snake_head == a and editor._snake_tail == a
    assert editor._snake_extend(tagger, far) == 2
    assert editor._snake_head == far and editor._snake_tail == a

    # near_a is one edge from the tail (a) but far from the head (far), so it
    # must extend the TAIL, not the head — this is the bidirectional choice.
    assert editor._snake_extend(tagger, near_a) == 3
    assert editor._snake_tail == near_a           # tail moved to the near pick
    assert editor._snake_head == far              # head unchanged
    assert editor._snake_path[0] == near_a and editor._snake_path[-1] == far
    # Still one continuous walk on the mesh.
    for u, v in zip(editor._snake_path[:-1], editor._snake_path[1:]):
        assert _edge_exists(tagger._graph, u, v)


def test_undo_last_point_restores_previous_geodesic():
    mesh, tagger = _sphere_tagger()
    editor = ManualEditor(mesh, plotter=None)
    editor.set_active_label(11)

    editor._snake_extend(tagger, 0)
    editor._snake_extend(tagger, mesh.n_points - 1)
    two_path = list(editor._snake_path)
    editor._snake_extend(tagger, 100)             # third anchor
    assert editor.snake_point_count == 3

    # Undo the third point → exactly the two-anchor state is back.
    assert editor.undo_last_point() == 2
    assert editor._snake_path == two_path
    assert editor.snake_point_count == 2

    # Undo down to one, then empty, then nothing-to-undo.
    assert editor.undo_last_point() == 1
    assert editor._snake_path == [0]
    assert editor.undo_last_point() == 0
    assert editor._snake_path == [] and editor.snake_point_count == 0
    assert editor.undo_last_point() == -1         # empty history


def test_snake_extend_repeat_endpoint_and_disconnected():
    mesh, tagger = _sphere_tagger()
    editor = ManualEditor(mesh, plotter=None)
    editor.set_active_label(11)
    editor._snake_extend(tagger, 0)
    editor._snake_extend(tagger, 50)
    assert editor._snake_extend(tagger, 50) == 0    # repeat pick on the head
    assert editor._snake_extend(tagger, 0) == 0     # repeat pick on the tail

    # A disconnected mesh: the second anchor is unreachable from the first.
    dmesh = _two_disjoint_triangles()
    dtag = RegionTagger(dmesh)
    ded = ManualEditor(dmesh, plotter=None)
    ded.set_active_label(11)
    assert ded._snake_extend(dtag, 0) == 1
    assert ded._snake_extend(dtag, 3) == -1         # other component


# ---------------------------------------------------------------------------
# triangles_incident_to
# ---------------------------------------------------------------------------
def test_triangles_incident_to_partitions_cells():
    _, tagger = _sphere_tagger()
    verts = {0, 1, 2, 3}
    cells = set(tagger.triangles_incident_to(verts).tolist())
    tris = tagger._triangles
    for c in range(tris.shape[0]):
        touches = bool(set(tris[c].tolist()) & verts)
        assert (c in cells) == touches
    assert tagger.triangles_incident_to([]).size == 0


# ---------------------------------------------------------------------------
# ManualEditor.commit_snake
# ---------------------------------------------------------------------------
def test_commit_snake_tags_incident_triangles_and_is_undoable():
    mesh, tagger = _sphere_tagger()
    mesh.cell_data["elemTag"] = np.full(mesh.n_cells, BODY_LABEL, dtype=np.int32)
    editor = ManualEditor(mesh, plotter=None)
    editor.set_active_label(11)
    editor._snake_extend(tagger, 0)
    editor._snake_extend(tagger, mesh.n_points - 1)

    expected = tagger.triangles_incident_to(editor._snake_path)

    n = editor.commit_snake(tagger)
    assert n == expected.size and n > 0
    tags = np.asarray(mesh.cell_data["elemTag"])
    assert np.all(tags[expected] == 11)
    assert editor.can_undo
    assert editor.snake_point_count == 0            # anchors consumed on commit

    editor.undo()
    assert np.all(np.asarray(mesh.cell_data["elemTag"]) == BODY_LABEL)


def test_commit_snake_body_and_too_few_points_are_noops():
    mesh, tagger = _sphere_tagger()
    mesh.cell_data["elemTag"] = np.full(mesh.n_cells, BODY_LABEL, dtype=np.int32)
    editor = ManualEditor(mesh, plotter=None)

    # body active label: builds no geodesic
    editor.set_active_label(BODY_LABEL)
    editor._snake_extend(tagger, 0)
    editor._snake_extend(tagger, 50)
    assert editor.commit_snake(tagger) == 0
    assert not editor.can_undo

    # fewer than two anchors
    editor = ManualEditor(mesh, plotter=None)
    editor.set_active_label(11)
    editor._snake_extend(tagger, 0)
    assert editor.commit_snake(tagger) == 0
    assert not editor.can_undo
