"""
test_volume_tags.py
===================
Atrial region labels for the tetrahedra of a labelled atrial volume, and
the band at an opening.

Uses the synthetic atrium of ``test_atrial_surfaces``: a spherical shell
two prisms thick with a vein cap (11, "LSPV cap") and a valve (7, "MV").

* the endocardium comes out as an open surface with maps back to the
  volume, and the seeds the recorded openings can tell;
* labels drawn on it reach through the whole wall, and the epicardium
  matches the endocardium;
* islands join their neighbours;
* the band at an opening tags the faces along that opening's edge only.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pytest
import pyvista as pv

from ccdaf.core import atrial_labels as al
from ccdaf.core import tag_map
from ccdaf.core.atrial_surfaces import label_atria
from ccdaf.core.chamber_extract import extract_chambers
from ccdaf.core.opening_band import band_faces, boundary_loops
from ccdaf.core.surface_labels import LABEL_MASK_FIELD, node_label_mask
from ccdaf.core.volume_tags import (
    VOLUME_CELL, VOLUME_NODE, _drop_islands, apply_to_volume, endocardium,
    endocardium_present, seeds_from_openings,
)
from test_atrial_surfaces import LA_MAP, R_IN, R_OUT, _shell


@pytest.fixture(scope="module")
def labelled():
    """The extracted, surface-labelled synthetic left atrium."""
    grid = extract_chambers(_shell(), LA_MAP, ["LA"]).grid
    labels = label_atria(grid, tag_map.read(grid))
    grid.point_data[LABEL_MASK_FIELD] = node_label_mask(grid, labels)
    return grid


def test_the_endocardium_is_an_open_surface_mapped_to_the_volume(labelled):
    assert endocardium_present(labelled) == ["LA"]
    endo = endocardium(labelled, "LA")
    r = np.linalg.norm(np.asarray(endo.points), axis=1)
    assert np.allclose(r, R_IN, atol=0.3)
    assert len(boundary_loops(endo)) == 2              # the cap and the valve
    assert np.allclose(np.asarray(labelled.points)[endo.point_data[VOLUME_NODE]],
                       endo.points)
    assert set(np.unique(endo.cell_data["elemTag"])) == {al.LA.body}
    assert endo.n_cells == len(np.unique(endo.cell_data[VOLUME_CELL]))


def test_a_volume_without_labelled_surfaces_has_no_endocardium():
    with pytest.raises(ValueError):
        endocardium(_shell(), "LA")


def test_the_openings_give_their_seeds(labelled):
    endo = endocardium(labelled, "LA")
    seeds = seeds_from_openings(endo, LA_MAP)
    assert set(seeds) == {"LSPV", "MV"}
    # The vein seed lies inside the vein's opening region, not on its edge.
    assert seeds["LSPV"][2] > 0.8 * R_IN


def _tag_cap_region(endo, label, axis=(0, 0, 1), cos_min=0.7):
    pts = np.asarray(endo.points)
    tri = np.asarray(endo.faces).reshape(-1, 4)[:, 1:]
    c = pts[tri].mean(1)
    c /= np.linalg.norm(c, axis=1, keepdims=True)
    tags = np.asarray(endo.cell_data["elemTag"]).copy()
    tags[c @ np.asarray(axis, float) > cos_min] = label
    endo.cell_data["elemTag"] = tags
    return tags


def test_labels_reach_through_the_wall(labelled):
    endo = endocardium(labelled, "LA")
    _tag_cap_region(endo, al.LA.regions["LSPV"])
    res = apply_to_volume(labelled, endo, "LA")
    assert res.agreement > 0.97
    centres = np.asarray(labelled.cell_centers().points)
    d = centres / np.linalg.norm(centres, axis=1, keepdims=True)
    lspv = res.tags == al.LA.regions["LSPV"]
    # Every layer, inner and outer, near the vein is the vein; far away, body.
    r = np.linalg.norm(centres, axis=1)
    near = d[:, 2] > 0.8
    far = d[:, 2] < 0.5
    assert lspv[near & (r < 22)].mean() > 0.95 and lspv[near & (r > 22)].mean() > 0.95
    assert not lspv[far].any()
    assert r[lspv].max() > R_OUT - 2.0


def test_islands_join_their_main_neighbour():
    # Four tets in a row: 1 1 5 1 — the 5 is a single island in the body,
    # and two separate 1-pieces become one once it joins them.
    tags = np.array([1, 1, 5, 1, 5, 5])
    a = np.array([0, 1, 2, 4])
    b = np.array([1, 2, 3, 5])
    moved = _drop_islands(tags, a, b, np.ones(6, dtype=bool))
    assert tags.tolist() == [1, 1, 1, 1, 5, 5]
    assert moved == {5: 1}


def test_the_band_follows_the_nearest_opening(labelled):
    endo = endocardium(labelled, "LA")
    loops = boundary_loops(endo)
    pts = np.asarray(endo.points)
    top = max(loops, key=lambda nodes: pts[nodes][:, 2].mean())
    faces, loop = band_faces(endo, pts[top].mean(0), width=3.0)
    assert set(loop.tolist()) == set(top.tolist())
    tri = np.asarray(endo.faces).reshape(-1, 4)[:, 1:]
    c = pts[tri[faces]].mean(1)
    # Near the cap's edge only: high on the sphere, none by the valve.
    assert c[:, 2].min() > 0.75 * R_IN
    assert len(faces) > 0


def test_a_closed_surface_has_no_opening_band():
    with pytest.raises(ValueError):
        band_faces(pv.Sphere(), (0, 0, 1), 1.0)


def test_unassigned_triangles_count_as_body(labelled):
    from ccdaf.core.mesh_loader import UNASSIGNED
    endo = endocardium(labelled, "LA")
    tags = _tag_cap_region(endo, al.LA.regions["LSPV"])
    tags[tags == al.LA.body] = UNASSIGNED
    endo.cell_data["elemTag"] = tags
    res = apply_to_volume(labelled, endo, "LA")
    assert set(np.unique(res.tags)) == {al.LA.body, al.LA.regions["LSPV"]}


def test_the_tagger_grows_a_vein_from_its_opening(labelled):
    from ccdaf.core.region_tagger import RegionTagger
    from ccdaf.core.volume_tags import opening_loops
    endo = endocardium(labelled, "LA")
    loops = opening_loops(endo, LA_MAP)
    assert set(loops) == {"LSPV", "MV"}
    pts = np.asarray(endo.points)
    # Every region needs a seed; the four absent veins and the LAA go far
    # from the cap, the LSPV seed off to one side of its opening.
    seeds = {"LSPV": int(loops["LSPV"][0])}
    for i, n in enumerate(("LIPV", "RSPV", "RIPV", "LAA")):
        ang = np.pi / 2 + i * 0.4
        seeds[n] = int(np.argmin(np.linalg.norm(
            pts - R_IN * np.array([np.cos(ang), np.sin(ang), -0.3]), axis=1)))
    tri = np.asarray(endo.faces).reshape(-1, 4)[:, 1:]
    edge = np.isin(tri, loops["LSPV"]).any(1)
    tags = RegionTagger(endo.copy()).tag(
        seeds, atrium=al.LA, sources={"LSPV": loops["LSPV"]})
    assert (tags[edge] == al.LA.regions["LSPV"]).mean() > 0.9


def test_known_openings_get_a_band(labelled):
    from ccdaf.core.volume_tags import band_openings
    endo = endocardium(labelled, "LA")
    counts = band_openings(endo, LA_MAP, "LA", width=3.0)
    assert set(counts) == {"LSPV"}                 # the valve is no region
    tags = np.asarray(endo.cell_data["elemTag"])
    pts = np.asarray(endo.points)
    tri = np.asarray(endo.faces).reshape(-1, 4)[:, 1:]
    c = pts[tri].mean(1)
    lspv = tags == al.LA.regions["LSPV"]
    assert lspv.sum() == counts["LSPV"] > 0
    assert c[lspv][:, 2].min() > 0.75 * R_IN       # only by the vein's cap
    assert set(np.unique(tags)) == {al.LA.body, al.LA.regions["LSPV"]}


def test_the_previous_tags_draw_with_their_names():
    from ccdaf.app.ccdaf import CATEGORICAL_FIELDS, SOURCE_TAG, _region_legend
    assert SOURCE_TAG in CATEGORICAL_FIELDS
    _, _, names = _region_legend(np.array([3, 21]), anatomical=True,
                                 mapping={3: "LA body", 21: "LSPV ring"})
    assert list(names.values()) == ["LA body", "LSPV ring"]


def test_a_band_stops_halfway_to_the_next_opening():
    # A flat strip with two slits close together: a band wider than the gap
    # round one slit must not reach the other.
    plane = pv.Plane(i_size=20, j_size=10, i_resolution=80, j_resolution=40
                     ).triangulate()
    c = np.asarray(plane.cell_centers().points)
    hole = (np.abs(c[:, 1]) < 1.0) & ((np.abs(c[:, 0] + 2) < 1.0)
                                     | (np.abs(c[:, 0] - 2) < 1.0))
    strip = plane.extract_cells(np.where(~hole)[0]).extract_surface()
    faces, _loop = band_faces(strip, (-2.0, 0.0, 0.0), width=4.0)
    x = np.asarray(strip.cell_centers().points)[faces][:, 0]
    assert x.max() < 0.5                         # never past the midpoint


def test_the_legend_keeps_names_with_unassigned_triangles():
    from ccdaf.app.ccdaf import _region_legend
    _, _, names = _region_legend(np.array([-1, 1, 11]))
    assert names[0] == "unassigned"
    assert "LA body" in names.values() and "LSPV" in names.values()


def test_fill_holes_never_fills_the_body():
    from ccdaf.core.region_tagger import RegionTagger
    sphere = pv.Sphere(theta_resolution=20, phi_resolution=20)
    t = RegionTagger(sphere)
    tags = np.full(sphere.n_cells, al.LA.body)
    z = np.asarray(sphere.cell_centers().points)[:, 2]
    tags[z > 0.3] = al.LA.regions["LAA"]          # one region, body below
    out = t._fill_holes(tags)
    assert (out == al.LA.body).sum() == (tags == al.LA.body).sum()
