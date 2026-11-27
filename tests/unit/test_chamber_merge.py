"""
test_chamber_merge.py
=====================
Putting extracted chambers' results back into their source mesh, on a bar
of tetrahedra tagged along its length: LV, LA body, LSPV ring, the shared
bundle (LA+RA), RA body.

* extraction stores source ids, and the merge uses them;
* without ids (or with wrong ones) points are found by coordinates;
* a moved point is refused;
* fields land on the atrium's rows; elsewhere a new field reads "not
  computed" and an existing field keeps its values;
* nodes the atrium shares with tissue already holding values keep them;
* the regions go back into elemTag in the source numbering, body elements
  only; a region with no tag in the source keeps its elements' tags.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pytest
import pyvista as pv

from ccdaf.core import tag_map
from ccdaf.core.chamber_extract import SOURCE_CELL, SOURCE_POINT, extract_chambers
from ccdaf.core.chamber_merge import NOT_COMPUTED, MergeError, merge_into

NAMES = tag_map.from_json({
    "1": {"name": "LV", "chamber": "LV", "role": "wall"},
    "3": {"name": "LA body", "chamber": "LA", "role": "wall"},
    "4": {"name": "RA body", "chamber": "RA", "role": "wall"},
    "21": {"name": "LSPV ring", "chamber": "LA", "role": "vein"},
    "26": {"name": "BB", "chamber": "LA+RA", "role": "wall"},
})


def _bar():
    bar = pv.ImageData(dimensions=(31, 3, 3)).cast_to_unstructured_grid().triangulate()
    x = np.asarray(bar.cell_centers().points)[:, 0]
    bar.cell_data["elemTag"] = np.select(
        [x < 5, x < 15, x < 18, x < 22], [1, 3, 21, 26], 4)
    bar.cell_data["fiber"] = np.tile([1.0, 0.0, 0.0], (bar.n_cells, 1))
    tag_map.write(bar, NAMES)
    return bar


_CACHE = {}


def _extract(chambers, source=None):
    """An extraction of the bar, computed once per chamber set."""
    key = (tuple(chambers), source is None)
    if key not in _CACHE:
        _CACHE[key] = extract_chambers(source if source is not None else _bar(),
                                       NAMES, chambers).grid
    return _CACHE[key].copy(deep=True)


def _x(grid):
    return np.asarray(grid.cell_centers().points)[:, 0]


def test_extraction_stores_the_source_ids():
    bar = _bar()
    la = _extract(["LA"])
    pid = np.asarray(la.point_data[SOURCE_POINT])
    cid = np.asarray(la.cell_data[SOURCE_CELL])
    assert np.allclose(np.asarray(bar.points)[pid], la.points)
    assert np.array_equal(np.asarray(bar.cell_data["elemTag"])[cid],
                          la.cell_data["elemTag"])


def test_ids_survive_a_second_extraction():
    bar = _bar()
    both = _extract(["LA", "RA"])
    la = extract_chambers(both, NAMES, ["LA"]).grid
    pid = np.asarray(la.point_data[SOURCE_POINT])
    assert np.allclose(np.asarray(bar.points)[pid], la.points)
    assert merge_into(bar, la).method == "ids"


def test_fields_land_on_the_atrium_and_read_not_computed_elsewhere():
    bar = _bar()
    la = _extract(["LA"])
    la.point_data["uac_alpha"] = np.linspace(0, 1, la.n_points)
    la.cell_data["uac_half"] = np.ones(la.n_cells, dtype=np.int8)
    la.cell_data["fiber"] = np.tile([0.0, 1.0, 0.0], (la.n_cells, 1))
    m = merge_into(bar, la)
    assert m.method == "ids"
    g = m.grid
    cid = np.asarray(la.cell_data[SOURCE_CELL])
    pid = np.asarray(la.point_data[SOURCE_POINT])
    out = np.ones(g.n_cells, bool)
    out[cid] = False
    alpha = np.asarray(g.point_data["uac_alpha"])
    assert np.allclose(alpha[pid], la.point_data["uac_alpha"])
    assert np.isnan(np.delete(alpha, pid)).all()
    half = np.asarray(g.cell_data["uac_half"])
    assert (half[cid] == 1).all() and (half[out] == NOT_COMPUTED).all()
    fibre = np.asarray(g.cell_data["fiber"])
    assert np.allclose(fibre[cid], [0, 1, 0]) and np.allclose(fibre[out], [1, 0, 0])
    for name in (SOURCE_POINT, SOURCE_CELL, "openingTag"):
        assert name not in g.point_data and name not in g.cell_data
    assert "uac_alpha" not in bar.point_data            # the source is untouched


def test_without_ids_points_are_found_by_coordinates():
    bar = _bar()
    la = _extract(["LA"])
    la.cell_data["v"] = np.arange(la.n_cells, dtype=float)
    by_ids = merge_into(bar, la).grid
    bare = la.copy()
    del bare.point_data[SOURCE_POINT]
    del bare.cell_data[SOURCE_CELL]
    m = merge_into(bar, bare)
    assert m.method == "coordinates"
    assert np.array_equal(m.grid.cell_data["v"], by_ids.cell_data["v"],
                          equal_nan=True)


def test_wrong_ids_fall_back_to_coordinates():
    bar = _bar()
    la = _extract(["LA"])
    la.cell_data[SOURCE_CELL] = np.roll(la.cell_data[SOURCE_CELL], 1)
    assert merge_into(bar, la).method == "coordinates"


def test_a_moved_point_is_refused():
    bar = _bar()
    la = _extract(["LA"])
    pts = np.asarray(la.points).copy()
    pts[0] += 0.3
    la.points = pts
    with pytest.raises(MergeError, match="not in this mesh"):
        merge_into(bar, la)


def test_shared_nodes_keep_the_first_atriums_values():
    bar = _bar()
    la = _extract(["LA"])
    ra = _extract(["RA"])
    la.point_data["s"] = np.full(la.n_points, 1.0)
    ra.point_data["s"] = np.full(ra.n_points, 2.0)
    m = merge_into(merge_into(bar, la).grid, ra)
    s = np.asarray(m.grid.point_data["s"])
    shared = np.intersect1d(la.point_data[SOURCE_POINT], ra.point_data[SOURCE_POINT])
    assert len(shared) > 0 and m.kept_nodes == len(shared)
    assert (s[shared] == 1).all()
    ra_only = np.setdiff1d(ra.point_data[SOURCE_POINT], shared)
    assert (s[ra_only] == 2).all()


def _tagged_la(bar):
    """The LA as the volume tagging leaves it: regions in elemTag, the
    source tags in sourceTag."""
    la = _extract(["LA"])
    src = np.asarray(la.cell_data["elemTag"]).copy()
    x = _x(la)
    region = np.where(src == 21, 11, 1)             # ring is LSPV, rest body
    region[(src == 3) & (x > 13)] = 11              # LSPV grows into the body
    region[(src == 21) & (x > 17)] = 1              # some ring left in the body
    region[src == 26] = 11                          # bundle in the vein region
    region[(src == 3) & (x < 7)] = 19               # an LAA with no tag here
    la.cell_data["sourceTag"] = src
    la.cell_data["elemTag"] = region
    la.field_data["sourceTagMap"] = np.array([tag_map.encode(NAMES)])
    tag_map.write(la, None)
    return la, src, x


def test_regions_go_back_in_the_source_numbering():
    bar = _bar()
    la, src, x = _tagged_la(bar)
    m = merge_into(bar, la)
    tags = np.asarray(m.grid.cell_data["elemTag"])
    cid = np.asarray(la.cell_data[SOURCE_CELL])
    grown = (src == 3) & (x > 13)
    assert (tags[cid[grown]] == 21).all()
    assert m.retagged == {21: int(grown.sum())}
    # Ring left in the body, and the bundle in the vein, keep their tags.
    assert (tags[cid[src == 21]] == 21).all()
    assert (tags[cid[src == 26]] == 26).all()
    assert m.kept_tags == int(((src == 21) & (x > 17)).sum() + (src == 26).sum())
    # No LAA tag in the source: those body elements stay body.
    laa = (src == 3) & (x < 7)
    assert (tags[cid[laa]] == 3).all() and m.no_tag == {"LAA": int(laa.sum())}
    # Everything outside the atrium is unchanged.
    out = np.ones(bar.n_cells, bool)
    out[cid] = False
    assert np.array_equal(tags[out], np.asarray(bar.cell_data["elemTag"])[out])
    assert "retagged" in m.summary()


def test_untagged_chambers_leave_the_tags_alone():
    bar = _bar()
    la = _extract(["LA"])
    m = merge_into(bar, la)
    assert np.array_equal(m.grid.cell_data["elemTag"], bar.cell_data["elemTag"])
    assert m.tag_mismatch == 0 and not m.retagged
