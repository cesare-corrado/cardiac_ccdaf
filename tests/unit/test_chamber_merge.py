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
from ccdaf.core.chamber_extract import (
    SOURCE_CELL, SOURCE_CELLS, SOURCE_POINT, extract_chambers,
)
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


def test_a_moved_interface_point_is_refused():
    bar = _bar()
    la = _extract(["LA"])
    pts = np.asarray(la.points).copy()
    pts[0] += 0.3
    la.points = pts
    # pts[0] is on the interface, so the remeshed reading fails as well.
    with pytest.raises(MergeError, match="not where they were"):
        merge_into(bar, la)
    bare = la.copy()
    del bare.field_data[SOURCE_CELLS]
    with pytest.raises(MergeError, match="not in this mesh"):
        merge_into(bar, bare)


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


# ---------------------------------------------------------------------
# Remeshed chambers: replaced, joined at the interface
# ---------------------------------------------------------------------
from ccdaf.core.chamber_extract import INTERFACE  # noqa: E402
from ccdaf.core.volume_mesh import tetrahedra  # noqa: E402
from ccdaf.core.volume_postprocessor import RemeshOptions, remesh  # noqa: E402


def _faces(tets):
    f = np.sort(np.concatenate([tets[:, [0, 1, 2]], tets[:, [0, 1, 3]],
                                tets[:, [0, 2, 3]], tets[:, [1, 2, 3]]]), 1)
    return np.unique(f, axis=0, return_counts=True)


def _volume(grid):
    return float(grid.compute_cell_sizes(length=False, area=False)
                 .cell_data["Volume"].sum())


def test_extraction_records_the_interface_and_the_elements():
    bar = _bar()
    la = _extract(["LA"])
    on = np.asarray(la.point_data[INTERFACE]) > 0
    x = np.asarray(la.points)[:, 0]
    # The LA touches the LV at x = 5, and the RA where the bundle is split.
    assert on.any() and set(np.round(x[on])) == {5, 20}
    cells = np.asarray(la.field_data[SOURCE_CELLS])
    assert np.array_equal(cells, np.sort(la.cell_data[SOURCE_CELL]))
    assert len(cells) < bar.n_cells


@pytest.fixture(scope="module")
def remeshed():
    la = _extract(["LA"])
    la.cell_data["v"] = np.ones(la.n_cells)
    out = remesh(la, RemeshOptions(target_edge=0.6, freeze_boundary=False,
                                   fix_interface=True))
    return la, out


def test_a_remesh_keeps_the_interface_exactly(remeshed):
    la, out = remeshed
    assert out.n_cells > la.n_cells
    on = np.asarray(la.point_data[INTERFACE]) > 0
    back = np.asarray(out.point_data[INTERFACE]) > 0
    assert back.sum() == on.sum()
    assert np.array_equal(np.sort(np.asarray(out.point_data[SOURCE_POINT])[back]),
                          np.sort(np.asarray(la.point_data[SOURCE_POINT])[on]))
    assert (np.asarray(out.cell_data[SOURCE_CELL]) == -1).all()
    assert np.array_equal(out.field_data[SOURCE_CELLS], la.field_data[SOURCE_CELLS])


def test_remeshed_chambers_replace_their_elements(remeshed):
    bar = _bar()
    la, out = remeshed
    m = merge_into(bar, out)
    assert m.method == "replaced" and m.removed == la.n_cells
    g = m.grid
    _u, count = _faces(tetrahedra(g))
    assert count.max() <= 2                                  # conforming
    _u0, count0 = _faces(tetrahedra(bar))
    assert np.isclose(_volume(g), _volume(bar), rtol=1e-6)
    # The other chambers are untouched, element for element.
    tags = np.asarray(g.cell_data["elemTag"])
    for t in (1, 4):
        assert (tags == t).sum() == (np.asarray(bar.cell_data["elemTag"]) == t).sum()
    assert (tags == 3).sum() > (np.asarray(bar.cell_data["elemTag"]) == 3).sum()
    # A field only the chambers had reads "not computed" elsewhere.
    v = np.asarray(g.cell_data["v"])
    assert (v[tags == 3] == 1).all() and np.isnan(v[tags == 1]).all()
    assert "Replaced" in m.summary()
    # Extracting again gives the remeshed LA back; the shared bundle is split
    # afresh on the new elements, so a few of its may change side.
    again = extract_chambers(g, NAMES, ["LA"]).grid
    own = np.asarray(out.cell_data["elemTag"]) != 26
    assert (np.asarray(again.cell_data["elemTag"]) != 26).sum() == own.sum()
    assert abs(again.n_cells - out.n_cells) < 0.01 * out.n_cells


def test_a_moved_interface_is_refused():
    la = _extract(["LA"])
    out = remesh(la, RemeshOptions(target_edge=0.6, freeze_boundary=False))
    with pytest.raises(MergeError, match="interface nodes"):
        merge_into(_bar(), out)


def test_source_tags_ride_through_the_remesh_exactly():
    la, src, _x0 = _tagged_la(_bar())
    out = remesh(la, RemeshOptions(target_edge=0.6, fix_interface=True))
    pairs_before = set(zip(np.asarray(la.cell_data["elemTag"]).tolist(), src.tolist()))
    pairs_after = set(zip(np.asarray(out.cell_data["elemTag"]).tolist(),
                          np.asarray(out.cell_data["sourceTag"]).tolist()))
    assert pairs_after <= pairs_before
