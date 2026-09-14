"""
test_tissue_property.py
=======================
Actions → Assign tissue property: the rules in ``ccdaf.core.tissue_property``.

The contract:

* only one-component float fields with data are offered — never a label
  field (``elemTag`` stored as float32 included), a vector, or bookkeeping;
* a point field is averaged per element over its finite vertices, and every
  statistic is weighted by element size;
* the healthy estimate recovers the healthy tissue's mean and SD with and
  without an infarct, and does not depend on the seed percentile while the
  seeds stay in healthy tissue;
* masked tissue in the selection (SD = 0) blocks, and one value repeated
  over more than 5% of the tissue warns;
* mean + k·SD: healthy below the first cut, each region up to the next cut,
  the top one open. Thresholds: rows half-open, the top row closed, and the
  rows must cover the data, while a row with no data in it is fine;
* excluded tissue gets its own ID, every element gets one, and none is
  negative;
* a no-data element takes its nearest measured neighbour's value along the
  mesh, never across excluded tissue, and one with nothing reachable blocks.

Synthetic meshes; no display, no Qt.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pytest
import pyvista as pv
import scipy.sparse as sp

from ccdaf.core import tissue_property as tp


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------
def _block(n: int = 12) -> pv.UnstructuredGrid:
    grid = pv.ImageData(dimensions=(n, n, n), spacing=(1.0, 1.0, 1.0)).triangulate()
    grid.cell_data["elemTag"] = np.ones(grid.n_cells, dtype=np.int32)
    return grid


def _lge(grid, *, infarct: bool, seed: int = 0) -> np.ndarray:
    """Healthy noise, plus a bright corner when *infarct*. Returns the core points."""
    rng = np.random.default_rng(seed)
    points = np.asarray(grid.points)
    lge = 0.12 + 0.045 * rng.standard_normal(grid.n_points)
    core = np.zeros(grid.n_points, dtype=bool)
    if infarct:
        core = np.linalg.norm(points - points.min(axis=0), axis=1) < 5.0
        lge[core] += 0.45
    grid.point_data["lge"] = lge
    return core


def _strip(values, tags=None) -> pv.PolyData:
    """Separate triangles, one cell value each."""
    points, faces = [], []
    for i in range(len(values)):
        base = len(points)
        points += [[i, 0.0, 0.0], [i + 1.0, 0.0, 0.0], [i, 1.0, 0.0]]
        faces += [3, base, base + 1, base + 2]
    mesh = pv.PolyData(np.asarray(points, dtype=float), faces=np.asarray(faces))
    mesh.cell_data["v"] = np.asarray(values, dtype=float)
    mesh.cell_data["elemTag"] = (np.ones(len(values), dtype=np.int32) if tags is None
                                 else np.asarray(tags, dtype=np.int32))
    return mesh


def _thresholds(edges, ids, tissue=(1,), excluded=None) -> tp.TissuePropertyOptions:
    return tp.TissuePropertyOptions(
        field="v", association=tp.CELL, tissue=tuple(tissue),
        excluded_ids=dict(excluded or {}), criterion=tp.CRITERION_THRESHOLDS,
        region_ids=tuple(ids), edges=tuple(edges))


def _dice(a, b, w) -> float:
    return 2.0 * w[a & b].sum() / (w[a].sum() + w[b].sum())


# ---------------------------------------------------------------------------
# Fields, values, sizes, neighbours
# ---------------------------------------------------------------------------
def test_only_scalar_float_fields_with_data_are_offered():
    mesh = pv.Sphere(theta_resolution=8, phi_resolution=8).triangulate()
    mesh.cell_data["elemTag"] = np.ones(mesh.n_cells, dtype=np.float32)  # as a saved VTK
    mesh.cell_data["tissueTag"] = np.zeros(mesh.n_cells, dtype=np.int32)
    mesh.cell_data["fiber"] = np.tile([1.0, 0.0, 0.0], (mesh.n_cells, 1))
    mesh.cell_data["render_idx"] = np.zeros(mesh.n_cells)
    mesh.cell_data["thickness"] = np.ones(mesh.n_cells)
    mesh.point_data["LAT"] = np.linspace(0.0, 1.0, mesh.n_points)
    mesh.point_data["A1"] = np.full(mesh.n_points, np.nan)
    mesh.point_data["count"] = np.arange(mesh.n_points)
    assert set(tp.eligible_fields(mesh)) == {("LAT", tp.POINT), ("thickness", tp.CELL)}


def test_a_point_field_is_averaged_over_finite_vertices():
    mesh = pv.PolyData(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float),
                       faces=np.array([3, 0, 1, 2, 3, 1, 3, 2]))
    mesh.point_data["v"] = np.array([1.0, np.nan, 3.0, np.nan])
    values = tp.element_values(mesh, "v", tp.POINT)
    assert values[0] == pytest.approx(2.0)          # (1 + 3) / 2, the NaN left out
    assert values[1] == pytest.approx(3.0)          # the one vertex with data
    mesh.point_data["v"] = np.full(4, np.nan)
    assert np.isnan(tp.element_values(mesh, "v", tp.POINT)).all()


def test_statistics_are_weighted_by_element_size():
    mesh = pv.PolyData(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                                 [5, 0, 0], [8, 0, 0], [5, 3, 0]], dtype=float),
                       faces=np.array([3, 0, 1, 2, 3, 3, 4, 5]))
    sizes = tp.element_sizes(mesh)
    assert sizes == pytest.approx([0.5, 4.5])
    mean, _ = tp.weighted_mean_sd(np.array([0.0, 1.0]), sizes)
    assert mean == pytest.approx(0.9)


def test_weighted_quantile_follows_the_weights():
    values = np.array([0.0, 1.0, 2.0])
    assert tp.weighted_quantile(values, np.array([1.0, 1.0, 1.0]), 0.5) == 1.0
    assert tp.weighted_quantile(values, np.array([10.0, 1.0, 1.0]), 0.5) == 0.0


def test_adjacency_links_elements_that_share_a_facet():
    sphere = pv.Sphere(theta_resolution=10, phi_resolution=10).triangulate()
    graph = tp.element_adjacency(sphere)
    assert (graph != graph.T).nnz == 0
    assert np.all(np.diff(graph.indptr) == 3)       # a closed triangle surface
    block = tp.element_adjacency(_block(4))
    neighbours = np.diff(block.indptr)
    assert neighbours.min() >= 1 and neighbours.max() <= 4
    assert graph.data.min() > 0.0


def test_labels_must_be_whole_numbers():
    mesh = _strip([0.1, 0.2])
    mesh.cell_data["elemTag"] = np.array([1.0, 2.0], dtype=np.float32)
    assert tp.element_labels(mesh).tolist() == [1, 2]
    mesh.cell_data["elemTag"] = np.array([1.0, 1.5])
    with pytest.raises(ValueError, match="whole numbers"):
        tp.element_labels(mesh)


# ---------------------------------------------------------------------------
# Healthy estimate
# ---------------------------------------------------------------------------
def test_the_estimate_recovers_healthy_tissue_without_an_infarct():
    grid = _block()
    _lge(grid, infarct=False)
    values = tp.element_values(grid, "lge", tp.POINT)
    weights = tp.element_sizes(grid)
    estimate = tp.estimate_healthy(values, weights, tp.element_adjacency(grid),
                                   np.ones(grid.n_cells, dtype=bool))
    mean, sd = tp.weighted_mean_sd(values, weights)
    assert estimate.settled
    assert weights[estimate.region].sum() / weights.sum() > 0.97
    assert estimate.mean == pytest.approx(mean, abs=0.1 * sd)
    assert estimate.sd == pytest.approx(sd, rel=0.05)


def test_the_estimate_matches_the_true_healthy_reference_with_an_infarct():
    grid = _block()
    core = _lge(grid, infarct=True)
    healthy = ~core[tp.element_connectivity(grid)].any(axis=1)
    values = tp.element_values(grid, "lge", tp.POINT)
    weights = tp.element_sizes(grid)
    true_mean, true_sd = tp.weighted_mean_sd(values[healthy], weights[healthy])

    result = tp.assign_tissue_property(
        grid, tp.TissuePropertyOptions(field="lge", association=tp.POINT, tissue=(1,)))
    assert result.mean == pytest.approx(true_mean, abs=0.15 * true_sd)
    assert result.sd == pytest.approx(true_sd, rel=0.1)
    true_scar = values >= true_mean + 3.0 * true_sd
    assert _dice(result.tags == 2, true_scar, weights) > 0.9


@pytest.mark.parametrize("percentile", [10.0, 35.0])
def test_the_seed_percentile_does_not_change_the_answer(percentile):
    grid = _block()
    _lge(grid, infarct=True)
    values = tp.element_values(grid, "lge", tp.POINT)
    weights = tp.element_sizes(grid)
    graph = tp.element_adjacency(grid)
    tissue = np.ones(grid.n_cells, dtype=bool)
    reference = tp.estimate_healthy(values, weights, graph, tissue)
    other = tp.estimate_healthy(values, weights, graph, tissue, seed_percentile=percentile)
    assert other.mean == pytest.approx(reference.mean, rel=0.02)
    assert other.sd == pytest.approx(reference.sd, rel=0.02)


def test_masked_tissue_in_the_selection_blocks():
    grid = _block()
    _lge(grid, infarct=False)
    points = np.asarray(grid.points)
    grid.point_data["lge"][points[:, 0] <= 4.0] = 0.0            # masked, one third
    centres = np.asarray(grid.cell_centers().points)
    grid.cell_data["elemTag"] = np.where(centres[:, 0] < 4.0, 2, 1).astype(np.int32)

    both = tp.TissuePropertyOptions(field="lge", association=tp.POINT, tissue=(1, 2))
    with pytest.raises(ValueError, match="no spread"):
        tp.assign_tissue_property(grid, both)

    lv = tp.TissuePropertyOptions(field="lge", association=tp.POINT, tissue=(1,),
                                  excluded_ids={2: 3})
    result = tp.assign_tissue_property(grid, lv)
    assert result.sd > 0.0
    assert set(np.unique(result.tags[grid.cell_data["elemTag"] == 2])) == {3}


def test_one_value_over_five_percent_of_the_tissue_warns():
    grid = _block()
    _lge(grid, infarct=False)
    values = tp.element_values(grid, "lge", tp.POINT)
    values[: len(values) // 10] = 0.05
    grid.cell_data["clipped"] = values
    result = tp.assign_tissue_property(
        grid, tp.TissuePropertyOptions(field="clipped", association=tp.CELL, tissue=(1,)))
    assert any("exactly the value" in note for note in result.warnings)


def test_a_typed_mean_outside_the_data_warns():
    mesh = _strip([0.0, 1.0, 2.0])
    options = tp.TissuePropertyOptions(field="v", association=tp.CELL, tissue=(1,),
                                       auto=False, mean=50.0, sd=1.0)
    result = tp.assign_tissue_property(mesh, options)
    assert any("outside the field's range" in note for note in result.warnings)


# ---------------------------------------------------------------------------
# Rows and IDs
# ---------------------------------------------------------------------------
def test_mean_sd_rows_start_at_each_cut_and_the_top_region_is_open():
    mesh = _strip([0.0, 1.9, 2.0, 2.9, 3.0, 50.0])
    options = tp.TissuePropertyOptions(field="v", association=tp.CELL, tissue=(1,),
                                       region_ids=(1, 2), ks=(2.0, 3.0), healthy_id=0,
                                       auto=False, mean=0.0, sd=1.0)
    assert tp.assign_tissue_property(mesh, options).tags.tolist() == [0, 0, 1, 1, 2, 2]


def test_threshold_rows_are_half_open_and_the_top_row_is_closed():
    mesh = _strip([0.0, 0.25, 0.5, 0.75, 1.0])
    result = tp.assign_tissue_property(mesh, _thresholds((0.0, 0.5, 1.0), (1, 2)))
    assert result.tags.tolist() == [1, 1, 2, 2, 2]


def test_threshold_rows_must_cover_the_data():
    mesh = _strip([0.0, 0.5, 1.0])
    with pytest.raises(ValueError, match="cover"):
        tp.assign_tissue_property(mesh, _thresholds((0.1, 0.5, 1.0), (1, 2)))
    with pytest.raises(ValueError, match="cover"):
        tp.assign_tissue_property(mesh, _thresholds((0.0, 0.5, 0.9), (1, 2)))


def test_rows_may_extend_past_the_data_and_hold_none_of_it():
    mesh = _strip([0.0, 0.5, 1.0])
    result = tp.assign_tissue_property(mesh, _thresholds((-1.0, 0.5, 1.0, 2.0), (1, 2, 3)))
    assert result.tags.tolist() == [1, 2, 3]
    result = tp.assign_tissue_property(mesh, _thresholds((-1.0, 0.5, 5.0, 9.0), (1, 2, 3)))
    assert result.tags.tolist() == [1, 2, 2]
    assert 3 not in result.counts


def test_excluded_tissue_gets_its_own_id_and_no_id_is_negative():
    mesh = _strip([0.1, 0.2, 0.3, 0.0], tags=[1, 1, 1, 2])
    result = tp.assign_tissue_property(
        mesh, _thresholds((0.0, 0.5), (5,), excluded={2: 7}))
    assert result.tags.tolist() == [5, 5, 5, 7]
    assert result.tags.dtype == np.int32
    assert result.tags.min() >= 0


def test_every_tissue_needs_an_id():
    mesh = _strip([0.1, 0.2], tags=[1, 2])
    with pytest.raises(ValueError, match="neither classified nor excluded"):
        tp.assign_tissue_property(mesh, _thresholds((0.0, 0.5), (1,)))


@pytest.mark.parametrize("value, used, expected", [
    (2, [0, 1, 2], 3),          # the right ventricle, with scar already 2
    (11, [0, 1, 2], 11),        # a vein keeps its number
    (2, [0, 1, 3], 2),
    (0, [0, 1, 2, 3], 4),
])
def test_default_excluded_ids(value, used, expected):
    assert tp.default_excluded_id(value, used) == expected


@pytest.mark.parametrize("changes, message", [
    (dict(tissue=()), "Tick at least one tissue"),
    (dict(region_ids=()), "between"),
    (dict(region_ids=tuple(range(1, 12)), ks=tuple(float(k) for k in range(1, 12))), "between"),
    (dict(region_ids=(1, 1)), "share an ID"),
    (dict(healthy_id=2), "healthy ID equals"),
    (dict(region_ids=(1, -2)), "negative"),
    (dict(excluded_ids={2: -1}), "negative"),
    (dict(ks=(3.0, 2.0)), "strictly increasing"),
    (dict(ks=(0.0, 3.0)), "greater than 0"),
    (dict(ks=(2.0,)), "one k value per region"),
    (dict(auto=False, mean=0.1, sd=0.0), "SD must be greater than 0"),
    (dict(auto=False, mean=None, sd=None), "finite healthy mean"),
    (dict(seed_percentile=60.0), "seed percentile"),
    (dict(growth_limit=1.0), "growth limit"),
    (dict(tolerance_percent=0.0), "tolerance"),
    (dict(criterion=tp.CRITERION_THRESHOLDS, edges=(0.0, 0.5, 0.5)), "greater than its 'from'"),
    (dict(criterion=tp.CRITERION_THRESHOLDS, edges=(0.0, 1.0)), "finite 'from' and 'to'"),
    (dict(tissue=(1, 2), excluded_ids={2: 5}), "both classified and excluded"),
])
def test_options_that_must_stop_apply(changes, message):
    base = dict(field="v", association=tp.CELL, tissue=(1,))
    base.update(changes)
    with pytest.raises(ValueError, match=message):
        tp.TissuePropertyOptions(**base).validate()


def test_options_worth_confirming():
    options = tp.TissuePropertyOptions(field="v", association=tp.CELL, tissue=(1,),
                                       seed_percentile=40.0, excluded_ids={2: 0, 3: 9, 4: 9})
    notes = options.warnings()
    assert any("percentile" in note for note in notes)
    assert any("elemTag 2 → 0" in note for note in notes)
    # Excluded tissues sharing an ID with each other is a choice, not a problem.
    assert not any("elemTag 3" in note for note in notes)


def test_availability_says_why_not():
    assert tp.availability(None)[0] is False
    bare = pv.Sphere().triangulate()
    assert tp.availability(bare) == (False, "The mesh has no elemTag array.")
    bare.cell_data["elemTag"] = np.ones(bare.n_cells, dtype=np.int32)
    enabled, why = tp.availability(bare)
    assert enabled is False and "scalar" in why
    bare.point_data["LAT"] = np.linspace(0.0, 1.0, bare.n_points)
    assert tp.availability(bare)[0] is True


# ---------------------------------------------------------------------------
# No data
# ---------------------------------------------------------------------------
def _path_graph(n: int) -> sp.csr_matrix:
    ones = np.ones(n - 1)
    return sp.diags([ones, ones], [-1, 1]).tocsr()


def test_no_data_takes_the_nearest_measured_value_along_the_mesh():
    values = np.array([0.0, 1.0, 2.0, np.nan, np.nan, 5.0, 6.0])
    out, filled, unreachable = tp.fill_along_mesh(values, _path_graph(7), np.ones(7, dtype=bool))
    assert out[3] == 2.0 and out[4] == 5.0
    assert (filled, unreachable) == (2, 0)
    assert np.isnan(values[3])                      # the field itself is left alone


def test_the_fill_never_crosses_excluded_tissue():
    values = np.array([0.0, 1.0, 2.0, np.nan, np.nan, 5.0, 6.0])
    tissue = np.ones(7, dtype=bool)
    tissue[[2, 5]] = False
    out, filled, unreachable = tp.fill_along_mesh(values, _path_graph(7), tissue)
    assert (filled, unreachable) == (0, 2)
    assert np.isnan(out[3]) and np.isnan(out[4])


def test_no_data_with_nothing_to_take_from_blocks():
    mesh = _strip([0.2, np.nan])                    # two triangles sharing no edge
    with pytest.raises(ValueError, match="no connected element with data"):
        tp.assign_tissue_property(mesh, _thresholds((0.0, 1.0), (1,)))


def test_filled_elements_are_classified_and_counted():
    sphere = pv.Sphere(theta_resolution=12, phi_resolution=12).triangulate()
    sphere.cell_data["elemTag"] = np.ones(sphere.n_cells, dtype=np.int32)
    values = np.linspace(0.0, 1.0, sphere.n_cells)
    values[[5, 40, 41]] = np.nan
    sphere.cell_data["v"] = values
    result = tp.assign_tissue_property(sphere, _thresholds((0.0, 0.5, 1.0), (1, 2)))
    assert result.filled == 3
    assert set(np.unique(result.tags)) <= {1, 2}
    assert "3 no-data elements filled" in result.summary()
