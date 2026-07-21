"""
test_label_islands.py
=====================
`RegionTagger.reduce_to_single_components` — each PV label reduced to one
connected patch, seed-free.

`tag()` already guarantees this via `_enforce_contiguity`, but a manual edit
paints picked cells with no connectivity check and a segmentation round trip's
nearest-cell `elemTag` copy can strand a single cell across a crevice. Either
leaves an island that `tag()` never re-checks, and CemrgApp's label check
rejects a label split across two regions (its auto-fix crashes on a one-cell
region). The contract:

* the largest component of each PV label is kept;
* every smaller island is reassigned to the majority label bordering it —
  body when it borders only body, the adjacent PV when it borders that;
* a label already forming one region is left untouched;
* the input array is not modified.

Uses synthetic meshes (no real EAM data required).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pyvista as pv
from scipy.sparse.csgraph import connected_components

from ccdaf.core.mesh_loader import BODY_LABEL
from ccdaf.core.region_tagger import RegionTagger, LABELS


def _grow_patch(adj, start, n):
    """A connected set of ~n cells grown from ``start`` over the adjacency."""
    seen, frontier = {start}, [start]
    while frontier and len(seen) < n:
        for nb in adj[frontier.pop()].indices:
            if nb not in seen:
                seen.add(nb)
                frontier.append(nb)
                if len(seen) >= n:
                    break
    return seen


def _isolated_cell(adj, avoid, n_cells):
    """A cell not in ``avoid`` and not adjacent to it — a true island site."""
    for c in range(n_cells):
        if c in avoid:
            continue
        if not (set(adj[c].indices) & avoid):
            return c
    raise AssertionError("no isolated cell available")


def _make():
    mesh = pv.Sphere(theta_resolution=30, phi_resolution=30).triangulate()
    tagger = RegionTagger(mesh)
    return mesh, tagger


def _n_regions(tagger, tags, lbl):
    mask = tags == lbl
    if not mask.any():
        return 0
    n, _ = connected_components(tagger._tri_adj[mask][:, mask], directed=False)
    return n


def test_island_is_removed_and_main_patch_kept():
    _, tagger = _make()
    adj = tagger._tri_adj
    n_cells = adj.shape[0]
    tags = np.full(n_cells, BODY_LABEL, dtype=np.int32)
    patch = _grow_patch(adj, 0, 40)
    tags[list(patch)] = 11
    island = _isolated_cell(adj, patch, n_cells)
    tags[island] = 11

    assert _n_regions(tagger, tags, 11) == 2          # main + island
    out = tagger.reduce_to_single_components(tags)
    assert _n_regions(tagger, out, 11) == 1           # island gone
    assert out[island] == BODY_LABEL                  # reassigned to its border
    assert (out[list(patch)] == 11).all()             # main patch untouched


def test_input_is_not_mutated():
    _, tagger = _make()
    adj = tagger._tri_adj
    tags = np.full(adj.shape[0], BODY_LABEL, dtype=np.int32)
    patch = _grow_patch(adj, 0, 30)
    tags[list(patch)] = 13
    island = _isolated_cell(adj, patch, adj.shape[0])
    tags[island] = 13
    before = tags.copy()
    tagger.reduce_to_single_components(tags)
    assert np.array_equal(tags, before)


def test_single_region_label_is_untouched():
    _, tagger = _make()
    adj = tagger._tri_adj
    tags = np.full(adj.shape[0], BODY_LABEL, dtype=np.int32)
    patch = _grow_patch(adj, 0, 50)
    tags[list(patch)] = 15
    out = tagger.reduce_to_single_components(tags)
    assert np.array_equal(out, tags)


def test_island_bordering_a_pv_goes_to_that_pv_not_body():
    """A stray cell whose neighbours are all another PV inherits that PV."""
    _, tagger = _make()
    adj = tagger._tri_adj
    tags = np.full(adj.shape[0], BODY_LABEL, dtype=np.int32)
    # main LSPV patch somewhere
    main = _grow_patch(adj, 0, 40)
    tags[list(main)] = 11
    # a cell far from it, with its whole neighbourhood painted RSPV, itself LSPV
    island = _isolated_cell(adj, main, adj.shape[0])
    tags[list(adj[island].indices)] = 15
    tags[island] = 11
    out = tagger.reduce_to_single_components(tags)
    assert out[island] == 15                          # border majority, not body
    assert _n_regions(tagger, out, 11) == 1


def test_every_pv_label_is_a_single_region_afterwards():
    _, tagger = _make()
    adj = tagger._tri_adj
    n_cells = adj.shape[0]
    tags = np.full(n_cells, BODY_LABEL, dtype=np.int32)
    used = set()
    for i, lbl in enumerate(LABELS.values()):
        patch = _grow_patch(adj, i * 3 + 1, 20)
        tags[list(patch)] = lbl
        used |= patch
    # scatter an island for one label
    island = _isolated_cell(adj, used, n_cells)
    tags[island] = 11
    out = tagger.reduce_to_single_components(tags)
    for lbl in LABELS.values():
        assert _n_regions(tagger, out, lbl) in (0, 1)
