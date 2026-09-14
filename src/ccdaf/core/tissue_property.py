"""
Tissue properties
=================

Mark every element of the working mesh with a material region, read off a
scalar field: ``tissueTag``, an integer cell array. It is what a simulation
assigns conductivities and cell models by, so it must cover every element
and hold no negative value.

Two criteria:

* **mean + k·SD** — region *i* starts at ``mean + kᵢ·SD`` of the *healthy*
  tissue and runs to the next row's start; the top region is open-ended and
  anything below the first row is healthy. With ``k = (2, 3)`` that is the
  border zone above 2 SD and scar above 3 SD, the definition of Chen et al.
  (Heart Rhythm 2015) as restated by Mendonca Costa et al. (Heart Rhythm
  2019).
* **thresholds** — contiguous rows ``[from, to)``, the top row closed at its
  upper edge, together covering at least the data.

Which tissue is healthy
-----------------------
The published method takes the mean and SD from a remote region an observer
draws by eye. On a mesh there is no observer, and marking healthy tissue is
the very thing being computed, so :func:`estimate_healthy` finds it:

1. **Seeds** — elements at or below a low weighted percentile (25 by
   default). Almost certainly healthy, but a biased sample: too dark and far
   too narrow.
2. **Grow** — from the seeds, through adjacent elements, keeping everything
   at or below ``mean + m·SD`` of the current region; refit; repeat. Every
   round restarts from the seeds, so the region can shrink as well as grow.
3. **Stop** when a round changes the region by less than a small share of
   the tissue (0.1% by default).

``m = 3`` rather than 2 because the limit also trims healthy tissue's own
upper tail. On exactly normal healthy tissue with no scar the refit settles
on an SD of 0.911 of the true one at ``m = 2`` — which labels 0.42% of a
scar-free wall as scar instead of 0.13% — against 0.993 at ``m = 3``.

The seed percentile is a safety margin, not a tuning knob: while the seeds
stay inside healthy tissue the answer does not depend on it, and once they
reach into scar the estimate collapses onto the whole wall. On a synthetic
ventricle with 30% of the wall healthy, seeds at 10, 25 and 35 all recovered
the truth and seeds at 50 labelled nothing as scar.

Weighting
---------
Every statistic is weighted by element volume (tetrahedra) or area
(triangles): the discrete integral over the tissue, which is what counting
equal voxels does on the image. Unweighted, a mesh refined around the scar
shifts the thresholds; on a ventricle refined to a size spread of CoV 1.14
the 3 SD cut moved from 0.865 to 0.927 counted per element, and stayed at
0.865 weighted.

No data
-------
A point field is averaged to each element over its *finite* vertex values,
so one missing vertex does not blank the element. An element whose vertices
are all missing takes the value of its nearest element with data, measured
through adjacent elements rather than in a straight line so that it can
never borrow from the opposite wall. The fill happens after the statistics,
which see measured elements only, and never touches the field itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyvista as pv
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, dijkstra

from ccdaf.core.mesh_loader import INTERNAL_ARRAYS
from ccdaf.core.volume_mesh import tetrahedra

#: The cell array this module writes.
TISSUE_TAG = "tissueTag"
#: The anatomical labelling the tissue selection reads.
ELEM_TAG = "elemTag"
#: Arrays whose values are labels, never a field to classify by. Named, not
#: detected from the dtype: a surface saved as VTK stores both as float32.
LABEL_FIELDS = frozenset({ELEM_TAG, TISSUE_TAG})

CRITERION_SD = "mean_sd"
CRITERION_THRESHOLDS = "thresholds"
POINT = "point"
CELL = "cell"

MIN_REGIONS = 1
MAX_REGIONS = 10
#: Border zone above 2 SD, scar above 3 SD.
DEFAULT_K: Tuple[float, ...] = (2.0, 3.0)

DEFAULT_SEED_PERCENTILE = 25.0
SEED_PERCENTILE_RANGE = (1.0, 50.0)
#: Above this the seeds start to risk reaching into a large infarct.
SEED_PERCENTILE_WARN = 35.0
DEFAULT_GROWTH_LIMIT = 3.0
GROWTH_LIMIT_RANGE = (2.0, 5.0)
DEFAULT_TOLERANCE_PERCENT = 0.1
MAX_ROUNDS = 100

#: One exact value holding more of the tissue than this is masked or
#: clipped data, not a measurement. Continuous fields stay under 1%: LGE on
#: the example ventricle 0%, Carto bipolar and LAT under 0.8%.
REPEATED_VALUE_WARN = 0.05
#: Healthy tissue below this share of the ticked tissue is worth a look —
#: correct for a very large infarct, but also what seeds reaching into scar
#: produce.
HEALTHY_FRACTION_WARN = 0.5


# ---------------------------------------------------------------------------
# Mesh helpers
# ---------------------------------------------------------------------------
def element_connectivity(dataset) -> np.ndarray:
    """The ``(n, 3)`` triangles of a surface or ``(n, 4)`` tetrahedra of a volume.

    Raises when the dataset holds anything else, because every rule here —
    a value per element, a size per element, a neighbour per shared facet —
    is written for one element shape.
    """
    if isinstance(dataset, pv.PolyData):
        faces = np.asarray(dataset.faces)
        if (faces.size == 0 or faces.size % 4 or np.any(faces[::4] != 3)
                or faces.size // 4 != dataset.n_cells):
            raise ValueError("the surface must contain triangles only")
        return faces.reshape(-1, 4)[:, 1:].astype(np.int64)
    tets = tetrahedra(dataset)
    if len(tets) == 0 or len(tets) != dataset.n_cells:
        raise ValueError("the volume must contain tetrahedra only")
    return tets


def element_sizes(dataset) -> np.ndarray:
    """Area of every triangle, or volume of every tetrahedron."""
    sizes = dataset.compute_cell_sizes(length=False, area=True, volume=True)
    key = "Area" if isinstance(dataset, pv.PolyData) else "Volume"
    return np.abs(np.asarray(sizes.cell_data[key], dtype=float))


def element_adjacency(dataset) -> sp.csr_matrix:
    """Elements sharing a face (tetrahedra) or an edge (triangles).

    Weighted by the distance between the two element centres, so a shortest
    path through it is a distance along the mesh.
    """
    conn = element_connectivity(dataset)
    n, nodes = conn.shape
    facets = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)) if nodes == 4 \
        else ((0, 1), (1, 2), (2, 0))
    keys = np.sort(np.vstack([conn[:, list(f)] for f in facets]), axis=1)
    owner = np.tile(np.arange(n), len(facets))
    npts = int(conn.max()) + 1
    if npts ** keys.shape[1] < 2 ** 62:
        key = keys[:, 0].copy()
        for col in keys[:, 1:].T:
            key = key * npts + col
    else:                                   # too many points to pack a key
        key = np.unique(keys, axis=0, return_inverse=True)[1].ravel()
    order = np.argsort(key, kind="stable")
    ordered = key[order]
    same = np.flatnonzero(ordered[1:] == ordered[:-1])
    i, j = owner[order[same]], owner[order[same + 1]]
    centres = np.asarray(dataset.points, dtype=float)[conn].mean(axis=1)
    # A shared facet between coincident centres is still an edge: keep its
    # weight positive so no graph routine reads it as absent.
    dist = np.maximum(np.linalg.norm(centres[i] - centres[j], axis=1), 1e-12)
    return sp.csr_matrix((np.r_[dist, dist], (np.r_[i, j], np.r_[j, i])),
                         shape=(n, n))


def eligible_fields(dataset) -> List[Tuple[str, str]]:
    """``(name, association)`` of every field a tissue property can be read from.

    One component, floating point, at least one finite value, and neither a
    label (``elemTag``, ``tissueTag``) nor bookkeeping.
    """
    out: List[Tuple[str, str]] = []
    if dataset is None:
        return out
    for association, attr in ((POINT, dataset.point_data),
                              (CELL, dataset.cell_data)):
        for name in attr.keys():
            if name in INTERNAL_ARRAYS or name in LABEL_FIELDS:
                continue
            values = np.asarray(attr[name])
            if values.ndim != 1 or not np.issubdtype(values.dtype, np.floating):
                continue
            if not np.isfinite(values).any():
                continue
            out.append((str(name), association))
    return out


def element_values(dataset, name: str, association: str) -> np.ndarray:
    """One value per element; NaN where the element has no data.

    A point field is averaged over each element's finite vertex values. For
    linear elements the plain average of the vertices *is* the interpolated
    value at the centre, since every barycentric weight is equal there; with
    a vertex missing it is the average of the rest.
    """
    if association == CELL:
        return np.asarray(dataset.cell_data[name], dtype=float).copy()
    conn = element_connectivity(dataset)
    corner = np.asarray(dataset.point_data[name], dtype=float)[conn]
    finite = np.isfinite(corner)
    count = finite.sum(axis=1)
    total = np.where(finite, corner, 0.0).sum(axis=1)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def element_labels(dataset) -> np.ndarray:
    """``elemTag`` as integers. Raises if it is missing or not whole numbers."""
    if ELEM_TAG not in dataset.cell_data:
        raise ValueError("the mesh has no elemTag array to select tissue by")
    raw = np.asarray(dataset.cell_data[ELEM_TAG])
    if np.issubdtype(raw.dtype, np.integer):
        return raw.astype(np.int64)
    rounded = np.rint(raw)
    if not np.all(np.isfinite(raw)) or not np.allclose(raw, rounded):
        raise ValueError("elemTag holds values that are not whole numbers")
    return rounded.astype(np.int64)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """The smallest value below which a share *q* of the total weight lies."""
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    index = int(np.searchsorted(cumulative, q * cumulative[-1], side="left"))
    return float(values[order][min(index, len(values) - 1)])


def weighted_mean_sd(values: np.ndarray, weights: np.ndarray) -> Tuple[float, float]:
    """Weighted mean and (population) standard deviation."""
    total = float(weights.sum())
    mean = float(np.sum(weights * values) / total)
    return mean, float(np.sqrt(np.sum(weights * (values - mean) ** 2) / total))


def repeated_value_share(values: np.ndarray, weights: np.ndarray) -> Tuple[float, float]:
    """``(share, value)`` of the single most common exact value, by weight."""
    unique, inverse = np.unique(values, return_inverse=True)
    mass = np.bincount(inverse.ravel(), weights=weights)
    k = int(np.argmax(mass))
    return float(mass[k] / weights.sum()), float(unique[k])


def _grow(adjacency: sp.csr_matrix, starts: np.ndarray,
          allowed: np.ndarray) -> np.ndarray:
    """Elements reachable from *starts* through *allowed* elements only."""
    index = np.flatnonzero(allowed)
    out = np.zeros(allowed.shape, dtype=bool)
    if index.size == 0:
        return out
    _, component = connected_components(adjacency[index][:, index],
                                        directed=False)
    out[index] = np.isin(component, np.unique(component[starts[index]]))
    return out


@dataclass(frozen=True)
class HealthyEstimate:
    """What :func:`estimate_healthy` found."""

    mean: float
    sd: float
    #: The elements the statistics came from.
    region: np.ndarray
    rounds: int
    #: False when the round limit was reached before the region settled.
    settled: bool
    #: The percentile value the seeds were cut at.
    seed_value: float


def estimate_healthy(values: np.ndarray, weights: np.ndarray,
                     adjacency: sp.csr_matrix, tissue: np.ndarray, *,
                     seed_percentile: float = DEFAULT_SEED_PERCENTILE,
                     growth_limit: float = DEFAULT_GROWTH_LIMIT,
                     tolerance_percent: float = DEFAULT_TOLERANCE_PERCENT,
                     max_rounds: int = MAX_ROUNDS) -> HealthyEstimate:
    """Find the healthy part of *tissue* and its weighted mean and SD.

    See the module docstring for the method. Only measured elements take
    part: a NaN is neither a seed nor a member.
    """
    measured = np.asarray(tissue, dtype=bool) & np.isfinite(values)
    index = np.flatnonzero(measured)
    if index.size == 0:
        raise ValueError("the field has no values in the ticked tissue")
    graph = adjacency[index][:, index]
    v, w = values[index], weights[index]
    total = float(w.sum())

    seed_value = weighted_quantile(v, w, seed_percentile / 100.0)
    seeds = v <= seed_value
    region = seeds.copy()
    settled = False
    rounds = 0
    for rounds in range(1, int(max_rounds) + 1):
        mean, sd = weighted_mean_sd(v[region], w[region])
        allowed = v <= mean + growth_limit * sd
        grown = _grow(graph, seeds & allowed, allowed)
        change = float(w[grown ^ region].sum()) / total
        region = grown
        if change < tolerance_percent / 100.0:
            settled = True
            break

    mean, sd = weighted_mean_sd(v[region], w[region])
    full = np.zeros(values.shape, dtype=bool)
    full[index[region]] = True
    return HealthyEstimate(mean, sd, full, rounds, settled, seed_value)


def fill_along_mesh(values: np.ndarray, adjacency: sp.csr_matrix,
                    tissue: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """Give each no-data element of *tissue* its nearest measured neighbour's value.

    Nearest is measured through adjacent elements of the tissue. Returns
    ``(values, filled, unreachable)``; *values* is a copy, and an element
    with no measured element reachable keeps its NaN.
    """
    out = np.array(values, dtype=float, copy=True)
    tissue = np.asarray(tissue, dtype=bool)
    missing = tissue & ~np.isfinite(out)
    if not missing.any():
        return out, 0, 0
    index = np.flatnonzero(tissue)
    local_missing = missing[index]
    sources = np.flatnonzero(~local_missing)
    targets = np.flatnonzero(local_missing)
    if sources.size == 0:
        return out, 0, int(targets.size)
    dist, _pred, nearest = dijkstra(adjacency[index][:, index], directed=False,
                                    indices=sources, min_only=True,
                                    return_predecessors=True)
    reached = targets[np.isfinite(dist[targets])]
    out[index[reached]] = values[index[nearest[reached]]]
    return out, int(reached.size), int(targets.size - reached.size)


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
def default_excluded_id(value: int, used: Sequence[int]) -> int:
    """An excluded tissue keeps its own ``elemTag`` number where it is free.

    Otherwise the smallest positive integer nobody uses: the right ventricle
    is ``elemTag`` 2, and on a two-row classification 2 is already scar.
    """
    taken = {int(u) for u in used}
    if int(value) >= 0 and int(value) not in taken:
        return int(value)
    candidate = 1
    while candidate in taken:
        candidate += 1
    return candidate


def threshold_edges(lo: float, hi: float, n_regions: int) -> Tuple[float, ...]:
    """Evenly spaced row boundaries from *lo* to *hi*."""
    return tuple(float(e) for e in np.linspace(lo, hi, int(n_regions) + 1))


@dataclass
class TissuePropertyOptions:
    """Everything the dialog asks for."""

    field: str
    association: str
    #: ``elemTag`` values to classify.
    tissue: Tuple[int, ...]
    #: The ID each unticked ``elemTag`` value is written as.
    excluded_ids: Dict[int, int] = dc_field(default_factory=dict)
    criterion: str = CRITERION_SD
    region_ids: Tuple[int, ...] = (1, 2)
    #: Mean + k·SD rows, one k per region, strictly increasing.
    ks: Tuple[float, ...] = DEFAULT_K
    #: Threshold rows as ``N + 1`` increasing boundaries.
    edges: Tuple[float, ...] = ()
    healthy_id: int = 0
    auto: bool = True
    mean: Optional[float] = None
    sd: Optional[float] = None
    seed_percentile: float = DEFAULT_SEED_PERCENTILE
    growth_limit: float = DEFAULT_GROWTH_LIMIT
    tolerance_percent: float = DEFAULT_TOLERANCE_PERCENT

    @property
    def n_regions(self) -> int:
        return len(self.region_ids)

    def classified_ids(self) -> Tuple[int, ...]:
        """IDs the classified tissue can receive."""
        ids = tuple(int(i) for i in self.region_ids)
        return ((int(self.healthy_id),) + ids) if self.criterion == CRITERION_SD else ids

    def validate(self) -> None:
        """Raise ``ValueError`` on anything that must stop Apply."""
        if self.association not in (POINT, CELL):
            raise ValueError(f"unknown field association '{self.association}'")
        if self.criterion not in (CRITERION_SD, CRITERION_THRESHOLDS):
            raise ValueError(f"unknown criterion '{self.criterion}'")
        if not self.tissue:
            raise ValueError("Tick at least one tissue to classify.")
        overlap = set(self.tissue) & set(self.excluded_ids)
        if overlap:
            raise ValueError(f"elemTag {sorted(overlap)} cannot be both classified and excluded.")
        if not MIN_REGIONS <= self.n_regions <= MAX_REGIONS:
            raise ValueError(f"Use between {MIN_REGIONS} and {MAX_REGIONS} regions.")

        ids = list(self.classified_ids()) + [int(i) for i in self.excluded_ids.values()]
        if any(i < 0 for i in ids):
            raise ValueError("IDs cannot be negative: tissueTag must hold no negative value.")
        if len(set(self.region_ids)) != len(self.region_ids):
            raise ValueError("Two region rows share an ID; each region needs its own.")

        if self.criterion == CRITERION_SD:
            if int(self.healthy_id) in {int(i) for i in self.region_ids}:
                raise ValueError("The healthy ID equals a region ID.")
            ks = np.asarray(self.ks, dtype=float)
            if ks.size != self.n_regions or not np.all(np.isfinite(ks)):
                raise ValueError("Give one k value per region.")
            if ks[0] <= 0.0:
                raise ValueError("The first k must be greater than 0.")
            if np.any(np.diff(ks) <= 0.0):
                raise ValueError("The k values must be strictly increasing.")
            if self.auto:
                lo, hi = SEED_PERCENTILE_RANGE
                if not lo <= self.seed_percentile <= hi:
                    raise ValueError(f"The seed percentile must lie between {lo:g} and {hi:g}.")
                lo, hi = GROWTH_LIMIT_RANGE
                if not lo <= self.growth_limit <= hi:
                    raise ValueError(f"The growth limit m must lie between {lo:g} and {hi:g}.")
                if not self.tolerance_percent > 0.0:
                    raise ValueError("The stop tolerance must be greater than 0.")
            else:
                if self.mean is None or self.sd is None \
                        or not np.isfinite(self.mean) or not np.isfinite(self.sd):
                    raise ValueError("Give a finite healthy mean and SD, or tick Auto.")
                if self.sd <= 0.0:
                    raise ValueError("The healthy SD must be greater than 0.")
        else:
            edges = np.asarray(self.edges, dtype=float)
            if edges.size != self.n_regions + 1 or not np.all(np.isfinite(edges)):
                raise ValueError("Every threshold row needs a finite 'from' and 'to'.")
            if np.any(np.diff(edges) <= 0.0):
                raise ValueError("Each row's 'to' must be greater than its 'from'.")

    def warnings(self) -> List[str]:
        """Things worth confirming before Apply; none of them stops it."""
        notes: List[str] = []
        if (self.criterion == CRITERION_SD and self.auto
                and self.seed_percentile > SEED_PERCENTILE_WARN):
            notes.append(
                f"Seeds above the {SEED_PERCENTILE_WARN:g}th percentile can reach "
                "into a large infarct, and the estimate then takes the whole "
                "tissue as healthy.")
        classified = set(self.classified_ids())
        shared = sorted({(t, i) for t, i in self.excluded_ids.items() if int(i) in classified})
        if shared:
            pairs = ", ".join(f"elemTag {t} → {i}" for t, i in shared)
            notes.append(
                f"Excluded tissue shares an ID with classified tissue ({pairs}); "
                "the simulation will not be able to tell them apart.")
        return notes


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
@dataclass
class TissueResult:
    """The ``tissueTag`` array and what went into it."""

    tags: np.ndarray
    mean: Optional[float]
    sd: Optional[float]
    healthy: Optional[HealthyEstimate]
    #: ``{id: (elements, share of the mesh by size)}``.
    counts: Dict[int, Tuple[int, float]]
    #: No-data elements given their nearest neighbour's value.
    filled: int
    warnings: List[str]
    #: Share of the ticked tissue, by size, the automatic estimate found healthy.
    healthy_fraction: Optional[float] = None

    def summary(self) -> str:
        parts = [f"{i}: {share:.1%}" for i, (_, share) in sorted(self.counts.items())]
        text = f"tissueTag written — {', '.join(parts)}"
        if self.mean is not None and self.sd is not None:
            text += f"; healthy mean {self.mean:.4g}, SD {self.sd:.4g}"
            if self.healthy is not None and self.healthy_fraction is not None:
                text += (f" (auto: {self.healthy_fraction:.0%} of the ticked tissue, "
                         f"{self.healthy.rounds} rounds)")
        if self.filled:
            text += f"; {self.filled} no-data elements filled from neighbours"
        return text + "."


def assign_tissue_property(dataset, options: TissuePropertyOptions, *,
                           adjacency: Optional[sp.csr_matrix] = None) -> TissueResult:
    """Classify *dataset* by *options*; raise ``ValueError`` on a blocking problem.

    Nothing is written to the dataset: the caller decides, after showing
    :attr:`TissueResult.warnings`. Pass *adjacency* to reuse one already
    built for this dataset.
    """
    options.validate()
    notes = options.warnings()
    labels = element_labels(dataset)
    present = {int(v) for v in np.unique(labels)}
    unassigned = sorted(present - set(options.tissue) - set(options.excluded_ids))
    if unassigned:
        raise ValueError(
            f"elemTag {unassigned} is neither classified nor excluded; "
            "every tissue needs an ID.")

    tissue = np.isin(labels, list(options.tissue))
    values = element_values(dataset, options.field, options.association)
    weights = element_sizes(dataset)
    measured = tissue & np.isfinite(values)
    if not measured.any():
        raise ValueError(f"'{options.field}' has no values in the ticked tissue.")

    share, value = repeated_value_share(values[measured], weights[measured])
    if share > REPEATED_VALUE_WARN:
        notes.append(
            f"{share:.0%} of the ticked tissue has exactly the value {value:.6g}: "
            "masked or clipped data, which biases the healthy estimate.")

    missing = tissue & ~np.isfinite(values)
    need_graph = missing.any() or (options.criterion == CRITERION_SD and options.auto)
    if need_graph and adjacency is None:
        adjacency = element_adjacency(dataset)

    mean = sd = None
    healthy = None
    healthy_fraction = None
    lo, hi = float(values[measured].min()), float(values[measured].max())
    if options.criterion == CRITERION_SD:
        if options.auto:
            healthy = estimate_healthy(
                values, weights, adjacency, tissue,
                seed_percentile=options.seed_percentile,
                growth_limit=options.growth_limit,
                tolerance_percent=options.tolerance_percent)
            mean, sd = healthy.mean, healthy.sd
            if not healthy.settled:
                notes.append(
                    f"The healthy estimate did not settle within {MAX_ROUNDS} rounds; "
                    "the last round is used.")
            fraction = float(weights[healthy.region].sum() / weights[measured].sum())
            healthy_fraction = fraction
            if fraction < HEALTHY_FRACTION_WARN:
                notes.append(
                    f"Only {fraction:.0%} of the ticked tissue was estimated healthy: "
                    "right for a very large infarct, but also what seeds reaching "
                    "into scar produce.")
        else:
            mean, sd = float(options.mean), float(options.sd)
            if not lo <= mean <= hi:
                notes.append(
                    f"The healthy mean {mean:.4g} lies outside the field's range "
                    f"[{lo:.4g}, {hi:.4g}] in the ticked tissue; is it on this "
                    "field's scale, per element?")
        if not sd > 1e-12 * max(1.0, abs(hi - lo)):
            raise ValueError(
                "The healthy tissue has no spread (SD = 0). Check the tissue "
                "selection: masked tissue, all one value, may be ticked.")

    filled_values, filled, unreachable = fill_along_mesh(values, adjacency, tissue) \
        if missing.any() else (values, 0, 0)
    if unreachable:
        raise ValueError(
            f"{unreachable} elements of the ticked tissue have no data and no "
            "connected element with data to take a value from.")

    v = filled_values[tissue]
    if options.criterion == CRITERION_SD:
        classes = np.full(v.shape, int(options.healthy_id), dtype=np.int64)
        for rid, k in zip(options.region_ids, options.ks):
            classes[v >= mean + float(k) * sd] = int(rid)
    else:
        edges = np.asarray(options.edges, dtype=float)
        lo_all, hi_all = float(v.min()), float(v.max())
        if edges[0] > lo_all or edges[-1] < hi_all:
            raise ValueError(
                f"The threshold rows cover [{edges[0]:.6g}, {edges[-1]:.6g}], but "
                f"the ticked tissue's values span [{lo_all:.6g}, {hi_all:.6g}]. "
                "Lower the first 'from' or raise the last 'to'.")
        row = np.clip(np.searchsorted(edges, v, side="right") - 1, 0, options.n_regions - 1)
        classes = np.asarray(options.region_ids, dtype=np.int64)[row]

    tags = np.full(labels.shape, -1, dtype=np.int64)
    for value_, rid in options.excluded_ids.items():
        tags[labels == int(value_)] = int(rid)
    tags[tissue] = classes
    if np.any(tags < 0):                    # guarded above; never silently
        raise ValueError("some elements received no ID")

    total = float(weights.sum())
    counts = {int(i): (int((tags == i).sum()), float(weights[tags == i].sum() / total))
              for i in np.unique(tags)}
    return TissueResult(tags=tags.astype(np.int32), mean=mean, sd=sd,
                        healthy=healthy, counts=counts, filled=filled,
                        warnings=notes, healthy_fraction=healthy_fraction)


def availability(dataset) -> Tuple[bool, str]:
    """Whether a tissue property can be assigned on *dataset*, and why not."""
    if dataset is None:
        return False, "Load a mesh first."
    if ELEM_TAG not in dataset.cell_data:
        return False, "The mesh has no elemTag array."
    if not eligible_fields(dataset):
        return False, ("The mesh has no scalar floating-point field to read a "
                       "tissue property from.")
    return True, "Mark elements with material regions read off a scalar field."


__all__ = [
    "TISSUE_TAG", "ELEM_TAG", "LABEL_FIELDS",
    "CRITERION_SD", "CRITERION_THRESHOLDS", "POINT", "CELL",
    "MIN_REGIONS", "MAX_REGIONS", "DEFAULT_K",
    "DEFAULT_SEED_PERCENTILE", "SEED_PERCENTILE_RANGE", "SEED_PERCENTILE_WARN",
    "DEFAULT_GROWTH_LIMIT", "GROWTH_LIMIT_RANGE", "DEFAULT_TOLERANCE_PERCENT",
    "MAX_ROUNDS", "REPEATED_VALUE_WARN", "HEALTHY_FRACTION_WARN",
    "element_connectivity", "element_sizes", "element_adjacency",
    "eligible_fields", "element_values", "element_labels",
    "weighted_quantile", "weighted_mean_sd", "repeated_value_share",
    "HealthyEstimate", "estimate_healthy", "fill_along_mesh",
    "default_excluded_id", "threshold_edges",
    "TissuePropertyOptions", "TissueResult", "assign_tissue_property",
    "availability",
]
