"""
RegionTagger
============

Multi-source geodesic segmentation of a left-atrial surface mesh.

Given a triangular `pyvista.PolyData` and user-provided seed points
(4 pulmonary veins + 1 appendage), this module assigns per-triangle
labels to the cell-data array ``elemTag``:

    LSPV = 11,  LIPV = 13,  RSPV = 15,  RIPV = 17,  LAA = 19

Background body triangles remain at the initialization value (1).

Method
------
1. Build the mesh 1-skeleton as a weighted undirected graph:
   nodes = vertices, edges = triangle edges, weights = Euclidean length.
2. Run truncated Dijkstra from each seed (SciPy CSR backend).
3. Multi-source Voronoi on the surface: each vertex is assigned to the
   closest seed within a per-seed radius cap.
4. Lift vertex labels to triangles via a majority-vote rule (≥2 of the
   3 vertices must share the same valid label).
5. Enforce per-label contiguity by keeping only the triangle-adjacency
   connected component that contains (or lies closest to) the seed.

Ostium detection (per-seed radius cap refinement) is area-based: each
distance-bin's wavefront vertices are projected onto their local PCA
plane, the 2D convex-hull area is measured, and the cut is placed
where the area grows above ``area_growth_threshold`` over ≥2
consecutive bins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pyvista as pv
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from ccdaf.core.mesh_loader import BODY_LABEL, UNASSIGNED

# ---------------------------------------------------------------------------
# Label conventions (from Claude.md)
# ---------------------------------------------------------------------------
LABELS: Dict[str, int] = {
    "LSPV": 11,
    "LIPV": 13,
    "RSPV": 15,
    "RIPV": 17,
    "LAA":  19,
}



@dataclass
class TaggerConfig:
    """Tunable parameters for the tagging pipeline.

    Per-seed radius factor (radius_cap = factor * median_edge_length).
    Each pulmonary vein carries its own factor so the GUI can tune the
    caps independently (vein calibres vary anatomically). LAA has its
    own factor because the appendage is lobed rather than tubular.

    All factors must be strictly positive; validation runs in
    ``__post_init__`` so an invalid config is rejected before any
    tagging work begins.
    """

    # Per-PV radius factors. Tuned independently by the GUI.
    lspv_radius_factor: float = 25.0
    lipv_radius_factor: float = 25.0
    rspv_radius_factor: float = 25.0
    ripv_radius_factor: float = 25.0
    # LAA radius factor (lobed pouch — typically expands more than PV tubes).
    laa_radius_factor:  float = 25.0

    # Optional hard overrides (world units). If set (non-None), they
    # take precedence over the corresponding factor for that seed.
    # A single PV abs override applies to ALL PVs; set to None to fall
    # back to the per-PV factors above.
    pv_radius_abs:  Optional[float] = None
    laa_radius_abs: Optional[float] = None

    # Area-based ostium detection. The ostium is flagged where the
    # wavefront cross-section grows by at least this ratio between
    # consecutive distance bins, sustained for ≥2 bins.
    area_growth_threshold: float = 1.8

    # Number of frontier bins used to estimate cross-section vs. distance.
    bottleneck_bins: int = 50

    def __post_init__(self) -> None:
        self._validate()

    # -- validation ------------------------------------------------------
    def _validate(self) -> None:
        factor_fields = (
            ("lspv_radius_factor", self.lspv_radius_factor),
            ("lipv_radius_factor", self.lipv_radius_factor),
            ("rspv_radius_factor", self.rspv_radius_factor),
            ("ripv_radius_factor", self.ripv_radius_factor),
            ("laa_radius_factor",  self.laa_radius_factor),
        )
        bad = [name for name, v in factor_fields
               if not (isinstance(v, (int, float)) and np.isfinite(v) and v > 0.0)]
        if bad:
            raise ValueError(
                f"TaggerConfig: radius factor(s) must be > 0 and finite: {bad}"
            )
        # Abs overrides, if provided, must also be strictly positive.
        for name, v in (("pv_radius_abs", self.pv_radius_abs),
                        ("laa_radius_abs", self.laa_radius_abs)):
            if v is not None and not (np.isfinite(v) and v > 0.0):
                raise ValueError(
                    f"TaggerConfig: {name} must be > 0 and finite (got {v!r})"
                )
        if not (np.isfinite(self.area_growth_threshold)
                and self.area_growth_threshold > 1.0):
            raise ValueError(
                "TaggerConfig: area_growth_threshold must be finite and > 1 "
                f"(got {self.area_growth_threshold!r})"
            )


# Map seed name -> config attribute that holds its radius factor.
_RADIUS_FACTOR_ATTR: Dict[str, str] = {
    "LSPV": "lspv_radius_factor",
    "LIPV": "lipv_radius_factor",
    "RSPV": "rspv_radius_factor",
    "RIPV": "ripv_radius_factor",
    "LAA":  "laa_radius_factor",
}


# ---------------------------------------------------------------------------
# RegionTagger
# ---------------------------------------------------------------------------
class RegionTagger:
    """Geodesic multi-source tagger for the LA surface mesh."""

    def __init__(
        self,
        mesh: pv.PolyData,
        config: Optional[TaggerConfig] = None,
    ) -> None:
        if not isinstance(mesh, pv.PolyData):
            raise TypeError("mesh must be a pyvista.PolyData")

        self.mesh: pv.PolyData = mesh
        self.config: TaggerConfig = config or TaggerConfig()

        self._points: np.ndarray = np.asarray(mesh.points, dtype=float)
        self._triangles: np.ndarray = self._extract_triangles(mesh)

        # Vertex graph (1-skeleton) -- built once and reused.
        self._graph, self._median_edge = self._build_vertex_graph(
            self._points, self._triangles
        )

        # Triangle adjacency graph -- built once for connectivity checks.
        self._tri_adj: csr_matrix = self._build_triangle_adjacency(self._triangles)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def tag(
        self,
        seeds: Mapping[str, int | Sequence[float]],
    ) -> np.ndarray:
        """Segment the mesh and write labels into ``mesh.cell_data['elemTag']``.

        Parameters
        ----------
        seeds
            Mapping ``{name -> seed}`` with names in ``LABELS``.
            A seed may be either a vertex index (int) or an (x,y,z) coordinate.

        Returns
        -------
        elemTag : (n_cells,) int array that has also been written to
            ``self.mesh.cell_data['elemTag']``.
        """
        # Guard against any post-construction mutation of the config: if a
        # GUI slider pushed a factor <= 0 after init, reject before work.
        self.config._validate()

        seed_vertices = self._resolve_seeds(seeds)

        # Per-seed geodesic distances with truncation (leakage guard #1).
        dist_stack, radius_per_seed, name_order = self._multi_source_dijkstra(
            seed_vertices
        )

        # Vertex-level Voronoi (within caps).
        vertex_label = self._vertex_voronoi(dist_stack, radius_per_seed, name_order)

        # Lift to triangles with majority-vote rule (leakage guard #2).
        tri_label = self._triangles_from_vertices(vertex_label)

        # Keep only the seed-containing connected component (leakage guard #3).
        tri_label = self._enforce_contiguity(tri_label, seed_vertices, name_order)
        

        # PV labels of interest
        pv_labels = {LABELS["LSPV"], LABELS["LIPV"], LABELS["RSPV"], LABELS["RIPV"], LABELS["LAA"]}
        
        # 2. Identify "Multi-tag" vertices
        # We find vertices where the incident triangles have >1 unique PV label
        multi_tag_vertices = self._find_shared_pv_vertices(tri_label, pv_labels)
        
        # 3. Identify and unassign triangles sharing these vertices
        if multi_tag_vertices.size > 0:
            # A triangle is marked UNASSIGNED if ANY of its vertices 
            # are shared between two different PV regions.
            mask_to_unassign = np.any(np.isin(self._triangles, multi_tag_vertices), axis=1)
            tri_label[mask_to_unassign] = UNASSIGNED

        # --- End of Shared Boundary Masking ---

        tri_label = self._fill_holes(tri_label)
        
        # Write back into the mesh.
        elem_tag = np.full(self.mesh.n_cells, BODY_LABEL, dtype=np.int32)
        assigned = tri_label != UNASSIGNED
        elem_tag[assigned] = tri_label[assigned]
        self.mesh.cell_data["elemTag"] = elem_tag
        return elem_tag

    # ------------------------------------------------------------------
    # Mesh preparation
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_triangles(mesh: pv.PolyData) -> np.ndarray:
        """Return an (M, 3) int array of triangle vertex indices."""
        faces = np.asarray(mesh.faces)
        if faces.size == 0:
            raise ValueError("mesh has no faces")
        # PyVista packs faces as [n, v0, v1, ..., vn-1, n, ...]. Require triangles.
        if faces.size % 4 != 0 or np.any(faces[::4] != 3):
            raise ValueError("mesh must contain triangles only")
        return faces.reshape(-1, 4)[:, 1:].astype(np.int64)

    @staticmethod
    def _build_vertex_graph(
        points: np.ndarray,
        triangles: np.ndarray,
    ) -> Tuple[csr_matrix, float]:
        """Weighted symmetric CSR graph on mesh vertices.

        For each unique undirected edge ``{u, v}`` the stored weight is
        the *minimum* Euclidean length observed across incident triangles.
        This makes no assumption about how many times the edge appears
        (boundary=1, manifold=2, non-manifold=3+) and is therefore safe
        for non-manifold meshes.
        """
        # One directed pair per triangle edge (directed (e0, e1)).
        e0 = triangles[:, [0, 1, 2]].ravel()
        e1 = triangles[:, [1, 2, 0]].ravel()
        lengths = np.linalg.norm(points[e0] - points[e1], axis=1)

        # Canonicalize to (min, max) so the same undirected edge collapses.
        u = np.minimum(e0, e1)
        v = np.maximum(e0, e1)

        # Aggregate minimum length per unique undirected edge via sort +
        # ``np.minimum.reduceat`` on run-starts — no duplicate-count assumption.
        n = int(points.shape[0])
        key = u.astype(np.int64) * (n + 1) + v.astype(np.int64)
        order = np.argsort(key, kind="stable")
        key_sorted = key[order]
        len_sorted = lengths[order]
        starts = np.empty(key_sorted.shape, dtype=bool)
        starts[0] = True
        starts[1:] = key_sorted[1:] != key_sorted[:-1]
        start_idx = np.flatnonzero(starts)
        min_lengths = np.minimum.reduceat(len_sorted, start_idx)
        u_unique = u[order][start_idx]
        v_unique = v[order][start_idx]

        # Symmetric entries: (u,v) and (v,u) with the same minimum weight.
        rows = np.concatenate([u_unique, v_unique])
        cols = np.concatenate([v_unique, u_unique])
        data = np.concatenate([min_lengths, min_lengths])
        graph = csr_matrix((data, (rows, cols)), shape=(n, n))

        median_edge = float(np.median(min_lengths)) if min_lengths.size else 0.0
        return graph, median_edge

    @staticmethod
    def _build_triangle_adjacency(triangles: np.ndarray) -> csr_matrix:
        """Triangle-to-triangle graph: edge if they share a mesh edge."""
        m = triangles.shape[0]
        edge_to_tri: Dict[Tuple[int, int], list] = {}
        for ti, (a, b, c) in enumerate(triangles):
            for u, v in ((a, b), (b, c), (c, a)):
                key = (u, v) if u < v else (v, u)
                edge_to_tri.setdefault(key, []).append(ti)

        rows, cols = [], []
        for tris in edge_to_tri.values():
            if len(tris) >= 2:
                # Manifold edges connect exactly two triangles; non-manifold
                # edges (rare) are fully connected among their incident triangles.
                for i in range(len(tris)):
                    for j in range(i + 1, len(tris)):
                        rows.append(tris[i])
                        cols.append(tris[j])
                        rows.append(tris[j])
                        cols.append(tris[i])

        data = np.ones(len(rows), dtype=np.uint8)
        return csr_matrix((data, (rows, cols)), shape=(m, m))

    # ------------------------------------------------------------------
    # Seed resolution
    # ------------------------------------------------------------------
    def _resolve_seeds(
        self,
        seeds: Mapping[str, int | Sequence[float]],
    ) -> Dict[str, int]:
        """Normalize seeds to vertex indices; ignore non-segmentation seeds."""
        out: Dict[str, int] = {}
        for name, seed in seeds.items():
            if name not in LABELS:
                # e.g. the mitral seed is stored alongside but not tagged here.
                continue
            if np.isscalar(seed):
                idx = int(seed)
                if not (0 <= idx < self._points.shape[0]):
                    raise ValueError(f"seed '{name}' index out of range")
            else:
                p = np.asarray(seed, dtype=float).reshape(3)
                idx = int(np.argmin(np.sum((self._points - p) ** 2, axis=1)))
            out[name] = idx

        missing = set(LABELS) - set(out)
        if missing:
            raise ValueError(f"missing required seeds: {sorted(missing)}")
        return out

    # ------------------------------------------------------------------
    # Geodesic distances
    # ------------------------------------------------------------------
    def _multi_source_dijkstra(
        self,
        seed_vertices: Mapping[str, int],
    ) -> Tuple[np.ndarray, np.ndarray, list[str]]:
        """Truncated single-source Dijkstra from each seed.

        Returns
        -------
        dist   : (K, N) distance matrix (inf beyond the cap).
        radii  : (K,) radius cap per seed.
        names  : ordered seed names matching rows of ``dist``.
        """
        names = list(seed_vertices.keys())
        k = len(names)
        radii = np.empty(k, dtype=float)
        dist = np.empty((k, self._points.shape[0]), dtype=float)

        for i, name in enumerate(names):
            r = self._radius_for(name)
            # Run an uncapped Dijkstra first up to a generous working radius so
            # we can refine ``r`` via the bottleneck detector, then re-cap.
            working_limit = r * 2.0
            d = dijkstra(
                self._graph,
                indices=seed_vertices[name],
                directed=False,
                limit=working_limit,
            )
            r_eff = self._bottleneck_radius(d, r)
            d = np.where(d <= r_eff, d, np.inf)

            dist[i] = d
            radii[i] = r_eff

        return dist, radii, names

    def _radius_for(self, name: str) -> float:
        """Per-seed radius cap.

        Lookup order:
          1. Absolute override (``pv_radius_abs`` for PVs, ``laa_radius_abs``
             for LAA) if set.
          2. Per-seed factor from TaggerConfig * median edge length.
        """
        if name == "LAA":
            if self.config.laa_radius_abs is not None:
                return float(self.config.laa_radius_abs)
            return self.config.laa_radius_factor * self._median_edge

        # Pulmonary veins: a single abs override covers all four if set;
        # otherwise read the vein-specific factor.
        if self.config.pv_radius_abs is not None:
            return float(self.config.pv_radius_abs)

        attr = _RADIUS_FACTOR_ATTR.get(name)
        if attr is None:
            raise ValueError(f"unknown seed name for radius lookup: {name!r}")
        return float(getattr(self.config, attr)) * self._median_edge


    def _bottleneck_radius(self, d: np.ndarray, r_max: float) -> float:
        """Area-based ostium detection on the wavefront cross-section.

        For each distance bin:
          1. Collect the 3D positions of vertices inside that bin.
          2. Center and project onto their local PCA plane (first two
             SVD components) — this gives a planar cross-section that
             is faithful even for tubes that curve in 3D.
          3. Compute the 2D convex-hull area of the projected points.

        Walking outward bin-by-bin, a sustained expansion is declared
        when the ratio ``area[b] / area[b-1]`` exceeds
        ``area_growth_threshold`` for ≥2 consecutive bins. The cut is
        placed at the edge of the first bin in that sustained run.

        Improved area-based ostium detection.
        Refinements:
        1. Baseline Tracking: Compares current area to the minimum area seen so far,
           allowing it to catch single large expansions that then plateau.
        2. Signal Smoothing: Uses a windowed average to filter out noise from 
           low-density mesh regions or "thick-slice" projection artifacts.
        3. Sustained Check: Ensures the expansion isn't a transient spike (like 
           passing an internal ridge) by checking a look-ahead window.

        Refinements:
        1. Warm-up Zone: Ignores the first 15% of the search radius to allow 
           the wavefront to move away from the seed's local curvature/flatness.
        2. Stable Baseline: Establishes a 'vein diameter' by looking for a 
           region where area growth is minimal before looking for a flare.
        3. Gradient vs. Ratio: Uses a smoothed area gradient to distinguish
           between the natural quadratic growth of a flat-wall expansion 
           and the sudden 'pop' of a vessel hitting the atrial body.

        Fallbacks
        ---------
        Returns ``r_max`` on any failure (too few samples, degenerate
        hulls, Qhull errors, SVD failures, or no sustained expansion).
        """
        try:
            # Local import keeps SciPy's spatial deps lazy.
            from scipy.spatial import QhullError

            finite = np.isfinite(d) & (d > 0)
            if np.count_nonzero(finite) < 16:
                return r_max

            bins = max(20   , int(self.config.bottleneck_bins))
            edges = np.linspace(0.0, r_max, bins + 1)
            d_finite = d[finite]
            pts_finite = self._points[finite]

            # Assign each reachable vertex to a bin index in [0, bins).
            bin_idx = np.digitize(d_finite, edges) - 1
            bin_idx = np.clip(bin_idx, 0, bins - 1)

            # A 2D convex hull needs ≥3 non-collinear points; use 4 as a
            # safety margin against near-collinear tube samplings.
            MIN_VERTS = 8
            raw_areas = np.full(bins, np.nan, dtype=float)

            for b in range(bins):
                sel = bin_idx == b
                if np.count_nonzero(sel) < MIN_VERTS:
                    continue
                try:
                    pts = pts_finite[sel]
                    centered = pts - pts.mean(axis=0)
                    _, _, vt = np.linalg.svd(centered, full_matrices=False)
                except np.linalg.LinAlgError:
                    continue
                if vt.shape[0] < 2:
                    continue
                basis = vt[:2]               # (2, 3) local plane basis
                proj = centered @ basis.T    # (n, 2) in-plane coords


                try:
                    #hull = ConvexHull(proj)
                    # For a 2D ConvexHull, .volume is the area.
                    #a = float(hull.volume)
                    cov = np.cov(proj.T)
                    area = np.sqrt(np.linalg.det(cov))
                    
                    
                except (QhullError, ValueError, Exception):
                    continue
                if np.isfinite(area) and area > 0.0:
                    raw_areas[b] = area

            mask = np.isfinite(raw_areas)
            if np.count_nonzero(mask) < 6:
                return r_max

            xp    = np.where(mask)[0]

            # Linear interpolation for missing bins
            #areas = np.interp(np.arange(bins), xp, fp)
            # Smoothing (simple 3-bin moving average) to suppress mesh noise
            #kernel = np.array([0.2, 0.6, 0.2])
            #areas = np.convolve(areas, kernel, mode='same')            
            #look_ahead      = 2 # Number of bins the expansion must persist
            threshold = float(self.config.area_growth_threshold)
            r_floor = max(4.0 * self._median_edge, 1e-9)
            
            # PROTECT: Ignore the first 15% of the radius (the 'warm-up')
            #warm_up_idx = int(bins * 0.15)
            # Find the minimum area in the 'stable' portion of the vein (warm-up zone)
            # This represents our best guess for the vessel's true diameter.
            #baseline_area = np.min(areas[:warm_up_idx + 1])



            #for b in range(warm_up_idx, bins - look_ahead):
            #    dist_at_bin  = edges[b]
            #    current_area = areas[b]
            #    prev_area    = areas[b-1] if areas[b-1]>0 else current_area
            #    if dist_at_bin < r_floor:
            #        continue                
            #    # Check if current bin is a significant expansion over the baseline
            #    if (current_area > baseline_area * threshold) or (current_area>prev_area * threshold):
            #        # Sustained expansion check
            #        future_samples = areas[b+1 : b+1+look_ahead]
            #        if np.all(future_samples > current_area * 0.9): # Must stay 'large'
            #            return float(dist_at_bin)
            #    baseline_area = min(baseline_area, current_area)

            # Walk the valid-bin sequence: count consecutive bins whose
            # area/prev-area ratio exceeds the threshold. The cut fires
            # only when this count reaches 2 (sustained expansion), and
            # the radius is the outer edge of the FIRST growing bin.
            sustained = 0
            growth_start_edge: Optional[float] = None
            areas = raw_areas
            for i in range(1, len(xp)):
                b_prev = xp[i - 1]
                b_curr = xp[i]
                a_prev = areas[b_prev]
                a_curr = areas[b_curr]
                if a_prev <= 0.0 or not np.isfinite(a_curr):
                    sustained = 0
                    growth_start_edge = None
                    continue

                ratio = a_curr / a_prev
                if ratio > threshold:
                    if sustained == 0:
                        # First growing bin: record the cut at its outer edge.
                        growth_start_edge = float(edges[b_curr])
                    sustained += 1
                    if sustained >= 2 and growth_start_edge is not None:
                        r_detected = growth_start_edge
                        if (not np.isfinite(r_detected)
                                or r_detected < r_floor
                                or r_detected >= r_max):
                            return r_max
                        return r_detected
                else:
                    sustained = 0
                    growth_start_edge = None


            return r_max
        except Exception:
            # Any unexpected failure -> plain cap.
            return r_max

    
    
    # ------------------------------------------------------------------
    # Label assignment
    # ------------------------------------------------------------------
    @staticmethod
    def _vertex_voronoi(
        dist: np.ndarray,
        radii: np.ndarray,
        names: Sequence[str],
    ) -> np.ndarray:
        """Assign each vertex to the closest seed within its own cap."""
        n = dist.shape[1]
        # Mask infeasible cells (beyond per-seed cap) with +inf.
        masked = np.where(dist <= radii[:, None], dist, np.inf)
        best = np.argmin(masked, axis=0)
        best_dist = masked[best, np.arange(n)]

        label = np.full(n, UNASSIGNED, dtype=np.int32)
        reachable = np.isfinite(best_dist)
        label_lookup = np.array([LABELS[nm] for nm in names], dtype=np.int32)
        label[reachable] = label_lookup[best[reachable]]
        return label

    def _triangles_from_vertices(self, vertex_label: np.ndarray) -> np.ndarray:
        """Majority-vote rule: a triangle is labeled iff at least 2 of its
        3 vertices share the same *valid* (non-UNASSIGNED) label.

        With three vertices a 2-2 tie is impossible, so the winning label
        is unambiguous. This reduces boundary holes compared to unanimity
        while leakage is still contained by (a) the per-seed radius cap
        at the Dijkstra step and (b) the downstream contiguity filter.
        """
        tri = self._triangles
        la = vertex_label[tri[:, 0]]
        lb = vertex_label[tri[:, 1]]
        lc = vertex_label[tri[:, 2]]

        # Pairwise agreements restricted to valid labels.
        ab = (la == lb) & (la != UNASSIGNED)
        ac = (la == lc) & (la != UNASSIGNED)
        bc = (lb == lc) & (lb != UNASSIGNED)

        out = np.full(tri.shape[0], UNASSIGNED, dtype=np.int32)
        # Any pairwise match is sufficient; with 3 vertices no tie can
        # produce conflicting winners.
        out = np.where(ab, la, out)
        out = np.where(ac & (out == UNASSIGNED), la, out)
        out = np.where(bc & (out == UNASSIGNED), lb, out)
        return out

    def _enforce_contiguity(
        self,
        tri_label: np.ndarray,
        seed_vertices: Mapping[str, int],
        names: Sequence[str],
    ) -> np.ndarray:
        """Keep only the seed-anchored component for each label.

        The anchor triangle for a given label is picked by:
          (1) preferring any labeled triangle *incident* to the seed vertex;
          (2) otherwise the labeled triangle whose *centroid* is closest
              to the seed point in Euclidean space.
        This fixes the prior "largest component" fallback, which could
        lock onto a spurious blob when the seed sat on a triangle that
        failed the majority test.
        """
        out = tri_label.copy()
        tri = self._triangles
        centroids = None  # lazy: computed once if the nearest-tri fallback fires

        for name in names:
            lbl = LABELS[name]
            mask = out == lbl
            if not np.any(mask):
                continue

            # Subgraph of triangles with this label.
            sub = self._tri_adj[mask][:, mask]
            n_comp, comp = connected_components(sub, directed=False)

            seed_v = seed_vertices[name]
            incident = np.any(tri == seed_v, axis=1) & mask

            local_idx = np.cumsum(mask) - 1  # global -> local index inside mask
            if np.any(incident):
                keep_comp = int(comp[local_idx[np.argmax(incident)]])
            else:
                # Nearest labeled-triangle centroid to the seed point.
                if centroids is None:
                    centroids = self._points[tri].mean(axis=1)
                seed_xyz = self._points[seed_v]
                labeled_global = np.where(mask)[0]
                d2 = np.sum(
                    (centroids[labeled_global] - seed_xyz) ** 2, axis=1
                )
                nearest_global = int(labeled_global[int(np.argmin(d2))])
                keep_comp = int(comp[local_idx[nearest_global]])

            # Unlabel triangles in other components.
            global_idx = np.where(mask)[0]
            drop = global_idx[comp != keep_comp]
            out[drop] = UNASSIGNED

        return out

    def reduce_to_single_components(self, tri_label: np.ndarray) -> np.ndarray:
        """Each PV label as one connected patch, seed-free.

        For every label in :data:`LABELS`, keep its largest connected
        component and reassign each smaller island to the majority label
        bordering it — BODY when it borders nothing else, which is the usual
        case (a stray triangle sitting in the body). Returns a new array;
        ``tri_label`` is untouched.

        :meth:`_enforce_contiguity` already does this during ``tag``, but it
        needs the seeds and runs only there. A manual edit paints picked
        cells with no connectivity check, and a segmentation round trip's
        nearest-cell ``elemTag`` copy can strand a single cell across a
        crevice — either leaves an island that ``tag`` never sees. CemrgApp's
        label check requires exactly one region per PV label and its auto-fix
        crashes on a one-cell region, so this runs on the way out of manual
        correction to keep an export from carrying one.
        """
        out = np.asarray(tri_label).copy()
        for lbl in LABELS.values():
            mask = out == lbl
            if not np.any(mask):
                continue
            sub = self._tri_adj[mask][:, mask]
            n_comp, comp = connected_components(sub, directed=False)
            if n_comp <= 1:
                continue
            global_idx = np.where(mask)[0]
            keep = int(np.argmax(np.bincount(comp)))
            for c in range(n_comp):
                if c == keep:
                    continue
                island = global_idx[comp == c]
                out[island] = self._border_majority_label(island, out)
        return out

    def dilate_label(self, tri_label: np.ndarray, label: int) -> np.ndarray:
        """Grow ``label`` one pass into the background fringe, seam-safe.

        A background cell (BODY or UNASSIGNED) becomes ``label`` when at least
        two of its edge-neighbours already carry ``label`` **and** none carries
        a *different* PV label. The majority rule fills the ``\\/`` gaps that
        make a per-triangle boundary zigzag; the other-PV guard keeps ``label``
        from bridging a thin body seam into a neighbouring region and merging
        the two. Returns a new array; ``tri_label`` is untouched.
        """
        out = np.asarray(tri_label).copy()
        if label not in LABELS.values():
            return out
        is_label = (out == label).astype(np.int64)
        is_bg = (out == BODY_LABEL) | (out == UNASSIGNED)
        is_other_pv = np.isin(out, list(LABELS.values())) & (out != label)
        n_label = self._tri_adj.dot(is_label)
        n_other = self._tri_adj.dot(is_other_pv.astype(np.int64))
        out[is_bg & (n_label >= 2) & (n_other == 0)] = label
        return out

    def erode_label(self, tri_label: np.ndarray, label: int) -> np.ndarray:
        """Shrink ``label`` one pass off the background, the inverse of
        :meth:`dilate_label`.

        A ``label`` cell with at least two background edge-neighbours reverts
        to BODY, which shaves the lone spikes a per-triangle boundary leaves.
        Returns a new array; ``tri_label`` is untouched.
        """
        out = np.asarray(tri_label).copy()
        if label not in LABELS.values():
            return out
        is_bg = ((out == BODY_LABEL) | (out == UNASSIGNED)).astype(np.int64)
        n_bg = self._tri_adj.dot(is_bg)
        out[(out == label) & (n_bg >= 2)] = BODY_LABEL
        return out

    def _border_majority_label(self, cells: np.ndarray,
                               tri_label: np.ndarray) -> int:
        """Majority label among the cells bordering ``cells`` but not in it.

        A component's external edge-neighbours never share its label (they
        would be in the component), so this is always a different label —
        BODY when the island borders only body.
        """
        external = np.setdiff1d(np.unique(self._tri_adj[cells, :].indices), cells)
        if external.size == 0:
            return BODY_LABEL
        vals, counts = np.unique(tri_label[external], return_counts=True)
        return int(vals[int(np.argmax(counts))])

    def _fill_holes(self, tri_label: np.ndarray) -> np.ndarray:
        """
        Identifies components of BODY_LABEL/UNASSIGNED triangles that are 
        entirely enclosed by a single anatomical region and fills them.
        """
        out = tri_label.copy()
        # Define what we consider "background" (holes)
        is_background = (out == UNASSIGNED) | (out == BODY_LABEL)
        
        if not np.any(is_background):
            return out

        # 1. Group background triangles into connected components
        sub = self._tri_adj[is_background][:, is_background]
        n_comp, comp = connected_components(sub, directed=False)
        
        # Map from local (background-only) index to global triangle index
        bg_global_indices = np.where(is_background)[0]
        
        for c_idx in range(n_comp):
            # Triangles belonging to this specific background cluster
            cluster_mask_local = (comp == c_idx)
            cluster_global_indices = bg_global_indices[cluster_mask_local]
            
            # 2. Find all neighbors of this cluster
            # Get rows from adjacency matrix for these triangles
            adj_sub = self._tri_adj[cluster_global_indices, :]
            # Find indices of all triangles adjacent to the cluster
            neighbor_indices = adj_sub.indices
            
            # Filter neighbors: must not be part of the cluster itself
            cluster_set = set(cluster_global_indices)
            external_neighbors = [idx for idx in neighbor_indices if idx not in cluster_set]
            
            if not external_neighbors:
                continue
                
            # 3. Check labels of external neighbors
            neighbor_labels = out[external_neighbors]
            # Exclude background labels from the "rim" check
            rim_labels = neighbor_labels[(neighbor_labels != UNASSIGNED) & (neighbor_labels != BODY_LABEL)]
            
            unique_rim_labels = np.unique(rim_labels)
            
            # 4. If the cluster is surrounded by exactly ONE anatomical label, fill it
            if len(unique_rim_labels) == 1:
                fill_label = unique_rim_labels[0]
                out[cluster_global_indices] = fill_label
                
        return out

    def _find_shared_pv_vertices(self, tri_label: np.ndarray, pv_labels: set) -> np.ndarray:
        """Finds vertex indices that are incident to triangles of different PV labels."""
        n_verts = self._points.shape[0]
        
        # Store the 'first' PV label found for each vertex
        first_label = np.full(n_verts, UNASSIGNED, dtype=np.int32)
        # Mark vertices that have encountered a second, different PV label
        is_shared = np.zeros(n_verts, dtype=bool)
        
        for i in range(3):
            v_indices = self._triangles[:, i]
            t_labels = tri_label
            
            # Only process triangles that are labeled as one of the 4 PVs
            mask = np.isin(t_labels, list(pv_labels))
            
            current_v = v_indices[mask]
            current_l = t_labels[mask]
            
            # Check for conflict: 
            # Vertex already has a label AND that label is different from current triangle label
            has_prev = first_label[current_v] != UNASSIGNED
            conflict = has_prev & (first_label[current_v] != current_l)
            is_shared[current_v[conflict]] = True
            
            # Update first_label for vertices that haven't been assigned yet
            first_label[current_v[~has_prev]] = current_l[~has_prev]
            
        return np.where(is_shared)[0]


__all__ = ["RegionTagger", "TaggerConfig", "LABELS", "BODY_LABEL"]
