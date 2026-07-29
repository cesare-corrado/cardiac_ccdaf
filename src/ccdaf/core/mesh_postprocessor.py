"""
MeshPostprocessor
=================

Mesh quality post-processing routines usable at any stage of the pipeline:

* ``decimate``  - simulated annealing subset + retriangulation
* ``refine``    - vtkAdaptiveSubdivisionFilter-like refinement to a target
                  edge length (split only)
* ``remesh``    - isotropic resampling into an edge-length band (split and
                  collapse)
* ``clean``     - merge duplicates, drop non-connected points, remove
                  non-manifold / degenerate cells, orient normals, and
                  repair low-quality triangles while preserving labelled
                  surfaces
* ``improve_quality`` - the repair step on its own: relocate vertices to
                  fix badly-shaped triangles without touching topology
* ``fill_holes``- close boundary loops below a radius threshold, leaving
                  larger (anatomical) openings open

All routines return a fresh ``pyvista.PolyData`` with the point / cell
data from the input mesh transferred onto the new topology via nearest-
neighbour lookup (integer arrays are preserved exactly). ``remesh`` is the
exception for cell data: every output triangle knows the input triangle it
descends from, so labels transfer exactly rather than by proximity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence, Set

import numpy as np
import pyvista as pv
import vtk
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist, pdist


# =====================================================================
# data transfer
# =====================================================================
def _transfer_arrays(src: pv.PolyData, dst: pv.PolyData) -> None:
    """Copy all point / cell arrays from *src* onto *dst* by nearest-point
    and nearest-cell-centroid lookup. Integer dtypes are preserved."""
    if src.n_points and dst.n_points and src.point_data:
        tree = cKDTree(np.asarray(src.points))
        _, pid = tree.query(np.asarray(dst.points), k=1)
        for name in list(src.point_data.keys()):
            arr = np.asarray(src.point_data[name])
            dst.point_data[name] = arr[pid]

    if src.n_cells and dst.n_cells and src.cell_data:
        src_c = np.asarray(src.cell_centers().points)
        dst_c = np.asarray(dst.cell_centers().points)
        tree = cKDTree(src_c)
        _, cid = tree.query(dst_c, k=1)
        for name in list(src.cell_data.keys()):
            if name == "render_idx":
                continue
            arr = np.asarray(src.cell_data[name])
            dst.cell_data[name] = arr[cid]


def _strip_new_arrays(mesh: pv.PolyData,
                      keep_point: Set[str],
                      keep_cell: Set[str]) -> None:
    """Drop every point / cell array whose name is not in the keep sets.

    Intermediate VTK filters attach bookkeeping arrays to their output
    (``RegionId`` from ``connectivity()``, ``vtkOriginalPointIds`` /
    ``vtkOriginalCellIds`` from ``extract_surface()``). Those must not
    reach a caller, who would otherwise see them as selectable fields.
    The keep sets are snapshotted from the input mesh, so an array the
    input already carried is retained rather than mistaken for one an
    intermediate filter introduced."""
    for name in list(mesh.point_data.keys()):
        if name not in keep_point:
            del mesh.point_data[name]
    for name in list(mesh.cell_data.keys()):
        if name not in keep_cell:
            del mesh.cell_data[name]


# =====================================================================
# geometry helpers
# =====================================================================
def _faces_to_tri(mesh: pv.PolyData) -> np.ndarray:
    f = np.asarray(mesh.faces).reshape(-1, 4)
    if np.any(f[:, 0] != 3):
        raise ValueError("non-triangle face detected")
    return f[:, 1:].astype(np.int64, copy=False)


def _tri_to_faces(tri: np.ndarray) -> np.ndarray:
    n = tri.shape[0]
    out = np.empty((n, 4), dtype=np.int64)
    out[:, 0] = 3
    out[:, 1:] = tri
    return out.ravel()


def _median_edge_length(mesh: pv.PolyData) -> float:
    tri = _faces_to_tri(mesh)
    p = np.asarray(mesh.points)
    e0 = np.linalg.norm(p[tri[:, 1]] - p[tri[:, 0]], axis=1)
    e1 = np.linalg.norm(p[tri[:, 2]] - p[tri[:, 1]], axis=1)
    e2 = np.linalg.norm(p[tri[:, 0]] - p[tri[:, 2]], axis=1)
    return float(np.median(np.concatenate([e0, e1, e2])))


# =====================================================================
# DECIMATE (simulated annealing)
# =====================================================================
def _calc_delta_energy(X: np.ndarray,
                       points: np.ndarray,
                       neigh,
                       neigh_ind: int,
                       choice_ind: int) -> float:
    """Energy change when moving ``points[choice_ind]`` to ``X[neigh[neigh_ind]]``.

    Direct port of ``meshutils.calc_delta_energy``: inverse-square
    interaction energy summed over all other chosen points (the
    self-index is excluded from both old and new sums).
    """
    old_dists = cdist(points[choice_ind:choice_ind + 1], points)[0]
    new_dists = cdist(X[neigh[neigh_ind]][None, :], points)[0]
    with np.errstate(divide="ignore"):
        old_energy = ((1.0 / old_dists[:choice_ind] ** 2).sum()
                      + (1.0 / old_dists[choice_ind + 1:] ** 2).sum())
        new_energy = ((1.0 / new_dists[:choice_ind] ** 2).sum()
                      + (1.0 / new_dists[choice_ind + 1:] ** 2).sum())
    return float(new_energy - old_energy)


def _subset_anneal(X: np.ndarray,
                   tri: np.ndarray,
                   num: int,
                   num_designs: int,
                   choice: Optional[np.ndarray] = None,
                   rng: Optional[np.random.Generator] = None,
                   verbose: bool = True) -> tuple[np.ndarray, bool]:
    """Distribute ``num`` vertices over the input mesh via simulated
    annealing (inverse-square repulsion). Direct port of
    ``meshutils.subset_anneal``: greedy acceptance of neighbour moves
    that lower the energy, progress report every 10 000 iterations,
    and early exit when the successful-move rate drops to ≤ 1 %.

    Parameters
    ----------
    choice
        Optional starting subset (indices into ``X``). When ``None`` a
        random initial subset of size ``num`` is drawn; otherwise the
        caller's array is taken as the seed (allowing outer-loop
        resumption).

    Returns
    -------
    choice : np.ndarray
        Final vertex indices (length ``num``).
    converged : bool
        ``True`` if the successful-move rate over this call dropped to
        ≤ 1 % — signals the caller that further iterations are unlikely
        to improve the layout.
    """
    import trimesh  # local import keeps GUI startup light

    if rng is None:
        rng = np.random.default_rng()

    #if verbose:
    #    print("Optimizing inducing point positions with simulated annealing...")

    mesh = trimesh.Trimesh(vertices=X, faces=tri, process=False)
    if choice is None:
        choice = np.arange(X.shape[0])
        rng.shuffle(choice)
        choice = choice[:num].copy()
    else:
        choice = np.asarray(choice, dtype=np.int64).copy()

    points = X[choice].copy()
    dists = pdist(points)
    dists = dists[dists > 0]
    best_cost = ((1.0 / dists) ** 2).sum() if dists.size else 0.0
    neighbours = mesh.vertex_neighbors

    converged = False
    batch_count = 0      # successful moves in current 10 000-design batch
    total_count = 0      # successful moves across the whole call

    for idesign in range(1, num_designs + 1):
        choice_ind = int(rng.integers(0, choice.shape[0]))
        neigh = neighbours[int(choice[choice_ind])]
        if len(neigh) == 0:
            continue
        neigh_ind = int(rng.integers(0, len(neigh)))
        diff_energy = _calc_delta_energy(X, points, neigh, neigh_ind, choice_ind)

        if diff_energy < 0:  # greedy acceptance (matches meshutils)
            batch_count += 1
            total_count += 1
            best_cost += diff_energy
            choice[choice_ind] = int(neigh[neigh_ind])
            points[:] = X[choice]

        if idesign % 10000 == 0:
            perc = 100.0 * batch_count / 10000.0
            if verbose:
                print(
                    "Progress {:02d}%, Percentage of successful moves: {:4.1f}%".format(
                        int(100 * idesign / num_designs), perc
                    ),
                    end="\r",
                )
            batch_count = 0
            if perc < 1.0:
                if verbose:
                    print("\nBreaking at <= 1% successful moves")
                converged = True
                break

    # Call-level convergence check (covers num_designs < 10 000 where the
    # per-batch test above never fires).
    if not converged and num_designs > 0:
        perc_total = 100.0 * total_count / num_designs
        if perc_total < 1.0:
            converged = True

    return choice, converged


def _subset_triangulate(X: np.ndarray,
                        tri: np.ndarray,
                        choice: np.ndarray,
                        verbose: bool = True) -> pv.PolyData:
    """Re-triangulate the decimated vertex subset.

    Direct port of ``meshutils.subset_triangulate``:

    1. Find the nearest chosen vertex for every original vertex
       (``closest_c``).
    2. Build the new edge list from the original ``edges_unique``,
       mapped through ``closest_c`` and de-duplicated.
    3. For every chosen vertex ``cc``, enumerate pairs of neighbours
       ``(a, b)`` such that the edge ``a—b`` also exists in the new
       edge list, and register ``(a, b, cc)`` as a face. This is
       equivalent to enumerating 3-cliques in the new edge graph.
    4. Iteratively drop faces touching edges shared by more than two
       faces (non-manifold), then remove boundary triangles with an
       angle ``> 135°`` (two passes, matches meshutils).
    5. Re-orient normals consistently.
    """
    import trimesh

    if verbose:
        print("Calculating nearest inducing point")

    tree = cKDTree(X[choice])
    _, closest_c = tree.query(X, k=1)

    if verbose:
        print("Building edge list...")

    mesh = trimesh.Trimesh(vertices=X, faces=tri, process=False)
    edges = np.asarray(mesh.edges_unique)
    closest_c_edges = closest_c[edges.flatten()].reshape(-1, 2)
    keep = closest_c_edges[:, 1] != closest_c_edges[:, 0]
    edge_list = np.sort(closest_c_edges[keep], axis=1)
    edge_list = np.unique(edge_list, axis=0)

    if verbose:
        print("Building face list...")

    # adjacency in the new-vertex graph
    adj: dict[int, Set[int]] = {i: set() for i in range(choice.shape[0])}
    for a, b in edge_list:
        adj[int(a)].add(int(b))
        adj[int(b)].add(int(a))

    face_list: list[tuple[int, int, int]] = []
    for cc in range(choice.shape[0]):
        neigh = adj[cc]
        for a in neigh:
            if a <= cc:
                continue
            for b in (neigh & adj[a]):
                if b <= a:
                    continue
                face_list.append((cc, a, b))
    if not face_list:
        raise RuntimeError("decimation produced no faces")
    face_arr = np.sort(np.array(face_list, dtype=np.int64), axis=1)
    face_arr = np.unique(face_arr, axis=0)

    if verbose:
        print("Removing offending vertices... (should be rapid, else stuck in while loop)")

    sub = trimesh.Trimesh(vertices=X[choice], faces=face_arr, process=False)
    # iterative manifoldisation: drop faces using any edge shared by > 2 faces
    while True:
        unique, counts = np.unique(sub.faces_unique_edges, return_counts=True)
        bad_edges = unique[counts > 2]
        if bad_edges.size == 0:
            break
        bad_face_mask = np.any(np.isin(sub.faces_unique_edges, bad_edges), axis=1)
        new_faces = sub.faces[~bad_face_mask]
        if new_faces.shape[0] == sub.faces.shape[0]:
            break
        sub = trimesh.Trimesh(vertices=sub.vertices, faces=new_faces, process=False)

    # two passes: remove open-boundary triangles with any angle > 135°
    DEG_TO_RAD = np.pi / 180.0
    for _ in range(2):
        angles = sub.face_angles
        bad_triangles = np.any(angles > 135 * DEG_TO_RAD, axis=1)
        unique, counts = np.unique(sub.faces_unique_edges, return_counts=True)
        boundary_edges = unique[counts == 1]
        is_edge_face = np.any(np.isin(sub.faces_unique_edges, boundary_edges), axis=1)
        keep = ~(bad_triangles & is_edge_face)
        sub = trimesh.Trimesh(vertices=sub.vertices, faces=sub.faces[keep], process=False)

    sub.fix_normals()
    return pv.PolyData(np.asarray(sub.vertices),
                       _tri_to_faces(np.asarray(sub.faces, dtype=np.int64)))


def decimate(mesh: pv.PolyData,
             target_points: int,
             n_iters: int = 200,
             seed: Optional[int] = None,
             max_hole_size: float = 0.0,
             on_progress: Optional["Callable[[int, int], None]"] = None
             ) -> pv.PolyData:
    """Decimate *mesh* to roughly ``target_points`` vertices via simulated
    annealing.

    Runs ``_subset_anneal`` inside an outer loop of at most ``n_iters``
    calls, each executing ``X.shape[0]`` design trials. The loop stops
    early as soon as ``_subset_anneal`` reports convergence. Point /
    cell data are transferred to the new topology.

    ``max_hole_size`` > 0 triggers a hole-filling pass
    (``vtkFillHolesFilter``) after retriangulation, closing any hole
    whose bounding-sphere radius is below the threshold. Large openings
    (mitral valve, PV ostia) stay open provided the threshold is set
    well below their radius. ``0`` disables hole filling.

    ``on_progress(i, n_iters)`` — when provided — is called after each
    outer-loop iteration (``i`` = number of completed outer calls).
    """
    if target_points <= 0 or target_points > mesh.n_points:
        raise ValueError(
            f"target_points ({target_points}) must be in (1, {mesh.n_points})"
        )
    X = np.asarray(mesh.points)
    tri = _faces_to_tri(mesh)
    rng = np.random.default_rng(seed)

    num_designs = int(X.shape[0])
    choice: Optional[np.ndarray] = None

    if on_progress is not None:
        on_progress(0, n_iters)

    for i in range(n_iters):
        choice, converged = _subset_anneal(
            X, tri,
            num=target_points,
            num_designs=num_designs,
            choice=choice,
            rng=rng,
        )
        if on_progress is not None:
            on_progress(i + 1, n_iters)
        if converged:
            break

    out = _subset_triangulate(X, tri, choice)
    if max_hole_size > 0.0:
        out = _fill_small_holes(out, max_hole_size)
    _transfer_arrays(mesh, out)
    return out


_DEDUP_TOL = 1e-9


def _drop_isolated_triangles(tri: np.ndarray) -> np.ndarray:
    """Drop triangles whose all three edges are boundary (no edge shared
    with any other triangle)."""
    n = tri.shape[0]
    if n == 0:
        return tri
    edges = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    edges_sorted = np.sort(edges, axis=1)
    _, inv, counts = np.unique(edges_sorted, axis=0,
                               return_inverse=True, return_counts=True)
    edge_counts = counts[inv].reshape(3, n)
    isolated = np.all(edge_counts == 1, axis=0)
    return tri[~isolated]


def _dedupe_points(pts: np.ndarray, tri: np.ndarray, tol: float
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge points whose coordinates coincide within absolute ``tol``.
    Returns (new_pts, new_tri, first_idx) where ``first_idx[k]`` is the
    original index that became unique row ``k`` (so external arrays of
    length ``len(pts)`` can be sliced as ``arr[first_idx]``)."""
    if pts.shape[0] == 0:
        return pts, tri, np.empty(0, dtype=np.int64)
    snapped = np.rint(pts / tol).astype(np.int64)
    _, first_idx, inverse = np.unique(snapped, axis=0,
                                      return_index=True,
                                      return_inverse=True)
    new_pts = pts[first_idx]
    new_tri = inverse[tri] if tri.size else tri
    return new_pts, new_tri, first_idx


def _dedupe_triangles(tri: np.ndarray) -> np.ndarray:
    """Drop duplicate triangles (winding-insensitive: two triangles with
    the same set of three vertex indices collapse to one)."""
    if tri.shape[0] == 0:
        return tri
    sorted_tri = np.sort(tri, axis=1)
    _, first_idx = np.unique(sorted_tri, axis=0, return_index=True)
    first_idx.sort()
    return tri[first_idx]


def _drop_degenerate_triangles(tri: np.ndarray) -> np.ndarray:
    """Drop triangles with fewer than three unique vertex indices."""
    if tri.shape[0] == 0:
        return tri
    keep = ((tri[:, 0] != tri[:, 1]) &
            (tri[:, 1] != tri[:, 2]) &
            (tri[:, 0] != tri[:, 2]))
    return tri[keep]


def _drop_nonmanifold_edges(tri: np.ndarray) -> np.ndarray:
    """Drop all triangles incident to any edge shared by ≥3 triangles."""
    n = tri.shape[0]
    if n == 0:
        return tri
    edges = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    edges_sorted = np.sort(edges, axis=1)
    _, inv, counts = np.unique(edges_sorted, axis=0,
                               return_inverse=True, return_counts=True)
    edge_counts = counts[inv].reshape(3, n)
    bad = np.any(edge_counts >= 3, axis=0)
    return tri[~bad]


def _compact_points(pts: np.ndarray, tri: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop points referenced by no triangle and remap indices. Returns
    (new_pts, new_tri, used) with ``used`` = original indices kept."""
    if tri.shape[0] == 0:
        return (np.empty((0, 3), dtype=pts.dtype),
                tri,
                np.empty(0, dtype=np.int64))
    used = np.unique(tri.ravel())
    remap = np.full(pts.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(used.shape[0])
    return pts[used], remap[tri], used


def _boundary_edges(tri: np.ndarray) -> np.ndarray:
    """Return Nx2 array of edges used by exactly one triangle."""
    if tri.shape[0] == 0:
        return np.empty((0, 2), dtype=np.int64)
    edges = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    edges_sorted = np.sort(edges, axis=1)
    uniq, counts = np.unique(edges_sorted, axis=0, return_counts=True)
    return uniq[counts == 1]


def _angular_partners(v: int,
                      nbrs: list[int],
                      pts: np.ndarray) -> dict[int, int]:
    """Pair ``v``'s boundary neighbours by sorting them angularly around
    ``v`` in the local tangent plane (PCA of ``{v} ∪ nbrs``) and pairing
    consecutive neighbours. Returns dict mapping each neighbour to its
    angular partner."""
    p_v = pts[v]
    p_nbrs = pts[nbrs]
    coords = np.vstack([p_v[None, :], p_nbrs])
    centred = coords - coords.mean(axis=0)
    try:
        _, _, Vt = np.linalg.svd(centred, full_matrices=False)
    except np.linalg.LinAlgError:
        # Degenerate plane — fall back to listing order.
        order = np.arange(len(nbrs))
    else:
        e1, e2 = Vt[0], Vt[1]
        rel = p_nbrs - p_v
        angles = np.arctan2(rel @ e2, rel @ e1)
        order = np.argsort(angles, kind="stable")
    partner: dict[int, int] = {}
    k = len(nbrs)
    for i in range(0, k - 1, 2):
        a = nbrs[order[i]]
        b = nbrs[order[i + 1]]
        partner[a] = b
        partner[b] = a
    return partner


def _build_partner_map(pts: np.ndarray, tri: np.ndarray
                       ) -> tuple[set[int], dict[int, dict[int, int]]]:
    """Build the per-vertex partner map used by the loop walker. At a
    manifold boundary vertex (2 boundary neighbours) the two neighbours
    are paired with each other; at a non-manifold one (≥4) the angular
    pairing rule from :func:`_angular_partners` is used."""
    boundary = _boundary_edges(tri)
    if boundary.shape[0] == 0:
        return set(), {}
    adj: dict[int, list[int]] = {}
    for a, b in boundary:
        adj.setdefault(int(a), []).append(int(b))
        adj.setdefault(int(b), []).append(int(a))
    partner: dict[int, dict[int, int]] = {}
    for v, nbrs in adj.items():
        # Collapse parallel edges (same neighbour appearing twice) to a
        # single entry; pairing is then between distinct neighbours.
        nbrs_unique = list(dict.fromkeys(nbrs))
        if len(nbrs_unique) == 2:
            a, b = nbrs_unique
            partner[v] = {a: b, b: a}
        elif len(nbrs_unique) >= 4 and len(nbrs_unique) % 2 == 0:
            partner[v] = _angular_partners(v, nbrs_unique, pts)
        # vertices with 1 or odd-count neighbours are left out → loops
        # touching them get aborted by the walker.
    return set(adj.keys()), partner


def _walk_loops(boundary_verts: set[int],
                partner: dict[int, dict[int, int]]
                ) -> list[list[int]]:
    """Walk every closed loop induced by the partner map. Each directed
    half-edge is consumed at most once. Returns loops as vertex lists
    (no repeat at end)."""
    loops: list[list[int]] = []
    used: set[tuple[int, int]] = set()
    safety = sum(len(p) for p in partner.values()) + 8
    for start_v in boundary_verts:
        if start_v not in partner:
            continue
        for first_nbr in list(partner[start_v].keys()):
            if (start_v, first_nbr) in used:
                continue
            loop = [start_v]
            prev, curr = start_v, first_nbr
            # Mark both directions so we don't re-walk the same loop
            # backwards from another seed.
            used.add((prev, curr))
            used.add((curr, prev))
            ok = True
            steps = 0
            while curr != start_v:
                loop.append(curr)
                if curr not in partner or prev not in partner[curr]:
                    ok = False
                    break
                nxt = partner[curr][prev]
                used.add((curr, nxt))
                used.add((nxt, curr))
                prev, curr = curr, nxt
                steps += 1
                if steps > safety:
                    ok = False
                    break
            if ok and len(loop) >= 3:
                loops.append(loop)
    return loops


def _fan_triangulate(loop_arr: np.ndarray) -> np.ndarray:
    """Fan triangulation from ``loop_arr[0]``. Used only as a last-ditch
    fallback when ear-clipping itself can't make progress (numerically
    degenerate polygon)."""
    n = loop_arr.shape[0]
    if n < 3:
        return np.empty((0, 3), dtype=np.int64)
    out = np.empty((n - 2, 3), dtype=np.int64)
    for i in range(1, n - 1):
        out[i - 1] = (loop_arr[0], loop_arr[i], loop_arr[i + 1])
    return out


def _point_in_triangle_2d(p: np.ndarray,
                          a: np.ndarray,
                          b: np.ndarray,
                          c: np.ndarray) -> bool:
    """Inclusive barycentric point-in-triangle test in 2D. Returns True
    if ``p`` lies inside ``abc`` or on its boundary."""
    v0 = c - a
    v1 = b - a
    v2 = p - a
    denom = v0[0] * v1[1] - v0[1] * v1[0]
    if abs(denom) < 1e-20:
        return False
    inv = 1.0 / denom
    u = (v1[1] * v2[0] - v1[0] * v2[1]) * inv
    v = (v0[0] * v2[1] - v0[1] * v2[0]) * inv
    return u >= 0.0 and v >= 0.0 and (u + v) <= 1.0


def _earclip_triangulate(loop_pts: np.ndarray,
                         loop_arr: np.ndarray,
                         forbidden: Optional[set] = None
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate a simple loop by ear-clipping in its best-fit 2D
    plane. Returns ``(triangles, residual)``.

    ``triangles`` are global-indexed (n,3) — admissible ears that have
    been clipped. ``residual`` is the array of *global vertex indices*
    of the sub-polygon that ear-clip could not finish (length 0 if the
    loop was fully triangulated, length ≥ 3 otherwise). The caller is
    expected to handle a non-empty residual (e.g., by inserting a new
    center vertex and fan-triangulating).

    ``forbidden`` is an optional set of ``(min(a,b), max(a,b))`` global
    edges that no new triangle may introduce as a diagonal — those are
    edges already 2-shared in the surrounding mesh, where adding a
    new incident triangle would force a non-manifold ≥3-shared edge."""
    empty_tri = np.empty((0, 3), dtype=np.int64)
    n = loop_pts.shape[0]
    if n < 3:
        return empty_tri, np.empty(0, dtype=np.int64)
    if n == 3:
        a, b, c = int(loop_arr[0]), int(loop_arr[1]), int(loop_arr[2])
        if forbidden and (
                (min(a, b), max(a, b)) in forbidden
                or (min(b, c), max(b, c)) in forbidden
                or (min(a, c), max(a, c)) in forbidden):
            return empty_tri, np.asarray(loop_arr, dtype=np.int64)
        return loop_arr[np.array([[0, 1, 2]], dtype=np.int64)], np.empty(0, dtype=np.int64)

    # PCA projection to 2D
    centred = loop_pts - loop_pts.mean(axis=0)
    try:
        _, _, Vt = np.linalg.svd(centred, full_matrices=False)
    except np.linalg.LinAlgError:
        return empty_tri, np.asarray(loop_arr, dtype=np.int64)
    if Vt.shape[0] < 2:
        return empty_tri, np.asarray(loop_arr, dtype=np.int64)
    P = np.column_stack([centred @ Vt[0], centred @ Vt[1]])
    if not np.all(np.isfinite(P)):
        return empty_tri, np.asarray(loop_arr, dtype=np.int64)

    # Orient CCW (positive signed area).
    x, y = P[:, 0], P[:, 1]
    signed_area = 0.5 * float(
        np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    if signed_area < 0.0:
        P = P[::-1].copy()
        order = np.arange(n - 1, -1, -1, dtype=np.int64)
    else:
        order = np.arange(n, dtype=np.int64)

    indices = list(range(n))           # current polygon as P-indices
    triangles: list[tuple[int, int, int]] = []
    safety = n * n + 4

    def _diagonal_blocked(ip: int, iN: int) -> bool:
        if not forbidden:
            return False
        ga = int(loop_arr[order[ip]])
        gb = int(loop_arr[order[iN]])
        return (min(ga, gb), max(ga, gb)) in forbidden

    while len(indices) > 3 and safety > 0:
        safety -= 1
        m = len(indices)
        ear_k = -1
        for k in range(m):
            ip = indices[(k - 1) % m]
            ic = indices[k]
            iN = indices[(k + 1) % m]
            a, b, c = P[ip], P[ic], P[iN]
            cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if cross <= 0.0:
                continue  # reflex or collinear
            if _diagonal_blocked(ip, iN):
                continue  # would create a forbidden diagonal
            inside = False
            for j in indices:
                if j == ip or j == ic or j == iN:
                    continue
                if _point_in_triangle_2d(P[j], a, b, c):
                    inside = True
                    break
            if not inside:
                ear_k = k
                break
        if ear_k < 0:
            break  # no admissible ear — leave the rest open
        ip = indices[(ear_k - 1) % m]
        ic = indices[ear_k]
        iN = indices[(ear_k + 1) % m]
        triangles.append((int(order[ip]), int(order[ic]), int(order[iN])))
        indices.pop(ear_k)

    if len(indices) == 3:
        i0, i1, i2 = indices
        if not (_diagonal_blocked(i0, i1)
                or _diagonal_blocked(i1, i2)
                or _diagonal_blocked(i0, i2)):
            triangles.append((int(order[i0]),
                              int(order[i1]),
                              int(order[i2])))
            indices = []  # fully consumed
    # Whatever remains (≥ 3 vertices) is the residual sub-polygon.
    residual = np.array([int(loop_arr[order[i]]) for i in indices],
                        dtype=np.int64)

    if not triangles:
        return empty_tri, residual
    tri_local = np.array(triangles, dtype=np.int64)
    return loop_arr[tri_local], residual


def _projected_loop_is_simple(proj: np.ndarray) -> bool:
    """True when the projected loop polygon has no self-crossing edges.

    A loop non-planar enough to cross itself once flattened cannot be
    triangulated by ``vtkDelaunay2D``: the crossing leaves at least one
    constraint edge unrecoverable, which the filter reports as "Edge not
    recovered, polygon fill suspect" and then returns a fill the caller has
    to reject anyway. Testing first skips the doomed call and its warning.

    Only proper crossings count. Loops that merely touch at a vertex are
    pinches, and are split upstream by :func:`_split_loop_at_pinches`.
    """
    n = proj.shape[0]
    if n < 4:
        return True

    def side(a, b, c) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    for i in range(n):
        p1, p2 = proj[i], proj[(i + 1) % n]
        for j in range(i + 1, n):
            # Edges sharing a vertex legitimately touch.
            if (j + 1) % n == i or (i + 1) % n == j:
                continue
            p3, p4 = proj[j], proj[(j + 1) % n]
            d1, d2 = side(p3, p4, p1), side(p3, p4, p2)
            d3, d4 = side(p1, p2, p3), side(p1, p2, p4)
            if (((d1 > 0) and (d2 < 0)) or ((d1 < 0) and (d2 > 0))) and \
               (((d3 > 0) and (d4 < 0)) or ((d3 < 0) and (d4 > 0))):
                return False
    return True


def _delaunay_triangulate(loop_pts: np.ndarray,
                          loop_arr: np.ndarray) -> Optional[np.ndarray]:
    """Constrained Delaunay triangulation of a single loop. Project the
    loop to its best-fit 2D plane (PCA), feed to ``vtkDelaunay2D`` with
    the loop polygon as constraint. Returns triangle array (global
    indices) or ``None`` on failure."""
    n = loop_pts.shape[0]
    if n < 3:
        return None
    centred = loop_pts - loop_pts.mean(axis=0)
    try:
        _, _, Vt = np.linalg.svd(centred, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if Vt.shape[0] < 2:
        return None
    e1, e2 = Vt[0], Vt[1]
    proj = np.column_stack([centred @ e1, centred @ e2])
    if not np.all(np.isfinite(proj)):
        return None
    if not _projected_loop_is_simple(proj):
        return None
    pts3d = np.zeros((n, 3), dtype=np.float64)
    pts3d[:, :2] = proj

    inp = pv.PolyData(pts3d)
    src = pv.PolyData(pts3d)
    polygon_conn = np.empty(n + 1, dtype=np.int64)
    polygon_conn[0] = n
    polygon_conn[1:] = np.arange(n, dtype=np.int64)
    src.faces = polygon_conn

    flt = vtk.vtkDelaunay2D()
    flt.SetInputData(inp)
    flt.SetSourceData(src)
    try:
        flt.Update()
    except Exception:
        return None
    out = flt.GetOutput()
    if out is None or out.GetNumberOfCells() == 0:
        return None
    try:
        face_arr = np.asarray(pv.wrap(out).faces).reshape(-1, 4)
    except Exception:
        return None
    if face_arr.shape[1] != 4 or np.any(face_arr[:, 0] != 3):
        return None
    local = face_arr[:, 1:].astype(np.int64)
    return loop_arr[local]


def _split_loop_at_pinches(loop: list[int]) -> list[list[int]]:
    """Split a loop at any vertex visited more than once (figure-8 pinch
    introduced by the angular re-pairing rule). Returns simple sub-loops
    with no repeated vertex; sub-loops with fewer than 3 vertices are
    dropped (they enclose no area)."""
    seen: dict[int, int] = {}
    for i, v in enumerate(loop):
        if v in seen:
            j = seen[v]
            inner = loop[j:i]
            outer = loop[i:] + loop[:j]
            return (_split_loop_at_pinches(inner)
                    + _split_loop_at_pinches(outer))
        seen[v] = i
    return [loop] if len(loop) >= 3 else []


def _filter_forbidden_triangles(tri: np.ndarray,
                                forbidden: set) -> np.ndarray:
    """Drop triangles that introduce any edge from ``forbidden`` (edges
    already 2-shared in the surrounding mesh, where adding a new
    incident triangle would create a non-manifold ≥3-shared edge)."""
    if tri.shape[0] == 0 or not forbidden:
        return tri
    keep = np.ones(tri.shape[0], dtype=bool)
    for i in range(tri.shape[0]):
        a, b, c = int(tri[i, 0]), int(tri[i, 1]), int(tri[i, 2])
        e0 = (min(a, b), max(a, b))
        e1 = (min(b, c), max(b, c))
        e2 = (min(a, c), max(a, c))
        if e0 in forbidden or e1 in forbidden or e2 in forbidden:
            keep[i] = False
    return tri[keep]


def _delaunay_covers_all_loop_edges(filler: np.ndarray,
                                    loop_arr: np.ndarray) -> bool:
    """True if every loop boundary edge appears in at least one filler
    triangle. Used to detect when the post-forbidden-filter Delaunay
    output left part of the loop uncovered."""
    if filler.shape[0] == 0:
        return False
    n = loop_arr.shape[0]
    needed = {(int(min(loop_arr[i], loop_arr[(i + 1) % n])),
               int(max(loop_arr[i], loop_arr[(i + 1) % n])))
              for i in range(n)}
    seen: set = set()
    for t in filler:
        a, b, c = int(t[0]), int(t[1]), int(t[2])
        for u, v in ((a, b), (b, c), (a, c)):
            seen.add((min(u, v), max(u, v)))
    return needed.issubset(seen)


def _fill_holes(pts: np.ndarray,
                tri: np.ndarray,
                max_size: float,
                point_origin: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Walk boundary loops, split figure-8 pinches into simple sub-loops,
    filter each by size, triangulate (Delaunay → ear-clip → centroid-
    fan), and append to ``tri``. Anatomical openings
    (radius > ``max_size``) stay open.

    Triangulators are constrained not to introduce any edge that
    already exists as a 2-shared interior edge of the surrounding mesh
    — preventing topologically-false holes (where opposite corners are
    connected through a different part of the surface) from being
    closed in a way that would create non-manifold geometry. When ear-
    clip cannot finish a loop under that constraint, a fresh center
    vertex is inserted at the residual sub-polygon's centroid and the
    residual is fan-triangulated from it. Adding a brand-new vertex is
    unconditionally manifold-safe: every edge incident to the new
    vertex is itself new, so it cannot collide with any existing 2-
    shared edge.

    Returns updated (pts, tri, point_origin) — the point array may
    grow when center vertices are inserted; ``point_origin`` is
    extended by inheriting the origin index of the first residual
    vertex, so downstream point_data slicing remains well-defined."""
    boundary_verts, partner = _build_partner_map(pts, tri)
    if not boundary_verts:
        return pts, tri, point_origin

    # Running edge-incidence count, kept up-to-date as fillers commit.
    # An edge with count ≥ 2 must not be re-introduced by any later
    # triangulation — including ones triggered by adjacent loops that
    # share a vertex with the just-filled loop.
    edges = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    edges_sorted = np.sort(edges, axis=1)
    edge_count: dict = {}
    for e in edges_sorted:
        key = (int(e[0]), int(e[1]))
        edge_count[key] = edge_count.get(key, 0) + 1

    def _commit(filler_block: np.ndarray) -> None:
        new_blocks.append(filler_block)
        for t in filler_block:
            for u, v in ((int(t[0]), int(t[1])),
                         (int(t[1]), int(t[2])),
                         (int(t[0]), int(t[2]))):
                key = (min(u, v), max(u, v))
                edge_count[key] = edge_count.get(key, 0) + 1

    raw_loops = _walk_loops(boundary_verts, partner)
    new_blocks: list[np.ndarray] = []
    new_pts_list: list[np.ndarray] = []
    new_origin_list: list[int] = []

    for raw in raw_loops:
        for loop in _split_loop_at_pinches(raw):
            loop_arr = np.array(loop, dtype=np.int64)
            loop_pts = pts[loop_arr]
            centre = loop_pts.mean(axis=0)
            radius = float(np.linalg.norm(loop_pts - centre, axis=1).max())
            if radius > max_size:
                continue

            # Snapshot of edges that would be ≥3-shared if a new
            # triangle used them. Loop edges (count 1) are deliberately
            # NOT in this set — covering them is the whole point.
            forbidden = {e for e, c in edge_count.items() if c >= 2}

            d = _delaunay_triangulate(loop_pts, loop_arr)
            if d is not None and d.shape[0] > 0:
                d = _filter_forbidden_triangles(d, forbidden)
                if (d.shape[0] > 0
                        and _delaunay_covers_all_loop_edges(d, loop_arr)):
                    _commit(d)
                    continue

            ec_tri, residual = _earclip_triangulate(
                loop_pts, loop_arr, forbidden=forbidden)
            if ec_tri.shape[0] > 0:
                _commit(ec_tri)

            if residual.shape[0] >= 3:
                new_idx = (pts.shape[0] + len(new_pts_list))
                res_pts = pts[residual]
                new_pts_list.append(res_pts.mean(axis=0))
                new_origin_list.append(int(point_origin[residual[0]]))
                k = residual.shape[0]
                fan = np.empty((k, 3), dtype=np.int64)
                for i in range(k):
                    fan[i] = (residual[i],
                              residual[(i + 1) % k],
                              new_idx)
                _commit(fan)

    if new_pts_list:
        pts = np.vstack([pts, np.asarray(new_pts_list, dtype=pts.dtype)])
        point_origin = np.concatenate(
            [point_origin, np.asarray(new_origin_list, dtype=np.int64)])
    if new_blocks:
        tri = np.vstack([tri] + new_blocks)
    return pts, tri, point_origin


def _topology_cleanup(pts: np.ndarray,
                      tri: np.ndarray,
                      point_origin: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Iterate cleanup steps 1+2 to a fixed point. Returns updated
    (pts, tri, point_origin). Idempotent — calling twice on a clean mesh
    is a no-op."""
    while True:
        n_tri_before = tri.shape[0]
        n_pts_before = pts.shape[0]

        tri = _drop_isolated_triangles(tri)
        pts, tri, first_idx = _dedupe_points(pts, tri, _DEDUP_TOL)
        point_origin = point_origin[first_idx]
        tri = _dedupe_triangles(tri)
        tri = _drop_degenerate_triangles(tri)
        tri = _drop_nonmanifold_edges(tri)

        if (tri.shape[0] == n_tri_before
                and pts.shape[0] == n_pts_before):
            return pts, tri, point_origin


def _fill_small_holes(mesh: pv.PolyData, max_size: float) -> pv.PolyData:
    """Repair mesh topology and close holes whose bounding-sphere radius
    is ≤ ``max_size``.

    Pipeline
    --------
    Cleanup (always runs, regardless of ``max_size``), iterated to a
    fixed point:

    1. Drop fully-isolated triangles (all three edges are boundary).
    2. Merge points whose coordinates coincide within ``1e-9``; dedupe
       triangles winding-insensitively; drop triangles that collapse to
       fewer than three unique vertices; drop every triangle incident
       to an edge still shared by ≥3 triangles.

    Hole filling (only when ``max_size > 0``):

    3. At every non-manifold boundary vertex, sort its boundary
       neighbours angularly in the local tangent plane (PCA) and pair
       them consecutively. Walk closed loops with this pairing. Any
       loop that visits the same vertex more than once (figure-8
       pinch) is split into simple sub-loops at the pinch. Loops
       larger than ``max_size`` (bounding-sphere radius around the
       loop centroid) stay open — that's how anatomical openings keep
       their identity. Each accepted (sub-)loop is triangulated with
       ``vtkDelaunay2D`` constrained by the loop polygon, projected to
       the loop's best-fit plane. If Delaunay fails or produces no
       triangles, fall back to fan triangulation and emit a
       ``RuntimeWarning``. Cleanup steps 1+2 are then re-run as a
       safety net to absorb any duplicate or non-manifold triangles
       the fan fallback may have introduced.

    A final ``vtkPolyDataNormals`` pass orients all triangles
    consistently. Orphaned points (referenced by no surviving triangle)
    are dropped and ``point_data`` is sliced accordingly. Cell data is
    intentionally not carried — the caller's ``_transfer_arrays`` step
    repopulates it via nearest-centroid lookup.
    """
    pts = np.asarray(mesh.points, dtype=np.float64).copy()
    tri = _faces_to_tri(mesh)
    if tri.size == 0:
        return mesh

    # Track which original point each current point came from so that
    # point_data can be sliced through both dedup and orphan compaction.
    point_origin = np.arange(pts.shape[0], dtype=np.int64)

    # ---- cleanup: iterate steps 1+2 to a fixed point ----------------
    pts, tri, point_origin = _topology_cleanup(pts, tri, point_origin)
    pts, tri, used = _compact_points(pts, tri)
    point_origin = point_origin[used]

    # ---- step 3: fill loops (only when requested) -------------------
    if max_size > 0.0 and tri.shape[0] > 0:
        pts, tri, point_origin = _fill_holes(pts, tri, max_size,
                                             point_origin)
        # Safety-net cleanup: should be a no-op given the constrained
        # triangulators, but kept as a defence-in-depth measure.
        pts, tri, point_origin = _topology_cleanup(pts, tri, point_origin)
        pts, tri, used = _compact_points(pts, tri)
        point_origin = point_origin[used]

    # ---- assemble output --------------------------------------------
    if tri.shape[0] == 0:
        out = pv.PolyData(pts) if pts.shape[0] else pv.PolyData()
    else:
        out = pv.PolyData(pts, _tri_to_faces(tri))

    for name in list(mesh.point_data.keys()):
        arr = np.asarray(mesh.point_data[name])
        out.point_data[name] = arr[point_origin]

    if tri.shape[0] == 0:
        return out

    # ---- consistent orientation -------------------------------------
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(out)
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.SplittingOff()
    normals.ComputePointNormalsOff()
    normals.ComputeCellNormalsOff()
    normals.Update()
    oriented = pv.wrap(normals.GetOutput())
    if not isinstance(oriented, pv.PolyData):
        return out
    for name in list(out.point_data.keys()):
        if name not in oriented.point_data:
            oriented.point_data[name] = np.asarray(out.point_data[name])
    return oriented


SMOOTH_TAUBIN = "taubin"
SMOOTH_LAPLACIAN = "laplacian"


def smooth(mesh: pv.PolyData,
           method: str = SMOOTH_TAUBIN,
           iterations: int = 40,
           passband: float = 0.001,
           relaxation: float = 0.1,
           feature_angle: float = 180.0) -> pv.PolyData:
    """Smooth the whole surface, stripping acquisition noise.

    Unlike :func:`clean`'s quality smoothing — which only nudges vertices of
    badly-shaped triangles and leaves the anatomy alone — this moves every
    vertex, and so reshapes the surface.

    * ``taubin`` (``vtkWindowedSincPolyDataFilter``) alternates a shrinking
      and an inflating pass, so it removes roughness without deflating the
      shell (~+0.1% volume on a Carto atrium). ``passband`` sets how much is
      removed; smaller is smoother. The filter is only stable for modest
      ``iterations`` relative to mesh density: 40 is right for a ~13k-point
      Carto surface, costs ~4% of the volume on the same surface decimated
      to 4k, and runs away on a very coarse one — so lower it when smoothing
      a decimated mesh.
    * ``laplacian`` (``vtkSmoothPolyDataFilter``) moves each vertex toward
      its neighbours' average. Simpler, but it shrinks progressively with
      ``iterations`` (~-2.5% volume at 100), which matters when the shell is
      later measured.

    Topology, point count and point order are preserved, so the caller can
    read the displacement off as ``out.points - mesh.points`` — that vertex
    correspondence is what lets electrodes follow the surface. Point and cell
    arrays are copied across by index.
    """
    if method == SMOOTH_TAUBIN:
        flt = vtk.vtkWindowedSincPolyDataFilter()
        flt.SetInputData(mesh)
        flt.SetNumberOfIterations(int(iterations))
        flt.SetPassBand(float(passband))
        flt.SetFeatureAngle(float(feature_angle))
        flt.FeatureEdgeSmoothingOn()
        flt.BoundarySmoothingOn()
        flt.NonManifoldSmoothingOn()
        flt.NormalizeCoordinatesOn()
    elif method == SMOOTH_LAPLACIAN:
        flt = vtk.vtkSmoothPolyDataFilter()
        flt.SetInputData(mesh)
        flt.SetNumberOfIterations(int(iterations))
        flt.SetRelaxationFactor(float(relaxation))
        flt.SetFeatureAngle(float(feature_angle))
        flt.FeatureEdgeSmoothingOn()
        flt.BoundarySmoothingOn()
        flt.SetConvergence(0.0)
    else:
        raise ValueError(f"unknown smoothing method: {method!r}")
    flt.Update()

    out = pv.wrap(flt.GetOutput())
    if out.n_points != mesh.n_points:
        raise RuntimeError(
            f"{method} smoothing changed the point count "
            f"({mesh.n_points} -> {out.n_points}); the vertex correspondence "
            "callers rely on is gone"
        )
    for name in list(mesh.point_data.keys()):
        out.point_data[name] = np.asarray(mesh.point_data[name])
    for name in list(mesh.cell_data.keys()):
        if name == "render_idx":
            continue
        out.cell_data[name] = np.asarray(mesh.cell_data[name])
    return out


def fill_holes(mesh: pv.PolyData, max_size: float) -> pv.PolyData:
    """Close boundary loops whose bounding-sphere radius is ≤ ``max_size``.

    ``max_size`` is an absolute length in mesh units (mm for Carto
    exports), measured as the maximum distance from a loop's vertices to
    their centroid. Openings larger than it stay open — that is how
    genuine anatomical openings (PV ostia, mitral valve) keep their
    identity, so the threshold must sit well below their radius. A
    ``max_size`` above the largest loop radius therefore closes
    everything.

    Topology cleanup (isolated / duplicate / degenerate / non-manifold
    cell removal) runs regardless, and the result is consistently
    oriented. Point and cell arrays are transferred onto the repaired
    topology by nearest-neighbour lookup, so integer cell arrays such as
    ``elemTag`` survive — ``_fill_small_holes`` alone does not carry
    cell data.
    """
    if max_size <= 0:
        raise ValueError("max_size must be positive")
    out = _fill_small_holes(mesh, max_size)
    if out is mesh:            # nothing to do (no faces) — never alias the input
        return mesh.copy()
    _transfer_arrays(mesh, out)
    return out


# =====================================================================
# REFINE (vtkAdaptiveSubdivisionFilter-like)
# =====================================================================
def refine(mesh: pv.PolyData,
           edge_len: float,
           max_area: Optional[float] = None) -> pv.PolyData:
    """Adaptively subdivide triangles whose longest edge exceeds
    ``edge_len``. ``max_area`` defaults to ``0.5 * edge_len**2``."""
    if edge_len <= 0:
        raise ValueError("edge_len must be positive")
    if max_area is None:
        max_area = 0.5 * edge_len * edge_len

    flt = vtk.vtkAdaptiveSubdivisionFilter()
    flt.SetInputData(mesh)
    flt.SetMaximumEdgeLength(edge_len)
    flt.SetMaximumTriangleArea(max_area)
    flt.SetMaximumNumberOfPasses(50)
    flt.Update()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(flt.GetOutputPort())
    normals.ComputeCellNormalsOn()
    normals.ComputePointNormalsOff()
    normals.SplittingOff()
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.Update()
    
    out = pv.wrap(normals.GetOutput())
    if not isinstance(out, pv.PolyData):
        raise RuntimeError("refinement returned non-PolyData")
    # vtkAdaptiveSubdivisionFilter interpolates point data itself; cell data
    # is passed through via original-cell inheritance. Re-transfer to be sure
    # integer arrays survive without being cast.
    _transfer_arrays(mesh, out)
    return out


# =====================================================================
# REMESH (isotropic resampling to an edge-length band)
# =====================================================================
REFINE_ADAPTIVE = "adaptive"
REFINE_RESAMPLE = "resample"

# Factors turning a target edge length into a ``[min, max]`` band. Which
# pair applies depends on how the target compares with the mesh's current
# mean edge length — see _band_from_target.
_BAND_FINE = (0.50, 1.8)     # target much finer than the mesh
_BAND_NEAR = (0.65, 1.7)     # target comparable to the mesh
_BAND_COARSE = (0.75, 1.8)   # target much coarser than the mesh

# How many collapse / flip batches one pass may run before moving on. Both
# operations lock what they touch for the rest of a batch, so one batch only
# thins a dense field of candidates — see remesh.
_COLLAPSE_BATCHES = 10
# 6 is where flipping has converged: raising it to 10 produces an identical
# mesh, it just spends more sweeps discovering there is nothing left to do.
_FLIP_BATCHES = 6
# How much local triangle quality a valence-improving flip may cost. Loose
# on purpose: a flip often pays off only once its neighbours follow, so a
# tight bound blocks the whole cascade and the connectivity never improves.
# This is a guard against a catastrophic single flip, not an optimiser.
_FLIP_QUALITY_TOL = 0.5


def _mean_edge_length(mesh: pv.PolyData) -> float:
    """Mean edge length over element edges, i.e. interior edges counted
    once per incident triangle."""
    tri = _faces_to_tri(mesh)
    p = np.asarray(mesh.points)
    e0 = np.linalg.norm(p[tri[:, 1]] - p[tri[:, 0]], axis=1)
    e1 = np.linalg.norm(p[tri[:, 2]] - p[tri[:, 1]], axis=1)
    e2 = np.linalg.norm(p[tri[:, 0]] - p[tri[:, 2]], axis=1)
    return float(np.concatenate([e0, e1, e2]).mean())


def _band_from_target(mesh: pv.PolyData,
                      target: float) -> tuple[float, float]:
    """Derive ``(min_edge, max_edge)`` from a target edge length.

    A single target is not enough to remesh with: splitting needs an upper
    bound and collapsing a lower one, and if the two sit too close the
    passes cycle, splitting an edge and collapsing the halves straight back.
    So the band is widened around the target, by an amount that depends on
    how far the target is from where the mesh already is:

    * target well below the current mean edge — the mesh is being refined,
      and a wide band lets one pass split an edge several ways without the
      collapse step immediately undoing it;
    * target comparable to it — the mesh is nearly there, so a tighter band
      holds it rather than letting edges churn either side of the target;
    * target well above it — the mesh is being coarsened, and a raised floor
      keeps collapsing going until edges genuinely reach the target.

    The comparison uses the true mean edge length of the mesh, so the band
    is a deterministic function of the geometry: the same surface always
    gives the same band, whatever order its triangles are stored in.
    """
    avrg = _mean_edge_length(mesh)
    if 0.5 * avrg > target:
        lo, hi = _BAND_FINE
    elif target <= 1.5 * avrg:
        lo, hi = _BAND_NEAR
    else:
        lo, hi = _BAND_COARSE
    return lo * target, hi * target


def _edge_table(tri: np.ndarray, n_pts: int
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Undirected edge decomposition of *tri*.

    Returns ``(uniq, inv, counts)``: the ``(E, 2)`` sorted unique edges, the
    ``(3M,)`` index of each triangle-edge into them (triangle ``i``'s edge
    ``k`` is at ``inv[k * M + i]``), and how many triangles use each edge.
    """
    e = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    lo = e.min(axis=1).astype(np.int64)
    hi = e.max(axis=1).astype(np.int64)
    # Pack each edge into one integer: np.unique on a 1-D key is much
    # faster than the structured-view path np.unique(..., axis=0) takes.
    ukey, inv, counts = np.unique(lo * n_pts + hi,
                                  return_inverse=True, return_counts=True)
    uniq = np.column_stack([ukey // n_pts, ukey % n_pts])
    return uniq, np.asarray(inv).ravel(), counts


def _edge_owners(inv: np.ndarray,
                 counts: np.ndarray,
                 n_tri: int) -> tuple[np.ndarray, np.ndarray]:
    """Triangles incident to each edge, as a CSR-style ``(owner, start)``
    pair: edge ``e`` is used by ``owner[start[e]:start[e] + counts[e]]``."""
    order = np.argsort(inv, kind="stable")
    owner = np.tile(np.arange(n_tri, dtype=np.int64), 3)[order]
    start = np.concatenate([[0], np.cumsum(counts)[:-1]])
    return owner, start


def _vertex_normals(pts: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Area-weighted unit vertex normals."""
    fn = np.cross(pts[tri[:, 1]] - pts[tri[:, 0]],
                  pts[tri[:, 2]] - pts[tri[:, 0]])
    nrm = np.zeros_like(pts)
    for k in range(3):
        for c in range(3):
            nrm[:, c] += np.bincount(tri[:, k], weights=fn[:, c],
                                     minlength=pts.shape[0])
    length = np.linalg.norm(nrm, axis=1)
    length[length == 0.0] = 1.0
    return nrm / length[:, None]


def _frozen_vertices(pts_n: int,
                     tri: np.ndarray,
                     origin: np.ndarray,
                     tags: Optional[np.ndarray],
                     preserve_labels: Optional[Sequence[int]],
                     fix_boundary: bool) -> np.ndarray:
    """Vertices that may neither move nor be removed: open-boundary
    vertices (when *fix_boundary*), label-seam vertices, and any vertex on
    a non-manifold edge.

    Derived from the current topology rather than carried along, so a
    vertex inserted on a seam edge is itself a seam vertex from the next
    step onwards, with no bookkeeping to fall out of step.
    """
    frozen = np.zeros(pts_n, dtype=bool)
    uniq, inv, counts = _edge_table(tri, pts_n)

    odd = (counts != 2) if fix_boundary else (counts >= 3)
    if odd.any():
        frozen[uniq[odd].ravel()] = True

    if tags is not None and preserve_labels != ():
        et = np.asarray(tags)[origin]
        owner, start = _edge_owners(inv, counts, tri.shape[0])
        two = counts == 2
        o0 = owner[start[two]]
        o1 = owner[start[two] + 1]
        seam = et[o0] != et[o1]
        if preserve_labels is not None:
            listed = list(preserve_labels)
            seam &= np.isin(et[o0], listed) | np.isin(et[o1], listed)
        if seam.any():
            frozen[uniq[two][seam].ravel()] = True
    return frozen


def _split_long_edges(pts: np.ndarray,
                      tri: np.ndarray,
                      origin: np.ndarray,
                      max_edge: float
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Split every edge longer than *max_edge* at its midpoint.

    All long edges of a pass go in one batch, so a triangle is cut into 2,
    3 or 4 depending on how many of its edges are marked.
    """
    n_pts = pts.shape[0]
    uniq, inv, _ = _edge_table(tri, n_pts)
    length = np.linalg.norm(pts[uniq[:, 0]] - pts[uniq[:, 1]], axis=1)
    long_e = length > max_edge
    n_split = int(long_e.sum())
    if n_split == 0:
        return pts, tri, origin, 0

    new_id = np.full(uniq.shape[0], -1, dtype=np.int64)
    new_id[long_e] = np.arange(n_split, dtype=np.int64) + n_pts
    mid = 0.5 * (pts[uniq[long_e, 0]] + pts[uniq[long_e, 1]])
    pts = np.vstack([pts, mid])

    # mid_of[:, k] = new vertex on edge k of the triangle, or -1
    mid_of = new_id[inv].reshape(3, tri.shape[0]).T
    n_cut = (mid_of >= 0).sum(axis=1)

    out_tri: list[np.ndarray] = [tri[n_cut == 0]]
    out_org: list[np.ndarray] = [origin[n_cut == 0]]

    def _rolled(sel: np.ndarray, first: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray]:
        """Relabel each selected triangle so its edge *first* becomes edge
        0, i.e. vertices ``(v0, v1, v2)`` with the edge of interest at
        ``(v0, v1)``."""
        idx = (np.arange(3)[None, :] + first[:, None]) % 3
        return (np.take_along_axis(tri[sel], idx, axis=1),
                np.take_along_axis(mid_of[sel], idx, axis=1))

    one = np.flatnonzero(n_cut == 1)
    if one.size:
        v, m = _rolled(one, np.argmax(mid_of[one] >= 0, axis=1))
        out_tri += [np.column_stack([v[:, 0], m[:, 0], v[:, 2]]),
                    np.column_stack([m[:, 0], v[:, 1], v[:, 2]])]
        out_org += [origin[one]] * 2

    two = np.flatnonzero(n_cut == 2)
    if two.size:
        # roll the *un-split* edge to position 0: cuts sit on (v1,v2) and
        # (v2,v0), so the polygon is [v0, v1, q, v2, r]
        v, m = _rolled(two, np.argmax(mid_of[two] < 0, axis=1))
        q, r = m[:, 1], m[:, 2]
        out_tri += [np.column_stack([v[:, 0], v[:, 1], q]),
                    np.column_stack([v[:, 0], q, r]),
                    np.column_stack([r, q, v[:, 2]])]
        out_org += [origin[two]] * 3

    three = np.flatnonzero(n_cut == 3)
    if three.size:
        v, m = tri[three], mid_of[three]
        p, q, r = m[:, 0], m[:, 1], m[:, 2]
        out_tri += [np.column_stack([v[:, 0], p, r]),
                    np.column_stack([p, v[:, 1], q]),
                    np.column_stack([r, q, v[:, 2]]),
                    np.column_stack([p, q, r])]
        out_org += [origin[three]] * 4

    return (pts, np.vstack(out_tri).astype(np.int64),
            np.concatenate(out_org), n_split)


def _vertex_triangles(tri: np.ndarray,
                      n_pts: int) -> tuple[np.ndarray, np.ndarray]:
    """CSR-style vertex-to-triangle map: vertex ``v`` is used by
    ``v2t[start[v]:start[v + 1]]``."""
    order = np.argsort(tri.ravel(), kind="stable")
    v2t = np.repeat(np.arange(tri.shape[0], dtype=np.int64), 3)[order]
    start = np.concatenate(
        [[0], np.cumsum(np.bincount(tri.ravel(), minlength=n_pts))])
    return v2t, start


def _collapse_short_edges(pts: np.ndarray,
                          tri: np.ndarray,
                          origin: np.ndarray,
                          frozen: np.ndarray,
                          min_edge: float,
                          max_edge: float,
                          surf_corr: float
                          ) -> tuple[np.ndarray, np.ndarray,
                                     np.ndarray, int]:
    """Collapse edges shorter than *min_edge*, shortest first.

    A collapse is vetoed when it would flatten a feature (endpoint normals
    correlating no more than *surf_corr*), break manifoldness (link
    condition), flip or degenerate an incident triangle, or create an edge
    longer than *max_edge*. Frozen vertices are never removed and never
    moved: an edge with one frozen endpoint collapses onto it, an edge with
    two is left alone.
    """
    n_pts = pts.shape[0]
    uniq, _, counts = _edge_table(tri, n_pts)
    length = np.linalg.norm(pts[uniq[:, 0]] - pts[uniq[:, 1]], axis=1)
    cand = np.flatnonzero(length < min_edge)
    if cand.size == 0:
        return pts, tri, origin, 0
    cand = cand[np.argsort(length[cand])]

    nrm = _vertex_normals(pts, tri)
    # The two vetoes that depend on nothing a previous collapse can change
    # are evaluated for the whole candidate list at once, so the sequential
    # loop below only sees edges that could actually be collapsed.
    ea, eb = uniq[cand, 0], uniq[cand, 1]
    survives = (np.sum(nrm[ea] * nrm[eb], axis=1) > surf_corr) & \
               ~(frozen[ea] & frozen[eb])
    cand = cand[survives]
    if cand.size == 0:
        return pts, tri, origin, 0

    v2t, v2t_start = _vertex_triangles(tri, n_pts)
    pts = pts.copy()
    merge = np.arange(n_pts, dtype=np.int64)
    locked = np.zeros(n_pts, dtype=bool)
    area_eps = 1e-12 * min_edge * min_edge
    n_done = 0

    for e in cand:
        a, b = int(uniq[e, 0]), int(uniq[e, 1])
        if locked[a] or locked[b]:
            continue
        if frozen[a]:
            keep, gone, target = a, b, pts[a]
        elif frozen[b]:
            keep, gone, target = b, a, pts[b]
        else:
            keep, gone, target = a, b, 0.5 * (pts[a] + pts[b])

        t_keep = v2t[v2t_start[keep]:v2t_start[keep + 1]]
        t_gone = v2t[v2t_start[gone]:v2t_start[gone + 1]]
        nb_keep = np.unique(tri[t_keep])
        nb_gone = np.unique(tri[t_gone])
        # Link condition: an edge may only be collapsed when its endpoints
        # share exactly the vertices opposite it, else the result is
        # non-manifold or a fold.
        shared = np.intersect1d(nb_keep, nb_gone).size - 2   # minus a, b
        if shared != (2 if counts[e] == 2 else 1):
            continue

        aff = tri[np.union1d(t_keep, t_gone)]
        dying = np.any(aff == keep, axis=1) & np.any(aff == gone, axis=1)
        chk = aff[~dying]
        old_n = _cross3(pts[chk[:, 1]] - pts[chk[:, 0]],
                        pts[chk[:, 2]] - pts[chk[:, 0]])
        sub = np.where(chk == gone, keep, chk)
        new_p = pts[sub]
        new_p[sub == keep] = target
        new_n = _cross3(new_p[:, 1] - new_p[:, 0],
                        new_p[:, 2] - new_p[:, 0])
        if np.any((old_n * new_n).sum(axis=1) <= 0.0):
            continue                                   # flipped a triangle
        if np.any(np.linalg.norm(new_n, axis=1) < area_eps):
            continue                                   # degenerate result
        new_len = np.linalg.norm(new_p - new_p[:, [1, 2, 0]], axis=2)
        if float(new_len.max()) > max_edge:
            continue                                   # over-long edge

        merge[gone] = keep
        pts[keep] = target
        locked[nb_keep] = True
        locked[nb_gone] = True
        n_done += 1

    if n_done == 0:
        return pts, tri, origin, 0

    tri = merge[tri]
    alive = ((tri[:, 0] != tri[:, 1]) &
             (tri[:, 1] != tri[:, 2]) &
             (tri[:, 0] != tri[:, 2]))
    tri, origin = tri[alive], origin[alive]
    pts, tri, _ = _compact_points(pts, tri)
    return pts, tri, origin, n_done


# How many nearby triangles a point is projected against. The result lands
# on the surface for any value ≥ 1; more candidates only make it the
# *closest* such point rather than a nearby one. 4 is where the repair stops
# improving: it reaches the same triangle quality as an exhaustive closest-
# point query, while 3 and below start snapping vertices to the wrong
# neighbour often enough to leave bad triangles behind.
_PROJECT_CANDIDATES = 4


def _closest_on_triangle(p: np.ndarray, a: np.ndarray, b: np.ndarray,
                         c: np.ndarray) -> np.ndarray:
    """Closest point to *p* on triangle ``(a, b, c)``, vectorised.

    Barycentric region test: the plane of the triangle is divided into the
    face, three edges and three corners, and the answer comes from
    whichever region *p* projects into.
    """
    ab, ac = b - a, c - a
    ap = p - a
    d1 = np.sum(ab * ap, axis=-1)
    d2 = np.sum(ac * ap, axis=-1)
    bp = p - b
    d3 = np.sum(ab * bp, axis=-1)
    d4 = np.sum(ac * bp, axis=-1)
    cp = p - c
    d5 = np.sum(ab * cp, axis=-1)
    d6 = np.sum(ac * cp, axis=-1)

    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    denom = va + vb + vc
    safe = np.where(denom == 0.0, 1.0, denom)
    # interior of the face
    out = a + ab * (vb / safe)[..., None] + ac * (vc / safe)[..., None]

    def _edge(base, along, num, den):
        t = np.where(den == 0.0, 0.0, num / np.where(den == 0.0, 1.0, den))
        return base + along * t[..., None]

    # the three edges, then the three corners: later assignments win, so
    # they run in increasing order of priority
    out = np.where(((vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0))[..., None],
                   _edge(a, ab, d1, d1 - d3), out)
    out = np.where(((vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0))[..., None],
                   _edge(a, ac, d2, d2 - d6), out)
    out = np.where(((va <= 0.0) & ((d4 - d3) >= 0.0)
                    & ((d5 - d6) >= 0.0))[..., None],
                   _edge(b, c - b, d4 - d3, (d4 - d3) + (d5 - d6)), out)
    out = np.where(((d1 <= 0.0) & (d2 <= 0.0))[..., None], a, out)
    out = np.where(((d3 >= 0.0) & (d4 <= d3))[..., None], b, out)
    out = np.where(((d6 >= 0.0) & (d5 <= d6))[..., None], c, out)
    return out


def _surface_projector(mesh: pv.PolyData
                       ) -> Callable[[np.ndarray], np.ndarray]:
    """Return a function snapping points onto *mesh*'s surface.

    ``PolyData.find_closest_cell`` answers one point per call, which on a
    dense mesh over many passes is millions of round trips into VTK. This
    builds one kd-tree over triangle centroids up front and answers a whole
    array at a time.
    """
    ref_pts = np.asarray(mesh.points, dtype=float)
    ref_tri = _faces_to_tri(mesh)
    tree = cKDTree(ref_pts[ref_tri].mean(axis=1))
    k = min(_PROJECT_CANDIDATES, ref_tri.shape[0])

    def project(query: np.ndarray) -> np.ndarray:
        _, cand = tree.query(query, k=k)
        cand = cand.reshape(query.shape[0], -1)
        corners = ref_pts[ref_tri[cand]]          # (N, k, 3, 3)
        near = _closest_on_triangle(query[:, None, :],
                                    corners[..., 0, :],
                                    corners[..., 1, :],
                                    corners[..., 2, :])
        best = np.argmin(np.sum((near - query[:, None, :]) ** 2, axis=-1),
                         axis=1)
        return near[np.arange(query.shape[0]), best]

    return project


def _cross3(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Cross product of stacked 3-vectors.

    ``np.cross`` spends far more time in ``moveaxis`` / axis normalisation
    than in the arithmetic, which matters when it is called once per
    candidate edge rather than once per array.
    """
    return np.stack([u[..., 1] * v[..., 2] - u[..., 2] * v[..., 1],
                     u[..., 2] * v[..., 0] - u[..., 0] * v[..., 2],
                     u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]],
                    axis=-1)


def _quality_of(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                twice_area: np.ndarray) -> np.ndarray:
    """``_triangle_quality`` for triangles whose ``|edge cross product|``
    (twice the area) is already known. Vectorised over the first axis."""
    sq = (np.sum((y - x) ** 2, axis=-1) + np.sum((z - y) ** 2, axis=-1)
          + np.sum((x - z) ** 2, axis=-1))
    out = np.zeros_like(sq)
    ok = sq > 0.0
    out[ok] = 2.0 * np.sqrt(3.0) * twice_area[ok] / sq[ok]
    return out


def _flip_edges(pts: np.ndarray,
                tri: np.ndarray,
                frozen: np.ndarray,
                surf_corr: float,
                max_edge: float) -> tuple[np.ndarray, int]:
    """Flip interior edges that bring vertex valences closer to the ideal
    (6 inside, 4 on a boundary). Edges with two frozen endpoints, edges
    across a feature (incident normals correlating no more than
    *surf_corr*), flips that would fold or flatten the surface and flips
    that would push the new edge past *max_edge* are left alone."""
    n_pts = pts.shape[0]
    uniq, inv, counts = _edge_table(tri, n_pts)
    interior = np.flatnonzero(counts == 2)
    if interior.size == 0:
        return tri, 0
    owner, start = _edge_owners(inv, counts, tri.shape[0])
    o0 = owner[start[interior]]
    o1 = owner[start[interior] + 1]
    a, b = uniq[interior, 0], uniq[interior, 1]
    # the third vertex of a triangle holding a and b
    c = tri[o0].sum(axis=1) - a - b
    d = tri[o1].sum(axis=1) - a - b

    valence = np.bincount(uniq.ravel(), minlength=n_pts)
    ideal = np.where(frozen, 4, 6)
    dev = np.abs(valence - ideal)
    before = dev[a] + dev[b] + dev[c] + dev[d]
    after = (np.abs(valence[a] - 1 - ideal[a]) +
             np.abs(valence[b] - 1 - ideal[b]) +
             np.abs(valence[c] + 1 - ideal[c]) +
             np.abs(valence[d] + 1 - ideal[d]))
    want = np.flatnonzero((after < before) & ~(frozen[a] & frozen[b]))
    if want.size == 0:
        return tri, 0

    # Every geometric veto is evaluated for all candidates at once. Done one
    # candidate at a time these are 3-vector operations whose numpy call
    # overhead dwarfs the arithmetic — it was 90% of a remesh.
    va, vb, vc, vd = a[want], b[want], c[want], d[want]
    p0, p1, p2, p3 = pts[va], pts[vb], pts[vc], pts[vd]
    n0 = _cross3(p1 - p0, p2 - p0)                     # (a, b, c) as is
    n1 = _cross3(p0 - p1, p3 - p1)                     # (b, a, d) as is
    m0 = _cross3(p3 - p0, p2 - p0)                     # (a, d, c) flipped
    m1 = _cross3(p1 - p3, p2 - p3)                     # (d, b, c) flipped
    s0, s1 = np.linalg.norm(n0, axis=1), np.linalg.norm(n1, axis=1)
    t0_, t1_ = np.linalg.norm(m0, axis=1), np.linalg.norm(m1, axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.sum(n0 * n1, axis=1) / (s0 * s1)
    old_q = np.minimum(_quality_of(p0, p1, p2, s0),
                       _quality_of(p1, p0, p3, s1))
    new_q = np.minimum(_quality_of(p0, p3, p2, t0_),
                       _quality_of(p3, p1, p2, t1_))
    ok = ((np.linalg.norm(p3 - p2, axis=1) <= max_edge)  # new edge in band
          & (s0 > 0.0) & (s1 > 0.0)
          & (corr > surf_corr)                           # not across a feature
          & (np.sum(m0 * n0, axis=1) > 0.0)              # would not fold
          & (np.sum(m1 * n1, axis=1) > 0.0)
          & (np.minimum(t0_, t1_) >= 1e-6 * np.maximum(s0, s1))
          # Valence alone would happily trade a well-shaped pair for a much
          # thinner one. Requiring no degradation at all blocks the whole
          # cascade — a flip often pays off only once its neighbours follow —
          # so allow a bounded local loss.
          & (new_q >= _FLIP_QUALITY_TOL * old_q))

    # Flipping onto an edge the mesh already has duplicates a triangle.
    # _edge_table hands back its keys sorted, so the test is a searchsorted
    # rather than a Python set of every edge in the mesh.
    ukey = uniq[:, 0] * n_pts + uniq[:, 1]
    new_key = np.minimum(vc, vd) * n_pts + np.maximum(vc, vd)
    slot = np.searchsorted(ukey, new_key).clip(0, ukey.size - 1)
    ok &= ukey[slot] != new_key

    added: set[int] = set()          # only the keys this batch creates
    tri = tri.copy()
    used = np.zeros(tri.shape[0], dtype=bool)
    n_flip = 0
    # What is left is inherently sequential: whether a flip is allowed
    # depends on the ones already applied in this batch.
    for k in np.flatnonzero(ok):
        i = want[k]
        t0, t1 = int(o0[i]), int(o1[i])
        if used[t0] or used[t1]:
            continue
        x, y = int(vc[k]), int(vd[k])
        key = int(new_key[k])
        if key in added:
            continue
        tri[t0] = (int(va[k]), y, x)
        tri[t1] = (y, int(vb[k]), x)
        used[t0] = used[t1] = True
        added.add(key)
        n_flip += 1
    return tri, n_flip


def _relax_tangential(pts: np.ndarray,
                      tri: np.ndarray,
                      frozen: np.ndarray,
                      project: Callable[[np.ndarray], np.ndarray],
                      iterations: int = 5,
                      factor: float = 0.5) -> np.ndarray:
    """Even out vertex spacing without moving the surface.

    Each free vertex drifts towards the centroid of its neighbours, but
    only by the component of that step lying in its tangent plane, and the
    result is put back on the surface by *project*. Tangential motion is
    surface-preserving to first order and the projection removes the rest,
    which is what keeps this from shrinking a curved wall the way a plain
    Laplacian pass would.
    """
    n_pts = pts.shape[0]
    uniq, _, _ = _edge_table(tri, n_pts)
    lo, hi = uniq[:, 0], uniq[:, 1]
    deg = (np.bincount(lo, minlength=n_pts) +
           np.bincount(hi, minlength=n_pts)).astype(float)
    deg[deg == 0.0] = 1.0
    free = ~frozen

    pts = pts.copy()
    for _ in range(iterations):
        acc = np.empty_like(pts)
        for c in range(3):
            acc[:, c] = (np.bincount(lo, weights=pts[hi, c],
                                     minlength=n_pts) +
                         np.bincount(hi, weights=pts[lo, c],
                                     minlength=n_pts))
        step = acc / deg[:, None] - pts
        nrm = _vertex_normals(pts, tri)
        step -= (step * nrm).sum(axis=1)[:, None] * nrm
        pts[free] += factor * step[free]

    pts[free] = project(pts)[free]
    return pts


def remesh(mesh: pv.PolyData,
           target_edge: Optional[float] = None,
           min_edge: Optional[float] = None,
           max_edge: Optional[float] = None,
           surf_corr: float = 0.95,
           fix_boundary: bool = True,
           relax: bool = True,
           n_passes: int = 10,
           preserve_labels: Optional[Sequence[int]] = None,
           on_progress: Optional[Callable[[int, int], None]] = None
           ) -> pv.PolyData:
    """Resample a triangle surface into an edge-length band.

    Unlike :func:`refine`, which only ever splits, this both splits edges
    longer than ``max_edge`` and collapses edges shorter than ``min_edge``,
    so the point count can fall as well as rise and vertices move.

    Give either *target_edge* or an explicit *min_edge* / *max_edge* pair,
    never both — see :func:`_band_from_target` for how a target becomes a
    band. Passing both raises ``ValueError`` rather than picking one and
    leaving the caller to wonder which.

    Each pass runs split → collapse → valence-improving edge flips →
    tangential relaxation, stopping early once no edge is out of band.

    ``surf_corr`` is the collapse veto: an edge may only be collapsed when
    the surface normals at its endpoints correlate by more than this, so
    collapsing cannot flatten a curved feature. Note it is a dot product —
    a value of 1.0 or above disables collapsing entirely, since no pair of
    normals can correlate by more than 1.

    ``fix_boundary`` freezes open-boundary vertices, so a clipped mesh
    keeps its PV ostia and mitral-valve rims exactly.

    ``preserve_labels`` controls label seams: ``None`` (the default) freezes
    every ``elemTag`` boundary, a sequence of labels freezes only the
    boundaries touching those labels, and ``()`` protects nothing. A frozen
    seam
    vertex is never moved and never collapsed away, so the seam curve of
    the input survives verbatim in the output; seam edges can still be
    split, which refines the seam without redrawing it.

    ``on_progress(i, n_passes)`` fires after every pass. Passes stop early
    once nothing is out of band, so it need not reach ``n_passes``.
    """
    ratio = _BAND_FINE[1] / _BAND_FINE[0]
    if target_edge is not None:
        if min_edge is not None or max_edge is not None:
            raise ValueError(
                "give target_edge or (min_edge, max_edge), not both")
        min_edge, max_edge = _band_from_target(mesh, target_edge)
    elif min_edge is None and max_edge is None:
        raise ValueError("give target_edge or (min_edge, max_edge)")
    elif min_edge is None:
        min_edge = max_edge / ratio
    elif max_edge is None:
        max_edge = min_edge * ratio
    if min_edge <= 0 or max_edge <= min_edge:
        raise ValueError("need 0 < min_edge < max_edge")
    if n_passes < 1:
        raise ValueError("n_passes must be positive")

    tri = _faces_to_tri(mesh)
    pts = np.asarray(mesh.points, dtype=float).copy()
    if tri.shape[0] == 0:
        return mesh.copy()
    # One kd-tree over the input surface, reused by every pass.
    project = _surface_projector(mesh)
    origin = np.arange(tri.shape[0], dtype=np.int64)
    tags = (np.asarray(mesh.cell_data["elemTag"])
            if "elemTag" in mesh.cell_data else None)

    for i_pass in range(n_passes):
        pts, tri, origin, n_split = _split_long_edges(
            pts, tri, origin, max_edge)
        # A collapse locks the 1-ring of both its endpoints for the rest of
        # the batch, so one batch only ever thins a dense field of short
        # edges. Repeat until it stalls.
        n_coll = 0
        for _ in range(_COLLAPSE_BATCHES):
            frozen = _frozen_vertices(pts.shape[0], tri, origin, tags,
                                      preserve_labels, fix_boundary)
            pts, tri, origin, done = _collapse_short_edges(
                pts, tri, origin, frozen, min_edge, max_edge, surf_corr)
            n_coll += done
            if done == 0:
                break
        # Each triangle takes part in at most one flip per batch, so a batch
        # only ever improves valence locally. Repeat until it settles, else
        # an anisotropic input keeps its connectivity however well its edge
        # lengths land in the band.
        frozen = _frozen_vertices(pts.shape[0], tri, origin, tags,
                                  preserve_labels, fix_boundary)
        n_flip = 0
        for _ in range(_FLIP_BATCHES):
            tri, done = _flip_edges(pts, tri, frozen, surf_corr, max_edge)
            n_flip += done
            if done == 0:
                break
        if relax:
            pts = _relax_tangential(pts, tri, frozen, project)
        if on_progress is not None:
            on_progress(i_pass + 1, n_passes)
        if n_split == 0 and n_coll == 0 and n_flip == 0:
            break

    out = pv.PolyData(pts, _tri_to_faces(tri))
    # Point data has to be looked up: relaxation moved the vertices and
    # collapses removed some. Cell data does not — every triangle still
    # knows the input triangle it descends from, so labels transfer exactly
    # rather than by proximity.
    if mesh.n_points and mesh.point_data:
        _, pid = cKDTree(np.asarray(mesh.points)).query(pts, k=1)
        for name in list(mesh.point_data.keys()):
            out.point_data[name] = np.asarray(mesh.point_data[name])[pid]
    for name in list(mesh.cell_data.keys()):
        if name == "render_idx":
            continue
        out.cell_data[name] = np.asarray(mesh.cell_data[name])[origin]
    return out


# =====================================================================
# CLEAN
# =====================================================================
def _triangle_quality(mesh: pv.PolyData) -> np.ndarray:
    """Shape quality in [0, 1]; 1 is equilateral, 0 is degenerate.

    ``q = 4·√3·A / (l₁² + l₂² + l₃²)``: area over summed squared side
    lengths, normalised so an equilateral triangle scores exactly 1. Both
    parts scale with the square of the mesh's units, so the score depends
    on shape alone and one threshold works at any resolution.

    It is more demanding than it looks — a right isoceles triangle scores
    0.87 and a 30-60-90 triangle 0.75 — so a threshold of 0.8 flags
    anything worse than roughly a 35-55-90 triangle.
    """
    return _triangle_quality_from(np.asarray(mesh.points, dtype=float),
                                  _faces_to_tri(mesh))


def clean(mesh: pv.PolyData,
          preserve_labels: Optional[Iterable[int]] = None,
          quality_threshold: float = 0.2,
          quality_relaxation: float = 0.1,
          smooth_iterations: int = 20,
          merge_tol: float = 0.0,
          on_progress: Optional[Callable[[int, int], None]] = None
          ) -> pv.PolyData:
    """Apply the full cleaning pipeline.

    * merge duplicate points, drop unused / non-connected points
    * remove non-manifold and degenerate cells
    * ensure consistent outward normals
    * smooth low-quality triangles (``quality < quality_threshold``) while
      freezing vertices belonging to cells whose ``elemTag`` is listed in
      ``preserve_labels``

    ``on_progress(i, n)`` is forwarded to :func:`improve_quality`, the one
    step long enough to be worth reporting.

    ``merge_tol`` is the absolute welding distance of the deduplication
    step, in mesh units (mm for Carto exports). The default ``0.0`` is
    geometry-safe: it merges only exactly coincident points and moves no
    vertex. A positive value welds near-duplicates — useful on exports
    whose seams are stitched to within a fraction of a millimetre rather
    than exactly.

    When ``preserve_labels`` is non-empty the "provided surfaces" are
    treated as inviolable:

    * their vertices are never moved (frozen during smoothing, never
      merged by the deduplication step — the default ``merge_tol=0``
      only touches coincident points);
    * their cells are never dropped by the non-manifold or
      connectivity passes;
    * a final restoration step re-appends any protected cell that still
      went missing and removes duplicate triangles.

    Raising ``merge_tol`` above ``0`` weakens the first of those
    guarantees: near-coincident protected vertices become weldable, so
    protected geometry can shift by up to ``merge_tol``. The restoration
    step still re-appends protected cells at their original coordinates.

    Bookkeeping arrays introduced by the intermediate filters
    (``RegionId``, ``vtkOriginalPointIds``, ``vtkOriginalCellIds``) are
    stripped from the result; arrays carried by ``mesh`` itself are kept
    with the input's values.
    """
    src = mesh.copy()
    preserve_labels = tuple(preserve_labels or ())

    # Snapshot the caller's array names before any filter runs: anything
    # else present at the end was introduced on the way and is stripped.
    input_point_arrays = set(src.point_data.keys())
    input_cell_arrays = set(src.cell_data.keys())

    # Snapshot the protected cells verbatim (original coordinates + tags);
    # this is what we use to guarantee surface preservation at the end.
    protected_snapshot: Optional[pv.PolyData] = None
    if preserve_labels and "elemTag" in src.cell_data:
        tags_src = np.asarray(src.cell_data["elemTag"])
        pres_idx = np.where(np.isin(tags_src, list(preserve_labels)))[0]
        if pres_idx.size:
            protected_snapshot = _extract_cells(src, pres_idx)

    # 1. merge duplicates + drop unused points (geometry-safe at tol=0).
    #    merge_tol is absolute (pyvista's clean defaults to absolute=True).
    cleaned = src.clean(
        point_merging=True,
        merge_tol=merge_tol,
        lines_to_points=False,
        polys_to_lines=False,
        strips_to_polys=False,
        inplace=False,
    )

    # 2. connectivity: with protection, keep every component containing at
    #    least one protected cell AND the largest; without protection, keep
    #    only the largest.
    cleaned = _keep_main_components(cleaned, preserve_labels)

    # 3. triangle filter - kills degenerate/strip cells
    cleaned = cleaned.triangulate()

    # Cell_data / point_data carried by vtk filters are not guaranteed to
    # map 1:1 to the simplified topology — refresh now so the protection
    # mask used by non-manifold removal is correct.
    _transfer_arrays(src, cleaned)

    # 4. remove non-manifold edges — protected cells survive.
    cleaned = _remove_non_manifold(cleaned, preserve_labels)

    # 5. drop zero-area triangles
    cleaned = _drop_degenerate(cleaned, preserve_labels)

    # Bring arrays across from the source mesh BEFORE normal / smoothing
    # operations that may further perturb topology.
    _transfer_arrays(src, cleaned)

    # 6. consistent outward normals (SplittingOff, no point/cell normals
    #    arrays: does not move vertices).
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(cleaned)
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.SplittingOff()
    normals.ComputePointNormalsOff()
    normals.ComputeCellNormalsOff()
    normals.Update()
    cleaned = pv.wrap(normals.GetOutput())
    _transfer_arrays(src, cleaned)

    # 7. quality repair with preserved surfaces
    if smooth_iterations > 0 and quality_threshold > 0.0 and quality_relaxation > 0.0:
        cleaned = improve_quality(
            cleaned,
            quality_threshold=quality_threshold,
            iterations=smooth_iterations,
            step=quality_relaxation,
            preserve_labels=preserve_labels,
            on_progress=on_progress,
        )

    # 8. final restoration — guarantees surface preservation even if any
    #    upstream VTK pass silently dropped a protected triangle.
    if protected_snapshot is not None:
        cleaned = _restore_protected(cleaned, protected_snapshot)

    # 9. drop bookkeeping arrays the intermediate filters attached. Arrays
    #    the input carried survive with the input's values: the
    #    _transfer_arrays passes above re-seeded them from ``src`` after
    #    connectivity() had overwritten any same-named array.
    _strip_new_arrays(cleaned, input_point_arrays, input_cell_arrays)

    return cleaned


# ---------------------------------------------------------------------
# protection helpers
# ---------------------------------------------------------------------
def _extract_cells(mesh: pv.PolyData, idx: np.ndarray) -> pv.PolyData:
    """Return a PolyData holding only ``mesh`` cells at ``idx`` (triangles,
    with elemTag and other cell arrays copied)."""
    tri = _faces_to_tri(mesh)[idx]
    out = pv.PolyData(np.asarray(mesh.points), _tri_to_faces(tri))
    for name in list(mesh.cell_data.keys()):
        if name == "render_idx":
            continue
        out.cell_data[name] = np.asarray(mesh.cell_data[name])[idx]
    for name in list(mesh.point_data.keys()):
        out.point_data[name] = np.asarray(mesh.point_data[name])
    # Drop points unused by the extracted cells so the snapshot doesn't
    # carry the whole-mesh vertex cloud around.
    return out.clean(point_merging=False, inplace=False)


def _keep_main_components(mesh: pv.PolyData,
                          preserve_labels: Sequence[int]) -> pv.PolyData:
    """Keep the largest connected component and any component containing
    a protected cell. If no protected labels are given, behaves like
    ``connectivity(largest=True)``."""
    try:
        labelled = mesh.connectivity()
    except Exception:
        return mesh
    try:
        region = np.asarray(labelled.cell_data["RegionId"])
    except KeyError:
        return mesh.extract_surface(algorithm='dataset_surface') if hasattr(mesh, "extract_surface") else mesh

    keep = set()
    # largest component
    counts = np.bincount(region)
    keep.add(int(np.argmax(counts)))
    # components touching a protected cell
    if preserve_labels and "elemTag" in labelled.cell_data:
        tags = np.asarray(labelled.cell_data["elemTag"])
        pres = np.isin(tags, list(preserve_labels))
        keep.update(int(r) for r in np.unique(region[pres]))

    mask = np.isin(region, list(keep))
    if mask.all():
        out = labelled
    else:
        idx = np.where(mask)[0]
        out = _extract_cells(labelled, idx)
    try:
        return out.extract_surface(algorithm='dataset_surface')
    except Exception:
        return out


def _restore_protected(cleaned: pv.PolyData,
                       protected: pv.PolyData) -> pv.PolyData:
    """Append ``protected`` back onto ``cleaned``, fuse coincident points,
    and drop duplicate triangles. Guarantees every cell in ``protected``
    is present in the result with its original vertex coordinates."""
    combined = cleaned.merge(protected, merge_points=True)
    combined = combined.clean(point_merging=True, merge_tol=0.0,
                              lines_to_points=False, polys_to_lines=False,
                              strips_to_polys=False, inplace=False)
    combined = combined.extract_surface(algorithm='dataset_surface')
    combined = combined.triangulate()

    tri = _faces_to_tri(combined)
    key = np.sort(tri, axis=1)
    _, first = np.unique(key, axis=0, return_index=True)
    first.sort()
    kept_tri = tri[first]

    out = pv.PolyData(np.asarray(combined.points), _tri_to_faces(kept_tri))
    for name in list(combined.cell_data.keys()):
        if name == "render_idx":
            continue
        arr = np.asarray(combined.cell_data[name])
        out.cell_data[name] = arr[first]
    for name in list(combined.point_data.keys()):
        out.point_data[name] = np.asarray(combined.point_data[name])
    return out


# ---------------------------------------------------------------------
# geometry clean-up steps (protection-aware)
# ---------------------------------------------------------------------
def _remove_non_manifold(mesh: pv.PolyData,
                         preserve_labels: Sequence[int] = ()) -> pv.PolyData:
    tri = _faces_to_tri(mesh)
    e = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    e = np.sort(e, axis=1)
    _, inv, counts = np.unique(e, axis=0, return_inverse=True, return_counts=True)
    cell_edges = inv.reshape(3, -1).T  # (n_cells, 3)
    bad = np.any(counts[cell_edges] > 2, axis=1)
    # Never drop protected cells — even if they participate in a non-
    # manifold edge: preserving the surface wins over manifold-ness.
    if preserve_labels and "elemTag" in mesh.cell_data:
        tags = np.asarray(mesh.cell_data["elemTag"])
        bad &= ~np.isin(tags, list(preserve_labels))
    if not bad.any():
        return mesh
    keep = np.where(~bad)[0]
    out = pv.PolyData(np.asarray(mesh.points), _tri_to_faces(tri[keep]))
    for name in list(mesh.cell_data.keys()):
        if name == "render_idx":
            continue
        out.cell_data[name] = np.asarray(mesh.cell_data[name])[keep]
    for name in list(mesh.point_data.keys()):
        out.point_data[name] = np.asarray(mesh.point_data[name])
    return out


def _drop_degenerate(mesh: pv.PolyData,
                     preserve_labels: Sequence[int] = ()) -> pv.PolyData:
    tri = _faces_to_tri(mesh)
    p = np.asarray(mesh.points)
    a = p[tri[:, 1]] - p[tri[:, 0]]
    b = p[tri[:, 2]] - p[tri[:, 0]]
    area = 0.5 * np.linalg.norm(np.cross(a, b), axis=1)
    keep = area > 1e-14
    # Never drop a protected cell (even if flagged degenerate) — the
    # caller asked us to preserve the surface exactly.
    if preserve_labels and "elemTag" in mesh.cell_data:
        tags = np.asarray(mesh.cell_data["elemTag"])
        keep = keep | np.isin(tags, list(preserve_labels))
    if keep.all():
        return mesh
    out = pv.PolyData(np.asarray(mesh.points), _tri_to_faces(tri[keep]))
    for name in list(mesh.cell_data.keys()):
        if name == "render_idx":
            continue
        out.cell_data[name] = np.asarray(mesh.cell_data[name])[keep]
    for name in list(mesh.point_data.keys()):
        out.point_data[name] = np.asarray(mesh.point_data[name])
    return out


# Exponent of the per-triangle weight (1 - q)**_BADNESS_POW in the
# objective the repair descends. High enough that a well-shaped triangle
# contributes almost nothing, so the step is driven by the bad ones.
_BADNESS_POW = 4


def _quality_gradient(pts: np.ndarray,
                      tri: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-triangle quality and its gradient with respect to each corner.

    Returns ``(q, grad)`` with ``grad`` of shape ``(M, 3, 3)``:
    ``grad[t, k]`` is ``∂q_t / ∂p`` for the ``k``-th vertex of triangle
    ``t``. Analytic, so no finite differences: with ``q = 4√3·A/S`` and
    ``S = Σl²``, ``∂A/∂a = ½·n̂ × (c − b)`` and ``∂S/∂a = 2(a−b) + 2(a−c)``.
    """
    p = [pts[tri[:, k]] for k in range(3)]
    nrm = np.cross(p[1] - p[0], p[2] - p[0])
    two_area = np.linalg.norm(nrm, axis=1)
    area = 0.5 * two_area
    unit = nrm / np.where(two_area == 0.0, 1.0, two_area)[:, None]
    sq = (np.sum((p[1] - p[0]) ** 2, axis=1) +
          np.sum((p[2] - p[1]) ** 2, axis=1) +
          np.sum((p[0] - p[2]) ** 2, axis=1))
    ok = sq > 0.0
    inv_sq = np.zeros_like(sq)
    inv_sq[ok] = 1.0 / sq[ok]

    q = 4.0 * np.sqrt(3.0) * area * inv_sq
    grad = np.zeros((tri.shape[0], 3, 3))
    for k in range(3):
        a, b, c = p[k], p[(k + 1) % 3], p[(k + 2) % 3]
        d_area = 0.5 * np.cross(unit, c - b)
        d_sq = 2.0 * (a - b) + 2.0 * (a - c)
        grad[:, k, :] = 4.0 * np.sqrt(3.0) * (
            d_area * inv_sq[:, None]
            - (area * inv_sq * inv_sq)[:, None] * d_sq)
    return q, grad


def improve_quality(mesh: pv.PolyData,
                    quality_threshold: float = 0.8,
                    iterations: int = 20,
                    step: float = 0.05,
                    max_shift: float = 0.5,
                    project: bool = True,
                    preserve_labels: Optional[Iterable[int]] = None,
                    fix_boundary: bool = True,
                    on_progress: Optional[Callable[[int, int], None]] = None
                    ) -> pv.PolyData:
    """Repair badly-shaped triangles by relocating vertices.

    The last step of :func:`clean`, and usable on its own when the repair
    is wanted without the topology passes — after a resample, say, where
    those would renumber points for nothing.

    Topology is untouched and the point count **and order**
    are preserved, so arrays are carried across by index and a caller can
    keep its own vertex references — the same contract as :func:`smooth`.

    Every vertex of a triangle whose quality is below *quality_threshold*
    steps along the gradient of a neighbourhood badness functional,
    ``Σ (1 − q)`` over its one-ring raised to a power, so the step is aimed
    at the shape rather than at the neighbours' centroid the way a
    Laplacian nudge is. Three things keep the repair from buying triangle
    shape with geometry, which is the failure mode of a bare gradient
    method — it can always make a triangle equilateral by pushing a vertex
    off the wall, and on a noisy surface that means crumpling it:

    * steps are projected into the vertex tangent plane, so to first order
      a vertex slides along the surface instead of off it, and with
      *project* the residual drift is removed by snapping back onto the
      input surface — vertices then slide along the wall without ever
      leaving it, however many iterations run;
    * no vertex may end up further than ``max_shift`` local edge lengths
      from where it started;
    * a step is only kept if it lowers the objective, and any triangle it
      would invert is rolled back; the best iterate seen is what gets
      returned, rather than the last one or the one before the first
      regression.

    ``quality_threshold`` follows :func:`_triangle_quality` — 1 is
    equilateral, so triangles **below** it are repaired.

    Frozen (never moved): vertices of cells whose ``elemTag`` is listed in
    *preserve_labels*, every ``elemTag`` seam, non-manifold vertices, and —
    when *fix_boundary* — open-boundary vertices, which on a clipped mesh
    are the PV ostia and mitral-valve rims.

    ``on_progress(i, iterations)`` fires after every iteration. The loop
    stops early, so it need not reach ``iterations``.
    """
    tri = _faces_to_tri(mesh)
    pts0 = np.asarray(mesh.points, dtype=float)
    out = pv.PolyData(pts0.copy(), _tri_to_faces(tri))
    for name in list(mesh.point_data.keys()):
        out.point_data[name] = np.asarray(mesh.point_data[name])
    for name in list(mesh.cell_data.keys()):
        if name != "render_idx":
            out.cell_data[name] = np.asarray(mesh.cell_data[name])
    if tri.shape[0] == 0 or iterations < 1 or step <= 0.0:
        return out

    n_pts = pts0.shape[0]
    tags = (np.asarray(mesh.cell_data["elemTag"])
            if "elemTag" in mesh.cell_data else None)
    frozen = _frozen_vertices(n_pts, tri, np.arange(tri.shape[0]),
                              tags, None, fix_boundary)
    if preserve_labels is not None and tags is not None:
        listed = list(preserve_labels)
        if listed:
            frozen[np.unique(tri[np.isin(tags, listed)].ravel())] = True

    # Per-vertex length scale: the mean length of its incident edges. Both
    # the step and the displacement cap are relative to it, so one setting
    # works across a graded mesh.
    uniq, _, _ = _edge_table(tri, n_pts)
    elen = np.linalg.norm(pts0[uniq[:, 0]] - pts0[uniq[:, 1]], axis=1)
    tot = (np.bincount(uniq[:, 0], weights=elen, minlength=n_pts) +
           np.bincount(uniq[:, 1], weights=elen, minlength=n_pts))
    deg = (np.bincount(uniq[:, 0], minlength=n_pts) +
           np.bincount(uniq[:, 1], minlength=n_pts)).astype(float)
    scale = tot / np.where(deg == 0.0, 1.0, deg)
    cap = max_shift * scale

    pts = pts0.copy()
    snap = _surface_projector(mesh) if project else None
    sign0 = _cross3(pts0[tri[:, 1]] - pts0[tri[:, 0]],
                    pts0[tri[:, 2]] - pts0[tri[:, 0]])

    def _objective(p: np.ndarray) -> tuple[np.ndarray, float, int]:
        q = _triangle_quality_from(p, tri)
        bad = q < quality_threshold
        return q, float(np.sum((1.0 - q) ** _BADNESS_POW)), int(bad.sum())

    _, best_obj, best_bad = _objective(pts)
    best = pts.copy()

    for i_iter in range(iterations):
        if on_progress is not None:
            on_progress(i_iter, iterations)
        q, grad = _quality_gradient(pts, tri)
        bad = q < quality_threshold
        if not bad.any():
            break
        active = np.zeros(n_pts, dtype=bool)
        active[np.unique(tri[bad].ravel())] = True
        active &= ~frozen
        if not active.any():
            break

        # dF/dp of F = Σ (1 - q)**pow, scattered from corners to vertices
        wgt = -_BADNESS_POW * (1.0 - q) ** (_BADNESS_POW - 1)
        force = np.zeros_like(pts)
        for k in range(3):
            contrib = wgt[:, None] * grad[:, k, :]
            for c in range(3):
                force[:, c] += np.bincount(tri[:, k], weights=contrib[:, c],
                                           minlength=n_pts)
        # descend, tangentially: sliding along the surface preserves it to
        # first order, moving along the normal is what deforms the wall.
        nrm = _vertex_normals(pts, tri)
        force -= (force * nrm).sum(axis=1)[:, None] * nrm
        mag = np.linalg.norm(force, axis=1)
        direction = np.zeros_like(force)
        good = mag > 0.0
        direction[good] = -force[good] / mag[good][:, None]
        direction[~active] = 0.0

        improved = False
        walk = step                             # fresh line search each pass
        for _ in range(3):                      # backtracking
            cand = pts + direction * (walk * scale)[:, None]
            if project:
                moving = active & good
                cand[moving] = snap(cand)[moving]
            off = cand - pts0
            dist = np.linalg.norm(off, axis=1)
            far = dist > cap
            if far.any():                       # clamp to the shift budget
                cand[far] = pts0[far] + off[far] * (cap[far] / dist[far])[:, None]
            flipped = np.sum(np.cross(cand[tri[:, 1]] - cand[tri[:, 0]],
                                      cand[tri[:, 2]] - cand[tri[:, 0]])
                             * sign0, axis=1) <= 0.0
            if flipped.any():                   # roll back only those
                cand[np.unique(tri[flipped].ravel())] = \
                    pts[np.unique(tri[flipped].ravel())]
            _, obj, n_bad = _objective(cand)
            if obj < best_obj or n_bad < best_bad:
                pts = cand
                improved = True
                if n_bad < best_bad or (n_bad == best_bad and obj < best_obj):
                    best, best_obj, best_bad = cand.copy(), obj, n_bad
                break
            walk *= 0.5
        if not improved:
            break

    if on_progress is not None:
        on_progress(iterations, iterations)
    out.points = best
    return out


def _triangle_quality_from(pts: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """``_triangle_quality`` on raw arrays."""
    a = np.linalg.norm(pts[tri[:, 1]] - pts[tri[:, 0]], axis=1)
    b = np.linalg.norm(pts[tri[:, 2]] - pts[tri[:, 1]], axis=1)
    c = np.linalg.norm(pts[tri[:, 0]] - pts[tri[:, 2]], axis=1)
    s = 0.5 * (a + b + c)
    area = np.sqrt(np.clip(s * (s - a) * (s - b) * (s - c), 0.0, None))
    sq = a * a + b * b + c * c
    q = np.zeros_like(area)
    m = sq > 0
    q[m] = 4.0 * np.sqrt(3.0) * area[m] / sq[m]
    return q


# =====================================================================
# orchestration
# =====================================================================
@dataclass
class PostprocessOptions:
    do_decimate: bool = False
    do_refine: bool = False
    do_clean: bool = False
    do_fill_holes: bool = False
    do_smooth: bool = False

    # decimate
    decimate_target_points: int = 5000
    decimate_iters: int = 200          # outer-loop iterations
    # Shared by decimate's internal pass and the fill_holes step: both mean
    # "close loops with radius ≤ this" on the same mesh. 0 = off.
    # 4.0 closes the acquisition gaps a Carto export carries while staying
    # well under the PV-ostia / mitral-valve radius, which must stay open.
    max_hole_size: float = 4.0

    # refine. Defaults are tuned for a clipped atrial wall in mm.
    refine_mode: str = REFINE_RESAMPLE   # or REFINE_ADAPTIVE
    refine_edge_len: float = 0.3   # 0 -> use median edge length
    # resample mode only; all API-only, none of them is in the GUI panel.
    # min/max are an alternative to refine_edge_len, not an addition to it:
    # set either of them and refine_edge_len must be 0. 0 = derive.
    remesh_min_edge: float = 0.0
    remesh_max_edge: float = 0.0
    remesh_surf_corr: float = 0.95
    remesh_fix_boundary: bool = True
    remesh_relax: bool = True
    remesh_passes: int = 10
    # None = freeze every elemTag seam, () = freeze none, a sequence =
    # freeze only the seams touching those labels.
    remesh_preserve_labels: Optional[Sequence[int]] = None

    # clean. 0.8 flags anything worse than roughly a 35-55-90 triangle;
    # see _triangle_quality for how the scale reads.
    clean_quality_threshold: float = 0.8
    clean_smooth_iterations: int = 20
    clean_quality_relaxation: float = 0.05
    clean_preserve_labels: Sequence[int] = field(default_factory=tuple)
    clean_merge_tol: float = 0.0   # absolute weld distance; 0 = coincident only

    # smooth
    smooth_method: str = SMOOTH_TAUBIN
    smooth_iterations: int = 40
    smooth_passband: float = 0.001   # taubin only; smaller = smoother
    smooth_relaxation: float = 0.1   # laplacian only


def apply(mesh: pv.PolyData,
          opts: PostprocessOptions,
          on_decimate_progress: Optional[Callable[[int, int], None]] = None,
          on_surface_moved: Optional[
              Callable[["pv.PolyData", "pv.PolyData"], None]] = None,
          on_progress: Optional[Callable[[str, int, int], None]] = None
          ) -> pv.PolyData:
    """Apply decimate -> refine -> clean -> fill_holes -> smooth in that
    order, skipping any step whose flag is off. Returns a new
    ``pv.PolyData``.

    The refine step runs :func:`refine` (``refine_mode=REFINE_ADAPTIVE``,
    the default) or :func:`remesh` (``REFINE_RESAMPLE``). In resample mode
    ``refine_edge_len`` is a target edge length; to give an explicit band
    instead, set it to 0 and fill in ``remesh_min_edge`` /
    ``remesh_max_edge``. Supplying both raises ``ValueError``.

    Hole filling runs last because :func:`clean` is itself a source of
    holes: its non-manifold and degenerate passes drop cells, which opens
    boundary loops that were previously hidden behind the bad geometry.
    Filling before cleaning would leave those newly-exposed loops open.
    The cost of the ordering is that filler triangles miss clean's
    quality smoothing — a quality nit, against a topology failure the
    other way round.

    Smoothing runs last, on the final topology, and is the only step that
    moves the surface as a whole.

    ``on_decimate_progress(i, n_iters)`` is forwarded to :func:`decimate`
    and fires after every outer-loop iteration of the annealing.

    ``on_progress(stage, i, n)`` reports the other long steps — ``stage``
    is ``"resample"`` for :func:`remesh`'s passes and ``"quality repair"``
    for :func:`improve_quality`'s iterations. Both stop early, so ``i``
    need not reach ``n``.

    ``on_surface_moved(old_mesh, new_mesh)`` fires after smoothing with the
    surface either side of it. It exists so a caller can carry other geometry
    (EAM electrodes) along with the wall. Surfaces rather than vertex arrays,
    because what rides along is read from the two surfaces as shapes — the
    vertex correspondence smoothing happens to preserve is over half
    tangential re-parameterisation, which is not motion anything should
    follow. No other step reports: decimate/refine/clean/fill_holes
    re-tessellate the same surface rather than move it.
    """
    def _stage(name: str) -> Optional[Callable[[int, int], None]]:
        if on_progress is None:
            return None
        return lambda i, n: on_progress(name, i, n)

    out = mesh
    if opts.do_decimate:
        out = decimate(out,
                       target_points=opts.decimate_target_points,
                       n_iters=opts.decimate_iters,
                       max_hole_size=opts.max_hole_size,
                       on_progress=on_decimate_progress)
    if opts.do_refine and opts.refine_mode == REFINE_RESAMPLE:
        band = (opts.remesh_min_edge > 0.0) or (opts.remesh_max_edge > 0.0)
        if band and opts.refine_edge_len > 0.0:
            raise ValueError(
                "resample mode takes refine_edge_len as a target *or* "
                "remesh_min_edge / remesh_max_edge as a band; set "
                "refine_edge_len=0 to use the band")
        el = opts.refine_edge_len
        if not band and el <= 0:
            el = _median_edge_length(out)
        out = remesh(out,
                     target_edge=None if band else el,
                     min_edge=opts.remesh_min_edge or None,
                     max_edge=opts.remesh_max_edge or None,
                     surf_corr=opts.remesh_surf_corr,
                     fix_boundary=opts.remesh_fix_boundary,
                     relax=opts.remesh_relax,
                     n_passes=opts.remesh_passes,
                     preserve_labels=opts.remesh_preserve_labels,
                     on_progress=_stage("resample"))
    elif opts.do_refine:
        el = opts.refine_edge_len
        if el <= 0:
            el = _median_edge_length(out)
        out = refine(out, edge_len=el)
    if opts.do_clean:
        out = clean(out,
                    preserve_labels=opts.clean_preserve_labels,
                    quality_threshold=opts.clean_quality_threshold,
                    quality_relaxation=opts.clean_quality_relaxation,
                    smooth_iterations=opts.clean_smooth_iterations,
                    merge_tol=opts.clean_merge_tol,
                    on_progress=_stage("quality repair"))
    # max_hole_size == 0 means "no hole filling" (as it does for decimate),
    # so honour that rather than letting fill_holes reject it.
    if opts.do_fill_holes and opts.max_hole_size > 0.0:
        out = fill_holes(out, max_size=opts.max_hole_size)

    if opts.do_smooth:
        # smooth() returns a new surface and leaves its input alone, so the
        # pre-smoothing mesh needs no copy of its own.
        before = out
        out = smooth(out,
                     method=opts.smooth_method,
                     iterations=opts.smooth_iterations,
                     passband=opts.smooth_passband,
                     relaxation=opts.smooth_relaxation)
        if on_surface_moved is not None:
            on_surface_moved(before, out)

    return out


__all__ = [
    "PostprocessOptions",
    "apply",
    "smooth",
    "SMOOTH_TAUBIN",
    "SMOOTH_LAPLACIAN",
    "decimate",
    "refine",
    "remesh",
    "REFINE_ADAPTIVE",
    "REFINE_RESAMPLE",
    "clean",
    "improve_quality",
    "fill_holes",
]
