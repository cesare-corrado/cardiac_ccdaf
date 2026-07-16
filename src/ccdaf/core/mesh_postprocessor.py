"""
MeshPostprocessor
=================

Mesh quality post-processing routines usable at any stage of the pipeline:

* ``decimate``  - simulated annealing subset + retriangulation
* ``refine``    - vtkAdaptiveSubdivisionFilter-like refinement to a target
                  edge length
* ``clean``     - merge duplicates, drop non-connected points, remove
                  non-manifold / degenerate cells, orient normals, and
                  smooth low-quality triangles while preserving labelled
                  surfaces
* ``fill_holes``- close boundary loops below a radius threshold, leaving
                  larger (anatomical) openings open

All routines return a fresh ``pyvista.PolyData`` with the point / cell
data from the input mesh transferred onto the new topology via nearest-
neighbour lookup (integer arrays are preserved exactly).
"""

from __future__ import annotations

import warnings
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
# CLEAN
# =====================================================================
def _triangle_quality(mesh: pv.PolyData) -> np.ndarray:
    """Radius-ratio style quality in [0, 1]; 1 is equilateral."""
    tri = _faces_to_tri(mesh)
    p = np.asarray(mesh.points)
    a = np.linalg.norm(p[tri[:, 1]] - p[tri[:, 0]], axis=1)
    b = np.linalg.norm(p[tri[:, 2]] - p[tri[:, 1]], axis=1)
    c = np.linalg.norm(p[tri[:, 0]] - p[tri[:, 2]], axis=1)
    s = 0.5 * (a + b + c)
    area = np.sqrt(np.clip(s * (s - a) * (s - b) * (s - c), 0.0, None))
    longest = np.maximum.reduce([a, b, c])
    q = np.zeros_like(area)
    m = longest > 0
    q[m] = (4.0 / np.sqrt(3.0)) * area[m] / (longest[m] ** 2)
    return q


def clean(mesh: pv.PolyData,
          preserve_labels: Optional[Iterable[int]] = None,
          quality_threshold: float = 0.2,
          quality_relaxation: float = 0.1,
          smooth_iterations: int = 20,
          merge_tol: float = 0.0) -> pv.PolyData:
    """Apply the full cleaning pipeline.

    * merge duplicate points, drop unused / non-connected points
    * remove non-manifold and degenerate cells
    * ensure consistent outward normals
    * smooth low-quality triangles (``quality < quality_threshold``) while
      freezing vertices belonging to cells whose ``elemTag`` is listed in
      ``preserve_labels``

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

    # 7. quality-driven smoothing with preserved surfaces
    if smooth_iterations > 0 and quality_threshold > 0.0 and quality_relaxation > 0.0:
        cleaned = _quality_smooth(
            cleaned,
            preserve_labels=set(preserve_labels),
            quality_threshold=quality_threshold,
            iterations=smooth_iterations,
            quality_relaxation=quality_relaxation,
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


def _quality_smooth(mesh: pv.PolyData,
                    preserve_labels: Set[int],
                    quality_threshold: float,
                    iterations: int,
                    quality_relaxation: float) -> pv.PolyData:
    """Laplacian-smooth vertices belonging to low-quality triangles,
    freezing any vertex that touches a preserved-label cell.
    quality_relaxation controls how much a point can depart from it original position
    """
    tri = _faces_to_tri(mesh)
    pts = np.asarray(mesh.points, dtype=float).copy()

    # vertices adjacent to preserved-label cells are frozen
    frozen = np.zeros(pts.shape[0], dtype=bool)
    if preserve_labels and "elemTag" in mesh.cell_data:
        tags = np.asarray(mesh.cell_data["elemTag"])
        pres_cells = np.isin(tags, list(preserve_labels))
        frozen[np.unique(tri[pres_cells].ravel())] = True

    # build vertex adjacency
    edges = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    adj: list[list[int]] = [[] for _ in range(pts.shape[0])]
    for a, b in edges:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))

    for _ in range(iterations):
        q = _triangle_quality_from(pts, tri)
        bad_cells = q < quality_threshold
        if not bad_cells.any():
            break
        active_v = np.zeros(pts.shape[0], dtype=bool)
        active_v[np.unique(tri[bad_cells].ravel())] = True
        active_v &= ~frozen
        if not active_v.any():
            break
        new_pts = pts.copy()
        for v in np.where(active_v)[0]:
            nb = adj[v]
            if nb:
                new_pts[v] = pts[v] + quality_relaxation*(pts[nb].mean(axis=0)-pts[v])
        pts = new_pts

    out = pv.PolyData(pts, _tri_to_faces(tri))
    for name in list(mesh.point_data.keys()):
        out.point_data[name] = np.asarray(mesh.point_data[name])
    for name in list(mesh.cell_data.keys()):
        if name == "render_idx":
            continue
        out.cell_data[name] = np.asarray(mesh.cell_data[name])
    return out


def _triangle_quality_from(pts: np.ndarray, tri: np.ndarray) -> np.ndarray:
    a = np.linalg.norm(pts[tri[:, 1]] - pts[tri[:, 0]], axis=1)
    b = np.linalg.norm(pts[tri[:, 2]] - pts[tri[:, 1]], axis=1)
    c = np.linalg.norm(pts[tri[:, 0]] - pts[tri[:, 2]], axis=1)
    s = 0.5 * (a + b + c)
    area = np.sqrt(np.clip(s * (s - a) * (s - b) * (s - c), 0.0, None))
    longest = np.maximum.reduce([a, b, c])
    q = np.zeros_like(area)
    m = longest > 0
    q[m] = (4.0 / np.sqrt(3.0)) * area[m] / (longest[m] ** 2)
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

    # refine
    refine_edge_len: float = 0.4   # 0 -> use median edge length

    # clean
    clean_quality_threshold: float = 0.2
    clean_smooth_iterations: int = 20
    clean_quality_relaxation: float = 0.1
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
              Callable[["pv.PolyData", "pv.PolyData"], None]] = None
          ) -> pv.PolyData:
    """Apply decimate -> refine -> clean -> fill_holes -> smooth in that
    order, skipping any step whose flag is off. Returns a new
    ``pv.PolyData``.

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

    ``on_surface_moved(old_mesh, new_mesh)`` fires after smoothing with the
    surface either side of it. It exists so a caller can carry other geometry
    (EAM electrodes) along with the wall. Surfaces rather than vertex arrays,
    because what rides along is read from the two surfaces as shapes — the
    vertex correspondence smoothing happens to preserve is over half
    tangential re-parameterisation, which is not motion anything should
    follow. No other step reports: decimate/refine/clean/fill_holes
    re-tessellate the same surface rather than move it.
    """
    out = mesh
    if opts.do_decimate:
        out = decimate(out,
                       target_points=opts.decimate_target_points,
                       n_iters=opts.decimate_iters,
                       max_hole_size=opts.max_hole_size,
                       on_progress=on_decimate_progress)
    if opts.do_refine:
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
                    merge_tol=opts.clean_merge_tol)
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
    "clean",
    "fill_holes",
]
