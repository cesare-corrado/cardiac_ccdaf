# Mesh post-processing

The **Mesh post-processing** panel exposes `ccdaf.core.mesh_postprocessor.apply`
as a set of togglable stages, run in order on the working mesh. Enable the
stages you want with their checkboxes and set their parameters.

## Decimate

Reduce the point count.

- **target points** — the point count to anneal down to.
- **anneal iters** — maximum simulated-annealing sweeps used to redistribute the
  decimated vertices.

## Refine

Bring the mesh to a target edge length, in one of two modes.

- **mode**
    - *adaptive* — **split only.** Triangles with an edge longer than **edge
      len** are subdivided. No vertex moves and none is removed, so the point
      count can only grow and every input vertex survives.
    - *resample* — **split and collapse.** The surface is resampled into an
      edge-length band derived from **edge len**, so over-fine regions are
      coarsened as well as coarse ones refined. Vertices move, the point count
      can fall, and input vertices are not guaranteed to survive.
- **edge len** — read differently per mode: a **maximum** edge length in
  *adaptive*, a **target mean** edge length in *resample*. The same number
  therefore gives a coarser mesh in *resample* than in *adaptive*. `0` uses the
  mesh's current median edge length.

### How resample turns the target into a band

A single target length is not enough to remesh with: you need an upper bound to
decide what to split and a lower one to decide what to collapse, and if the two
are too close the algorithm cycles, splitting an edge one pass and collapsing it
back the next. The band is therefore widened around the target, by how much
depending on how far the target is from where the mesh already is — comparing
the target `t` with the mesh's current mean edge length `a`:

| when | band | why |
|---|---|---|
| `t < a/2` — target much finer | `[0.50 t, 1.8 t]` | the mesh is being refined; a wide band lets one pass split an edge into several without immediately collapsing the pieces |
| `a/2 ≤ t ≤ 1.5 a` — target comparable | `[0.65 t, 1.7 t]` | the mesh is already close to the target, so the band is tightened to hold it there rather than churn |
| `t > 1.5 a` — target much coarser | `[0.75 t, 1.8 t]` | the mesh is being coarsened; the raised floor keeps collapsing going until edges really reach the target |

`a` is the mean edge length of the input, so the band is a deterministic
function of the geometry: the same mesh always gives the same band, whatever
order its triangles happen to be stored in.

### What a pass does

Each pass splits over-long edges at their midpoints, collapses over-short ones,
flips edges that bring vertex valences closer to the ideal (6 in the interior,
4 on a boundary), and then relaxes vertices *tangentially* — each vertex slides
towards the centroid of its neighbours, but only by the part of that step lying
in its tangent plane, and the result is projected back onto the input surface.
That is what lets the triangles even out without the wall deflating: vertices
move **along** the surface, never off it.

Passes repeat until nothing is out of band, up to `n_passes`.

Two things are preserved throughout: **label seams** (an `elemTag` boundary is
never redrawn — its vertices are neither moved nor collapsed away, though seam
edges may still be split, which refines the seam without moving it) and **open
boundaries** (PV ostia and the mitral valve keep their rims).

### Resample parameters available from the API only

The panel exposes only **mode** and **edge len**. The remaining arguments of
`ccdaf.core.mesh_postprocessor.remesh` (and their `PostprocessOptions` fields)
are settable from scripts:

| argument | option field | default | meaning |
|---|---|---|---|
| `min_edge`, `max_edge` | `remesh_min_edge`, `remesh_max_edge` | derived | explicit band instead of a target. Mutually exclusive with the target: set `refine_edge_len=0` to use them, else `apply` raises. |
| `surf_corr` | `remesh_surf_corr` | `0.95` | collapse veto — an edge is only collapsible when the surface normals at its endpoints correlate by more than this. It is a dot product, so `1.0` or above disables collapsing entirely. |
| `fix_boundary` | `remesh_fix_boundary` | `True` | freeze open-boundary vertices, so a clipped mesh keeps its rims exactly. |
| `relax` | `remesh_relax` | `True` | run the tangential relaxation step. |
| `n_passes` | `remesh_passes` | `10` | maximum passes; stops early once no edge is out of band. |
| `preserve_labels` | `remesh_preserve_labels` | `None` | `None` freezes every `elemTag` seam, a sequence of labels freezes only the seams touching them, and `()` freezes none. |

## Clean

Merge duplicate points, drop disconnected points, remove non-manifold and
degenerate cells, orient normals, and repair low-quality triangles while
preserving the region labels. The first five are topology work; the last is
vertex relocation and is also available on its own as
`ccdaf.core.mesh_postprocessor.improve_quality`, for when you want the repair
without the topology passes — after a resample, say, where those passes would
renumber points for nothing.

- **quality threshold** — triangles whose shape quality is below this are
  repaired. `1.0` = equilateral, `0.0` disables the repair.
- **smooth iters** — maximum repair iterations; stops early once nothing is
  below the threshold or a step stops helping, and returns the best result seen.
- **Relaxation factor** — repair step size as a fraction of the local edge
  length.

### The quality metric

`q = 4√3·A / (l₁² + l₂² + l₃²)` — **1 is equilateral, 0 is degenerate**, and
triangles *below* the threshold are the ones repaired.

The numerator is the triangle's area, the denominator the sum of its squared
side lengths; the constant normalises the ratio so an equilateral triangle
scores exactly 1. Both parts scale with the square of the mesh's units, so the
score depends only on the triangle's *shape*, not its size — the same threshold
works on a 0.3 mm mesh and a 3 mm one.

The threshold is more demanding than it looks. A right isoceles triangle scores
0.87 and a 30-60-90 triangle scores 0.75, so `0.8` flags anything worse than
roughly a 35-55-90 triangle. On a well-resampled surface that is a fraction of
a percent of the mesh; on a raw clinical export it can be half of it.

### How the repair works

Every vertex of a flagged triangle steps along the gradient of a one-ring
badness functional — aimed at triangle *shape*, unlike a Laplacian nudge, which
only pulls a vertex toward its neighbours' centroid and has no notion of whether
that helps. Three guards keep it from buying shape with geometry:

- steps are tangential and the result is projected back onto the input surface,
  so vertices slide along the wall without ever leaving it;
- no vertex ends up further than `max_shift` local edge lengths from where it
  started;
- a step is kept only if it lowers the objective, any triangle it would invert
  is rolled back, and the **best** iterate is returned.

Without those, a bare gradient method will happily reach zero bad triangles by
wrecking the surface: it can always make a triangle equilateral by pushing a
vertex off the wall, and on a noisy input that means crumpling it — measurably,
a test sphere's area inflating by 63% while every quality score improves.

Frozen vertices, never moved: those of cells whose `elemTag` is listed in
**preserve labels**, every `elemTag` seam, non-manifold vertices, and — with
`fix_boundary` (default) — open-boundary vertices. On a clipped mesh those are
the PV ostia and mitral-valve rims.

!!! note "The defaults are tuned for a clipped atrial wall"
    Mode `resample`, edge len `0.3` mm, quality threshold `0.8`, relaxation
    `0.05`. Ticking Refine and Clean and pressing Apply is the intended
    pipeline for a clipped mesh; nothing needs typing.

    Repairing quality deliberately does **not** include a global smoothing
    pass. Denoising the wall is a different operation with a different
    trade-off, and it has its own stage — **Smooth** — so that reshaping the
    surface is always something you asked for rather than a side effect of
    fixing triangles.

## Fill holes

Close boundary loops up to a maximum size (absolute length, mesh units).
Openings larger than the threshold stay open, preserving genuine anatomical
openings (PV ostia, mitral valve).

## Smooth

Smooth the **whole** surface (reshapes the wall — unlike Clean's quality
smoothing, which only nudges bad triangles).

- **method** — *Taubin* (`vtkWindowedSincPolyDataFilter`, removes roughness
  without deflating the shell) or *Laplacian* (simpler, shrinks with iterations).
- **iterations**, **passband/relaxation** — strength of the smoothing.

!!! warning "Only the *Smooth* stage moves the electrodes"
    When a mapping is loaded, the **Smooth** stage carries the electrodes onto
    the moved wall. The **Clean** stage's quality smoothing also nudges
    vertices but does **not** trigger electrode displacement. See
    [Concepts → Electrode displacement](../concepts.md#electrode-displacement).
