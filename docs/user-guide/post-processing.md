# Mesh post-processing

The panel has two forms, and which one you see follows the mesh. A surface
gets the stages below. A **tetrahedral volume** gets the volumetric panel
instead — different operations, not a greyed-out version of these — described
in [Volumes](#volumes) at the end.

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
degenerate cells, drop connected components too small to be anatomy, orient
normals, and repair low-quality triangles while preserving the region labels. The first five are topology work; the last is
vertex relocation and is also available on its own as
`ccdaf.core.mesh_postprocessor.improve_quality`, for when you want the repair
without the topology passes — after a resample, say, where those passes would
renumber points for nothing.

!!! note "Enclosed cavities survive"

    The connectivity pass keeps every component holding at least 1% of the
    cells and drops the rest. That threshold matters on a wall around a
    chamber: the endocardium is a separate component of comparable size, and
    keeping only the largest would delete it, returning a solid lump instead
    of a wall. A stray fleck of segmentation is far below 1% and still goes.

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

## Volumes

A tetrahedral volume is adapted in one pass by MMG3D rather than through a
sequence of stages, so the panel asks for a size rather than for steps.

- **target edge** — one uniform edge length, in mesh units. `auto` (0) leaves
  the size to MMG, which keeps roughly what the mesh has and repairs quality.
- **min edge** / **max edge** — an edge-length band instead of one target.
  Mutually exclusive with the target: MMG refuses both, so setting either
  disables the other.
- **gradation** — the largest ratio allowed between the lengths of two
  adjacent edges. It does not set the size; it limits how *fast* the size may
  change. `auto` leaves MMG's own default of 1.3.
- **boundary tol.** — the Hausdorff distance, how far an adapted boundary may
  stray from the original. `auto` uses **a fifth of the element size**, which
  is what the validated runs used.

!!! warning "Do not tighten the boundary tolerance casually"

    Cost rises steeply as it tightens. At a 1.5 mm target on the 290,000-element
    example:

    | Tolerance | Time | Elements |
    |---:|---:|---:|
    | 0.3 | 23 s | 334,000 |
    | 0.1 | 27 s | 358,000 |
    | 0.05 | 56 s | 613,000 |

    `auto` deliberately does **not** fall back to MMG's own default of 0.01
    mesh units. That is a sensible figure for unit-scale geometry and absurd
    for a heart in millimetres — it asks for the surface to within 10 µm, is
    five times tighter than the slowest row above, and does not finish in any
    usable time. It simply looks as though the application has hung.

Both of those only apply while the boundary is being adapted, and are disabled
otherwise. The tolerance is obvious — a frozen boundary does not move.
Gradation is less so: it constrains how a *varying* size may change, and the
size only varies when MMG derives it from surface curvature, which it does
only while adapting the boundary. Measured on a ventricle with the boundary
frozen, changing gradation produced byte-identical meshes, so leaving the
control live would offer something that does nothing.

Where it does apply, it is the strongest knob in the panel:

| Gradation | Tetrahedra | Mean edge |
|---|---:|---:|
| 1.05 | 148,866 | 1.92 |
| auto (1.3) | 56,772 | 2.68 |
| 3.0 | 41,031 | 3.06 |

(Adapting boundary, 0.5–4.0 mm band, tolerance 0.5, on the 290,474-element
example.) Smaller grades more gently and costs elements; larger lets the size
jump and saves them.

### Adapt the boundary too

Left unticked — the default — the boundary is **frozen**. It comes back vertex
for vertex identical: measured on a 66,819-vertex ventricle, all 38,823
boundary vertices returned at distance 0.0 with the surface area unchanged.
Only the interior changes, so the anatomy is untouched.

Ticked, the surface is re-approximated within the tolerance. On the same mesh
that moved the wall by up to 1.5 mm, half a millimetre on average, and left it
with a third as many boundary vertices. That is sometimes exactly what you
want; it is never what you want by accident, which is why it is off by
default.

### What happens to the fields

MMG carries an integer reference per element through the adaptation, so the
labels ride across exactly. Everything else it returns is bare geometry, so
CCDAF carries those itself, by rules that differ because they are different
kinds of quantity:

| Field | Rule |
|---|---|
| point fields (`scar_probability`, …) | interpolated within the source tetrahedron the new vertex falls in |
| labels (`elemTag`) | carried by MMG itself as an element *reference*, so they come back exact rather than re-derived by proximity |
| directions (`fiber`) | averaged over the source elements the new element covers, as an **axis** |
| surface labels (`surfaceLabelMask`) | copied between nodes that coincide, never interpolated; every new interior node is on no surface. If the boundary was adapted, the labels are dropped and the status bar asks you to label the surfaces again |

That last rule matters more than it looks. A fibre direction is *axial*: `f`
and `−f` are the same direction, and which one a file happens to store is
arbitrary. Averaging two elements whose stored vectors point opposite ways
gives, as vectors, nothing at all — a direction pointing nowhere that renders
and exports like real data. CCDAF averages the outer products `f·fᵀ` and takes
the dominant eigenvector, which is sign-free and gives the right answer
whichever way each contributor was written down.

The surface labels need their own rule because they are bit flags, not
measurements. Interpolated, a new interior node between an epicardial node (2)
and an interior one (0) reads 1, "base", and a Laplace solve would hold it there.
With a frozen boundary every boundary node survives unchanged, so copying by
coincident node is exact: on the example ventricle all 45,138 boundary faces
keep their label.

### Clean volume

A second button on the same panel, for a different kind of problem. The remesh
changes element *sizes*; the clean repairs the mesh's *connectivity* and
reports its shape. It moves no vertex.

It runs these passes, in this order:

1. **merge points** — weld coincident points. `exact` (0) merges only points
   that are already identical, so no vertex moves. A positive tolerance welds
   near-duplicates, and the survivor is the first point of each group rather
   than a centroid.
2. **degenerate elements** — drop any tetrahedron with a repeated node or no
   volume. Welding is what turns a sliver into one of these, which is why it
   runs first.
3. **duplicate elements** — drop a second copy of the same four nodes.
4. **repair the boundary** — make the boundary manifold, where it is not. See
   below.
5. **detached pieces** — drop any face-connected piece holding less than
   **min. piece** of the elements. The largest piece is always kept, so this
   cannot empty a mesh, and a separately meshed second body is safe.
6. **reorient inverted elements** — swap two nodes of any tetrahedron with a
   negative signed volume. Same points, same shape, and the remesher refuses a
   mesh that still has them.

A detached element is not cosmetic. Any solve that puts conditions on named
surfaces builds one system over the whole mesh, and a piece carrying no
condition makes that system singular. The 290,474-element example ventricle
has exactly three: single tetrahedra held to the body by two or three nodes
and by no face at all, lying outside the body rather than plugging a void, so
removing them leaves no hole.

#### Repairing a non-manifold boundary

A boundary that is not manifold renders as a hole in a wall that has none, and
it is where a surface label leaks from epicardium to endocardium. Until it is
repaired the tunnel and cavity counts cannot be derived at all. There are two
kinds of defect and they are repaired differently.

**Material that only touches** — two pieces meeting at a single node or along a
single edge, sharing no face — can be **separated**: each piece gets its own copy
of the node. That is exact. No element is removed, no coordinate moves, and the
copies sit on top of the original, so the only thing that changes is which
element refers to which node. It can leave a piece attached by nothing at all,
which is the right answer: the detached-pieces pass then sees it for the stray
it is.

!!! warning "Separating is off, and is not on the panel"

    It is the one repair here that does not run, and not because it is wrong.
    A non-manifold edge is skipped by anything that walks faces across shared
    edges, so before the split those edges act as *accidental cuts* in the
    boundary. **Actions → Label ventricular surfaces depends on exactly that**:
    measured on the example ventricle, with the split it finds two surface
    pieces where it needs three, and the labelling fails.

    That is not a one-off. The conditions that create touching material — a
    wall pinching to nothing at the basal rim, where the epicardium and the
    endocardium run into each other — are the same conditions where those
    edges are what separates the two surfaces. So it is not offered as a
    control at all: it lives in `CleanOptions.separate_touching` for scripts
    and tests, and the contacts it would have separated are still reported.

    Welding is unaffected. It also closes *more* pinholes without the split
    than with it — 43 against 25 — because a contact the split would have
    separated stays in the shape a weld can close.

    The assumption the labelling makes was never sound, and the honest fix is
    there rather than here. When it lands, this comes back.

**A pinhole** — a passage through the wall that has closed to a point, so the
surface passes through the same node twice while the material runs continuously
around it — cannot be separated: the tetrahedra there form one connected fan,
and splitting the node would tear apart material that is genuinely joined. It
is **welded** shut instead, by filling the empty wedge at the contact with one
tetrahedron at an edge, or by capping the opening and coning it back to the node
at a vertex.

Welding adds material, so it is bounded. **weld limit** is the largest element a
weld may add, as a multiple of the elements already at that contact; at the
default of 2 a pinhole costs about one element, while a passage that is
genuinely open would need an element many times the local one and is reported
instead. Set the limit to *report only* and the repair still separates touching
material, but every pinhole is counted rather than filled. Untick **Repair a
non-manifold boundary** and both are reported and nothing is changed.

On the example ventricle, at the default settings, the clean welds 43 pinholes
shut with 56 elements and leaves one contact alone: 290,474 → 290,530
tetrahedra, with the volume up by 0.013% (14.7 mm³ of 109,203). With `separate_touching=True` it separates 15 contacts and welds 25, and the 18
non-manifold edges and 17 pinch points both go to none — at the cost described
in the warning above. The added elements are
ordinary in shape, not slivers — their median shape distortion matches the mesh
mean, and the mesh's worst element is unchanged.

!!! note "What it reports but will not repair"

    **An open tunnel** — a hole you could see through — is counted, never
    filled. It is either anatomy or a segmentation artefact, nothing local
    tells those apart, and filling one would invent material that was never
    imaged. With the boundary repaired, the example ventricle reports 2.

    A contact the repair could not resolve is counted too, never silently
    left.

The report gives the Euler characteristic χ = V − E + F − T, which is exact for
any mesh, and the number of tunnels and cavities **only when the boundary is
manifold**. Deriving those needs a count of boundary sheets, and that count is
wrong on a boundary that pinches, so before the repair they are reported as
*not determined* rather than guessed.

### Check the wall

A fourth button, beside **Clean volume**, that measures and changes nothing.

It counts the **handles** of the boundary — the ways through the wall, a handle
being a loop you cannot shrink to a point — and compares that with what the
valve openings imply. The target is not zero: a shell around *p* cavities opened
by *o* valve openings has **o − p** handles when every cavity opens at least
once. A cavity opened twice must have one, because you can travel in through the
mitral opening, along the cavity, and out through the aortic one. Anything beyond
that count is a hole in the wall.

Both measurements are undefined on a boundary that is not manifold, and the
clean deliberately leaves one that is (see the warning above), so the check
**repairs a copy**, measures that, and reports. Your mesh is untouched, and what
it tells you is a property of the anatomy rather than of which repairs happen to
be switched on.

On the example ventricle it takes about 16 seconds and says:

```
The wall has 1 hole in it: 2 handles measured against the 1 that
3 openings into 2 cavities imply.

Cavities: 150.7 mL, 141.2 mL
Openings: 1226 mm², 177 mm², 606 mm²

4 join(s) between the epicardium and a cavity, largest first:
  232.2 around, centre (10.2, -3.34, -6.11)
  179.3 around, centre (42.6, 25.4, 12.9)
  91.3 around, centre (31.7, -11.4, 38.8)
  11.7 around, centre (56.4, -9.97, 24)   <- too small to be a valve opening
```

The three large rings are the valve rims; the fourth is a perforation, in a wall
that thins to 0.39 mm beside it. The order-of-magnitude gap between the third
and the fourth is what makes "the largest *n* are the valves" a sound rule
rather than a guess.

!!! tip "Run it on the mesh as loaded, before cleaning"

    A cleaned mesh can be *harder* to measure than the one it came from. Welding
    is on by default and separating is not, so a clean fuses the contacts the
    check's own full repair would otherwise have separated — and a contact the
    weld refused stays refused. The copy then cannot be made manifold and the
    counts have no value.

    When that happens the report says so and names the places left, so it still
    tells you where to look:

    ```
    The wall could not be measured: the boundary is not manifold even
    after a full repair, at 1 place.

    Still not manifold at:
      (56.1, -9.68, 23.9)
    ```

    On the example ventricle that coordinate is the perforation itself.

!!! note "It finds joins, not every hole"

    The ring finder reports where the epicardium and a cavity are **joined**. A
    hole that does not join two surfaces is counted in the handle total but has
    no ring listed. Locating every one of them needs a homology basis rather
    than a minimum cut, which is not built.

### Improve quality

A third button, for a third kind of problem. The remesh changes element
*sizes*; the clean repairs *connectivity*; this repairs element *shape*, and
only where it is bad. Everything else is left exactly as it was.

Shape is measured as `q = 1 − √2·6V / rms_edge³`, where `rms_edge` is the root
mean square of the six edge lengths. **Lower is better**: 0 is a regular
tetrahedron, 1 is flat, and an element turned inside out is reported as 2. So
**repair above** is an upper bound, and its default of 0.8 means "the bad
ones": on the example ventricle that is 889 elements out of 290,508, or 0.31%.

!!! warning "0.2 is not a strict threshold, it is nearly every element"

    In this measure the example ventricle has a *mean* of 0.22, and 48% of its
    elements sit above 0.2. A repair pass given 0.2 is not repairing the bad
    elements, it is smoothing the whole mesh. Measured on a 14-million-element
    mesh, that moved all 2.7 million vertices, by 0.125 mm on average, and cost
    **15% of the myocardial volume**.

Three passes run per round, and the round is undone whole if it did not reduce
the count of bad elements:

1. **flips** — connectivity only, no vertex moves. Three elements around an
   interior edge become two (3-to-2), and a node whose only four elements span
   five nodes collapses into one (4-to-1). Each is applied only where the worst
   element involved gets better, and only around an interior edge, so the
   boundary is untouched by construction.

    A 4-to-1 flip **removes its node from the mesh**: the replacement element
    is built from the other four nodes, so the node it collapsed has nothing
    left using it and is dropped, and the point fields follow. The report says
    how many nodes went this way. Node numbering therefore changes, which
    matters if you are matching results back to a mesh saved earlier.
2. **smoothing** — quality-guarded Taubin over the bad elements and two layers
   around them. A node's move is kept only if its own worst element does not
   get worse, and the result is checked a second time with every move in place,
   because two nodes of the same element can each be right alone and wrong
   together.
3. **shifting** — gradient descent on `Σ 10^(q + 1 − thr)` over the elements at
   each bad node, with a backtracking step. The power makes the worst element
   dominate the sum, so the step is aimed at the damage rather than at the
   average.

#### What happens to the wall

81% of the elements above 0.8 on the example ventricle have at least one node
on the wall and 17% have all four, so freezing the boundary would leave most of
the problem untouchable. **Let wall nodes slide** (on by default) lets a
boundary node move within its own tangent plane: the component of every step
along the surface normal is removed, and no node may end up further from where
it started than 30% of its shortest edge.

Measured on the example ventricle, with the wall sliding:

| | |
|---|---|
| set of boundary faces | unchanged |
| surface area | +0.00007% |
| total volume | −0.0000% |
| wall drift off its original surface | 0.07 mm worst, 1.4 µm at p99 |
| furthest any node travelled | 0.36 mm, along the wall |

Nodes on a **sharp edge** never move, whatever the setting: an edge where two
boundary faces meet at more than 30° is a feature, which is what keeps the rim
of a valve opening a rim. Nodes where the boundary is not manifold are held
too, because a surface normal is not defined there — run **Clean volume**
first and there will not be any.

On the example ventricle, after the clean, the pass takes 13 seconds and brings
**889 elements above 0.8 down to 7**, with the worst element improving from
0.895 to 0.870 and 872 flips applied.

### A note on speed

MMG is fast when it is told a size and slow when it is not. On the 290,474-tet
ventricle, a frozen-boundary remesh at a 2 mm target takes about 10 seconds —
7 in MMG and the rest carrying the fields.
Asking MMG to optimise quality with no size at all is the pathological case —
it invents a size map and multiplied the same mesh 23-fold over 8 minutes — so
the panel always sends a size and never exposes that mode.
