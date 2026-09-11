# Segmentation

The segmentation workflow builds a surface from a **label image** (`.nii` /
`.nii.gz`) instead of loading a mesh. It opens the **2×2 segmentation view**
(three orthogonal slices + a 3D preview).

Open it from **Segmentation → Load**; export the result with
**Segmentation → Export to VTK…**.

A label image stored in an orientation other than `LPS` is re-indexed on load
so the slices, the brush and the 3D preview agree on where each voxel is; the
status bar says so, and saving writes the original orientation back. See
[Segmentation images](../file-formats.md#segmentation-images-nii-niigz).

<figure markdown="span">
  ![The 2×2 segmentation view](../assets/screenshots/segmentation-view.png)
  <figcaption>The segmentation view: three orthogonal slices plus a 3D preview.</figcaption>
</figure>

## Morphology

### The kernel is a radius

The three boxes are a **radius per axis**, not the number of voxels across.
The element spans `2r+1` voxels on each axis, independently:

| Radius (x,y,z) | Element | Effect |
|---|---|---|
| (0,0,0) | 1×1×1 | nothing — a single voxel |
| (1,0,0) | 3×1×1 | acts along **x** only |
| (1,1,1) | 3×3×3 | the smallest isotropic element |
| (2,1,1) | 5×3×3 | twice as far along x as along y and z |

Only **odd** sizes exist, because the element is centred on a voxel: an
even-sized kernel cannot be expressed, and repeating the operation does not
work around it — dilation with a flat element composes, so radius 1 applied
twice is byte-identical to radius 2 (5×5×5), not a 2-voxel effect.

Under the hood these are SimpleITK's `BinaryDilate` / `BinaryErode` with
`kernelType=sitkBox`, a flat box element; opening and closing are composed
from them. The second argument is the **kernel radius**, which is where the
`2r+1` comes from.

!!! note "A single-axis kernel still changes all three views"

    `(5,0,0)` dilates along **x** only — measured, the bounding box grows on
    x and not at all on y or z. But a sagittal view is a y–z plane at a fixed
    x, and dilating along x makes a voxel foreground if any voxel within ±5
    *in x* was, so that slice gains tissue wherever the structure existed at
    neighbouring x positions. In-plane it looks like the cross-section
    filling out, even though no y or z voxel was involved.

    Dilation is also **clipped at the image border**: the voxelisation
    allocates one voxel of margin, so a large dilation is truncated rather
    than growing the volume.

!!! warning "A myocardium is thin, and the radius is in voxels"

    Erosion — and the erode half of an **opening** — removes `r × spacing`
    from *every* side, so anything thinner than twice that disappears
    entirely. That is arithmetic, not a fault. Because the radius is in
    voxels, the same number means different things at different spacings.
    Measured on the biventricular example:

    | Radius | Off each side | Opening | Erosion | Closing |
    |---:|---:|---:|---:|---:|
    | **at 1 mm spacing** ||||
    | 1 | 1.0 mm | 80.0% | 23.6% | 100.4% |
    | 2 | 2.0 mm | 13.5% | 0.9% | 102.6% |
    | 3 | 3.0 mm | **0.0%** | **0.0%** | 107.1% |
    | **at 0.5 mm spacing** ||||
    | 1 | 0.5 mm | 98.7% | 59.2% | 100.1% |
    | 2 | 1.0 mm | 83.9% | 23.7% | 100.6% |
    | 3 | 1.5 mm | 36.6% | 6.6% | 101.6% |

    Voxelising finer buys you finer control: a radius of 2 wipes the wall at
    1 mm and barely touches it at 0.5 mm.

    An operation that removes most of the segmentation asks first, rather
    than leaving you to find the undo button. The threshold depends on what
    the operation promises:

    - **Erode** is a request to shrink, so a large loss is the point. It asks
      only below **10%**.
    - **Opening** is a request to *smooth* — take off small protrusions. One
      that deletes most of the object is not smoothing, so it asks below
      **50%**. At 1 mm an opening at radius 2 leaves 13.5%, which is exactly
      the case this exists for.

    Dilation and closing can only grow a region, so neither can trip it.

    Note also that a **closing** can only grow the region — it is a dilation
    followed by an erosion, so its result contains the original. On a thin
    wall it bridges the cavities, which is why it can trip the growth guard
    when converting back to a volume.

Binary operations on the voxel labels, using the per-axis **kernel radius**
(voxels) below them:

- **Binary Dilate / Erode** — grow / shrink labelled regions.
- **Morph. opening** (erode → dilate) — remove small specks.
- **Morph. closing** (dilate → erode) — fill small holes.

## Cleanup

- **Fill Holes** — fill fully-enclosed holes in the segmentation.

## Manual edit (paint)

Repaint voxels interactively:

- **Actual label** / **New label** — only voxels matching *Actual* are changed,
  and become *New*.
- **Brush** — Sphere / Square / Cylinder, with a **Brush radius** (voxels).
- **2D** paints in the current slice; **3D** paints through several slices (see
  **3D depth**).
- **Activate paint mode**, then drag on a slice.
- **Convert All** relabels every *Actual* voxel to *New* at once.
- **Plane relabel** shows an orientable plane; **Apply plane relabel** converts
  *Actual* → *New* only on the side the normal points to (an oblique cut the
  axis planes cannot make).
- **Undo** reverts the last segmentation edit.

## Image smoothing → surface

Two per-axis parameter rows control the marching-cubes surface:

- **gaussian std (vx)** — Gaussian pre-smoothing standard deviation.
- **radius factor (vx)** — Gaussian kernel size.

Then:

- **Update 3D** (button over the 3D quadrant) — rebuilds the preview surface
  from the current volume and smoothing parameters. **Rendering only** — it does
  not create a working mesh.
- **Segmentation → Export to VTK…** — runs marching cubes with the same
  parameters, adopts the result as the working mesh, and (if this segmentation
  came from a mesh loaded here) carries its fields and electrodes onto the new
  surface. Converting closes the volume, so it offers to write it out as a
  NIfTI file first.

## Coming back as a volume

!!! danger "A round trip through an image can perforate a thin wall"

    Voxelising drops tissue thinner than a voxel, which opens holes through a
    thin wall — and **finer voxels do not reliably help**. Measured on a
    biventricular mesh whose source had *no* perforations at all:

    | Spacing | Handles in the reconstruction |
    |---|---:|
    | 1 mm | 28 |
    | 0.5 mm | 12 |
    | 0.25 mm | **15** |

    It does not converge. The Gaussian smoothing is not the cause — the genus
    is identical with and without it — and nothing downstream is either. It is
    inherent to going through an image, and a perforated wall is unusable for
    simulation.

    **This route is for building a mesh from an image.** To smooth or repair a
    tetrahedral mesh you already have, use
    [Remesh volume](post-processing.md#volumes) with *Adapt the boundary too*:
    it moves the elements you have rather than re-deriving them, and on the
    same mesh returned 100.0% of the volume with the Euler characteristic
    unchanged and not one new non-manifold edge.

    CCDAF says this once per session before converting a volume. It does not
    try to predict *which* segmentations perforate, because that turned out
    not to be cheaply measurable: the minimum of the distance field is one
    voxel for any shape, and the share removed by a one-voxel erosion scores a
    solid sphere the same as this ventricle.

When the segmentation was made from a **tetrahedral mesh** in the same session,
**Export to VTK…** returns a volume rather than a surface. That is the point of
keeping the tetrahedra: converting to a surface would throw away the elements
and the fibres, which is what the file was loaded for.

The working mesh is cut down to the corrected boundary in two steps: the
elements whose centres fall inside the corrected segmentation are kept, then
that mesh is remeshed with its boundary free, which pulls the element-resolution
staircase onto the corrected surface. Labels, fibres and point fields are
carried onto the result. On a 290,000-element biventricular mesh this takes
about 25 seconds and returns 99.2% of the volume when the segmentation has not
been edited at all.

!!! warning "An edit that grows the anatomy cannot be carved"

    The mesh can only be *cut*. A correction that pushes the segmentation
    outside the original boundary — a dilation, or a hole filled outward — has
    no elements out there to claim, so that material would be silently
    dropped: you would ask for a dilation and get your original shape back
    with nothing to say why.

    CCDAF measures this before anything is meshed. Material outside the mesh
    is **always dropped and always reported** in the status bar; past a
    tolerance of **5%** it stops and offers three answers instead: convert to
    a **surface** (keeps the whole correction, loses the tetrahedra and the
    fibres), **carve anyway** (keeps the volume, drops the outside), or
    cancel.

    The tolerance is set against the method's own error rather than picked
    for tidiness. Carving an *unedited* segmentation should be the identity
    and is not: it returns 99.24% of the volume at 1 mm spacing and 99.99% at
    0.5 mm. A tolerance below that 0.76% would interrupt you about quantities
    the round trip cannot resolve in the first place.

    | Edit | Growth at 1 mm | at 0.5 mm |
    |---|---:|---:|
    | Closing r=1 | 0.37% | 0.14% |
    | Closing r=2 | 2.53% | 0.62% |
    | Closing r=3 | **6.66%** — asks | 1.55% |
    | Fill holes | 0.00% | 0.00% |
    | Dilate r=2 | **62%** — asks | **45%** — asks |

### The smoothing applies to both routes

The **Image smoothing** parameters are the Gaussian applied to the distance
field before the boundary is decided. They govern all three things that read
that boundary — **Update 3D**, the surface export, and the volumetric carve —
so the preview predicts what the export produces whichever route it takes.
Zero on either parameter means no smoothing.

### Which button to press

A volume can only be **cut** from the mesh you have: elements are dropped,
never created.

| | Keeps | Loses | On a pure dilation |
|---|---|---|---|
| **Carve anyway** | tetrahedra, `elemTag`, fibres | everything outside the current mesh | **no effect** — returns your mesh unchanged |
| **Convert to a surface** | the whole corrected shape, plus `elemTag` and fibres carried onto the triangles | the tetrahedra | a hollow shell of the dilated anatomy |

That last column is not a figure of speech. Measured: a radius-2 dilation grew
the segmentation by 82%, and carving returned **100.0%** of the original
mesh's volume. A dilation only adds, so no element fell outside, nothing was
dropped, and there was nothing out there to pick up.

**Carve anyway** earns its keep when the edit *removes* something — smoothing
that shaves a bump, an erosion, repairing a spur — because then the mesh only
has to lose elements. For a purely additive edit the dialog is really saying
"this correction cannot be expressed as a cut".

The surface produced by **Convert to a surface** is a closed shell tracing
both faces of the wall; it looks solid until you notice there is nothing
between the two layers.

A segmentation loaded from a NIfTI has no mesh behind it, and one made from a
surface has no tetrahedra to keep; both convert to a surface as they always
have.

## Keeping the volume

The volume is written by **Segmentation → Save segmentation…**, in the
orientation it was loaded in.

Anything that edits it — the brush, morphology, fill holes, the plane relabel,
undo — and voxelising a mesh mark it *modified*. The two operations that drop
it then offer to write it out first, through the same prompt:

- **Segmentation → Close segmentation**, when there are unsaved edits;
- **Segmentation → Export to VTK…**, always — the volume it converts from may
  never have been on disk at all.

**Yes** opens the NIfTI save dialog, **No** goes ahead without writing, and
**Cancel** — like backing out of the save dialog — leaves the segmentation
exactly as it was. Quitting, and **File → Close**, ask about it too.

!!! warning "Electrodes and the two paths"
    **Export to VTK** recomputes electrode positions onto the reconstructed
    surface; **Update 3D** is a preview and does not. See
    [Concepts → Electrode displacement](../concepts.md#electrode-displacement).

### What the converter cleans up

Marching cubes needs tidying after it, and both paths get it:

- **Near-duplicate points are welded.** Where the isosurface grazes a voxel
  corner, marching cubes emits a triangle whose two points sit a rounding
  error apart — a sliver with almost no area. Those slivers are worth removing:
  a triangle whose area rounds to zero is skipped by VTK when it builds the
  map it picks against, which shifts every later triangle and makes clicks land
  somewhere other than the cursor. The weld distance is 1% of the voxel pitch
  (10 µm on 1 mm voxels), far below any anatomical scale, and the surface stays
  closed and manifold.

**Export to VTK** additionally drops **stray shells**. Marching cubes returns
every surface it finds in one mesh, so a speck of stray voxels — a slip of the
brush, or anything the Gaussian was too weak to dissolve — arrives as a second
closed surface floating beside the anatomy, and nothing downstream tells the
two apart: it takes a label, it is pickable, and it is exported. The largest
surface is kept as the anatomy and the specks are dropped, with the status bar
reporting what went:

> Dropped 1 stray shell (188 cells, 0.4%).

A component holding **1% or more** of the mesh is not a speck, so it is kept
rather than deleted, and reported instead:

> The surface is still in 2 separate pieces (48312 cells plus 3020) — too large
> to be specks; check the segmentation.

**Update 3D** leaves components alone: it is a preview of the volume, so it
shows you the stray shell rather than hiding it.

Both paths pad the volume with background before meshing, so a label running
into the edge of the image is still closed there — a scan cropped tight around
the anatomy no longer comes back with holes on the faces it touched. The
surface is capped half a voxel outside the last labelled plane, and cropping a
volume does not change the mesh built from it. The padding widens with the
smoothing parameters above, which is what keeps the Gaussian kernel from
reaching past the edge of the image and shaping the cap.
