# Universal atrial coordinates

**Actions → Compute UAC…**

Gives every point of a labelled atrial surface, left or right, two coordinates
on a fixed unit square, so the same place in different atria has the same
coordinates. The veins and the appendage land in the same place for every
patient. The coordinates are written as point arrays and can be shown like any
other field. Which atrium it is comes from the labels; the right atrium is
described in [Right atrium](#right-atrium). A surface that carries labels of
both atria is refused for now: the two together are not supported yet.

**The two coordinates and their ranges**

| Array | Coordinate | Range |
|---|---|---|
| `uac_alpha` | septal to lateral | 0 on the septal side (right veins) to 1 on the lateral side (left veins) |
| `uac_beta` | posterior to anterior, over the roof | 0 at the posterior mitral valve, 0.5 on the roof, 1 at the anterior mitral valve |

The posterior wall is the lower half of the square (`uac_beta` below 0.5), the
anterior wall the upper half.

<figure markdown="span">
  ![The UAC template](../assets/img/uac_template.png)
  <figcaption>The template square. The inferior veins are circles on the
  posterior half; the superior veins are half-ellipses standing on the roof
  line; the appendage is an ellipse on the anterior half. The small circles are
  the veins' open ends.</figcaption>
</figure>

## What it needs

The item is enabled on a **triangle surface** whose `elemTag` has all six left
atrial labels:

| Label | Region |
|---:|---|
| 1 | body |
| 11 | LSPV |
| 13 | LIPV |
| 15 | RSPV |
| 17 | RIPV |
| 19 | LAA |

These are the labels [Seeds & tagging](seeds-tagging.md) writes. The surface
must be open at the mitral valve and at the end of each vein (see
[Clipping](clipping.md)). The appendage stays closed. No landmarks are needed.
Lengths are taken to be in millimetres.

Otherwise the item is greyed out, and its tooltip says what is missing.

## The window

There is nothing to choose. **Compute** runs in the background (a few minutes
on a typical atrium) and reports progress; **Stop** stops it and changes
nothing. When the run finishes, the surface is replaced by its repaired copy
(see [Repairs to the surface](#repairs-to-the-surface)), with `uac_alpha` and
`uac_beta` added and shown. Every other field comes along. **Undo** puts back
the surface as it was before Compute.

**Save flat map…** saves the square itself as a mesh (see
[The flat map](#the-flat-map)). It works on any surface that has `uac_alpha`
and `uac_beta`, also one loaded from a file, with no need to compute again.
**File → Save** ticks the two arrays by default.

If the surface already has `uac_alpha` and `uac_beta` (from an earlier run or
from the file), the window says so: **Compute** replaces them, computed from
the labels as they are now. Use this to recompute after changing the tagging.
The replaced values stay in the file on disk until you save.

The window is not modal, so you can keep working in the view. If the surface or
its labels change while a run is in progress, the result is discarded.

**The report** gives:

- the solver used (see [Speed](#speed));
- how the atrium was cut open;
- for each half, the number of folded triangles (0 is the normal result) and
  how far each ring ended from its place on the template (in units of the
  square; below 0.01 is a close fit);
- any patch the map collapsed to a point (see [When it fails](#when-it-fails));
- what the repair changed on the surface, and any vein extensions added;
- warnings, if the fit is poor.

## How it works

The method follows Roney et al. (*Med Image Anal* 2019), with changes that make
it fully automatic and keep the map free of folds.

1. **Repair** the surface (see below). The repairs are kept.
2. **Vein extensions.** Each vein fills a ring between its ostium and its open
   end on the template. Clipped veins are usually shorter than that ring
   needs, and a short vein would be crushed. Each vein is lengthened with a
   virtual extension for the computation only; it is never written to the
   surface and gets no coordinates.
3. **Cut.** The atrium is opened along a line from the mitral valve up the
   lateral wall, round the posterior half of the LSPV ostium, over the roof,
   round the posterior half of the RSPV ostium and down the septal wall back to
   the valve. Both inferior veins must end on the posterior half; both
   superior veins and the appendage on the anterior half.
4. **Flat map.** Each half is laid on its half of the square with a map that
   cannot fold, then warped so every ring reaches its template place, again
   without folding a triangle.

**Points on the cut** belong to both halves and so have two values. The point
arrays carry the posterior value. The anterior value follows from it: on the
lateral and septal lines it is (`uac_alpha`, 1 − `uac_beta`), and on the roof
the two are the same. The flat map has both.

### Repairs to the surface

A flat map cannot be made of every surface a segmentation or a clip leaves.
Compute repairs these, in this order, and keeps the result as your surface.
No point moves.

**What Compute repairs, and which other step does the same**

| Repair | What it fixes | Also done by |
|---|---|---|
| Label islands | A small patch of one label inside another joins the label round it, so each label is one patch | Tagging (on accept and on load) |
| Duplicate triangles | A zero-area flap listed twice goes; the wall keeps its triangle | [Clean](post-processing.md#clean) |
| Winding | Triangle orientation made consistent | Clean |
| Ears and slivers on openings | A triangle with two sides on an opening, or a very flat one along it, is removed with any point it leaves unused | [Apply clip](clipping.md) |
| Edge flips | A long thin pair of triangles is re-cut along its other diagonal | Clean (tick box), [Flip edges](post-processing.md#flip-edges) |
| Pinched points | A point the rim passes through twice gets one copy per side | Clean, Apply clip |
| Thin tunnels | A bridge of tissue is cut along its shortest loop, under 15 mm, and both cut ends are closed with a patch that takes the label round it | Clean |

A surface already repaired by those steps needs nothing more, so Compute leaves
it as it is and the report says "no repair needed". A tunnel whose shortest
loop is longer than 15 mm is not a thin bridge: it is not cut, and the run stops
and asks you to fix the segmentation.

A copied point (pinch or tunnel cut) keeps its fields; a tunnel patch's centre
point gets the mean of its neighbours. That value is filled in, not measured.

## The flat map

**Save flat map…** writes the square as a VTK surface in the plane z = 0:

- the points' x and y are `uac_alpha` and `uac_beta`;
- points on the cut appear once for each half;
- it is drawn from the stored `uac_alpha` and `uac_beta`, so it is the same
  map whether you save it right after Compute or after loading the atrium
  from a file;
- point array `uac_source` is the index of the atrium's point each flat point
  comes from, so any field of the atrium can be carried to the square;
- every other point field and every cell field of the atrium (`elemTag`
  included) is carried across;
- vein extensions are left out (they are never part of the surface).

## When it fails

**Messages that stop a run, and what to do**

| Message | Meaning | What to do |
|---|---|---|
| Labels missing | One of the six labels is absent | Tag the atrium |
| No mitral valve opening | No opening borders body tissue | Clip the mitral valve |
| Left and right veins look swapped | A "left" vein is farther from the LAA than the "right" vein at the same level. The LAA lies next to the left veins (on 226 labelled atria the right vein was always at least 1.18 times as far), so the labels are reversed | Swap the left and right vein labels |
| The surface has N tunnel(s) … longer than 15 mm | A wide tunnel, not a thin bridge | Fix the segmentation |
| No valid cut. The left and right vein labels touch … | No body tissue between the left and right veins for the roof | Shrink the vein labels |
| No valid cut. … The vein layout may be twisted, or vein labels swapped | No cut can put both inferior veins on the posterior half | Check which vein is which |

A run can also finish with a **warning** that a ring is far from its template
place. The coordinates are still fold-free, but they are less precise around
that ring. The usual cause is a vein label that runs well onto the body; the
warning names any ostium longer than 90 mm.

The report can also list a **patch collapsed to a point**, with its size in
millimetres and its label. This is almost always a thin spike in the
segmentation: a closed finger of tissue a few millimetres long and a fraction
of a millimetre wide. Every flat map squeezes such a spike to a point, so all
its points get the coordinates of its base. The rest of the map is not
affected. Triangles inside the patch have no size on the map, so they are not
counted as folded. To remove the spike, smooth or clean the segmentation.

## Speed

The warp solves a large sparse system about 200 times per atrium. The fastest
solver installed is used, without asking:

1. CHOLMOD (`scikit-sparse`, installed by `environment.yml`);
2. PARDISO (`pypardiso`, Intel and AMD processors only);
3. SciPy's built-in solver, always present and slowest.

The report says which one ran. All three give the same coordinates.

## Right atrium

On a surface tagged as a right atrium ([Seeds & tagging](seeds-tagging.md#right-atrium)),
Compute UAC runs the right-atrial version of the same method: the same repairs,
virtual extensions of short vessels, cut, first map and warp to a template,
with the right atrium's cut and template.

**What it needs:** all five right-atrial labels (RA body 2, SVC 21, IVC 23,
CS 25, RAA 27), the surface open at the tricuspid valve and at the end of the
SVC, IVC and CS.

**The two coordinates on the right atrium**

| Array | Coordinate | Range |
|---|---|---|
| `uac_alpha` | lateral to septal, over the roof | 0 at the lateral tricuspid valve, 0.5 on the roof between the venae cavae, 1 at the septal tricuspid valve |
| `uac_beta` | IVC to SVC | 0 on the IVC side, 1 on the SVC side |

This is the published convention for the right atrium. The cut runs along the
roof between the venae cavae and from each cava to the tricuspid valve; it
splits the atrium into a lateral half, which holds the appendage, and a septal
half, which holds the coronary sinus.

**The right-atrial template, from the published fibre atlas**

| Structure | Place on the square | Size |
|---|---|---|
| SVC ostium | half-ellipse on the roof, centre (0.5, 0.885) | 0.115 along beta, 0.11 across alpha |
| IVC ostium | half-ellipse on the roof, centre (0.5, 0.115) | same |
| SVC, IVC ends | circles at (0.555, 0.885), (0.555, 0.115) | radius 0.02 |
| CS ostium, end | circles at (0.85, 0.38) | radius 0.03, 0.015 |
| RAA ostium | ellipse at (0.135, 0.839) | 0.075 along alpha, 0.069 along beta |

These places were measured on the 16 labelled right atria of the published
fibre atlas (Roney et al., 2021), whose coordinates put each structure at a
fixed place. On those 16 atria the run gave 0 folded triangles and every ring
within 0.008 of its place; its coordinates differ from the published ones by
a median of 2 to 8 mm per atrium, about as much as on the left atrium.

The window, the report, Undo and **Save flat map…** work as for the left
atrium. A right atrium whose superior vena cava and inferior vena cava are
labelled the wrong way round is refused: the appendage lies nearer the SVC.

## Atrial fibres

**Actions > Atrial fibres** maps fibres onto an atrium that has UAC, left or
right. The fibres come from the average fibre atlas of Roney et al. (2021),
built from ex vivo diffusion-tensor data and published under CC BY 4.0, which
has a left-atrial and a right-atrial average for each layer. The atlas of the
atrium the labels describe is used; a surface holding both atria is refused.

**The choices in the window**

| Choice | What it writes |
|---|---|
| Layer: endocardium | Cell field `fibre_endo` |
| Layer: epicardium | Cell field `fibre_epi` |
| Layer: bilayer | Both fields, and point field `thickness` (mm) |

For a bilayer, the thickness is the distance between the two layers:

- **Fixed:** one value for every point. The default is 0.1 mm, the
  separation of the published bilayer models.
- **Point field:** a point field of the mesh. Only fields with a finite value
  of 0 or more at every point are offered.
- **File:** one value per point, as text (`.txt`, `.dat`, `.csv`; an optional
  first line with the point count) or `.npy`. The file must hold exactly one
  value per point, or the window says what is wrong.

**How a fibre is mapped.** Each triangle of the atrium is placed on the flat
map by its coordinates, and the atlas triangle at the same place gives the
fibre. The atlas fibre is written as a direction on the flat map, then turned
back into 3D through the atrium's own triangle. So a fibre keeps its course
relative to the anatomy (round a vein, along the roof) whatever the shape.
The fibres are unit vectors in the plane of each triangle.

**Beyond the atlas.** The atlas vessels are short, so the outer part of a long
vein, or of a long vena cava, has no atlas tissue at the same place on the
map. Those triangles take the nearest atlas fibre on the map, which lies
further in along the same vessel at the same position round it. The report
gives that area per vessel. A
triangle squeezed to nothing on the map (a collapsed spike) takes the mean
fibre of its neighbours.

**The atlas files.** The first use on each atrium downloads its two atlas files
(about 13 MB for each atrium) from the ccdaf release `fibre-atlas-2` into
`~/.cache/ccdaf/fibre_atlas`
(on Windows, under `%LOCALAPPDATA%`) and checks their checksums. To work
offline, copy the files into a folder and set the environment variable
`CCDAF_FIBRE_ATLAS` to it.

**Exporting.** Export > Carp offers `fibre_endo` and `fibre_epi` as the fibre
direction for `.lon`.

**Checks.** Mapping each of the four atlases onto its own mesh gives back its
fibres (median 0.00°). Mapped onto the individual hearts of the atlas (six of
the seven, endocardium), the average differs from each heart's own fibres by a
median of 35° to 41° on the right atrium and 30° to 39° on the left: that is
how far one heart is from the average, not an error of the mapping. On seven
left atria with fibres from the original pipeline, the same
mapping, given the original coordinates and the average fibre field that
pipeline used, gives back those fibres exactly. So the method is the same;
the fibres still differ in two ways:

- **The coordinates.** With ccdaf's coordinates the fibres differ by a median
  of about 25° to 30°. That is what a shift of about 5 mm gives: the atlas
  fibres change fast along the wall (points 5 mm apart differ by a median of
  26°), and ccdaf's coordinates differ from the original ones by about 5 mm.
- **The average.** ccdaf uses the average published on Zenodo. The original
  pipeline used another version of the average on the same anatomy; the two
  differ by a median of about 28°, so with the original coordinates and the
  Zenodo average the fibres differ from that pipeline's by 24° to 31°.
