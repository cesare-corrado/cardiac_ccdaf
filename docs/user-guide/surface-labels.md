# Labelling the ventricular surfaces

**Actions → Label ventricular surfaces…**

Splits a truncated tetrahedral volume's boundary into four named surfaces:
the **base**, the **epicardium**, the **LV endocardium** and the
**RV endocardium**. Those four are what a rule-based fibre generator needs, and
they are the input a Laplace–Dirichlet method puts its boundary conditions on:
see [Fibres](fibres.md).

## Why a plane is needed at all

On a mesh that still carries its valve orifices, the epicardium and both
endocardia are **one connected surface**, joined around each opening. No local
geometric rule separates them, because an orifice is about as wide as the
chamber it opens into.

Cut the mesh above the valves and the problem disappears: the boundary falls
into exactly three pieces. So the plane is not a convenience, it is the
information the geometry does not contain.

This tool **selects** faces on a plane; it does not cut. The mesh must already
be truncated.

## Using it

The dialog opens with an answer. The cut plane is detected, the labelling is
run, and the result is shown. If it looks right, press OK.

The window is **not modal**: it stays open while you rotate the view, drag the
plane and check again. It closes itself if the working volume is replaced, for
instance by a remesh or a clean, because what it measured no longer exists.

- **point on the plane** and **normal** — the plane, as six numbers. Edit them
  to override the detection; the sign of the normal does not matter.
- **Detect again** — find the largest flat, coplanar patch of the boundary.
- **Modify plane** — show a draggable plane in the 3D view. Drag its centre to
  move it and its arrow to turn it; the six numbers follow as you go. Press the
  button again to put it away, then press *Check this plane*. It is the same
  gizmo the segmentation view uses for its half-space relabel.
- **Check this plane** — label with the plane above and report, writing nothing.
  Each check refreshes the preview: the plane is drawn as a translucent sheet
  and the faces that would become the base are highlighted in red. A refused
  plane clears the preview, so what you see never contradicts the report.

!!! tip "A dragged plane snaps onto the cut"

    This tool selects faces on a cut the mesh already has, so a plane placed by
    hand lands *beside* the answer rather than on it, and nothing lies on it.
    Check therefore snaps: if your plane selects nothing, it looks for a real
    cut within **8 mean edge lengths** and within **45°** of parallel, uses that
    instead, moves the six numbers onto it, and says how far it moved.

    So dragging means "use that cut", not "cut exactly here". Measured on the
    example, a plane slid up to 6 units off the cut, or tilted by up to 40°,
    snaps back onto it exactly; at 10 units it reports that it found nothing
    rather than guessing.
- **distance (× mean edge)** — a face joins the base when all three of its nodes
  lie within this many mean edge lengths of the plane. Default 0.5.
- **| cos angle | >** — how parallel a face must be to the plane. Default 0.95.

Editing the plane clears the previous answer, so OK is only ever pressed
against a result you have seen.

## The rule, and why it has two halves

A face is part of the base when **all three nodes lie within the distance band**
**and** its normal is nearly parallel to the plane's. Measured against a known
cut on the example ventricle:

| Rule | Precision | Recall | Patches |
|---|---:|---:|---:|
| distance + angle | 1.000 | 1.000 | 1 |
| angle only | 0.960 | 1.000 | 28 |

The distance test is what makes the answer exact: a planar cut is exactly
planar, so nothing lies *near* the plane without lying *on* it, and the
tolerance barely matters (0.5 and 1.0 select the same faces). The angle test
guards the case where the surface runs tangent to the plane. Without it the
selection scatters into 28 patches, pinning nodes to ψ = 1 in places that have
nothing to do with the base.

## The acceptance test

Removing the base must leave **exactly three** connected surfaces. If it leaves
one, the tool refuses and says so.

That refusal is the useful part. It is what tells you the plane is in the wrong
place, or that the mesh was never truncated, instead of letting a meaningless
solve proceed. It also catches the detector's honest limitation: detection
maximises coplanar area and knows nothing about anatomy, so on a mesh flat at
both ends it can return the end the cavities do not open onto.

## How the surfaces are named

- **Epicardium**: the piece lying on the convex hull. On the example its median
  depth beneath the hull is 0.05, against 3.55 and 4.26 for the two cavities.
- **LV against RV**: by the tags of the elements behind each surface. The LV
  cavity is enclosed entirely by LV and septal tissue, which carry one tag,
  while the RV cavity is bounded by the RV free wall **and** by the septum, so
  it comes back mixed (measured: 1.00 pure against 0.74 / 0.26 mixed).

!!! warning "Size is the wrong signal"

    On the example ventricle the LV endocardium is the **smaller** of the two
    cavities, 21% against 30% of the boundary area, so naming them by area gets
    them backwards. Area is used only as a last resort, and when it is, the
    report says the naming is uncertain.

A mesh with a single myocardial tag gives the rule no signal. The labelling
still completes, and says the naming is uncertain, which means checking which
cavity is which before relying on it.

## What is written

Pressing OK writes `surfaceLabel`:

| Where | Array | Why |
|---|---|---|
| the volume's **points** | `surfaceLabelMask`, membership bit flags | the complete record: it is the only thing written to the file |
| the boundary's **cells** | `surfaceLabel`, one label per face | what you colour by; derived from the mask whenever the boundary is rebuilt |

One array is stored, not two. A plain label per node is exactly the mask's
lowest set bit — checked on the example ventricle, all 35,856 nodes agree — so
keeping both would store the same thing twice, and the mask is the half that
can describe a node lying on a seam. `surfaceLabelMask` is therefore saved and
exported but not offered as something to colour by; `surfaceLabel` is.

Values are `0` unlabelled, `1` base, `2` epicardium, `3` LV endocardium,
`4` RV endocardium. The field is categorical, so the visualisation panel draws
it with discrete colours rather than a ramp.

Saving the volume keeps both node arrays; the per-face form is rebuilt from
them, because the boundary itself is rebuilt whenever the volume changes.

`surfaceLabelMask` is what makes that rebuild exact. Each surface owns one bit
(1 base, 2 epicardium, 4 LV endocardium, 8 RV endocardium), so a node on the
rim between base and epicardium reads **3**: it belongs to both. A face is then
whichever surface all three of its nodes belong to.

That matters because one label per node cannot describe a seam. On the example
ventricle 1,162 nodes belong to two surfaces, so a single label has to choose,
and the faces around those nodes disagree and come back unlabelled: 1,775 of
45,138 faces, about 4%, showing as a thin bare line along every join.
Membership does not have to choose, and recovers all 45,138.

!!! note "Why the mask has its own name"

    The value 3 means "LV endocardium" as a label and "base + epicardium" as a
    mask. Storing the mask under the name `surfaceLabel` would make files
    written by earlier versions silently wrong, so it is a separate array.

## On the example ventricle

Truncated across the ventricles, the tool reports:

| Surface | Faces | Nodes | Share of area |
|---|---:|---:|---:|
| base | 7,210 | 4,179 | 4.8% |
| epicardium | 17,023 | 8,754 | 46.1% |
| LV endocardium | 8,740 | 4,525 | 20.3% |
| RV endocardium | 12,165 | 6,263 | 28.8% |

Run on the same mesh **before** truncation, it refuses, reporting that removing
the base left one surface rather than three because the valve orifices join the
epicardium to the endocardium whatever is cut.
