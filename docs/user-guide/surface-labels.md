# Labelling the ventricular surfaces

**Actions → Label ventricular surfaces…**

Splits a tetrahedral ventricular volume's boundary into four named surfaces:
the **base**, the **epicardium**, the **LV endocardium** and the
**RV endocardium**. Those four are what a rule-based fibre generator needs, and
they are the input a Laplace–Dirichlet method puts its boundary conditions on:
see [Fibres](fibres.md).

## Two kinds of mesh, two methods

On a mesh that still carries its valve openings, the epicardium and both
endocardia are **one connected surface**, joined through each opening. No local
geometric rule separates them, because an opening is about as wide as the
chamber it opens into. What separates them is the base, and where the base
lies depends on the mesh:

- **Flat cut.** A mesh truncated above the valves has a flat basal face.
  Removing it leaves exactly three pieces. The tool finds that face from a
  plane; it **selects** faces on the plane and does not cut.
- **Valve openings.** A mesh with its valves open has a ring at each opening
  instead: the short band lining the inside of the hole, which is not flat
  (a mitral annulus is saddle-shaped). The tool finds each opening and builds a
  ring there. See [Meshes with valve openings](#meshes-with-valve-openings).

## Choosing the method

- **Method**: *Automatic* (the default), *Flat cut* or *Valve openings*.
  Automatic tries the flat cut first, because it is fast and exact on a
  truncated mesh. If the cut does not split the surface into three, the mesh
  still has its openings, and Automatic uses them instead. The report says
  which method was used and why.
- **Mesh units**: *mm*, *cm* or *µm*. Only the valve-opening method uses it,
  because its sizes are anatomical lengths in millimetres. It is guessed from
  the mesh: the RMS distance of a ventricle from its centre is about 45 mm
  (measured 43.7 to 44.8 mm on three hearts), so the unit that brings the mesh
  closest to that is chosen. The guess is shown next to the box; change it if
  it is wrong. Changing it clears the openings until you detect them again.

The section below the method shows the controls of the method in use: the
plane for a flat cut, the list of openings for valve openings.

Forcing the wrong method is refused, not silently answered. *Flat cut* on a
mesh with open valves fails its own check (the surface stays in one piece).
*Valve openings* on a truncated mesh would pass its check and still be wrong:
on the truncated example it labelled the whole cut, and 38% of the LV
endocardium, as epicardium. So it refuses whenever a flat cut splits the mesh
into three. That fact is learnt when the window opens, so the refusal costs
nothing.

## Using it

The dialog opens with an answer. The method is chosen, the base is found, the
labelling is run, and the result is shown. If it looks right, press OK.

The window is **not modal**: it stays open while you rotate the view, drag the
plane and check again. It closes itself if the working volume is replaced, for
instance by a remesh or a clean, because what it measured no longer exists.

### Flat cut

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

## Meshes with valve openings

### Controls

- **The list**: the openings found, largest first within each blood pool.
  A mitral and an aortic valve lying side by side can come back as one
  opening. That is fine: the base there is one ring instead of two.
- **Detect again**: find the openings again, with the units above, then label.
- **Remove selected**: drop an opening found where there is none, then press
  *Check these openings*.
- **Check these openings**: build a ring at each listed opening and report,
  writing nothing. The preview draws the rings in red and a yellow marker at
  the centre of each opening.

### How the openings are found

1. **The blood pools.** The solid is turned into 1 mm voxels and closed with a
   30 mm ball, which fills the cavities. That fill also lays a thin layer over
   the whole base, so it is then opened with a 6 mm ball. What is left is
   exactly the two pools, and each touches only its own endocardium.
2. **The openings.** An opening is where a pool meets the outside air. Air
   only counts as outside if it reaches the edge of the grid through gaps
   wider than 1.5 voxels, which leaves out thin pockets between a pool and
   its wall.
3. **The rings.** Faces within 3 mm of an opening form a search band. Inside
   it, the shortest loop that separates the epicardial side from the
   endocardial side is found. The narrowest section of an opening is its
   throat, which is what a shortest loop finds. The base is the faces
   touching that loop.

The band is narrow on purpose. At 8 mm the loop has room to slip down into
the LV cavity, and the LV agreement below falls from 99.3% to 96.9%.

Two small corrections follow. A few stray faces cut off between the ring and a
wall defect join the base, and so does any face whose three nodes all lie on
the ring: it sits inside the ring, and the saved form would read it back as
base anyway.

Edges shared by more than two faces (a defect some meshes from segmentation
have) do not count as connections. On the example ventricle 18 such edges join
the epicardium directly to an endocardium, and counting them would keep the
surface in one piece whatever is cut.

### Measured accuracy

**Agreement with reference labels, as a share of each surface's area**

| Mesh | Epicardium | LV endocardium | RV endocardium | Time |
|---|---:|---:|---:|---:|
| Biventricular reference, labels from its coordinates | 99.1% | 99.3% | 99.9% | 18 s |
| Second biventricular mesh, labels from another pipeline | 100% | 97.3% | 98.7% | 31 s |

On the first mesh 93% of the ring's nodes lie exactly on the reference base.
On the second, the shortfall is one strip beside the ring in a merged
mitral-aortic opening, where the reference base itself has a gap.

!!! note "No seed placement, yet"

    Every opening was found on every mesh tried, so there is no way yet to add
    one by hand. If a mesh turns up where one is missed, the labelling refuses
    (the surface stays in one piece), and adding an opening at a clicked point
    is the planned remedy.

## The acceptance test

Removing the base must leave **exactly three** connected surfaces. If it leaves
one, the tool refuses and says so.

That refusal is the useful part. It is what tells you the plane is in the wrong
place, or that an opening is missing, instead of letting a meaningless solve
proceed. It also catches the detector's honest limitation: detection
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

Truncated across the ventricles, the flat-cut method reports:

| Surface | Faces | Nodes | Share of area |
|---|---:|---:|---:|
| base | 7,210 | 4,179 | 4.8% |
| epicardium | 17,023 | 8,754 | 46.1% |
| LV endocardium | 8,740 | 4,525 | 20.3% |
| RV endocardium | 12,165 | 6,263 | 28.8% |

Before truncation, with its valves open, *Automatic* finds that no flat cut
splits it and uses the openings. It finds two pools (151 and 141 mL) and three
openings, two on one pool and one on the other, and reports base 1.6%,
epicardium 48.4%, LV endocardium 22.5% and RV endocardium 27.5% of the area,
in about 6 s.
