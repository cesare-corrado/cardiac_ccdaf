# Labelling the atrial surfaces

An atrial volume has a wall with two sides: the **endocardium** (the blood
side) and the **epicardium** (the outside). They meet only at the openings:
the mitral or tricuspid valve, the clipped veins and a clipped appendage.
There, the wall ends in a crown one wall thick, the **rim**.

**Actions → Label atrial surfaces…** splits the boundary of an atrial volume
into the rim, the epicardium and the endocardium of each atrium present: the
left, the right, or both. Volume UAC need these surfaces.

## From a four-chamber mesh: Extract chambers

A four-chamber mesh holds all four chambers, the valve planes, the caps that
close the veins and the appendage, and the great vessels. **Actions →
Extract chambers…** keeps the tissue of the chambers you tick and removes the
rest.

1. Name the tags first (**Actions → Tag names…**, see
   [Tag names](tag-names.md)). Extraction reads each tag's chamber and role,
   not its number. Load the cohort's tag file if it has one.
2. Open **Extract chambers…**. It lists each chamber with tissue in the mesh,
   with the tags it would keep. The atria are ticked.
3. Press OK. The kept tissue replaces the working mesh.

**What extraction keeps and removes**

| Kept | Removed |
|---|---|
| tags whose chamber is ticked and whose role is wall, layer, vein or appendage | caps, valves, vessels, the other chambers, and tags with no role |
| tissue on two chambers (BB, "LA+RA"): all of it when both are ticked, the part lining the ticked atrium when one is | the part lining the other atrium |
| every cell and point array, and the tag names | the old surface labels, which described the old boundary |

**Tissue shared by two chambers.** When you keep only one of them, a tag on
two chambers (BB, "LA+RA") is split between them. In the four-chamber
cohort BB is not a thin bundle: it is full-thickness wall over the roof, the
septum and round the right veins, and lines about a quarter of the LA
endocardium. So each BB element goes to the atrium **whose blood it
lines**: ccdaf first labels both atria together (see below), takes the BB
elements on each endocardium, and gives every other BB element to the
endocardium nearer along BB. The cut then runs inside the wall, as the
septum between the LA and RA bodies does, and becomes epicardium of each
atrium. Measured on one four-chamber mesh: each atrium's endocardium is the
same, face for face, whether it is extracted alone or with the other.
Extraction takes about 15 seconds instead of 6 when it splits.

When the two atria cannot be labelled together, the split falls back to the
tags: each BB element goes to the atrium whose own tissue (body and rings)
is nearer along BB.

Where a removed cap or valve met the kept tissue is exactly the rim. Before
the caps go, each node of those faces gets the cap's tag in a point array,
`openingTag` (see [File formats](../file-formats.md)). Labelling reads it, so
the rims come from the tags, not from the shape.

## Labelling

**Actions → Label atrial surfaces…** opens a window that labels the volume at
once and shows the result: the rims in red in the 3D view, and a report of the
area of each surface. Nothing is written until you press OK.

The window uses one of two methods, shown at the top.

**How the rim is found**

| Method | When | How |
|---|---|---|
| Recorded rims | the volume came from **Extract chambers** | a boundary face whose three nodes carry `openingTag` is a rim face |
| Openings from the shape | any other atrial volume | the volume is voxelised; each blood pool is filled with a large ball; an opening is where a pool meets the outside; the rim is the shortest ring round each opening |

The shape method needs the mesh units (guessed from the mesh size) and a voxel
size about half the thinnest wall (0.5 mm by default). It takes 10 to 30
seconds; the recorded rims take about one second.

Removing the rims leaves one epicardium, and one endocardium per atrium.
Stray pieces of a few faces next to a rim join the rim. Anything else (an
opening missed, or a hole in the wall) is refused with a message, rather than
labelled wrongly.

### Which atrium is which

The epicardium is the surface nearest the outside (the convex hull). Each
endocardium is named from the tags behind it, through the tag names: an
endocardium on "LA body" and the left veins is the LA. Tissue shared by both
atria (BB) has no vote.

Without tag names, or on a volume with no tags, the name is a guess (a
mesh loaded without tags gets the default tag 1, which is the LA body's
number, but that proves nothing and is not used): the
cavity with more openings is the LA (four veins and the mitral valve, against
two caval veins and the tricuspid valve). The report then says the naming is
uncertain. Press **Swap LA and RA** if the guess is wrong; on a single atrium
it renames its endocardium.

## What is written

The same arrays as the ventricular labels (see
[Surface labels](surface-labels.md#what-is-written)): `surfaceLabelMask` on the
volume's points, and `surfaceLabel` on the boundary's faces.

**Atrial surface labels**

| Surface | `surfaceLabel` | `surfaceLabelMask` bit |
|---|---|---|
| rim | 1 | 1 |
| epicardium | 2 | 2 |
| LA endocardium | 5 | 16 |
| RA endocardium | 6 | 32 |

The rim shares value 1 with the ventricular base: both are where an
endocardium meets the epicardium. The legend calls it "rim" on an atrial
volume and "base" on a ventricular one.

## Tagging the regions of the volume

Volume UAC need each tetrahedron's region: the body, each vein, the
appendage. The tagging tools all work on triangles, so the regions are drawn
on the endocardium and then carried into the wall.

1. **Actions → Tag the endocardium…** (after Label atrial surfaces). The
   volume steps aside and its endocardium becomes the working surface. With
   both atria, you choose which one.
2. On a volume from **Extract chambers**, the seeds the openings can tell
   are placed for you: each vein and appendage seed just inside the opening
   its cap closed, and the MV or TV seed at the valve. Seeds with no cap
   (the CS and the RAA) are left for you to place.
3. Each vein and appendage opening the tag names explain gets a **3 mm
   band** along its edge at once, as **Band at opening** would give. In the
   four-chamber cohorts the veins are clipped at or near the ostium, so
   there is too little stub to grow a region from a seed, and the band is
   the reliable answer: on one Zenodo mesh every tetrahedron of the
   segmentation's own vein rings took its vein's label. For veins with a
   longer stub, **Run tagging** grows them from their seeds instead (it
   replaces the bands); each such vein whose opening is known grows from
   its whole cut edge too, so it wraps round its circumference and cannot
   take the next vein's edge. On a clipped vein that growth reaches too far
   into the body, which is why the band comes first.
   Two openings closer than twice the band width would meet halfway, on a
   line of shared vertices; the band triangles at those vertices go back to
   the body, which keeps a corridor one triangle wide between the veins (on
   one four-chamber mesh, 40 of the LSPV and LIPV bands' 697 triangles).
   Where the openings are only about two rim edges apart, the corridor
   reaches the rim and a few of its triangles are body (9 of 287 there,
   after a 0.7 mm remesh). The UAC are unaffected: on that mesh they had 0
   folded triangles with and without the corridor.
4. Correct as usual (see [Manual correction](manual-correction.md)): every
   tool works, including **Band at opening** for a vein clipped at its
   ostium. Press **Accept tagging** when done; it gives any triangle left
   unassigned the body label.
5. **Actions → Apply to volume** carries the labels into the tetrahedra and
   makes the volume the working mesh again. To change something, tag the
   endocardium again and apply again.

**How the labels cross the wall.** Copying the nearest endocardial label
fails when the wall is several tetrahedra thick: the border between two
regions can zig-zag inside the wall even when it is clean on the
endocardium. Instead, for each label one Laplace problem is solved in the
volume: 1 on the endocardium with that label, 0 on the rest of the
endocardium, and no condition on the epicardium and the rims. Each
tetrahedron takes the label with the highest value. Borders between regions
are then smooth surfaces through the wall, and a vein on the endocardium is
the same vein on the epicardium, which takes its labels from the
tetrahedra. Small islands join the neighbouring region they touch most. The
report gives how well the tetrahedra behind the endocardium agree with it
(99% on the two four-chamber meshes measured).

**What is kept.** The new regions are written as `elemTag` in ccdaf's
atrial labels (1 LA body, 11 LSPV, …). When the volume had tag names (from
a four-chamber mesh), its previous tags are kept as the cell array
`sourceTag`, with their names in the field `sourceTagMap`, for a later
merge back into the four-chamber mesh. Colour by `sourceTag` to see them
with their names.

!!! note "Rings in the four-chamber cohorts"

    The vein rings of both four-chamber cohorts lie on the outside of each
    vein end, with no endocardium of their own: the veins on the
    endocardium are tagged as LA body. That is why a volume from Extract
    chambers still needs tagging, and why the seeds come from the caps
    rather than from the rings.

## Merging back into the four-chamber mesh

**Actions → Merge into source mesh…** writes what was computed on the
extracted atrium (its regions, UAC, depth, fibres, …) back onto the same
elements of the four-chamber mesh. Load the atrium, choose the action, and
pick the four-chamber file it was extracted from. The merged mesh opens as
the working mesh; save it as usual. To merge both atria, merge the left
atrium, then merge the right atrium into the result.

The atrium's elements must be the four-chamber mesh's own: an atrium that
was remeshed or cleaned after extraction is refused, and so is a different
mesh. Extraction stores each point's and element's id in the four-chamber
mesh (`sourcePoint`, `sourceCell`), and the merge uses them. An atrium
extracted before these ids were stored is matched by its coordinates
instead, with the same result.

**The fields.** Each field of the atrium is written on its elements or
nodes.

- **Same name in the four-chamber mesh** (for example `fiber` and `sheet`,
  which the ventricles have too): one array for the whole heart. The
  atrium's rows are replaced and the rest is unchanged.
- **New field:** outside the atrium it reads "not computed": NaN for
  decimal fields such as `uac_alpha`, and -32000 for integer fields such as
  `uac_half`.
- **Nodes shared with other tissue:** a value already there is kept. So
  after merging the left atrium and then the right, the nodes the two atria
  share keep the left atrium's α and β. The report says how many.
- **Not written:** `openingTag`, `surfaceLabelMask` and `sourceTag`, which
  describe the extracted atrium alone.

**The tags.** The regions of the tagged atrium are the same entities as the
four-chamber tags (body, vein rings, appendage ring), numbered differently,
and the tagging moves their borders into the body. They are written into
the four-chamber `elemTag` in its own numbering, matched by name: an
element of the atrium's body tag (LA body, RA body) that the tagging put in
a vein takes the tag of that vein's ring. Every other element keeps its
tag:

- a ring element the tagging left in the body;
- a tag shared by both atria, such as the interatrial bundle, where it lies
  in a vein's region;
- the elements of a region with no tag in the four-chamber mesh (for
  example the CS).

The report gives each count. On a Zenodo mesh, the left atrium's merge
retagged 11,539 body elements to the veins and the appendage, and left
151,093 as body.

## Remeshing an extracted atrium

An atrium can be remeshed on its own and put back, for example to refine it
where two veins are less than an element apart. On one four-chamber mesh
the LSPV and LIPV openings are 1.5 mm apart on the wall, about one element,
so only a one-triangle corridor of body fits between their bands; at
0.5 mm the strip between them is three elements wide.

1. **Extract chambers…**, choosing the atrium.
2. **Remesh volume** (see [Post-processing](post-processing.md#keep-the-interface-with-the-source-mesh))
   with **Adapt the boundary too** and **Keep the interface with the source
   mesh** ticked. The faces the atrium shares with the rest of the heart
   stay where they are; the endocardium and the rest of the epicardium are
   remeshed.
3. **Merge into source mesh…**, choosing the four-chamber file. The old
   atrial elements are removed and the remeshed ones put in their place,
   joined at the interface. The result is a new four-chamber mesh.
4. Save it, and continue from there: **Extract chambers…** again, then
   label, tag, and compute as usual.

The merge refuses an atrium whose interface moved (remeshed without the
option, or from another mesh), and checks that every face between the old
atrium and the rest of the heart is on the remeshed atrium, so the result
is conforming. The report gives the change in volume (+0.01% on the
four-chamber mesh above, where the remesh took the atrium from 223,067 to
785,158 elements).

Two things to know:

- **The mesh is renumbered.** Nodes only the old atrium used are dropped,
  so atria extracted from the old four-chamber mesh no longer match the new
  one by their ids. Extract again from the new mesh.
- **A tag shared by both atria is split again** when you extract from the
  new mesh, on its new elements, so a few of its elements can change side
  (271 of 193,717 on the mesh above).

## Measured accuracy

On a four-chamber mesh from each of two cohorts, the recorded rims gave one
epicardium and one endocardium per atrium, for the left atrium, the right and
both, with at most 24 stray faces. The naming from the tags was certain in
every case.

The shape method was run on the same volumes with their tags and rims
removed, and compared with the recorded rims.

**Agreement of the shape method with the recorded rims (share of area)**

| Volume | Epicardium | LA endocardium | RA endocardium |
|---|---|---|---|
| LA, cohort 1 | 0.996 | 0.968 | |
| LA, cohort 2 | 0.986 | 0.989 | |
| LA + RA, cohort 1 | | 0.966 | 0.959 |
| LA + RA, cohort 2 | | 0.989 | 0.980 |

The rims themselves differ more, because they are different things: the
recorded rim is the crown the cap covered, and the shape rim is the shortest
ring round the opening. Both separate the same two sides. On both biatrial
volumes the guess from the number of openings named the atria correctly.
