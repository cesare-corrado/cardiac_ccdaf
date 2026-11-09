# Generating fibres

**Actions → Generate fibres…**

Sets a fibre and a sheet direction in every element of a labelled ventricular
volume, with the rule-based method of Bayer et al. (Ann Biomed Eng, 2012). It
is the same rule `GlRuleFibers` applies, and it gives the same answer: on a
2.9-million-element biventricular mesh, every element with a well-defined frame
agrees with `GlRuleFibers` to within 0.2°.

The item is enabled on a tetrahedral volume whose surfaces are labelled
(see [Surface labels](surface-labels.md)). Otherwise it is greyed out, and its
tooltip says what is missing.

## What it does

1. **Four Laplace problems** are solved on the volume, each with the value 1 on
   one surface and 0 on others:

    | Field | 1 on | 0 on |
    |---|---|---|
    | `ldrb_epi` | epicardium | both endocardia |
    | `ldrb_lv` | LV endocardium | epicardium, RV endocardium |
    | `ldrb_rv` | RV endocardium | epicardium, LV endocardium |
    | `ldrb_ab` | base | apex patch (see below) |

    Both values are held exactly (true Dirichlet conditions), and the three
    transmural fields add up to 1 everywhere. The report checks this.

2. **A frame per element** is built from the gradients of those fields and
   turned by the helix angle α and the sheet angle β. Each angle runs from its
   endocardial value to its epicardial value across the wall. Frames are blended
   between the LV, RV and epicardial sides.

## The window

The window is **not modal**, so you can rotate the view and click the apex
while it is open. It opens with the apex already found.

**Apex**

- The apex is found automatically, by the same rule as **Compute UVC**, so the
  two place the apex in the same spot: the LV-endocardial node farthest from
  the centre of the base, then the epicardial node nearest it. That epicardial
  node is the apex here (see [Universal ventricular coordinates](uvc.md)).
- The apex node is widened to a **patch**: the epicardial nodes within 2 mm
  of it, never a base node. A single held node would put a point singularity
  in the apex-to-base field. The apex nodes are marked in yellow.
- The patch is sized in millimetres, not in mesh edges, so a coarse mesh and a
  fine mesh of the same heart get the same patch. The mesh unit (mm, cm or µm)
  is guessed from the size of the heart, as for the surface labelling.
- The patch is kept small and on the epicardium on purpose. The fibres use only
  the *direction* of the apex-to-base gradient. A larger patch, or one that
  also holds the endocardial apex, spreads the field's values more evenly, but
  it makes that gradient parallel to the transmural one across the thin apical
  wall, so many elements there lose their frame.
- **Pick on surface**: click the epicardium in the 3D view. The nearest
  epicardial node becomes the apex, and the patch is grown around it the same
  way. Press the button again to stop picking.
- **Find automatically**: go back to the automatic apex.

**Angles**, in degrees. The defaults are those of `GlRuleFibers`:

| | endocardium | epicardium |
|---|---:|---:|
| helix α | 40 | −50 |
| sheet β | −65 | 25 |

**Defaults** puts them back.

**Generate** runs in the background, so the window stays responsive. The report
shows the preconditioner, each solver's iterations, residual and time, the
conservation check, and how many elements were rebuilt from their neighbours
(see below). The window stays open, so you can try other angles.

!!! tip "Changing only the angles is quick"

    The four Laplace fields are kept on the mesh. A second run on the same mesh,
    labels and apex reuses them and only rebuilds the frames: on the
    2.9-million-element mesh, 19 s instead of about 45 s; on the
    148,000-element example, about 1 s.

    A new apex on the same mesh and labels reuses the three transmural fields,
    which do not depend on the apex, and solves only the apex-to-base field
    again. The report says which fields were reused.

## What is written

| Where | Array | Content |
|---|---|---|
| cells | `fiber` | the fibre direction, unit length |
| cells | `sheet` | the in-sheet direction across the fibre, unit length |
| points | `ldrb_ab`, `ldrb_epi`, `ldrb_lv`, `ldrb_rv` | the four Laplace fields |
| field data | `ldrb_digest`, `ldrb_geometry`, `ldrb_apex` | what the fields were solved on, and the apex used |

`fiber` and `sheet` are the names **Export → Carp** reads, so the export writes
them to `.lon` directly (see [Export](export.md)). The four fields can be shown
in the visualisation panel like any other point field.

The window says where the mesh's current fibres came from: generated here,
carried from a previous mesh, or brought by the loaded file. Generating
replaces them.

### When the mesh changes

Fibres describe one mesh with one labelling. Whenever the volume is replaced,
the status bar (and the Generate fibres window, if it is open) says what
happened to them:

**What each kind of change does to generated fibres**

| Change | Fibres | Note |
|---|---|---|
| nothing the fibres depend on (a relabelling that gives the same labels, a tissue property) | kept | *Mesh and labels unchanged: fibres kept.* |
| the geometry (a remesh, a clean that changed something) | kept, marked as **carried** | *The mesh changed: the fibres were carried across by interpolation.* |
| the labels, on the same geometry | removed | *The surface labels changed, so the fibres built on the old labels were removed.* |

Carried fibres are what the remesh is designed to give: it averages each new
element's fibre from the old elements it covers, as a direction without sign
(see [Post-processing](post-processing.md)). They are an interpolation, not a
solution on the new mesh, so the next **Generate** solves from scratch rather
than reusing the old fields. The mark survives saving.

Fibres removed after a relabelling cannot be carried: they were built on
surfaces that no longer exist.

A remesh with the boundary **adapted** (Freeze boundary unticked) moves the
boundary nodes, so the surface labels cannot follow and are dropped; the remesh
message says so. The fibres are still carried, but **Generate fibres** is greyed
out until the surfaces are labelled again.

Fibres that came with a loaded file, and were not generated here, are never
touched.

## Showing the fibres

In the **Visualisation** panel:

- **Show fibres** draws short white line segments along the directions at a
  sample of elements, and makes the surface see-through so the segments inside
  the wall show. It is greyed out until the volume carries `fiber`.
- **Direction** chooses `fiber` or `sheet`.
- **Segments** sets how many elements get a segment (default 20,000). The
  segments get shorter as there are more of them, so they do not overlap.

Segments are lines, not arrows, because a fibre has no sign.

## Two things the method leaves open

### Elements with no frame

A frame needs two independent directions. In a few elements lying against a
labelled surface there is only one:

- **all four nodes on one surface**: every node carries the same value, so the
  field has no gradient there;
- **three nodes on the epicardium at the base rim or the apex**: the apex-to-base
  field is held constant on the same face as the transmural one, so both
  gradients point along the face normal.

The paper does not say what to do there. `GlRuleFibers` writes a fixed
placeholder direction. Here, every gradient of such an element is rebuilt
instead, from the volume-weighted average of the elements around its four nodes,
so its frame follows the surrounding tissue.

**Distance from each rebuilt fibre to its neighbours at the same wall depth, on the 2.9-million-element mesh (median, degrees)**

| Elements | Count | Rebuilt here | `GlRuleFibers` | Neighbours among themselves |
|---|---:|---:|---:|---:|
| all four nodes on the epicardium | 401 | 23.5 | 45.8 | 30.5 |
| all four nodes on an endocardium | 69 | 20.3 | 72.6 | 20.7 |
| base rim or apex, three nodes | 32 | 16.4 | 32.5 | 19.0 |

The last column is the yardstick. Near the base the epicardium is rough, so even
well-defined neighbours disagree with each other by 20° to 30°. The rebuilt
fibres sit inside that spread; the placeholder does not. These are the **only**
elements where the result differs from `GlRuleFibers`.

### Nodes on two surfaces

Where a wall has no thickness, one node can lie on the epicardium and an
endocardium at once (a *pinch point*; **Clean volume** reports them). No solve can
hold such a node at 1 and at 0 together, so it is held by none of the transmural
solves and takes the value its neighbours give it. The three fields still add up
to 1. The report says how many nodes were released: 12 of 35,856 on the
truncated example.

## Large meshes

The four Laplace problems are solved with conjugate gradients preconditioned by
algebraic multigrid (`pyamg`). The apex-to-base problem holds a small apex patch
against a large base, which makes it hard for simpler preconditioners; multigrid
solves it in a few dozen iterations. The three transmural problems hold the same
nodes, so they share one matrix and one multigrid setup.

The stiffness matrix is assembled a million elements at a time, so assembly
needs little more memory than the matrix itself.

**Fibre generation on a 35-million-element ventricle (400 µm), before and after the multigrid solver and chunked assembly**

| | Jacobi preconditioner | Multigrid |
|---|---:|---:|
| Apex-to-base solve | 45 min | 6 min |
| Whole fibre step | 56 min | 18 min |
| Peak memory | about 30 GB (assembly) | 13.5 GB (building the frames) |

The solver stops at a relative residual of 1e-9. At that tolerance the fibres
match a fully converged solution to within 0.1°; at 1e-7, 100 to 200 of 2.9
million elements, whose frames are close to undefined, came out more than 1°
off.

## Choosing the apex

The apex changes the apex-to-base field, and with it the fibres, mostly near the
apex: moving the patch by a few millimetres changes the fibres by about a degree
over most of the heart and by more within a centimetre of the apex. The
automatic apex is the one **Compute UVC** uses, so fibres and coordinates on the
same mesh agree on where the apex is. If you have a preferred apex, pick it.
