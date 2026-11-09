# Universal ventricular coordinates

**Actions → Compute UVC…**

Gives every node of a labelled biventricular volume four coordinates that say
where it is in the heart, independently of the heart's size and shape (Bayer et
al., *Med Image Anal* 2018). They are written as point arrays and can be shown
like any other field.

**The four coordinates and their ranges**

| Array | Coordinate | Range |
|---|---|---|
| `uvc_ven` | transventricular, ν | −1 in the LV, +1 in the RV |
| `uvc_z` | apicobasal | 0 at the apex to 1 at the base, in each ventricle |
| `uvc_rho` | transmural | 0 on the endocardium to 1 on the epicardium |
| `uvc_phi` | rotational, radians | 0 mid-septum, ±π/2.5 at the septal edges, ±π/2 at the LV–RV junctions, ±π on the LV free wall; in the RV, 0 mid free wall and ±π/2 at the junctions. Positive is anterior |

Some conventions worth knowing:

- `uvc_z` starts again at 0 in the RV, from the RV's own apex.
- In the LV, the outer wall for `uvc_rho` includes the septum, which faces the
  RV. In the RV, `uvc_rho` is 0 on the free-wall endocardium.
- A node shared by LV and RV elements belongs to the LV.

## What it needs

The item is enabled on a tetrahedral volume that has:

- **surface labels**, from [Surface labels](surface-labels.md). The same
  labelling serves the fibres, so fibres and coordinates are built on the same
  surfaces;
- **two element tags**, LV and RV, in an integer element array (`elemTag`). The
  LV tag is the one whose elements touch the LV endocardium.

Otherwise the item is greyed out, and its tooltip says what is missing.

## The window

The window is **not modal**, so you can rotate the view and click while it is
open.

**Apices**. Each apex is a short line of nodes across the wall, from an
endocardial node to the epicardial node nearest it.

- **Automatic** (the default): the LV apex starts from the LV-endocardial node
  farthest from the centre of the base; the RV apex from the RV-endocardial
  node farthest from its base rim, measured along the RV endocardium. Both are
  found when the window opens (3.6 s on a 6.2-million-node mesh) and marked in
  the view: LV in gold, RV in cyan. **Compute** uses exactly the apices shown.
- **Pick LV apex** / **Pick RV apex**: click that ventricle's endocardium in
  the 3D view; the nearest endocardial node starts the line. Press the button
  again to stop picking.
- **Automatic** forgets picked apices.

The LV apex is also the automatic apex of [Generate fibres](fibres.md), which
holds the epicardial end of the same line.

**Compute** runs in the background and writes the four arrays. The report gives
the LV and RV tags, the apices used, the angles of the septal edges and
junctions, and the time of each distance computation.

## How the coordinates are computed

1. **Surfaces.** From the labels and the tags: the RV septum (the RV-endocardial
   faces of LV elements), the RV free-wall endocardium, the LV outer wall (the
   boundary of the LV elements, less the LV endocardium and the base), and the
   interface between LV and RV elements.
2. **Distances.** Sixteen geodesic distance fields, each from one of those
   surfaces, the base or an apex, measured through the tissue by solving the
   Eikonal equation with the fast iterative method. Paths along mesh edges would
   be quicker, but their gradients turn by several degrees from one element to
   the next, which matters wherever the coordinates are differentiated.
3. **z and rho** are distance ratios, `g0 / (g0 + g1)`: from the apex and the
   base for `uvc_z`, from the endocardium and the outer wall for `uvc_rho`.
   A node on an endocardium and the outer wall at once, where the wall has no
   thickness (a pinch, or the rim of a hole through the wall), is held by
   neither for `uvc_rho` and takes its value from the tissue around it, as the
   fibre solves treat the same nodes; the report counts them. It still shows as
   a spot on the surface, since no single value suits both walls: **Clean
   volume** reports such nodes, and repairing the mesh removes them.
4. **phi.** An axis frame is built from the apices: the long axis runs from the
   LV apex to the centre of the LV endocardium, and the LV–RV direction points
   at the RV apex. Each element gets an angle around the long axis. The septal
   edges are the angular extremes of the RV septum, the junctions those of the
   LV–RV interface. That splits the LV into six sectors and the RV into two, and
   `uvc_phi` is interpolated across each sector by the distance ratio to its two
   boundaries.

This is the distance-based construction of the reference UVC tool the
pipeline's models were built with, not the Laplace-based one of the 2018 paper.

## Agreement with the reference tool

**ccdaf UVC against the reference tool on 29 pipeline ventricles: median [90th percentile] of the mean absolute difference**

| Apices | `uvc_z` LV | `uvc_z` RV | `uvc_rho` LV | `uvc_rho` RV | `uvc_phi` LV | `uvc_phi` RV |
|---|---:|---:|---:|---:|---:|---:|
| the reference's own | 0.009 [0.019] | 0.004 [0.005] | 0.005 [0.006] | 0.008 [0.008] | 0.007 [0.010] | 0.010 [0.019] |
| automatic | 0.017 [0.026] | 0.024 [0.032] | 0.005 [0.006] | 0.008 [0.009] | 0.080 [0.143] | 0.106 [0.263] |

`uvc_ven` agreed everywhere. Given the same apices, the coordinates agree to
within about half a degree in `uvc_phi`. The automatic apices differ: the
reference tool chooses its apices by a rule it does not document, and no rule
tried reproduced it. The apices set the LV–RV direction, and with it where
`uvc_phi` is 0, so where matching a reference model matters, pick its apices.

## Time and memory

**Compute UVC on two ventricles, automatic apices, 24 threads**

| Mesh | Nodes | Elements | Time | Peak memory |
|---|---:|---:|---:|---:|
| 2.9-million-element ventricle | 637,000 | 2.9 million | 47 s | — |
| 400 µm ventricle | 6.2 million | 35 million | 8.5 min | 22 GB |

On the 400 µm mesh each of the sixteen distance fields takes about 17 s; most
of the rest, and the memory peak, goes into finding the surfaces from the
mesh's 140 million element faces. The distance solver uses every core; on a
four-core laptop expect the distance part to take about four times longer
(measured: 6.1 s on 4 threads against 1.6 s on 24, for one field on the
637,000-node mesh).

## Labels that cannot be right

The epicardium is the outside of the heart, so it must spread farther from the
centre than either endocardium. Labels where it does not are refused with that
reason, rather than turned into coordinates that look plausible and are wrong.
Three of the 32 pipeline ventricles above had their epicardium and RV
endocardium swapped this way; labelled again with
[Surface labels](surface-labels.md), they compute normally.

## What it does not do

- The coordinates are not written to the CARP export.
- There is no special treatment of the bridges between valve openings
  (`uvc_z` above 1 in the 2018 paper).
- Compute again after the mesh or its labels change: the arrays are not
  updated or removed on their own.
