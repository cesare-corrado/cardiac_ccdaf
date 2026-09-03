# Getting started

This tutorial walks the **happy path**: from a raw surface mesh to a tagged,
clipped result. Each step maps to a side panel; the full detail is in the
[User guide](user-guide/index.md).

!!! tip "The two shortcuts to remember"
    Over the 3D view, **`X`** picks — a triangle, or a point for the geodesic
    *snake* tools — and **`C`** commits the manual-correction selection batch.
    See [Keyboard & mouse](shortcuts.md).

<figure markdown="span">
  ![The CCDAF main window with a mesh loaded](assets/screenshots/overview.png)
  <figcaption>The main window: the 3D viewport with the side-panel column and menubar.</figcaption>
</figure>

## 1. Load the data

**File → Load data** and pick a `.vtk` surface — or a `.pkl` bundle, or a
`.nii`/`.nii.gz` segmentation; the reader follows the extension. The same path
can be given on the command line: `ccdaf path/to/mesh.vtk`. It appears in the 3D view; the
**Mesh info** panel reports point/cell counts and available fields. Untagged
cells start as *body*.

## 2. Place the six seeds

Open **Seed selection → Start seed selection** and click, in order:

`LSPV → LIPV → RSPV → RIPV → LAA → MV`

Each click snaps to the nearest visible surface vertex and the prompt advances.
Use **Undo** to re-pick the last one. Save them with **Save seeds…** — they are
stored as names + coordinates, so they reload onto a clipped or refined mesh.

## 3. Tag the regions automatically

In **Tagging**, set the per-region radius factors if needed, then
**Run automatic tagging**. CCDAF grows each pulmonary vein and the appendage as
a geodesic region from its seed and writes the labels into `elemTag`.

![Region labels](assets/img/region-legend.png){ width="320" }

<figure markdown="span">
  ![Atrium coloured by region labels after tagging](assets/screenshots/tagged-regions.png)
  <figcaption>The atrium coloured by region after automatic tagging.</figcaption>
</figure>

## 4. Correct labels by hand

In **Manual correction**, choose the active **Label**, then either:

- **Activate selection mode** and press **`X`** over triangles to build a
  batch, then press **`C`** to commit them; or
- toggle **Snake tag**, press **`X`** to drop points, and **Commit snake** to
  tag every triangle along the geodesic through them.

**Fill Holes**, **Smooth active label**, and multi-level **Undo** are here too.
Finish with **Accept tagging** (unassigned cells become body).

## 5. Clip the veins and valve

In **Clipping**, tick **Clipping active** (this hands the `X` key to clipping),
then:

Pick a **Region** (a vein or `MV`) and a **Mode**, then **Start** → place it →
**Apply clip**:

- **Contour:** drop points with `X` around the ostium; **Undo / reset** takes
  the last one back.
- **Sphere / plane:** position the widget (left-drag to move or tilt it,
  right-drag to resize the sphere), or type the centre and radius into the
  boxes below. **Undo / reset** returns it to the region's seed.

**Revert clip** undoes an applied clip. The sphere or plane is remembered, so
pressing **Start** again brings it back where you left it.

Ticking **Clipping active** accepts the tagging if that has not happened yet —
asking first when accepting would change tags — so a mesh reloaded from an
earlier session can start at this step. An already-tagged mesh opens clipping
as soon as it loads.

## 6. Export

**File → Save data** writes the mesh (with tags, seeds, and — if a mapping is
loaded — electrodes). If you close or quit with unsaved changes, CCDAF asks
first — **Save**, **Discard** or **Cancel**. For EAM, **EAM → Export** writes a Carto binary bundle or
a VTK. See [File formats](file-formats.md).

---

Next: the **[User guide](user-guide/index.md)** covers every panel, and
**[Concepts & methods](concepts.md)** explains what happens under the hood.
