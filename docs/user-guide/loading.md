# Loading & mesh info

## Loading a mesh

**File → Load data** opens a `.vtk` triangular surface, a `.vtk` tetrahedral
volume, a `.pkl` session bundle, or a `.nii`/`.nii.gz` segmentation — the
reader follows the extension. The same file can be named on the command line
(`ccdaf path/to/file`), which opens it as the window comes up.

Loading a mesh, CCDAF:

- reads it via VTK (`ccdaf.io.vtkfunctions.read_dataset`);
- restores no-data sentinels to `NaN` for legacy **ASCII** `.vtk` files (whose
  reader cannot parse a `nan` token — see [File formats](../file-formats.md));
- initialises the `elemTag` cell array to **body** (`1`) if the mesh has none.

## Surfaces and volumes

A file holding **tetrahedra** is kept as a volume. Its boundary surface is
derived and is what you see, pick and colour, but the volume is the working
mesh: it is what gets saved, and it keeps every field the file carried,
including a per-element fibre direction.

What a mesh *is* comes from its cells, not from the file's dataset type. A
triangular surface exported as an unstructured grid is a surface and behaves
exactly as it always did.

The **Mesh info** panel says which you have: for a volume, *Npt* and *nElem*
count the volume's points and tetrahedra, an extra row gives the boundary
triangle count, and *inverted* counts tetrahedra whose nodes wind the other
way — harmless to view, and normalised before any remeshing sees them.

!!! note "What a volume can and cannot do"

    Seed selection, tagging, manual correction and clipping all act on a
    surface. On a volume they are switched off, each saying so.

    Reshaping a volume is done by the volumetric branch of
    [Mesh post-processing](post-processing.md#volumes), which adapts the
    tetrahedra directly and carries the labels, fibres and point fields onto
    the result.

Surfaces must be **purely triangular** and volumes purely **tetrahedral**;
anything else is rejected with a message rather than partly handled.

## Mesh info panel

The **Mesh info** panel summarises the working mesh:

- point and cell counts;
- the point-data **fields** present (e.g. Carto voltage/LAT maps);
- which region **labels** are currently in `elemTag`.

It updates whenever the mesh changes — after tagging, clipping, or a
post-processing pass.

## Saving

**File → Save data** writes the mesh back out, carrying its tags, the current seeds,
and — when an EAM mapping is loaded — the electrodes, so the file reloads with
everything in place. The save dialog lets you choose the VTK encoding; the
project default is ASCII. See [File formats](../file-formats.md) for what each
container preserves.

!!! note "Bundles"
    A **File → Save data** to a `.pkl` bundle stores the surface, tagging, seeds,
    and electrodes together; it is the format that round-trips a full session.

## Unsaved changes

Anything that changes the working data — placing seeds, tagging, a manual
edit, a clip, a post-processing pass, a segmentation edit — marks the session
*modified*. **File → Close** and quitting (**File → Quit**, or the window's
close button) then ask before throwing that away:

- **Save** writes whatever is unsaved. With a segmentation open, that is the
  volume first (**Segmentation → Save segmentation…**) and then the mesh
  (**File → Save data**) — two dialogs in that order, since the volume is what
  you are looking at. Backing out of either cancels the close too, so nothing
  is lost by accident.
- **Discard** closes without saving.
- **Cancel** leaves everything as it was.

Loading, saving or closing clears the flag: straight after any of them there
is nothing to warn about.
