# Loading & mesh info

## Loading a mesh

**File → Load** opens a `.vtk` triangular surface. On load, CCDAF:

- reads the surface via VTK (`ccdaf.io.vtkfunctions.readvtk`);
- restores no-data sentinels to `NaN` for legacy **ASCII** `.vtk` files (whose
  reader cannot parse a `nan` token — see [File formats](../file-formats.md));
- initialises the `elemTag` cell array to **body** (`1`) if the mesh has none.

The mesh must be **purely triangular**; non-triangular input is rejected.

## Mesh info panel

The **Mesh info** panel summarises the working mesh:

- point and cell counts;
- the point-data **fields** present (e.g. Carto voltage/LAT maps);
- which region **labels** are currently in `elemTag`.

It updates whenever the mesh changes — after tagging, clipping, or a
post-processing pass.

## Saving

**File → Save** writes the mesh back out, carrying its tags, the current seeds,
and — when an EAM mapping is loaded — the electrodes, so the file reloads with
everything in place. The save dialog lets you choose the VTK encoding; the
project default is ASCII. See [File formats](../file-formats.md) for what each
container preserves.

!!! note "Bundles"
    A **File → Save** to a `.pkl` bundle stores the surface, tagging, seeds, and
    electrodes together; it is the format that round-trips a full session.
