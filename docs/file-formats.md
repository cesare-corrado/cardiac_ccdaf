# File formats

## Region labels (`elemTag`)

Tags live in the mesh's `elemTag` **cell** array:

![Region labels](assets/img/region-legend.png){ width="320" }

| Label | Region |
|---|---|
| 11 | LSPV |
| 13 | LIPV |
| 15 | RSPV |
| 17 | RIPV |
| 19 | LAA |
| 1  | body (background) |
| −1 | unassigned (internal; becomes body on *Accept tagging*) |

## Meshes — `.vtk`

Triangular surfaces are read/written through `ccdaf.io.vtkfunctions`.

- **Encoding:** legacy ASCII, legacy binary, and XML are supported. The project
  default on save is ASCII.
- **No-data / NaN:** binary and XML `.vtk` carry `NaN` natively. The legacy
  **ASCII** reader cannot parse a `nan` token — the first one trips the stream
  and every value after it misreads — so CCDAF writes a sentinel on ASCII export
  and restores it to `NaN` on import. If a downstream tool (e.g. ParaView) must
  read the file, prefer **binary** to keep `NaN` intact.

## Seeds — `.json` / `.pkl`

- **`.json`** — human-readable `{"seed_LA": {name: [x, y, z], ...}}`. Stores
  **coordinates only**, no vertex ids, so seeds reload onto a clipped or refined
  mesh (each snapped to the nearest surface point). Every seed type uses the
  same layout under its own key — **landmarks_LA_UAC** under
  `landmarks_LA_UAC`, and so on.
- **`.pkl`** — the seeds alongside a Carto-dict surface.

!!! note "The `seeds` key was renamed"

    The six-seed set is written under `seed_LA`; it used to be `seeds`. Both
    spellings are accepted on read (`seed` too), and CCDAF says in the status
    bar when it read a file under the old key. Saving writes `seed_LA`, so a
    file converts the first time you save it.

## Session bundles — `.pkl`

A **File → Save data** bundle packs the surface, tagging, every completed point
set (each under its seed type's own key) and electrodes together, so a session
round-trips in one file (`read_bundle` / write path in
`ccdaf.core.eam_loader`).

A set appears **only when it has been completed**. That is load-bearing, not
tidiness: nothing in the file says which anatomy it holds, so an absent key has
to mean absent. A ventricular mesh therefore carries no `seed_LA` key, and an
atrial one carries no right-atrial key, rather than each carrying an empty one
that a reader would take at face value. A seed type with no points defined
(`seed_RA` today) is never written at all.

## EAM export

`ccdaf.core.eam_export` writes:

- **Binary** (`EXPORT_BINARY`) — a pickled `{'surface', 'electrodes'}`
  dictionary, as the reference Carto pipeline dumps.
- **VTK** (`EXPORT_VTK`) — the surface with every field, for ParaView.
  Electrodes are separate geometry and are **not** embedded in the VTK.

## Segmentation images — `.nii` / `.nii.gz`

Label images are read/written with SimpleITK. Surfaces are reconstructed with
marching cubes; see [Segmentation](user-guide/segmentation.md).

**Orientation.** A NIfTI header records which anatomical direction each voxel
axis grows towards — `LPS`, `RAS`, `LAS` and so on. CCDAF works internally in
`LPS`, so a volume in any other orientation is **re-indexed on load**: the voxel
array is permuted and flipped, and the origin moved to match. Nothing moves in
world coordinates — every label keeps its physical position, and so does the
surface built from it — but it lets the slice views, the brush and the plane
gizmo place voxels correctly. The status bar names the change, e.g.
*"Loaded PVeinsLabelled.nii (reoriented RAS → LPS)"*.

**Saving** writes the orientation the file arrived in, so a `RAS` volume loaded
and edited is saved back as `RAS` and the rest of your pipeline sees the layout
it expects.

Volumes whose voxel axes are **oblique** to the world axes cannot be squared up
by re-indexing — that would need resampling. CCDAF warns on load and the slice
views stay approximate; resample such a volume to an axis-aligned grid before
editing it.
