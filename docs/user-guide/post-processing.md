# Mesh post-processing

The **Mesh post-processing** panel exposes `ccdaf.core.mesh_postprocessor.apply`
as a set of togglable stages, run in order on the working mesh. Enable the
stages you want with their checkboxes and set their parameters.

## Decimate

Reduce the point count.

- **target points** — the point count to anneal down to.
- **anneal iters** — maximum simulated-annealing sweeps used to redistribute the
  decimated vertices.

## Refine

Adaptively subdivide triangles whose longest edge exceeds the target.

- **edge len** — target maximum edge length; longer edges are split. `0` uses
  the mesh's current median edge length.

## Clean

Merge duplicate points, drop disconnected points, remove non-manifold and
degenerate cells, orient normals, and smooth low-quality triangles while
preserving the region labels.

- **quality threshold** — triangles below this radius-ratio quality are smoothed
  (`1.0` = equilateral, `0.0` = no quality smoothing).
- **smooth iters** — max Laplacian sweeps over bad-triangle vertices; exits early
  once no triangle is below the threshold.
- **Relaxation factor** — how far each point moves toward its neighbours' mean.

## Fill holes

Close boundary loops up to a maximum size (absolute length, mesh units).
Openings larger than the threshold stay open, preserving genuine anatomical
openings (PV ostia, mitral valve).

## Smooth

Smooth the **whole** surface (reshapes the wall — unlike Clean's quality
smoothing, which only nudges bad triangles).

- **method** — *Taubin* (`vtkWindowedSincPolyDataFilter`, removes roughness
  without deflating the shell) or *Laplacian* (simpler, shrinks with iterations).
- **iterations**, **passband/relaxation** — strength of the smoothing.

!!! warning "Only the *Smooth* stage moves the electrodes"
    When a mapping is loaded, the **Smooth** stage carries the electrodes onto
    the moved wall. The **Clean** stage's quality smoothing also nudges
    vertices but does **not** trigger electrode displacement. See
    [Concepts → Electrode displacement](../concepts.md#electrode-displacement).
