# EAM integration — verification against real Carto data

Everything in `tests/` runs on synthetic meshes, so it runs anywhere. The
checks below were run against real Carto exports, which are **not** in the
repository and cannot be. They exist here so they can be re-run — and so
they can become the nightly suite when it runs on a private mirror with a
local runner.

Each section states what was checked, how, and the number that came back.
Where a number is quoted it is what the current code actually produced; a
re-run that disagrees means something changed and is worth understanding
before it is dismissed.

## Environment

This matters more than it looks. An entire session's testing was once
invalidated by using the wrong interpreter.

```bash
PYTHONPATH=src /home/cc14/Libraries/anaconda3/envs/ccdaf/bin/python …
```

`envs/ccdaf` is Python 3.14 / VTK 9.6.2 / pyvista 0.48.4 / scipy 1.18. The
**base anaconda** `python` is Python 3.9 / VTK 9.0.3 / pyvista 0.45 and
behaves differently — the `fmt="%d"` scalar-bar crash reproduces only under
VTK 9.6.2, and base anaconda bundles networkx by accident where the ccdaf
env does not.

Headless: `pv.Plotter(off_screen=True)` works. `CCDAF()` constructs only
with **DISPLAY unset**:

```bash
env -u DISPLAY QT_QPA_PLATFORM=offscreen XDG_CACHE_HOME=<scratch> …
```

With DISPLAY set plus the offscreen plugin, VTK talks to X directly and
raises BadWindow. `XDG_CACHE_HOME` isolation is load-bearing (fontconfig
segfault; see `environment.yml`).

**Anything that renders cannot be driven headless here.** `_set_segmentation`
renders slices; with no GPU the shader compile fails and VTK aborts the
process with `std::bad_array_new_length`. To exercise the segmentation round
trip, set `_seg_sitk` / `_seg_array` / `_seg_origin` / `_seg_spacing`
directly and skip `_set_segmentation`. Extracting the segmentation logic out
of the GUI class removes this obstacle entirely — that is the main argument
for doing it.

## Data

```
/home/ccorrad1/data/EGM/patient_data/Carto/SN00{1..5}/*/*.mesh
/data/cc14/Amsterdam-EGM-MRI/data/Q110001/0_data/Q110001_EAMexport/Carto_Q110001/*.mesh
```

17 mappings across SN001–SN004 carry fields; `1-NONE` (SN003/NIEDERER,
20979 points, 2215 electrodes) and `1-TACHY 335MS` (20352 points, 6720
electrodes) are the two used most below.

Reference pipeline being replaced:
`/home/ccorrad1/data/EGM-MRI/export_Amsterdam_Carto.py` plus
`Carto_tools/{udfunctions,vtkfunctions}.py`. Its downstream consumer is
`post_CemrgApp_postprocessing.py`.

## Reading

- **A Carto export arrives closed.** Raw boundary edges: `1-NONE` 8 in 2
  loops, `1-TACHY 335MS` 21 in 5, `2-IC AFL 1 259ms` 4 in 1, `1-LA` 16 in 4.
  `fill_holes(4.0)` and `fill_holes(1000.0)` both take every one to 0. There
  are no open PV ostia to preserve; the reference's `hole_size=1000` and our
  `max_hole_size=4.0` are equivalent on this data.
- **No field is ever partially no-data.** Across all 17 mappings, every field
  is either fully valid or fully sentinel. On `1-NONE`, 10 of 13 are entirely
  no-data (including `Impedance` and `Force`, both of which the downstream
  interpolates); `Unipolar`, `Bipolar`, `LAT` are entirely valid. So mixed
  triangles do not occur in this cohort — the renormalising path in
  `field_transfer` is unexercised here, and is kept because Carto's format
  permits per-point invalid data.
- **Map discovery from `.mesh` files** takes 0.089s over 61,284 files, and
  the 8 stems match the study XML's map names exactly.
- `read_carto_mesh_file` handles `1-1-ReTACHY 335MS`, which has no
  `[VerticesColorsSection]`, and `load_carto_electrodes` handles a
  header-only `_car.txt` (`1-2-PACING`, 4 electrodes).

## Downstream compatibility

- `post_CemrgApp_postprocessing.py:154` reads **only** `['electrodes']` from
  the pickle; `['surface']` is never read back.
- Our electrode record keeps the raw sentinel (masking is on `VertexColors`
  only). On `1-NONE`, 747 of 2215 electrodes carry `LAT ≤ -1000`, so the
  `LAT > -1000` filter at line 158 keeps 1468 — identical to a reference
  pickle.
- The vtk is read at line 276 and fed to
  `pointDataInterpolation(…, ['LAT','Bipolar','Unipolar','Force','Impedance'])`.
- `gpmi.set_data` (`qulati/gpmi.py:89`) computes `y_mean`/`y_std` and scales
  `(y - y_mean) / y_std`, reversing it on output. The GP is therefore
  invariant to a constant offset — which is why the reference's `LAT - LMIN`
  shift has no numerical consumer.

## Electrodes

Both on `1-TACHY 335MS` (20352 points, 6720 electrodes), same Taubin
smoothing, correspondence available so the RBF is on its home turf:

```
TIME   rbf 15.13s | distance 1.13s        (13.4x)
MOVE   rbf mean 0.387 max 1.521 | distance mean 0.315 max 1.453
AGREE  the two answers differ by mean 0.297 max 1.379
WALL   rbf      distance-to-wall error: mean 0.1905 max 1.3595 | wrong side: 15
       distance distance-to-wall error: mean 0.0000 max 0.0000 | wrong side:  0
```

The decisive figure is not the speed. Taubin's vertex motion decomposes as
mean normal 0.293 against mean tangential 0.208 — **54% of what the RBF fits
is the mesh re-parameterising itself**, not the wall moving. An electrode has
no identity on the surface to slide with.

Distance-field cost scales with electrode count, not mesh size: BSP build
~20ms, ~145ms per distance sweep, ~150ms per gradient sweep, three Newton
steps.

Robustness: the signed distance field degrades gracefully with holes. Cutting
away 20% of `1-NONE` still gives 0 sign flips in 2215 electrodes; at 40%
(470 boundary edges) only 4 flip. A closed-mesh precondition for the fallback
would therefore never fire, which is why the fallback triggers on a fault
rather than a threshold.

## Segmentation round trip

`1-NONE`, voxelised at 1mm, Gaussian σ=1, marching cubes.

- **Fidelity**: 20979 → 22796 points, no correspondence. Every rebuilt vertex
  within **0.821mm** of the source wall (mean 0.103, p99 0.449). **0 of 22796**
  beyond 1mm — so a faithful round trip never trips a 2mm guard.
- **Guard fires only on edits**: after a `BinaryMorphologicalClosing` with a
  radius-4 ball (+3045 voxels), 797 vertices (3.69%) sit beyond 1mm and 145
  (0.67%) beyond 2mm. Those 145 would otherwise inherit LAT spanning
  −52.4..119.5 ms against a true range of −53..138 — indistinguishable from
  measurement. With the guard they are NaN.
- **The guard's scale**: `2 × max(voxel spacing, median source edge)`, not
  `2 × voxel spacing`. The original rule was wrong and a real run found it.
  What moves the wall is the smoothing and the morphology, in millimetres —
  the voxel size does not — so refining the grid shrank the threshold past a
  drift that had not moved, and the *better* reconstruction discarded more
  data. On `1-Pre-PVI SR ANALYSE` (Q110001, 13579 points, median edge 1.29mm,
  vertices 0.87mm apart) at 0.5mm voxels, σ=1.5, morphological opening and
  closing at r=2:
  ```
  guard 2 x 0.5 = 1.0mm  -> 283 of 105526 NaN, in 22 clusters, largest 38
                            their drift 1.00..1.49mm
                            each within 1.67mm of a real measurement
                            i.e. inside one source triangle
  same run at 1mm voxels -> 0 NaN
  guard 2 x max(0.5, 1.29) = 2.57mm -> 0 NaN, and a 4mm dilation still 100%
  ```
  At a 1mm voxelisation of a Carto shell the voxels win and the rule is
  unchanged, so every number above still holds.
- **End to end, both flip pairings, with the closing applied**:
  ```
  new mesh 21614 pts | point fields 14 | cell ['elemTag']
  LAT NaN 145/21614 | range -53.0..136.8
  wall offset preserved |d1-d0|: mean 0.0000 max 0.0000
  same side of the wall: 2215/2215
  Transferred 13 point and 1 cell fields; 145 of 21614 vertices sit further
    than 2 from anything measured and were left as no-data.
  ```
  These counts are the **flip=False pairing's**. The flip=True pairing
  legitimately differs (a 2026-07 re-run: 21692 pts, LAT NaN 138): negating
  x/y re-anchors the voxel grid on the negated bounds — a different
  sub-voxel offset — so it is not a mirror of the same rasterisation. Both
  are correct; compare a re-run against the pairing it actually ran.
- **Orientation**: `_segmentation_to_polydata` calls `FlipNormalsOn()`, which
  inverts the signed distance — the sign follows triangle **winding**, not the
  stored `Normals` array. On a sphere: `[-9.95, 10.02]` becomes
  `[9.95, -10.02]`. Unhandled, every electrode is driven through the wall
  (9mm moves instead of 0.15mm). `_outward()` normalises both surfaces.
  Note the MIRTK flip itself is a **180° rotation about z** (determinant +1),
  so it does *not* reverse winding.
- **Voxelisation padding** is symmetric (one voxel per side). It changed
  nothing measurable — mean 0.1030 → 0.1026, max 0.8210 → 0.8207.

## Export

- pickle: `['surface', 'electrodes']`; surface keys `['X','Tri',
  'VertexColors','ColorsIDs','ColorsNames']`. On `1-NONE` after a round trip:
  `X (22796,3)`, `VertexColors (22796,13)`, 13 names, 13 ids, columns matching
  names, `LAT` reading back at its own id.
  A 3-component field must never enter `VertexColors` — it contributes three
  columns under one name and shifts every later `ColorsID` onto the wrong
  column, so `Unipolar` reads back as a normal's y-component.
- Save with defaults writes cell `['elemTag', 'Normals']`, Normals computed
  and unit-length across all cells. Ticking `LAT`+`Bipolar` keeps both with
  ranges intact.
- `writevtk` defaults to `binary=False`; `mesh.save()` defaults to binary.

## Known, accepted

- **VTK 9.6 offloads a few filters to Viskores and reaches for CUDA.** Where
  the PTX does not match the driver every such call fails
  (`cudaErrorUnsupportedPtxVersion`), prints a screenful, and redoes the work
  on the CPU — correct, but `cell_data_to_point_data` on a 42k-cell shell
  costs 0.427s against 0.002s with the overrides off, and clipping calls it
  every pass. Disabled at the package root; `CCDAF_VTK_ACCEL=1` restores it.
- `vtkDelaunay2D`'s "edge not recovered, polygon fill not possible" is the
  hole filler's constrained triangulation failing on a loop.
  `_delaunay_covers_all_loop_edges` catches it and falls back to ear-clip,
  then a centroid fan. Benign, and VTK prints it regardless.
- Decimate followed by 40 smoothing iterations loses 4–7% of a decimated
  shell's volume. Reported rather than prevented; volume is a poor metric on
  a Carto shell anyway.
- `_define_image_from_mesh`'s upper pad is ~1.5 voxels against the lower
  pad's 1, because `dims = ceil(extent/spacing)` rounds up.
- Tagging and clipping have **not** been exercised on a Carto mapping. Their
  stack is untouched by this work, and `elemTag` now survives a round trip,
  but that is an expectation rather than a result.
