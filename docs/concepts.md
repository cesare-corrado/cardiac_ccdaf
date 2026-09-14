# Concepts & methods

The non-obvious algorithms behind the panels. Each maps to a module under
`ccdaf.core` / `ccdaf.interaction`.

## Seed geometry {#seed-geometry}

`ccdaf.core.seed_geometry.SeedGeometryResolver` turns a raw pick into a
deterministic vertex and validates it:

- **Snap** — nearest mesh vertex by a KD-tree (deterministic for a fixed point
  order).
- **Duplicate guard** — reject a pick within ~2% of the bounding-box diagonal of
  an existing seed.
- **PV prior** — a pulmonary-vein seed must sit on a *protrusion*: its geodesic
  distance from a body anchor (the vertex nearest the mesh centroid) must exceed
  a robust threshold. Body-wall picks fall below it and are rejected.

## Geodesic tagging & the snake {#geodesic}

Region tagging and the manual "snake" both work on the mesh **1-skeleton** (its
edge graph):

- **Automatic tagging** (`ccdaf.core.region_tagger`) grows each region outward
  from its seed along geodesics, capped by *radius factor × median edge length*.
- **The snake** (`ManualEditor` / `ClippingTool`) builds an **open geodesic**
  through user-dropped points, growing **bidirectionally** — each new point
  extends whichever endpoint (head or tail) reaches it by the shorter geodesic,
  so the path never doubles back. Committing tags the triangles incident to the
  path (manual) or clips inside the closed loop (PV contour).

## Picking {#picking}

All interactive picking uses VTK's **hardware (z-buffer) picker**
(`enable_point_picking(picker="hardware")`): it returns the front-most
*visible* surface hit, so a pick can never bleed through to an occluded
back-wall vertex.

- **Vertex tools** (seeds, snakes) snap the hit to the nearest vertex.
- **Cell tagging** maps the hit point to its triangle with
  `mesh.find_closest_cell` — the hit lies on a triangle *face*, so it resolves
  to exactly one cell (no vertex-sharing ambiguity).

## Electrode displacement {#electrode-displacement}

When the wall moves under a loaded mapping, electrodes are re-warped to preserve
their **signed distance to the wall** — an electrode 2 mm off the tissue stays
2 mm off it, on the same side. The default (`displace_electrodes_by_distance` in
`ccdaf.core.eam_loader`) needs **no** vertex correspondence between the two
surfaces, so it works even after a marching-cubes round trip.

Each electrode moves along the new wall's normal until the new wall is as far
away as the old one was:

```text
x  ←  x + (d0 − d1(x)) · ∇d1(x)
```

where `d0` is the (fixed) distance to the old wall, `d1` the distance to the new
wall, and `∇d1` its gradient. Because a signed distance field satisfies the
eikonal property `|∇d| = 1`, one step lands on the target distance to first
order; the residual is second-order (wall curvature) and negligible for the
sub-millimetre motion smoothing and reconstruction produce.

There is **no convergence loop** — the update `(d0 − d1)·∇d1` vanishes on its
own as the residual goes to zero, so a fixed `EAM_SDF_ITERATIONS = 3` is a safe
over-count:

![Electrode displacement convergence](assets/img/electrode-convergence.png){ width="440" }

Properties worth knowing:

- **Bounded** — no electrode moves further than the wall itself shifted.
- **Local by closest point** — an electrode tracks the piece of wall it is
  nearest to; a uniform wall offset moves every electrode with it.
- **Sign consistency** — both surfaces are re-wound outward first
  (`_outward`), so a Carto mesh and a flipped marching-cubes surface agree on
  which side is which, and electrodes are never driven through the wall.

Where this fires: the **Smooth** post-processing stage, **Clipping**, and the
segmentation **Export to VTK** round trip. It is a no-op when no mapping is
loaded. The RBF method `displace_electrodes` survives as a fallback for a
degenerate surface where the distance field faults.

## Healthy-tissue estimate {#healthy-tissue-estimate}

`ccdaf.core.tissue_property.estimate_healthy` supplies the healthy mean and SD
that **Actions → Assign tissue property** classifies by. The criterion is that
of Chen et al. (2015), as restated by Mendonca Costa et al. (2019): "Scar and
BZ were segmented as the regions with signal intensity above 3 and 2 SD from
the mean signal intensity within healthy myocardium, respectively". Those
studies take the mean and SD from a remote region an observer draws on the
image. A mesh has no observer, and what is healthy is the output, so it is
estimated:

1. **Seeds**: elements at or below the 25th weighted percentile of the field.
2. **Grow**: from the seeds, through elements sharing a face (an edge, on a
   surface), keep those at or below `mean + m·SD` of the region so far, with
   `m = 3`. Recompute, and grow again from the seeds, so the region can shrink
   as well as grow.
3. **Stop** when a round changes the region by less than 0.1% of the tissue.

The final region still contains the elements later labelled border zone (2 to
3 SD). That is intended: healthy tissue has its own tail there, and trimming it
shrinks the SD.

### Why the limit is 3 SD

On exactly normal healthy tissue with no scar, the loop settles as follows.

**Where the refit settles on scar-free, normally distributed tissue, by growth limit**

| m | Healthy tissue kept | SD found ÷ true SD | Labelled border zone | Labelled scar |
|---|---:|---:|---:|---:|
| 2.0 | 95.8% | 0.911 | 3.78% | 0.42% |
| 2.5 | 99.2% | 0.973 | 2.53% | 0.19% |
| 3.0 | 99.85% | 0.993 | 2.23% | 0.15% |
| true mean and SD | 100% | 1 | 2.14% | 0.13% |

The last row is not zero: the definition itself labels part of healthy
tissue's noise tail. At `m = 2` the loop trims real healthy tissue at every
round, the SD shrinks, and false scar roughly triples. On the example ventricle
`m = 2` gives 22.1% scar where `m = 3` gives 13.9%.

### Why grow from seeds, not stop at an edge

Stopping growth at a *jump* in intensity sounds natural, and it fails where it
matters. A border zone is by definition the gradual part, each step up it is
no larger than noise, and growth climbs straight into the scar. On synthetic
ventricles with a gradual border zone, a jump rule found scar with Dice 0.10
(10% infarct) and 0.00 (30% infarct); the `mean + 3·SD` limit found it with
Dice 1.00 in both.

The seed percentile is a safety margin rather than a tuning knob. While the
seeds stay in healthy tissue the answer does not depend on it: on the example
ventricle seeds at 10, 25, 35 and 50% give identical results. Once they reach
into scar the estimate takes the whole wall as healthy. With a 60% infarct (30%
of the wall healthy), seeds at 10, 25 and 35% recovered the truth and seeds at
50% found no scar at all.

### Weighting

Every statistic is weighted by element volume, or area on a surface: the
integral over the tissue, which is what counting equal voxels does on the
image. Counting elements instead lets refinement move the thresholds. On a
probability field of the segmentation example, a 3 SD cut came out at 0.867
on the original mesh and at 0.927 counted per element on a version refined
unevenly (element-size coefficient of variation 1.14); weighted by volume it
was 0.865 on the refined mesh.

### Limits

- Heiberg et al. (2022) found the n-SD method "unreliable for infarct
  quantification due to high variability which depends on different placement
  and size of remote ROI, number 'n-SD', and image signal properties". An
  automatic reference removes the placement choice, not the other two.
- Masked tissue among the ticked tissue skews the estimate before it breaks
  it. On the example, 5% of exact zeros lowered scar from 14.0% to 12.0%; at
  25%, the seed percentile, every seed is a zero, the SD is 0 and Apply
  refuses. One value covering more than 5% of the tissue triggers a warning.
- The synthetic tests used normally distributed healthy tissue. Real healthy
  tissue is skewed: 0.76 on the example's estimated healthy region.

**References.** Chen et al., *Heart Rhythm* 2015,
[doi:10.1016/j.hrthm.2014.12.020](https://doi.org/10.1016/j.hrthm.2014.12.020);
Mendonca Costa et al., *Heart Rhythm* 2019,
[PMC6774764](https://pmc.ncbi.nlm.nih.gov/articles/PMC6774764/); Heiberg et
al., *J Cardiovasc Magn Reson* 2022,
[PMC9639305](https://pmc.ncbi.nlm.nih.gov/articles/PMC9639305/).
