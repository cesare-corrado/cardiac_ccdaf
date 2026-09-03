# Troubleshooting

## The app won't start / crashes on launch

- **Wrong VTK stack.** CCDAF needs VTK ≥ 9.6 / PyVista ≥ 0.48. Check:
  ```bash
  python -c "import vtk, pyvista; print(vtk.vtkVersion.GetVTKVersion(), pyvista.__version__)"
  ```
  Base Anaconda's VTK 9.0.3 is unsupported — use the `ccdaf` conda env.
- **Font-cache crash in `libfontconfig`.** Ensure `XDG_CACHE_HOME` points at a
  private directory (the conda env sets this automatically).

## Picking selects the wrong point or triangle

CCDAF uses a hardware/z-buffer picker, which picks the visible surface. If picks
still feel off on a **HiDPI / Retina** display, that is the classic device-pixel
scaling issue — verify on a standard-DPI display. See
[Concepts → Picking](concepts.md#picking).

## A pulmonary-vein seed is rejected

PV seeds must sit on a protrusion (the anatomical prior). Click **inside** the
vein, further from the atrial body, not on the ostial rim. See
[Concepts → Seed geometry](concepts.md#seed-geometry).

## Electrodes didn't move after editing the surface

Electrode displacement only runs when a **mapping with electrodes** is loaded,
and only on the paths that commit a surface change:

- **Post-processing:** only the **Smooth** stage moves them — **Clean**'s quality
  smoothing does not.
- **Segmentation:** **Export to VTK** moves them; **Update 3D** is a preview and
  does not.

See [Concepts → Electrode displacement](concepts.md#electrode-displacement).

## The 3D surface sits away from the slice planes

A label image whose NIfTI header is not `LPS` used to draw its slice planes in
one place and its surface in another — often more than 100 mm apart, since the
two are mirrored about the image origin. CCDAF now re-indexes such a volume on
load and says so in the status bar. If they still disagree, check for the
**oblique segmentation** warning: voxel axes tilted relative to the world axes
need resampling to an axis-aligned grid. See
[File formats](file-formats.md#segmentation-images-nii-niigz).

## A tool switched itself off when I started another one

Seed selection, **selection mode**, **snake tag** and the **PV contour clip**
all drive the same surface picker, and only one may hold it. Starting any of
them stops whichever had it and says so in the status bar — the panel's
controls go back to "off" so nothing looks live while another tool has the
mouse.

Nothing is lost when this happens. Seed selection is *paused*, not reset:
clicking **Start seed selection** again carries on from the seeds you had,
at the next one in the order — the status bar names it. (Starting over is what
**Reset seeds** is for; after a reset, **Start** begins from the first seed as
usual.) A pending triangle selection survives too, and tags are untouched.

Only a **contour in progress is abandoned**, since the clip needs its picks
to mean anything; restart it with **Start**.

## The clipping controls stay greyed out

Clipping needs an accepted tagging, and ticking **Clipping active** now
supplies one — silently when accepting changes no tag, otherwise after asking,
since accepting commits any pending selection batch and gives every unassigned
triangle the body label. Declining that question withdraws the tick, so a
ticked box with dead controls is not a state you can reach; if the panel is
grey, the box is unticked. With no mesh loaded the tick is refused outright and
the status bar says so.

A mesh that arrives already tagged skips the question entirely: clipping is
available from the moment it loads.

## `X` does nothing

`X` is owned by one tool at a time. For clipping, tick **Clipping active**; for
manual correction, make sure clipping is **not** active and the relevant mode
(selection or snake) is on. See [Keyboard & mouse](shortcuts.md).

## A `.vtk` opens blank in ParaView

An ASCII `.vtk` containing `NaN` values cannot be read by the legacy ASCII
reader ParaView uses. Re-export as **binary** VTK. See
[File formats](file-formats.md).
