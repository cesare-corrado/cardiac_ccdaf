# Clipping

Remove the pulmonary-vein cuffs and open the mitral valve.

!!! note "The `X` key belongs to whoever is active"
    Clipping's snake and manual correction both use `X`. Tick **Clipping
    active** to hand the key to clipping; untick it to give it back to manual
    correction. Only one tool owns `X` at a time.

## Getting in

Clipping works on an accepted tagging, and the panel reaches that on its own:

- A mesh that arrives **already tagged** — reloaded from an earlier session,
  say — opens clipping the moment it loads. Starting a session at this step is
  a click on **Clipping active** and nothing more.
- Otherwise, ticking **Clipping active** accepts the tagging for you. If
  accepting would change the tags — a pending selection batch to commit, or
  triangles still unassigned that would become body — you are asked first,
  because that body fill is not on the undo stack. Decline and the tick is
  withdrawn, so the box never sits ticked over greyed-out controls.

**Accept tagging** in the manual-correction panel does the same thing, and is
still the way to finish tagging deliberately.

!!! note "Stray label cells are repaired without asking"
    Both routes reduce each region to a single connected patch, reassigning
    any cell of a label that got stranded away from its main patch — a label
    split in two is rejected on export. This is a repair rather than a choice,
    so it is not part of the question; the status bar reports how many cells
    it moved.

## PV contour (snake)

Clip a pulmonary vein by drawing a closed loop around its ostium:

1. Pick the vein in the **PV** drop-down. The snake is confined to that tag's
   region.
2. **Start PV contour**.
3. Press **`X`** to drop points; a geodesic contour follows on that tag and
   grows bidirectionally.
4. **Undo last point** removes the most recent point.
5. **Close & Clip PV** closes the contour into a loop and clips off the cuff
   inside it.

A point is only taken when the **triangle under the cursor** carries the
selected vein's tag — a press on a neighbouring vein, on the body, or off the
surface altogether is refused, and the status bar says which tag it found.
Points on the rim of the region are refused too: the contour needs vertices
inside the tag to travel on.

<figure markdown="span">
  ![PV contour snake around a vein ostium](../assets/screenshots/clipping_vein.png)
  <figcaption>A PV contour: the geodesic snake tracing the vein ostium.</figcaption>
</figure>

## Mitral valve

Clip the valve with a positionable widget:

Both widgets appear at the **MV seed**, so that seed has to exist first — the
buttons say so if it does not.

- **Mitral: sphere** — an adjustable sphere. **Left-drag** it to move,
  **right-drag** to resize. **Apply clip** removes every triangle whose centre
  falls inside it.
- **Mitral: plane** — an adjustable cutting plane, drawn as an arrow through a
  box with a ball at its centre. **Left-drag the arrowhead** to tilt it,
  **the rim** (where the plane meets the box) to slide it along the arrow, and
  **the centre ball** to shift it sideways; **middle-drag** moves it freely.
  **Apply clip** removes the mitral side.

!!! warning "Drags that start on the widget do not move the camera"
    Right-drag normally dollies the view, but over the plane widget it resizes
    the widget's bounding box — which changes nothing about the clip. Start
    the drag off the widget to move the camera.

<figure markdown="span">
  ![Mitral-valve sphere clip widget](../assets/screenshots/clipping_MV_sphere.png)
  <figcaption>Clipping the mitral valve with the adjustable sphere widget.</figcaption>
</figure>

## Apply / revert

- **Apply clip** commits the pending mitral clip.
- **Reject / revert clip** discards a pending clip, or undoes the last applied
  one, restoring the previous mesh (each clip snapshots the mesh first).

!!! tip "Electrodes follow the clip"
    Clipping changes the surface, so a loaded mapping's electrodes are carried
    across automatically — see
    [Concepts → Electrode displacement](../concepts.md#electrode-displacement).
