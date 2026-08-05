# Clipping

Remove the pulmonary-vein cuffs and open the mitral valve.

!!! note "The `X` key belongs to whoever is active"
    Clipping's snake and manual correction both use `X`. Tick **Clipping
    active** to hand the key to clipping; untick it to give it back to manual
    correction. Only one tool owns `X` at a time.

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

- **Mitral: sphere** — an adjustable sphere; **Apply clip** removes the surface
  inside it.
- **Mitral: plane** — an adjustable cutting plane; **Apply clip** removes the
  surface on one side.

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
