# Keyboard & mouse

## 3D view — navigation

| Action | Result |
|---|---|
| Left-drag | Rotate (trackball) |
| Scroll wheel | Zoom |
| Right-drag | Zoom / dolly |
| Middle-drag | Pan |

## Mitral clip widgets

A drag that starts **on** the sphere or plane drives the widget, not the camera.

| Input | Result |
|---|---|
| Sphere: **left-drag** | Move it |
| Sphere: **right-drag** | Resize it |
| Plane: **left-drag the arrowhead** | Tilt the plane |
| Plane: **left-drag the rim** | Slide it along the arrow |
| Plane: **left-drag the centre ball** | Shift it sideways |
| Plane: **middle-drag** | Move it freely |
| Plane: **right-drag** | Resize the widget's box only — the clip is unchanged |

## Picking & committing

| Input | Result |
|---|---|
| **Left-click** | Place a seed (seed selection only) |
| **`X`** | Pick — add a triangle to the selection batch, or drop a point for a snake tool |
| **`C`** | Commit the manual-correction selection batch |

!!! info "Who owns `X`"
    `X` is the *pick* key for every tool that picks with the keyboard, and they
    cannot all listen at once:

    - **Manual correction — selection mode:** `X` adds the triangle under the
      mouse to the batch.
    - **Manual correction — snake / Clipping — PV contour:** `X` drops a
      geodesic point.

    The **Clipping active** checkbox decides whether the key belongs to
    clipping or to manual correction, so only one tool responds at a time.
    Seed selection is the exception: seeds are still placed by left-click.

    Committing never shares a key with picking: `C` commits the selection
    batch, **Commit snake** commits the snake, and clipping has its own
    **Close & Clip PV** / **Apply clip**.

## Per-tool buttons

Undo is per tool, not global:

- **Seeds:** Undo / Reset.
- **Manual correction:** Undo last edit (up to 3 levels); Undo last point / Clear
  (snake).
- **Clipping:** Undo last point (PV); Reject / revert clip.
- **Segmentation:** Undo (last segmentation edit).

Every hoverable control has a **tooltip** — hover to see what it does.
