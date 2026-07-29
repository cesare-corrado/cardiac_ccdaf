# Keyboard & mouse

## 3D view — navigation

| Action | Result |
|---|---|
| Left-drag | Rotate (trackball) |
| Scroll wheel | Zoom |
| Right-drag | Zoom / dolly |
| Middle-drag | Pan |

## Picking & committing

| Input | Result |
|---|---|
| **Left-click** | Pick — place a seed, or add a triangle to the selection batch |
| **`X`** | Commit the active tool's batch, **or** drop a point for a snake tool |

!!! info "Who owns `X`"
    `X` is shared by several tools that cannot all listen at once:

    - **Manual correction — selection mode:** `X` commits the pending triangle
      batch.
    - **Manual correction — snake / Clipping — PV contour:** `X` drops a geodesic
      point.

    The **Clipping active** checkbox decides whether the key belongs to clipping
    or to manual correction, so only one tool responds at a time.

## Per-tool buttons

Undo is per tool, not global:

- **Seeds:** Undo / Reset.
- **Manual correction:** Undo last edit (up to 3 levels); Undo last point / Clear
  (snake).
- **Clipping:** Undo last point (PV); Reject / revert clip.
- **Segmentation:** Undo (last segmentation edit).

Every hoverable control has a **tooltip** — hover to see what it does.
