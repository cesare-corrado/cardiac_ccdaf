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
