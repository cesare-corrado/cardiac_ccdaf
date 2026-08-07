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
| **`X`** | Drop a point for a snake tool |
| **`C`** | Commit the manual-correction selection batch |

!!! info "Who owns `X`"
    `X` is the *pick* key, shared by the two tools that cannot both listen at
    once — the **manual snake** and the **PV contour** — and the **Clipping
    active** checkbox decides which of them responds.

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
