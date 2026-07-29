# Architecture

CCDAF separates **pure logic** from the **GUI**, so the algorithms are
headless-testable and Qt-free. The package lives under `src/ccdaf/`:

```mermaid
flowchart TD
    app["app<br/>main window · view specs"]
    gui["gui<br/>Qt panels &amp; dialogs"]
    interaction["interaction<br/>VTK/PyVista tools"]
    core["core<br/>algorithms &amp; state (Qt-free)"]
    io["io<br/>readers/writers"]

    app --> gui
    app --> interaction
    app --> core
    gui -->|signals| app
    interaction --> core
    core --> io
```

| Package | Responsibility |
|---|---|
| `ccdaf.core` | Pure algorithms and state: mesh load, region tagging, seed geometry/state machine, post-processing, field transfer, segmentation, EAM load/export, seed I/O. **No Qt, no VTK interaction** — unit-testable headlessly. |
| `ccdaf.io` | Low-level file readers/writers (`vtkfunctions`, `carto_functions`). |
| `ccdaf.interaction` | On-surface tools that own a picker: `seed_selector`, `manual_editor`, `clipping_tool`. Logic (state, geometry) is delegated to `core`; these add markers, HUD, and pick callbacks. |
| `ccdaf.gui` | Qt widgets — one panel per stage plus dialogs. Widgets expose **signals**; they hold no business logic. |
| `ccdaf.app` | The main window (`ccdaf.app.ccdaf.CCDAF`) wires widget signals to `core` / `interaction`, and `views` defines the window layouts. |

## Design principles

- **Signals up, calls down.** Widgets emit intent (`snake_commit_requested`, …);
  the app translates that into calls on `core`/`interaction`. Widgets never
  import `core`.
- **Logic is Qt-free.** Anything worth testing lives in `core` and is exercised
  by the headless suite. Interaction tools keep a thin, mockable plotter surface
  so their logic is testable too.
- **One view registry.** New window layouts are entries in `ccdaf.app.views.VIEWS`,
  not new code paths.

## Where things live

- The `X`-key routing and tool arbitration is in `ccdaf.app.ccdaf`.
- Picking uses VTK's hardware picker; see
  [Concepts → Picking](../concepts.md#picking).
- Electrode displacement is in `ccdaf.core.eam_loader`; see
  [Concepts → Electrode displacement](../concepts.md#electrode-displacement).

The full symbol-level documentation is generated from the docstrings in the
[API reference](../reference/index.md).
