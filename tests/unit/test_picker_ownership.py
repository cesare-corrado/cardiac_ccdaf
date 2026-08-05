"""
test_picker_ownership.py
========================
Guards the one rule the shared surface picker has: exactly one tool holds it.

The bug this pins: PyVista keeps a single picker per render window, and
``enable_point_picking`` raises ``PyVistaPickingError`` when a second tool
grabs it. Four tools call it on the same plotter — seed selection, the manual
editor's selection mode, its snake, and the PV contour clip — but the only
guard was between the editor's own two modes. So starting seed selection and
then ticking *Snake tag* (both panels are live from the moment a mesh loads)
raised straight out of a Qt slot, which does not unwind: the app aborted.

The contract:

* every tool releases the picker before taking it, so no pairing can raise —
  including the pairings that used to, seeds→snake and seeds→selection;
* the host's ``_release_picker`` stops whichever tool holds it and puts that
  panel's controls back in step, so nothing looks live while another tool
  drives the mouse;
* it never fights the caller: ``keep`` exempts the tool taking over;
* stopping is not destructive — seeds already picked survive, so the
  interrupted tool can carry on;
* the render window is left with no picker even when no tool admits to
  holding one.

Real tools against a real off-screen plotter for the picker half; a stand-in
host for the arbiter, since constructing the window needs a GL context.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pytest
import pyvista as pv
from ccdaf.core.mesh_loader import BODY_LABEL
from ccdaf.interaction.clipping_tool import ClipMode
from ccdaf.interaction.manual_editor import EditState, ManualEditor
from ccdaf.interaction.seed_selector import SeedSelector

pv.OFF_SCREEN = True


@pytest.fixture
def mesh():
    m = pv.Sphere(theta_resolution=16, phi_resolution=16).triangulate()
    m.cell_data["elemTag"] = np.full(m.n_cells, BODY_LABEL, dtype=np.int32)
    return m


@pytest.fixture
def plotter(mesh):
    p = pv.Plotter(off_screen=True)
    p.add_mesh(mesh)
    yield p
    try:
        p.close()
    except Exception:
        pass


def _editor(mesh, plotter) -> ManualEditor:
    return ManualEditor(mesh=mesh, plotter=plotter, on_render=lambda *a: None,
                        on_state=lambda s: None, on_commit=lambda *a: None)


def _selector(mesh, plotter) -> SeedSelector:
    return SeedSelector(plotter=plotter, mesh=mesh,
                        on_progress=lambda *a, **k: None)


# ------------------------------------------------------- the tools themselves


def test_seed_selection_then_snake(mesh, plotter):
    """The reported crash: seeds in progress, user ticks Snake tag."""
    _selector(mesh, plotter).start()
    _editor(mesh, plotter).start_snake(tagger=None)      # must not raise


def test_seed_selection_then_selection_mode(mesh, plotter):
    """The same hole, mirror image: seeds in progress, user ticks Edit."""
    _selector(mesh, plotter).start()
    _editor(mesh, plotter).activate()                     # must not raise


def test_snake_then_seed_selection(mesh, plotter):
    _editor(mesh, plotter).start_snake(tagger=None)
    _selector(mesh, plotter).start()


def test_every_ordered_pairing_survives(mesh, plotter):
    """No order of any two grabs may raise."""
    def grabs():
        ed, sel = _editor(mesh, plotter), _selector(mesh, plotter)
        return {
            "seeds": sel.start,
            "selection": ed.activate,
            "snake": lambda: ed.start_snake(tagger=None),
        }

    names = list(grabs())
    for first in names:
        for second in names:
            g = grabs()
            g[first]()
            g[second]()          # raised PyVistaPickingError before the fix


# ------------------------------------------------------------- the arbiter


def _host(*, snake=False, selecting=False, clip=False, seeds=False):
    """Stand-in carrying just what ``_release_picker`` touches."""
    host = MagicMock()
    host.editor.snake_active = snake
    host.editor.state = EditState.SELECTING if selecting else EditState.IDLE
    host.clipper.mode = ClipMode.PV_CONTOUR if clip else ClipMode.NONE
    sel = MagicMock()
    sel.is_active = seeds
    host._selectors = {"pv": sel}
    host._seed_sel = sel
    return host


def test_release_stops_the_holder_and_syncs_its_panel():
    from ccdaf.app.ccdaf import CCDAF

    host = _host(seeds=True)
    stopped = CCDAF._release_picker(host)

    assert stopped == ["seed selection"]
    host._seed_sel.stop.assert_called_once()
    host.seed_widget.set_prompt.assert_called_once()   # panel no longer "live"
    host.plotter.disable_picking.assert_called_once()


def test_release_unchecks_the_manual_toggles():
    from ccdaf.app.ccdaf import CCDAF

    host = _host(snake=True, selecting=True)
    stopped = CCDAF._release_picker(host)

    assert stopped == ["snake tagging", "selection mode"]
    host.manual_widget.uncheck_snake.assert_called_once()
    host.manual_widget.uncheck_edit_toggle.assert_called_once()
    host.editor.stop_snake.assert_called_once()
    host.editor.deactivate.assert_called_once()


def test_release_cancels_a_clip_in_flight():
    from ccdaf.app.ccdaf import CCDAF

    host = _host(clip=True)
    assert CCDAF._release_picker(host) == ["the clip in progress"]
    host.clipper.cancel.assert_called_once()
    host.clipping_widget.clear_in_flight.assert_called_once()
    # Activation and accepted-tagging are the user's — not reset here.
    host.clipping_widget.reset_state.assert_not_called()


@pytest.mark.parametrize("keep,attr,method", [
    ("seeds", "_seed_sel", "stop"),
    ("snake", "editor", "stop_snake"),
    ("selection", "editor", "deactivate"),
    ("clip", "clipper", "cancel"),
])
def test_keep_exempts_the_caller(keep, attr, method):
    """A tool taking the picker must never be stopped by its own request."""
    from ccdaf.app.ccdaf import CCDAF

    host = _host(snake=True, selecting=True, clip=True, seeds=True)
    CCDAF._release_picker(host, keep=keep)
    getattr(getattr(host, attr), method).assert_not_called()


def test_release_is_quiet_when_nobody_holds_it():
    from ccdaf.app.ccdaf import CCDAF

    host = _host()
    assert CCDAF._release_picker(host) == []
    # Still cleared: a tool dropped by a plotter rebuild can hold the picker
    # without any of them admitting to it.
    host.plotter.disable_picking.assert_called_once()


def test_release_survives_a_tool_that_throws():
    from ccdaf.app.ccdaf import CCDAF

    host = _host(seeds=True, snake=True)
    host._seed_sel.stop.side_effect = RuntimeError("stale actor")
    host.editor.stop_snake.side_effect = RuntimeError("stale actor")

    assert CCDAF._release_picker(host) == ["seed selection", "snake tagging"]
    host.plotter.disable_picking.assert_called_once()


def test_release_without_tools_does_not_crash():
    """Before any mesh is loaded there is no editor and no clipper."""
    from ccdaf.app.ccdaf import CCDAF

    host = MagicMock()
    host._selectors = {}
    host.editor = None
    host.clipper = None
    assert CCDAF._release_picker(host) == []


def _bind_release(host):
    """Give the stand-in the real ``_release_picker`` — MagicMock stubs it."""
    from ccdaf.app.ccdaf import CCDAF

    host._release_picker = lambda **kw: CCDAF._release_picker(host, **kw)
    return host


def test_take_picker_reports_what_it_interrupted():
    from ccdaf.app.ccdaf import CCDAF

    host = _bind_release(_host(seeds=True))
    CCDAF._take_picker(host, "snake")
    msg = host.statusBar.return_value.showMessage.call_args[0][0]
    assert "seed selection" in msg
    host._seed_sel.stop.assert_called_once()

    quiet = _bind_release(_host())
    CCDAF._take_picker(quiet, "snake")
    quiet.statusBar.return_value.showMessage.assert_not_called()


# --------------------------------------------------- resuming a paused set


def _seed_host(*, paused, complete=False, seeds=2):
    """Stand-in for the seed half of the host, with one selector for "pv"."""
    host = MagicMock()
    host.loader.mesh = object()
    host._seed_paused = set(paused)
    sel = MagicMock()
    sel.is_active = False
    sel.is_complete = complete
    sel.seeds = {f"s{i}": object() for i in range(seeds)}
    sel.next_name.return_value = "RSPV"
    host.selector = sel
    host._selectors = {"pv": sel}
    profile = MagicMock()
    profile.type_id = "pv"
    profile.label = "PV"
    host._current_profile.return_value = profile
    return host


def test_start_seeds_resumes_a_set_another_tool_interrupted():
    """The whole point of stopping non-destructively."""
    from ccdaf.app.ccdaf import CCDAF

    host = _seed_host(paused={"pv"})
    kept = host.selector
    CCDAF._action_start_seeds(host)

    kept.resume.assert_called_once()
    kept.start.assert_not_called()          # start() wipes the seed state
    host._new_selector.assert_not_called()  # and a new selector loses them
    assert host.selector is kept
    assert host._seed_paused == set()       # the pause is spent
    assert "resumed" in host.statusBar.return_value.showMessage.call_args[0][0]


def test_start_seeds_is_a_fresh_start_when_nothing_was_paused():
    """Unchanged behaviour for every ordinary start."""
    from ccdaf.app.ccdaf import CCDAF

    host = _seed_host(paused=set())
    CCDAF._action_start_seeds(host)

    host._new_selector.assert_called_once()
    host._new_selector.return_value.start.assert_called_once()


def test_start_seeds_does_not_resume_a_completed_set():
    from ccdaf.app.ccdaf import CCDAF

    host = _seed_host(paused={"pv"}, complete=True)
    CCDAF._action_start_seeds(host)

    host.selector.resume.assert_not_called()
    host._new_selector.assert_called_once()


def test_reset_seeds_cancels_a_pending_resume():
    """An explicit reset must not be undone by an earlier pause."""
    from ccdaf.app.ccdaf import CCDAF

    host = _seed_host(paused={"pv"})
    CCDAF._action_reset_seeds(host)
    assert host._seed_paused == set()

    host._new_selector.reset_mock()
    CCDAF._action_start_seeds(host)
    host._new_selector.assert_called_once()   # starts over, seeds not revived


def test_release_records_the_pause():
    from ccdaf.app.ccdaf import CCDAF

    host = _host(seeds=True)
    host._seed_paused = set()
    CCDAF._release_picker(host)
    assert host._seed_paused == {"pv"}


def test_take_picker_does_not_stop_its_own_tool():
    """``_take_picker`` must pass its name through as ``keep``."""
    from ccdaf.app.ccdaf import CCDAF

    host = _bind_release(_host(snake=True, seeds=True))
    CCDAF._take_picker(host, "snake")
    host.editor.stop_snake.assert_not_called()   # the caller
    host._seed_sel.stop.assert_called_once()     # the holder
