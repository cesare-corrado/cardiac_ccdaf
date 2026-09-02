"""
test_cli_and_unsaved_changes.py
===============================
Two window-level behaviours, exercised without building the window:

* the command-line parser, and the extension routing (``_open_path``) that
  both it and File → Load data go through — a ``.pkl`` must reach the bundle
  reader, not the mesh reader;
* the unsaved-work flag and the Save / Discard / Cancel prompt that File →
  Close and quitting put in front of it.

The methods under test are called unbound on a stub host carrying only the
attributes they touch, the way ``test_manual_after_segmentation`` does — a
real ``CCDAF`` needs a live plotter, which is far more than these need.

Runs headless with the offscreen Qt platform (the project's standard pytest
invocation sets ``QT_QPA_PLATFORM=offscreen``).
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pytest
from PyQt5 import QtWidgets

import ccdaf
from ccdaf.app.ccdaf import CCDAF, build_parser


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------
def test_no_argument_means_no_file():
    assert build_parser().parse_args([]).path is None


def test_a_path_is_taken_as_the_file_to_open():
    assert build_parser().parse_args(["mesh.vtk"]).path == "mesh.vtk"


def test_version_reports_the_package_version(capsys):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["--version"])
    assert exc.value.code == 0
    assert ccdaf.__version__ in capsys.readouterr().out


def test_a_second_path_is_rejected():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["a.vtk", "b.vtk"])


class _Opener:
    """The three readers ``_open_path`` dispatches to, and nothing else."""

    def __init__(self):
        self.calls = []
        self._load_mesh = lambda fn: self.calls.append(("mesh", fn))
        self._load_bundle = lambda fn: self.calls.append(("bundle", fn))
        self._load_segmentation = lambda fn: self.calls.append(("seg", fn))

    def open(self, fn):
        CCDAF._open_path(self, fn)
        return self.calls[-1][0]


@pytest.mark.parametrize("name, reader", [
    ("surface.vtk", "mesh"),
    ("surface.vtp", "mesh"),
    ("surface.STL", "mesh"),
    ("bundle.pkl", "bundle"),
    ("bundle.pickle", "bundle"),
    ("bundle.PKL", "bundle"),
    ("volume.nii", "seg"),
    ("volume.nii.gz", "seg"),
])
def test_the_extension_picks_the_reader(name, reader):
    assert _Opener().open(name) == reader


# ---------------------------------------------------------------------------
# Unsaved work
# ---------------------------------------------------------------------------
class _Host(QtWidgets.QWidget):
    """The parts of the window the dirty-state helpers reach for.

    A real ``QWidget``: the prompt parents its message box to it.
    """

    def __init__(self, mesh=object(), seg=None):
        super().__init__()
        self._dirty = False
        self._seg_dirty = False
        self.loader = MagicMock(mesh=mesh)
        self._seg_array = seg
        self.saved = 0
        self.seg_saved = 0

    # The real methods, bound here.
    _mark_dirty = CCDAF._mark_dirty
    _clear_dirty = CCDAF._clear_dirty
    _mark_seg_dirty = CCDAF._mark_seg_dirty
    _clear_seg_dirty = CCDAF._clear_seg_dirty
    _unsaved = CCDAF._unsaved
    _confirm_discard = CCDAF._confirm_discard

    def _action_save(self):
        self.saved += 1

    def _action_seg_save(self):
        self.seg_saved += 1


def _answer(monkeypatch, button):
    monkeypatch.setattr(QtWidgets.QMessageBox, "exec_", lambda self: button)


def test_clean_data_never_prompts(qapp, monkeypatch):
    host = _Host()
    _answer(monkeypatch, pytest.fail)     # calling it at all is the failure
    assert host._confirm_discard("quitting") is True


def test_discard_lets_the_close_through_without_saving(qapp, monkeypatch):
    host = _Host()
    host._mark_dirty()
    _answer(monkeypatch, QtWidgets.QMessageBox.Discard)
    assert host._confirm_discard("quitting") is True
    assert host.saved == 0


def test_cancel_stops_the_close(qapp, monkeypatch):
    host = _Host()
    host._mark_dirty()
    _answer(monkeypatch, QtWidgets.QMessageBox.Cancel)
    assert host._confirm_discard("quitting") is False
    assert host.saved == 0
    assert host._dirty, "cancelling leaves the work unsaved"


def test_save_runs_the_file_menu_save(qapp, monkeypatch):
    host = _Host()
    host._mark_dirty()
    _answer(monkeypatch, QtWidgets.QMessageBox.Save)
    # A save that went through is what clears the flag.
    host._action_save = lambda: (setattr(host, "saved", host.saved + 1),
                                 host._clear_dirty())
    assert host._confirm_discard("quitting") is True
    assert host.saved == 1


def test_a_cancelled_save_dialog_cancels_the_close(qapp, monkeypatch):
    """Save → then backing out of the file dialog must not discard the work."""
    host = _Host()
    host._mark_dirty()
    _answer(monkeypatch, QtWidgets.QMessageBox.Save)   # _action_save writes nothing
    assert host._confirm_discard("quitting") is False
    assert host.saved == 1
    assert host._dirty


def test_a_segmentation_with_no_mesh_saves_the_segmentation(qapp, monkeypatch):
    host = _Host(mesh=None, seg=object())
    host._mark_seg_dirty()
    _answer(monkeypatch, QtWidgets.QMessageBox.Save)
    host._confirm_discard("quitting")
    assert (host.saved, host.seg_saved) == (0, 1)


def test_both_unsaved_are_offered_segmentation_first(qapp, monkeypatch):
    """The volume is what the user is looking at, so it is asked about first."""
    host = _Host(mesh=object(), seg=object())
    host._mark_dirty()
    host._mark_seg_dirty()
    _answer(monkeypatch, QtWidgets.QMessageBox.Save)
    order = []
    host._action_save = lambda: (order.append("mesh"), host._clear_dirty())
    host._action_seg_save = lambda: (order.append("seg"), host._clear_seg_dirty())
    assert host._confirm_discard("quitting") is True
    assert order == ["seg", "mesh"]


def test_a_flag_left_over_from_unloaded_data_never_blocks_the_close(qapp, monkeypatch):
    """Both flags set, nothing loaded: there is nothing to save, so no prompt."""
    host = _Host(mesh=None, seg=None)
    host._mark_dirty()
    host._mark_seg_dirty()
    _answer(monkeypatch, pytest.fail)
    assert host._confirm_discard("quitting") is True


def test_a_mesh_edit_does_not_make_the_segmentation_unsaved(qapp):
    host = _Host(mesh=object(), seg=object())
    host._mark_dirty()
    assert host._unsaved() == (True, False)


def test_close_event_ignores_the_event_when_the_user_cancels(qapp, monkeypatch):
    host = _Host()
    host._mark_dirty()
    _answer(monkeypatch, QtWidgets.QMessageBox.Cancel)
    event = MagicMock()
    CCDAF.closeEvent(host, event)
    event.ignore.assert_called_once()
    event.accept.assert_not_called()


def test_close_event_accepts_when_there_is_nothing_to_lose(qapp):
    host = _Host()
    event = MagicMock()
    CCDAF.closeEvent(host, event)
    event.accept.assert_called_once()


# ---------------------------------------------------------------------------
# The File menu says "data", not "mesh" — it handles more than meshes.
# ---------------------------------------------------------------------------
def test_file_menu_labels_say_data():
    src = Path(ccdaf.app.__file__).with_name("ccdaf.py").read_text()
    assert '"&Load data…"' in src and '"&Save data…"' in src
    assert "&Load mesh" not in src and "&Save mesh" not in src


# ---------------------------------------------------------------------------
# Segmentation → Close segmentation offers to write the volume out
# ---------------------------------------------------------------------------
class _SegHost(QtWidgets.QWidget):
    """The parts of the window ``_offer_save_segmentation`` reaches for."""

    _offer_save_segmentation = CCDAF._offer_save_segmentation
    _action_seg_close = CCDAF._action_seg_close

    def __init__(self, seg=object(), dirty=True):
        super().__init__()
        self._seg_array = seg
        self._seg_dirty = dirty
        self.recent_folder = Path(".")
        self.written = []
        self.closed = 0

    def _write_segmentation(self, fn):
        self.written.append(fn)
        return True

    def _close_segmentation(self):
        self.closed += 1


def _file_dialog(monkeypatch, path):
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (path, "")))


def test_no_volume_means_nothing_to_offer(qapp, monkeypatch):
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", pytest.fail)
    assert _SegHost(seg=None)._offer_save_segmentation("?") is True


def test_yes_writes_the_file_the_dialog_returns(qapp, monkeypatch):
    host = _SegHost()
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Yes))
    _file_dialog(monkeypatch, "/tmp/seg.nii")
    assert host._offer_save_segmentation("?") is True
    assert host.written == ["/tmp/seg.nii"]


def test_no_goes_on_without_writing(qapp, monkeypatch):
    host = _SegHost()
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        staticmethod(lambda *a, **k: QtWidgets.QMessageBox.No))
    assert host._offer_save_segmentation("?") is True
    assert host.written == []


def test_cancel_aborts_the_caller(qapp, monkeypatch):
    host = _SegHost()
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Cancel))
    assert host._offer_save_segmentation("?") is False
    assert host.written == []


def test_a_cancelled_file_dialog_aborts_too(qapp, monkeypatch):
    """Yes then no filename must not be read as "go ahead and drop it"."""
    host = _SegHost()
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Yes))
    _file_dialog(monkeypatch, "")
    assert host._offer_save_segmentation("?") is False
    assert host.written == []


def test_close_segmentation_asks_when_the_volume_was_edited(qapp, monkeypatch):
    host = _SegHost(dirty=True)
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        staticmethod(lambda *a, **k: QtWidgets.QMessageBox.No))
    host._action_seg_close()
    assert host.closed == 1


def test_close_segmentation_is_cancellable(qapp, monkeypatch):
    host = _SegHost(dirty=True)
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Cancel))
    host._action_seg_close()
    assert host.closed == 0, "Cancel must leave the segmentation open"


def test_an_untouched_volume_closes_without_a_prompt(qapp, monkeypatch):
    host = _SegHost(dirty=False)
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", pytest.fail)
    host._action_seg_close()
    assert host.closed == 1
