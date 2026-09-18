"""
test_window_title.py
====================
The main window's title names the open file: no folder, no extension.
"""

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pytest

from ccdaf.app.ccdaf import APP_TITLE, CCDAF, display_name, window_title


@pytest.mark.parametrize("filename, shown", [
    ("/data/meshes/AJR5d03295_truncated.vtk", "AJR5d03295_truncated"),
    ("relative/heart.vtu", "heart"),
    ("bundle.pkl", "bundle"),
    ("/scans/case 7.nii.gz", "case 7"),      # two extensions, both dropped
    ("/scans/CASE.NII.GZ", "CASE"),
    ("/scans/case.nii", "case"),
    ("model.v2.vtk", "model.v2"),            # only the last extension goes
])
def test_the_title_shows_the_file_name_alone(filename, shown):
    assert display_name(filename) == shown
    assert window_title(display_name(filename)) == f"{APP_TITLE} — {shown}"


def test_with_nothing_open_the_title_is_the_application_alone():
    assert window_title() == APP_TITLE
    assert window_title("") == APP_TITLE


@pytest.mark.parametrize("method", ["_adopt_mesh", "_load_segmentation",
                                    "_load_eam_mapping", "_action_close"])
def test_every_change_of_file_updates_the_title(method):
    # Loading a mesh or bundle (both adopt it), a segmentation or an EAM
    # map, and closing are the places the open file changes; a title left behind by
    # any of them would name a file that is no longer open.
    source = inspect.getsource(getattr(CCDAF, method))
    assert "setWindowTitle(window_title(" in source
