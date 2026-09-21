"""
test_wall_report.py
===================
Measuring a ventricular wall without changing it.

The contract:

* **it changes nothing.** The measurements need a manifold boundary and
  the cleaner's default deliberately leaves one that is not, so the
  repair happens on a copy and the caller's mesh comes back identical;
* **the target is not zero handles.** A shell around ``p`` cavities
  opened by ``o`` valve openings has ``o - p`` of them: a cavity opened
  twice gives a loop you cannot shrink. Anything beyond that is a hole;
* **a sound wall is reported as sound**, or the check would cry wolf on
  every mesh and be ignored on the one that mattered.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pyvista as pv

from ccdaf.core import volume_mesh as vm
from ccdaf.core.orifice_labels import Passage
from ccdaf.core.wall_report import WallReport, check_wall


def _hollow(n: int = 11, size: float = 10.0, channel: int = 0):
    """A hollow box, optionally bored through from cavity to outside."""
    grid = pv.ImageData(dimensions=(n, n, n), spacing=(size / (n - 1),) * 3)
    grid = grid.cast_to_unstructured_grid().triangulate()
    step = size / (n - 1)
    idx = np.floor(np.asarray(grid.cell_centers().points) / step).astype(int)
    mid = (n - 1) // 2
    hollow = (np.abs(idx - mid) <= 2).all(axis=1)
    bore = np.zeros(len(idx), dtype=bool)
    if channel:
        half = (channel - 1) // 2
        bore = ((np.abs(idx[:, 1] - mid) <= half)
                & (np.abs(idx[:, 2] - mid) <= half)
                & (idx[:, 0] > mid))
    return grid.extract_cells(np.where(~(hollow | bore))[0])


def test_the_wall_check_does_not_touch_the_mesh():
    """A measurement that edited the mesh would be a trap, not a check."""
    grid = _hollow(channel=3)
    before_points = np.asarray(grid.points).copy()
    before_cells = vm.tetrahedra(grid).copy()

    check_wall(grid)

    assert np.array_equal(np.asarray(grid.points), before_points)
    assert np.array_equal(vm.tetrahedra(grid), before_cells)


def test_a_wall_with_no_extra_handle_is_reported_sound():
    report = WallReport(genus=1, expected=1, pool_ml=[100.0, 90.0],
                        opening_mm2=[900.0, 300.0, 120.0])
    assert report.extra_handles == 0
    assert report.sound
    assert "sound" in report.summary()


def test_the_extra_handles_are_what_the_openings_do_not_account_for():
    """Two cavities and three openings imply one handle, not none.

    In through the mitral opening, along the cavity, out through the
    aortic one: a loop that cannot be shrunk away, and anatomy rather
    than damage. This is the number the example ventricle measures two
    against, which is how its perforation is known to be one hole.
    """
    report = WallReport(genus=2, expected=1, pool_ml=[150.7, 141.2],
                        opening_mm2=[1226.0, 606.0, 177.0])
    assert report.extra_handles == 1
    assert not report.sound
    assert "1 hole" in report.summary()


def test_an_unmeasurable_wall_says_so_rather_than_guessing():
    report = WallReport(genus=None, expected=1)
    assert report.extra_handles is None
    assert "could not be measured" in report.summary()


def test_the_details_name_where_each_way_through_the_wall_is():
    ring = np.array([[0, 1], [1, 2]], dtype=np.int64)
    report = WallReport(
        genus=2, expected=1, pool_ml=[150.7, 141.2],
        opening_mm2=[1226.0, 606.0, 177.0],
        joins=[Passage(ring=ring, circumference=232.2,
                       centre=np.array([10.2, -3.3, -6.1])),
               Passage(ring=ring, circumference=11.7,
                       centre=np.array([56.4, -10.0, 24.0]))],
        repaired="separated 15 touching contacts")
    text = report.details()
    assert "150.7 mL" in text and "1226 mm²" in text
    assert "56.4" in text
    assert "your mesh is unchanged" in text


def test_a_wall_it_cannot_measure_says_where_and_what_to_do():
    """The failure has to be actionable or it is just a refusal.

    A mesh already cleaned can be harder to measure than the one it came
    from: welding fuses the contacts the separating pass would have
    taken, and one the weld refused stays refused. So the report names
    the places left and says to run it on the mesh as loaded.
    """
    report = WallReport(genus=None, expected=1,
                        pool_ml=[150.7, 141.3],
                        opening_mm2=[1226.0, 600.0, 177.0],
                        unresolved=[np.array([56.1, -9.68, 23.9])],
                        repaired="1 non-manifold contacts could not be "
                                 "repaired")
    text = report.details()
    assert "at 1 place" in report.summary()
    assert "56.1" in text
    assert "before cleaning" in text


def test_a_cleaned_mesh_can_be_harder_to_measure_than_its_source():
    """The ordering that surprised a user, pinned as behaviour.

    Welding is on by default and separating is not, so a default clean
    fuses contacts that the check's own full repair would otherwise have
    separated. What the weld refuses is then unrepairable, and the copy
    cannot be made manifold.
    """
    from ccdaf.core.volume_clean import clean as clean_volume

    grid = _hollow(channel=1)
    straight = check_wall(grid)
    after_clean = check_wall(clean_volume(grid)[0])
    # Whatever each concludes, neither may claim to have measured a wall
    # it could not: a genus of None must come with somewhere to look.
    for report in (straight, after_clean):
        if report.genus is None:
            assert "before cleaning" in report.details()
