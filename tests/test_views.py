"""
test_views.py
=============
Tests for the view registry — each purpose the plotter can serve, bound to
its layout.

The contract:

* every view names all of its panes: roles cover the shape exactly, and no
  two roles share a pane;
* titles only describe roles that exist;
* the segmentation view's quadrants and title strings are pinned exactly —
  they are what the GUI shows today, and moving a pane must be a deliberate
  act that changes this test in the same commit;
* the title-actor name derivation is pinned, because three sites (the
  builder, the slice refresher, the 3D-pane clear) replace the same actor
  by name and must agree on it.

Pure data — no Qt, no pyvista, no display.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from ccdaf.app.views import VIEWS, ViewSpec, title_actor_name


def test_registry_carries_the_two_shipped_views():
    assert {"general", "segmentation"} <= set(VIEWS)


@pytest.mark.parametrize("key", sorted(VIEWS))
def test_name_matches_registry_key(key):
    assert VIEWS[key].name == key


def test_general_is_a_single_untitled_3d_view():
    spec = VIEWS["general"]
    assert spec.shape == (1, 1)
    assert not spec.is_multiview
    assert dict(spec.roles) == {"3d": (0, 0)}
    assert dict(spec.titles) == {}


def test_segmentation_quadrants_are_pinned():
    spec = VIEWS["segmentation"]
    assert spec.shape == (2, 2)
    assert spec.is_multiview
    assert dict(spec.roles) == {
        "axial": (0, 0),
        "sagittal": (0, 1),
        "3d": (1, 0),
        "coronal": (1, 1),
    }
    assert dict(spec.titles) == {
        "axial": "Axial (XY)",
        "sagittal": "Sagittal (YZ)",
        "coronal": "Coronal (XZ)",
        "3d": "3D",
    }


@pytest.mark.parametrize("key", sorted(VIEWS))
def test_roles_cover_the_shape_exactly(key):
    spec = VIEWS[key]
    locators = list(spec.roles.values())
    assert len(set(locators)) == len(locators), "two roles share a pane"
    if isinstance(spec.shape, tuple):
        rows, cols = spec.shape
        assert len(locators) == rows * cols, "every pane needs a role"
        for row, col in locators:
            assert 0 <= row < rows and 0 <= col < cols


@pytest.mark.parametrize("key", sorted(VIEWS))
def test_titles_only_describe_existing_roles(key):
    spec = VIEWS[key]
    assert set(spec.titles) <= set(spec.roles)


def test_title_actor_name_matches_the_historic_derivation():
    # The 2×2 build wrote f"_title_{row}_{col}" long before this module
    # existed; the derivation must not drift under existing plotters.
    assert title_actor_name((1, 0)) == "_title_1_0"
    assert title_actor_name((0, 1)) == "_title_0_1"
    # Single-index locators, as pyvista's string shapes ("1|2") use.
    assert title_actor_name((2,)) == "_title_2"


def test_a_future_string_shape_view_is_expressible():
    # The registry must carry pyvista's column layouts without new code:
    # e.g. a 3D pane beside two stacked graphs.
    spec = ViewSpec(
        name="example",
        shape="1|2",
        roles={"3d": (0,), "graph_top": (1,), "graph_bottom": (2,)},
        titles={"3d": "3D"},
    )
    assert spec.is_multiview
    assert set(spec.titles) <= set(spec.roles)
