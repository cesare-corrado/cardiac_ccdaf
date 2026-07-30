"""
test_seed_profiles.py
=====================
Tests for the seed-type profiles and a SeedSelector driven by a
non-default one (the LA-UAC landmark set).

The contract:

* the two shipped profiles are distinct sets under distinct export keys,
  and the default reproduces the original six-seed workflow;
* a selector built on the landmark profile picks that set, in that order,
  with no pulmonary-vein prior (none of its points are PVs) and no
  tagging exclusion (``seeds_for_tagging`` returns them all);
* ``hide``/``show`` drop and restore a selector's actors without losing
  the picks, which is what lets the two sets coexist on one plotter.

Reuses the synthetic atrium and StubPlotter of test_seed_persistence;
no display, no Qt.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np

from ccdaf.core.seed_profiles import (
    SEED_PROFILE, LANDMARKS_LA_UAC_PROFILE, SEED_PROFILES, DEFAULT_PROFILE,
)
from ccdaf.interaction.seed_selector import SeedSelector

from test_seed_persistence import StubPlotter, make_mesh


def test_registry_and_default():
    assert DEFAULT_PROFILE is SEED_PROFILE
    assert SEED_PROFILES["seed"] is SEED_PROFILE
    assert SEED_PROFILES["landmarks_LA_UAC"] is LANDMARKS_LA_UAC_PROFILE
    # Distinct point sets under distinct export keys.
    assert SEED_PROFILE.export_key == "seeds"
    assert LANDMARKS_LA_UAC_PROFILE.export_key == "landmarks_LA_UAC"
    assert set(SEED_PROFILE.order).isdisjoint(LANDMARKS_LA_UAC_PROFILE.order)


def test_landmark_profile_shape():
    p = LANDMARKS_LA_UAC_PROFILE
    assert p.order == ("LSPV_BODY_JCN", "RSPV_BODY_JCN",
                       "SEPTAL_WALL", "LATERAL_WALL")
    assert p.pv_names == frozenset()        # no PV prior
    assert p.no_tag_names == frozenset()    # nothing excluded from tagging
    assert p.tags is False                  # landmarks do not drive tagging
    assert set(p.prompts) == set(p.order)
    assert set(p.colors) == set(p.order)


def _pick_all(sel, profile):
    """Drive the selector by feeding each point's target coordinate."""
    targets = {
        "LSPV_BODY_JCN": [10.0, 0.0, 0.0],
        "RSPV_BODY_JCN": [0.0, 10.0, 0.0],
        "SEPTAL_WALL":   [0.0, 0.0, 10.0],
        "LATERAL_WALL":  [0.0, 0.0, -10.0],
    }
    for name in profile.order:
        sel._on_pick(np.asarray(targets[name], dtype=float))


def test_landmark_selector_end_to_end():
    profile = LANDMARKS_LA_UAC_PROFILE
    sel = SeedSelector(mesh=make_mesh(), plotter=StubPlotter(), profile=profile)
    sel.start()
    _pick_all(sel, profile)

    assert sel.is_complete
    assert list(sel.seeds.keys()) == list(profile.order)
    # No name is excluded from tagging for this set.
    assert set(sel.seeds_for_tagging()) == set(profile.order)


def test_hide_show_preserve_picks():
    profile = LANDMARKS_LA_UAC_PROFILE
    sel = SeedSelector(mesh=make_mesh(), plotter=StubPlotter(), profile=profile)
    sel.start()
    _pick_all(sel, profile)
    before = {n: s.xyz.copy() for n, s in sel.seeds.items()}

    sel.hide()
    assert not sel.is_active                # hide stops picking
    assert set(sel.seeds) == set(before)    # but keeps the picks

    sel.show()
    for n, xyz in before.items():
        assert np.allclose(sel.seeds[n].xyz, xyz)
