"""
test_seed_profiles.py
=====================
Tests for the seed-type profiles and a SeedSelector driven by a
non-default one (the LA-UAC landmark set).

The contract:

* the shipped profiles are distinct sets under distinct export keys, and
  the default reproduces the original six-seed workflow under its new
  name;
* each profile's capability lists (radii, labels, clip regions) say what
  the Tagging, Manual-correction and Clipping panels may offer, and the
  label values agree with the modules that own them;
* a signpost profile has no points, never persists, and reads as
  complete to the state machine — which is exactly why the panel has to
  gate on ``count`` rather than on ``is_complete``;
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

from ccdaf.core.mesh_loader import BODY_LABEL
from ccdaf.core.region_tagger import LABELS
from ccdaf.core.seed_profiles import (
    SEED_LA_PROFILE, LANDMARKS_LA_UAC_PROFILE, SEED_RA_PROFILE,
    SEED_PROFILE_ORDER, SEED_PROFILES, DEFAULT_PROFILE, profile_for_key,
)
from ccdaf.interaction.manual_editor import ALLOWED_LABELS
from ccdaf.interaction.seed_selector import SeedSelector

from test_seed_persistence import StubPlotter, make_mesh


def test_registry_and_default():
    assert DEFAULT_PROFILE is SEED_LA_PROFILE
    assert SEED_PROFILES["seed_LA"] is SEED_LA_PROFILE
    assert SEED_PROFILES["landmarks_LA_UAC"] is LANDMARKS_LA_UAC_PROFILE
    assert SEED_PROFILES["seed_RA"] is SEED_RA_PROFILE
    # Distinct point sets under distinct export keys.
    assert SEED_LA_PROFILE.export_key == "seed_LA"
    assert LANDMARKS_LA_UAC_PROFILE.export_key == "landmarks_LA_UAC"
    assert set(SEED_LA_PROFILE.order).isdisjoint(LANDMARKS_LA_UAC_PROFILE.order)
    # Every export key is unique, or a bundle would overwrite one set with
    # another.
    keys = [p.export_key for p in SEED_PROFILE_ORDER]
    assert len(keys) == len(set(keys))


def test_legacy_key_maps_to_seed_la():
    """A file written before the rename still finds its profile."""
    assert SEED_LA_PROFILE.read_keys[0] == "seed_LA"
    assert "seeds" in SEED_LA_PROFILE.read_keys
    assert profile_for_key("seeds") is SEED_LA_PROFILE
    assert profile_for_key("seed") is SEED_LA_PROFILE
    assert profile_for_key("seed_LA") is SEED_LA_PROFILE
    assert profile_for_key("landmarks_LA_UAC") is LANDMARKS_LA_UAC_PROFILE
    assert profile_for_key("nothing_like_it") is None


def test_capabilities_per_profile():
    """Only the left-atrial seed set drives tagging, labels and clipping."""
    assert SEED_LA_PROFILE.radius_names == ("LSPV", "LIPV", "RSPV", "RIPV", "LAA")
    assert SEED_LA_PROFILE.clip_regions == ("LSPV", "LIPV", "RSPV", "RIPV", "MV")
    assert SEED_LA_PROFILE.tags is True

    for profile in (LANDMARKS_LA_UAC_PROFILE, SEED_RA_PROFILE):
        assert profile.radius_names == ()
        assert profile.label_values == ()
        assert profile.clip_regions == ()
        assert profile.tags is False


def test_label_values_agree_with_their_owners():
    """The profile's labels are the ones the editor and the tagger use.

    seed_profiles spells the values out rather than importing them, to
    stay free of the geometry stack. This is what stops the copy drifting.
    """
    assert set(SEED_LA_PROFILE.label_values) == set(ALLOWED_LABELS)
    assert set(SEED_LA_PROFILE.label_values) == set(LABELS.values()) | {BODY_LABEL}
    # Radius names are tagged regions, so each must be a real label.
    assert set(SEED_LA_PROFILE.radius_names) <= set(LABELS)
    # ... and each must have a config field the GUI can set.
    from ccdaf.core.region_tagger import TaggerConfig
    cfg = TaggerConfig()
    for name in SEED_LA_PROFILE.radius_names:
        assert hasattr(cfg, f"{name.lower()}_radius_factor")


def test_signpost_profile_has_no_points_and_never_persists():
    p = SEED_RA_PROFILE
    assert p.order == ()
    assert p.count == 0
    assert p.persists is False
    # The trap this exists to document: an empty order is "complete", so a
    # panel gating Save on is_complete alone would offer to write nothing.
    from ccdaf.core.seed_state_machine import SeedStateMachine
    assert SeedStateMachine(p.order).is_complete is True
    for other in (SEED_LA_PROFILE, LANDMARKS_LA_UAC_PROFILE):
        assert other.persists is True


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
