"""
test_seed_profiles.py
=====================
Tests for the seed-type profiles and a SeedSelector driven by a
non-default one.

The contract:

* the shipped profiles (left and right atrium) are distinct sets under
  distinct export keys, and the default reproduces the original six-seed
  workflow under its new name;
* each profile's capability lists (radii, labels, clip regions) say what
  the Tagging, Manual-correction and Clipping panels may offer, and the
  label values agree with the modules that own them
  (``ccdaf.core.atrial_labels``, the tagger and the editor);
* the retired ``landmarks_LA_UAC`` key maps to no profile, so a file that
  carries it loads with that set ignored;
* a selector built on another profile picks that set, in that order;
* ``hide``/``show`` drop and restore a selector's actors without losing
  the picks, which is what lets several sets coexist on one plotter.

Reuses the synthetic atrium and StubPlotter of test_seed_persistence;
no display, no Qt.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np

from ccdaf.core import atrial_labels as al
from ccdaf.core.region_tagger import LABELS, TaggerConfig
from ccdaf.core.seed_profiles import (
    SEED_LA_PROFILE, SEED_RA_PROFILE, SeedProfile,
    SEED_PROFILE_ORDER, SEED_PROFILES, DEFAULT_PROFILE, profile_for_key,
)
from ccdaf.interaction.manual_editor import ALLOWED_LABELS
from ccdaf.interaction.seed_selector import SeedSelector

from test_seed_persistence import StubPlotter, make_mesh

#: A four-point set with no vein prior and nothing excluded from tagging,
#: to drive a selector on something other than the default profile.
FOUR = SeedProfile(
    type_id="four", label="four", export_key="four",
    order=("A", "B", "C", "D"),
    prompts={n: n for n in "ABCD"}, colors={n: "#000000" for n in "ABCD"},
)


def test_registry_and_default():
    assert DEFAULT_PROFILE is SEED_LA_PROFILE
    assert SEED_PROFILES["seed_LA"] is SEED_LA_PROFILE
    assert SEED_PROFILES["seed_RA"] is SEED_RA_PROFILE
    assert "landmarks_LA_UAC" not in SEED_PROFILES          # retired
    assert [p.type_id for p in SEED_PROFILE_ORDER] == ["seed_LA", "seed_RA"]
    assert SEED_LA_PROFILE.export_key == "seed_LA"
    assert SEED_RA_PROFILE.export_key == "seed_RA"
    assert set(SEED_LA_PROFILE.order).isdisjoint(SEED_RA_PROFILE.order)
    keys = [p.export_key for p in SEED_PROFILE_ORDER]
    assert len(keys) == len(set(keys))


def test_legacy_key_maps_to_seed_la():
    """A file written before the rename still finds its profile."""
    assert SEED_LA_PROFILE.read_keys[0] == "seed_LA"
    assert "seeds" in SEED_LA_PROFILE.read_keys
    assert profile_for_key("seeds") is SEED_LA_PROFILE
    assert profile_for_key("seed") is SEED_LA_PROFILE
    assert profile_for_key("seed_LA") is SEED_LA_PROFILE
    assert profile_for_key("seed_RA") is SEED_RA_PROFILE
    assert profile_for_key("landmarks_LA_UAC") is None      # ignored on load
    assert profile_for_key("nothing_like_it") is None


def test_capabilities_per_profile():
    """Each atrium offers its own radii, labels and clip regions."""
    assert SEED_LA_PROFILE.radius_names == ("LSPV", "LIPV", "RSPV", "RIPV", "LAA")
    assert SEED_LA_PROFILE.clip_regions == ("LSPV", "LIPV", "RSPV", "RIPV", "MV")
    assert SEED_LA_PROFILE.tags is True and SEED_LA_PROFILE.atrium == "LA"
    assert SEED_RA_PROFILE.order == ("SVC", "IVC", "CS", "RAA", "TV")
    assert SEED_RA_PROFILE.radius_names == ("SVC", "IVC", "CS", "RAA")
    assert SEED_RA_PROFILE.clip_regions == ("SVC", "IVC", "CS", "TV")
    assert SEED_RA_PROFILE.no_tag_names == frozenset(("TV",))
    assert SEED_RA_PROFILE.tags is True and SEED_RA_PROFILE.atrium == "RA"
    for p in SEED_PROFILE_ORDER:
        assert p.persists
        assert set(p.prompts) == set(p.order) == set(p.colors)


def test_label_values_agree_with_their_owners():
    """The profiles' labels are the ones the label table, the editor and
    the tagger use. seed_profiles spells the values out rather than
    importing them, to stay free of the geometry stack; this is what stops
    the copy drifting."""
    for profile, atrium in ((SEED_LA_PROFILE, al.LA), (SEED_RA_PROFILE, al.RA)):
        assert tuple(profile.label_values) == atrium.values
        assert set(profile.radius_names) == set(atrium.regions)
        assert set(profile.label_values) <= set(ALLOWED_LABELS)
        cfg = TaggerConfig()
        for name in profile.radius_names:
            assert hasattr(cfg, f"{name.lower()}_radius_factor")
        # every clip region is a tagged region or the atrium's valve
        assert set(profile.clip_regions) <= set(atrium.regions) | {atrium.valve}
    assert LABELS == dict(al.LA.regions)
    assert set(ALLOWED_LABELS) == set(al.LA.values) | set(al.RA.values)
    # the two atria never share a label, bodies included
    assert not set(al.LA.values) & set(al.RA.values)


def _pick_all(sel, profile):
    """Drive the selector by feeding each point's target coordinate."""
    targets = {
        "A": [10.0, 0.0, 0.0],
        "B": [0.0, 10.0, 0.0],
        "C": [0.0, 0.0, 10.0],
        "D": [0.0, 0.0, -10.0],
    }
    for name in profile.order:
        sel._on_pick(np.asarray(targets[name], dtype=float))


def test_another_profile_selector_end_to_end():
    profile = FOUR
    sel = SeedSelector(mesh=make_mesh(), plotter=StubPlotter(), profile=profile)
    sel.start()
    _pick_all(sel, profile)

    assert sel.is_complete
    assert list(sel.seeds.keys()) == list(profile.order)
    # No name is excluded from tagging for this set.
    assert set(sel.seeds_for_tagging()) == set(profile.order)


def test_hide_show_preserve_picks():
    profile = FOUR
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
