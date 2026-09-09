"""
Seed profiles
=============

A *seed profile* describes one selectable set of surface points and, with
it, what the rest of the workflow may do while that set is chosen: which
points to pick and in what order, how each is prompted and coloured,
which are pulmonary veins (so the anatomical prior applies), which are
excluded from region tagging, the key the set is written under in a saved
pickle/JSON, and which radii, labels and clip regions the Tagging,
Manual-correction and Clipping panels offer.

That last part is why the profile is the single source of truth rather
than three panels each hard-coding the atrium: adding an anatomy is
adding a profile, not editing every panel that mentions a pulmonary vein.

Three profiles ship:

* :data:`SEED_LA_PROFILE` — the six-seed left-atrium workflow
  (LSPV, LIPV, RSPV, RIPV, LAA, MV). It feeds region tagging, offers the
  five radius factors, the six ``elemTag`` labels and the five clip
  regions, and is written under the ``"seed_LA"`` key.
* :data:`LANDMARKS_LA_UAC_PROFILE` — four left-atrium landmarks used for
  a universal-atrial-coordinate step (LSPV_BODY_JCN, RSPV_BODY_JCN,
  SEPTAL_WALL, LATERAL_WALL). Picked, saved and exported exactly like
  seeds, under the ``"landmarks_LA_UAC"`` key, and never tagged. Seed
  selection is all it offers: no radii, no labels, no clip regions.
* :data:`SEED_RA_PROFILE` — the right atrium, with no points defined
  yet. A signpost: choosing it says what the mesh is and switches every
  atrial-left tool off, and it reserves the slot the right-atrial seeds
  will occupy.

The sets are independent: a session can complete several, and a saved
bundle can carry every non-empty one at once.

Persistence and the old key
---------------------------
The six-seed set was written under ``"seeds"`` before it was named
``seed_LA``. New files carry ``"seed_LA"``; :attr:`SeedProfile.read_keys`
lists the old spellings still accepted on read, so a file saved by an
earlier version loads unchanged and is converted on the next save.

A profile with no points never reaches a file at all — see
:attr:`SeedProfile.persists`. There is nothing to write, and a key
carrying an empty mapping would claim the mesh has a set it does not.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Tuple


#: ``elemTag`` values of the left-atrial tagging, in the order the
#: manual-correction dropdown offers them: the four veins, the appendage,
#: then the body.
#:
#: Mirrors ``region_tagger.LABELS`` plus ``mesh_loader.BODY_LABEL``, spelled
#: out here rather than imported so this module stays free of the geometry
#: stack (numpy/scipy/pyvista) that every panel would otherwise drag in to
#: read a dropdown. ``test_seed_profiles`` asserts the two never drift.
LA_LABEL_VALUES: Tuple[int, ...] = (11, 13, 15, 17, 19, 1)

#: Regions the left-atrial clipping panel offers: the four veins, which
#: carry their own tag, and the mitral valve, which sits on the body.
LA_CLIP_REGIONS: Tuple[str, ...] = ("LSPV", "LIPV", "RSPV", "RIPV", "MV")


@dataclass(frozen=True)
class SeedProfile:
    """One selectable set of surface points and what it enables.

    Attributes
    ----------
    type_id
        Stable identifier used as the dropdown value and the dict key that
        tracks this set's selector.
    label
        Text shown in the seed-type dropdown.
    export_key
        Key the set is written under in a saved pickle/JSON bundle.
    order
        The points to pick, in strict acquisition order. Empty for a
        signpost profile, which has no points yet.
    prompts
        Per-point instruction shown while it is the next to pick.
    colors
        Per-point marker/HUD colour (hex).
    pv_names
        Points to which the pulmonary-vein anatomical prior applies.
    no_tag_names
        Points excluded from the region-tagging seed map (e.g. MV).
    tags
        Whether completing this set enables region tagging.
    radius_names
        Seeds the Tagging panel shows a radius factor for. Empty means the
        panel offers no radii and cannot run.
    label_values
        ``elemTag`` values the Manual-correction panel offers, in dropdown
        order. Empty disables manual correction entirely.
    clip_regions
        Regions the Clipping panel offers. Empty disables clipping.
    legacy_keys
        Older on-disk spellings of ``export_key`` still accepted on read.
    """

    type_id: str
    label: str
    export_key: str
    order: Tuple[str, ...]
    prompts: Dict[str, str]
    colors: Dict[str, str]
    pv_names: FrozenSet[str] = field(default_factory=frozenset)
    no_tag_names: FrozenSet[str] = field(default_factory=frozenset)
    tags: bool = False
    radius_names: Tuple[str, ...] = ()
    label_values: Tuple[int, ...] = ()
    clip_regions: Tuple[str, ...] = ()
    legacy_keys: FrozenSet[str] = field(default_factory=frozenset)

    @property
    def count(self) -> int:
        return len(self.order)

    @property
    def persists(self) -> bool:
        """Whether this set is ever written to a file.

        False for a profile with no points. Nothing would be written but
        the key itself, and a key carrying an empty mapping states that
        the mesh has a set it does not have — which is exactly what a
        reader would then act on.
        """
        return bool(self.order)

    @property
    def read_keys(self) -> Tuple[str, ...]:
        """Keys to look for when loading, current spelling first.

        The legacy spellings follow, so a file written before the set was
        renamed still loads; saving then writes ``export_key``, which is
        what converts it.
        """
        return (self.export_key, *sorted(self.legacy_keys))


# -- The six-seed left-atrium workflow ----------------------------------
SEED_PROMPT: Dict[str, str] = {
    "LSPV": "Click INSIDE the left superior pulmonary vein (LSPV)",
    "LIPV": "Click INSIDE the left inferior pulmonary vein (LIPV)",
    "RSPV": "Click INSIDE the right superior pulmonary vein (RSPV)",
    "RIPV": "Click INSIDE the right inferior pulmonary vein (RIPV)",
    "LAA":  "Click INSIDE the left atrial appendage (LAA)",
    "MV":   "Click NEAR the center of the mitral valve (MV)",
}

SEED_COLOR: Dict[str, str] = {
    "LSPV": "#e41a1c",
    "LIPV": "#377eb8",
    "RSPV": "#4daf4a",
    "RIPV": "#984ea3",
    "LAA":  "#ff7f00",
    "MV":   "#f7e111",
}

SEED_LA_PROFILE = SeedProfile(
    type_id="seed_LA",
    label="seed_LA",
    export_key="seed_LA",
    order=("LSPV", "LIPV", "RSPV", "RIPV", "LAA", "MV"),
    prompts=SEED_PROMPT,
    colors=SEED_COLOR,
    pv_names=frozenset(("LSPV", "LIPV", "RSPV", "RIPV")),
    no_tag_names=frozenset(("MV",)),
    tags=True,
    # MV is not tagged, so it has no radius to cap: the five that are.
    radius_names=("LSPV", "LIPV", "RSPV", "RIPV", "LAA"),
    label_values=LA_LABEL_VALUES,
    clip_regions=LA_CLIP_REGIONS,
    # "seeds" is what every file written before the rename carries; "seed"
    # is accepted alongside it because that was the set's name in the UI,
    # and a hand-written sidecar may well spell it that way.
    legacy_keys=frozenset(("seeds", "seed")),
)


# -- Left-atrium UAC landmarks ------------------------------------------
LANDMARKS_LA_UAC_PROMPT: Dict[str, str] = {
    "LSPV_BODY_JCN": "Click at the LSPV–body junction (LSPV_BODY_JCN)",
    "RSPV_BODY_JCN": "Click at the RSPV–body junction (RSPV_BODY_JCN)",
    "SEPTAL_WALL":   "Click on the septal wall (SEPTAL_WALL)",
    "LATERAL_WALL":  "Click on the lateral wall (LATERAL_WALL)",
}

LANDMARKS_LA_UAC_COLOR: Dict[str, str] = {
    "LSPV_BODY_JCN": "#1b9e77",
    "RSPV_BODY_JCN": "#d95f02",
    "SEPTAL_WALL":   "#7570b3",
    "LATERAL_WALL":  "#e7298a",
}

LANDMARKS_LA_UAC_PROFILE = SeedProfile(
    type_id="landmarks_LA_UAC",
    label="landmarks_LA_UAC",
    export_key="landmarks_LA_UAC",
    order=("LSPV_BODY_JCN", "RSPV_BODY_JCN", "SEPTAL_WALL", "LATERAL_WALL"),
    prompts=LANDMARKS_LA_UAC_PROMPT,
    colors=LANDMARKS_LA_UAC_COLOR,
    pv_names=frozenset(),
    no_tag_names=frozenset(),
    tags=False,
    # Picking landmarks is all this set does. No radii (it does not tag),
    # no labels (no manual correction) and no clip regions: the tools that
    # act on a tagging belong to the seed set that produced it, and a
    # landmark set is not that. Switch back to seed_LA to correct or clip.
    radius_names=(),
    label_values=(),
    clip_regions=(),
)


# -- Right atrium (signpost) --------------------------------------------
SEED_RA_PROFILE = SeedProfile(
    type_id="seed_RA",
    label="seed_RA",
    export_key="seed_RA",
    # No points yet. Choosing this type says what the mesh is and turns
    # the left-atrial tools off; the right-atrial seeds fill the slot when
    # they are defined.
    order=(),
    prompts={},
    colors={},
    tags=False,
    radius_names=(),
    label_values=(),
    clip_regions=(),
)


# Dropdown / registry order: the default set first.
SEED_PROFILE_ORDER: Tuple[SeedProfile, ...] = (
    SEED_LA_PROFILE,
    LANDMARKS_LA_UAC_PROFILE,
    SEED_RA_PROFILE,
)
SEED_PROFILES: Dict[str, SeedProfile] = {p.type_id: p for p in SEED_PROFILE_ORDER}
DEFAULT_PROFILE = SEED_LA_PROFILE


def profile_for_key(key: str) -> "SeedProfile | None":
    """The profile a file's point-set *key* belongs to, or ``None``.

    Current spellings are matched first across every profile, then the
    legacy ones, so a new key can never be shadowed by another profile's
    old alias.
    """
    for profile in SEED_PROFILE_ORDER:
        if key == profile.export_key:
            return profile
    for profile in SEED_PROFILE_ORDER:
        if key in profile.legacy_keys:
            return profile
    return None


__all__ = [
    "SeedProfile",
    "SEED_LA_PROFILE",
    "LANDMARKS_LA_UAC_PROFILE",
    "SEED_RA_PROFILE",
    "SEED_PROFILE_ORDER",
    "SEED_PROFILES",
    "DEFAULT_PROFILE",
    "SEED_PROMPT",
    "SEED_COLOR",
    "LA_LABEL_VALUES",
    "LA_CLIP_REGIONS",
    "profile_for_key",
]
