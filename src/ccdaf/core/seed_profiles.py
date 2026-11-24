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

Two profiles ship:

* :data:`SEED_LA_PROFILE` — the six-seed left-atrium workflow
  (LSPV, LIPV, RSPV, RIPV, LAA, MV). It feeds region tagging, offers the
  five radius factors, the six ``elemTag`` labels and the five clip
  regions, and is written under the ``"seed_LA"`` key.
* :data:`SEED_RA_PROFILE` — the five-seed right-atrium workflow (SVC, IVC,
  CS, RAA, TV), the same tools with the right-atrial labels (body 2, SVC
  21, IVC 23, CS 25, RAA 27; see :mod:`ccdaf.core.atrial_labels`), written
  under ``"seed_RA"``.

Choosing a profile is what decides which atrium every panel works on: the
Tagging, Manual-correction and Clipping panels show only that atrium's
radii, labels and regions.

A file that still carries the retired ``"landmarks_LA_UAC"`` set loads
unchanged; that set is ignored (the atrial coordinates need no manual
landmarks).

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

#: Right-atrial labels in dropdown order (vessels, appendage, body) and the
#: regions its clipping offers: the three vessels and the tricuspid valve.
RA_LABEL_VALUES: Tuple[int, ...] = (21, 23, 25, 27, 2)
RA_CLIP_REGIONS: Tuple[str, ...] = ("SVC", "IVC", "CS", "TV")


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
    atrium
        ``"LA"`` or ``"RA"``: whose labels tagging, correction and
        clipping use (:mod:`ccdaf.core.atrial_labels`).
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
    atrium: str = "LA"

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


# -- The five-seed right-atrium workflow --------------------------------
RA_SEED_PROMPT: Dict[str, str] = {
    "SVC": "Click INSIDE the superior vena cava (SVC)",
    "IVC": "Click INSIDE the inferior vena cava (IVC)",
    "CS":  "Click INSIDE the coronary sinus (CS)",
    "RAA": "Click INSIDE the right atrial appendage (RAA)",
    "TV":  "Click NEAR the center of the tricuspid valve (TV)",
}

RA_SEED_COLOR: Dict[str, str] = {
    "SVC": "#a65628",
    "IVC": "#f781bf",
    "CS":  "#17becf",
    "RAA": "#e6ab02",
    "TV":  "#bcbd22",
}

SEED_RA_PROFILE = SeedProfile(
    type_id="seed_RA",
    label="seed_RA",
    export_key="seed_RA",
    order=("SVC", "IVC", "CS", "RAA", "TV"),
    prompts=RA_SEED_PROMPT,
    colors=RA_SEED_COLOR,
    # The tubular vessels get the same snap-inside prior as the veins.
    pv_names=frozenset(("SVC", "IVC", "CS")),
    no_tag_names=frozenset(("TV",)),
    tags=True,
    radius_names=("SVC", "IVC", "CS", "RAA"),
    label_values=RA_LABEL_VALUES,
    clip_regions=RA_CLIP_REGIONS,
    atrium="RA",
)


# Dropdown / registry order: the default set first.
SEED_PROFILE_ORDER: Tuple[SeedProfile, ...] = (
    SEED_LA_PROFILE,
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
    "SEED_RA_PROFILE",
    "SEED_PROFILE_ORDER",
    "SEED_PROFILES",
    "DEFAULT_PROFILE",
    "SEED_PROMPT",
    "SEED_COLOR",
    "LA_LABEL_VALUES",
    "LA_CLIP_REGIONS",
    "RA_LABEL_VALUES",
    "RA_CLIP_REGIONS",
    "RA_SEED_PROMPT",
    "RA_SEED_COLOR",
    "profile_for_key",
]
