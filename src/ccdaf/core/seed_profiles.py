"""
Seed profiles
=============

A *seed profile* describes one selectable set of surface points: which
points to pick and in what order, how each is prompted and coloured,
which are pulmonary veins (so the anatomical prior applies), which are
excluded from region tagging, and the key the set is written under in a
saved pickle/JSON.

Two profiles ship:

* :data:`SEED_PROFILE` — the original six-seed workflow
  (LSPV, LIPV, RSPV, RIPV, LAA, MV). It feeds region tagging and is
  written under the ``"seeds"`` key.
* :data:`LANDMARKS_LA_UAC_PROFILE` — four left-atrium landmarks used for
  a universal-atrial-coordinate step (LSPV_BODY_JCN, RSPV_BODY_JCN,
  SEPTAL_WALL, LATERAL_WALL). Picked, saved and exported exactly like
  seeds, but under the ``"landmarks_LA_UAC"`` key and never tagged.

The two sets are independent: a session can complete both, and a saved
bundle can carry both keys at once.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Tuple


@dataclass(frozen=True)
class SeedProfile:
    """One selectable set of surface points and how it is handled.

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
        The points to pick, in strict acquisition order.
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

    @property
    def count(self) -> int:
        return len(self.order)


# -- The original six-seed workflow -------------------------------------
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

SEED_PROFILE = SeedProfile(
    type_id="seed",
    label="seed",
    export_key="seeds",
    order=("LSPV", "LIPV", "RSPV", "RIPV", "LAA", "MV"),
    prompts=SEED_PROMPT,
    colors=SEED_COLOR,
    pv_names=frozenset(("LSPV", "LIPV", "RSPV", "RIPV")),
    no_tag_names=frozenset(("MV",)),
    tags=True,
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
)


# Dropdown / registry order: the default set first.
SEED_PROFILE_ORDER: Tuple[SeedProfile, ...] = (
    SEED_PROFILE,
    LANDMARKS_LA_UAC_PROFILE,
)
SEED_PROFILES: Dict[str, SeedProfile] = {p.type_id: p for p in SEED_PROFILE_ORDER}
DEFAULT_PROFILE = SEED_PROFILE


__all__ = [
    "SeedProfile",
    "SEED_PROFILE",
    "LANDMARKS_LA_UAC_PROFILE",
    "SEED_PROFILE_ORDER",
    "SEED_PROFILES",
    "DEFAULT_PROFILE",
    "SEED_PROMPT",
    "SEED_COLOR",
]
