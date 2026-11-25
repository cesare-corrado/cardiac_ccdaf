"""
tag_map
=======
What each ``elemTag`` number of a mesh *means*: its name ("LV", "LA body",
"LSPV ring"), the chamber it belongs to, and its role (wall, vein, cap, …).

Why a map
---------
Tag numbers are not universal. ccdaf's own atrial scheme numbers the left
atrial body 1 and the right 2 (:mod:`ccdaf.core.atrial_labels`), a
ventricular mesh calls its left ventricle 1 and its right 2, and
four-chamber cohorts number their chambers, valves, vessel caps and vein
rings in schemes of their own, which do not even agree with each other
(the ring numbered 18 is the LAA ring in one cohort and the LSPV ring in
another). A number alone therefore cannot say what the tissue is.

The map is the answer the mesh does not carry: one :class:`TagEntry` per
tag number. Actions read the chamber and the role, never the number; the
legend shows the name.

Names, chambers and roles
-------------------------
:data:`VOCABULARY` lists the names ccdaf knows. Each has a fixed legend
colour, a chamber and a role, so picking the name fills in the rest. A
name typed by the user starts with no chamber and no role: actions ignore
that tag until both are set.

The chamber is one of :data:`CHAMBERS` ("LA", "LV+LA", or "" for none).
The role is one of :data:`ROLES`:

==========  ==============================================================
wall        myocardium of its chamber
layer       wall tissue with its own conduction (a fast endocardial layer);
            counts as wall
vein        a vein, or the ring of tissue left where it was clipped
appendage   the atrial appendage, or its ring when it was clipped
cap         a disc closing a vein or the appendage; not tissue, its edge is
            an opening
valve       a valve plane; not tissue, its edge is the valve rim
vessel      the aorta or the pulmonary artery; not part of a chamber
other       anything actions should leave alone
==========  ==============================================================

Where it lives
--------------
On the dataset's field data, as one JSON string under :data:`FIELD`, so it
travels with the mesh through a VTK file. A pickle bundle stores it under
its own key (see :mod:`ccdaf.core.eam_export`). :func:`save_file` and
:func:`load_file` keep it in a JSON file of its own, to name the first mesh
of a cohort and reuse the names on the rest.

Proposed, then confirmed
------------------------
:func:`propose` fills in what can be told without asking: a mesh in ccdaf's
atrial scheme names itself, and a labelled ventricular volume with two tags
gets LV and RV from which tag lines the LV endocardium. Anything else stays
unnamed and is shown by its number. Only a map the user has confirmed is
written to the mesh.

Pure numpy; no pyvista import, so the GUI can import it cheaply.
"""
from __future__ import annotations

import json
from typing import Dict, Iterable, List, Mapping, NamedTuple, Optional, Tuple, Union

import numpy as np

from ccdaf.core import atrial_labels

#: Field-data key the map is stored under.
FIELD: str = "tagMap"

# Roles.
WALL, LAYER, VEIN, APPENDAGE = "wall", "layer", "vein", "appendage"
CAP, VALVE, VESSEL, OTHER = "cap", "valve", "vessel", "other"
#: Every role, in the order the dialog offers them ("" is none).
ROLES: Tuple[str, ...] = ("", WALL, LAYER, VEIN, APPENDAGE, CAP, VALVE,
                          VESSEL, OTHER)
#: Roles that are tissue of their chamber.
TISSUE_ROLES: Tuple[str, ...] = (WALL, LAYER, VEIN, APPENDAGE)

#: Every chamber value, in the order the dialog offers them ("" is none).
#: A tag on the boundary of two chambers names both, joined by "+".
CHAMBERS: Tuple[str, ...] = ("", "LV", "RV", "LA", "RA", "LV+RV", "LV+LA",
                             "RV+RA", "LA+RA")


class TagEntry(NamedTuple):
    """What one tag is."""

    name: str
    chamber: str = ""
    role: str = ""

    @property
    def chambers(self) -> Tuple[str, ...]:
        """The chambers this tag belongs to, split."""
        return tuple(c for c in self.chamber.split("+") if c)


class Meaning(NamedTuple):
    """A name ccdaf knows: its colour, chamber, role, and the atrial region
    it counts as ("" when it is not atrial vein or appendage tissue)."""

    color: str
    chamber: str
    role: str
    region: str = ""


LV: str = "LV"
RV: str = "RV"
#: A generic cap or valve, for a mesh that does not say which.
CAP_NAME: str = "Cap / valve"
IGNORE: str = "Ignore"


def _atrial_vocabulary() -> Dict[str, Meaning]:
    out: Dict[str, Meaning] = {}
    for a in atrial_labels.ATRIA:
        out[atrial_labels.NAMES[a.body]] = Meaning(a.colors[a.body], a.name, WALL)
        for region, tag in a.regions.items():
            role = APPENDAGE if region in ("LAA", "RAA") else VEIN
            out[region] = Meaning(a.colors[tag], a.name, role, region)
    return out


_ATRIAL = _atrial_vocabulary()
_C = {r: m.color for r, m in _ATRIAL.items()}

#: Every name ccdaf knows, in the order the dialog lists them.
VOCABULARY: Dict[str, Meaning] = {
    LV: Meaning("#f4a582", "LV", WALL),
    RV: Meaning("#92c5de", "RV", WALL),
    "FEC_LV": Meaning("#d6604d", "LV", LAYER),
    "FEC_RV": Meaning("#4393c3", "RV", LAYER),
    "FEC_SV": Meaning("#b2abd2", "LV+RV", LAYER),
    **_ATRIAL,
    "BB": Meaning("#35978f", "LA+RA", WALL),
    # Vein rings are the vein tissue left at a clipped vein: the vein's
    # own colour, since they are that vein.
    "LSPV ring": Meaning(_C["LSPV"], "LA", VEIN, "LSPV"),
    "LIPV ring": Meaning(_C["LIPV"], "LA", VEIN, "LIPV"),
    "RSPV ring": Meaning(_C["RSPV"], "LA", VEIN, "RSPV"),
    "RIPV ring": Meaning(_C["RIPV"], "LA", VEIN, "RIPV"),
    "LAA ring": Meaning(_C["LAA"], "LA", APPENDAGE, "LAA"),
    "SVC ring": Meaning(_C["SVC"], "RA", VEIN, "SVC"),
    "IVC ring": Meaning(_C["IVC"], "RA", VEIN, "IVC"),
    # Caps close the openings: a dark shade of the vein they close.
    "LSPV cap": Meaning("#99000d", "LA", CAP),
    "LIPV cap": Meaning("#08306b", "LA", CAP),
    "RSPV cap": Meaning("#00441b", "LA", CAP),
    "RIPV cap": Meaning("#3f007d", "LA", CAP),
    "LAA cap": Meaning("#7f2704", "LA", CAP),
    "SVC cap": Meaning("#543005", "RA", CAP),
    "IVC cap": Meaning("#7a0177", "RA", CAP),
    "MV": Meaning("#fee08b", "LV+LA", VALVE),
    "TV": Meaning("#d9ef8b", "RV+RA", VALVE),
    "AV": Meaning("#e08214", "LV", VALVE),
    "Pulmonary valve": Meaning("#8073ac", "RV", VALVE),
    "AV plane": Meaning("#525252", "", OTHER),
    "Ao": Meaning("#b2182b", "", VESSEL),
    "PArt": Meaning("#2166ac", "", VESSEL),
    CAP_NAME: Meaning("#969696", "", CAP),
    IGNORE: Meaning("#f0f0f0", "", OTHER),
}

#: The names the dialog offers, in its order. Free text is allowed too.
MEANINGS: Tuple[str, ...] = tuple(VOCABULARY)

#: How the four-chamber cohorts' ``{name: tag}`` files name each region.
#: Their bare vein names ("LSPV", "LAA", "SVC") are the closing caps; the
#: vein tissue is the ring.
GROUP_NAMES: Dict[str, str] = {
    "LV": LV, "RV": RV, "LA": "LA body", "RA": "RA body",
    "Ao": "Ao", "PArt": "PArt",
    "MV": "MV", "TV": "TV", "AV": "AV", "PV": "Pulmonary valve",
    "LSPV": "LSPV cap", "LIPV": "LIPV cap", "RSPV": "RSPV cap",
    "RIPV": "RIPV cap", "LAA": "LAA cap", "SVC": "SVC cap", "IVC": "IVC cap",
    "LSPV_ring": "LSPV ring", "LIPV_ring": "LIPV ring",
    "RSPV_ring": "RSPV ring", "RIPV_ring": "RIPV ring",
    "LAA_ring": "LAA ring", "SVC_ring": "SVC ring", "IVC_ring": "IVC ring",
    "FEC_LV": "FEC_LV", "FEC": "FEC_LV", "FEC_RV": "FEC_RV",
    "FEC_SV": "FEC_SV", "BB": "BB", "AV_plane": "AV plane",
}

#: Legend colour of each ventricle (kept for callers that ask directly).
VENTRICLE_COLORS: Dict[str, str] = {LV: VOCABULARY[LV].color,
                                    RV: VOCABULARY[RV].color}

TagMap = Dict[int, TagEntry]
_Value = Union[str, TagEntry]


# ---------------------------------------------------------------------
# Entries
# ---------------------------------------------------------------------
def entry(value: _Value) -> TagEntry:
    """*value* as a :class:`TagEntry`: a bare name takes the chamber and
    role :data:`VOCABULARY` gives it (none for a name it does not know)."""
    if isinstance(value, TagEntry):
        return value
    if isinstance(value, Mapping):
        return TagEntry(str(value.get("name", "")).strip(),
                        str(value.get("chamber", "") or ""),
                        str(value.get("role", "") or ""))
    name = str(value).strip()
    known = VOCABULARY.get(name)
    return TagEntry(name, known.chamber, known.role) if known else TagEntry(name)


def normalise(mapping: Optional[Mapping[int, _Value]]) -> TagMap:
    """*mapping* with every value a :class:`TagEntry`, blank names dropped."""
    out: TagMap = {}
    for k, v in (mapping or {}).items():
        e = entry(v)
        if e.name:
            out[int(k)] = e
    return out


# ---------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------
def read(dataset) -> Optional[TagMap]:
    """The map stored on *dataset*, or ``None`` when it has none."""
    if dataset is None or FIELD not in dataset.field_data:
        return None
    raw = np.asarray(dataset.field_data[FIELD]).ravel()
    if raw.size == 0:
        return None
    try:
        return decode(str(raw[0])) or None
    except (ValueError, TypeError):
        return None


def write(dataset, mapping: Optional[Mapping[int, _Value]]) -> None:
    """Store *mapping* on *dataset*; ``None`` or empty removes it."""
    if dataset is None:
        return
    if not mapping:
        if FIELD in dataset.field_data:
            del dataset.field_data[FIELD]
        return
    dataset.field_data[FIELD] = np.array([encode(mapping)])


def _as_json(mapping: Mapping[int, _Value]) -> Dict[str, dict]:
    return {str(int(k)): entry(v)._asdict()
            for k, v in sorted(mapping.items(), key=lambda kv: int(kv[0]))}


def encode(mapping: Mapping[int, _Value]) -> str:
    """JSON text of *mapping*, keys as strings (JSON has no integer keys)."""
    return json.dumps(_as_json(mapping))


def decode(text: str) -> TagMap:
    """The map from :func:`encode`'s text, or from one of the file formats
    :func:`from_json` reads. Raises ``ValueError`` if malformed."""
    return from_json(json.loads(text))


def from_json(data) -> TagMap:
    """A map from parsed JSON, in any of the formats ccdaf reads:

    * ``{"21": {"name": …, "chamber": …, "role": …}}``: ccdaf's own;
    * ``{"21": "LSPV ring"}``: names only, chamber and role from
      :data:`VOCABULARY`;
    * ``{"LSPV_ring": 21}``: name to number, as the four-chamber cohorts
      ship it; their names are translated through :data:`GROUP_NAMES`.
      When two names share a number, the first wins.
    """
    if not isinstance(data, dict):
        raise ValueError("a tag map is a JSON object")
    if data and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                    for v in data.values()):
        out: TagMap = {}
        for name, tag in data.items():
            if int(tag) not in out:
                e = entry(GROUP_NAMES.get(str(name), str(name)))
                if e.name:
                    out[int(tag)] = e
        return out
    try:
        return normalise({int(k): v for k, v in data.items()})
    except (TypeError, ValueError) as exc:
        raise ValueError(f"not a tag map: {exc}") from None


def save_file(path: str, mapping: Mapping[int, _Value]) -> None:
    """Write *mapping* to a JSON file in ccdaf's own format."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_as_json(mapping), fh, indent=2)
        fh.write("\n")


def load_file(path: str) -> TagMap:
    """Read a tag map from a JSON file (see :func:`from_json`)."""
    with open(path, encoding="utf-8") as fh:
        return from_json(json.load(fh))


# ---------------------------------------------------------------------
# Proposal and lookup
# ---------------------------------------------------------------------
def propose(tags: Iterable[int], ventricles: Optional[Tuple[int, int]] = None
            ) -> TagMap:
    """The meanings that can be told without asking, for the *tags* present.

    ``ventricles`` is ``(lv, rv)`` when the caller could tell them apart
    (a labelled ventricular volume, see :func:`ccdaf.core.uvc.lv_rv_tags`).
    Otherwise a tagging in ccdaf's atrial scheme names itself; anything
    else is left out, which the legend shows as the bare number.
    """
    present = sorted({int(t) for t in tags})
    if ventricles is not None and set(ventricles) == set(present):
        lv, rv = ventricles
        return {int(lv): entry(LV), int(rv): entry(RV)}
    if atrial_labels.legend_atria(present):
        return {t: entry(atrial_labels.NAMES[t]) for t in present
                if t in atrial_labels.NAMES}
    return {}


def name_of(tag: int, mapping: Optional[Mapping[int, _Value]]) -> str:
    """The legend name of *tag*: its name, or the number itself."""
    if mapping and int(tag) in mapping:
        return entry(mapping[int(tag)]).name
    return str(int(tag))


def color_of(name: str) -> Optional[str]:
    """The fixed legend colour of a known name, or ``None``."""
    known = VOCABULARY.get(name)
    return known.color if known else None


def follows_atrial_scheme(mapping: Optional[Mapping[int, _Value]],
                          tags: Iterable[int]) -> bool:
    """Whether *mapping* names every tag in *tags* as ccdaf's atrial scheme
    does (or says nothing about it). Such a mesh keeps the atrial legend."""
    if not mapping:
        return True
    names = {int(k): entry(v).name for k, v in mapping.items()}
    return all(names.get(int(t), atrial_labels.NAMES.get(int(t)))
               == atrial_labels.NAMES.get(int(t)) for t in tags)


def tags_meaning(mapping: Optional[Mapping[int, _Value]], name: str) -> List[int]:
    """Every tag *mapping* gives the name *name*, ascending."""
    return sorted(int(t) for t, v in (mapping or {}).items()
                  if entry(v).name == name)


def tags_of(mapping: Optional[Mapping[int, _Value]], chamber: str,
            roles: Iterable[str] = TISSUE_ROLES) -> List[int]:
    """Every tag that belongs to *chamber* with one of *roles*, ascending:
    ``tags_of(m, "LA")`` is the left atrial tissue, caps and valves left
    out."""
    want = set(roles)
    return sorted(int(t) for t, v in (mapping or {}).items()
                  if chamber in entry(v).chambers and entry(v).role in want)


def atrial_region(e: _Value) -> str:
    """The atrial region a tag counts as: the vein or appendage it is, the
    atrium's body for wall tissue, or "" when it is not atrial tissue."""
    e = entry(e)
    atria = [a for a in atrial_labels.ATRIA if a.name in e.chambers]
    if len(atria) != 1 or e.role not in TISSUE_ROLES:
        return ""
    a = atria[0]
    if e.role in (WALL, LAYER):
        return atrial_labels.NAMES[a.body]
    known = VOCABULARY.get(e.name)
    if known and known.region and known.chamber == a.name and known.role == e.role:
        return known.region
    if e.role == APPENDAGE:
        return next(r for r in a.regions if r.endswith("AA"))
    return ""


__all__ = [
    "FIELD", "WALL", "LAYER", "VEIN", "APPENDAGE", "CAP", "VALVE", "VESSEL",
    "OTHER", "ROLES", "TISSUE_ROLES", "CHAMBERS", "TagEntry", "Meaning",
    "LV", "RV", "CAP_NAME", "IGNORE", "VOCABULARY", "MEANINGS", "GROUP_NAMES",
    "VENTRICLE_COLORS", "entry", "normalise", "read", "write", "encode",
    "decode", "from_json", "save_file", "load_file", "propose", "name_of",
    "color_of", "follows_atrial_scheme", "tags_meaning", "tags_of",
    "atrial_region",
]
