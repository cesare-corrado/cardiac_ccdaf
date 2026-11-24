"""
atrial_labels.py
================
The ``elemTag`` labels of the atria, in one place: the numbers the tagging
writes (and every export carries), the names the legend shows and the
colours it draws them in. Tagging, manual correction, clipping, the legend
and the atrial coordinates all read this table, so what is written and what
is shown cannot drift apart.

Left atrium: body 1, LSPV 11, LIPV 13, RSPV 15, RIPV 17, LAA 19.
Right atrium: body 2, SVC 21, IVC 23, CS 25, RAA 27.

The two bodies have different numbers so a mesh holding both atria keeps
them apart. The valves (MV, TV) carry no label of their own: they are
openings in their atrium's body.

Pure Python on purpose (no numpy/pyvista), so the light GUI modules can
import it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Tuple


@dataclass(frozen=True)
class Atrium:
    """One atrium's labels."""

    name: str                       # "LA" / "RA"
    body: int
    regions: Mapping[str, int]      # tagged regions, in tagging order
    valve: str                      # the valve opening, which sits on the body
    colors: Mapping[int, str]       # legend colour of every label, body included

    @property
    def values(self) -> Tuple[int, ...]:
        """Every label, regions first, then the body."""
        return (*self.regions.values(), self.body)

    def name_of(self, tag: int) -> str:
        if tag == self.body:
            return f"{self.name} body"
        for k, v in self.regions.items():
            if v == tag:
                return k
        raise KeyError(tag)

    def tag_of(self, region: str) -> int:
        """The label a region (or the valve, which is body) carries."""
        if region == self.valve or region == "body":
            return self.body
        return int(self.regions[region])


LA = Atrium(
    name="LA", body=1,
    regions={"LSPV": 11, "LIPV": 13, "RSPV": 15, "RIPV": 17, "LAA": 19},
    valve="MV",
    colors={1: "#d9d9d9", 11: "#e41a1c", 13: "#377eb8", 15: "#4daf4a",
            17: "#984ea3", 19: "#ff7f00"},
)

RA = Atrium(
    name="RA", body=2,
    regions={"SVC": 21, "IVC": 23, "CS": 25, "RAA": 27},
    valve="TV",
    # Distinct from every LA hue, and the body a different grey, so a mesh
    # holding both atria shows which tissue is which.
    colors={2: "#c6dbef", 21: "#a65628", 23: "#f781bf", 25: "#17becf",
            27: "#e6ab02"},
)

ATRIA: Tuple[Atrium, ...] = (LA, RA)

#: Every atrial label: its name and colour.
NAMES: Dict[int, str] = {t: a.name_of(t) for a in ATRIA for t in a.values}
COLORS: Dict[int, str] = {t: c for a in ATRIA for t, c in a.colors.items()}
BODIES: Tuple[int, ...] = tuple(a.body for a in ATRIA)
#: The tagged (non-body) labels of both atria.
REGION_LABELS: Tuple[int, ...] = tuple(v for a in ATRIA for v in a.regions.values())


def atrium_of(tag: int) -> Optional[Atrium]:
    """The atrium a label belongs to, or ``None``."""
    for a in ATRIA:
        if tag in a.values:
            return a
    return None


def region_tag(region: str) -> Optional[int]:
    """The label a named region carries, whichever atrium it is in; a valve
    (MV, TV) is its atrium's body. ``None`` for an unknown name."""
    for a in ATRIA:
        if region in a.regions or region == a.valve:
            return a.tag_of(region)
    return None


def atria_present(tags: Iterable[int]) -> Tuple[Atrium, ...]:
    """The atria whose labels appear in *tags*; a mesh with only the LA
    body (every fresh mesh) counts as the left atrium."""
    present = {int(t) for t in tags}
    out = tuple(a for a in ATRIA if present & set(a.values))
    return out


def legend_atria(tags: Iterable[int]) -> Optional[Tuple[Atrium, ...]]:
    """The atria a set of labels is a tagging of, or ``None`` when it is
    not an atrial tagging at all.

    An atrium counts when one of its *regions* is present: a body number
    alone is not enough, because other meshes use small numbers too (a
    ventricular mesh labels the left ventricle 1 and the right ventricle 2).
    A fresh mesh, body 1 everywhere, is the left atrium. Every label present
    must belong to an atrium that counts."""
    present = {int(t) for t in tags}
    if present == {LA.body}:
        return (LA,)
    atria = tuple(a for a in ATRIA if present & set(a.regions.values()))
    if not atria or not present <= {v for a in atria for v in a.values}:
        return None
    return atria


def body_for(tags: Iterable[int]) -> int:
    """The body label to fill with: the RA body when only right-atrial
    labels are present, else the LA body."""
    atria = atria_present(tags)
    return RA.body if atria == (RA,) else LA.body


__all__ = ["Atrium", "LA", "RA", "ATRIA", "NAMES", "COLORS", "BODIES",
           "REGION_LABELS", "region_tag", "legend_atria", "atrium_of", "atria_present", "body_for"]
