"""
SeedStateMachine
================

Pure-logic layer for the 6-seed selection workflow. No VTK, no PyVista,
no numpy geometry calls. The class is fully unit-testable in isolation.

Responsibilities
----------------
* Enforce SEED_ORDER (LSPV, LIPV, RSPV, RIPV, LAA, MV) strictly.
* Track the next expected seed slot (the queue is derived, never mutated
  by external callers).
* Accept/reject candidate commits:
    - OUT_OF_ORDER     — commit name does not match next expected slot
    - DUPLICATE_VID    — vertex id already used by an earlier seed
    - NO_ACTIVE_SLOT   — state machine is already complete
    - OK               — committed; history snapshot pushed
* Maintain a deepcopy history stack; undo pops the top snapshot and
  restores the previous full state. reset() clears both seeds and
  history.

The state machine holds only primitive/POD data, so deepcopy snapshots
are cheap and truly independent of live state.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


SEED_ORDER: Tuple[str, ...] = ("LSPV", "LIPV", "RSPV", "RIPV", "LAA", "MV")
PV_NAMES: Tuple[str, ...] = ("LSPV", "LIPV", "RSPV", "RIPV")


@dataclass
class Seed:
    name: str
    vertex_id: int
    xyz: np.ndarray  # shape (3,), float64


class CommitResult(Enum):
    OK = "ok"
    OUT_OF_ORDER = "out_of_order"
    DUPLICATE_VID = "duplicate_vid"
    NO_ACTIVE_SLOT = "no_active_slot"


class SeedStateMachine:
    """Pure state machine for sequential seed acquisition."""

    def __init__(self, order: Sequence[str] = SEED_ORDER) -> None:
        self._order: Tuple[str, ...] = tuple(order)
        self._seeds: Dict[str, Seed] = {}
        self._history: List[Dict[str, Seed]] = []

    # -- queries --------------------------------------------------------
    @property
    def order(self) -> Tuple[str, ...]:
        return self._order

    @property
    def seeds(self) -> Dict[str, Seed]:
        # Return a shallow copy so external mutation cannot corrupt state.
        return dict(self._seeds)

    @property
    def is_complete(self) -> bool:
        return all(n in self._seeds for n in self._order)

    @property
    def history_depth(self) -> int:
        return len(self._history)

    def next_name(self) -> Optional[str]:
        for n in self._order:
            if n not in self._seeds:
                return n
        return None

    def queue(self) -> List[str]:
        """Remaining seeds in canonical order — always derived, never stored."""
        return [n for n in self._order if n not in self._seeds]

    def committed_names(self) -> List[str]:
        return [n for n in self._order if n in self._seeds]

    # -- mutations ------------------------------------------------------
    def try_commit(self, seed: Seed) -> CommitResult:
        expected = self.next_name()
        if expected is None:
            return CommitResult.NO_ACTIVE_SLOT
        if seed.name != expected:
            return CommitResult.OUT_OF_ORDER
        for existing in self._seeds.values():
            if existing.vertex_id == seed.vertex_id:
                return CommitResult.DUPLICATE_VID

        self._seeds[seed.name] = seed
        # Full deepcopy: snapshots must be independent of future mutations
        # (including any in-place numpy edits to seed.xyz, however unlikely).
        self._history.append(copy.deepcopy(self._seeds))
        return CommitResult.OK

    def undo(self) -> Optional[str]:
        """Pop the top snapshot and restore the preceding state.

        Returns the name of the seed that was removed, or ``None`` if
        there is nothing to undo.
        """
        if not self._history:
            return None
        self._history.pop()
        prev = copy.deepcopy(self._history[-1]) if self._history else {}
        removed = [n for n in self._seeds if n not in prev]
        self._seeds = prev
        return removed[-1] if removed else None

    def reset(self) -> None:
        self._seeds.clear()
        self._history.clear()


__all__ = [
    "Seed",
    "SeedStateMachine",
    "CommitResult",
    "SEED_ORDER",
    "PV_NAMES",
]
