"""
test_seed_state_machine.py
==========================
Tests for the pure SeedStateMachine logic.

Exercises only the state-machine layer — no VTK, PyVista, or mesh geometry.

Run with pytest:
    pytest tests/test_seed_state_machine.py

Run as a standalone script:
    python tests/test_seed_state_machine.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np

from ccdaf.core.seed_state_machine import (
    SEED_ORDER,
    PV_NAMES,
    Seed,
    SeedStateMachine,
    CommitResult,
)


def make_seed(name: str, vid: int, xyz=(0.0, 0.0, 0.0)) -> Seed:
    return Seed(name=name, vertex_id=vid, xyz=np.asarray(xyz, dtype=float))


def test_order_enforcement():
    sm = SeedStateMachine()
    assert sm.try_commit(make_seed("LIPV", 1)) is CommitResult.OUT_OF_ORDER
    assert sm.next_name() == "LSPV"
    for i, name in enumerate(SEED_ORDER):
        assert sm.next_name() == name
        assert sm.try_commit(make_seed(name, 100 + i)) is CommitResult.OK
    assert sm.is_complete
    assert sm.next_name() is None
    assert list(sm.seeds.keys()) == list(SEED_ORDER)


def test_no_commit_after_complete():
    sm = SeedStateMachine()
    for i, name in enumerate(SEED_ORDER):
        assert sm.try_commit(make_seed(name, 10 + i)) is CommitResult.OK
    assert sm.try_commit(make_seed("LSPV", 999)) is CommitResult.NO_ACTIVE_SLOT


def test_duplicate_vertex_rejected():
    sm = SeedStateMachine()
    assert sm.try_commit(make_seed("LSPV", 42)) is CommitResult.OK
    assert sm.try_commit(make_seed("LIPV", 42)) is CommitResult.DUPLICATE_VID
    assert sm.next_name() == "LIPV"


def test_history_deepcopy_isolation():
    sm = SeedStateMachine()
    xyz = np.array([1.0, 2.0, 3.0])
    assert sm.try_commit(Seed("LSPV", 7, xyz)) is CommitResult.OK
    xyz[0] = 999.0
    snap = sm._history[-1]["LSPV"].xyz
    assert snap[0] == 1.0, "history snapshot must be deep-copied"


def test_undo_restores_prior_state():
    sm = SeedStateMachine()
    sm.try_commit(make_seed("LSPV", 1))
    sm.try_commit(make_seed("LIPV", 2))
    sm.try_commit(make_seed("RSPV", 3))
    assert sm.history_depth == 3
    removed = sm.undo()
    assert removed == "RSPV"
    assert sm.next_name() == "RSPV"
    assert sm.history_depth == 2
    assert set(sm.seeds.keys()) == {"LSPV", "LIPV"}
    assert sm.try_commit(make_seed("RSPV", 30)) is CommitResult.OK
    assert sm.seeds["RSPV"].vertex_id == 30


def test_undo_empty_is_noop():
    sm = SeedStateMachine()
    assert sm.undo() is None
    assert sm.next_name() == "LSPV"


def test_full_undo_chain():
    sm = SeedStateMachine()
    for i, name in enumerate(SEED_ORDER):
        sm.try_commit(make_seed(name, i))
    for name in reversed(SEED_ORDER):
        assert sm.undo() == name
    assert sm.next_name() == "LSPV"
    assert sm.history_depth == 0
    assert not sm.seeds


def test_reset_clears_history():
    sm = SeedStateMachine()
    sm.try_commit(make_seed("LSPV", 1))
    sm.try_commit(make_seed("LIPV", 2))
    sm.reset()
    assert sm.history_depth == 0
    assert sm.next_name() == "LSPV"
    assert not sm.seeds


def test_queue_is_derived():
    sm = SeedStateMachine()
    assert sm.queue() == list(SEED_ORDER)
    sm.try_commit(make_seed("LSPV", 1))
    assert sm.queue() == list(SEED_ORDER[1:])
    sm.undo()
    assert sm.queue() == list(SEED_ORDER)


def test_pv_names_constant():
    assert PV_NAMES == ("LSPV", "LIPV", "RSPV", "RIPV")
    assert all(n in SEED_ORDER for n in PV_NAMES)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
