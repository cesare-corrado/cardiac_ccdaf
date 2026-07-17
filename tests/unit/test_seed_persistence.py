"""
test_seed_persistence.py
========================
Tests for saving and reloading a seed selection.

The contract:

* a saved seed is a name and a coordinate — never a vertex id, because
  ids do not survive a clip or a refinement;
* both formats round-trip exactly: the ``.json`` sidecar, and the
  ``.pkl`` bundle that carries the surface dict beside the same
  ``"seeds"`` key (the EAM export pickle's convention);
* loading rejects files without a ``"seeds"`` key, non-3-vector
  coordinates, and unknown suffixes — loudly, not by guessing;
* ``apply_positions`` rebuilds the selection through the same snapping
  and validation as clicking, completes on a mesh of a *different*
  resolution (the id-independence the coordinates buy), and stops at the
  first missing seed leaving the rest resumable.

Uses the synthetic atrium of test_seed_geometry; no display, no Qt.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pytest
import pyvista as pv

from ccdaf.core.seed_io import SEEDS_KEY, load_seeds, save_seeds
from ccdaf.core.seed_state_machine import SEED_ORDER
from ccdaf.interaction.seed_selector import SeedSelector


def make_mesh(theta=60, phi=60, reach=9.0, cone_deg=14.0) -> pv.PolyData:
    """One connected surface: a body sphere with four pulled-out PV tubes.

    test_seed_geometry's merge-of-spheres will not do here — merge only
    concatenates, and validate_pv rejects a geodesically disconnected
    protrusion. Pulling vertices outward keeps the surface connected, and
    reach=9 keeps the tips past the PV outlier fence at every resolution
    used below.
    """
    m = pv.Sphere(radius=10.0, theta_resolution=theta, phi_resolution=phi)
    pts = np.asarray(m.points).copy()
    n = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    cos_cone = np.cos(np.radians(cone_deg))
    for d in ([1.0, 0, 0], [-1.0, 0, 0], [0, 1.0, 0], [0, -1.0, 0]):
        d = np.asarray(d, dtype=float)
        sel = (n @ d) > cos_cone
        t = (n[sel] @ d - cos_cone) / (1.0 - cos_cone)
        pts[sel] += (t[:, None] * reach) * d
    m.points = pts
    return m


POSITIONS = {
    "LSPV": [18.9, 0.0, 0.0],
    "LIPV": [-18.9, 0.0, 0.0],
    "RSPV": [0.0, 18.9, 0.0],
    "RIPV": [0.0, -18.9, 0.0],
    "LAA":  [0.0, 0.0, 10.0],
    "MV":   [0.0, 0.0, -10.0],
}


class StubPlotter:
    """Just enough plotter for markers, HUD and picking calls to no-op."""

    def add_mesh(self, *a, **k): return object()
    def add_text(self, *a, **k): return object()
    def add_point_labels(self, *a, **k): return object()
    def remove_actor(self, *a, **k): pass
    def enable_point_picking(self, *a, **k): pass
    def disable_picking(self): pass
    def render(self): pass


def _selector(mesh, complete_log):
    return SeedSelector(
        mesh=mesh, plotter=StubPlotter(),
        on_complete=lambda seeds: complete_log.append(dict(seeds)),
    )


def test_json_round_trip(tmp_path):
    path = tmp_path / "case.seeds.json"
    save_seeds(path, POSITIONS)
    back = load_seeds(path)
    assert set(back) == set(POSITIONS)
    for name in POSITIONS:
        assert np.allclose(back[name], POSITIONS[name])


def test_pickle_bundles_surface_beside_seeds(tmp_path):
    import pickle
    path = tmp_path / "case.pkl"
    save_seeds(path, POSITIONS, mesh=make_mesh())
    with open(path, "rb") as fh:
        raw = pickle.load(fh)
    assert set(raw) == {"surface", SEEDS_KEY}
    assert {"X", "Tri"} <= set(raw["surface"])
    back = load_seeds(path)
    assert np.allclose(back["MV"], POSITIONS["MV"])


def test_pickle_requires_a_mesh(tmp_path):
    with pytest.raises(ValueError):
        save_seeds(tmp_path / "case.pkl", POSITIONS, mesh=None)


def test_load_rejects_malformed_files(tmp_path):
    no_key = tmp_path / "no_key.json"
    no_key.write_text('{"points": []}')
    with pytest.raises(ValueError):
        load_seeds(no_key)

    bad_vec = tmp_path / "bad.json"
    bad_vec.write_text('{"seeds": {"MV": [1.0, 2.0]}}')
    with pytest.raises(ValueError):
        load_seeds(bad_vec)

    with pytest.raises(ValueError):
        save_seeds(tmp_path / "seeds.vtk", POSITIONS)
    with pytest.raises(ValueError):
        load_seeds(tmp_path / "seeds.vtk")


def test_apply_positions_completes_and_fires(tmp_path):
    mesh = make_mesh()
    done: list = []
    sel = _selector(mesh, done)
    problems = sel.apply_positions(POSITIONS)
    assert problems == []
    assert sel.is_complete
    assert len(done) == 1
    tagging = sel.seeds_for_tagging()
    assert set(tagging) == set(SEED_ORDER) - {"MV"}
    for vid in tagging.values():
        assert 0 <= vid < mesh.n_points


def test_saved_seeds_reload_onto_a_refined_mesh(tmp_path):
    coarse = make_mesh(theta=60, phi=60)
    done: list = []
    sel = _selector(coarse, done)
    assert sel.apply_positions(POSITIONS) == []
    path = tmp_path / "case.seeds.json"
    save_seeds(path, {n: s.xyz for n, s in sel.seeds.items()})

    fine = make_mesh(theta=100, phi=100)  # different vertex numbering entirely
    done2: list = []
    sel2 = _selector(fine, done2)
    assert sel2.apply_positions(load_seeds(path)) == []
    assert sel2.is_complete
    for name, seed in sel2.seeds.items():
        assert np.linalg.norm(seed.xyz - np.asarray(POSITIONS[name])) < 1.0


def test_apply_stops_at_the_first_missing_seed():
    mesh = make_mesh()
    done: list = []
    sel = _selector(mesh, done)
    partial = {k: v for k, v in POSITIONS.items() if k != "RSPV"}
    problems = sel.apply_positions(partial)
    assert problems == ["RSPV: not in the file"]
    assert not sel.is_complete
    assert done == []
    # LSPV and LIPV landed before the failure; the queue resumes at RSPV.
    assert set(sel.seeds) == {"LSPV", "LIPV"}
    assert sel.next_name() == "RSPV"
