"""
test_seed_geometry.py
=====================
Tests for SeedGeometryResolver — determinism and correctness contracts.

Uses a synthetic sphere + tube mesh (no real atrial data required).

Run with pytest:
    pytest tests/test_seed_geometry.py

Run as a standalone script:
    python tests/test_seed_geometry.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pyvista as pv

from ccdaf.core.seed_geometry import SeedGeometryResolver, GeometryError, snap_point


def make_mesh(theta=40, phi=40):
    m = pv.Sphere(radius=10.0, theta_resolution=theta, phi_resolution=phi)
    for d in (
        np.array([1.0, 0, 0]),
        np.array([-1.0, 0, 0]),
        np.array([0, 1.0, 0]),
        np.array([0, -1.0, 0]),
    ):
        m = m.merge(pv.Sphere(radius=2.0, center=(10 * d).tolist()))
    return m


def test_snap_is_deterministic_same_mesh():
    mesh = make_mesh()
    p = np.array([10.0, 0.0, 0.0])
    ids = [snap_point(mesh, p) for _ in range(50)]
    assert len(set(ids)) == 1, f"snap_point not deterministic: got {set(ids)}"


def test_snap_deterministic_across_resolvers():
    mesh = make_mesh()
    r1 = SeedGeometryResolver(mesh)
    r2 = SeedGeometryResolver(mesh)
    probes = [
        np.array([10.0, 0.0, 0.0]),
        np.array([-10.0, 0.0, 0.0]),
        np.array([0.0, 10.0, 0.0]),
        np.array([0.0, -10.0, 0.0]),
        np.array([0.0, 0.0, 10.0]),
        np.array([5.77, 5.77, 5.77]),
    ]
    for p in probes:
        a = r1.snap(p).vertex_id
        b = r2.snap(p).vertex_id
        assert a == b, f"snap disagreed across resolvers for {p}: {a} vs {b}"


def test_snap_rejects_invalid_input():
    mesh = make_mesh()
    r = SeedGeometryResolver(mesh)
    try:
        r.snap(np.array([np.nan, 0.0, 0.0]))
    except GeometryError:
        pass
    else:
        raise AssertionError("non-finite pick must raise GeometryError")


def test_snap_point_bare_function():
    mesh = make_mesh()
    vid = snap_point(mesh, np.array([10.0, 0.0, 0.0]))
    assert 0 <= vid < mesh.n_points
    x, y, z = mesh.points[vid]
    assert abs(y) < 3 and abs(z) < 3 and x > 7, (
        f"snap landed off-protrusion: {(x, y, z)}"
    )


def test_anchor_is_near_centroid():
    mesh = make_mesh()
    r = SeedGeometryResolver(mesh)
    c = np.asarray(mesh.points).mean(axis=0)
    ap = mesh.points[r.anchor_vertex_id]
    assert np.linalg.norm(ap - c) <= 11.0


def test_validate_pv_rejects_anchor():
    mesh = make_mesh()
    r = SeedGeometryResolver(mesh)
    ok, _ = r.validate_pv(r.anchor_vertex_id)
    assert not ok, "PV validation must reject the body anchor itself"


def test_validate_pv_rejects_out_of_range():
    mesh = make_mesh()
    r = SeedGeometryResolver(mesh)
    ok, _ = r.validate_pv(-1)
    assert not ok
    ok, _ = r.validate_pv(mesh.n_points + 1000)
    assert not ok


def test_duplicate_position_check():
    mesh = make_mesh()
    r = SeedGeometryResolver(mesh)
    p = np.array([10.0, 0.0, 0.0])
    assert not r.is_duplicate_position(p, [])
    assert r.is_duplicate_position(p, [p])
    far = p + np.array([0, 0, 100.0])
    assert not r.is_duplicate_position(p, [far])


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
