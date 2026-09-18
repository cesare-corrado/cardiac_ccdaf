"""
test_ldrb.py
============
The rule-based fibre stage (Bayer et al., Ann Biomed Eng 2012).

The contract:

* the rotation helpers are consistent: a frame survives the round trip
  through a quaternion, and bislerp returns its end frames at t = 0 and 1;
* every frame the stage writes is orthonormal and right-handed;
* where an element's frame is well defined, the result matches
  ``GlRuleFibers`` element by element. The fixture holds real
  ``GlRuleFibers`` output on a small ventricle, for its default angles and
  for a steeper set, so the measured conventions (septal beta flip,
  quaternion order, which column is the sheet) are all exercised;
* where the frame is undefined, the gradients are rebuilt from the
  neighbours instead of copying ``GlRuleFibers``' placeholder, and those
  elements are exactly the ones that differ from it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pytest

from ccdaf.core.laplace import basis_gradients, element_gradients
from ccdaf.core.ldrb import (
    LEFT, LINEAR, RIGHT, FibreAngles, axis, bislerp, degenerate_elements,
    fibres, orient, quat_to_rot, repair_gradients, rot_to_quat,
)

ORACLE = Path(__file__).resolve().parents[1] / "data" / "glrulefibers_oracle.npz"


@pytest.fixture(scope="module")
def oracle():
    d = np.load(ORACLE)
    fields = {k: d[k] for k in ("ab", "epi", "lv", "rv")}
    return d, fields


def _random_frames(n, seed=0):
    q = np.random.default_rng(seed).normal(size=(n, 4))
    return quat_to_rot(q / np.linalg.norm(q, axis=1, keepdims=True))


def _axial_angle(a, b):
    """Angle between two direction fields, ignoring sign, in degrees."""
    c = np.abs(np.sum(a * b, axis=1)) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return np.degrees(np.arccos(np.clip(c, 0.0, 1.0)))


def _assert_orthonormal_right_handed(f):
    assert np.allclose(np.linalg.norm(f.fibre, axis=1), 1.0)
    assert np.allclose(np.linalg.norm(f.sheet, axis=1), 1.0)
    assert np.allclose(np.sum(f.fibre * f.sheet, axis=1), 0.0, atol=1e-9)
    assert np.allclose(np.sum(f.fibre * f.sheet_normal, axis=1), 0.0, atol=1e-9)
    # Frame columns are (fibre, sheet normal, sheet): F x S = T.
    assert np.allclose(np.cross(f.fibre, f.sheet_normal), f.sheet, atol=1e-9)


# ---------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------
def test_a_frame_survives_the_quaternion_round_trip():
    r = _random_frames(500)
    assert np.allclose(quat_to_rot(rot_to_quat(r)), r, atol=1e-12)


def _same_axes(p, q):
    """Frames equal up to the sign of each axis."""
    return np.all(np.abs(np.abs(np.einsum("nik,nik->nk", p, q)) - 1.0) < 1e-9, axis=1)


def test_bislerp_returns_its_end_frames():
    """Up to axis signs, since bislerp may pick an equivalent frame."""
    a, b = _random_frames(200, 1), _random_frames(200, 2)
    assert _same_axes(bislerp(a, b, np.zeros(200), RIGHT), a).all()
    assert _same_axes(bislerp(a, b, np.ones(200), RIGHT), b).all()


def test_the_printed_order_does_not_keep_the_start_frame():
    """Why RIGHT is the default: u . q turns the frame about the world
    axes, so at t = 0 it can return a different frame altogether."""
    a, b = _random_frames(200, 1), _random_frames(200, 2)
    assert not _same_axes(bislerp(a, b, np.zeros(200), LEFT), a).all()


def test_axis_builds_a_right_handed_frame_from_two_gradients():
    rng = np.random.default_rng(3)
    psi, phi = rng.normal(size=(100, 3)), rng.normal(size=(100, 3))
    q = axis(psi, phi)
    assert np.allclose(np.einsum("nki,nkj->nij", q, q), np.eye(3), atol=1e-12)
    assert np.allclose(np.linalg.det(q), 1.0)
    # e1 follows the apicobasal gradient; e2 is the transmural one, made
    # orthogonal to it.
    assert np.allclose(q[:, :, 1], psi / np.linalg.norm(psi, axis=1, keepdims=True))


def test_orient_keeps_a_frame_a_rotation():
    q = orient(_random_frames(50), np.full(50, 37.0), np.full(50, -21.0))
    assert np.allclose(np.linalg.det(q), 1.0)


# ---------------------------------------------------------------------
# Against GlRuleFibers
# ---------------------------------------------------------------------
@pytest.mark.parametrize("name", ["default", "steep"])
def test_well_defined_elements_match_glrulefibers(oracle, name):
    d, fields = oracle
    r = fibres(d["points"], d["tets"], fields, FibreAngles(*d[f"{name}_angles"]))
    _assert_orthonormal_right_handed(r)
    good = ~(r.repaired | r.unresolved)
    assert good.sum() > 0.95 * len(good)
    assert _axial_angle(r.fibre, d[f"{name}_fibre"])[good].max() < 0.1
    assert _axial_angle(r.sheet, d[f"{name}_sheet"])[good].max() < 0.1


def test_the_default_angles_are_glrulefibers_defaults():
    a = FibreAngles()
    assert (a.alpha_endo, a.alpha_epi, a.beta_endo, a.beta_epi) == (40.0, -50.0, -65.0, 25.0)


# ---------------------------------------------------------------------
# Undefined frames
# ---------------------------------------------------------------------
def test_an_element_on_one_surface_is_degenerate(oracle):
    """Four nodes held on the epicardium: phi_epi is 1 at all of them, so
    the element has no transmural gradient."""
    d, fields = oracle
    tets = d["tets"]
    g = {k: element_gradients(d["points"], tets, fields[k]) for k in fields}
    bad = degenerate_elements(tets, fields, g)
    on_epi = np.all(fields["epi"][tets] == 1.0, axis=1)
    assert on_epi.any()
    assert bad[on_epi].all()


def test_the_repaired_elements_are_the_only_ones_that_differ(oracle):
    d, fields = oracle
    r = fibres(d["points"], d["tets"], fields)
    assert r.repaired.any() and not r.unresolved.any()
    off = _axial_angle(r.fibre, d["default_fibre"]) > 1.0
    assert not (off & ~r.repaired).any()


def test_repaired_fibres_follow_their_neighbours_better_than_the_placeholder(oracle):
    """GlRuleFibers writes a placeholder that ignores the tissue around it;
    the rebuilt frame should sit closer to the neighbouring fibres."""
    d, fields = oracle
    tets = d["tets"]
    r = fibres(d["points"], tets, fields)
    ref = d["default_fibre"]
    good = ~r.repaired
    ours, theirs = [], []
    for e in np.where(r.repaired)[0]:
        nb = np.where(np.isin(tets, tets[e]).any(axis=1) & good)[0]
        if len(nb) == 0:
            continue
        ours.append(np.median(_axial_angle(np.repeat(r.fibre[[e]], len(nb), 0), r.fibre[nb])))
        theirs.append(np.median(_axial_angle(np.repeat(ref[[e]], len(nb), 0), ref[nb])))
    assert np.median(ours) < np.median(theirs)


def test_a_zero_gradient_is_rebuilt_from_its_neighbours():
    """A slab with phi = x: every element's gradient is (1, 0, 0). Zero
    one out and mark it; the repair must bring it back."""
    import pyvista as pv
    from ccdaf.core import volume_mesh as vm
    grid = pv.ImageData(dimensions=(6, 6, 6)).cast_to_unstructured_grid().triangulate()
    pts, tets = np.asarray(grid.points, float), vm.tetrahedra(grid)
    basis, volume = basis_gradients(pts, tets)
    grad = np.einsum("nij,nj->ni", basis, pts[:, 0][tets])
    target = np.zeros(len(tets), dtype=bool)
    target[len(tets) // 2] = True
    grad[target] = 0.0
    fixed, left = repair_gradients(tets, grad, volume, len(pts), target)
    assert not left.any()
    assert np.allclose(fixed[target], [1.0, 0.0, 0.0])


# ---------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------
def test_linear_mode_agrees_with_published_at_the_epicardium(oracle):
    """Both modes reduce to the epicardial frame turned by the epicardial
    angles where phi_epi is 1; they differ only inside the wall."""
    d, fields = oracle
    tets = d["tets"]
    pub = fibres(d["points"], tets, fields)
    lin = fibres(d["points"], tets, fields, mode=LINEAR)
    _assert_orthonormal_right_handed(lin)
    epi = fields["epi"][tets].mean(axis=1)
    surface = epi >= 1.0 - 1e-12     # all four nodes on it, so all repaired
    assert surface.any()
    assert _axial_angle(pub.fibre, lin.fibre)[surface].max() < 1e-6


def test_an_unknown_mode_is_refused(oracle):
    d, fields = oracle
    with pytest.raises(ValueError, match="unknown mode"):
        fibres(d["points"], d["tets"], fields, mode="cubic")
