"""
test_uac_right.py
=================
Universal atrial coordinates for the right atrium.

The test atrium is a sphere of radius 25 mm. The tricuspid valve is an
opening on the anterior side (+y); the SVC and IVC are open vessels high
and low on the posterior side, so the shortest line between them (the
roof) runs down the posterior wall; the appendage is a patch on the
lateral (+x) side next to the SVC; the CS is a small open vessel on the
septal (-x) side between the IVC and the valve.

The contract:

* the action needs all five labels, and says why not;
* the left atrium's layout is the left atrium's template, unchanged;
* a full run: no folded triangles, every ring near its template target,
  alpha and beta in [0, 1], the appendage on the lateral half, the CS on
  the septal half, each structure at its template place;
* the flat map from the stored values is the run's own map;
* SVC and IVC labels swapped are refused.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pytest
import pyvista as pv

from ccdaf.core import surface_repair
from ccdaf.core.uac import layout as lay
from ccdaf.core.uac import right
from ccdaf.core.uac import template as la
from ccdaf.core.uac import template_ra as tr
from ccdaf.core.uac.cut import CutError

#: label -> (direction, label radius, opening radius) in radians
VESSELS = {tr.SVC: ((0.0, -0.5, 0.85), 0.30, 0.14),
           tr.IVC: ((0.0, -0.5, -0.85), 0.30, 0.14),
           tr.CS: ((-0.6, 0.35, -0.6), 0.20, 0.09)}
RAA_DIR = (0.7, 0.3, 0.6)


def _atrium(res: int = 90, radius: float = 25.0, swap: bool = False) -> pv.PolyData:
    s = pv.Sphere(radius=radius, theta_resolution=res, phi_resolution=res).triangulate().clean()
    P = np.asarray(s.points)
    t = s.faces.reshape(-1, 4)[:, 1:]
    c = P[t].mean(1)
    u = c / np.linalg.norm(c, axis=1, keepdims=True)
    tag = np.full(len(t), tr.BODY, int)
    keep = u[:, 1] < 0.75                                     # tricuspid valve opening
    d = np.array(RAA_DIR) / np.linalg.norm(RAA_DIR)
    tag[np.arccos(np.clip(u @ d, -1, 1)) < 0.32] = tr.RAA
    for L, (dv, r_label, r_open) in VESSELS.items():
        dv = np.array(dv) / np.linalg.norm(dv)
        ang = np.arccos(np.clip(u @ dv, -1, 1))
        if swap and L in (tr.SVC, tr.IVC):
            L = tr.IVC if L == tr.SVC else tr.SVC
        tag[ang < r_label] = L
        keep &= ang > r_open
    m = pv.PolyData(P, np.c_[np.full(keep.sum(), 3), t[keep]].ravel())
    m.cell_data["elemTag"] = tag[keep]
    return m.clean()


def _inputs(m):
    return (np.asarray(m.points, float), m.faces.reshape(-1, 4)[:, 1:].astype(np.int64),
            np.asarray(m.cell_data["elemTag"]).astype(int))


@pytest.fixture(scope="module")
def atrium():
    return _atrium()


@pytest.fixture(scope="module")
def run(atrium):
    from ccdaf.core.uac.pipeline import Options
    return right.compute(*_inputs(atrium), Options(iterations=40))


def test_availability_says_why_not(atrium):
    ok, _ = right.availability(atrium)
    assert ok
    m = atrium.copy()
    tag = np.asarray(m.cell_data["elemTag"]).copy()
    tag[tag == tr.CS] = tr.BODY
    m.cell_data["elemTag"] = tag
    ok, why = right.availability(m)
    assert not ok and "CS" in why
    assert not right.availability(None)[0]


def test_the_left_layout_is_the_left_template():
    assert lay.LA.far_c == la.LSPV_C and lay.LA.near_c == la.RSPV_C
    assert lay.LA.closed[la.LAA].c == la.LAA_C and lay.LA.closed[la.LAA].side == 1
    assert {L: c for L, (c, _) in lay.LA.ends.items()} == la.END_C


def test_the_right_template_fits_the_square_without_overlap():
    boxes = {tr.RAA: (tr.RAA_C, tr.RAA_R_ALPHA, tr.RAA_R_BETA),
             tr.CS: (tr.CS_C, tr.CS_R, tr.CS_R),
             tr.SVC: ((tr.SVC_C[0] + tr.CAVA_ACROSS / 2, tr.SVC_C[1]), tr.CAVA_ACROSS / 2,
                      tr.CAVA_ALONG),
             tr.IVC: ((tr.IVC_C[0] + tr.CAVA_ACROSS / 2, tr.IVC_C[1]), tr.CAVA_ACROSS / 2,
                      tr.CAVA_ALONG)}
    for c, ra, rb in boxes.values():
        assert 0 < c[0] - ra and c[0] + ra < 1 and 0 < c[1] - rb and c[1] + rb < 1
    keys = list(boxes)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            (ca, ra1, rb1), (cb, ra2, rb2) = boxes[a], boxes[b]
            assert (abs(ca[0] - cb[0]) > ra1 + ra2) or (abs(ca[1] - cb[1]) > rb1 + rb2), (a, b)
    assert tr.RAA_C[0] + tr.RAA_R_ALPHA < 0.5 < tr.CS_C[0] - tr.CS_R   # lateral / septal


def test_a_run_is_fold_free_and_on_template(run):
    a = run.attempt
    assert a.folded_real == 0 and a.collapsed == []
    assert a.worst_miss < 0.02
    for x in (run.alpha, run.beta):
        assert np.isfinite(x).all() and x.min() >= -1e-6 and x.max() <= 1 + 1e-6
    assert run.atrium is right.RIGHT
    assert "Lateral" in run.details() and "SVC" in run.details()


def test_each_structure_lands_in_its_template_place(run):
    s = run.surface
    places = {tr.RAA: tr.RAA_C, tr.CS: tr.CS_C,
              tr.SVC: (0.55, tr.SVC_C[1]), tr.IVC: (0.55, tr.IVC_C[1])}
    for L, (a0, b0) in places.items():
        pts = np.unique(s.t[s.tag == L])
        centre = (np.median(run.alpha[pts]), np.median(run.beta[pts]))
        assert abs(centre[0] - a0) < 0.08 and abs(centre[1] - b0) < 0.08, (tr.NAMES[L], centre)
    # the appendage's side of the sphere is the lateral half
    P = s.P
    lateral = run.alpha < 0.5
    body = np.unique(s.t[s.tag == tr.BODY])
    assert np.median(P[body][lateral[body], 0]) > 0 > np.median(P[body][~lateral[body], 0])


def test_the_flat_map_from_the_stored_values_is_the_run_s_own(run):
    s = run.surface
    uv, tri, cells, src = right.flat_from_values(run.alpha, run.beta, s.t)
    assert uv.min() >= -1e-6 and uv.max() <= 1 + 1e-6
    assert len(src) > len(np.unique(src))                   # cut points appear once per half
    n0 = len(s.P)
    want: dict = {}
    for h in run.attempt.halves:                            # halves are in the working frame
        ok = h.nodes < n0
        for p, q in zip(h.nodes[ok], h.uv[ok][:, ::-1]):
            want.setdefault(int(p), []).append(q)
    got: dict = {}
    for p, q in zip(src, uv):
        got.setdefault(int(p), []).append(q)
    assert got.keys() == want.keys()
    for p in got:
        g = np.array(sorted(map(tuple, got[p])))
        w = np.array(sorted(map(tuple, want[p])))
        assert len(g) == len(w) and np.abs(g - w).max() < 1e-12, p


def test_the_run_writes_back_like_the_left_atrium(run, atrium):
    out = surface_repair.apply(atrium, run.surface, label="elemTag")
    assert out.n_points == len(run.alpha)


def test_swapped_cavae_are_refused():
    from ccdaf.core.uac.cut_ra import check_sides
    P, t, tag = _inputs(_atrium(res=60, swap=True))
    with pytest.raises(CutError, match="swapped"):
        check_sides(P, t, tag)
    P, t, tag = _inputs(_atrium(res=60))
    check_sides(P, t, tag)


def _stored_map_folds(alpha, beta, t):
    """Folds of the flat map drawn from the stored values, counted the
    caller's way: sign against the median sign, collapsed triangles skipped."""
    from ccdaf.core.uac.flatten import signed_areas
    uv, tri, _, _ = right.flat_from_values(alpha, beta, t)
    sa = signed_areas(uv, tri)
    e = np.stack([np.linalg.norm(uv[tri[:, k]] - uv[tri[:, (k + 1) % 3]], axis=1)
                  for k in range(3)], 1).max(1)
    ok = e >= 1e-6
    return int((ok & (np.sign(sa) == -np.sign(np.median(sa[ok])))).sum())


def test_the_report_counts_every_fold_of_the_stored_map(run):
    s = run.surface
    assert run.attempt.folded_real == _stored_map_folds(run.alpha, run.beta, s.t)
    # a fold made on purpose (one point pushed across its neighbours) is counted
    from ccdaf.core.uac.pipeline import stored_folds
    a, b = run.alpha.copy(), run.beta.copy()
    body = np.unique(s.t[s.tag == tr.BODY])
    inner = body[(np.abs(a[body] - 0.25) < 0.1) & (np.abs(b[body] - 0.5) < 0.2)]
    v = int(inner[0])
    nb = np.unique(s.t[(s.t == v).any(1)])
    nb = nb[nb != v]
    c = np.array([a[nb].mean(), b[nb].mean()])
    a[v], b[v] = c + 3.0 * (c - (a[v], b[v])) + (0.01, 0.0)
    want = _stored_map_folds(a, b, s.t)
    assert want > 0
    assert sum(stored_folds(b, a, s.t)) == want                 # working frame is transposed


def test_both_halves_hold_the_cut_at_the_same_place():
    """Starting the warp with the hard points at their targets keeps them
    there exactly, whatever the step sizes."""
    from ccdaf.core.uac.slim import slim
    disk = pv.Disc(inner=0.0, outer=1.0, r_res=12, c_res=48).triangulate().clean()
    R = np.asarray(disk.points)[:, :2]
    t = disk.faces.reshape(-1, 4)[:, 1:]
    rim = np.where(np.isclose(np.linalg.norm(R, axis=1), 1.0))[0]
    target = R[rim] * np.c_[1.0 + 0.2 * R[rim, 0] ** 2, np.ones(len(rim))]   # stretched rim
    init = R.copy()
    init[:, 0] *= 1.0 + 0.2 * R[:, 0] ** 2                                    # same stretch inside
    U, _, _ = slim(R, t, np.zeros(0, int), np.zeros((0, 2)), rim, target, iters=5, init=init)
    assert np.abs(U[rim] - target).max() == 0.0
    U2, _, _ = slim(R, t, np.zeros(0, int), np.zeros((0, 2)), rim, target, iters=5)
    assert np.abs(U2[rim] - target).max() > 0.0                              # without: only in the limit
