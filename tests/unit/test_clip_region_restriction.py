"""
test_clip_region_restriction.py
===============================
A geometric clip cuts one region, not one volume of space.

A sphere placed on a vein seed, or a plane pushed through it, inevitably
covers more than the vein: the body behind it, the neighbouring vein beside
it. Before this, everything covered was removed, so clipping the LSPV took
body triangles with it. The geometry now says *where* to cut and the tag says
*what* may be cut — only triangles carrying the clipped region's ``elemTag``
go, however much else the widget encloses.

The preview must agree with the apply by construction, so both are exercised
against the same mesh: a red overlay that promises more than the clip delivers
is the same bug wearing a different hat.

The app side answers the other half: which tag a region maps to (the four
veins carry their tagging label, MV clips the body it sits on) and what
happens when that tag is absent from the mesh.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pyvista as pv
import pytest

from ccdaf.app.ccdaf import CCDAF
from ccdaf.core.mesh_loader import BODY_LABEL
from ccdaf.interaction.clipping_tool import ClippingTool, ClipMode

PV_TAG = 11          # LSPV, per region_tagger.LABELS


class _FakeSphereWidget:
    def __init__(self, center, radius):
        self._c, self._r = tuple(center), float(radius)

    def GetCenter(self):
        return self._c

    def GetRadius(self):
        return self._r

    def Off(self):
        pass


class _FakePlaneWidget:
    def __init__(self, origin, normal):
        self._o, self._n = tuple(origin), tuple(normal)

    def GetOrigin(self):
        return self._o

    def GetNormal(self):
        return self._n

    def Off(self):
        pass


def _two_region_mesh() -> pv.PolyData:
    """A sphere whose upper half is tagged as a vein, lower half as body.

    Both halves surround the origin, so any sphere or plane wide enough to
    reach one reaches the other — which is exactly the situation the bug
    report describes."""
    mesh = pv.Sphere(theta_resolution=16, phi_resolution=16).triangulate()
    centroids = mesh.cell_centers().points
    tags = np.where(centroids[:, 2] > 0.0, PV_TAG, BODY_LABEL)
    mesh.cell_data["elemTag"] = tags.astype(np.int32)
    return mesh


def _tool(mesh=None):
    holder = {"mesh": _two_region_mesh() if mesh is None else mesh}
    tool = ClippingTool(
        mesh_getter=lambda: holder["mesh"],
        mesh_setter=lambda m: holder.__setitem__("mesh", m),
        plotter=MagicMock(),
    )
    return tool, holder


def _tags(mesh) -> np.ndarray:
    return np.asarray(mesh.cell_data["elemTag"])


# ---------------------------------------------------------------------------
# Sphere
# ---------------------------------------------------------------------------
def test_sphere_swallowing_the_whole_mesh_removes_only_the_tagged_region():
    tool, holder = _tool()
    before = _two_region_mesh()
    tool._mode = ClipMode.SPHERE
    tool._clip_tag = PV_TAG
    tool._sphere_widget = _FakeSphereWidget((0.0, 0.0, 0.0), 100.0)

    res = tool.apply_sphere()

    assert res.n_removed == int(np.sum(_tags(before) == PV_TAG))
    survivors = _tags(holder["mesh"])
    assert survivors.size == int(np.sum(_tags(before) == BODY_LABEL))
    # Not one body triangle went, though every one of them was inside.
    assert np.all(survivors == BODY_LABEL)


def test_sphere_still_respects_its_own_geometry():
    """The tag narrows the cut; it does not widen it.

    Only tagged triangles *inside* the sphere go — a tagged triangle outside
    it is as safe as it ever was."""
    tool, holder = _tool()
    before = _two_region_mesh()
    centroids = before.cell_centers().points
    tool._mode = ClipMode.SPHERE
    tool._clip_tag = PV_TAG
    # A sphere hugging the north pole: some of the vein, none of the body.
    tool._sphere_widget = _FakeSphereWidget((0.0, 0.0, 0.5), 0.35)

    res = tool.apply_sphere()

    expected = ((np.linalg.norm(centroids - np.array([0.0, 0.0, 0.5]),
                                axis=1) <= 0.35)
                & (_tags(before) == PV_TAG))
    assert 0 < int(expected.sum()) < int(np.sum(_tags(before) == PV_TAG))
    assert res.n_removed == int(expected.sum())


def test_untagged_sphere_clip_still_cuts_everything_it_covers():
    """No tag named, no regions to confuse — the old geometric behaviour."""
    tool, holder = _tool()
    before = _two_region_mesh()
    centroids = before.cell_centers().points
    tool._mode = ClipMode.SPHERE
    tool._clip_tag = None
    # Straddling the equator, so the sphere covers both regions.
    tool._sphere_widget = _FakeSphereWidget((1.0, 0.0, 0.0), 0.6)

    res = tool.apply_sphere()

    covered = np.linalg.norm(centroids - np.array([1.0, 0.0, 0.0]),
                             axis=1) <= 0.6
    # The sphere covers body triangles as well as vein ones, and with no tag
    # named all of them go — the pre-existing geometric rule, unchanged.
    assert np.any(covered & (_tags(before) == BODY_LABEL))
    assert res.n_removed == int(covered.sum())
    assert holder["mesh"].n_cells == before.n_cells - int(covered.sum())


# ---------------------------------------------------------------------------
# Plane
# ---------------------------------------------------------------------------
def test_plane_removes_only_the_tagged_half_of_the_half_space():
    tool, holder = _tool()
    before = _two_region_mesh()
    centroids = before.cell_centers().points
    tool._mode = ClipMode.PLANE
    tool._clip_tag = PV_TAG
    # A plane cutting in x: its clipped half holds body and vein alike.
    tool._plane_widget = _FakePlaneWidget((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))

    res = tool.apply_plane(side_seed=(0.5, 0.0, 0.0))

    expected = (centroids[:, 0] >= 0.0) & (_tags(before) == PV_TAG)
    assert res.n_removed == int(expected.sum())
    # Body triangles on the clipped side survived.
    survivors = holder["mesh"]
    kept_body_x_pos = np.sum(
        (survivors.cell_centers().points[:, 0] >= 0.0)
        & (_tags(survivors) == BODY_LABEL))
    assert kept_body_x_pos > 0


# ---------------------------------------------------------------------------
# Preview agrees with apply
# ---------------------------------------------------------------------------
def test_sphere_preview_highlights_exactly_what_the_clip_removes():
    tool, _ = _tool()
    tool._mode = ClipMode.SPHERE
    tool._clip_tag = PV_TAG
    tool._sphere_widget = _FakeSphereWidget((0.0, 0.0, 0.0), 100.0)

    tool._update_sphere_preview()

    preview = tool.plotter.add_mesh.call_args[0][0]
    assert preview.n_cells == int(
        np.sum(_tags(_two_region_mesh()) == PV_TAG))
    assert np.all(np.asarray(preview.cell_data["elemTag"]) == PV_TAG)


def test_plane_preview_highlights_exactly_what_the_clip_removes():
    tool, _ = _tool()
    tool._mode = ClipMode.PLANE
    tool._clip_tag = PV_TAG
    tool._side_seed = np.array([0.5, 0.0, 0.0])
    tool._plane_widget = _FakePlaneWidget((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))

    tool._update_plane_preview()

    preview = tool.plotter.add_mesh.call_args[0][0]
    assert np.all(np.asarray(preview.cell_data["elemTag"]) == PV_TAG)


# ---------------------------------------------------------------------------
# The tag is clip-scoped, not sticky
# ---------------------------------------------------------------------------
def test_cancel_forgets_the_clip_tag():
    """A tag outliving its clip would silently constrain the next one."""
    tool, _ = _tool()
    tool._mode = ClipMode.SPHERE
    tool._clip_tag = PV_TAG
    tool.cancel()
    assert tool._clip_tag is None


def test_eligible_mask_is_all_true_on_an_untagged_mesh():
    mesh = pv.Sphere(theta_resolution=8, phi_resolution=8).triangulate()
    tool, _ = _tool(mesh)
    tool._clip_tag = PV_TAG
    assert np.all(tool._eligible_mask(mesh))


# ---------------------------------------------------------------------------
# The app side: which tag a region clips
# ---------------------------------------------------------------------------
class _TagHost:
    def __init__(self, mesh):
        self.loader = MagicMock(mesh=mesh)
        self._clip_tag_for = lambda name: CCDAF._clip_tag_for(self, name)


@pytest.fixture
def warned(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "ccdaf.app.ccdaf.QtWidgets.QMessageBox.warning",
        lambda *a, **kw: calls.append(a[1:]))
    return calls


def test_a_vein_clips_its_own_tagging_label(warned):
    host = _TagHost(_two_region_mesh())
    assert host._clip_tag_for("LSPV") == PV_TAG
    assert not warned


def test_mv_clips_the_body_it_sits_on(warned):
    """MV is excluded from tagging, so the body label is its region."""
    host = _TagHost(_two_region_mesh())
    assert host._clip_tag_for("MV") == BODY_LABEL
    assert not warned


def test_a_region_absent_from_the_tagging_is_refused(warned):
    """Nothing to clip is a stop, not a licence to cut geometrically."""
    host = _TagHost(_two_region_mesh())          # holds LSPV and body only
    assert host._clip_tag_for("RIPV") is None
    assert len(warned) == 1
    assert "not present in the current tagging" in warned[0][1]


def test_an_untagged_mesh_still_yields_a_tag(warned):
    """No elemTag at all: there is no tagging to be absent from."""
    mesh = pv.Sphere(theta_resolution=8, phi_resolution=8).triangulate()
    host = _TagHost(mesh)
    assert host._clip_tag_for("LSPV") == PV_TAG
    assert not warned
