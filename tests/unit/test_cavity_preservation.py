"""
test_cavity_preservation.py
===========================
An enclosed cavity must survive being turned into a segmentation, being
turned back into a surface, and being post-processed.

The anatomy this is about is a ventricle: a thick wall around a chamber.
The atrial workflow never had one — a left-atrial shell is a single
surface with the blood pool inside it, and filling that pool is exactly
what voxelising it is for. A ventricle is the opposite: the chamber is
*not* tissue, and a step that fills it turns a wall into a solid lump
with no way back.

The contract:

* ``voxelise_polydata`` leaves an enclosed cavity as background. It is
  the even-odd parity of the stencil that does this — the endocardium is
  a second crossing — so the test measures the volume rather than
  trusting the mechanism;
* the same code on a surface with *no* cavity fills it solid, which is
  what proves the measurement above is measuring the cavity and not some
  general emptiness;
* ``segmentation_to_polydata`` brings the cavity back as its own inner
  surface, so the round trip returns a wall and not a lump;
* every post-processing step keeps the cavity, and keeps the cell fields
  (labels, and a per-cell vector such as a fibre direction) that a
  volumetric mesh's boundary carries.

The fixture is a thick-walled sphere — two concentric shells in one
surface — because it is the smallest thing with the property under test.
Sizes are small enough to keep the suite quick.

No display, no Qt.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pytest
import pyvista as pv
import SimpleITK as sitk

import ccdaf.core.mesh_postprocessor as mp
from ccdaf.core.segmentation import segmentation_to_polydata, voxelise_polydata

R_OUT = 10.0
R_IN = 6.0
SPACING = (0.5, 0.5, 0.5)

WALL_VOLUME = 4.0 / 3.0 * np.pi * (R_OUT ** 3 - R_IN ** 3)
SOLID_VOLUME = 4.0 / 3.0 * np.pi * R_OUT ** 3


def _sphere(radius: float) -> pv.PolyData:
    return pv.Sphere(radius=radius, theta_resolution=48,
                     phi_resolution=48).triangulate()


@pytest.fixture(scope="module")
def shell() -> pv.PolyData:
    """A hollow ball: outer wall plus the endocardium inside it."""
    m = (_sphere(R_OUT) + _sphere(R_IN)).triangulate()
    m.cell_data["elemTag"] = np.full(m.n_cells, 1, dtype=np.int32)
    # A per-cell vector, standing in for the fibre direction a volumetric
    # mesh's boundary carries. Three components, so it also pins that a
    # non-scalar cell array survives every step.
    rng = np.random.default_rng(0)
    m.cell_data["fiber"] = rng.normal(size=(m.n_cells, 3))
    m.point_data["scar"] = np.linspace(0.0, 1.0, m.n_points)
    return m


def _n_components(mesh: pv.PolyData) -> int:
    ids = np.asarray(mesh.connectivity("all").point_data["RegionId"])
    return int(ids.max()) + 1


def _foreground_volume(img: sitk.Image) -> float:
    arr = sitk.GetArrayFromImage(img)
    voxel = float(np.prod(img.GetSpacing()))
    return float((arr > 0).sum()) * voxel


# ------------------------------------------------------------ voxelising
def test_fixture_is_two_shells(shell):
    assert _n_components(shell) == 2


def test_voxelising_leaves_the_cavity_empty(shell):
    img = voxelise_polydata(shell, SPACING, flip=False)
    got = _foreground_volume(img)
    # The wall, not the ball: a filled cavity would be ~28% larger.
    assert got == pytest.approx(WALL_VOLUME, rel=0.02)
    assert got < 0.9 * SOLID_VOLUME


def test_the_cavity_centre_stays_background(shell):
    img = voxelise_polydata(shell, SPACING, flip=False)
    arr = sitk.GetArrayFromImage(img)
    centre = tuple(n // 2 for n in arr.shape)
    assert arr[centre] == 0


def test_a_surface_without_a_cavity_fills_solid():
    """The control. Without this, the tests above prove nothing.

    A single closed surface has no inner crossing, so the same code fills
    it — which is the atrial behaviour, and must not change.
    """
    img = voxelise_polydata(_sphere(R_OUT), SPACING, flip=False)
    assert _foreground_volume(img) == pytest.approx(SOLID_VOLUME, rel=0.02)


# --------------------------------------------------------- and back again
def test_the_round_trip_returns_a_wall_not_a_lump(shell):
    img = voxelise_polydata(shell, SPACING, flip=False)
    rebuilt = pv.wrap(segmentation_to_polydata(
        img, flip=False, filt_stdev=[0.5] * 3, filt_rfact=[1.5] * 3))
    # The endocardium comes back as its own surface. One component would
    # mean the cavity was filled somewhere along the way.
    assert _n_components(rebuilt) == 2


# ------------------------------------------------------- post-processing
@pytest.mark.parametrize("name,opts", [
    ("clean", dict(do_clean=True)),
    ("fill_holes", dict(do_fill_holes=True)),
    ("smooth", dict(do_smooth=True, smooth_iterations=10)),
    ("refine_adaptive", dict(do_refine=True, refine_mode=mp.REFINE_ADAPTIVE,
                             refine_edge_len=0.8)),
    ("refine_resample", dict(do_refine=True, refine_mode=mp.REFINE_RESAMPLE,
                             refine_edge_len=1.2, remesh_passes=3)),
    ("decimate", dict(do_decimate=True, decimate_target_points=800,
                      decimate_iters=20)),
])
def test_post_processing_keeps_the_cavity_and_the_fields(shell, name, opts):
    out = mp.apply(shell, mp.PostprocessOptions(**opts))
    assert _n_components(out) == 2, f"{name} dropped the cavity"
    assert "elemTag" in out.cell_data
    assert "scar" in out.point_data
    fiber = np.asarray(out.cell_data["fiber"])
    assert fiber.shape == (out.n_cells, 3)


def test_clean_still_drops_a_speck(shell):
    """The atrial behaviour the cavity fix must not have cost.

    ``clean`` keeps a component large enough to be anatomy; a stray fleck
    of segmentation is not, and still goes.

    48 cells against the wall's 8832 is 0.54%, comfortably under
    ``MIN_COMPONENT_FRACTION``. The endocardium in the same mesh is 50%,
    which is the gap the threshold sits in.
    """
    speck = pv.Sphere(radius=0.2, center=(40.0, 0.0, 0.0),
                      theta_resolution=6, phi_resolution=6).triangulate()
    speck.cell_data["elemTag"] = np.full(speck.n_cells, 1, dtype=np.int32)
    speck.cell_data["fiber"] = np.zeros((speck.n_cells, 3))
    speck.point_data["scar"] = np.zeros(speck.n_points)
    dirty = (shell + speck).triangulate()
    assert _n_components(dirty) == 3
    assert speck.n_cells < mp.MIN_COMPONENT_FRACTION * dirty.n_cells

    out = mp.clean(dirty, quality_threshold=0.8, smooth_iterations=5)
    assert _n_components(out) == 2      # the wall and its cavity, not the fleck
    assert out.bounds[1] < 2.0 * R_OUT  # nothing left out at x = 40
