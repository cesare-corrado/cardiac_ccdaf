"""
test_volume_from_segmentation.py
================================
Rebuilding a tetrahedral volume from a corrected segmentation.

The segmentation round trip has always returned a surface. For a
volumetric mesh that loses the tetrahedra and the fibres, which are the
reason the file was kept, so the working volume is cut down to the
corrected boundary instead — and where that cannot be done honestly, the
caller is told rather than handed a mesh that ignored half the edit.

The contract:

* the distance field is **positive inside**, and squared unless asked for
  millimetres. Both are stated here because assuming otherwise selects
  the background, which on a ventricle is thirteen times the anatomy;
* an unedited segmentation is close to the identity — if carving with a
  correction that changed nothing moved the mesh, nothing else here could
  be trusted;
* an edit that shrinks the anatomy shrinks the mesh;
* an edit that **grows** it past the original boundary is detected before
  anything is meshed. A mesh can only be cut, never extended, so the
  alternative is silently dropping the new material;
* the tolerance distinguishes a smoothing pass from a dilation — it is
  not zero, because voxelising and cutting both round at the boundary;
* labels and fields come through;
* a morphological operation that would remove almost everything is
  arithmetic, not a fault — but it is worth measuring, because on a wall
  two or three voxels thick a radius of 2 leaves under 1%.

The MMG call is real. Meshes are small enough to keep it quick.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pytest
import pyvista as pv
import SimpleITK as sitk

from ccdaf.core import volume_mesh as vm
from ccdaf.core.segmentation import (
    binary_mask_image, distance_field, voxelise_polydata,
)
from ccdaf.core.volume_from_segmentation import (
    GROWTH_TOLERANCE, carve, growth_outside,
)


def _block(n: int = 9, size: float = 8.0) -> pv.UnstructuredGrid:
    """A tetrahedralised box, labelled in two halves, with fields."""
    grid = pv.ImageData(
        dimensions=(n, n, n),
        spacing=(size / (n - 1),) * 3).cast_to_unstructured_grid().triangulate()
    centres = np.asarray(grid.cell_centers().points)
    grid.cell_data["elemTag"] = np.where(
        centres[:, 0] < size / 2.0, 1, 2).astype(np.int32)
    grid.cell_data["fiber"] = np.tile(
        np.array([0.0, 0.0, 1.0]), (grid.n_cells, 1))
    grid.point_data["scar_probability"] = np.linspace(0.0, 1.0, grid.n_points)
    return grid


def _segmentation(grid, spacing: float = 0.5) -> sitk.Image:
    return voxelise_polydata(grid, (spacing,) * 3, flip=False)


def _volume(grid) -> float:
    return float(np.abs(vm.signed_volumes(np.asarray(grid.points),
                                          vm.tetrahedra(grid))).sum())


@pytest.fixture(scope="module")
def block():
    return _block()


@pytest.fixture(scope="module")
def segmentation(block):
    return _segmentation(block)


# ------------------------------------------------------- the distance field
def test_the_distance_field_is_positive_inside(block, segmentation):
    """Stated as a test because getting it backwards is invisible.

    A level set built on "negative is inside" selects the background,
    which looks like the mesher being slow rather than like an error.
    """
    field = pv.wrap(distance_field(binary_mask_image(segmentation)))
    values = np.asarray(field.point_data[field.active_scalars_name
                                         or list(field.point_data)[0]], float)
    mask = sitk.GetArrayFromImage(segmentation).ravel() > 0
    assert values[mask].min() > 0.0
    assert values[~mask].max() < 0.0


def test_metric_gives_millimetres_and_keeps_the_zero_crossing(segmentation):
    """The raw field is squared; a level set needs distances."""
    binary = binary_mask_image(segmentation)
    raw = pv.wrap(distance_field(binary))
    metric = pv.wrap(distance_field(binary, metric=True))

    def values(g):
        return np.asarray(g.point_data[g.active_scalars_name
                                       or list(g.point_data)[0]], float)

    r, m = values(raw), values(metric)
    assert np.array_equal(r > 0, m > 0)               # same zero crossing
    assert np.abs(m).max() < np.abs(r).max()          # ... but smaller values
    assert np.allclose(np.abs(m), np.sqrt(np.abs(r)), atol=1e-6)


# ------------------------------------------------------------------ growth
def test_an_unedited_segmentation_has_grown_nothing(segmentation):
    growth = growth_outside(segmentation, segmentation)
    assert growth.added == 0
    assert growth.fraction == 0.0
    assert not growth.exceeds_tolerance


def test_a_dilation_is_detected(block, segmentation):
    dilated = sitk.BinaryDilate(
        sitk.Cast(segmentation > 0, sitk.sitkUInt8), [3, 3, 3])
    growth = growth_outside(dilated, segmentation)
    assert growth.added > 0
    assert growth.volume > 0.0
    assert growth.exceeds_tolerance, "a dilation must not be carved"


def test_an_erosion_has_grown_nothing(segmentation):
    """Shrinking is always safe: the mesh only has to lose elements."""
    eroded = sitk.BinaryErode(
        sitk.Cast(segmentation > 0, sitk.sitkUInt8), [1, 1, 1])
    growth = growth_outside(eroded, segmentation)
    assert growth.added == 0
    assert not growth.exceeds_tolerance


def test_the_tolerance_exceeds_the_methods_own_error():
    """It must not interrupt about what the method cannot resolve.

    Carving an *unedited* segmentation should be the identity and is not:
    it returns 99.24% of the volume at 1 mm spacing (0.01% loss at
    0.5 mm). A tolerance below that discretisation error would ask the
    user about quantities smaller than the noise floor — which is what a
    0.5% tolerance did, prompting on a 0.6% closing.
    """
    intrinsic_loss_at_1mm = 0.0076
    assert GROWTH_TOLERANCE > intrinsic_loss_at_1mm
    # ... and still well under a deliberate edit.
    assert GROWTH_TOLERANCE < 0.5


def test_mismatched_grids_are_refused(block, segmentation):
    other = _segmentation(block, spacing=1.0)
    with pytest.raises(ValueError, match="different grids"):
        growth_outside(other, segmentation)


# ------------------------------------------------------------------- carve
def test_carving_with_an_unedited_segmentation_is_near_identity(
        block, segmentation):
    """If this moved the mesh, nothing else here could be trusted."""
    out = carve(block, segmentation)
    assert vm.kind_of(out) == vm.VOLUME
    assert vm.inverted_count(out) == 0
    assert _volume(out) == pytest.approx(_volume(block), rel=0.05)


def test_carving_keeps_the_labels_and_the_fields(block, segmentation):
    out = carve(block, segmentation)
    assert set(np.unique(out.cell_data["elemTag"]).tolist()) <= {1, 2}
    assert np.asarray(out.cell_data["fiber"]).shape == (out.n_cells, 3)
    assert "scar_probability" in out.point_data
    assert np.isfinite(np.asarray(out.point_data["scar_probability"])).all()
    # Bookkeeping from the element selection must not ride along.
    assert not ({"vtkOriginalCellIds", "vtkOriginalPointIds"}
                & set(out.cell_data.keys()) | set(out.point_data.keys())
                & {"vtkOriginalCellIds", "vtkOriginalPointIds"})


def test_an_erosion_shrinks_the_mesh(block, segmentation):
    eroded = sitk.BinaryErode(
        sitk.Cast(segmentation > 0, sitk.sitkUInt8), [2, 2, 2])
    out = carve(block, eroded)
    assert _volume(out) < 0.9 * _volume(block)
    assert vm.inverted_count(out) == 0


def test_carving_against_nothing_is_refused(block, segmentation):
    """An empty correction must raise, not return an empty mesh."""
    empty = sitk.Image(segmentation.GetSize(), sitk.sitkUInt8)
    empty.CopyInformation(segmentation)
    with pytest.raises(RuntimeError, match="nothing to rebuild"):
        carve(block, empty)


def test_the_source_is_not_modified(block, segmentation):
    cells = block.n_cells
    tags = np.array(block.cell_data["elemTag"], copy=True)
    carve(block, segmentation)
    assert block.n_cells == cells
    assert np.array_equal(np.asarray(block.cell_data["elemTag"]), tags)


# ------------------------------------------------- what morphology costs
def _slab(thickness: int, size: int = 21) -> sitk.Image:
    """A wall *thickness* voxels thick — the shape a myocardium is."""
    arr = np.zeros((size, size, size), dtype=np.uint8)
    start = (size - thickness) // 2
    arr[:, :, start:start + thickness] = 1
    return sitk.GetImageFromArray(arr)


def _opening(mask: sitk.Image, radius: int) -> int:
    rad = [radius] * 3
    out = sitk.BinaryDilate(
        sitk.BinaryErode(mask, rad, kernelType=sitk.sitkBox),
        rad, kernelType=sitk.sitkBox)
    return int(sitk.GetArrayFromImage(out).sum())


@pytest.mark.parametrize("thickness,radius,survives", [
    (3, 1, True),       # 3 voxels thick, 1 off each side: 1 left, survives
    (3, 2, False),      # 2 off each side of 3: gone before the dilation
    (9, 2, True),       # a thick structure is untouched by the same radius
])
def test_an_opening_eats_anything_thinner_than_twice_its_radius(
        thickness, radius, survives):
    """Why the application asks before applying one.

    An opening erodes then dilates, and the erosion takes *radius* voxels
    off every side, so a structure thinner than twice the radius is gone
    before the dilation can run. A myocardial wall is two or three voxels
    at millimetre spacing, which makes a radius of 2 destructive — correct
    arithmetic, and almost never what the user meant. Hence the
    confirmation, whose necessity this pins.
    """
    mask = _slab(thickness)
    before = int(sitk.GetArrayFromImage(mask).sum())
    after = _opening(mask, radius)
    assert before > 0
    if survives:
        assert after > 0
    else:
        assert after == 0


def test_repeating_a_dilation_only_adds_radii():
    """There is no way to an even-sized element, including by repeating.

    Dilation with a flat structuring element composes: applying radius 1
    twice is the same operation as radius 2, not a 2-voxel effect. This
    was documented wrongly once.
    """
    single = np.zeros((15, 15, 15), dtype=np.uint8)
    single[7, 7, 7] = 1
    img = sitk.GetImageFromArray(single)

    once = sitk.BinaryDilate(img, [1, 1, 1], kernelType=sitk.sitkBox)
    twice = sitk.BinaryDilate(once, [1, 1, 1], kernelType=sitk.sitkBox)
    radius_two = sitk.BinaryDilate(img, [2, 2, 2], kernelType=sitk.sitkBox)

    assert np.array_equal(sitk.GetArrayFromImage(twice),
                          sitk.GetArrayFromImage(radius_two))
    assert int(sitk.GetArrayFromImage(once).sum()) == 27      # 3x3x3
    assert int(sitk.GetArrayFromImage(twice).sum()) == 125     # 5x5x5


@pytest.mark.parametrize("radius,expected", [
    ((0, 0, 0), (1, 1, 1)),
    ((1, 0, 0), (3, 1, 1)),      # one axis only: the zeros do the work
    ((1, 1, 1), (3, 3, 3)),
    ((2, 1, 1), (5, 3, 3)),
    ((3, 1, 2), (7, 3, 5)),
])
def test_the_element_spans_two_r_plus_one_per_axis(radius, expected):
    """The control is a radius per axis, not a count of voxels."""
    single = np.zeros((17, 17, 17), dtype=np.uint8)
    single[8, 8, 8] = 1
    out = sitk.GetArrayFromImage(sitk.BinaryDilate(
        sitk.GetImageFromArray(single), list(radius), kernelType=sitk.sitkBox))
    idx = np.argwhere(out > 0)
    span = tuple(int(v) for v in (idx.max(axis=0) - idx.min(axis=0) + 1)[::-1])
    assert span == expected


def test_the_thresholds_differ_by_what_the_operation_promises(segmentation):
    """One threshold cannot serve erode and opening.

    Erode is a request to shrink, so a big loss is the point and only
    near-total loss is a surprise. An opening is a request to *smooth*,
    and one that deletes most of the object is not smoothing — which is
    the case a single 10% threshold let through silently.
    """
    from ccdaf.app.ccdaf import SEG_SHRINK_WARN_FRACTION

    assert SEG_SHRINK_WARN_FRACTION["morph_open"] > \
        SEG_SHRINK_WARN_FRACTION["erode"]
    # Growing operations cannot shrink anything, so they have no entry.
    assert "dilate" not in SEG_SHRINK_WARN_FRACTION
    assert "morph_close" not in SEG_SHRINK_WARN_FRACTION

    wall = _slab(3)
    before = int(sitk.GetArrayFromImage(wall).sum())
    # An opening that guts a thin wall must trip its threshold ...
    assert _opening(wall, 2) < SEG_SHRINK_WARN_FRACTION["morph_open"] * before
    # ... while a structure thicker than the radius must not.
    thick = _slab(9)
    bulk = int(sitk.GetArrayFromImage(thick).sum())
    assert _opening(thick, 2) >= SEG_SHRINK_WARN_FRACTION["morph_open"] * bulk


#: What an opening at radius 2 leaves of the biventricular example at
#: 1 mm spacing, measured. Recorded as a number because it is the case
#: that drove the thresholds apart, and a synthetic fixture does not
#: reproduce it: a *uniform* slab is all-or-nothing under an opening
#: (erode by r then dilate by r restores the full thickness whenever
#: anything survives), and a real wall varies in thickness, which is
#: where a partial loss comes from.
OPENING_RADIUS_2_AT_1MM: float = 0.135


def test_the_case_that_passed_silently_now_warns():
    """13.5% left: above erode's threshold, below opening's.

    Judged as an erosion this applies with no warning, which is what
    happened and what prompted splitting the thresholds — an opening is a
    request to smooth, and one that deletes 86% of the object is not
    smoothing whatever the arithmetic says.
    """
    from ccdaf.app.ccdaf import SEG_SHRINK_WARN_FRACTION

    assert OPENING_RADIUS_2_AT_1MM > SEG_SHRINK_WARN_FRACTION["erode"]
    assert OPENING_RADIUS_2_AT_1MM < SEG_SHRINK_WARN_FRACTION["morph_open"]


def test_a_closing_only_ever_grows(block, segmentation):
    """Which is exactly why it can trip the growth guard.

    A closing is a dilation followed by an erosion, so its result
    contains the original. It cannot shrink the anatomy, and on a thin
    wall it bridges cavities — real growth, correctly refused rather than
    quietly clipped.
    """
    mask = binary_mask_image(segmentation)
    rad = [2, 2, 2]
    closed = sitk.BinaryErode(
        sitk.BinaryDilate(mask, rad, kernelType=sitk.sitkBox),
        rad, kernelType=sitk.sitkBox)
    before = sitk.GetArrayFromImage(mask) > 0
    after = sitk.GetArrayFromImage(closed) > 0
    assert np.all(after[before]), "a closing must not remove anything"
    assert after.sum() >= before.sum()
    assert growth_outside(closed, segmentation).added > 0


# --------------------------------------------------------- saving a volume
@pytest.mark.parametrize("typed,expected", [
    ("/tmp/volumetric", "/tmp/volumetric.nii.gz"),   # no suffix at all
    ("/tmp/seg.nii", "/tmp/seg.nii"),
    ("/tmp/seg.nii.gz", "/tmp/seg.nii.gz"),
    ("/tmp/seg.mha", "/tmp/seg.mha"),                # another writable one
    ("/tmp/my.mesh.thing", "/tmp/my.mesh.thing.nii.gz"),
])
def test_a_segmentation_filename_always_gets_a_writable_suffix(typed, expected):
    """A name typed without an extension must still be writable.

    SimpleITK picks its writer from the extension alone, so a bare name
    fails inside ITK with "Unable to determine ImageIO writer for …" — a
    message about a library the user never invoked, for a mistake the
    dialog could have fixed. The mesh save has always appended a suffix;
    this one did not.
    """
    from ccdaf.app.ccdaf import CCDAF

    assert CCDAF._with_segmentation_suffix(typed) == expected


# ------------------------------------------------------ the smoothing knobs
def test_smoothing_reaches_the_volume_route(block, segmentation):
    """What Update 3D previews must be what the export produces.

    The Gaussian belongs to the segmentation panel and was wired only to
    marching cubes and the 3D preview. The volumetric route ignored it, so
    you could set a smoothing, watch the preview change, and get an
    unsmoothed boundary out of the export.
    """
    plain = carve(block, segmentation)
    smoothed = carve(block, segmentation,
                     filt_stdev=[1.0, 1.0, 1.0], filt_rfact=[2.0, 2.0, 2.0])
    # Both are valid volumes of roughly the same anatomy ...
    for out in (plain, smoothed):
        assert vm.kind_of(out) == vm.VOLUME
        assert vm.inverted_count(out) == 0
    # ... but the parameter is not ignored.
    assert smoothed.n_cells != plain.n_cells or \
        not np.isclose(_volume(smoothed), _volume(plain))


def test_zero_smoothing_is_the_same_as_none(block, segmentation):
    """Zero has always meant "no smoothing" in that panel."""
    from ccdaf.core.segmentation import binary_mask_image, smooth_field

    field = distance_field(binary_mask_image(segmentation))
    assert smooth_field(field, [0.0] * 3, [1.5] * 3) is field
    assert smooth_field(field, [0.5] * 3, [0.0] * 3) is field
    assert smooth_field(field, [0.5] * 3, [1.5] * 3) is not field


# ------------------------------------------------------- perforation risk
def test_perforation_cannot_be_predicted_cheaply():
    """Recorded so the two failed heuristics are not tried again.

    The minimum of the distance field over the tissue is ~1 voxel for any
    shape, because every surface voxel is one voxel from the background —
    it flagged a solid cube. And the share a one-voxel erosion removes
    tracks surface-to-volume, not thinness: the ventricle scores 20.6% at
    0.25 mm and a solid sphere 21.0%. The application therefore *states*
    the risk once per session rather than predicting it.
    """
    import ccdaf.core.volume_from_segmentation as vfs
    from ccdaf.app.ccdaf import CCDAF

    assert not hasattr(vfs, "perforation_risk")
    assert not hasattr(vfs, "thinnest_wall")
    assert hasattr(CCDAF, "_warn_perforation_risk")
