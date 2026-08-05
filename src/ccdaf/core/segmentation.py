"""
segmentation
============
Mesh ↔ volume conversion: voxelise a surface into a binary SimpleITK
volume, and rebuild a surface from a (possibly edited) label volume by
signed-distance marching cubes.

Extracted from the GUI class so the whole segmentation round trip runs
with no Qt and no display — a mapping can be voxelised, edited and
re-meshed from a plain script, which is also what makes it testable.
The functions take the image they operate on instead of reading GUI
state; nothing here may import PyQt5 (tests enforce it).

Border note: both directions add a plane of background around the volume
they mesh. ``define_image_from_mesh`` allocates one voxel of margin so a
face on the bounding box still has background outside it; likewise
``segmentation_to_polydata`` pads the label mask, because a label running
into the volume border would otherwise produce an open surface there.

Orientation note: ``segmentation_to_polydata`` calls ``FlipNormalsOn``,
which reverses triangle winding — the sign of any downstream signed
distance follows the winding, not the stored ``Normals`` array. Callers
comparing surfaces (electrode displacement, field transfer) must
normalise winding first; see ``docs/eam-real-data-verification.md``.

Direction note: the segmentation view places every slice, crosshair and
brush stroke at ``origin + index * spacing``, i.e. it assumes the image's
direction matrix is the identity. Marching cubes, in contrast, honours
the direction matrix, so a volume stored in any other orientation would
render its surface and its slice planes in two different places.
``reorient_to_lps`` removes the discrepancy at load time by re-indexing
the volume into LPS, which is a lossless axis permutation and flip;
``restore_orientation`` puts it back for saving.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import SimpleITK as sitk
import vtk
from vtk.util import numpy_support


def negate_xy_inplace(poly: vtk.vtkPolyData) -> None:
    """Vectorised X/Y flip used by the MIRTK-orientation paths.

    A 180° rotation about z (determinant +1): it does *not* reverse
    triangle winding.
    """
    pts = poly.GetPoints()
    if pts is None or pts.GetNumberOfPoints() == 0:
        return
    arr = numpy_support.vtk_to_numpy(pts.GetData()).copy()
    arr[:, 0] *= -1.0
    arr[:, 1] *= -1.0
    new_data = numpy_support.numpy_to_vtk(arr, deep=True)
    new_pts = vtk.vtkPoints()
    new_pts.SetData(new_data)
    poly.SetPoints(new_pts)


#: Orientation the segmentation view works in. ITK reports direction
#: cosines in LPS, so an LPS-coded volume is exactly the identity.
LPS = "LPS"


def orientation_code(img: sitk.Image) -> str:
    """The three-letter DICOM orientation code of *img* (e.g. ``"RAS"``).

    The code names the anatomical direction each voxel axis grows
    towards, so ``"LPS"`` is the identity direction matrix.
    """
    return sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
        img.GetDirection()
    )


def is_axis_aligned(img: sitk.Image, tol: float = 1e-6) -> bool:
    """True when every voxel axis runs along a world axis.

    Only such a volume can be re-indexed into LPS losslessly: an oblique
    acquisition keeps its residual rotation through ``reorient_to_lps``,
    because that is an axis permutation, not a resampling.
    """
    d = np.abs(np.asarray(img.GetDirection(), dtype=float).reshape(3, 3))
    # A signed permutation matrix has exactly one 1 per row and column.
    return bool(np.all(np.abs(np.sort(d, axis=1) - [0.0, 0.0, 1.0]) <= tol)
                and np.all(np.abs(d.sum(axis=0) - 1.0) <= tol))


def reorient_to_lps(img: sitk.Image) -> Tuple[sitk.Image, str]:
    """Re-index *img* into LPS; return it with its original code.

    A pure permutation and flip of the voxel array with a matching
    origin, so every label keeps its world position — the surface built
    by marching cubes is unchanged. What does change is that
    ``origin + index * spacing`` becomes true, which is the placement
    rule the whole segmentation view is built on.

    The returned code is what ``restore_orientation`` needs to write the
    volume back the way it arrived. An already-LPS image is returned
    untouched.
    """
    code = orientation_code(img)
    if code == LPS:
        return img, code
    return sitk.DICOMOrient(img, LPS), code


def restore_orientation(img: sitk.Image, code: Optional[str]) -> sitk.Image:
    """Re-index *img* back into *code*, undoing ``reorient_to_lps``.

    A falsy or already-matching *code* is a no-op, so callers can pass
    whatever they recorded at load time without checking.
    """
    if not code or code == orientation_code(img):
        return img
    return sitk.DICOMOrient(img, code)


def define_image_from_mesh(poly: vtk.vtkPolyData,
                           spacing: np.ndarray) -> vtk.vtkImageData:
    """Allocate a vtkImageData covering the mesh bounds at *spacing*."""
    bounds = poly.GetBounds()  # (xmin, xmax, ymin, ymax, zmin, zmax)
    extents_world = np.array([
        bounds[1] - bounds[0],
        bounds[3] - bounds[2],
        bounds[5] - bounds[4],
    ], dtype=float)
    dims = np.maximum(np.ceil(extents_world / spacing).astype(int), 1)

    img = vtk.vtkImageData()
    img.SetSpacing(float(spacing[0]), float(spacing[1]), float(spacing[2]))
    img.SetDimensions(int(dims[0]), int(dims[1]), int(dims[2]))
    # One voxel of background on every side. The upper buffer was always
    # here, so the stencil never clips a face lying on the bbox max; the
    # lower one matches it, rather than leaving the mesh's min bound
    # sitting exactly on the grid's first plane with no background
    # outside it for the distance transform to measure against.
    img.SetExtent(0, int(dims[0]) + 2,
                  0, int(dims[1]) + 2,
                  0, int(dims[2]) + 2)
    img.SetOrigin(float(bounds[0] - spacing[0]),
                  float(bounds[2] - spacing[1]),
                  float(bounds[4] - spacing[2]))
    img.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    return img


def vtk_image_to_sitk(vtk_img: vtk.vtkImageData) -> sitk.Image:
    """Convert a vtkImageData to a SimpleITK image (no Python loops)."""
    ext = vtk_img.GetExtent()
    dims = (ext[1] - ext[0] + 1, ext[3] - ext[2] + 1, ext[5] - ext[4] + 1)
    scalars = vtk_img.GetPointData().GetScalars()
    arr = numpy_support.vtk_to_numpy(scalars).reshape(dims[2], dims[1], dims[0])
    out = sitk.GetImageFromArray(arr.astype(np.uint8))
    out.SetSpacing(tuple(float(s) for s in vtk_img.GetSpacing()))
    out.SetOrigin(tuple(float(o) for o in vtk_img.GetOrigin()))
    return out


def sitk_to_vtk_image(img: sitk.Image) -> vtk.vtkImageData:
    """Build a vtkImageData mirroring *img*'s geometry (vectorised)."""
    size = list(img.GetSize())
    spacing = list(img.GetSpacing())
    origin = list(img.GetOrigin())
    direction = list(img.GetDirection())

    vimg = vtk.vtkImageData()
    vimg.SetDimensions(int(size[0]), int(size[1]), int(size[2]))
    vimg.SetSpacing(float(spacing[0]), float(spacing[1]), float(spacing[2]))
    vimg.SetOrigin(float(origin[0]), float(origin[1]), float(origin[2]))
    vimg.SetExtent(0, size[0] - 1, 0, size[1] - 1, 0, size[2] - 1)
    if vtk.vtkVersion.GetVTKMajorVersion() >= 9 and len(direction) == 9:
        vimg.SetDirectionMatrix(direction)

    # SITK array shape is (Z, Y, X) C-contiguous; ravel matches VTK
    # linear indexing where X varies fastest.
    arr = sitk.GetArrayFromImage(img)
    flat = np.ascontiguousarray(arr).ravel()
    vimg.GetPointData().SetScalars(numpy_support.numpy_to_vtk(flat, deep=True))
    return vimg


def voxelise_polydata(mesh,
                      spacing: Tuple[float, float, float],
                      *, flip: bool) -> sitk.Image:
    """Convert a polydata surface to a binary SITK volume.

    Self-contained stencil-based rasterisation. Foreground fill is
    vectorised (single allocation through numpy_support).
    """
    # Take a writable deep copy of the polydata so optional flips
    # don't mutate the caller's mesh.
    poly = vtk.vtkPolyData()
    poly.DeepCopy(mesh)

    if flip:
        negate_xy_inplace(poly)

    spacing_arr = np.asarray(spacing, dtype=float)
    if (spacing_arr <= 0).any():
        raise ValueError(f"Spacing must be strictly positive, got {spacing}.")

    white = define_image_from_mesh(poly, spacing_arr)

    # Vectorised foreground fill (replaces per-voxel SetTuple1 loop).
    n_pts = white.GetNumberOfPoints()
    ones = np.ones(n_pts, dtype=np.uint8)
    white.GetPointData().SetScalars(
        numpy_support.numpy_to_vtk(ones, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    )

    stencil = vtk.vtkPolyDataToImageStencil()
    stencil.SetInputData(poly)
    stencil.SetOutputOrigin(white.GetOrigin())
    stencil.SetOutputSpacing(white.GetSpacing())
    stencil.SetOutputWholeExtent(white.GetExtent())
    stencil.Update()

    cutter = vtk.vtkImageStencil()
    cutter.SetInputData(white)
    cutter.SetStencilConnection(stencil.GetOutputPort())
    cutter.ReverseStencilOff()
    cutter.SetBackgroundValue(0)
    cutter.Update()

    return vtk_image_to_sitk(cutter.GetOutput())


def binary_mask_image(img: Optional[sitk.Image]) -> sitk.Image:
    """Build a uint8 0/1 mask of the segmentation *img*.

    Done in numpy to dodge ITK's ``BinaryThreshold`` parameter-range
    checks (which fail when ``upperThreshold`` exceeds the pixel
    type's max — e.g. ``2**31-1`` on a uint8 voxelisation).
    """
    if img is None:
        raise RuntimeError("No segmentation loaded.")
    arr = (sitk.GetArrayFromImage(img) > 0).astype(np.uint8)
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(img)
    return out


def label_mask_image(img: Optional[sitk.Image], label: int) -> sitk.Image:
    """Return a uint8 0/1 mask for the single label value *label*."""
    if img is None:
        raise RuntimeError("No segmentation loaded.")
    arr = (sitk.GetArrayFromImage(img) == label).astype(np.uint8)
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(img)
    return out


def sync_sitk_from_array(array: np.ndarray,
                         reference: sitk.Image) -> sitk.Image:
    """A new SITK image carrying *array*'s voxels on *reference*'s geometry.

    The (Z, Y, X) numpy edits become the image; origin, spacing and
    direction come from *reference*.
    """
    new = sitk.GetImageFromArray(array.astype(np.int16))
    new.SetOrigin(reference.GetOrigin())
    new.SetSpacing(reference.GetSpacing())
    new.SetDirection(reference.GetDirection())
    return new


def relabel_halfspace(array: np.ndarray,
                      origin, spacing, point, normal,
                      from_label: int, to_label: int) -> np.ndarray:
    """Relabel voxels on the normal-positive side of a plane.

    Every voxel whose label equals ``from_label`` and whose centre lies on the
    side the ``normal`` points to — the plane passing through ``point`` — is
    set to ``to_label``. The cut direction is the normal, so flipping the
    normal relabels the other half. Other labels and the other half are left
    untouched, and the input array is not modified.

    Voxel centres are ``origin + index * spacing`` with the array in
    ``(Z, Y, X)`` order — identity direction, the same assumption every slice
    in the segmentation view is placed under. A zero-length normal selects
    nothing (returns a copy). No mask matches when ``from_label`` is absent, so
    the call is a safe no-op there.
    """
    arr = np.asarray(array)
    nz, ny, nx = arr.shape
    ox, oy, oz = (float(v) for v in origin)
    sx, sy, sz = (float(v) for v in spacing)
    px, py, pz = (float(v) for v in point)
    n = np.asarray(normal, dtype=float)
    norm = float(np.linalg.norm(n))
    out = arr.copy()
    if norm == 0.0:
        return out
    n = n / norm
    xs = (ox + np.arange(nx) * sx - px) * n[0]
    ys = (oy + np.arange(ny) * sy - py) * n[1]
    zs = (oz + np.arange(nz) * sz - pz) * n[2]
    # Signed distance to the plane, broadcast to (Z, Y, X) only at compare time.
    signed = xs[None, None, :] + ys[None, :, None] + zs[:, None, None]
    out[(signed > 0.0) & (out == int(from_label))] = int(to_label)
    return out


def border_padding(filt_stdev: "list[float]",
                   filt_rfact: "list[float]") -> "list[int]":
    """Background voxels to add on each side before meshing, per axis.

    A label that runs into the volume border has no background outside it,
    so the signed distance never changes sign there and marching cubes
    leaves the surface open — the segmentation comes back with holes
    wherever the anatomy met the edge of a tightly cropped scan. One plane
    of background is all it takes to close it.

    The Gaussian smoothing that follows reads ``stdev * radius_factor``
    voxels either side of the point it writes, and at the image edge that
    kernel is truncated. Sitting the closure that far inside the padding
    keeps the truncation away from the new surface, hence the radius on
    top of the one plane. Smoothing off (either factor zero) leaves the
    single plane, which is the whole requirement.
    """
    radii = np.ceil(np.maximum(np.asarray(filt_stdev, dtype=float)
                               * np.asarray(filt_rfact, dtype=float), 0.0))
    return [1 + int(r) for r in radii]


def segmentation_to_polydata(img: Optional[sitk.Image], *, flip: bool,
                             filt_stdev: "list[float]",
                             filt_rfact: "list[float]",
                             label: Optional[int] = None,
                             pad: Optional[int] = None,
                             ) -> vtk.vtkPolyData:
    """Signed-distance marching cubes + smoothing; no preprocessing.

    When *label* is given, only that label value is meshed. Otherwise
    all voxels > 0 form the surface (legacy binary behaviour).

    The mask is padded with background first, so a label touching the
    volume border is still closed — see ``border_padding`` for how much
    and why. *pad* overrides that count on every axis; 0 disables it and
    meshes the mask as stored. Padding shifts the origin with the voxels,
    so the surface lands where it did before.
    """
    if img is None:
        raise RuntimeError("No segmentation loaded.")

    binary = label_mask_image(img, label) if label is not None \
        else binary_mask_image(img)
    widths = ([int(pad)] * 3 if pad is not None
              else border_padding(filt_stdev, filt_rfact))
    if any(w > 0 for w in widths):
        binary = sitk.ConstantPad(binary, widths, widths, 0)
    vimg = sitk_to_vtk_image(binary)

    outside_dist = vtk.vtkImageEuclideanDistance()
    outside_dist.SetInputData(vimg)
    outside_dist.SetConsiderAnisotropy(True)
    outside_dist.SetAlgorithmToSaito()
    outside_dist.Update()

    # Flip binary {0,1} → {1,0} so vtkImageEuclideanDistance can compute
    # distances from outside pixels to the nearest inside boundary.
    # SetOperationToInvert would compute 1/x, giving inf for 0-pixels
    # (no background for the distance filter). Use threshold instead.
    thresh = vtk.vtkImageThreshold()
    thresh.SetInputData(vimg)
    thresh.ThresholdByLower(0.5)
    thresh.SetInValue(1.0)
    thresh.SetOutValue(0.0)
    thresh.ReplaceInOn()
    thresh.ReplaceOutOn()
    thresh.Update()
    inside_dist = vtk.vtkImageEuclideanDistance()
    inside_dist.SetInputData(thresh.GetOutput())
    inside_dist.SetConsiderAnisotropy(True)
    inside_dist.SetAlgorithmToSaito()
    inside_dist.Update()

    sdf = vtk.vtkImageMathematics()
    sdf.SetInput1Data(outside_dist.GetOutput())
    sdf.SetInput2Data(inside_dist.GetOutput())
    sdf.SetOperationToSubtract()
    sdf.Update()
    vimg_sdf = sdf.GetOutput()

    mc = vtk.vtkMarchingCubes()
    if np.any(np.array(filt_stdev) > 0.) and np.any(np.array(filt_rfact) > 0.):
        gaussian = vtk.vtkImageGaussianSmooth()
        gaussian.SetStandardDeviations(filt_stdev[0], filt_stdev[1], filt_stdev[2])
        gaussian.SetRadiusFactors(filt_rfact[0], filt_rfact[1], filt_rfact[2])
        gaussian.SetDimensionality(3)
        gaussian.SetInputData(vimg_sdf)
        gaussian.Update()
        mc.SetInputConnection(gaussian.GetOutputPort())
    else:
        mc.SetInputData(vimg_sdf)
    mc.ComputeScalarsOff()
    mc.ComputeNormalsOff()
    mc.ComputeGradientsOff()
    mc.SetValue(0, 0.0)
    mc.Update()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(mc.GetOutputPort())
    normals.ComputePointNormalsOn()
    normals.ComputeCellNormalsOff()
    normals.AutoOrientNormalsOn()
    normals.FlipNormalsOn()
    normals.Update()

    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(normals.GetOutputPort())
    tri.PassVertsOff()
    tri.PassLinesOff()
    tri.Update()
    out: vtk.vtkPolyData = tri.GetOutput()

    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(out)
    clean.PointMergingOn()
    clean.ConvertLinesToPointsOff()
    clean.ConvertPolysToLinesOff()
    clean.ConvertStripsToPolysOff()
    clean.Update()
    out = clean.GetOutput()
    if flip:
        negate_xy_inplace(out)
    return out


__all__ = [
    "LPS", "orientation_code", "is_axis_aligned", "reorient_to_lps",
    "restore_orientation",
    "negate_xy_inplace", "define_image_from_mesh", "vtk_image_to_sitk",
    "sitk_to_vtk_image", "voxelise_polydata", "binary_mask_image",
    "label_mask_image", "sync_sitk_from_array", "border_padding",
    "segmentation_to_polydata", "relabel_halfspace",
]
