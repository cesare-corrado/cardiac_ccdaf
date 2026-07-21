"""
test_vtk_nodata.py
==================
No-data (NaN) survival across a VTK save/load round trip.

VTK's legacy *ASCII* reader cannot parse a ``nan`` token: the first one
poisons the stream, the tuple count drifts, and a leftover ``nan`` is read as
the next array's *name* — so every field past the first no-data field is lost
and a phantom ``nan`` field appears (this is what makes such a file open in
neither ParaView nor ccdaf). The contract enforced here:

* ASCII encodes no-data as the Carto sentinel (``CARTO_NODATA``), so the file
  a raw reader sees carries every field and no ``nan`` token;
* loading an ASCII ``.vtk`` folds the sentinel back to NaN, so the round trip
  is transparent — NaN in, NaN out, real values intact;
* binary carries NaN natively and is left completely untouched — no sentinel
  is written, and a binary file is never folded on read;
* the fold keys on the sentinel magnitude, which real Carto data never
  reaches, so a partially-no-data field (what a segmentation edit's guard
  produces) round-trips too.

Uses synthetic meshes (no real EAM data required).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pyvista as pv

from ccdaf.core.eam_loader import CARTO_NODATA
from ccdaf.core.mesh_loader import (
    MeshLoader,
    nodata_to_sentinel,
    sentinel_to_nodata,
    _is_ascii_legacy_vtk,
)
from ccdaf.io.vtkfunctions import readvtk


def _mesh_with_nodata():
    """A triangulated sphere carrying a real field, an all-NaN field, and a
    partially-NaN field — the three shapes a Carto mapping produces."""
    m = pv.Sphere(theta_resolution=20, phi_resolution=20).triangulate()
    n = m.n_points
    rng = np.random.default_rng(0)
    m.point_data["Bipolar"] = rng.uniform(-5, 5, n).astype(np.float32)  # real
    m.point_data["Impedance"] = np.full(n, np.nan, np.float32)          # all no-data
    lat = rng.uniform(-50, 50, n).astype(np.float32)
    lat[: n // 3] = np.nan                                              # partial
    m.point_data["LAT"] = lat
    m.point_data["Tail"] = rng.uniform(0, 1, n).astype(np.float32)      # after the NaN
    m.cell_data["elemTag"] = np.ones(m.n_cells, np.int32)
    return m


def _save_load(tmp_path, binary):
    src = _mesh_with_nodata()
    loader = MeshLoader()
    loader.mesh = src.copy(deep=True)
    fn = tmp_path / "mesh.vtk"
    fields = list(src.point_data.keys()) + list(src.cell_data.keys())
    loader.save(str(fn), fields=fields, binary=binary)
    return src, fn, loader.load(str(fn))


# --------------------------------------------------------------------------
# The failure this fixes: an ASCII file with NaN loses fields to a phantom.
# --------------------------------------------------------------------------
def test_ascii_file_carries_every_field_to_a_raw_reader(tmp_path):
    src, fn, _ = _save_load(tmp_path, binary=False)
    raw = pv.wrap(readvtk(str(fn)))            # what ParaView / downstream see
    assert "nan" not in raw.point_data.keys()  # no phantom array
    for name in src.point_data.keys():
        assert name in raw.point_data.keys()


def test_ascii_file_has_no_nan_token_but_has_the_sentinel(tmp_path):
    _, fn, _ = _save_load(tmp_path, binary=False)
    text = fn.read_bytes()
    assert b"nan" not in text.lower()
    assert str(int(CARTO_NODATA)).encode() in text


# --------------------------------------------------------------------------
# Transparency: NaN in, NaN out, real values intact — both encodings.
# --------------------------------------------------------------------------
def _assert_faithful(src, back):
    for name, o in src.point_data.items():
        r = np.asarray(back.point_data[name], dtype=float)
        o = np.asarray(o, dtype=float)
        assert np.array_equal(np.isnan(o), np.isnan(r)), f"{name} nan-mask moved"
        fin = np.isfinite(o)
        assert np.allclose(o[fin], r[fin], atol=1e-3), f"{name} values drifted"


def test_ascii_round_trip_is_transparent(tmp_path):
    src, _, back = _save_load(tmp_path, binary=False)
    _assert_faithful(src, back)


def test_binary_round_trip_is_transparent(tmp_path):
    src, _, back = _save_load(tmp_path, binary=True)
    _assert_faithful(src, back)


def test_partial_nodata_field_survives_ascii(tmp_path):
    """The segmentation-guard case: some points NaN inside a real field."""
    src, _, back = _save_load(tmp_path, binary=False)
    o = np.asarray(src.point_data["LAT"], dtype=float)
    r = np.asarray(back.point_data["LAT"], dtype=float)
    assert 0 < np.isnan(o).sum() < len(o)          # genuinely partial
    assert np.array_equal(np.isnan(o), np.isnan(r))


# --------------------------------------------------------------------------
# Binary is untouched: no sentinel written, and never folded on read.
# --------------------------------------------------------------------------
def test_binary_keeps_nan_not_the_sentinel(tmp_path):
    _, fn, back = _save_load(tmp_path, binary=True)
    assert np.isnan(np.asarray(back.point_data["Impedance"])).all()


def test_binary_read_is_not_folded(tmp_path):
    """A literal value at the sentinel magnitude in a *binary* file stays put —
    only ASCII files are folded, so binary is a byte-faithful carrier."""
    m = pv.Sphere(theta_resolution=8, phi_resolution=8).triangulate()
    m.point_data["odd"] = np.full(m.n_points, CARTO_NODATA, np.float32)
    loader = MeshLoader()
    loader.mesh = m
    fn = tmp_path / "b.vtk"
    loader.save(str(fn), fields=["odd", "elemTag"], binary=True)
    back = loader.load(str(fn))
    vals = np.asarray(back.point_data["odd"], dtype=float)
    assert np.isfinite(vals).all() and np.allclose(vals, CARTO_NODATA)


# --------------------------------------------------------------------------
# Helpers.
# --------------------------------------------------------------------------
def test_is_ascii_legacy_vtk(tmp_path):
    m = pv.Sphere().triangulate()
    m.point_data["f"] = np.ones(m.n_points, np.float32)
    loader = MeshLoader()
    loader.mesh = m
    a = tmp_path / "a.vtk"
    b = tmp_path / "b.vtk"
    loader.save(str(a), fields=["f"], binary=False)
    loader.save(str(b), fields=["f"], binary=True)
    assert _is_ascii_legacy_vtk(a) is True
    assert _is_ascii_legacy_vtk(b) is False
    assert _is_ascii_legacy_vtk(tmp_path / "x.vtp") is False
    assert _is_ascii_legacy_vtk(tmp_path / "missing.vtk") is False


def test_sentinel_helpers_are_inverse_and_leave_real_data(tmp_path):
    m = _mesh_with_nodata()
    before = {k: np.asarray(v, dtype=float).copy() for k, v in m.point_data.items()}
    nodata_to_sentinel(m)
    assert np.isfinite(np.asarray(m.point_data["Impedance"])).all()  # NaN gone
    sentinel_to_nodata(m)
    for k, o in before.items():
        r = np.asarray(m.point_data[k], dtype=float)
        assert np.array_equal(np.isnan(o), np.isnan(r))
        fin = np.isfinite(o)
        assert np.allclose(o[fin], r[fin])


def test_elem_tag_and_labels_are_untouched_by_the_fold():
    """elemTag labels never approach the sentinel, so folding is a no-op."""
    m = pv.Sphere(theta_resolution=8, phi_resolution=8).triangulate()
    m.cell_data["elemTag"] = np.full(m.n_cells, 19, np.float32)
    sentinel_to_nodata(m)
    assert np.all(np.asarray(m.cell_data["elemTag"]) == 19)
