"""
test_bundle_io.py
=================
Tests for the File → Save pickle bundle: ``eam_export.export_binary`` with
the seed / elemTag extensions, read back by ``eam_loader.read_bundle``.

The contract:

* geometry, point fields (NaN and all), seeds, electrodes and elemTag
  survive the round trip;
* the seed / elemTag keys are opt-in — a plain ``export_binary`` still
  writes exactly ``{'surface', 'electrodes'}``, so the EAM export path is
  unchanged;
* ``read_bundle`` rejects a pickle that is not a bundle;
* a bundle's ``"seeds"`` key is readable by the seed loader too, so the
  two entry points interoperate.

Synthetic mesh; no display, no Qt.
"""

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pytest
import pyvista as pv

from ccdaf.core.eam_export import export_binary
from ccdaf.core.eam_loader import read_bundle
from ccdaf.core.seed_io import load_seeds


def _mesh() -> pv.PolyData:
    m = pv.Sphere(radius=5.0, theta_resolution=20, phi_resolution=20).triangulate()
    lat = np.linspace(-50.0, 80.0, m.n_points)
    lat[::7] = np.nan                         # scattered no-data
    m.point_data["LAT"] = lat
    m.cell_data["elemTag"] = np.full(m.n_cells, 11, dtype=np.int32)
    m.cell_data["elemTag"][: m.n_cells // 2] = 17
    return m


SEEDS = {"LSPV": [5.0, 0.0, 0.0], "MV": [0.0, 0.0, -5.0]}


def test_plain_export_is_unchanged(tmp_path):
    path = tmp_path / "eam.pkl"
    export_binary(path, _mesh())
    with open(path, "rb") as fh:
        payload = pickle.load(fh)
    assert set(payload) == {"surface", "electrodes"}


def test_bundle_round_trip(tmp_path):
    path = tmp_path / "bundle.pkl"
    mesh = _mesh()
    export_binary(path, mesh, seeds=SEEDS, include_elem_tag=True)

    back, seeds, electrodes = read_bundle(path)
    assert back.n_points == mesh.n_points
    assert back.n_cells == mesh.n_cells
    assert electrodes is None

    lat0 = np.asarray(mesh.point_data["LAT"], dtype=float)
    lat1 = np.asarray(back.point_data["LAT"], dtype=float)
    assert np.array_equal(np.isnan(lat0), np.isnan(lat1))
    assert np.allclose(lat1[~np.isnan(lat1)], lat0[~np.isnan(lat0)])

    assert np.array_equal(np.asarray(back.cell_data["elemTag"]),
                          np.asarray(mesh.cell_data["elemTag"]))

    assert set(seeds) == set(SEEDS)
    for name in SEEDS:
        assert np.allclose(seeds[name], SEEDS[name])


def test_field_selection_governs_the_surface(tmp_path):
    # The app drops unselected point fields before calling export_binary;
    # here we prove the writer keeps exactly what it is handed.
    path = tmp_path / "no_lat.pkl"
    mesh = _mesh()
    mesh.point_data.remove("LAT")
    export_binary(path, mesh, include_elem_tag=True)
    back, _, _ = read_bundle(path)
    assert "LAT" not in back.point_data


def test_electrodes_round_trip(tmp_path):
    path = tmp_path / "elec.pkl"
    record = {"data": np.array([[0.0, 5.0, 0.0, 0.0, 42.0],
                                [1.0, 0.0, 5.0, 0.0, 43.0]])}
    pts = np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
    export_binary(path, _mesh(), electrodes=record, electrode_points=pts)
    _, _, electrodes = read_bundle(path)
    assert electrodes is not None
    assert np.allclose(np.asarray(electrodes["data"])[:, 1:4], pts)


def test_read_bundle_rejects_non_bundle(tmp_path):
    path = tmp_path / "junk.pkl"
    with open(path, "wb") as fh:
        pickle.dump({"nope": 1}, fh)
    with pytest.raises(ValueError):
        read_bundle(path)


def test_seed_loader_reads_a_bundle(tmp_path):
    path = tmp_path / "bundle.pkl"
    export_binary(path, _mesh(), seeds=SEEDS, include_elem_tag=True)
    seeds = load_seeds(path)          # the seed panel's Load path
    assert set(seeds) == set(SEEDS)
    assert np.allclose(seeds["MV"], SEEDS["MV"])
