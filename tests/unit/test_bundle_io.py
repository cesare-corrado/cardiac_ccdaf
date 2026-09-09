"""
test_bundle_io.py
=================
Tests for the File → Save pickle bundle: ``eam_export.export_binary`` with
the seed / elemTag extensions, read back by ``eam_loader.read_bundle``.

The contract:

* geometry, point fields (NaN and all), point sets, electrodes and
  elemTag survive the round trip;
* the point-set / elemTag keys are opt-in — a plain ``export_binary``
  still writes exactly ``{'surface', 'electrodes'}``, so the EAM export
  path is unchanged;
* only non-empty sets are written, so a bundle never claims a set the
  session does not hold — which is what keeps atrial keys off a
  non-atrial mesh;
* a set stored under a profile's legacy key (the six seeds under
  ``"seeds"``, from before they were named ``seed_LA``) is read back
  under the current profile, so old files keep working;
* a point set may not claim one of the payload's own keys;
* ``read_bundle`` rejects a pickle that is not a bundle;
* a bundle's point-set key is readable by the seed loader too, so the
  two entry points interoperate;
* the LA-UAC landmark set rides in the same bundle under its own
  ``"landmarks_LA_UAC"`` key, alongside the seeds, and round-trips
  independently.

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
from ccdaf.core.seed_io import load_point_set, load_seeds
from ccdaf.core.seed_profiles import (
    LANDMARKS_LA_UAC_PROFILE, SEED_LA_PROFILE,
)

LA = SEED_LA_PROFILE.export_key                 # "seed_LA"
UAC = LANDMARKS_LA_UAC_PROFILE.export_key       # "landmarks_LA_UAC"


def _mesh() -> pv.PolyData:
    m = pv.Sphere(radius=5.0, theta_resolution=20, phi_resolution=20).triangulate()
    lat = np.linspace(-50.0, 80.0, m.n_points)
    lat[::7] = np.nan                         # scattered no-data
    m.point_data["LAT"] = lat
    m.cell_data["elemTag"] = np.full(m.n_cells, 11, dtype=np.int32)
    m.cell_data["elemTag"][: m.n_cells // 2] = 17
    return m


SEEDS = {"LSPV": [5.0, 0.0, 0.0], "MV": [0.0, 0.0, -5.0]}
LANDMARKS = {"LSPV_BODY_JCN": [0.0, 5.0, 0.0], "SEPTAL_WALL": [-5.0, 0.0, 0.0]}


def test_plain_export_is_unchanged(tmp_path):
    path = tmp_path / "eam.pkl"
    export_binary(path, _mesh())
    with open(path, "rb") as fh:
        payload = pickle.load(fh)
    assert set(payload) == {"surface", "electrodes"}


def test_bundle_round_trip(tmp_path):
    path = tmp_path / "bundle.pkl"
    mesh = _mesh()
    export_binary(path, mesh, point_sets={LA: SEEDS}, include_elem_tag=True)

    back, point_sets, electrodes = read_bundle(path)
    seeds = point_sets[SEED_LA_PROFILE.type_id]
    assert back.n_points == mesh.n_points
    assert back.n_cells == mesh.n_cells
    assert electrodes is None
    # None supplied → key absent → no entry, rather than an empty one.
    assert LANDMARKS_LA_UAC_PROFILE.type_id not in point_sets

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


def test_landmarks_round_trip_alongside_seeds(tmp_path):
    # Both point sets ride in one bundle under their own keys and come back
    # independently, matching the "coexist / both exported" behaviour.
    path = tmp_path / "both.pkl"
    export_binary(path, _mesh(), point_sets={LA: SEEDS, UAC: LANDMARKS})

    with open(path, "rb") as fh:
        payload = pickle.load(fh)
    assert LA in payload and UAC in payload
    assert "seeds" not in payload      # new files use the new spelling

    _, point_sets, _ = read_bundle(path)
    seeds = point_sets[SEED_LA_PROFILE.type_id]
    landmarks = point_sets[LANDMARKS_LA_UAC_PROFILE.type_id]
    assert set(seeds) == set(SEEDS)
    assert set(landmarks) == set(LANDMARKS)
    for name in LANDMARKS:
        assert np.allclose(landmarks[name], LANDMARKS[name])


def test_empty_sets_are_not_written(tmp_path):
    """A set with no points leaves no key behind.

    Nothing in the file says which anatomy it is, so an absent key has to
    mean absent: an empty ``seed_LA`` on a mesh that has no left-atrial
    seeds would be read back as a set that exists.
    """
    path = tmp_path / "empty.pkl"
    export_binary(path, _mesh(), point_sets={LA: {}, UAC: None})
    with open(path, "rb") as fh:
        payload = pickle.load(fh)
    assert set(payload) == {"surface", "electrodes"}


def test_reserved_keys_are_refused(tmp_path):
    """A point set cannot overwrite the payload's own keys."""
    path = tmp_path / "clash.pkl"
    for key in ("surface", "electrodes", "elemTag"):
        with pytest.raises(ValueError):
            export_binary(path, _mesh(), point_sets={key: SEEDS})


def test_legacy_seeds_key_is_read_as_seed_la(tmp_path):
    """A bundle written before the rename still loads its six seeds.

    Written by hand under the old key, because that is what an existing
    file on disk looks like — the current writer can no longer produce it.
    """
    path = tmp_path / "old.pkl"
    export_binary(path, _mesh(), point_sets={LA: SEEDS})
    with open(path, "rb") as fh:
        payload = pickle.load(fh)
    payload["seeds"] = payload.pop(LA)          # rewind to the old spelling
    with open(path, "wb") as fh:
        pickle.dump(payload, fh)

    _, point_sets, _ = read_bundle(path)
    seeds = point_sets[SEED_LA_PROFILE.type_id]
    assert set(seeds) == set(SEEDS)
    assert np.allclose(seeds["MV"], SEEDS["MV"])

    # The seed panel's Load path reaches it through the same alias list,
    # and reports which key it matched so the caller can say so.
    key, points = load_point_set(path, SEED_LA_PROFILE.read_keys)
    assert key == "seeds"
    assert set(points) == set(SEEDS)


def test_plain_export_has_no_landmarks_key(tmp_path):
    path = tmp_path / "plain.pkl"
    export_binary(path, _mesh())
    with open(path, "rb") as fh:
        payload = pickle.load(fh)
    assert UAC not in payload


def test_read_bundle_rejects_non_bundle(tmp_path):
    path = tmp_path / "junk.pkl"
    with open(path, "wb") as fh:
        pickle.dump({"nope": 1}, fh)
    with pytest.raises(ValueError):
        read_bundle(path)


def test_seed_loader_reads_a_bundle(tmp_path):
    path = tmp_path / "bundle.pkl"
    export_binary(path, _mesh(), point_sets={LA: SEEDS}, include_elem_tag=True)
    seeds = load_seeds(path, key=LA)  # the seed panel's Load path
    assert set(seeds) == set(SEEDS)
    assert np.allclose(seeds["MV"], SEEDS["MV"])
