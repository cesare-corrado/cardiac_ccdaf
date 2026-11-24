"""
test_fibre_atlas.py
===================
The left-atrial fibre atlases: fetched once with a checksum, refused when
built by another coordinate method, and (when the published files are on
this machine) carrying what fibre mapping needs. No test uses the network.
"""
from __future__ import annotations

import hashlib
import io
import os

import numpy as np
import pytest
import pyvista as pv

from ccdaf.core.uac import fibre_atlas as fa


def _tiny(path, method=fa.UAC_METHOD):
    m = pv.Sphere(theta_resolution=8, phi_resolution=8).triangulate()
    m.field_data["uac_method"] = [method]
    m.save(path)
    return open(path, "rb").read()


class _Response(io.BytesIO):
    headers = {}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture
def server(tmp_path, monkeypatch):
    """A fake release: serves bytes by file name, and counts requests."""
    files, calls = {}, []
    monkeypatch.delenv(fa.ENV, raising=False)
    monkeypatch.setattr(fa, "folder", lambda: str(tmp_path / "cache"))

    def urlopen(url, timeout=None):
        calls.append(url)
        name = url.rsplit("/", 1)[1]
        if name not in files:
            raise OSError("404")
        return _Response(files[name])

    monkeypatch.setattr(fa.urllib.request, "urlopen", urlopen)
    return files, calls


def test_a_file_is_downloaded_once_and_checked(tmp_path, server, monkeypatch):
    files, calls = server
    data = _tiny(str(tmp_path / "x.vtp"))
    name = fa.file_name("atlas", "endo")
    files[name] = data
    monkeypatch.setitem(fa.FILES, name, hashlib.sha256(data).hexdigest())
    m = fa.load("atlas", "endo")
    assert m.n_cells > 0 and len(calls) == 1
    fa.load("atlas", "endo")
    assert len(calls) == 1                                    # cached: no second download


def test_labarthe_is_never_downloaded_only_read_locally(tmp_path, server):
    files, calls = server
    with pytest.raises(fa.AtlasError, match="not published"):
        fa.load("labarthe", "epi")
    assert not calls
    os.makedirs(fa.folder(), exist_ok=True)
    _tiny(os.path.join(fa.folder(), fa.file_name("labarthe", "epi")))
    assert fa.load("labarthe", "epi").n_cells > 0 and not calls


def test_a_damaged_download_is_refused_and_not_kept(tmp_path, server, monkeypatch):
    files, _ = server
    name = fa.file_name("atlas", "epi")
    files[name] = _tiny(str(tmp_path / "x.vtp"))
    monkeypatch.setitem(fa.FILES, name, "0" * 64)
    with pytest.raises(fa.AtlasError, match="damaged"):
        fa.load("atlas", "epi")
    assert not os.listdir(fa.folder())                       # no partial file left


def test_no_network_says_how_to_work_offline(server):
    with pytest.raises(fa.AtlasError, match=fa.ENV):
        fa.load("atlas", "endo")


def test_an_offline_folder_is_used_and_another_method_refused(tmp_path, monkeypatch):
    monkeypatch.setenv(fa.ENV, str(tmp_path))
    assert fa.folder() == str(tmp_path)
    _tiny(str(tmp_path / fa.file_name("atlas", "endo")))
    assert fa.load("atlas", "endo").n_cells > 0
    _tiny(str(tmp_path / fa.file_name("atlas", "epi")), method="older")
    with pytest.raises(fa.AtlasError, match="rebuild"):
        fa.load("atlas", "epi")
    with pytest.raises(fa.AtlasError, match="not published"):
        fa.load("labarthe", "endo")
    with pytest.raises(ValueError):
        fa.file_name("labarthe", "middle")


# ---------------------------------------------------------------------
# The published files, when this machine has them (never downloaded here)
# ---------------------------------------------------------------------
def _local(set_name, layer, chamber="la"):
    p = fa.path(set_name, layer, chamber)
    if not os.path.exists(p):
        pytest.skip(f"{p} not on this machine")
    return fa.load(set_name, layer, chamber)


@pytest.mark.parametrize("set_name", fa.SETS)
@pytest.mark.parametrize("layer", fa.LAYERS)
def test_each_published_atlas_has_what_mapping_needs(set_name, layer):
    m = _local(set_name, layer)
    for name in ("uac_alpha", "uac_beta", "uac_v2_alpha", "uac_v2_beta"):
        v = np.asarray(m.point_data[name])
        assert np.isfinite(v).all() and v.min() >= -1e-6 and v.max() <= 1 + 1e-6
    f = np.asarray(m.cell_data["fibre"], float)
    assert np.allclose(np.linalg.norm(f, axis=1), 1.0, atol=1e-3)
    tag = np.asarray(m.cell_data["elemTag"])
    assert set(np.unique(tag)) == {1, 11, 13, 15, 17, 19}
    # the left veins are the pair next to the LAA
    c = m.cell_centers().points
    laa = c[tag == 19].mean(0)
    dist = {L: np.linalg.norm(c[tag == L].mean(0) - laa) for L in (11, 13, 15, 17)}
    assert max(dist[11], dist[13]) < min(dist[15], dist[17])
    # ours and the atlas' own coordinates agree on which side each vein is
    t = m.faces.reshape(-1, 4)[:, 1:]
    for L in (11, 15):
        p = np.unique(t[tag == L])
        assert abs(m.point_data["uac_alpha"][p].mean() - m.point_data["uac_v2_alpha"][p].mean()) < 0.15
    if (set_name, layer) == ("labarthe", "epi"):
        bb = np.asarray(m.cell_data["bb"]) == 1
        assert bb.any() and np.isfinite(np.asarray(m.cell_data["fibre_bb"])[bb]).all()


@pytest.mark.parametrize("layer", fa.LAYERS)
def test_each_right_atrial_atlas_has_what_mapping_needs(layer):
    from ccdaf.core import atrial_labels as al
    m = _local("atlas", layer, "ra")
    for name in ("uac_alpha", "uac_beta", "uac_v2_alpha", "uac_v2_beta"):
        v = np.asarray(m.point_data[name])
        assert np.isfinite(v).all() and v.min() >= -1e-6 and v.max() <= 1 + 1e-6
    f = np.asarray(m.cell_data["fibre"], float)
    assert np.allclose(np.linalg.norm(f, axis=1), 1.0, atol=1e-3)
    tag = np.asarray(m.cell_data["elemTag"])
    assert set(np.unique(tag)) == set(al.RA.values)
    # the appendage on the lateral half (alpha below 0.5), as in the template
    t = m.faces.reshape(-1, 4)[:, 1:]
    assert m.point_data["uac_alpha"][np.unique(t[tag == 27])].mean() < 0.5
    assert fa.file_name("atlas", layer, "ra") in fa.FILES
