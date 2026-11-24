"""
test_uac_atrium_choice.py
=========================
Compute UAC reads the atrium from the labels: the left atrium as before,
the right one through ``uac.right``, a surface with both refused; the flat
map and Atrial fibres follow (the atlas of the atrium on the mesh).
"""
from __future__ import annotations

import numpy as np
import pytest

from ccdaf.core import uac
from ccdaf.core.uac import right

from test_uac_right import _atrium


def test_the_atrium_is_read_from_the_labels():
    assert uac.which_atrium([1, 11, 13, 15, 17, 19]) == "LA"
    assert uac.which_atrium([1]) == "LA"                 # untagged: as before
    assert uac.which_atrium([2, 21, 23, 25, 27]) == "RA"
    assert uac.which_atrium([1, 11, 2, 21]) == "both"


def test_a_right_atrium_is_offered_and_both_are_refused():
    m = _atrium()
    assert uac.availability(m) == (True, "")
    tag = np.asarray(m.cell_data["elemTag"]).copy()
    m2 = m.copy()
    m2.cell_data["elemTag"] = np.where(tag == 25, 2, tag)     # no CS
    ok, why = uac.availability(m2)
    assert not ok and "CS" in why and "body 2" in why
    both = m.copy()
    tag[:3] = 11
    both.cell_data["elemTag"] = tag
    ok, why = uac.availability(both)
    assert not ok and why == uac.BOTH_ATRIA
    with pytest.raises(uac.UacError, match="both atria"):
        uac.compute(*uac.inputs(both))


def test_compute_and_the_flat_map_use_the_right_atrium():
    m = _atrium()
    run = uac.compute(*uac.inputs(m), uac.Options(iterations=40))
    done = uac.write(m, run)
    flat = uac.flat_mesh(done)
    uv, tri, cells, src = right.flat_from_values(done.point_data["uac_alpha"],
                                                done.point_data["uac_beta"],
                                                done.faces.reshape(-1, 4)[:, 1:])
    assert np.allclose(np.asarray(flat.points)[:, :2], uv)
    # alpha runs lateral (0) to septal (1): the appendage on the lateral half
    raa = np.unique(done.faces.reshape(-1, 4)[:, 1:][done.cell_data["elemTag"] == 27])
    assert done.point_data["uac_alpha"][raa].mean() < 0.5


def test_fibres_on_a_right_atrium_use_its_own_flat_map_and_atlas(monkeypatch):
    """A right atrium is mapped with the right atrium's cut rules and atlas:
    an atlas mapped onto itself gives back its fibres, also on the cut."""
    from ccdaf.core.uac import fibres
    m = _atrium()
    run = uac.compute(*uac.inputs(m), uac.Options(iterations=40))
    done = uac.write(m, run)
    t = done.faces.reshape(-1, 4)[:, 1:]
    P = np.asarray(done.points)
    e = P[t[:, 1]] - P[t[:, 0]]                        # a fibre in each triangle
    done.cell_data["fibre"] = e / np.linalg.norm(e, axis=1, keepdims=True)
    asked = []
    monkeypatch.setattr(fibres.fibre_atlas, "load",
                        lambda s, layer, chamber="la": asked.append(chamber) or done)
    f, rep = fibres.map_layer(done, "endo")
    assert asked == ["ra"] and "right-atrial" in rep.lines()[0]
    ang = np.degrees(np.arccos(np.clip(np.abs((f * done.cell_data["fibre"]).sum(1)), 0, 1)))
    assert np.percentile(ang, 99) < 1.0


def test_fibres_refuse_a_surface_with_both_atria():
    from ccdaf.core.uac import fibres
    m = _atrium()
    tag = np.asarray(m.cell_data["elemTag"]).copy()
    tag[:3] = 11
    m.cell_data["elemTag"] = tag
    m.point_data["uac_alpha"] = np.zeros(m.n_points)
    m.point_data["uac_beta"] = np.zeros(m.n_points)
    with pytest.raises(ValueError, match="both atria"):
        fibres.apply(m, "endo")
