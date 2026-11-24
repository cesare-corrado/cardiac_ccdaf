"""
test_uac_atrium_choice.py
=========================
Compute UAC reads the atrium from the labels: the left atrium as before,
the right one through ``uac.right``, a surface with both refused; the flat
map and Atrial fibres follow.
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


def test_fibres_are_left_atrial_only_for_now():
    from ccdaf.core.uac import fibres
    m = _atrium()
    m.point_data["uac_alpha"] = np.zeros(m.n_points)
    m.point_data["uac_beta"] = np.zeros(m.n_points)
    with pytest.raises(ValueError, match="left atrium only"):
        fibres.apply(m, "endo")
