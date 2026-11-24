"""
test_right_atrium_tagging.py
============================
The right atrium in the tagging workflow: one label table for both atria,
tagging that writes the right-atrial labels when asked, island joining and
smoothing that keep each atrium's body, the editor filling unassigned cells
with the right body, the tricuspid valve treated like the mitral one in
clipping, and radius defaults that follow the region.
"""
from __future__ import annotations

import numpy as np
import pytest
import pyvista as pv

from ccdaf.core import atrial_labels as al
from ccdaf.core.region_tagger import RegionTagger, TaggerConfig

from test_uac_right import VESSELS, RAA_DIR, _atrium


def test_the_label_table_keeps_the_atria_apart():
    assert al.LA.values == (11, 13, 15, 17, 19, 1)
    assert al.RA.values == (21, 23, 25, 27, 2)
    assert al.NAMES[1] == "LA body" and al.NAMES[2] == "RA body" and al.NAMES[25] == "CS"
    assert set(al.COLORS) == set(al.LA.values) | set(al.RA.values)
    assert len(set(al.COLORS.values())) == len(al.COLORS)          # one colour each
    assert al.region_tag("MV") == 1 and al.region_tag("TV") == 2
    assert al.region_tag("SVC") == 21 and al.region_tag("nothing") is None
    # a body number alone is not a tagging: a ventricular mesh uses 1 and 2
    assert al.legend_atria([1, 2]) is None
    assert al.legend_atria([1]) == (al.LA,)
    assert al.legend_atria([2, 21]) == (al.RA,)
    assert al.legend_atria([1, 11, 2, 23]) == (al.LA, al.RA)
    assert al.body_for([21, 2]) == 2 and al.body_for([11]) == 1 and al.body_for([]) == 1


def _seeds(m):
    """A seed at the centre of each labelled right-atrial region."""
    c = m.cell_centers().points
    tag = np.asarray(m.cell_data["elemTag"])
    out = {}
    for name, L in al.RA.regions.items():
        idx = np.where(tag == L)[0]
        out[name] = c[idx][np.argmin(np.linalg.norm(c[idx] - c[idx].mean(0), axis=1))]
    return out


def test_tagging_a_right_atrium_writes_its_labels():
    ref = _atrium()
    seeds = _seeds(ref)
    m = ref.copy()
    m.cell_data["elemTag"] = np.ones(m.n_cells, np.int32)            # a fresh mesh
    # the synthetic "vessels" are caps on a sphere, not tubes, so no ostium
    # stops the front: give radii that match their size (about 8 edges)
    tg = RegionTagger(m, TaggerConfig(svc_radius_factor=8, ivc_radius_factor=8,
                                      cs_radius_factor=6, raa_radius_factor=8))
    tag = tg.tag(seeds, atrium=al.RA)
    assert set(np.unique(tag)) == set(al.RA.values)                    # body 2, no LA label
    truth = np.asarray(ref.cell_data["elemTag"])
    for L in al.RA.regions.values():
        a, b = tag == L, truth == L
        assert 2 * (a & b).sum() / (a.sum() + b.sum()) > 0.5            # found where it is
    # the left atrium is still the default
    m2 = ref.copy()
    lsd = {k: v for k, v in zip(al.LA.regions, seeds.values())}
    lsd["LAA"] = seeds["RAA"]
    tag2 = RegionTagger(m2).tag(lsd)
    assert set(np.unique(tag2)) <= set(al.LA.values)


def test_islands_and_smoothing_keep_each_atrium_s_body():
    m = _atrium()
    tg = RegionTagger(m)
    tag = np.asarray(m.cell_data["elemTag"]).copy()
    body = np.where(tag == al.RA.body)[0]
    lonely = body[len(body) // 2]
    tag[lonely] = 21                                                     # a one-cell SVC island
    out = tg.reduce_to_single_components(tag)
    assert out[lonely] == al.RA.body                                     # back to the RA body, not 1
    assert al.LA.body not in out
    grown = tg.dilate_label(out, 21)
    assert (grown == 21).sum() >= (out == 21).sum() and al.LA.body not in grown
    shrunk = tg.erode_label(out, 21)
    assert al.LA.body not in shrunk


def test_the_editor_fills_unassigned_cells_with_the_right_body():
    from ccdaf.core.mesh_loader import UNASSIGNED
    from ccdaf.interaction.manual_editor import ALLOWED_LABELS, ManualEditor
    assert set(ALLOWED_LABELS) == set(al.LA.values) | set(al.RA.values)
    m = _atrium()
    tag = np.asarray(m.cell_data["elemTag"]).copy()
    tag[:5] = UNASSIGNED
    m.cell_data["elemTag"] = tag

    class _P:
        def __getattr__(self, name):
            return lambda *a, **k: None

    ed = ManualEditor(mesh=m, plotter=_P(), on_render=None, on_state=lambda s: None,
                      on_commit=lambda: None, active_label=21)
    ed.accept()
    assert (np.asarray(m.cell_data["elemTag"])[:5] == al.RA.body).all()


def test_radius_defaults_follow_the_region():
    from ccdaf.gui.tagging_widget import radius_default
    cfg = TaggerConfig()
    assert cfg.lspv_radius_factor == 25.0 and cfg.laa_radius_factor == 25.0
    assert cfg.svc_radius_factor == cfg.ivc_radius_factor == cfg.raa_radius_factor == 50.0
    assert cfg.cs_radius_factor == 25.0
    assert radius_default("SVC") == 50.0 and radius_default("LSPV") == 25.0
    assert radius_default("unknown") == 25.0


@pytest.fixture(scope="module")
def qapp():
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def test_the_tricuspid_valve_has_no_contour_like_the_mitral(qapp):
    from ccdaf.core.seed_profiles import SEED_RA_PROFILE
    from ccdaf.gui.clipping_widget import MODE_CONTOUR, ClippingWidget
    w = ClippingWidget(list(SEED_RA_PROFILE.clip_regions))
    w.cmb_region.setCurrentIndex(w.cmb_region.findData("TV"))
    item = w.cmb_mode.model().item(w.cmb_mode.findData(MODE_CONTOUR))
    assert not item.isEnabled() and "TV" in item.toolTip()
    w.cmb_region.setCurrentIndex(w.cmb_region.findData("SVC"))
    assert w.cmb_mode.model().item(w.cmb_mode.findData(MODE_CONTOUR)).isEnabled()


def test_clipping_a_right_region_takes_its_own_label():
    from ccdaf.app.ccdaf import CCDAF

    class _Host:
        loader = type("L", (), {"mesh": _atrium()})()
    host = _Host()
    assert CCDAF._clip_tag_for(host, "SVC") == 21
    assert CCDAF._clip_tag_for(host, "TV") == al.RA.body


_ = (VESSELS, RAA_DIR, pv)
