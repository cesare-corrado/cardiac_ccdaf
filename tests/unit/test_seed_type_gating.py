"""
test_seed_type_gating.py
========================
What each panel offers under each seed type.

One dropdown — the Seed selection panel's **Seed type** — decides the
anatomy, and Tagging, Manual correction and Clipping follow it. Each is
handed the active profile's own lists and must switch itself off when
that list is empty, rather than offering the left atrium's veins on a
mesh that has none.

The contract:

* Tagging shows exactly the profile's ``radius_names``, in that order,
  and its Run button is dead for a profile that does not tag — even with
  seeds reported complete;
* ``radius_factors`` reports only the active profile's radii, so a value
  tuned under one seed type cannot reach another's tagger config;
* a tuned radius survives a switch away and back, because the rows are
  hidden rather than destroyed;
* Manual correction offers exactly the profile's ``label_values`` and
  cannot be made active without any — ``set_active(True)`` is refused,
  not obeyed;
* Clipping offers exactly the profile's ``clip_regions``, and with none
  the activation checkbox goes too: a tick that can never lead to a clip
  would still take the X key from manual correction;
* repopulating a combo is not a user selection, so it emits no
  ``label_changed`` / ``selection_changed``;
* the label entries the app builds match the profile;
* the mesh kind is a *second*, independent gate on the same panels: a
  volume switches all of them off whatever the seed type says, and says
  so in its own words rather than blaming the seed type.

Qt only — no display beyond the offscreen platform, no mesh, no window
(constructing one needs a GL context).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pytest
from PyQt5 import QtWidgets

from ccdaf.app.ccdaf import CCDAF
from ccdaf.core.seed_profiles import (
    LANDMARKS_LA_UAC_PROFILE, SEED_LA_PROFILE, SEED_PROFILE_ORDER,
    SEED_RA_PROFILE,
)
from ccdaf.gui.clipping_widget import ClippingWidget
from ccdaf.gui.manual_correction_widget import ManualCorrectionWidget
from ccdaf.gui.postprocessing_widget import PostprocessingWidget
from ccdaf.gui.volume_postprocessing_widget import VolumePostprocessingWidget
from ccdaf.gui.tagging_widget import TaggingWidget


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def panels(qapp):
    """The three dependent panels, as the window builds them."""
    tagging = TaggingWidget()
    manual = ManualCorrectionWidget(CCDAF._label_entries(SEED_LA_PROFILE))
    clipping = ClippingWidget(list(SEED_LA_PROFILE.clip_regions))
    return tagging, manual, clipping


def _apply(profile, panels):
    """Do to the panels exactly what ``CCDAF._sync_profile_panels`` does."""
    tagging, manual, clipping = panels
    tagging.set_profile(profile)
    manual.set_label_entries(CCDAF._label_entries(profile), follows=profile.label)
    clipping.set_regions(list(profile.clip_regions), follows=profile.label)


# ---------------------------------------------------------------- tagging
def test_radii_are_the_profiles_own(panels):
    tagging, _, _ = panels
    for profile in SEED_PROFILE_ORDER:
        _apply(profile, panels)
        tagging.set_seeds_complete(True)
        assert tuple(tagging.radius_factors()) == tuple(profile.radius_names)


def test_run_button_needs_a_profile_that_tags(panels):
    tagging, _, _ = panels
    _apply(SEED_LA_PROFILE, panels)
    tagging.set_seeds_complete(True)
    assert tagging.btn_tag.isEnabled()

    # Complete seeds are not enough: the seed type has to tag something.
    for profile in (LANDMARKS_LA_UAC_PROFILE, SEED_RA_PROFILE):
        _apply(profile, panels)
        tagging.set_seeds_complete(True)
        assert not tagging.btn_tag.isEnabled()


def test_tuned_radius_survives_a_round_trip(panels):
    tagging, _, _ = panels
    _apply(SEED_LA_PROFILE, panels)
    tagging.spn_radius["LAA"].setValue(42.0)

    _apply(SEED_RA_PROFILE, panels)
    assert tagging.radius_factors() == {}      # nothing of another set leaks

    _apply(SEED_LA_PROFILE, panels)
    assert tagging.radius_factors()["LAA"] == pytest.approx(42.0)


# ------------------------------------------------------- manual correction
def test_labels_are_the_profiles_own(panels):
    _, manual, _ = panels
    for profile in SEED_PROFILE_ORDER:
        _apply(profile, panels)
        offered = [manual.cmb_label.itemData(i)
                   for i in range(manual.cmb_label.count())]
        assert offered == [int(v) for v in profile.label_values]


def test_manual_cannot_be_activated_without_labels(panels):
    _, manual, _ = panels
    _apply(SEED_LA_PROFILE, panels)
    manual.set_active(True)
    assert manual.btn_edit_toggle.isEnabled()
    assert manual.current_label() is not None

    _apply(SEED_RA_PROFILE, panels)
    assert manual.current_label() is None
    manual.set_active(True)               # refused, not obeyed
    assert not manual.btn_edit_toggle.isEnabled()
    assert not manual.btn_accept.isEnabled()
    assert not manual.btn_snake.isEnabled()
    assert not manual.cmb_label.isEnabled()

    # ... and accepting a tagging must not resurrect it either.
    manual.on_accepted()
    assert not manual.btn_edit_toggle.isEnabled()


def test_repopulating_labels_is_not_a_user_selection(panels):
    _, manual, _ = panels
    _apply(SEED_LA_PROFILE, panels)
    seen = []
    manual.label_changed.connect(seen.append)
    # Same set again: the current label is still there, so nothing changed.
    _apply(SEED_LA_PROFILE, panels)
    assert seen == []


def test_kept_label_survives_a_switch_back(panels):
    _, manual, _ = panels
    _apply(SEED_LA_PROFILE, panels)
    idx = manual.cmb_label.findData(int(SEED_LA_PROFILE.label_values[-1]))
    manual.set_label_index(idx)
    kept = manual.current_label()

    _apply(SEED_RA_PROFILE, panels)
    _apply(SEED_LA_PROFILE, panels)
    assert manual.current_label() == kept


# --------------------------------------------------------------- clipping
def test_regions_are_the_profiles_own(panels):
    _, _, clipping = panels
    for profile in SEED_PROFILE_ORDER:
        _apply(profile, panels)
        offered = [clipping.cmb_region.itemData(i)
                   for i in range(clipping.cmb_region.count())]
        assert offered == list(profile.clip_regions)


def test_clipping_is_dead_without_regions(panels):
    _, _, clipping = panels
    _apply(SEED_LA_PROFILE, panels)
    clipping.set_enabled_after_accept()
    clipping.chk_active.setChecked(True)
    assert clipping.btn_start.isEnabled()

    for profile in (LANDMARKS_LA_UAC_PROFILE, SEED_RA_PROFILE):
        _apply(profile, panels)
        assert clipping.selected_region() == ""
        assert not clipping.chk_active.isEnabled()
        assert not clipping.chk_active.isChecked()   # a live tick is withdrawn
        assert not clipping.btn_start.isEnabled()
        assert not clipping.btn_apply.isEnabled()
        assert not clipping.cmb_region.isEnabled()

    # Coming back restores it — the accepted tagging was never in question.
    _apply(SEED_LA_PROFILE, panels)
    assert clipping.chk_active.isEnabled()
    clipping.chk_active.setChecked(True)
    assert clipping.btn_start.isEnabled()


def test_kept_region_survives_a_switch_back(panels):
    _, _, clipping = panels
    _apply(SEED_LA_PROFILE, panels)
    clipping.cmb_region.setCurrentIndex(
        clipping.cmb_region.findData(SEED_LA_PROFILE.clip_regions[-1]))
    kept = clipping.selected_region()

    _apply(SEED_RA_PROFILE, panels)
    _apply(SEED_LA_PROFILE, panels)
    assert clipping.selected_region() == kept


def test_repopulating_regions_is_not_a_user_selection(panels):
    _, _, clipping = panels
    _apply(SEED_LA_PROFILE, panels)
    seen = []
    clipping.selection_changed.connect(lambda r, m: seen.append((r, m)))
    _apply(SEED_LA_PROFILE, panels)
    assert seen == []


# ------------------------------------------------------------ app helper
def test_label_entries_follow_the_profile():
    entries = CCDAF._label_entries(SEED_LA_PROFILE)
    assert [v for v, _ in entries] == [int(v) for v in SEED_LA_PROFILE.label_values]
    assert all(name for _, name in entries)          # every label is named
    assert CCDAF._label_entries(SEED_RA_PROFILE) == []


# ----------------------------------------------------------- volume mode
def _apply_volume(profile, panels):
    """Do to the panels what ``_sync_profile_panels`` does on a volume."""
    tagging, manual, clipping = panels
    reason = CCDAF.VOLUME_NOTE
    tagging.set_profile(profile, reason=reason, enabled=False)
    manual.set_label_entries([], follows=profile.label, reason=reason)
    clipping.set_regions([], follows=profile.label, reason=reason)


def test_a_volume_switches_every_surface_panel_off(panels):
    tagging, manual, clipping = panels
    _apply(SEED_LA_PROFILE, panels)          # the fully-capable seed type
    manual.set_active(True)
    clipping.set_enabled_after_accept()
    clipping.chk_active.setChecked(True)
    assert tagging.radius_factors() and manual.current_label() is not None

    _apply_volume(SEED_LA_PROFILE, panels)
    tagging.set_seeds_complete(True)         # not enough on a volume
    assert not tagging.btn_tag.isEnabled()
    assert tagging.radius_factors() == {}
    assert manual.current_label() is None
    assert not manual.btn_edit_toggle.isEnabled()
    assert not clipping.chk_active.isEnabled()
    assert not clipping.chk_active.isChecked()


def test_a_volume_says_why_rather_than_blaming_the_seed_type(panels):
    """The message names the real reason.

    seed_LA defines radii, labels and clip regions. Saying "this seed
    type has none" would be false and would send the user to the one
    control that cannot help.
    """
    tagging, manual, clipping = panels
    _apply_volume(SEED_LA_PROFILE, panels)
    for widget in (tagging, manual, clipping):
        text = widget.lbl_follows.text()
        assert "volumetric" in text
        assert "no regions to tag" not in text
        assert "no labels to correct" not in text
        assert "no regions to clip" not in text
        assert SEED_LA_PROFILE.label in text      # still names the seed type


def test_leaving_volume_mode_restores_the_panels(panels):
    tagging, manual, clipping = panels
    _apply_volume(SEED_LA_PROFILE, panels)
    _apply(SEED_LA_PROFILE, panels)
    tagging.set_seeds_complete(True)
    assert tagging.btn_tag.isEnabled()
    assert manual.current_label() is not None
    assert clipping.chk_active.isEnabled()


def test_one_post_processing_panel_or_the_other(qapp):
    """The two panels swap with the mesh kind; they are never both up.

    The surface steps rebuild a surface and hand it back, which on a
    volume would replace the tetrahedra with their boundary. They are not
    disabled-in-place but replaced, because what a volume needs is a
    different set of controls, not a greyed-out version of these.
    """
    surface = PostprocessingWidget(
        mesh_getter=lambda: None, mesh_setter=lambda m: None,
        on_status=lambda msg: None)
    volume = VolumePostprocessingWidget()

    for is_volume in (False, True, False):
        surface.setVisible(not is_volume)
        volume.setVisible(is_volume)
        assert surface.isVisibleTo(surface) is not volume.isVisibleTo(volume)


def test_the_volume_panel_refuses_a_target_and_a_band_together(qapp):
    """MMG rejects both; the panel makes the pairing untypable.

    Reported by the controls rather than by an error box after the click.
    """
    panel = VolumePostprocessingWidget()
    assert panel.spn_target.isEnabled() and panel.spn_min.isEnabled()

    panel.spn_target.setValue(2.0)
    assert not panel.spn_min.isEnabled()
    assert not panel.spn_max.isEnabled()
    panel.options().validate()                 # a target alone is valid

    panel.spn_target.setValue(0.0)
    panel.spn_min.setValue(1.0)
    assert not panel.spn_target.isEnabled()
    panel.options().validate()                 # a band alone is valid


def test_the_boundary_is_frozen_unless_asked(qapp):
    """The default must not move the anatomy."""
    panel = VolumePostprocessingWidget()
    assert panel.options().freeze_boundary is True
    # The tolerance is meaningless while the boundary is frozen, and says so.
    assert not panel.spn_hausdorff.isEnabled()

    panel.chk_adapt_boundary.setChecked(True)
    assert panel.options().freeze_boundary is False
    assert panel.spn_hausdorff.isEnabled()


def test_the_boundary_knobs_are_inert_while_it_is_frozen(qapp):
    """Gradation measurably does nothing with a frozen boundary.

    It limits how fast a *varying* size may change, and the size only
    varies while MMG derives it from surface curvature — which it does
    only when adapting the boundary. Changing it with the boundary frozen
    gave byte-identical meshes, so the control is disabled and the value
    is not sent: a number in the options that had no effect on the result
    would misdescribe what produced it.
    """
    panel = VolumePostprocessingWidget()
    panel.spn_gradation.setValue(1.05)
    panel.spn_hausdorff.setValue(0.4)

    assert not panel.spn_gradation.isEnabled()
    assert not panel.spn_hausdorff.isEnabled()
    frozen = panel.options()
    assert frozen.gradation == 0.0 and frozen.hausdorff == 0.0
    assert "hgrad" not in frozen.as_mmg_options()
    assert "hausd" not in frozen.as_mmg_options()

    panel.chk_adapt_boundary.setChecked(True)
    assert panel.spn_gradation.isEnabled()
    adapting = panel.options()
    assert adapting.gradation == pytest.approx(1.05)
    assert adapting.as_mmg_options()["hgrad"] == pytest.approx(1.05)
