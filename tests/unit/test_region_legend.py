"""
test_region_legend.py
=====================
What the Regions view draws when the tags are not the atrium's.

The colour table was the six left-atrial labels and nothing else, and a
value outside it fell to index 0 — the body colour. A mesh arriving with
its own labelling — a ventricular mesh labels the left ventricle 1 and
the right ventricle 2 — therefore rendered as one flat grey with the
pulmonary veins named in the legend beside it. Nothing said the mesh
carried a labelling at all.

The contract:

* an atrial tagging keeps the full six-entry legend in its anatomical
  colours, tagged or not — the legend doubles as a key to what tagging
  will produce, so it lists regions that are not there yet;
* any other set of values gets a legend built from what is present,
  named by number and coloured so that two labels are two colours;
* every tag present is in the table, so nothing is drawn in another
  label's colour;
* the field the visualisation panel starts on is the mesh's own
  labelling, so the panel and the view agree from the moment it opens.

Qt is imported (the app module needs it) but no window is built.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pytest

from ccdaf.app.ccdaf import (
    FALLBACK_TAG_COLORS, LABEL_COLORS, _region_legend,
)
from ccdaf.core.mesh_loader import BODY_LABEL


ATRIAL = sorted(LABEL_COLORS)


def test_an_atrial_tagging_keeps_the_anatomical_legend():
    for tags in ([BODY_LABEL] * 5, ATRIAL, [1, 11, 19]):
        ordered, colours, names = _region_legend(np.asarray(tags))
        assert ordered == ATRIAL
        assert colours == [LABEL_COLORS[t] for t in ATRIAL]
        assert names[0] == "body"
        assert "LSPV" in names.values()


def test_an_unknown_labelling_is_named_by_number():
    """Calling the right ventricle "LSPV" would be false."""
    ordered, colours, names = _region_legend(np.asarray([1, 2]))
    assert ordered == [1, 2]
    assert list(names.values()) == ["1", "2"]
    assert "LSPV" not in names.values()
    assert "body" not in names.values()


def test_two_labels_get_two_colours():
    """The bug: everything the table did not know became body grey."""
    _, colours, _ = _region_legend(np.asarray([1, 2]))
    assert len(set(colours)) == 2
    assert LABEL_COLORS[BODY_LABEL] not in colours


@pytest.mark.parametrize("tags", [
    [1, 2],
    [1, 2, 7],
    [0, 3, 4, 5, 6, 8, 9, 20, 21, 22, 23],      # more than the palette holds
    [42],
])
def test_every_tag_present_is_in_the_table(tags):
    """No value may fall through to another label's colour."""
    ordered, colours, names = _region_legend(np.asarray(tags))
    assert set(ordered) == set(tags)
    assert len(colours) == len(ordered) == len(names)
    # Wrapping the palette is allowed; silently dropping a tag is not.
    if len(ordered) <= len(FALLBACK_TAG_COLORS):
        assert len(set(colours)) == len(colours)


def test_a_mixed_tagging_is_not_treated_as_atrial():
    """One foreign value is enough: the atrial names no longer describe it."""
    ordered, _, names = _region_legend(np.asarray([1, 11, 2]))
    assert ordered == [1, 2, 11]
    assert list(names.values()) == ["1", "2", "11"]


def test_an_empty_tagging_still_yields_a_table():
    ordered, colours, names = _region_legend(np.asarray([], dtype=int))
    assert ordered and colours and names
