"""
test_help_menu.py
=================
The Help-menu content module (`ccdaf.gui.help_dialogs`): bundled offline assets
(logos, licence) resolve, the dialogs build headlessly, SVG logos render to
pixmaps, the local-docs resolver behaves, and external links are well-formed.

Runs headless with the offscreen Qt platform (the project's standard pytest
invocation sets ``QT_QPA_PLATFORM=offscreen``).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pytest
from PyQt5 import QtWidgets

import ccdaf
from ccdaf.gui import help_dialogs


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def test_bundled_assets_exist():
    for name in ("logo.svg", "CemrgLogo.svg", "LICENSE.txt"):
        assert help_dialogs.asset_path(name).is_file(), name


def test_license_text_is_bundled():
    text = help_dialogs.license_text()
    assert "BSD" in text and len(text.strip()) > 50


def test_cemrg_url():
    assert help_dialogs.CEMRG_URL == "https://cemrg.com"


def test_contact_email():
    assert help_dialogs.CONTACT_EMAIL == "c.corrado@imperial.ac.uk"


def test_svg_renders_to_pixmap(qapp):
    pm = help_dialogs.svg_pixmap("logo.svg", 64)
    assert not pm.isNull()
    assert pm.width() == 64


def test_dialogs_build(qapp):
    hs = help_dialogs.HelpDialog("Getting started", help_dialogs.GETTING_STARTED_HTML)
    assert hs.browser.toPlainText().strip()

    about = help_dialogs.AboutDialog(ccdaf.__version__)
    assert about.windowTitle() == "About CCDAF"


def test_local_docs_index_returns_none_or_existing():
    idx = help_dialogs.local_docs_index()
    assert idx is None or Path(idx).is_file()


def test_external_urls_are_https():
    for url in (help_dialogs.REPO_URL, help_dialogs.DOCS_URL,
                help_dialogs.ISSUES_URL, help_dialogs.CEMRG_URL):
        assert url.startswith("https://")
