"""
Help dialogs and Help-menu content.

Static, offline help shown from the **Help** menu: a *Getting started*
walkthrough, a keyboard/mouse reference, and an *About* box. External links
(full documentation, issue tracker, the CEMRG site) are opened in the browser by
the host app.

Everything needed offline — the logos and the licence — is bundled in
``ccdaf/gui/assets`` so the dialogs render with no network and no external
viewer. The full documentation site is opened in the browser (local build if
present, see :func:`local_docs_index`).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtSvg import QSvgRenderer

# The app prefers a local docs build (see local_docs_index); DOCS_URL is the
# online fallback — the GitHub Pages site published by .github/workflows/docs.yml.
REPO_URL = "https://github.com/cesare-corrado/cardiac_ccdaf"
DOCS_URL = "https://cesare-corrado.github.io/cardiac_ccdaf/"
ISSUES_URL = f"{REPO_URL}/issues"
CEMRG_URL = "https://cemrg.com"
CONTACT_EMAIL = "c.corrado@imperial.ac.uk"

_ASSETS = Path(__file__).resolve().parent / "assets"


# ---------------------------------------------------------------------------
# Asset helpers
# ---------------------------------------------------------------------------
def asset_path(name: str) -> Path:
    """Absolute path to a bundled asset (logo / licence)."""
    return _ASSETS / name


def svg_pixmap(name: str, width: int) -> QtGui.QPixmap:
    """Render a bundled SVG to a transparent pixmap of the given width."""
    renderer = QSvgRenderer(str(asset_path(name)))
    size = renderer.defaultSize()
    height = int(width * size.height() / size.width()) if size.width() else width
    pixmap = QtGui.QPixmap(width, height)
    pixmap.fill(Qt.transparent)
    painter = QtGui.QPainter(pixmap)
    renderer.render(painter)
    painter.end()
    return pixmap


def license_text() -> str:
    """The bundled licence text (BSD-3), or a short fallback."""
    try:
        return asset_path("LICENSE.txt").read_text(encoding="utf-8")
    except OSError:
        return "Licence file not found. See the project's LICENSE."


def local_docs_index() -> Optional[str]:
    """Path to a locally built docs site (``site/index.html``), or ``None``.

    Searches upward from this module for a repository checkout whose docs have
    been built with ``mkdocs build``. Lets *Documentation* open with no network.
    """
    for parent in Path(__file__).resolve().parents:
        index = parent / "site" / "index.html"
        if index.is_file():
            return str(index)
        if (parent / "mkdocs.yml").is_file():
            break
    return None


# ---------------------------------------------------------------------------
# Rich-text help (Getting started, Shortcuts)
# ---------------------------------------------------------------------------
GETTING_STARTED_HTML = """
<h2>Getting started</h2>
<p>CCDAF takes a left-atrial surface from a raw mesh to a tagged, clipped
result. Each step is a side panel; they unlock in order.</p>
<p><b>The two shortcuts to remember:</b> over the 3D view, <b>X</b> picks —
a triangle, or a point for the geodesic <i>snake</i> tools — and <b>C</b>
commits the manual-correction selection batch.</p>
<ol>
  <li><b>Load a mesh</b> — <i>File &rarr; Load</i> a <code>.vtk</code> surface.</li>
  <li><b>Place the six seeds</b> — <i>Seed selection &rarr; Start</i>, then click
      LSPV, LIPV, RSPV, RIPV, LAA, MV in order.</li>
  <li><b>Tag automatically</b> — set the radius factors, then
      <i>Run automatic tagging</i>.</li>
  <li><b>Correct by hand</b> — in <i>Manual correction</i>, pick a label, then
      select triangles with <b>X</b> (press <b>C</b> to commit) or use the
      snake.</li>
  <li><b>Clip</b> — tick <i>Clipping active</i>, then draw a PV contour or place
      a mitral sphere/plane and <i>Apply clip</i>.</li>
  <li><b>Export</b> — <i>File &rarr; Save</i>, or <i>EAM &rarr; Export</i> for a
      Carto bundle / VTK.</li>
</ol>
<p>See <b>Help &rarr; Documentation</b> for the full guide.</p>
"""


SHORTCUTS_HTML = """
<h2>Keyboard &amp; mouse</h2>
<h3>3D view</h3>
<table cellpadding="4">
  <tr><td><b>Left-drag</b></td><td>Rotate</td></tr>
  <tr><td><b>Scroll</b></td><td>Zoom</td></tr>
  <tr><td><b>Right-drag</b></td><td>Zoom / dolly</td></tr>
  <tr><td><b>Middle-drag</b></td><td>Pan</td></tr>
</table>
<h3>Picking &amp; committing</h3>
<table cellpadding="4">
  <tr><td><b>Left-click</b></td><td>Place a seed (seed selection only)</td></tr>
  <tr><td><b>X</b></td><td>Pick — add a triangle to the selection batch, or
      drop a point for a snake tool</td></tr>
  <tr><td><b>C</b></td><td>Commit the manual-correction selection batch</td></tr>
</table>
<p><b>Who owns <code>X</code>:</b> it is the <i>pick</i> key, shared by
manual-correction selection, the manual snake and the PV-contour snake; the
<i>Clipping active</i> checkbox decides whether it belongs to clipping or to
manual correction. Seeds are the exception — still placed by left-click.
Committing never shares a key with picking: <b>C</b> commits the selection
batch, and the snake and the clip have their own commit buttons.</p>
"""


class HelpDialog(QtWidgets.QDialog):
    """A simple modal dialog showing scrollable rich-text help."""

    def __init__(self, title: str, html: str,
                 parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(580, 540)

        layout = QtWidgets.QVBoxLayout(self)
        self.browser = QtWidgets.QTextBrowser()
        self.browser.setOpenExternalLinks(True)
        self.browser.setHtml(html)
        layout.addWidget(self.browser)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)


class AboutDialog(QtWidgets.QDialog):
    """The About box: title + app logo, description, CEMRG credit, licence."""

    def __init__(self, version: str,
                 parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("About CCDAF")
        self.setMinimumWidth(520)
        layout = QtWidgets.QVBoxLayout(self)

        # --- Header row: title (left) + app logo (top right) ---------------
        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel(
            "<h2>Cardiac Clinical Data Analysis Framework (CCDAF)</h2>"
            f"<p style='color:gray;'>Version {version}</p>"
        )
        title.setTextFormat(Qt.RichText)
        header.addWidget(title, 1)

        logo = QtWidgets.QLabel()
        logo.setPixmap(svg_pixmap("logo.svg", 72))
        logo.setAlignment(Qt.AlignTop | Qt.AlignRight)
        header.addWidget(logo, 0, Qt.AlignTop)
        layout.addLayout(header)

        # --- Description + disclaimer -------------------------------------
        body = QtWidgets.QLabel(
            "<p>A GUI for post-processing left-atrial surface meshes in cardiac "
            "electrophysiology research.</p>"
            "<p style='color:#b3202f;'><b>NOT FOR CLINICAL USE.</b> Research "
            "software only; not validated for clinical decision-making.</p>"
        )
        body.setWordWrap(True)
        body.setTextFormat(Qt.RichText)
        layout.addWidget(body)

        # --- Developed by: clickable CEMRG logo ---------------------------
        dev_row = QtWidgets.QHBoxLayout()
        dev_row.addWidget(QtWidgets.QLabel("Developed by:"))
        cemrg = QtWidgets.QPushButton()
        cemrg.setIcon(QtGui.QIcon(svg_pixmap("CemrgLogo.svg", 132)))
        cemrg.setIconSize(QtCore.QSize(132, int(132 * 286.7 / 446.0)))
        cemrg.setFlat(True)
        cemrg.setCursor(Qt.PointingHandCursor)
        cemrg.setToolTip(CEMRG_URL)
        cemrg.clicked.connect(
            lambda: QtGui.QDesktopServices.openUrl(QtCore.QUrl(CEMRG_URL)))
        dev_row.addWidget(cemrg)
        dev_row.addStretch(1)
        layout.addLayout(dev_row)

        # --- Links: source, issues, licence -------------------------------
        links = QtWidgets.QLabel(
            f"<a href='{REPO_URL}'>Source code</a> &nbsp;·&nbsp; "
            f"<a href='{ISSUES_URL}'>Report an issue</a> &nbsp;·&nbsp; "
            f"<a href='mailto:{CONTACT_EMAIL}'>Contact</a> &nbsp;·&nbsp; "
            "<a href='ccdaf:license'>Licence</a>"
        )
        links.setTextFormat(Qt.RichText)
        links.setOpenExternalLinks(False)     # handle the licence link ourselves
        links.linkActivated.connect(self._on_link)
        layout.addWidget(links)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

    def _on_link(self, href: str) -> None:
        if href == "ccdaf:license":
            HelpDialog("Licence", f"<pre>{license_text()}</pre>", self).exec_()
        else:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl(href))


__all__ = [
    "HelpDialog",
    "AboutDialog",
    "GETTING_STARTED_HTML",
    "SHORTCUTS_HTML",
    "asset_path",
    "svg_pixmap",
    "license_text",
    "local_docs_index",
    "REPO_URL",
    "DOCS_URL",
    "ISSUES_URL",
    "CEMRG_URL",
    "CONTACT_EMAIL",
]
