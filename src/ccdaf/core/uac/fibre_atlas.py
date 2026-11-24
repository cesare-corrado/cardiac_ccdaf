"""
fibre_atlas.py
==============
The atrial fibre atlases, for mapping fibres onto an atrium (left or right)
through its universal atrial coordinates.

Two published fibre sets, each for the endocardium and the epicardium
(Roney et al., the UAC fibre atlas):

* ``labarthe``: the rule-based bilayer fibres of Labarthe et al.; the
  epicardial file also carries Bachmann's bundle (``fibre_bb`` where
  ``bb`` is 1);
* ``atlas``: the average of the ex vivo diffusion-tensor fibre sets, as
  published on Zenodo (record 3764917, CC BY 4.0).

Each file is a surface in millimetres with ccdaf labels (``elemTag``),
ccdaf's coordinates (``uac_alpha``, ``uac_beta``), the atlas' own original
coordinates (``uac_v2_alpha``, ``uac_v2_beta``) and one unit fibre per
triangle (``fibre``).

The files are not part of the package; ``tools/build_fibre_atlas.py``
builds them. The ``atlas`` files (CC BY 4.0, from the published fibre
atlas) are assets of the GitHub release :data:`RELEASE`: :func:`load`
downloads a missing one once into :func:`folder` and checks it against
:data:`FILES`. The ``labarthe`` files are not published (their source is
not public), so they are used only from a local folder: put them in
``$CCDAF_FIBRE_ATLAS`` or in :func:`folder`. With ``CCDAF_FIBRE_ATLAS``
set, nothing is downloaded (offline machines). :data:`UAC_METHOD` names the coordinate method the files were
computed with, so a file built by another method is refused rather than
used with coordinates that no longer match.
"""
from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import urllib.request
from typing import Callable, Dict, Optional, Tuple

import pyvista as pv

#: The coordinate method the published files were computed with. Bump it,
#: rebuild the files and publish them under a new :data:`RELEASE` when a
#: change to ``ccdaf.core.uac`` moves the coordinates.
UAC_METHOD = "ccdaf-uac-la-1"

#: The GitHub release holding the files, and where to fetch them.
RELEASE = "fibre-atlas-2"
URL = f"https://github.com/cesare-corrado/cardiac_ccdaf/releases/download/{RELEASE}/"

#: SHA-256 of each file published in :data:`RELEASE`. A file not listed
#: here (the Labarthe sets) is never downloaded, only read locally.
FILES: Dict[str, str] = {
    "la_atlas_endo.vtp": "c438742c2bd462a8f773cb6edc26b74fdab10f26820580632cc3b0dd8a9aab60",
    "la_atlas_epi.vtp": "264536d03ef838ddb991c3e91e30cd19a1dd89b862db56b4bd9c4d5d572324e8",
    "ra_atlas_endo.vtp": "a74d93c23e1d20200d1f19b727d1ec60888fefe1495553f1d0f4729eec5f65ac",
    "ra_atlas_epi.vtp": "bd18b9e3dbf7828aa26bac49c700f5b008512032cecad59e67b3e734d7776011",
}

SETS: Tuple[str, ...] = ("labarthe", "atlas")
LAYERS: Tuple[str, ...] = ("endo", "epi")
ENV = "CCDAF_FIBRE_ATLAS"


class AtlasError(RuntimeError):
    """A fibre atlas file cannot be found, fetched or used."""


def folder() -> str:
    """Where the files are looked for: ``$CCDAF_FIBRE_ATLAS`` if set, else
    a per-user cache folder named after :data:`RELEASE`."""
    if os.environ.get(ENV):
        return os.environ[ENV]
    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    else:
        base = os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "ccdaf", "fibre_atlas", RELEASE)


def file_name(set_name: str, layer: str, chamber: str = "la") -> str:
    if set_name not in SETS or layer not in LAYERS:
        raise ValueError(f"unknown atlas {set_name!r} / {layer!r}")
    return f"{chamber}_{set_name}_{layer}.vtp"


def path(set_name: str, layer: str, chamber: str = "la") -> str:
    return os.path.join(folder(), file_name(set_name, layer, chamber))


def _sha256(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(name: str, on_progress: Optional[Callable[[int, int], None]] = None) -> str:
    """Download *name* from the release into :func:`folder` (if it is not
    there yet) and check its checksum. Returns the local path."""
    dest = os.path.join(folder(), name)
    if os.path.exists(dest):
        return dest
    if name not in FILES:
        raise AtlasError(f"{name} is not published: copy it into {folder()!r}, "
                         f"or set {ENV} to a folder that holds it")
    if os.environ.get(ENV):
        raise AtlasError(f"{name} is not in {dest!r} (set by {ENV})")
    os.makedirs(folder(), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=folder(), suffix=".part")
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(URL + name, timeout=60) as r:
            total = int(r.headers.get("Content-Length") or 0)
            got = 0
            for block in iter(lambda: r.read(1 << 20), b""):
                out.write(block)
                got += len(block)
                if on_progress:
                    on_progress(got, total)
        if _sha256(tmp) != FILES[name]:
            raise AtlasError(f"{name}: the download is damaged (checksum mismatch)")
        os.replace(tmp, dest)
    except OSError as exc:
        raise AtlasError(f"could not download {name} from {URL}: {exc}. To work "
                         f"offline, put the files in a folder and set {ENV} to it.") from None
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return dest


def load(set_name: str, layer: str, chamber: str = "la",
         on_progress: Optional[Callable[[int, int], None]] = None) -> pv.PolyData:
    """The atlas surface for *set_name* (``labarthe`` or ``atlas``) and
    *layer* (``endo`` or ``epi``), downloaded on first use."""
    p = fetch(file_name(set_name, layer, chamber), on_progress)
    m = pv.read(p)
    method = str(m.field_data["uac_method"][0]) if "uac_method" in m.field_data else "?"
    if method != UAC_METHOD:
        raise AtlasError(f"{os.path.basename(p)} was built with {method}, "
                         f"this ccdaf uses {UAC_METHOD}: rebuild it")
    return m
