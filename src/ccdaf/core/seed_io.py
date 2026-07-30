"""
seed_io
=======
Persist a seed selection: names and coordinates, nothing mesh-indexed.

Coordinates survive what a session does to a mesh — clipping, refinement,
repair — where vertex ids do not. A saved seed therefore carries no id;
on load it is re-snapped to the current surface by nearest point, and the
id is recovered rather than trusted.

Two formats, chosen by extension:

* ``.json`` — ``{"seeds": {name: [x, y, z], ...}}``. A human-readable
  sidecar carrying the seeds alone.
* ``.pkl`` — the surface as ``polydata_to_carto_dict`` plus the same
  ``"seeds"`` key, following the EAM export pickle's convention
  (``{'surface': ..., 'electrodes': ...}``), so the bundle is
  self-contained. Loading reads only ``"seeds"``: any pickle carrying
  that key loads, whatever else it holds.

A plain ``.vtk`` cannot carry seeds — the downstream pipeline reads vtk
and does not need them, so that loss is accepted rather than worked
around.
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Dict, Mapping, Sequence, Union

import numpy as np

from ccdaf.core.eam_export import polydata_to_carto_dict

SEEDS_KEY = "seeds"
_JSON_SUFFIXES = {".json"}
_PICKLE_SUFFIXES = {".pkl", ".pickle"}


def _as_plain(seeds: Mapping[str, Sequence[float]]) -> Dict[str, list]:
    out: Dict[str, list] = {}
    for name, xyz in seeds.items():
        arr = np.asarray(xyz, dtype=float).reshape(-1)
        if arr.shape != (3,) or not np.isfinite(arr).all():
            raise ValueError(f"seed '{name}' is not a finite 3-vector: {xyz}")
        out[str(name)] = [float(v) for v in arr]
    return out


def save_seeds(path: Union[str, Path],
               seeds: Mapping[str, Sequence[float]],
               mesh=None,
               key: str = SEEDS_KEY) -> None:
    """Write *seeds* (name → xyz) to *path*; format follows the suffix.

    ``.pkl`` embeds *mesh* as its ``"surface"`` — required there, unused
    for ``.json``. *key* is the dict/JSON key the set is stored under
    (``"seeds"`` by default; the LA-UAC landmark set uses
    ``"landmarks_LA_UAC"``), so different point sets round-trip through the
    same two formats without colliding.
    """
    path = Path(path)
    plain = _as_plain(seeds)
    if not plain:
        raise ValueError("no seeds to save")
    suffix = path.suffix.lower()
    if suffix in _JSON_SUFFIXES:
        path.write_text(json.dumps({key: plain}, indent=2) + "\n")
    elif suffix in _PICKLE_SUFFIXES:
        if mesh is None:
            raise ValueError("the pickle format embeds the surface — "
                             "a mesh is required")
        payload = {"surface": polydata_to_carto_dict(mesh), key: plain}
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)
    else:
        raise ValueError(f"unknown seed-file suffix '{path.suffix}' "
                         "(use .json or .pkl)")


def load_seeds(path: Union[str, Path],
               key: str = SEEDS_KEY) -> Dict[str, np.ndarray]:
    """Read a seed file back as ``{name: xyz array}``.

    Reads only the *key* mapping (``"seeds"`` by default), whichever
    format carries it.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in _JSON_SUFFIXES:
        data = json.loads(path.read_text())
    elif suffix in _PICKLE_SUFFIXES:
        with open(path, "rb") as fh:
            data = pickle.load(fh)
    else:
        raise ValueError(f"unknown seed-file suffix '{path.suffix}' "
                         "(use .json or .pkl)")
    if not isinstance(data, dict) or key not in data:
        raise ValueError(f"{path.name} carries no '{key}' key")
    seeds = data[key]
    if not isinstance(seeds, MappingABC):
        raise ValueError(f"'{key}' is not a name → xyz mapping")
    out: Dict[str, np.ndarray] = {}
    for name, xyz in seeds.items():
        arr = np.asarray(xyz, dtype=float).reshape(-1)
        if arr.shape != (3,) or not np.isfinite(arr).all():
            raise ValueError(f"seed '{name}' is not a finite 3-vector: {xyz}")
        out[str(name)] = arr
    return out


__all__ = ["save_seeds", "load_seeds", "SEEDS_KEY"]
