"""
EAM export
==========

Write a loaded (and possibly repaired / smoothed) EAM mapping out again:

* ``EXPORT_BINARY`` — a pickled ``{'surface': ..., 'electrodes': ...}``
  dictionary, in the shape the reference Carto pipeline dumps, so its
  downstream steps can read ours.
* ``EXPORT_VTK`` — the surface with every field, for ParaView and the like.

Both reflect the mesh as it currently stands, not as it was read: repairs,
smoothing and the electrode displacement that follows it are all included.

Two deliberate differences from what the reference Carto pipeline writes,
recorded here because they look like omissions and are not:

* **No-data is NaN**, where a raw Carto export carries ``+/-10000``. The
  sentinel is masked at load. Nothing downstream is worse off: the pickled
  electrodes — the only part read back — keep the raw sentinel, so the
  ``LAT > -1000`` artifact filter still works, and the scripts that consume
  the vtk already use NaN as their own no-data marker. A magic number that
  interpolates to a plausible wrong answer is the worse of the two.
* **LAT is not zero-based.** ``generate_polydata`` subtracts the smallest
  valid LAT so its exported field starts at 0; ours stays in Carto's own
  frame. That shift has no numerical consumer — the Gaussian-process step
  is fed the *unshifted* electrode LAT from the pickle, and ``gpmi``
  centres and standardises its data anyway, so a constant offset is a
  no-op to it. Zero-basing would only make the file disagree with what the
  app shows, and lose the absolute reference. ``LAT - np.nanmin(LAT)``
  restores it if a downstream reader ever wants it.
"""
from __future__ import annotations

import pickle
from typing import Optional

import numpy as np
import pyvista as pv


EXPORT_BINARY = "binary"
EXPORT_VTK = "vtk"

# Suffix each format is written with.
EXPORT_SUFFIX = {
    EXPORT_BINARY: ".pkl",
    EXPORT_VTK: ".vtk",
}


def polydata_to_carto_dict(mesh: pv.PolyData) -> dict:
    """Rebuild the reader's mesh dictionary from a surface.

    Inverse of :func:`eam_loader.carto_mesh_to_polydata`: point fields go
    back into a ``VertexColors`` matrix, named by ``ColorsNames`` and indexed
    by ``ColorsIDs``, so the result has the same shape as
    ``read_carto_mesh_file`` returns.
    """
    X = np.asarray(mesh.points, dtype=float)
    faces = np.asarray(mesh.faces).reshape(-1, 4)
    if faces.size and np.any(faces[:, 0] != 3):
        raise ValueError("mesh must contain triangles only")
    tri = faces[:, 1:].astype(int) if faces.size else np.zeros((0, 3), dtype=int)

    # One column per name is the format's whole premise, so a multi-component
    # field cannot go in: it would contribute three columns under one name and
    # silently shift every later field's ColorsID onto the wrong column. Only
    # scalars are Carto "colours" anyway — a surface rebuilt from a
    # segmentation arrives carrying marching cubes' own point Normals, which
    # are derived geometry, not a measured quantity.
    names = [n for n in mesh.point_data.keys()
             if np.asarray(mesh.point_data[n]).ndim == 1]
    if names:
        colors = np.column_stack(
            [np.asarray(mesh.point_data[n], dtype=float) for n in names])
    else:
        colors = np.zeros((X.shape[0], 0), dtype=float)

    return {
        "X": X,
        "Tri": tri,
        "VertexColors": colors,
        "ColorsIDs": np.arange(len(names), dtype=int),
        "ColorsNames": np.array(names),
    }


def electrodes_at(electrodes: Optional[dict],
                  points: Optional[np.ndarray]) -> Optional[dict]:
    """Copy the electrode record with its coordinates set to ``points``.

    The stored record keeps Carto's own columns; only x/y/z are rewritten,
    so a displaced electrode still carries its voltages, LAT and catheter id.
    Mismatched lengths leave the record untouched — better to export the
    original coordinates than to pair measurements with the wrong points.
    """
    if electrodes is None:
        return None
    out = dict(electrodes)
    data = np.array(electrodes.get("data"), dtype=float, copy=True)
    pts = None if points is None else np.asarray(points, dtype=float)
    if (pts is not None and data.ndim == 2 and data.shape[1] >= 4
            and data.shape[0] == pts.shape[0]):
        data[:, 1:4] = pts
    out["data"] = data
    return out


def export_binary(path: str, mesh: pv.PolyData,
                  electrodes: Optional[dict] = None,
                  electrode_points: Optional[np.ndarray] = None) -> None:
    """Pickle ``{'surface': <reader dict>, 'electrodes': <record>}``."""
    payload = {
        "surface": polydata_to_carto_dict(mesh),
        "electrodes": electrodes_at(electrodes, electrode_points),
    }
    with open(path, "wb") as fh:
        pickle.dump(payload, fh)


def export_vtk(path: str, mesh: pv.PolyData, binary: bool = True) -> None:
    """Write the surface with all of its fields, for ParaView.

    Unlike :meth:`MeshLoader.save` — which strips to the project format —
    nothing is dropped. Electrodes are not part of this file; they are
    separate geometry.

    ``binary`` is the default because this file carries every field and a
    text copy of it is large; pass ``False`` where a reader needs ASCII.
    """
    mesh.save(str(path), binary=binary)


__all__ = [
    "EXPORT_BINARY",
    "EXPORT_VTK",
    "EXPORT_SUFFIX",
    "polydata_to_carto_dict",
    "electrodes_at",
    "export_binary",
    "export_vtk",
]
