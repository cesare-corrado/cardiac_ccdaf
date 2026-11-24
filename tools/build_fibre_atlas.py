"""Build the atrial fibre atlas files used by ccdaf (left and right atrium).

usage: python tools/build_fibre_atlas.py OUT_DIR [--zenodo DIR] [--uac-codes DIR] [NAME ...]

Two sources, each a labelled surface in micrometres (``*.pts``, ``*.elem``
with the region tag of each triangle), one fibre per triangle (``*.lon``)
and the source's own UAC of every point:

* ``--zenodo``: the published fibre atlas, Zenodo record 3764917 (CC BY 4.0),
  folders ``LA_Endo/A``, ``LA_Epi/A``, ``RA_Endo/A`` and ``RA_Epi/A`` (the
  average fibre field, shown on anatomy 6); its UAC are the ``alpha``/``beta``
  point fields of the ``.vtk``. Builds ``la_atlas_endo``, ``la_atlas_epi``,
  ``ra_atlas_endo`` and ``ra_atlas_epi``. Right-atrial tags there: body 1,
  RAA 2, CS 5, SVC 6, IVC 7.
* ``--uac-codes``: the ``fibre_files`` folder of the UAC codes, for the
  Labarthe sets ``la/{endo,epi}/l`` and Bachmann's bundle ``bb/la``; its UAC
  are ``Labelled_Coords_2D_Rescaling_v3_C.pts``. Builds ``la_labarthe_endo``
  and ``la_labarthe_epi``. These are not published (their source is not
  public); ccdaf reads them from a local folder only.

For each set the script converts to millimetres and ccdaf labels, computes
ccdaf's own UAC (``ccdaf.core.uac``, default options) and writes one
compressed ``.vtp`` to OUT_DIR:

  point data  uac_alpha, uac_beta          ccdaf's coordinates
              uac_v2_alpha, uac_v2_beta    the source's own coordinates
  cell data   elemTag                      ccdaf labels
              fibre                        unit fibre direction
              fibre_bb, bb                 Bachmann's bundle (epi Labarthe
                                           only): its fibre, and 1 where it
                                           applies
  field data  uac_method, set, layer, source

The files are not part of the package: publish them as assets of a GitHub
release and put the release name and each file's SHA-256 (printed at the
end) into ``ccdaf.core.uac.fibre_atlas`` (RELEASE, FILES). Rerun it whenever
the UAC method changes, with a new fibre_atlas.UAC_METHOD and a new release.
"""
import os
import sys
import time

import numpy as np
import pyvista as pv
import vtk

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from ccdaf.core import uac  # noqa: E402
from ccdaf.core.uac import fibre_atlas  # noqa: E402

V2_TAGS = {11: 1, 13: 19, 21: 11, 23: 13, 25: 15, 27: 17}   # LA atlas tag -> ccdaf
RA_TAGS = {1: 2, 2: 27, 5: 25, 6: 21, 7: 23}                 # RA atlas tag -> ccdaf
SETS = {   # file: (source, folder, layer, set name)
    "la_atlas_endo": ("zenodo", "LA_Endo/A", "endo", "atlas"),
    "la_atlas_epi": ("zenodo", "LA_Epi/A", "epi", "atlas"),
    "la_labarthe_endo": ("uac_codes", "la/endo/l", "endo", "labarthe"),
    "la_labarthe_epi": ("uac_codes", "la/epi/l", "epi", "labarthe"),
    "ra_atlas_endo": ("zenodo", "RA_Endo/A", "endo", "atlas"),
    "ra_atlas_epi": ("zenodo", "RA_Epi/A", "epi", "atlas"),
}
SOURCE = {
    "zenodo": "Fibre atlas, Roney et al. 2021, Zenodo 3764917 (CC BY 4.0)",
    "uac_codes": "UAC codes, fibre_files",
}


def _one(d, ext):
    hits = sorted(f for f in os.listdir(d) if f.endswith(ext))
    if len(hits) != 1:
        raise RuntimeError(f"expected one *{ext} in {d}, found {hits}")
    return os.path.join(d, hits[0])


def _read(d):
    P = np.loadtxt(_one(d, ".pts"), skiprows=1) / 1000.0
    E = np.loadtxt(_one(d, ".elem"), skiprows=1, usecols=(1, 2, 3, 4)).astype(int)
    return P, E[:, :3], E[:, 3]


def _source_uac(kind, d, n):
    """The source's own coordinates of each of the *n* points."""
    if kind == "zenodo":
        v = pv.read(_one(d, ".vtk"))
        uv = np.c_[v.point_data["alpha"], v.point_data["beta"]]
    else:
        uv = np.loadtxt(os.path.join(d, "Labelled_Coords_2D_Rescaling_v3_C.pts"), skiprows=1)
    if len(uv) != n:
        raise RuntimeError(f"{d}: {len(uv)} coordinates for {n} points")
    return np.asarray(uv, float)[:, :2]


def _left_veins_by_the_laa(m, tag, name):
    """The LAA lies next to the left veins. The Labarthe files label the
    two pairs the other way round (their own coordinates place them
    correctly), so a pair whose superior vein is nearer the LAA as "right"
    is swapped."""
    c = m.cell_centers().points
    laa = c[tag == 19].mean(0)
    d = {L: np.linalg.norm(c[tag == L].mean(0) - laa) for L in (11, 15)}
    if d[15] < d[11]:
        print(f"{name}: left and right vein labels swapped (LAA {d[15]:.0f} mm from "
              f"the 'RSPV', {d[11]:.0f} mm from the 'LSPV')", flush=True)
        tag = tag.copy()
        for a, b in ((11, 15), (13, 17)):
            ia, ib = tag == a, tag == b
            tag[ia], tag[ib] = b, a
    return tag.astype(np.int32)


def _unit(v):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return np.where(n > 0, v / np.where(n > 0, n, 1), np.nan)


def build(roots, name):
    kind, folder, layer, set_name = SETS[name]
    if not roots.get(kind):
        raise RuntimeError(f"{name} needs --{kind.replace('_', '-')}")
    d = os.path.join(roots[kind], folder)
    P, T, tag = _read(d)
    m = pv.PolyData(P, np.c_[np.full(len(T), 3), T].ravel())
    if name.startswith("ra_"):
        m.cell_data["elemTag"] = np.vectorize(RA_TAGS.get)(tag).astype(np.int32)
    else:
        tag = np.where(tag % 2 == 0, tag - 1, tag)            # some epi tags are endo + 1
        m.cell_data["elemTag"] = _left_veins_by_the_laa(m, np.vectorize(V2_TAGS.get)(tag), name)
    uv = _source_uac(kind, d, len(P))
    m.point_data["uac_v2_alpha"], m.point_data["uac_v2_beta"] = uv[:, 0], uv[:, 1]
    m.cell_data["fibre"] = _unit(np.loadtxt(_one(d, ".lon"))[-len(T):, :3])
    if name == "la_labarthe_epi":
        b = os.path.join(roots[kind], "bb", "la")
        Pb, Tb, tb = _read(b)
        if not (np.allclose(Pb, P) and (Tb == T).all()):
            raise RuntimeError("bb/la is not on the epi Labarthe mesh")
        fb = _unit(np.loadtxt(_one(b, ".lon"))[-len(T):, :3])
        on = tb == 10
        fb[~on] = np.nan
        m.cell_data["fibre_bb"] = fb
        m.cell_data["bb"] = on.astype(np.uint8)
    t0 = time.time()
    run = uac.compute(*uac.inputs(m))
    out = uac.write(m, run)                                   # repairs carry every field
    a = run.attempt
    if a.folded_real or a.worst_miss > 0.01:
        raise RuntimeError(f"{name}: {a.folded_real} folds, worst ring {a.worst_miss:.4f}")
    # store compactly: single precision, small integers
    out.points = np.asarray(out.points, np.float32)
    for k in list(out.point_data.keys()):
        out.point_data[k] = np.asarray(out.point_data[k], np.float32)
    for k in ("fibre", "fibre_bb"):
        if k in out.cell_data:
            out.cell_data[k] = np.asarray(out.cell_data[k], np.float32)
    out.cell_data["elemTag"] = np.asarray(out.cell_data["elemTag"], np.int8)
    out.field_data["uac_method"] = [fibre_atlas.UAC_METHOD]
    out.field_data["set"] = [set_name]
    out.field_data["layer"] = [layer]
    out.field_data["source"] = [f"{SOURCE[kind]}, {folder}"]
    print(f"{name}: {out.n_points} points, {out.n_cells} triangles, "
          f"{time.time() - t0:.0f} s, worst ring {a.worst_miss:.4f}", flush=True)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out_dir")
    ap.add_argument("names", nargs="*", help=f"sets to build (default: those with a source): {list(SETS)}")
    ap.add_argument("--zenodo", help="the downloaded Zenodo 3764917 folder (LA_Endo, LA_Epi, ...)")
    ap.add_argument("--uac-codes", help="the fibre_files folder of the UAC codes")
    a = ap.parse_args()
    roots = {"zenodo": a.zenodo, "uac_codes": a.uac_codes}
    names = a.names or [n for n, v in SETS.items() if roots.get(v[0])]
    os.makedirs(a.out_dir, exist_ok=True)
    for name in names:
        p = os.path.join(a.out_dir, name + ".vtp")
        save(build(roots, name), p)
        print(f'    "{name}.vtp": "{fibre_atlas._sha256(p)}",', flush=True)


def save(mesh, path):
    """XML, LZMA-compressed, 32-bit ids (a quarter smaller than 64-bit)."""
    w = vtk.vtkXMLPolyDataWriter()
    w.SetInputData(mesh)
    w.SetIdTypeToInt32()
    w.SetCompressorTypeToLZMA()
    w.SetDataModeToAppended()
    w.EncodeAppendedDataOff()
    w.SetFileName(path)
    if not w.Write():
        raise RuntimeError(f"could not write {path}")


if __name__ == "__main__":
    main()
