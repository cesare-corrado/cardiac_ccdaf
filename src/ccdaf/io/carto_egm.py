"""
Carto electrograms
==================

Read the per-point electrograms (EGMs) of one Carto map, and the positions of
the mapping catheter's electrodes, into one plain dictionary that can be kept
in memory and pickled.

Vocabulary
----------
* **point** — one entry of ``<map>_car.txt`` / ``<map>_P<ID>_Point_Export.xml``:
  one bipole of the mapping catheter that Carto annotated (LAT, voltages).
* **placement** — one position of the catheter. All points recorded together
  share one ``<map>_ECG_Export_<StartTime>.txt`` and, per connected catheter,
  one ``<map>_<Connector>_Eleclectrode_Positions_OnAnnotation_<StartTime>.txt``
  (the *position file* of that catheter at that placement).

What is stored
--------------
Per map: the channel names (trace column order), the gain (mV per count), the
window of interest. Per placement: the raw int16 traces and the reference
annotation. Per catheter (only the mapping catheter for now): the electrode
positions at every placement, which trace column each electrode and bipole
uses, and the link from each bipole to its ``_car.txt`` row (-1 when that
bipole is not a point at that placement).

Trace columns are ``[reference, point bipoles, their unipoles, extra]``:

* **point bipoles** — every bipolar channel that some point of the map uses
  (10 on a 20-electrode PentaRay map: 1-2 and 3-4 of each spline);
* **unipoles** — the two electrodes of each point bipole;
* **extra** — any other channel the user asked for (traces only, no position).

Time
----
An ECG file has no time column: row ``i`` is ``i`` ms after the start of the
recording (1 kHz). The reference annotation is the row where the reference
channel activates, so ``t = row - reference_annotation`` puts that activation
at 0 ms and a point's activation at its LAT. Rows are counted from 0, the
convention openep uses; the Carto export does not state it.

Electrode layout of the mapping catheter
----------------------------------------
A position file lists the electrodes in blocks: optional leading rows (e.g.
two rows numbered 1, 2; probably shaft electrodes; they have no trace) and
then one block per spline, numbered 1..L from distal to proximal. The layout
(spline length L, number of splines S) is taken from the placements whose
position file is complete. Unipole ``<prefix>k`` is electrode
``(k-1) % L + 1`` of spline ``(k-1) // L``: an assumption on Carto's channel
numbering that every point of the test maps agrees with, and that the loader
checks against ``_car.txt`` (see ``warnings``).

Some position files lack whole splines. The splines that hold a point at that
placement are present; the missing ones are among the others. A block keeps
its positions only when its spline is the same for every possible choice of
missing splines; otherwise its electrodes get NaN positions. Traces are kept
in all cases.
"""
from __future__ import annotations

import itertools
import os
import re
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ccdaf.io.carto_functions import (
    MISSING_TRACE, get_exported_points, get_point_info, load_carto_electrodes,
    read_ecg_file, read_ecg_header, read_electrode_positions,
)


EGM_FORMAT_VERSION: int = 1

# Two positions closer than this are the same electrode. Carto writes the
# point position and the position-file row with the same digits, so a real
# match is exact; this only absorbs float parsing.
_SAME_POSITION_MM: float = 1e-3

ProgressFn = Optional[Callable[[int, int, str], None]]


def _ecg_name(directory: str, map_name: str, start_time: int) -> str:
    return os.path.join(directory, f"{map_name}_ECG_Export_{start_time}.txt")


def _position_name(directory: str, map_name: str, connector: str,
                   start_time: int) -> str:
    return os.path.join(
        directory,
        f"{map_name}_{connector}_Eleclectrode_Positions_OnAnnotation_{start_time}.txt")


def _split_number(name: str) -> Tuple[str, int]:
    """'20A_7' -> ('20A_', 7); 'MCC_Dx_UniPolar_12' -> ('MCC_Dx_UniPolar_', 12)."""
    m = re.match(r"^(.*?)(\d+)$", name)
    if m is None:
        raise ValueError(f"channel '{name}' does not end with an electrode number")
    return m.group(1), int(m.group(2))


def _blocks(ids: np.ndarray) -> List[Tuple[int, int]]:
    """Split the electrode-number column into blocks: a new block starts each
    time the number does not increase. Returns (start, stop) row ranges."""
    out: List[Tuple[int, int]] = []
    start = 0
    for i in range(1, len(ids) + 1):
        if i == len(ids) or ids[i] <= ids[i - 1]:
            out.append((start, i))
            start = i
    return out


@dataclass
class _Point:
    point_id: int
    start_time: int
    uni: str
    bip: str
    ref: str
    ref_annotation: int
    woi: Tuple[Optional[int], Optional[int]]
    connectors: List[str]


@dataclass
class EGMScan:
    """What :func:`scan_carto_egm` learned from the point files and ECG headers.

    Enough to offer the user the channel choice before any trace is read.
    """
    directory: str
    map_name: str
    points: List[_Point]
    start_times: List[int]                 # one per placement, ascending
    mapping_prefix: str                    # e.g. '20A_'
    bipoles: List[str]                     # point bipoles, by first electrode
    bipole_electrodes: List[Tuple[int, int]]   # electrode numbers of each bipole
    electrodes: List[int]                  # unipole numbers, ascending
    reference_channels: List[str]          # distinct reference channels
    available_channels: List[str]          # every channel in the ECG headers
    connectors: List[str] = field(default_factory=list)   # the points' connectors
    mapping_connector: Optional[str] = None  # detected; the user may override it
    car_row: Dict[int, int] = field(default_factory=dict)        # point id -> row
    car_xyz: Dict[int, np.ndarray] = field(default_factory=dict)  # point id -> xyz
    warnings: List[str] = field(default_factory=list)

    @property
    def unipoles(self) -> List[str]:
        return [f"{self.mapping_prefix}{k}" for k in self.electrodes]

    @property
    def default_channels(self) -> List[str]:
        """The channels always stored: reference(s), point bipoles, unipoles."""
        return list(self.reference_channels) + self.bipoles + self.unipoles


def scan_carto_egm(directory: str, map_name: str,
                   progress: ProgressFn = None) -> EGMScan:
    """Read the point files of *map_name* and the ECG file headers.

    Raises ``FileNotFoundError`` when the map has no point export.
    """
    pts_file = os.path.join(directory, f"{map_name}_Points_Export.xml")
    if not os.path.isfile(pts_file):
        raise FileNotFoundError(f"no point export for map '{map_name}': {pts_file}")
    warnings: List[str] = []

    point_list = get_exported_points(pts_file)
    points: List[_Point] = []
    for i, (pid, fname) in enumerate(point_list):
        if progress is not None and i % 100 == 0:
            progress(i, len(point_list), "Reading point files")
        info = get_point_info(fname, directory)
        ecg_file, uni, bip, ref = info["ECG"]
        start = info["StartTime"]
        if start is None or not ecg_file or not uni or not bip:
            warnings.append(f"point {pid}: no ECG export, skipped")
            continue
        points.append(_Point(int(pid), int(start), uni, bip, ref,
                             int(info["Annotations"][0]), tuple(info["WOI"]),
                             list(info["Connectors"])))
    if not points:
        raise ValueError(f"map '{map_name}' has no point with an ECG export")

    # Mapping catheter = the unipolar prefix most points use. A point taken
    # with another catheter (e.g. one ablation-catheter point) is not part of
    # it; its row in _car.txt is untouched, it just has no stored EGM.
    prefixes = Counter(_split_number(p.uni)[0] for p in points)
    mapping_prefix = prefixes.most_common(1)[0][0]
    other = [p for p in points if _split_number(p.uni)[0] != mapping_prefix]
    if other:
        warnings.append(f"{len(other)} point(s) recorded with another catheter "
                        f"({', '.join(sorted({p.uni for p in other}))}): no EGM stored")
    points = [p for p in points if _split_number(p.uni)[0] == mapping_prefix]

    # Point bipoles and their two electrodes. The second electrode is the
    # number after the dash when the name has one ('20A_7-8'), otherwise the
    # next number ('MCC_Dx_BiPolar_27' with unipole 'MCC_Dx_UniPolar_31' -> 32).
    first: Dict[str, int] = {}
    for p in points:
        n = _split_number(p.uni)[1]
        if first.setdefault(p.bip, n) != n:
            raise ValueError(f"bipole {p.bip} has two first electrodes "
                             f"({first[p.bip]} and {n})")
    bipoles = sorted(first, key=lambda b: first[b])
    pairs: List[Tuple[int, int]] = []
    for b in bipoles:
        n = first[b]
        m = re.search(rf"{n}-(?:{re.escape(mapping_prefix)})?(\d+)$", b)
        pairs.append((n, int(m.group(1)) if m else n + 1))
    electrodes = sorted({e for pair in pairs for e in pair})

    start_times = sorted({p.start_time for p in points})
    reference_channels = sorted({p.ref for p in points if p.ref})
    if len(reference_channels) > 1:
        warnings.append("the reference channel changes within the map: "
                        + ", ".join(reference_channels))

    available: List[str] = []
    seen = set()
    for i, st in enumerate(start_times):
        if progress is not None and i % 50 == 0:
            progress(i, len(start_times), "Reading ECG headers")
        path = _ecg_name(directory, map_name, st)
        if not os.path.isfile(path):
            continue
        for name in read_ecg_header(path)[0]:
            if name not in seen:
                seen.add(name)
                available.append(name)

    car_row, car_xyz = _read_car(directory, map_name)
    if not car_row:
        warnings.append("no _car.txt: bipoles are not linked to points")
    scan = EGMScan(directory, map_name, points, start_times, mapping_prefix,
                   bipoles, pairs, electrodes, reference_channels, available,
                   connectors=[c for c, _ in Counter(
                       c for p in points for c in p.connectors).most_common()],
                   car_row=car_row, car_xyz=car_xyz, warnings=warnings)
    scan.mapping_connector = _mapping_connector(scan) if car_xyz else None
    return scan


def _read_car(directory: str, map_name: str
              ) -> Tuple[Dict[int, int], Dict[int, np.ndarray]]:
    """Point id -> row index and point id -> xyz, from ``<map>_car.txt``."""
    car_path = os.path.join(directory, f"{map_name}_car.txt")
    row_of: Dict[int, int] = {}
    xyz_of: Dict[int, np.ndarray] = {}
    if os.path.isfile(car_path):
        data = np.asarray(load_carto_electrodes(car_path)["data"], dtype=float)
        for r, row in enumerate(data):
            row_of[int(row[0])] = r
            xyz_of[int(row[0])] = row[1:4]
    return row_of, xyz_of


def _mapping_connector(scan: EGMScan) -> Optional[str]:
    """The connector whose position files hold the mapping catheter.

    Tried on the points' own connectors: the one whose position file, at a
    point's placement, has an electrode exactly where the point is.
    """
    car_xyz = scan.car_xyz
    sample = [p for p in scan.points if p.point_id in car_xyz][:20]
    best, best_hits = None, 0
    for conn in scan.connectors:
        hits = 0
        for p in sample:
            path = _position_name(scan.directory, scan.map_name, conn, p.start_time)
            if not os.path.isfile(path):
                continue
            _, xyz = read_electrode_positions(path)
            if len(xyz) and np.min(np.linalg.norm(xyz - car_xyz[p.point_id], axis=1)) \
                    < _SAME_POSITION_MM:
                hits += 1
        if hits > best_hits:
            best, best_hits = conn, hits
    return best


def _spline_layout(files: Dict[int, Tuple[np.ndarray, np.ndarray]]
                   ) -> Optional[Tuple[int, int, int, int]]:
    """Layout of a catheter from its complete position files.

    Complete = the files with the most rows. Returns ``(L, S, n_extra,
    n_rows)``: the spline length (the most common block length), the number
    of splines (blocks of that length), the number of other blocks (the
    leading extra rows) and the row count of a complete file.
    """
    if not files:
        return None
    longest = max(len(ids) for ids, _ in files.values())
    ids = next(ids for ids, _ in files.values() if len(ids) == longest)
    lengths = [b - a for a, b in _blocks(ids)]
    L = Counter(lengths).most_common(1)[0][0]
    S = lengths.count(L)
    return L, S, len(lengths) - S, longest


def _block_labels(n_blocks: int, S: int, present: set) -> List[Optional[int]]:
    """Spline of each spline block, or None where it is not certain.

    *present*: splines known to be in the file (they hold a point).
    """
    n_missing = S - n_blocks
    if n_missing == 0:
        return list(range(S))
    free = [s for s in range(S) if s not in present]
    if n_missing < 0 or n_missing > len(free):
        return [None] * n_blocks
    options = [set() for _ in range(n_blocks)]
    for missing in itertools.combinations(free, n_missing):
        kept = [s for s in range(S) if s not in missing]
        for j, s in enumerate(kept):
            options[j].add(s)
    return [next(iter(o)) if len(o) == 1 else None for o in options]


def _catheter_positions(scan: EGMScan, connector: str, warnings: List[str]
                        ) -> np.ndarray:
    """(n_placements, n_electrodes, 3) electrode positions, NaN where unknown."""
    n_p, n_e = len(scan.start_times), len(scan.electrodes)
    pos = np.full((n_p, n_e, 3), np.nan, dtype=np.float32)
    files: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for st in scan.start_times:
        path = _position_name(scan.directory, scan.map_name, connector, st)
        if os.path.isfile(path):
            files[st] = read_electrode_positions(path)
    layout = _spline_layout(files)
    if layout is None:
        warnings.append(f"no position file for {connector}: positions are NaN")
        return pos
    L, S, n_template_extra, _ = layout
    if max(scan.electrodes) > L * S:
        warnings.append(f"{connector}: electrode {max(scan.electrodes)} does not fit "
                        f"{S} splines of {L}; positions are NaN")
        return pos

    spline_of = {k: (k - 1) // L for k in scan.electrodes}
    points_at: Dict[int, set] = defaultdict(set)
    for p in scan.points:
        points_at[p.start_time].add(spline_of[_split_number(p.uni)[1]])

    uncertain = 0
    incomplete = 0
    for ip, st in enumerate(scan.start_times):
        if st not in files:
            continue
        ids, xyz = files[st]
        blocks = _blocks(ids)
        spline_blocks = [(a, b) for a, b in blocks if b - a == L]
        labels = _block_labels(len(spline_blocks), S, points_at[st])
        # More short blocks than a complete file has: one of them may be a
        # partial spline, which looks like the extra rows. Do not guess.
        if len(blocks) - len(spline_blocks) != n_template_extra:
            labels = [None] * len(spline_blocks)
        uncertain += labels.count(None)
        incomplete += len(spline_blocks) < S
        for (a, b), s in zip(spline_blocks, labels):
            if s is None:
                continue
            for ie, k in enumerate(scan.electrodes):
                if spline_of[k] == s:
                    pos[ip, ie] = xyz[a + (k - 1) % L]
    if incomplete:
        warnings.append(f"{incomplete} placement(s) with missing splines; "
                        f"{uncertain} spline block(s) could not be identified (NaN)")
    if len(files) < len(scan.start_times):
        warnings.append(f"{len(scan.start_times) - len(files)} placement(s) without "
                        f"a {connector} position file (NaN)")
    return pos


def load_carto_egm(scan: EGMScan, extra_channels: Sequence[str] = (),
                   connector: Optional[str] = None,
                   progress: ProgressFn = None) -> dict:
    """Read the traces and the mapping catheter positions of a scanned map.

    *connector* is the mapping catheter's connector; ``None`` uses the one
    the scan detected (``scan.mapping_connector``).

    Returns the EGM dictionary described in the module docstring, with the
    traces uncompressed (``int16`` array). :func:`pack_egm` compresses it for
    the pickle.
    """
    warnings = list(scan.warnings)
    directory, map_name = scan.directory, scan.map_name
    n_p = len(scan.start_times)
    placement_of = {st: i for i, st in enumerate(scan.start_times)}

    # --- per placement: reference annotation and channel ---------------
    ref_ann = np.zeros(n_p, dtype=np.int16)
    ref_name: List[str] = [""] * n_p
    seen = [False] * n_p
    for p in scan.points:
        ip = placement_of[p.start_time]
        if not seen[ip]:
            ref_ann[ip], ref_name[ip], seen[ip] = p.ref_annotation, p.ref, True
        elif (ref_ann[ip], ref_name[ip]) != (p.ref_annotation, p.ref):
            warnings.append(f"placement {p.start_time}: points disagree on the "
                            "reference; the first point's is used")

    woi_counts = Counter(p.woi for p in scan.points)
    woi = woi_counts.most_common(1)[0][0]
    if len(woi_counts) > 1:
        warnings.append(f"the window of interest varies within the map; {woi} is stored")

    # --- channel columns -------------------------------------------------
    one_ref = len(set(ref_name)) == 1
    mapping = scan.bipoles + scan.unipoles
    extra = [c for c in extra_channels if c not in mapping and c not in ref_name]
    channel_names = [ref_name[0] if one_ref else "reference"] + mapping + extra
    n_bip, n_uni = len(scan.bipoles), len(scan.unipoles)

    # --- traces ------------------------------------------------------------
    traces = None
    gain = None
    missing = Counter()
    for ip, st in enumerate(scan.start_times):
        if progress is not None and ip % 20 == 0:
            progress(ip, n_p, "Reading electrograms")
        path = _ecg_name(directory, map_name, st)
        wanted = [ref_name[ip]] + mapping + extra
        if not os.path.isfile(path):
            warnings.append(f"placement {st}: ECG file missing")
            continue
        block, g, present = read_ecg_file(path, wanted)
        if traces is None:
            traces = np.full((n_p, len(wanted), block.shape[1]), MISSING_TRACE,
                             dtype=np.int16)
            gain = g
        if g != gain:
            raise ValueError(f"{os.path.basename(path)}: gain {g} differs from {gain}")
        if block.shape[1] != traces.shape[2]:
            raise ValueError(f"{os.path.basename(path)}: {block.shape[1]} samples, "
                             f"expected {traces.shape[2]}")
        traces[ip] = block
        for name, ok in zip(wanted, present):
            if not ok:
                missing[name] += 1
    if traces is None:
        raise ValueError(f"map '{map_name}': no ECG file could be read")
    for name, n in missing.items():
        warnings.append(f"channel {name} missing in {n} placement(s)")

    # --- mapping catheter ------------------------------------------------
    car_row, car_xyz = scan.car_row, scan.car_xyz

    bip_index = {b: i for i, b in enumerate(scan.bipoles)}
    link = np.full((n_p, n_bip), -1, dtype=np.int32)
    for p in scan.points:
        r = car_row.get(p.point_id, -1)
        if r < 0 and car_row:
            warnings.append(f"point {p.point_id} is not in _car.txt")
        link[placement_of[p.start_time], bip_index[p.bip]] = r

    if progress is not None:
        progress(n_p, n_p, "Reading catheter positions")
    connector = connector or scan.mapping_connector
    if connector is None:
        connector = "mapping"
        positions = np.full((n_p, len(scan.electrodes), 3), np.nan, dtype=np.float32)
        warnings.append("the mapping catheter's connector is not known: "
                        "positions are NaN")
    else:
        positions = _catheter_positions(scan, connector, warnings)

    # Check the electrode numbering against the points: each point must sit on
    # the electrode its unipole names (where that position is known).
    e_index = {k: i for i, k in enumerate(scan.electrodes)}
    off = 0
    for p in scan.points:
        if p.point_id in car_xyz:
            xyz = positions[placement_of[p.start_time], e_index[_split_number(p.uni)[1]]]
            if np.all(np.isfinite(xyz)) and \
                    np.linalg.norm(xyz - car_xyz[p.point_id]) > _SAME_POSITION_MM:
                off += 1
    if off:
        warnings.append(f"{off} point(s) are not on the electrode their channel names: "
                        "the electrode numbering assumption does not hold here")

    catheter = {
        "electrode_channels": scan.unipoles,
        "positions": positions,
        "electrode_column": np.arange(1 + n_bip, 1 + n_bip + n_uni, dtype=np.int32),
        "bipole_electrodes": np.array([[e_index[a], e_index[b]]
                                       for a, b in scan.bipole_electrodes],
                                      dtype=np.int32).reshape(-1, 2),
        "bipole_column": np.arange(1, 1 + n_bip, dtype=np.int32),
        "point_link": link,
    }
    egm = {
        "version": EGM_FORMAT_VERSION,
        "map_name": map_name,
        "gain": float(gain),
        "woi": woi,
        "channel_names": channel_names,
        "missing_value": int(MISSING_TRACE),
        "traces": traces,
        "reference_annotation": ref_ann,
        "catheters": {connector: catheter},
        "warnings": warnings,
    }
    if not one_ref:
        egm["reference_channels"] = ref_name
    return egm


def pack_egm(egm: dict) -> dict:
    """Copy of *egm* with the traces zlib-compressed, for the pickle."""
    out = dict(egm)
    traces = np.ascontiguousarray(egm["traces"], dtype=np.int16)
    out["traces"] = {"zlib": zlib.compress(traces.tobytes(), 6),
                     "shape": tuple(traces.shape), "dtype": "int16"}
    return out


def unpack_egm(egm: dict) -> dict:
    """Inverse of :func:`pack_egm`; an unpacked dict is returned unchanged."""
    t = egm.get("traces")
    if not isinstance(t, dict):
        return egm
    out = dict(egm)
    out["traces"] = np.frombuffer(zlib.decompress(t["zlib"]), dtype=t["dtype"]
                                  ).reshape(t["shape"]).copy()
    return out


def egm_positions(egm: Optional[dict]) -> np.ndarray:
    """Every known (finite) catheter electrode position, as one (n, 3) array.

    The order is fixed (catheters in key order, then placement, electrode),
    so :func:`set_egm_positions` can write moved positions back. Used to carry
    the catheters along when the surface is smoothed or mirrored, with the
    same rule as the ``_car.txt`` electrodes.
    """
    if not egm:
        return np.empty((0, 3))
    parts = []
    for cat in egm.get("catheters", {}).values():
        pos = np.asarray(cat["positions"], dtype=float).reshape(-1, 3)
        parts.append(pos[np.isfinite(pos).all(axis=1)])
    return np.concatenate(parts) if parts else np.empty((0, 3))


def set_egm_positions(egm: Optional[dict], points: np.ndarray) -> None:
    """Write positions from :func:`egm_positions` back, in place. NaN stays NaN."""
    if not egm:
        return
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    i = 0
    for cat in egm.get("catheters", {}).values():
        pos = np.asarray(cat["positions"])
        flat = pos.reshape(-1, 3)
        ok = np.isfinite(flat).all(axis=1)
        n = int(ok.sum())
        if i + n > len(points):
            raise ValueError("fewer positions than the catheters hold")
        flat[ok] = points[i:i + n]
        cat["positions"] = flat.reshape(pos.shape).astype(pos.dtype, copy=False)
        i += n
    if i != len(points):
        raise ValueError("more positions than the catheters hold")


def mapping_catheter(egm: Optional[dict]) -> Optional[dict]:
    """The mapping catheter's entry of ``egm['catheters']`` (the first one)."""
    if not egm or not egm.get("catheters"):
        return None
    return next(iter(egm["catheters"].values()))


def placement_bipoles(egm: dict, placement: int
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The bipoles of one placement: ``(first, second, has_point)``.

    *first* / *second* are (n_bipoles, 3) positions of each bipole's two
    electrodes (NaN where unknown); *has_point* is True where the bipole is a
    ``_car.txt`` point at this placement.
    """
    cat = mapping_catheter(egm)
    pos = np.asarray(cat["positions"][placement], dtype=float)
    pairs = np.asarray(cat["bipole_electrodes"], dtype=int).reshape(-1, 2)
    return pos[pairs[:, 0]], pos[pairs[:, 1]], cat["point_link"][placement] >= 0


def nearest_bipole(egm: dict, xyz: Sequence[float],
                   car_points: Optional[np.ndarray] = None,
                   placement: Optional[int] = None) -> Optional[Tuple[int, int]]:
    """``(placement, bipole)`` nearest to *xyz* among the pickable ones.

    Pickable: every bipole that is a ``_car.txt`` point (at its row of
    *car_points*, the electrode positions as displayed; without them, at its
    catheter position), plus every bipole of *placement*, the one on show.
    """
    cat = mapping_catheter(egm)
    if cat is None:
        return None
    link = np.asarray(cat["point_link"])
    ip, ib = np.nonzero(link >= 0)
    if car_points is not None and len(car_points):
        rows = link[ip, ib]
        ok = rows < len(car_points)
        ip, ib, cand = ip[ok], ib[ok], np.asarray(car_points, dtype=float)[rows[ok]]
    else:
        e1 = np.asarray(cat["bipole_electrodes"])[ib, 0]
        cand = np.asarray(cat["positions"], dtype=float)[ip, e1]
    keys = list(zip(ip.tolist(), ib.tolist()))
    if placement is not None:
        first, _, _ = placement_bipoles(egm, placement)
        keys += [(placement, b) for b in range(len(first))]
        cand = np.vstack([cand.reshape(-1, 3), first])
    ok = np.isfinite(cand).all(axis=1)
    if not ok.any():
        return None
    d = np.linalg.norm(cand - np.asarray(xyz, dtype=float), axis=1)
    d[~ok] = np.inf
    return keys[int(np.argmin(d))]
