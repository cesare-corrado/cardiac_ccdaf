"""
test_carto_egm.py
=================
Tests for the Carto point / ECG / position readers (``io.carto_functions``)
and the per-map EGM assembly (``io.carto_egm``).

The synthetic map mimics a real export with a 3-spline, 4-electrode-per-spline
catheter (prefix ``20A_``), two leading extra position rows, and three
placements:

* ``1000`` — complete position file; points on spline 0 and spline 2;
* ``2000`` — spline 0 missing; points on splines 1 and 2, so the missing one
  must be spline 0 and every remaining block is known;
* ``3000`` — one spline missing; a point on spline 1 only, so the missing one
  is spline 0 or 2 and no block can be placed (NaN positions).

Every trace sample equals ``100 * placement + column`` so the tests can see
which file and channel each stored trace came from.

No display, no Qt.
"""

import pickle
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pytest

from ccdaf.io.carto_egm import (
    _block_labels, _blocks, load_carto_egm, nearest_bipole, pack_egm,
    placement_anchors, scan_carto_egm, unpack_egm,
)
from ccdaf.io.carto_functions import (
    get_exported_points, get_point_info, read_ecg_file,
    read_ecg_header, read_electrode_positions,
)


MAP = "1-Test MAP"
CONN = "MAGNETIC_20_POLE_A_CONNECTOR"
L, S = 4, 3
N_SAMPLES = 50


def electrode_xyz(placement: int, k: int) -> list:
    """A distinct, exactly representable position for electrode k."""
    return [float(placement), float(k), 0.5 * k]


def write_point_xml(folder: Path, pid: int, start: int, uni_k: int,
                    ref_ann: int = 2000, cycle: bool = False,
                    bip: Optional[str] = None):
    bip = bip or f"20A_{uni_k}-{uni_k + 1}"
    ref_annotations = ('    <ReferenceAnnotations Beat0="2000" Beat-1="1679" '
                       'Beat-2="1347" CycleLength="321" />\n') if cycle else ""
    (folder / f"{MAP}_P{pid}_Point_Export.xml").write_text(
        f'<Point ID="{pid}" Date="01/01/26" Time="10:00:00">\n'
        f'    <Annotations StartTime="{start}" Reference_Annotation="{ref_ann}" '
        f'Map_Annotation="{ref_ann + 5}" />\n'
        '    <WOI From="-151" To="89" />\n'
        '    <Voltages Unipolar="0.936" Bipolar="0.096" />\n'
        '    <Positions>\n'
        f'        <Connector {CONN}="{MAP}_{CONN}_Eleclectrode_Positions_{start}.txt" />\n'
        f'        <Connector CS_CONNECTOR="{MAP}_CS_CONNECTOR_Eleclectrode_Positions_{start}.txt" />\n'
        '    </Positions>\n'
        f'    <Impedances Rate="100" Number="0" StartTime="{start}" />\n'
        f'{ref_annotations}'
        f'    <ECG FileName="{MAP}_ECG_Export_{start}.txt" UnipolarMappingChannel="20A_{uni_k}" '
        f'BipolarMappingChannel="{bip}" ReferenceChannel="CS1-CS2" />\n'
        '</Point>\n')


def write_ecg(folder: Path, start: int, placement: int, with_v1: bool):
    names = ["CS1-CS2"] + [f"20A_{k}" for k in range(1, L * S + 1)] \
        + [f"20A_{k}-{k + 1}" for k in range(1, L * S, 2)]
    if with_v1:
        names.append("V1")
    header = " ".join(f"{n}({i + 1})" for i, n in enumerate(names))
    rows = [" ".join(str(100 * placement + c) for c in range(len(names)))] * N_SAMPLES
    (folder / f"{MAP}_ECG_Export_{start}.txt").write_text(
        "ECG_Export_4.1\nRaw ECG to MV (gain) = 0.003000\n" + header + "\n"
        + "\n".join(rows) + "\n")
    return names


def write_positions(folder: Path, start: int, placement: int, splines: list):
    lines = ["Eleclectrode_Positions_2.0", "Electrode#\tTime\tX\tY\tZ",
             "1\t1995\t-1\t-1\t-1\t", "2\t1995\t-2\t-2\t-2\t"]       # extra rows
    for s in splines:
        for e in range(1, L + 1):
            k = s * L + e
            x, y, z = electrode_xyz(placement, k)
            lines.append(f"{e}\t1995\t{x}\t{y}\t{z}\t")
    (folder / f"{MAP}_{CONN}_Eleclectrode_Positions_OnAnnotation_{start}.txt"
     ).write_text("\n".join(lines) + "\n")


# (point id, start time, placement index, unipole number)
POINTS = [(1, 1000, 0, 1), (2, 1000, 0, 11),
          (3, 2000, 1, 5), (4, 2000, 1, 11),
          (5, 3000, 2, 7)]


@pytest.fixture
def carto_map(tmp_path):
    folder = tmp_path
    (folder / f"{MAP}_Points_Export.xml").write_text(
        f'<Points Map_Name="{MAP}">\n'
        + "".join(f'    <Point ID="{p[0]}" File_Name="{MAP}_P{p[0]}_Point_Export.xml" />\n'
                  for p in POINTS)
        + "</Points>\n")
    for pid, start, placement, k in POINTS:
        write_point_xml(folder, pid, start, k, cycle=(pid == 1))
    for placement, start in enumerate((1000, 2000, 3000)):
        write_ecg(folder, start, placement, with_v1=(start != 3000))
    write_positions(folder, 1000, 0, [0, 1, 2])
    write_positions(folder, 2000, 1, [1, 2])
    write_positions(folder, 3000, 2, [0, 1])          # spline 2 missing; undecidable
    # _car.txt: column C = point id, E..G = xyz (the bipole's first electrode).
    rows = ["VERSION_6_0 1-Test_MAP"]
    for pid, start, placement, k in POINTS:
        x, y, z = electrode_xyz(placement, k)
        cols = ["P", "0", str(pid), "0", str(x), str(y), str(z), "0", "0", "0",
                "0.9", "0.1", "5"] + ["0"] * 7
        rows.append("\t".join(cols))
    (folder / f"{MAP}_car.txt").write_text("\n".join(rows) + "\n")
    return folder


# --- readers ------------------------------------------------------------

def test_point_info_reads_rate_and_start_time(carto_map):
    info = get_point_info(f"{MAP}_P1_Point_Export.xml", str(carto_map))
    assert info["Impedances"] == [100, 0]
    assert info["StartTime"] == 1000
    assert info["Annotations"] == [2000, 2005]
    assert info["CycleLength"] == 321
    assert info["ECG"] == [f"{MAP}_ECG_Export_1000.txt", "20A_1", "20A_1-2", "CS1-CS2"]
    assert info["Connectors"] == sorted([CONN, "CS_CONNECTOR"])


def test_point_info_without_reference_annotations(carto_map):
    assert get_point_info(f"{MAP}_P2_Point_Export.xml", str(carto_map))["CycleLength"] is None


def test_exported_points(carto_map):
    pts = get_exported_points(str(carto_map / f"{MAP}_Points_Export.xml"))
    assert pts[0] == ["1", f"{MAP}_P1_Point_Export.xml"] and len(pts) == len(POINTS)


def test_ecg_channels_match_exactly(carto_map):
    path = str(carto_map / f"{MAP}_ECG_Export_1000.txt")
    names, gain = read_ecg_header(path)
    assert names[0] == "CS1-CS2" and gain == pytest.approx(0.003)
    # '20A_1' must not pick up '20A_10', '20A_11' or '20A_1-2'.
    traces, g, present = read_ecg_file(path, ["20A_1", "20A_1-2", "20A_10", "NOPE"])
    assert traces.dtype == np.int16 and traces.shape == (4, N_SAMPLES)
    assert traces[:, 0].tolist() == [names.index("20A_1"), names.index("20A_1-2"),
                                     names.index("20A_10"), 0]
    assert present.tolist() == [True, True, True, False]


def test_ecg_keeps_saturated_samples(tmp_path):
    # The recorder clips at the int16 limits: both are real samples.
    f = tmp_path / "x_ECG_Export_1.txt"
    f.write_text("ECG_Export_4.1\nRaw ECG to MV (gain) = 0.003\nA(1) B(2)\n"
                 "-32768 32767\n5 -32768\n")
    traces, _, present = read_ecg_file(str(f), ["A", "B"])
    assert traces.tolist() == [[-32768, 5], [32767, -32768]]
    assert present.tolist() == [True, True]


def test_ecg_rejects_values_outside_int16(tmp_path):
    f = tmp_path / "x_ECG_Export_1.txt"
    f.write_text("ECG_Export_4.1\nRaw ECG to MV (gain) = 0.003\nA(1) B(2)\n1 40000\n")
    with pytest.raises(ValueError):
        read_ecg_file(str(f), ["A"])


def test_read_electrode_positions(carto_map):
    ids, xyz = read_electrode_positions(
        str(carto_map / f"{MAP}_{CONN}_Eleclectrode_Positions_OnAnnotation_1000.txt"))
    assert ids.tolist()[:6] == [1, 2, 1, 2, 3, 4] and xyz.shape == (2 + L * S, 3)


# --- block logic --------------------------------------------------------

def test_blocks_restart_on_electrode_number():
    assert _blocks(np.array([1, 2, 1, 2, 3, 4, 1, 2, 3, 4])) == [(0, 2), (2, 6), (6, 10)]


@pytest.mark.parametrize("n_blocks, present, expected", [
    (5, set(), [0, 1, 2, 3, 4]),                 # complete
    (4, {1, 2, 3, 4}, [1, 2, 3, 4]),             # missing spline is 0
    (4, {1, 3}, [None, None, None, None]),       # 0, 2 or 4 missing: nothing certain
    (4, {0, 1, 2}, [0, 1, 2, None]),             # 3 or 4 missing: last block unsure
    (3, {2, 3, 4}, [2, 3, 4]),                   # two missing: 0 and 1
    (2, {0, 1, 2}, [None, None]),                # more present than blocks: inconsistent
])
def test_block_labels(n_blocks, present, expected):
    assert _block_labels(n_blocks, 5, present) == expected


# --- whole map ------------------------------------------------------------

def test_scan_finds_mapping_catheter(carto_map):
    scan = scan_carto_egm(str(carto_map), MAP)
    assert scan.mapping_prefix == "20A_"
    assert scan.start_times == [1000, 2000, 3000]
    assert scan.bipoles == ["20A_1-2", "20A_5-6", "20A_7-8", "20A_11-12"]
    assert scan.unipoles == ["20A_1", "20A_2", "20A_5", "20A_6", "20A_7", "20A_8",
                             "20A_11", "20A_12"]
    assert scan.reference_channels == ["CS1-CS2"]
    assert "V1" in scan.available_channels


def test_load_traces_columns_and_missing_extra(carto_map):
    scan = scan_carto_egm(str(carto_map), MAP)
    egm = load_carto_egm(scan, extra_channels=["V1"])
    names = egm["channel_names"]
    assert names == ["CS1-CS2"] + scan.bipoles + scan.unipoles + ["V1"]
    assert egm["traces"].shape == (3, len(names), N_SAMPLES)
    ecg_order = ["CS1-CS2"] + [f"20A_{k}" for k in range(1, L * S + 1)] \
        + [f"20A_{k}-{k + 1}" for k in range(1, L * S, 2)] + ["V1"]
    for ip in range(3):
        for col, name in enumerate(names):
            got = int(egm["traces"][ip, col, 0])
            missing = name == "V1" and ip == 2                # not in that file
            assert bool(egm["channel_missing"][ip, col]) == missing
            assert got == (0 if missing else 100 * ip + ecg_order.index(name))
    assert any("V1 missing in 1" in w for w in egm["warnings"])
    assert egm["gain"] == pytest.approx(0.003) and egm["woi"] == (-151, 89)
    assert egm["reference_annotation"].tolist() == [2000, 2000, 2000]


def test_load_positions_and_links(carto_map):
    egm = load_carto_egm(scan_carto_egm(str(carto_map), MAP))
    (conn, cat), = egm["catheters"].items()
    assert conn == CONN
    electrodes = [1, 2, 5, 6, 7, 8, 11, 12]
    pos = cat["positions"]
    # placement 0: complete, every electrode known
    for ie, k in enumerate(electrodes):
        assert pos[0, ie].tolist() == electrode_xyz(0, k)
    # placement 1: spline 0 (electrodes 1, 2) missing, the rest known
    for ie, k in enumerate(electrodes):
        if k <= L:
            assert np.isnan(pos[1, ie]).all()
        else:
            assert pos[1, ie].tolist() == electrode_xyz(1, k)
    # placement 2: undecidable, all NaN
    assert np.isnan(pos[2]).all()
    # bipoles -> electrode indices and trace columns
    assert cat["bipole_electrodes"].tolist() == [[0, 1], [2, 3], [4, 5], [6, 7]]
    assert cat["bipole_column"].tolist() == [1, 2, 3, 4]
    assert [egm["channel_names"][c] for c in cat["electrode_column"]] == \
        [f"20A_{k}" for k in electrodes]
    # point link: _car.txt row of each (placement, bipole), -1 elsewhere
    assert cat["point_link"].tolist() == [[0, -1, -1, 1],
                                          [-1, 2, -1, 3],
                                          [-1, -1, 4, -1]]
    assert not any("numbering" in w for w in egm["warnings"])


def move_point_3_to_electrode_6(folder: Path):
    """Point 3 (placement 1, bipole 20A_5-6) taken on the bipole's second
    electrode, as Carto does for some points: unipole 20A_6, _car.txt at 6."""
    write_point_xml(folder, 3, 2000, 6, bip="20A_5-6")
    car = folder / f"{MAP}_car.txt"
    old = "\t".join(["P", "0", "3", "0"] + [str(v) for v in electrode_xyz(1, 5)])
    new = "\t".join(["P", "0", "3", "0"] + [str(v) for v in electrode_xyz(1, 6)])
    assert old in car.read_text()
    car.write_text(car.read_text().replace(old, new))


def test_point_on_second_electrode(carto_map):
    move_point_3_to_electrode_6(carto_map)
    scan = scan_carto_egm(str(carto_map), MAP)
    # The bipole still spans electrodes 5 and 6; it is not read as 6-7.
    assert scan.bipoles == ["20A_1-2", "20A_5-6", "20A_7-8", "20A_11-12"]
    assert scan.bipole_electrodes == [(1, 2), (5, 6), (7, 8), (11, 12)]
    egm = load_carto_egm(scan)
    cat = egm["catheters"][CONN]
    assert cat["point_link"][1].tolist() == [-1, 2, -1, 3]
    assert cat["point_electrode"].tolist() == [[0, 0, 0, 0],
                                               [0, 1, 0, 0],
                                               [0, 0, 0, 0]]
    assert not any("numbering" in w for w in egm["warnings"])
    # Shown, and picked, where the point is.
    assert placement_anchors(egm, 1)[1].tolist() == electrode_xyz(1, 6)
    assert placement_anchors(egm, 1)[3].tolist() == electrode_xyz(1, 11)
    assert nearest_bipole(egm, electrode_xyz(1, 6)) == (1, 1)


def test_points_on_both_electrodes_of_one_bipole(carto_map):
    # Point 6 uses 20A_2 of bipole 20A_1-2 at another placement than point 1,
    # which uses 20A_1. Both belong to the same bipole.
    write_point_xml(carto_map, 6, 2000, 2, bip="20A_1-2")
    pts = carto_map / f"{MAP}_Points_Export.xml"
    pts.write_text(pts.read_text().replace(
        "</Points>", f'    <Point ID="6" File_Name="{MAP}_P6_Point_Export.xml" />\n</Points>'))
    scan = scan_carto_egm(str(carto_map), MAP)
    assert scan.bipole_electrodes[0] == (1, 2)
    assert len(scan.bipoles) == 4


def test_point_off_its_bipole_raises(carto_map):
    write_point_xml(carto_map, 3, 2000, 9, bip="20A_5-6")
    with pytest.raises(ValueError, match="20A_5-6"):
        scan_carto_egm(str(carto_map), MAP)


def test_point_on_another_catheter_is_skipped(carto_map):
    write_point_xml(carto_map, 6, 1000, 1)
    xml = (carto_map / f"{MAP}_P6_Point_Export.xml").read_text()
    (carto_map / f"{MAP}_P6_Point_Export.xml").write_text(
        xml.replace('UnipolarMappingChannel="20A_1"', 'UnipolarMappingChannel="M1"')
           .replace('BipolarMappingChannel="20A_1-2"', 'BipolarMappingChannel="M1-M2"'))
    pts = carto_map / f"{MAP}_Points_Export.xml"
    pts.write_text(pts.read_text().replace(
        "</Points>", f'    <Point ID="6" File_Name="{MAP}_P6_Point_Export.xml" />\n</Points>'))
    scan = scan_carto_egm(str(carto_map), MAP)
    assert len(scan.points) == len(POINTS)
    assert any("another catheter" in w for w in scan.warnings)


def test_pack_round_trip_through_pickle(carto_map):
    egm = load_carto_egm(scan_carto_egm(str(carto_map), MAP))
    packed = pack_egm(egm)
    assert isinstance(packed["traces"]["zlib"], bytes)
    back = unpack_egm(pickle.loads(pickle.dumps(packed)))
    assert np.array_equal(back["traces"], egm["traces"])
    assert back["traces"].dtype == np.int16
    assert unpack_egm(egm) is egm                    # already unpacked: unchanged


def test_missing_point_export_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        scan_carto_egm(str(tmp_path), "nothing")


def test_scan_detects_mapping_connector(carto_map):
    scan = scan_carto_egm(str(carto_map), MAP)
    assert scan.mapping_connector == CONN
    assert sorted(scan.connectors) == sorted([CONN, "CS_CONNECTOR"])


def test_user_can_override_connector(carto_map):
    # The user picks a connector that has no position files: traces are still
    # read, positions are NaN, and the reason is reported.
    egm = load_carto_egm(scan_carto_egm(str(carto_map), MAP), connector="CS_CONNECTOR")
    (conn, cat), = egm["catheters"].items()
    assert conn == "CS_CONNECTOR"
    assert np.isnan(cat["positions"]).all()
    assert any("no position file for CS_CONNECTOR" in w for w in egm["warnings"])
    assert egm["traces"].shape[0] == 3


# --- pkl bundle and moving positions ----------------------------------------

def test_egm_rides_in_the_bundle(carto_map, tmp_path):
    import pyvista as pv
    from ccdaf.core.eam_export import export_binary
    from ccdaf.core.eam_loader import read_bundle
    egm = load_carto_egm(scan_carto_egm(str(carto_map), MAP))
    path = tmp_path / "bundle.pkl"
    export_binary(path, pv.Sphere().triangulate(), egm=egm)
    with open(path, "rb") as fh:
        raw = pickle.load(fh)
    assert isinstance(raw["egm"]["traces"], dict)          # stored compressed
    _, _, _, back = read_bundle(path)
    assert np.array_equal(back["traces"], egm["traces"])
    (conn, cat), = back["catheters"].items()
    assert np.array_equal(cat["positions"], egm["catheters"][conn]["positions"],
                          equal_nan=True)
    assert cat["point_link"].tolist() == egm["catheters"][conn]["point_link"].tolist()
    assert cat["point_electrode"].tolist() == \
        egm["catheters"][conn]["point_electrode"].tolist()
    assert np.array_equal(back["channel_missing"], egm["channel_missing"])
    assert back["channel_names"] == egm["channel_names"]


def test_positions_round_trip_and_keep_nan(carto_map):
    from ccdaf.io.carto_egm import egm_positions, set_egm_positions
    egm = load_carto_egm(scan_carto_egm(str(carto_map), MAP))
    pos = next(iter(egm["catheters"].values()))["positions"]
    nan_before = np.isnan(pos).all(axis=2).copy()
    pts = egm_positions(egm)
    assert len(pts) == int((~nan_before).sum())
    set_egm_positions(egm, pts + 1.0)
    pos = next(iter(egm["catheters"].values()))["positions"]
    assert np.array_equal(np.isnan(pos).all(axis=2), nan_before)
    assert np.allclose(egm_positions(egm), pts + 1.0)
    with pytest.raises(ValueError):
        set_egm_positions(egm, pts[:-1])


# --- GUI: dialogs and the app flow (offscreen Qt, no plotter) ----------------

@pytest.fixture(scope="module")
def qapp():
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def test_load_dialog_electrograms_off_by_default(qapp):
    from ccdaf.gui.eam_load_dialog import EAMLoadDialog
    dlg = EAMLoadDialog()
    assert dlg.load_electrograms() is False
    dlg._chk_egm.setChecked(True)
    assert dlg.load_electrograms() is True


def test_channel_dialog_defaults_and_extras(qapp, carto_map):
    from PyQt5 import QtCore
    from ccdaf.gui.egm_channel_dialog import EGMChannelDialog
    scan = scan_carto_egm(str(carto_map), MAP)
    dlg = EGMChannelDialog(scan)
    assert dlg.selected_connector() == CONN                 # detected one pre-set
    offered = [dlg._list.item(i).text() for i in range(dlg._list.count())]
    assert "V1" in offered
    assert not set(offered) & set(scan.default_channels)   # always-stored not offered
    assert dlg.selected_extra_channels() == []              # nothing ticked
    dlg._list.item(offered.index("V1")).setCheckState(QtCore.Qt.Checked)
    assert dlg.selected_extra_channels() == ["V1"]
    dlg._cmb_connector.setCurrentIndex(scan.connectors.index("CS_CONNECTOR"))
    assert dlg.selected_connector() == "CS_CONNECTOR"


class _Host:
    """Only what the app's electrogram methods touch."""
    def __init__(self, eam_data, electrode_points=None):
        from unittest.mock import MagicMock
        self._eam_data = eam_data
        self._eam_electrode_points = electrode_points
        self._eam_warp_note = None
        self._status = MagicMock()

    def statusBar(self):
        return self._status

    def _on_egm_available(self):
        self.egm_available = True


def _quiet_boxes(monkeypatch):
    from PyQt5 import QtWidgets
    shown = []
    for kind in ("information", "warning", "critical"):
        monkeypatch.setattr(QtWidgets.QMessageBox, kind,
                            lambda *a, _k=kind, **k: shown.append(_k))
    return shown


def _fake_dialog(accept: bool, extras=(), connector=None):
    from PyQt5 import QtWidgets

    class Dlg:
        def __init__(self, scan, parent=None):
            self.scan = scan

        def exec_(self):
            return QtWidgets.QDialog.Accepted if accept else QtWidgets.QDialog.Rejected

        def selected_extra_channels(self):
            return list(extras)

        def selected_connector(self):
            return connector or self.scan.mapping_connector
    return Dlg


def test_app_loads_electrograms_after_the_dialog(qapp, carto_map, monkeypatch):
    import types
    import ccdaf.app.ccdaf as app
    shown = _quiet_boxes(monkeypatch)
    monkeypatch.setattr(app, "EGMChannelDialog", _fake_dialog(True, extras=["V1"]))
    host = _Host(types.SimpleNamespace(egm=None))
    app.CCDAF._load_eam_electrograms(host, str(carto_map), MAP)
    assert host._eam_data.egm is not None
    assert host._eam_data.egm["channel_names"][-1] == "V1"
    assert shown == ["information"]           # the missing-spline note is shown


def test_app_cancel_loads_no_electrograms(qapp, carto_map, monkeypatch):
    import types
    import ccdaf.app.ccdaf as app
    _quiet_boxes(monkeypatch)
    monkeypatch.setattr(app, "EGMChannelDialog", _fake_dialog(False))
    host = _Host(types.SimpleNamespace(egm=None))
    app.CCDAF._load_eam_electrograms(host, str(carto_map), MAP)
    assert host._eam_data.egm is None


def test_app_scan_failure_keeps_the_map(qapp, tmp_path, monkeypatch):
    import types
    import ccdaf.app.ccdaf as app
    shown = _quiet_boxes(monkeypatch)
    host = _Host(types.SimpleNamespace(egm=None))
    app.CCDAF._load_eam_electrograms(host, str(tmp_path), "nothing")
    assert host._eam_data.egm is None and shown == ["warning"]


def test_catheters_move_with_the_electrodes(qapp, carto_map, monkeypatch):
    # Smoothing moves the _car.txt electrodes and the catheter electrodes in
    # one call with the same rule.
    import types
    import pyvista as pv
    import ccdaf.app.ccdaf as app
    from ccdaf.io.carto_egm import egm_positions
    calls = []

    def fake_displace(pts, old, new, on_status=None):
        calls.append(len(pts))
        return np.asarray(pts) + 1.0
    monkeypatch.setattr(app, "displace_electrodes_for", fake_displace)
    egm = load_carto_egm(scan_carto_egm(str(carto_map), MAP))
    before = egm_positions(egm).copy()
    elec = np.zeros((5, 3))
    host = _Host(types.SimpleNamespace(egm=egm), electrode_points=elec)
    sphere = pv.Sphere().triangulate()
    app.CCDAF._on_surface_moved(host, sphere, sphere)
    assert calls == [len(elec) + len(before)]                # one call for both
    assert np.allclose(host._eam_electrode_points, elec + 1.0)
    assert np.allclose(egm_positions(egm), before + 1.0)
