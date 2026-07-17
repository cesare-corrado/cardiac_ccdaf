"""
test_headless_boundary.py
=========================
The {core, io} vs {gui, app} boundary, enforced rather than conventional.

Everything under ``ccdaf.core`` and ``ccdaf.io`` must import — every
module, transitively — without PyQt5 ever appearing in ``sys.modules``.
That is what keeps the CLI pipeline and this whole test suite runnable
on a machine with no Qt and no display; today it holds by discipline,
and this test is what turns an accidental ``from PyQt5 import ...`` in
core next month into a red build instead of a silent regression.

The probe runs in a fresh interpreter: the pytest process itself is
free to load Qt for other tests, so checking ``sys.modules`` in-process
would depend on test order.
"""

import os
import subprocess
import sys
from pathlib import Path

SRC = str(Path(__file__).resolve().parents[2] / "src")

_PROBE = """
import importlib, pkgutil, sys
import ccdaf.core, ccdaf.io
for pkg in (ccdaf.core, ccdaf.io):
    for mod in pkgutil.iter_modules(pkg.__path__):
        importlib.import_module(pkg.__name__ + "." + mod.name)
offenders = sorted(m for m in sys.modules
                   if m == "PyQt5" or m.startswith("PyQt5."))
assert not offenders, "Qt reached core/io via: " + ", ".join(offenders)
print("qt-free")
"""


def test_core_and_io_import_without_qt():
    env = dict(os.environ, PYTHONPATH=SRC)
    res = subprocess.run(
        [sys.executable, "-c", _PROBE],
        env=env, capture_output=True, text=True, timeout=180,
    )
    assert res.returncode == 0, res.stderr
    assert "qt-free" in res.stdout
