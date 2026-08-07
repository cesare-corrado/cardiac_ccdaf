# Contributing

## Environment

Use the pinned `ccdaf` conda environment (never base Anaconda — its VTK is too
old). See [Installation](../installation.md).

## Tests

The default suite is headless and synthetic (no private data). Qt widgets need
the offscreen platform:

```bash
QT_QPA_PLATFORM=offscreen PYTHONPATH=src python -m pytest tests/unit -q
```

- **Tier-1** (`tests/unit`) is the CI collection.
- **Tier-2** (`tests/nightly`, marked `nightly`) holds real Carto-data
  regressions, run by a local runner against private data.

Add or update a unit test with every change; keep the suite green.

## Lint

```bash
ruff check src tests
```

The gate is rules `E4,E7,E9,F` (see `ruff.toml`).

## Documentation

```bash
pip install -e ".[docs]"
mkdocs serve            # live preview
```

- Prose pages live in `docs/`; the [API reference](../reference/index.md) is
  auto-generated from docstrings by `mkdocstrings`.
- Regenerate the static figures with `python docs/assets/gen_figures.py`.
- GUI screenshots are captured by hand and dropped into
  `docs/assets/img/screenshots/`.

## Git workflow

- Branch off `develop`; never commit directly to `develop` or `main`. Name the
  branch for what the work is: `feature/<short-name>` for new capability,
  `hotfix/<short-name>` for a fix, `release/<version>` for a version bump.
- Merge into `develop` with `--no-ff`; promote `develop → main` with `--no-ff`
  on release, keeping them in sync.
- Releases are tagged `vMAJOR.MINOR.PATCH` (e.g. `v1.1.0`), annotated, on the
  merge commit on `main`. The version literal lives in `pyproject.toml`,
  `src/ccdaf/__init__.py` and the `README.md` heading; the Help → About dialog
  reads `__version__`, so it follows on its own.
