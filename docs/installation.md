# Installation

CCDAF is a PyQt5 + PyVista/VTK desktop application. It is validated against a
specific scientific stack — **VTK 9.6.2 / PyVista 0.48.4** on **Python ≥ 3.12**
(tested on 3.14). Older VTK (e.g. 9.0.3 in base Anaconda) behaves differently
and is unsupported.

## With conda (recommended)

The repository ships an `environment.yml` that pins the validated stack and
installs CCDAF in editable mode:

```bash
git clone https://github.com/cesare-corrado/cardiac_ccdaf.git
cd cardiac_ccdaf
conda env create -f environment.yml
conda activate ccdaf
```

!!! note "Font-cache isolation"
    The environment sets `XDG_CACHE_HOME` to a private directory. conda-forge's
    VTK/PyQt combo can crash inside `libfontconfig` when the shared
    `~/.cache/fontconfig` holds cache files written by another fontconfig
    version; isolating the cache avoids that.

## With pip

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Editable install exposes the package **and** the `ccdaf` launcher (see
`[project.scripts]` in `pyproject.toml`).

## Running

```bash
ccdaf                      # launch the GUI
ccdaf path/to/mesh.vtk     # open a surface on start
ccdaf path/to/session.pkl  # open a bundle on start
ccdaf path/to/scan.nii.gz  # open a segmentation on start
ccdaf --version            # print the version and exit
ccdaf --help               # usage
```

The optional `PATH` is opened exactly as **File → Load data** would open it:
the reader is chosen from the extension, so a `.pkl` bundle comes back with its
tagging, seeds and electrodes, and a `.nii`/`.nii.gz` volume opens the
segmentation view. A path that does not exist is reported on the command line
and the window never opens.

## Documentation build (optional)

To build this documentation site locally:

```bash
pip install -e ".[docs]"
mkdocs serve          # live preview at http://127.0.0.1:8000
mkdocs build          # static site into ./site
```

## Verifying the environment

```bash
python -c "import vtk, pyvista; print(vtk.vtkVersion.GetVTKVersion(), pyvista.__version__)"
# expect 9.6.x  0.48.x
```

Running the test suite (see [Contributing](developer/contributing.md)) is the
most thorough check that the stack is healthy.
