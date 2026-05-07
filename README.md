# CCDAF 0.1.0

> **NOT FOR CLINICAL USE.** This software is intended for research purposes
> only and has not been validated for clinical decision-making.

**CCDAF** (Cardiac Clinical Data Analysis Framework) is a GUI application for
post-processing left-atrial surface meshes in the context of cardiac
electrophysiology research.

The main workflow covers:

- Loading triangular surface meshes (`.vtk`)
- Interactive placement of 6 anatomical seed points (LSPV, LIPV, RSPV, RIPV, LAA, MV)
- Automatic geodesic region tagging (pulmonary veins, left atrial appendage)
- Manual label correction with undo support
- PV ostium and mitral valve clipping (contour snake, sphere, or plane)
- Mesh quality post-processing (decimate, refine, clean)

A parallel workflow supports voxel segmentation from `.nii` images, including
morphological operations, manual paint, and export to VTK.


## Pre-requisites

Python ≥ 3.9 is required.

### Option 1 — pip

Create and activate an environment (venv or conda), then install from the
repository root:

```bash
cd cardiac_ccdaf
pip install .
```

### Option 2 — conda

Create the full environment from the provided `environment.yml`:

```bash
cd cardiac_ccdaf
conda env create -f environment.yml
conda activate ccdaf
```

Both options can coexist. The conda environment resolves binary dependencies
(VTK, PyQt5) from conda-forge. Option 1 resolves them from PyPI.


## Run the code

After installation, launch the GUI with:

```bash
ccdaf                        # open with no mesh
ccdaf path/to/mesh.vtk       # open with a mesh pre-loaded
```

During development, the app can also be run directly without installing:

```bash
python src/ccdaf/app/ccdaf.py
python src/ccdaf/app/ccdaf.py path/to/mesh.vtk
```


## Tests

Run the full test suite with:

```bash
pytest tests/
```

Individual test modules can also be run as standalone scripts:

```bash
python tests/test_seed_state_machine.py
python tests/test_seed_geometry.py
```
