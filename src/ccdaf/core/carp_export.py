"""
CARP export
===========

Write the working mesh in the CARP format a simulator such as openCARP reads:
three ASCII files sharing one prefix.

* ``<name>.pts`` — a header line with the node count, then ``x y z`` per node.
  **CARP works in micrometres**, so the coordinates are multiplied by a scale
  factor on the way out (1000 for a mesh in millimetres).
* ``<name>.elem`` — a header line with the element count, then a type code
  (``Tt`` for a tetrahedron, ``Tr`` for a triangle), the element's nodes, and
  optionally an integer region tag. The tag is what a simulation groups
  elements by: "openCARP regions are then conglomerations of element region
  tags", which is how ``tissueTag`` reaches a simulation.
* ``<name>.lon`` — a header line holding 1 or 2 (fibre only, or fibre and
  sheet), then one direction per element.

Fibres
------
A mesh with no fibre field is written with the fibre-only form and every
element pointing along ``(1, 0, 0)``. That is a placeholder, valid only where
conductivity is isotropic, and it is reported as one. Vectors that are not
unit length are normalised in the written file, never in the mesh; a vector of
zero length has no direction to normalise and stops the export.

Bilayer
-------
An atrial surface with endocardial and epicardial fibres (``fibre_endo``,
``fibre_epi``) can be written as a bilayer, the format of Roney et al. 2023
(*Interface Focus* 13: 20230038) and Labarthe et al.: the surface is the
endocardial layer; a copy moved outwards along the point normals by the
wall thickness is the epicardial layer; and a line element (``Ln``) joins
each node to its copy. Wall thickness enters a simulation through the
lines' conductivity, so the distance itself is nominal (0.1 mm in the
published models).

* Points: the endocardial nodes, then their copies in the same order.
* Elements: the endocardial triangles, then the epicardial ones, then one
  line per node.
* Tags: the endocardial triangles carry the chosen region tag; the
  epicardial ones the same tag plus :attr:`CarpExportOptions.epi_tag_offset`
  (100 by default: the published +1 would turn ccdaf's LA body, 1, into its
  RA body, 2); the lines :attr:`CarpExportOptions.line_tag` (32, as
  published).
* Fibres: ``fibre_endo`` and ``fibre_epi`` on the triangles. On each line,
  its own direction: if the solver uses the usual finite-element
  conductivity, a line then conducts at the longitudinal conductivity of
  its region whichever way it points. The published files write (1, 0, 0)
  on every line, which is the same only when the line region is
  isotropic.

Region tags
-----------
The tag column is **always written**. With no array chosen every element is
written as :data:`DEFAULT_TAG`, so the file states which region each element
is in instead of leaving a reader to assume one. A reader that expects no tag
column has to skip it.

Tags must be whole, non-negative numbers. Values above 255 are reported
because older CARP stored the tag as an unsigned char, so a larger tag will
not survive there. Every tag written is listed in the result, since a tag that
no region of the simulation lists is silently taken as region 0.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ccdaf.core.volume_mesh import element_connectivity

#: CARP's element type code, by nodes per element.
ELEMENT_CODE: Dict[int, str] = {3: "Tr", 4: "Tt"}

#: CARP expects micrometres; a mesh in millimetres needs this factor.
DEFAULT_SCALE = 1000.0

#: Cell fields offered as fibres, and as sheets. Named rather than guessed
#: from the component count: a displacement is also three numbers per
#: element and is not a direction. Mirrors
#: :data:`ccdaf.core.field_transfer.AXIAL_CELL_FIELDS`.
FIBRE_FIELDS: Tuple[str, ...] = ("fiber", "fibre", "fibre_endo", "fibre_epi")
SHEET_FIELDS: Tuple[str, ...] = ("sheet",)

#: The placeholder direction written when the mesh carries no fibres.
PLACEHOLDER_FIBRE = (1.0, 0.0, 0.0)

#: Older CARP held an element's region tag in an unsigned char.
MAX_TAG = 255

#: Written for every element when no region array is chosen. The tag column
#: is always written, so the file says which region each element is in rather
#: than leaving a reader to assume one; 0 is where a simulation's default
#: (normally healthy) properties sit.
DEFAULT_TAG = 0

#: Shorter than this, a vector gives no direction to normalise.
MIN_FIBRE_NORM = 1e-6

#: How far a sheet may be from perpendicular to its fibre before it is worth
#: reporting: |cos| above this is not a sheet direction.
SHEET_ORTHOGONALITY_TOLERANCE = 1e-2

#: A scaled mesh whose longest side falls outside this, in micrometres
#: (0.5 cm to 50 cm), is almost certainly the wrong scale factor.
PLAUSIBLE_EXTENT_UM: Tuple[float, float] = (5.0e3, 5.0e5)

#: Written file suffixes, in the order they are written.
SUFFIXES: Tuple[str, ...] = (".pts", ".elem", ".lon")

# Six significant digits, not a fixed number of decimals. Mesh points are
# single precision, which carries about seven significant digits in total, so
# a fixed "%.3f" in micrometres writes down to the nanometre: digits the data
# never had, padded with zeros. Six keeps what a millimetre-scale mesh
# actually holds and writes no more.
_POINT_FORMAT = "%.6g"
_FIBRE_FORMAT = "%.6f"        # unit vectors, so six decimals is six digits
# A bilayer's lines are 100 um long between nodes about 50,000 um from the
# origin: at six digits their lengths come out between 99.4 and 100.5 um,
# and the coupling between the layers goes with that length. Eight digits
# keep it to about 0.01 um. The extra digits are rounding of computed
# positions, not measured data.
_BILAYER_POINT_FORMAT = "%.8g"


def region_fields(dataset) -> List[str]:
    """Cell arrays that can be written as region tags: whole numbers, one per element."""
    out: List[str] = []
    if dataset is None:
        return out
    for name in dataset.cell_data.keys():
        values = np.asarray(dataset.cell_data[name])
        if values.ndim != 1 or values.shape[0] != dataset.n_cells:
            continue
        if np.issubdtype(values.dtype, np.integer):
            out.append(str(name))
        elif np.issubdtype(values.dtype, np.floating) and np.all(np.isfinite(values)) \
                and np.allclose(values, np.rint(values)):
            out.append(str(name))
    return out


def direction_fields(dataset, names: Tuple[str, ...]) -> List[str]:
    """The three-component cell arrays among *names* that the mesh carries."""
    out: List[str] = []
    if dataset is None:
        return out
    for name in dataset.cell_data.keys():
        if str(name) not in names:
            continue
        values = np.asarray(dataset.cell_data[name])
        if values.ndim == 2 and values.shape == (dataset.n_cells, 3):
            out.append(str(name))
    return out


def availability(dataset) -> Tuple[bool, str]:
    """Whether *dataset* can be written as CARP, and why not."""
    if dataset is None:
        return False, "Load a mesh first."
    try:
        element_connectivity(dataset)
    except ValueError as exc:
        return False, str(exc).capitalize()
    return True, "Write the mesh as CARP .pts, .elem and .lon files."


@dataclass
class CarpExportOptions:
    """Where the files go, and what rides in them."""

    directory: str
    name: str
    scale: float = DEFAULT_SCALE
    #: Cell array written as the ``.elem`` region tag; ``None`` writes none.
    region_field: Optional[str] = None
    #: Cell array written to ``.lon``; ``None`` writes the placeholder.
    fibre_field: Optional[str] = None
    #: Second direction, making the ``.lon`` header 2.
    sheet_field: Optional[str] = None
    #: Write the surface as an endocardial and an epicardial layer joined
    #: by lines (see the module notes). Needs ``fibre_endo`` and ``fibre_epi``.
    bilayer: bool = False
    #: Point field giving the distance between the layers, in mm; ``None``
    #: uses :attr:`thickness_mm` everywhere.
    thickness_field: Optional[str] = None
    thickness_mm: float = 0.1
    #: Millimetres per mesh unit, to move the epicardial layer by a
    #: thickness given in mm.
    mm_per_unit: float = 1.0
    epi_tag_offset: int = 100
    line_tag: int = 32

    @property
    def prefix(self) -> Path:
        return Path(self.directory) / self.name

    def paths(self) -> List[Path]:
        return [self.prefix.with_suffix(suffix) for suffix in SUFFIXES]

    def existing(self) -> List[Path]:
        """Those of :meth:`paths` already on disk."""
        return [path for path in self.paths() if path.exists()]

    def validate(self) -> None:
        """Raise ``ValueError`` on anything that must stop the export."""
        if not str(self.name).strip():
            raise ValueError("Give the files a name.")
        if Path(self.name).suffix in SUFFIXES:
            raise ValueError(
                f"Leave the suffix off the name: all of {', '.join(SUFFIXES)} "
                "are written from it.")
        if not Path(self.directory).is_dir():
            raise ValueError(f"No such directory:\n{self.directory}")
        if not np.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("The scale factor must be greater than 0.")
        if self.sheet_field and not self.fibre_field:
            raise ValueError("A sheet direction needs a fibre direction as well.")
        if self.bilayer:
            if not self.thickness_field and not self.thickness_mm > 0.0:
                raise ValueError("The wall thickness must be greater than 0.")
            if self.epi_tag_offset < 0 or self.line_tag < 0:
                raise ValueError("Tags cannot be negative.")


@dataclass
class CarpExportResult:
    """What was written."""

    paths: List[Path]
    n_points: int
    n_elements: int
    element_code: str
    #: ``{tag: elements}``, empty when no region tag was written.
    tags: Dict[int, int] = dc_field(default_factory=dict)
    #: 1 for fibres only, 2 with sheets.
    fibre_axes: int = 1
    #: True when the fibres are the placeholder rather than the mesh's own.
    placeholder_fibres: bool = False
    #: Vectors that were not unit length and were normalised on the way out.
    normalised: int = 0
    warnings: List[str] = dc_field(default_factory=list)

    def summary(self) -> str:
        text = (f"Wrote {self.paths[0].with_suffix('')}.[pts|elem|lon] — "
                f"{self.n_points} nodes, {self.n_elements} {self.element_code} elements")
        if self.tags:
            listed = ", ".join(f"{tag}: {count}" for tag, count in sorted(self.tags.items()))
            text += f"; region tags {listed}"
        else:
            text += "; no region tag"
        text += "; fibres " + ("placeholder (1, 0, 0)" if self.placeholder_fibres
                               else f"from the mesh ({self.fibre_axes} per element)")
        if self.normalised:
            text += f", {self.normalised} normalised"
        return text + "."


def _directions(dataset, options: CarpExportOptions, n_elements: int,
                notes: List[str]) -> Tuple[np.ndarray, int, bool, int]:
    """``(vectors, axes, placeholder, normalised)`` for the ``.lon`` file."""
    normalised = 0
    if not options.fibre_field:
        notes.append(
            "The mesh carries no fibre direction, so every element is written "
            f"as {PLACEHOLDER_FIBRE}. That is only valid where conductivity is "
            "isotropic.")
        return np.tile(PLACEHOLDER_FIBRE, (n_elements, 1)), 1, True, 0

    columns = []
    for name in (options.fibre_field, options.sheet_field):
        if not name:
            continue
        vectors = np.asarray(dataset.cell_data[name], dtype=float)
        if vectors.shape != (n_elements, 3):
            raise ValueError(
                f"'{name}' holds {vectors.shape} values, not one 3-vector per element.")
        norms = np.linalg.norm(vectors, axis=1)
        if not np.all(np.isfinite(vectors)) or np.any(norms < MIN_FIBRE_NORM):
            raise ValueError(
                f"{int(np.count_nonzero(norms < MIN_FIBRE_NORM))} vectors of "
                f"'{name}' have no length, so they give no direction.")
        off = ~np.isclose(norms, 1.0, atol=1e-6)
        if off.any():
            normalised += int(off.sum())
            vectors = vectors / norms[:, None]
        if len(np.unique(np.round(vectors, 6), axis=0)) == 1:
            single = tuple(round(float(v), 3) for v in vectors[0])
            notes.append(
                f"'{name}' holds the same direction on every element, {single}. "
                "The file will be valid, but the tissue is oriented one way "
                "throughout rather than following a fibre architecture, which "
                "only behaves sensibly with isotropic conductivities.")
        columns.append(vectors)

    if len(columns) == 2:
        cosine = np.abs(np.einsum("ij,ij->i", columns[0], columns[1]))
        crooked = int(np.count_nonzero(cosine > SHEET_ORTHOGONALITY_TOLERANCE))
        if crooked:
            notes.append(
                f"{crooked} sheet directions are not perpendicular to their fibre.")
    return np.hstack(columns), len(columns), False, normalised


def _tags(dataset, options: CarpExportOptions, n_elements: int,
          notes: List[str]) -> Optional[np.ndarray]:
    """The ``.elem`` region column, or ``None``."""
    if not options.region_field:
        notes.append(
            f"No region array was chosen, so every element is written with tag "
            f"{DEFAULT_TAG}: one region, taking that region's properties, "
            "normally the healthy ones. Any material regions the mesh carries "
            "are left out of the file.")
        return np.full(n_elements, DEFAULT_TAG, dtype=np.int64)
    values = np.asarray(dataset.cell_data[options.region_field])
    if values.ndim != 1 or values.shape[0] != n_elements:
        raise ValueError(
            f"'{options.region_field}' does not hold one value per element.")
    if not np.issubdtype(values.dtype, np.integer):
        if not np.all(np.isfinite(values)) or not np.allclose(values, np.rint(values)):
            raise ValueError(
                f"'{options.region_field}' holds values that are not whole "
                "numbers, so they cannot be region tags.")
    tags = np.rint(np.asarray(values, dtype=float)).astype(np.int64)
    if tags.min() < 0:
        raise ValueError(
            f"'{options.region_field}' holds negative values "
            f"(lowest {int(tags.min())}); a region tag cannot be negative.")
    if tags.max() > MAX_TAG:
        notes.append(
            f"The highest region tag is {int(tags.max())}. Older CARP versions "
            f"store the tag as 0 to {MAX_TAG}, and will not read it.")
    return tags


#: The fibre fields a bilayer reads, for the endocardial and epicardial layers.
BILAYER_FIBRES: Tuple[str, str] = ("fibre_endo", "fibre_epi")


def bilayer_availability(dataset) -> Tuple[bool, str]:
    """Whether *dataset* can be written as a bilayer, and if not, why."""
    if dataset is None:
        return False, "Load a mesh first."
    try:
        conn = element_connectivity(dataset)
    except ValueError as exc:
        return False, str(exc).capitalize()
    if conn.shape[1] != 3:
        return False, "A bilayer is made from a surface, not a volume."
    missing = [n for n in BILAYER_FIBRES if n not in direction_fields(dataset, BILAYER_FIBRES)]
    if missing:
        return False, ("Map bilayer fibres first (Actions > Atrial fibres…, "
                       "layer bilayer).")
    return True, "Write an endocardial and an epicardial layer joined by lines."


def _unit_rows(vectors: np.ndarray, name: str) -> Tuple[np.ndarray, int]:
    vectors = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(vectors, axis=1)
    if not np.all(np.isfinite(vectors)) or np.any(norms < MIN_FIBRE_NORM):
        raise ValueError(
            f"{int(np.count_nonzero(~(norms >= MIN_FIBRE_NORM)))} vectors of "
            f"'{name}' have no length, so they give no direction.")
    off = ~np.isclose(norms, 1.0, atol=1e-6)
    return vectors / norms[:, None], int(off.sum())


def _outward_normals(dataset) -> np.ndarray:
    """Unit point normals pointing out of the atrium."""
    import pyvista as pv
    surf = pv.wrap(dataset)
    if not isinstance(surf, pv.PolyData):
        surf = surf.extract_surface()
    with_n = surf.compute_normals(point_normals=True, cell_normals=False,
                                  consistent_normals=True,
                                  auto_orient_normals=True, split_vertices=False,
                                  inplace=False)
    normals = np.asarray(with_n.point_data["Normals"], dtype=float)
    # Auto-orientation is made for closed surfaces; an atrium is open at its
    # valve and veins. Check against the centre and turn them all if most
    # point inwards.
    pts = np.asarray(surf.points, dtype=float)
    if np.median(np.einsum("ij,ij->i", normals, pts - pts.mean(0))) < 0:
        normals = -normals
    return normals / np.linalg.norm(normals, axis=1, keepdims=True)


def write_bilayer(dataset, options: CarpExportOptions, *,
                  dry_run: bool = False) -> CarpExportResult:
    """Write *dataset*, an atrial surface with bilayer fibres, as a CARP
    bilayer (see the module notes). Raises ``ValueError`` on a blocking
    problem."""
    options.validate()
    ok, why = bilayer_availability(dataset)
    if not ok:
        raise ValueError(why)
    notes: List[str] = []
    tri = element_connectivity(dataset)
    n_nodes, n_tri = int(dataset.n_points), len(tri)

    if options.thickness_field:
        thickness = np.asarray(dataset.point_data[options.thickness_field], dtype=float)
        if thickness.shape != (n_nodes,) or not np.all(np.isfinite(thickness)):
            raise ValueError(f"'{options.thickness_field}' is not one thickness per node.")
    else:
        thickness = np.full(n_nodes, float(options.thickness_mm))
    if np.any(thickness <= 0):
        raise ValueError(
            f"{int(np.count_nonzero(thickness <= 0))} nodes have no wall "
            "thickness: their two layers would coincide.")
    used = np.unique(tri)
    if len(used) != n_nodes:
        notes.append(
            f"{n_nodes - len(used)} nodes are in no triangle; their lines "
            "join nodes no triangle uses.")

    endo_pts = np.asarray(dataset.points, dtype=float)
    offset = thickness / float(options.mm_per_unit)
    epi_pts = endo_pts + _outward_normals(dataset) * offset[:, None]
    points = np.vstack([endo_pts, epi_pts]) * float(options.scale)
    extent = float((points.max(axis=0) - points.min(axis=0)).max())
    low, high = PLAUSIBLE_EXTENT_UM
    if not low <= extent <= high:
        notes.append(
            f"Scaled, the mesh is {extent / 1e3:.1f} mm across. CARP reads "
            "micrometres, so check the scale factor.")

    endo_tags = _tags(dataset, options, n_tri, notes)
    epi_tags = endo_tags + int(options.epi_tag_offset)
    if set(np.unique(endo_tags)) & set(np.unique(epi_tags)):
        notes.append("Some endocardial and epicardial triangles share a tag: "
                     "raise the epicardial offset to keep the layers apart.")
    if int(options.line_tag) in set(np.unique(np.r_[endo_tags, epi_tags])):
        notes.append(f"The line tag {options.line_tag} is also a triangle "
                     "tag: the lines would share that region's properties.")
    if max(int(epi_tags.max()), int(options.line_tag)) > MAX_TAG:
        notes.append(f"A tag is above {MAX_TAG}. Older CARP versions store the "
                     f"tag as 0 to {MAX_TAG}, and will not read it.")

    f_endo, n1 = _unit_rows(dataset.cell_data[BILAYER_FIBRES[0]], BILAYER_FIBRES[0])
    f_epi, n2 = _unit_rows(dataset.cell_data[BILAYER_FIBRES[1]], BILAYER_FIBRES[1])
    lines = epi_pts - endo_pts
    f_line = lines / np.linalg.norm(lines, axis=1, keepdims=True)

    points_path, elements_path, fibres_path = options.paths()
    n_elements = 2 * n_tri + n_nodes
    if not dry_run:
        with open(points_path, "w") as handle:
            handle.write(f"{len(points)}\n")
            np.savetxt(handle, points, fmt=_BILAYER_POINT_FORMAT)
        with open(elements_path, "w") as handle:
            handle.write(f"{n_elements}\n")
            np.savetxt(handle, np.column_stack([tri, endo_tags]), fmt="Tr %d %d %d %d")
            np.savetxt(handle, np.column_stack([tri + n_nodes, epi_tags]),
                       fmt="Tr %d %d %d %d")
            ids = np.arange(n_nodes)
            np.savetxt(handle, np.column_stack([ids, ids + n_nodes,
                                                np.full(n_nodes, int(options.line_tag))]),
                       fmt="Ln %d %d %d")
        with open(fibres_path, "w") as handle:
            handle.write("1\n")
            np.savetxt(handle, np.vstack([f_endo, f_epi, f_line]), fmt=_FIBRE_FORMAT)

    counts: Dict[int, int] = {}
    for values in (endo_tags, epi_tags, np.full(n_nodes, int(options.line_tag))):
        for v, c in zip(*np.unique(values, return_counts=True)):
            counts[int(v)] = counts.get(int(v), 0) + int(c)
    return CarpExportResult(
        paths=[points_path, elements_path, fibres_path],
        n_points=len(points), n_elements=n_elements, element_code="Tr+Ln",
        tags=counts, fibre_axes=1, placeholder_fibres=False,
        normalised=n1 + n2, warnings=notes)


def write_carp(dataset, options: CarpExportOptions, *,
               dry_run: bool = False) -> CarpExportResult:
    """Write *dataset* as CARP files; raise ``ValueError`` on a blocking problem.

    Nothing outside the three files is touched: a fibre normalised here is
    normalised on the way out, not in the mesh.

    ``dry_run`` checks everything and reports what would be written without
    creating a file, so a caller can put the warnings to the user while the
    directory is still untouched.
    """
    if options.bilayer:
        return write_bilayer(dataset, options, dry_run=dry_run)
    options.validate()
    notes: List[str] = []
    connectivity = element_connectivity(dataset)
    n_elements, nodes = connectivity.shape
    code = ELEMENT_CODE[nodes]

    points = np.asarray(dataset.points, dtype=float) * float(options.scale)
    extent = float((points.max(axis=0) - points.min(axis=0)).max()) if len(points) else 0.0
    low, high = PLAUSIBLE_EXTENT_UM
    if extent and not low <= extent <= high:
        notes.append(
            f"Scaled, the mesh is {extent / 1e3:.1f} mm across. CARP reads "
            "micrometres, so check the scale factor.")

    unused = len(points) - len(np.unique(connectivity)) if n_elements else len(points)
    if unused:
        noun = "node is" if unused == 1 else "nodes are"
        notes.append(
            f"{unused} {noun} in the mesh but in no element. A solver "
            "numbers its nodes from the element list, so it will refuse "
            "this mesh: clean the volume before exporting.")

    tags = _tags(dataset, options, n_elements, notes)
    vectors, axes, placeholder, normalised = _directions(
        dataset, options, n_elements, notes)

    points_path, elements_path, fibres_path = options.paths()
    if not dry_run:
        with open(points_path, "w") as handle:
            handle.write(f"{len(points)}\n")
            np.savetxt(handle, points, fmt=_POINT_FORMAT)
        with open(elements_path, "w") as handle:
            handle.write(f"{n_elements}\n")
            rows = connectivity if tags is None else np.column_stack([connectivity, tags])
            np.savetxt(handle, rows, fmt=f"{code} " + " ".join(["%d"] * rows.shape[1]))
        with open(fibres_path, "w") as handle:
            handle.write(f"{axes}\n")
            np.savetxt(handle, vectors, fmt=_FIBRE_FORMAT)

    counts: Dict[int, int] = {}
    if tags is not None:
        values, occurrences = np.unique(tags, return_counts=True)
        counts = {int(v): int(c) for v, c in zip(values, occurrences)}
    return CarpExportResult(
        paths=[points_path, elements_path, fibres_path],
        n_points=len(points), n_elements=n_elements, element_code=code,
        tags=counts, fibre_axes=axes, placeholder_fibres=placeholder,
        normalised=normalised, warnings=notes)


__all__ = [
    "ELEMENT_CODE", "DEFAULT_SCALE", "FIBRE_FIELDS", "SHEET_FIELDS",
    "BILAYER_FIBRES", "bilayer_availability", "write_bilayer",
    "PLACEHOLDER_FIBRE", "MAX_TAG", "DEFAULT_TAG", "MIN_FIBRE_NORM",
    "PLAUSIBLE_EXTENT_UM",
    "SUFFIXES", "region_fields", "direction_fields", "availability",
    "CarpExportOptions", "CarpExportResult", "write_carp",
]
