# Export

The **Export** menu writes the working mesh in another tool's format. It
leaves the mesh alone: nothing an export does changes the data in CCDAF.

## Carp

**Export → Carp…** writes three ASCII files that share one prefix, in the
format a simulator such as openCARP reads. It is available whenever a mesh is
loaded, surface or volume.

| Control | What it does |
|---|---|
| **directory** | Where the files go. **Browse…** picks it. |
| **file name** | The prefix of all three files, without a suffix. |
| **scale factor** | Point coordinates are multiplied by this. CARP works in micrometres, so a mesh in millimetres needs **1000**, the default. |
| **region tag** | The cell array written as the `.elem` region column. Only whole-numbered arrays are offered, so `tissueTag` and `elemTag` qualify and a measured field does not. It starts on `tissueTag` when the mesh has one. With *(none)* the column is still written, as 0 for every element. |
| **fibres** | The direction written to `.lon`, or the placeholder. Only the names the project treats as directions are offered (`fiber`, `fibre`). |
| **sheets** | A second direction per element (`sheet`), which makes the `.lon` header 2. It needs a fibre direction, so it stays greyed until one is chosen. |

### What the three files hold

| File | Contents |
|---|---|
| `<name>.pts` | A line with the node count, then `x y z` per node, in micrometres |
| `<name>.elem` | A line with the element count, then one row per element: `Tt` and four nodes for a tetrahedron, `Tr` and three for a triangle, then the region tag |
| `<name>.lon` | A line holding `1` (fibre only) or `2` (fibre and sheet), then one direction per element |

Node numbers are the mesh's own, counted from 0.

### The region tag is how a simulation sees your regions

openCARP groups elements by that column: regions there are "conglomerations of
element region tags", and the tags decide which ionic model and which
conductivities an element gets. Writing `tissueTag` is therefore what carries
the work of [Tissue properties](tissue-properties.md) into a simulation.

Two things to know:

- **A tag that no region of your simulation lists falls into region 0**, with
  region 0's properties, and nothing says so. The status message lists every
  tag written, with its element count, so you can check them against your
  parameter file.
- **The column is always written.** The format allows leaving it out, but a
  file without it only says which region an element is in by convention.
  Choosing *(none)* therefore writes 0 for every element, which is the region a
  simulation's default (normally healthy) properties sit in. A reader written
  for files without the column has to skip the last field of each row.

### Fibres

With no fibre field the file is written in the fibre-only form with every
element pointing along `(1, 0, 0)`. That is a placeholder, and it is only
valid where conductivity is isotropic; the export says so before writing.

Vectors that are not unit length are normalised **in the file**, never in the
mesh, and the count is reported. A vector of zero length has no direction to
normalise, and stops the export.

## Checks before writing

The export refuses, with a message, when:

- the name is empty, or carries one of the suffixes it writes;
- the directory does not exist;
- the scale factor is not above 0;
- a sheet direction is chosen without a fibre direction;
- the region array holds values that are not whole numbers, or holds a
  negative value;
- a fibre or sheet vector has no length.

It asks before writing when:

- files with those names already exist;
- the scaled mesh is smaller than 0.5 cm or larger than 50 cm across, which
  usually means the wrong scale factor;
- no region array was chosen, so every element is written as 0 and the mesh's
  material regions are left out of the file;
- the fibres are the placeholder, or a fibre field holds the same direction on
  every element, which is a placeholder in disguise;
- the highest region tag is above 255, which older CARP versions cannot hold
  because they store the tag in an unsigned char;
- a sheet direction is not perpendicular to its fibre.

## Worked example

With `examples/AJR5d03295_myo_scar_lge_masked_lvonly.vtk` loaded and a
`tissueTag` assigned (see [Tissue properties](tissue-properties.md)):

1. **Export → Carp…**, choose a directory, name it `ventricle`.
2. Leave the scale factor at 1000: the mesh is in millimetres.
3. Set **region tag** to `tissueTag` and **fibres** to `fiber`.

That writes 66,819 nodes and 290,474 `Tt` elements, with the four tags from
the tissue assignment listed in the status bar, and the fibres taken from the
mesh.
