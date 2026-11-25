# Tag names

**Actions → Tag names…** says what each element tag (`elemTag`) of the mesh
is: the left ventricle, the left atrial body, a vein ring, a valve, and so
on. The legend then shows these names instead of the numbers, and actions
read each tag's chamber and role to find the tissue they work on.

## Why tags need names

Tag numbers are not the same in every mesh:

| Mesh | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| ccdaf atrial tagging | LA body | RA body | | |
| Ventricular volume | LV | RV | | |
| A four-chamber cohort | LV | RV | LA wall | RA wall |

The number 1 alone cannot say which tissue it is. Even two four-chamber
cohorts from the same group differ:

**What tag 18 is in two four-chamber cohorts**

| Cohort | Tag 18 |
|---|---|
| one cohort (`tags_lvrv.json`) | LAA ring |
| another cohort | LSPV ring |

The tag names give the answer once, for each mesh, or once for a whole
cohort with a saved file.

## The dialog

The dialog has one row for each tag in the mesh:

**Columns of the Tag names dialog**

| Column | Shows |
|---|---|
| **Tag** | the number, with its legend colour |
| **Elements** | how many elements carry it |
| **Name** | the name: pick one from the list, or type your own |
| **Chamber** | the chamber the tag belongs to: LV, RV, LA, RA, two of them (such as "LV+LA" for the mitral valve), or none |
| **Role** | what kind of tissue or surface it is (see below) |

Picking a name from the list fills in its chamber and role. You can change
them, for example to keep a ring as appendage tissue when the appendage was
clipped. A name you type yourself starts with no chamber and no role:
actions leave that tag alone until you set them. Leave a name blank to show
the number.

**Roles and what actions do with each**

| Role | Meaning | Actions |
|---|---|---|
| wall | myocardium | keep it as tissue of its chamber |
| layer | wall tissue with its own conduction (a fast endocardial layer) | keep it, as wall |
| vein | a vein, or the ring of vein tissue left where it was clipped | keep it, as that vein |
| appendage | the appendage, or its ring when it was clipped | keep it, as the appendage |
| cap | a disc that closes a vein or the appendage | remove it; its edge is an opening |
| valve | a valve plane | remove it; its edge is the valve rim |
| vessel | the aorta or the pulmonary artery | remove it |
| other | anything else, such as the fibrous plane between atria and ventricles | leave it alone |

The list holds ccdaf's own names (LV, RV, the left and right atrial
regions) and the regions of four-chamber meshes: the fast endocardial
layers (FEC_LV, FEC_RV, FEC_SV), the interatrial tissue (BB), the vein and
appendage rings and caps, the valves, the aorta (Ao), the pulmonary artery
(PArt) and the AV plane. Each has its own colour; a ring takes the colour
of the vein it belongs to.

**Proposed names** fills in what ccdaf can tell by itself:

- a tagging in ccdaf's own atrial labels names itself (1 is "LA body", 11 is
  "LSPV", …);
- a volume with exactly two tags and labelled surfaces (see
  [Surface labels](surface-labels.md)) gets LV and RV: the LV is the tag whose
  elements line the LV endocardium, the same rule UVC use.

Anything else stays blank until you name it.

**Save…** writes the names to a JSON file. **Load…** reads one back, so you
can name the first mesh of a cohort and reuse the file on the others. Load
also reads a cohort's own `{name: number}` file, such as `tags_lvrv.json`,
and translates its names (see [File formats](../file-formats.md)). Tags the
file does not list keep their names.

Until you press OK, the legend uses the proposed names. Only names you
confirm with OK are saved with the mesh.

## The legend

With names, the **Regions** legend shows "LV" and "RV" instead of 1 and 2.
A known name keeps its own colour: the ventricles a light orange and a light
blue, every atrial region its usual colour, and each four-chamber region a
colour of its own. Other names, and tags with no name, take colours from the
general palette, which has 30 colours so that a four-chamber mesh shows
every tag apart.

A mesh in ccdaf's atrial labels keeps its full atrial legend, which also lists
the regions not yet tagged, as long as the names agree with those labels.

The **surfaceLabel** field is named too: base, epicardium, LV endocardium and
RV endocardium instead of 1 to 4, each in a fixed colour.

## Saving

The names are saved with the mesh: in a `.vtk` file as field data, and in a
`.pkl` bundle under their own key (see [File formats](../file-formats.md)).
They reload with it.
