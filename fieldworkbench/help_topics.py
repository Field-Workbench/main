"""Built-in user documentation for Field Workbench."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QLineEdit,
    QSplitter,
    QTextBrowser,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .window_utils import enable_standard_window_controls
from .theme import current_theme

_TOPIC_ROLE = Qt.ItemDataRole.UserRole
_HELP_ASSET_DIRECTORY = Path(__file__).resolve().parent / "assets" / "help"
_BLENDER_SURFACE_REGION_EXPORTER = _HELP_ASSET_DIRECTORY / "blender_surface_regions_export.py"


_HELP_CSS = """
<style>
body { color: #1f2937; font-family: sans-serif; font-size: 10pt; line-height: 1.42; }
h1 { color: #0f172a; font-size: 20pt; margin: 0 0 8px 0; }
h2 { color: #0f4f94; font-size: 15pt; margin: 18px 0 7px 0; }
h3 { color: #334155; font-size: 12pt; margin: 14px 0 5px 0; }
p { margin: 5px 0 9px 0; }
ul, ol { margin-top: 4px; margin-bottom: 10px; }
li { margin-bottom: 4px; }
code { background: #eef2f7; color: #172033; padding: 1px 4px; font-family: monospace; }
table { border-collapse: collapse; width: 100%; margin: 8px 0 14px 0; }
th { background: #eaf2fb; color: #163a5f; text-align: left; }
th, td { border: 1px solid #cbd5e1; padding: 6px 7px; vertical-align: top; }
.note { background: #eff6ff; border-left: 4px solid #1769c2; padding: 8px 10px; margin: 10px 0; }
.warning { background: #fff7ed; border-left: 4px solid #c75b12; padding: 8px 10px; margin: 10px 0; }
.formula { background: #f8fafc; border: 1px solid #d7dee8; padding: 7px 9px; margin: 7px 0; font-family: monospace; }
.small { color: #64748b; font-size: 9pt; }
.figure { background: #f8fafc; border: 1px solid #cbd5e1; padding: 8px; margin: 10px 0 14px 0; }
.figure img { display: block; margin: 0 auto 7px auto; }
.caption { color: #475569; font-size: 9pt; margin: 4px 2px 0 2px; }
.term { color: #0f4f94; font-weight: 600; }
</style>
"""

_DARK_HELP_CSS = """
<style>
body { background: #2a3037; color: #e4e8ec; font-family: sans-serif; font-size: 10pt; line-height: 1.42; }
h1 { color: #d9dee3; font-size: 20pt; margin: 0 0 8px 0; }
h2 { color: #6299be; font-size: 15pt; margin: 18px 0 7px 0; }
h3 { color: #cbd2d8; font-size: 12pt; margin: 14px 0 5px 0; }
p { margin: 5px 0 9px 0; }
ul, ol { margin-top: 4px; margin-bottom: 10px; }
li { margin-bottom: 4px; }
code { background: #252b31; color: #e4e8ec; padding: 1px 4px; font-family: monospace; }
table { border-collapse: collapse; width: 100%; margin: 8px 0 14px 0; }
th { background: #344c5e; color: #ffffff; text-align: left; }
th, td { border: 1px solid #48535e; padding: 6px 7px; vertical-align: top; }
.note { background: #344c5e; border-left: 4px solid #3d7197; padding: 8px 10px; margin: 10px 0; }
.warning { background: #292f35; color: #e6ad63; border-left: 4px solid #e6ad63; padding: 8px 10px; margin: 10px 0; }
.formula { background: #252b31; border: 1px solid #48535e; padding: 7px 9px; margin: 7px 0; font-family: monospace; }
.small { color: #9ca7b1; font-size: 9pt; }
.figure { background: #252b31; border: 1px solid #48535e; padding: 8px; margin: 10px 0 14px 0; }
.figure img { display: block; margin: 0 auto 7px auto; }
.caption { color: #9ca7b1; font-size: 9pt; margin: 4px 2px 0 2px; }
.term { color: #6299be; font-weight: 600; }
</style>
"""


def _themed_help_html(source: str, theme: str | None = None) -> str:
    selected = current_theme() if theme is None else str(theme).strip().lower()
    if selected == "dark":
        return source.replace(_HELP_CSS, _DARK_HELP_CSS)
    return source


def _page(title: str, body: str) -> str:
    return f"<!doctype html><html><head>{_HELP_CSS}</head><body><h1>{title}</h1>{body}</body></html>"


HELP_TOPICS: dict[str, tuple[str, str, str]] = {
    "overview": (
        "Introduction",
        "Overview",
        _page(
            "Field Workbench overview",
            """
<p>Field Workbench is a scene-based magnetic design and analysis application built around
Magpylib and Magpylib Studio. A scene can contain current sources, permanent magnets,
uniform background fields, sensors, passive geometry, measurement volumes, exclusion zones,
and generated physical coil construction.</p>
<h2>What the application separates</h2>
<table>
<tr><th>Layer</th><th>Purpose</th></tr>
<tr><td>Electromagnetic scene</td><td>Sources and sensors used to calculate <b>B</b> or <b>H</b>.</td></tr>
<tr><td>Physical coil specification</td><td>Turns, wire, winding dimensions, bobbin, mass, resistance, voltage, and power estimates.</td></tr>
<tr><td>Passive geometry</td><td>Visual objects, exclusion zones, and measurement volumes. These do not create a magnetic field.</td></tr>
<tr><td>Analysis results</td><td>2D/3D maps, sensor and magnetometer-probe readings, Brain Areas statistics, measurement-volume statistics, snapshots, comparisons, and optimizer finalists.</td></tr>
</table>
<div class="note"><b>Important:</b> the visible bobbin and winding envelope are electrically inert.
They are used for visualization and design-rule checking. The selected coil field model controls
the electromagnetic representation.</div>
<p>Specialized terms are defined where they first matter. The <b>Glossary</b> topic collects the
most common terms, and Help search includes both topic text and definitions.</p>
""",
        ),
    ),
    "workflow": (
        "Introduction",
        "Typical workflow",
        _page(
            "Typical workflow",
            """
<ol>
<li>Add or load coils, magnets, sensors, and geometry. Add a Background field when the apparatus operates inside a known static bias such as the Earth's field.</li>
<li>Position and rotate objects in the scene.</li>
<li>For coils, enter turns, drive current, wire information, winding width/build, and optional bobbin dimensions.</li>
<li>Select <b>Edit → Preferences</b> to choose the Light/Dark theme, coil field model, solver temporary-memory budget, worker concurrency, and optional solver diagnostics. Theme changes apply immediately to open windows.</li>
<li>Use Sensor Plot for scene sensors, calculate magnetometer-probe channels in the inspector, open a 2D/3D field-map window, or use Brain view for LPBA40 regional analysis.</li>
<li>Assign a closed object the Measurement volume role to calculate volume statistics and save snapshots.</li>
<li>Assign anatomy or keep-out geometry the Exclusion zone role and use <b>Check design</b>.</li>
<li>Save the scene as a <code>.magpy.json</code> document.</li>
</ol>
<p>Map windows deliberately open before calculating. This lets you choose the plane, extent,
resolution, component, and display options before committing to an expensive field evaluation.</p>
""",
        ),
    ),
    "limitations": (
        "Introduction",
        "Limitations and model boundaries",
        _page(
            "Limitations and model boundaries",
            """
<p>Field Workbench is primarily a <b>magnetostatic / quasi-static magnetic-field design tool</b>.
Its results are useful only when the physical system is well represented by the source models and
approximations below. The program does not infer whether those assumptions are valid for a particular
experiment; that remains an engineering judgement and should be checked against measurement or an
independent solver when the distinction matters.</p>
<h2>Frequency and time dependence</h2>
<ul>
<li>The magnetic field solver itself has <b>no frequency input and no built-in maximum frequency</b>. It evaluates the static B or H field produced by the instantaneous source currents.</li>
<li>Waveform playback and video animation change source current with time, but each displayed frame is still a quasi-static field solution. The field solver does not calculate propagation delay, radiation, phase lag through space, or electric fields induced by the changing magnetic field.</li>
<li>The optional waveform electrical model may use entered resistance and self-inductance to estimate current response. It does not derive inductance from scene geometry and does not solve mutual inductance between coils.</li>
<li>Skin effect, proximity effect, parasitic capacitance, frequency-dependent conductor impedance, resonances, and transmission-line/full-wave effects are outside the model.</li>
</ul>
<div class="warning"><b>There is no universal frequency cutoff.</b> Quasi-static validity depends on apparatus size, conductor/material properties, surrounding conductive structures, and the accuracy required. As frequency rises or induced-current/material effects become significant, validate with measurement or an appropriate electromagnetic finite-element/full-wave/transient solver.</div>
<h2>Iron, ferrite, shielding, and magnetic materials</h2>
<ul>
<li>Workbench does <b>not</b> solve ferromagnetic cores, ferrite cores, magnetic shielding, or the permeability of imported passive geometry.</li>
<li>Imported meshes, anatomy, bobbins, coil formers, and other passive construction are magnetically inert in the Workbench field solution even if the real object would be conductive or magnetic.</li>
<li>No nonlinear B-H curves, saturation, hysteresis, domain response, or core losses are modeled.</li>
<li>Permanent-magnet source objects use their prescribed magnetization/polarization model; they are not a substitute for solving a nonlinear iron-core magnetic circuit.</li>
<li>Eddy currents and induced magnetization in nearby conductors or magnetic materials are not solved.</li>
</ul>
<h2>Coil-source approximations</h2>
<ul>
<li><b>Centreline</b> treats a complete winding as an ideal current path carrying the effective ampere-turn current.</li>
<li><b>Exact turns</b> places one ideal centreline at each calculated turn position. "Exact" therefore means exact turn placement within this filament model, not finite copper cross-section or microscopic current-density physics.</li>
<li><b>Bundled</b> and <b>Current sheet</b> reduce the physical winding to lower-cost representative sources. Auto performs sparse checks against Exact turns when practical, but that validation is an error estimate over sampled points rather than a proof everywhere.</li>
<li>Visible winding packs, insulation, bobbins, flanges, and hardware are primarily construction/DRC geometry; they do not themselves modify the field unless represented by the selected coil source model.</li>
</ul>
<h2>Sampling and numerical interpretation</h2>
<ul>
<li>2D maps, 3D maps, Brain view, measurement volumes, sensor paths, optimizer targets, and comparisons evaluate a finite set of observer points. Unsampled space is not mathematically guaranteed to satisfy the displayed limits.</li>
<li>Interpolated colour maps and rendered fluxlines are visualization aids. Fluxline seed density is not magnetic flux density.</li>
<li>Optimizer results are solutions of the chosen sampled objective, parameter bounds, solver model, and search effort. They are not proofs of a unique or global physical optimum.</li>
<li>Imported geometry is only as trustworthy as its scale, topology, coordinate frame, and alignment.</li>
</ul>
<h2>Electrical and thermal scope</h2>
<ul>
<li>Wire length, DC resistance, resistive voltage, power loss, current density, and mass are engineering estimates derived from the entered winding data.</li>
<li>Workbench does not perform coupled thermal simulation, insulation-temperature prediction, mechanical stress analysis, or automatic safety certification.</li>
</ul>
<div class="note"><b>Recommended final check:</b> for a critical design, increase spatial sampling, compare simplified coil models with Exact turns where practical, verify the built geometry with physical measurements, and use a material-aware electromagnetic solver whenever conductive or ferromagnetic surroundings could materially change the result.</div>
""",
        ),
    ),
    "portable_export": (
        "Introduction",
        "Portable scene export",
        _page(
            "Portable scene export",
            """
<p><b>File → Export…</b> exports selected scene objects through a small, explicit
<span class="term">portable scene</span> representation. This is deliberately separate from the
native <code>.magpy.json</code> project file: editor state, optimizer history, plots, and other
Workbench-only data do not become dependencies of every external translator.</p>
<h2>Export window</h2>
<ul>
<li>Choose a translator, then check the objects that should be exported. The hierarchy is shown
for selection convenience; Collection/Group rows are containers rather than physical solids.</li>
<li><b>Select visible</b> selects supported objects that are currently visible. Unsupported objects
remain listed with the reason they cannot be sent to the selected translator.</li>
<li><b>Portable scene package (.fwpkg)</b> writes a ZIP package containing
<code>manifest.json</code> and <code>scene.json</code>. The scene uses explicit millimetre transforms
and stores physically meaningful source/geometry metadata.</li>
<li><b>OpenSCAD (.scad)</b> writes physical geometry. Coil current, turn count, current convention,
and empirical field-scale correction are retained as comments and portable-scene metadata rather
than becoming OpenSCAD physics.</li>
<li><b>FEMM axisymmetric builder (.lua)</b> writes a FEMM Lua script for magnetostatic, rotationally
symmetric coil systems. Select any circular coil or coaxial set: Workbench automatically removes
the source rigid-body translation/rotation and maps the selected common coil axis onto FEMM's
canonical Z axis. The Workbench scene itself is not modified. Non-coaxial or non-parallel circular
selections are rejected rather than approximated. Each winding becomes an axisymmetric bulk copper
region carrying the saved turns and drive current, with an open-space IABC boundary. Passive
<code>workbench.Mesh</code> objects may also be selected as <b>reference outlines only</b>. Choose
Auto, XY, XZ, or YZ for the source-world reference-slice orientation. The plane offset is
derived automatically from the selected coils' common axis, so the reference plane follows the
axis without manual tuning. Workbench intersects each selected passive solid with that
<em>literal</em> source plane; it no longer rotates the request into a different meridional section.
Because FEMM only represents r &ge; 0, <b>Export half</b> chooses which source-world side of the axis
is retained and folded onto FEMM +r. The resulting 2D outline is only rigidly rotated/translated
for the FEMM display, preserving its source-slice shape. The Export dialog also shows a live 3D
plane preview plus a 2D Slice tab generated by the same cross-section core as the 2D viewer; the
Slice preview marks the common axis and the selected FEMM +r direction. Reference-only guides
are deliberately kept 0.25 mm clear of FEMM's r=0 axis, and closed guide contours receive a
0.25 mm break after simplification. A final topology sanity pass removes duplicate edges and opens
any remaining endpoint-graph reference cycle. The generated model also marks its Air label as FEMM's default block label, so any region that FEMM itself creates from guide/boundary intersections inherits Air instead of becoming an unlabeled material region.</li>
<li><b>Gmsh/GetDP 3D magnetostatic project (.getdp.zip)</b> writes a complete editable project
rather than a frozen mesh. Version 8 accepts multi-coil selections of circular, Square/rectangular, and/or parametric
racetrack coils (up to 64 sources per project) plus optional Linear axis sensors. The archive contains matched <code>.geo</code> and <code>.pro</code>
sources, one-command Linux and Windows runners, the normalized portable scene, project metadata,
documentation, and sample coordinates with centreline references where applicable. The runner asks
Gmsh to create an ASCII MSH 2.2 tetrahedral mesh, solves the linear 3D magnetic-vector-potential
formulation in GetDP, then writes magnetic flux density in microtesla, source current density, and
smoothed values at every sample point. The smoothed verification samples reduce element-local noise
from the lowest-order H(curl) field. This conservative output targets Gmsh 4.8.4 and GetDP 3.2.0 or
later. Circular windings use annular stranded-current volumes. Square/rectangular coils use exact mitered
path-band volumes with a current-conserving tangent through the four corners. Racetracks use a continuous swept
capsule-band winding with a tangent field that follows both straight sections and semicircular ends.
The Export dialog exposes <b>Use saved physical winding construction when available</b>. With it enabled, valid saved winding-pack dimensions become the stranded-current
region; coils without construction still receive a compact numerical pack derived from turns, finished-wire diameter, and packing factor. Turn it off to force the numerical surrogate for every selected coil, which is useful for idealized cross-checks against Workbench centrelines. In both cases the uniform tangent-current magnitude
integrates to the selected signed ampere-turns. Square/rectangular and racetrack sample CSVs also include references from
the same Polyline centrelines used by Field Workbench. Every GetDP export now uses the same
mathematically unbounded exterior: FWB automatically sizes an ordinary-air sphere around all selected
sources and requested samples, surrounds it with a modest finite spherical shell, and assigns GetDP's
<code>VolSphShell</code> Jacobian to that shell so its outer A=0 surface represents infinity rather than
a large artificial finite boundary. Mesh quality is selectable as <b>Fast</b>, <b>Standard</b>, or
<b>Validation</b>; these presets change winding, source/sample-neighbourhood, and far-air discretization.
Standard is deliberately memory-conscious for representative 16 GB-class systems, while Validation is
allowed to target a modern higher-memory workstation; neither changes the open-boundary physics. For
Validation-scale winding refinement, FWB automatically adds a distance-based source-to-air grading field.
The winding geometry keeps its fine source target, while the surrounding air starts at a separately derived
interface size (bounded by the near-air target) and grows smoothly outward. This avoids both an abrupt
sub-millimetre-to-millimetre transition and the pathological over-refinement that would result from imposing
the winding's thinnest characteristic length across every winding surface. Validation also switches the 3-D
mesher to Gmsh HXT and lets it use the system thread count; HXT is Gmsh's parallel Delaunay implementation
and is better suited to these strongly graded large meshes. Fast and Standard retain their lighter accepted
Delaunay mesh behaviour. Advanced controls expose only the high-value local mesh settings. Users
who need solver- or mesher-specific tuning can edit the generated <code>.geo</code> and <code>.pro</code>
files directly in Gmsh/GetDP. The dialog previews source/near/far/shell characteristic lengths and the
automatically derived inner/outer shell radii before export. Generated solve commands also request PETSc
to fail explicitly when a linear solve does not converge, preventing invalid NaN result files from being
mistaken for a successful run.</li>
<li><b>COMSOL 6.4 Java builder (.java)</b> is an experimental 3D magnetic translator. Circular and
planar-path coils become closed 3D edge loops and are excited with Magnetic Fields Edge Current
features. The exported edge current is the saved drive current multiplied by turn count, so the
ideal line loop represents the coil ampere-turns. The builder creates a surrounding free-space
sphere, mesh, and stationary-study scaffold but deliberately does not solve automatically.</li>
</ul>
<h2>Magnetic excitation</h2>
<p>FEMM, GetDP, and COMSOL exports expose <b>Apply empirical Field correction to exported magnetic
excitation</b>. It is off by default so an independent finite-element solver can be used as a
cross-check of the underlying physical model. Enable it only when the intended exported model is
the empirically calibrated Workbench model; the saved correction then multiplies the exported
excitation.</p>
<h2>Coil construction</h2>
<p>If a coil has physical construction enabled, the OpenSCAD translator uses the same winding-pack,
bobbin-barrel, and flange solids used by Workbench visualization and DRC. If construction is not
enabled, the translator emits a thin centreline reference so the source placement remains visible
without inventing a bobbin. GetDP uses only the conductive winding pack, not bobbin/flange solids:
the GetDP-specific checkbox chooses whether a valid saved winding pack should be used as the finite
stranded-current region or whether every selected coil should use a compact numerical surrogate.</p>
<div class="note"><b>Magnetic translator boundaries:</b> FEMM is intentionally limited to
axisymmetric circular-coil systems, but those systems may be positioned and oriented arbitrarily
in the Workbench scene because export normalizes their common axis automatically. Passive FEMM
reference outlines are deliberately <em>not</em> material or magnetic regions. Workbench first
stitches and deduplicates the raw triangle-mesh intersection into clean contour guides, removes only
microscopic tessellation slivers/near-collinear points, and leaves one tiny gap in each closed contour
so it cannot become an unlabeled FEMM region. The FEMM physics geometry, air label, and asymptotic
boundary are created before these guides are appended, and no segment-selection/grouping pass is used
for the imported outlines. Because the guides are still FEMM input segments, they can slightly alter
triangulation if you solve with them present. GetDP v10 validates multi-coil selections of circular, Square/rectangular, and/or parametric racetrack sources in the same 3D solve;
arbitrary non-rectangular Polyline winding volumes remain outside the current scope. It is linear magnetostatic only: conductivity, eddy currents, nonlinear
materials, and time/frequency-domain physics are not silently inferred. COMSOL currently exports ideal current-loop coils only; permanent
magnets, passive solids, sensors, dipoles, and Background fields are not yet translated into COMSOL
physics. The COMSOL Java output is generated against 6.4 API conventions and should be inspected in
COMSOL before relying on a solution.</div>
<div class="note"><b>Schema v1:</b> the portable representation is intended to be the common
input for additional translators. New exporters should consume this normalized scene rather than
parsing the complete Workbench save document.</div>
""",
        ),
    ),
    "coordinates": (
        "Coordinates and anatomy",
        "Coordinates and units",
        _page(
            "Coordinates and units",
            """
<p>The user interface presents geometry in millimetres, angles in degrees, magnetic flux density
in microtesla (µT), current in amperes, resistance in ohms, voltage in volts, power in watts,
and mass in grams. Internally, Magpylib geometry and observer coordinates use metres and field
vectors are returned in SI units before display conversion.</p>
<h2>Local and world coordinates</h2>
<ul>
<li><span class="term">Local coordinates</span> belong to an object before its scene position,
rotation, and scale are applied.</li>
<li><span class="term">World coordinates</span> are the final shared scene coordinates after
those object transforms are applied.</li>
<li><span class="term">Origin</span> means the point whose coordinates are zero. An object's
local origin and the scene/world origin coincide only while its position is <code>(0, 0, 0)</code>.</li>
</ul>
<h2>Vector quantities</h2>
<ul>
<li><b>Bx, By, Bz</b> are world-coordinate components of magnetic flux density.</li>
<li><b>|B|</b> is the Euclidean magnitude: <code>sqrt(Bx² + By² + Bz²)</code>.</li>
<li>Rotating a source changes the direction of its field in world coordinates.</li>
<li>Sensor pixel coordinates are local to the sensor and are transformed with the sensor pose.</li>
</ul>
<p>Position and rotation are independent of the construction dimensions. A winding width or
bobbin flange changes the modeled winding envelope, not the object's scene origin.</p>
""",
        ),
    ),
    "head_frame": (
        "Coordinates and anatomy",
        "NAS/LPA/RPA head frame",
        _page(
            "Canonical NAS/LPA/RPA head frame",
            """
<p>The Generic bust, SRI24 brain, and imported UPENN-GBM tumors share one
<span class="term">head frame</span>: a reproducible three-dimensional coordinate system fixed
to anatomical landmarks. Using one frame lets anatomy and coil positions be described without
manually aligning every new object.</p>
<h2>The three landmarks</h2>
<table>
<tr><th>Abbreviation</th><th>Meaning</th><th>Placement used here</th></tr>
<tr><td><b>NAS</b></td><td><span class="term">Nasion</span>, the midline depression at the root
of the nose where the frontal and nasal bones meet.</td><td>The external nasion.</td></tr>
<tr><td><b>LPA</b></td><td>Left preauricular point: a landmark immediately in front of the left
ear.</td><td>The left helix–tragus junction.</td></tr>
<tr><td><b>RPA</b></td><td>Right preauricular point: the corresponding landmark immediately in
front of the right ear.</td><td>The right helix–tragus junction.</td></tr>
</table>
<p>The <span class="term">helix</span> is the outer rim of the ear. The
<span class="term">tragus</span> is the small cartilaginous projection in front of the ear canal.
The junction provides a visually repeatable external point, although other EEG/neuroimaging
conventions can define preauricular points somewhat differently.</p>
<h2>Origin and positive axes</h2>
<table>
<tr><th>Element</th><th>Definition</th></tr>
<tr><td>Origin</td><td>The perpendicular projection of NAS onto the line from LPA to RPA.</td></tr>
<tr><td>+X</td><td>Subject/model right, along LPA→RPA.</td></tr>
<tr><td>+Y</td><td>Anterior (toward the face/NAS), made perpendicular to +X.</td></tr>
<tr><td>+Z</td><td>Superior (toward the top of the head), completing a right-handed frame.</td></tr>
</table>
<p>A <span class="term">right-handed frame</span> means the positive axes follow the standard
right-hand rule: +X crossed with +Y points along +Z. In this convention, negative X is left,
negative Y is posterior, and negative Z is inferior.</p>
<div class="note"><b>Screen side is not anatomical side:</b> when looking at a subject from the
front, the subject's right side appears on the left side of the screen. LPA and RPA always name
the subject/model sides.</div>
""",
        ),
    ),
    "well_plates": (
        "Objects",
        "Built-in well plates",
        _page(
            "Built-in cell-culture well plates",
            """
<p><b>Object → 6/12/24/48/96-well plate</b> adds a ready-to-analyze cell-culture plate.
Each built-in plate uses the established Field Workbench experimental envelope of
<b>84 × 126 × 15 mm</b>: local X is the short plate side, local Y is the long side, and
local Z is plate thickness.</p>
<h2>Exact well-centre samples</h2>
<p>The plate is created directly as a <b>Measurement volume</b>, but its sampling mode is
<b>User defined points</b>. The stored samples are the exact well centres in the local Z=0
plane rather than a generated voxel grid. Moving or rotating the plate therefore carries the
whole well grid with it. These same discrete points are used by volume analysis, snapshots,
comparison, and optimizer targets.</p>
<table>
<tr><th>Format</th><th>Grid</th><th>Centre pitch</th><th>Representative well opening</th></tr>
<tr><td>6 well</td><td>2 × 3</td><td>39.12 mm</td><td>35.43 mm</td></tr>
<tr><td>12 well</td><td>3 × 4</td><td>26.01 mm</td><td>22.73 mm</td></tr>
<tr><td>24 well</td><td>4 × 6</td><td>19.30 mm</td><td>16.26 mm</td></tr>
<tr><td>48 well</td><td>6 × 8</td><td>13.08 mm</td><td>11.56 mm</td></tr>
<tr><td>96 well</td><td>8 × 12</td><td>9.00 mm</td><td>6.96 mm</td></tr>
</table>
<p>The 6–48 well pitch/opening values use representative Corning cell-culture plate dimensions;
the 96-well grid uses the conventional 9 mm pitch. The rendered well circles are a convenient
visual reference, not a claim that every manufacturer's well wall and bottom geometry is
identical.</p>
<div class="note"><b>Why the envelope is 84 × 126 × 15 mm:</b> this intentionally matches the
plate abstraction already used by the RILZ/FRILZ experimental scenes and source paper rather
than silently changing those analyses to a manufacturer-specific outer height.</div>
<p>Rows run across local X and are labelled A, B, C…; numbered columns run along local Y.
The well-centre markers visible in the 3D scene are the actual measurement sample locations.
You can still edit the User defined points list in the inspector for a custom plate.</p>
""",
        ),
    ),
    "generic_bust": (
        "Generic bust",
        "Built-in Generic bust",
        _page(
            "Built-in Generic bust",
            """
<p><b>Object → Generic bust</b> adds a packaged human head/neck/shoulder reference mesh for
visualization, design-rule exclusion, measurement, snapshots, and optimizer geometry.</p>
<p>A <span class="term">triangle mesh</span> represents a surface with connected triangular
faces. <span class="term">STL</span> is the mesh-file format used for the packaged object; it
stores surface triangles rather than medical image voxels or tissue labels.</p>
<h2>Mesh provenance</h2>
<p><code>generic_bust_v1.stl</code> is derived from Blender's <b>Human Base Meshes v1.4.1</b>
asset bundle, specifically the <b>Body Female - Realistic</b> base mesh by Blender Studio and
community contributors. Blender distributes the Human Base Meshes bundle under
<b>CC0 1.0 Universal</b>. CC0 does not require attribution, but Field Workbench records the
source in <code>THIRD_PARTY_NOTICES.txt</code> for provenance and acknowledgement.</p>
<p>The source body was cropped to retain the head, neck, shoulders, upper torso, and short upper
arms. Cut surfaces were closed; facial/eye volumes were consolidated; the result was voxel
remeshed and decimated; and the packaged mesh was checked as a watertight, manifold,
consistently oriented surface suitable for inside/outside geometry tests.</p>
<ul>
<li><span class="term">Watertight</span> means the surface has no boundary holes.</li>
<li><span class="term">Manifold</span> means each surface edge has the expected local
neighbourhood—normally exactly two adjoining faces for a closed mesh.</li>
<li><span class="term">Consistently oriented</span> means face directions/normals agree on which
side is outside.</li>
<li><span class="term">Decimation</span> reduces triangle count while attempting to preserve the
overall shape.</li>
</ul>
<h2>Authored surface regions</h2>
<p>The packaged bust also has a companion <code>generic_bust_v1.regions.json</code> file. The
regions were authored in Blender by assigning mesh faces to material slots and exporting those
face assignments without splitting the watertight bust into separate objects. The current map
contains five named regions:</p>
<table>
<tr><th>Region</th><th>Triangles</th><th>Current purpose</th></tr>
<tr><td>Scalp</td><td>9,106</td><td>Placement + DRC exclusion (default)</td></tr>
<tr><td>Face</td><td>8,117</td><td>DRC exclusion (default)</td></tr>
<tr><td>Ear L</td><td>1,103</td><td>DRC exclusion (default)</td></tr>
<tr><td>Ear R</td><td>1,167</td><td>DRC exclusion (default)</td></tr>
<tr><td>Neck/Shoulders</td><td>18,939</td><td>DRC exclusion (default)</td></tr>
</table>
<p>Together the regions account for all 38,432 bust triangles exactly once. Because version 1
region files identify triangles by ordered face index, Field Workbench verifies the fixed STL
identity, mesh counts, and its own ordered geometry digest before enabling region highlighting.
The exporter-provided authoring fingerprint is retained as provenance as well.</p>
<p>Select a Generic bust and use <b>Surface regions</b> in the inspector to configure each authored
region independently. <b>Colour</b> controls the overlay colour; <b>Show</b> is display-only and keeps
the coloured overlay visible; selecting a row gives a temporary preview even when Show is off.
<b>Active</b> is the master semantic switch. <b>Place</b> and <b>DRC</b> are independent capabilities,
so one region can be both a future placement surface and a physical exclusion surface. Region
settings are saved per bust object and survive normal save/open, duplicate, import, undo, and redo.</p>
<p>Each DRC-enabled region inherits the object's clearance override when one is enabled, otherwise
it inherits the scene-wide exclusion clearance. A region can override that inherited value with its
own clearance. The built-in defaults make Scalp active for both placement and DRC; Face, both ears,
and Neck/Shoulders are active for DRC only.</p>
<p>Regional DRC separates <b>penetration/containment</b> from <b>local clearance</b>. Whenever any
region on the bust has active DRC, the complete watertight bust remains the physical shell used to
prevent a coil from crossing or hiding inside the body. Named region triangle patches then apply
their individual clearance distances. If the whole bust is also assigned the Exclusion role, any
faces not covered by an active regional DRC rule retain the base object clearance.</p>
<div class="note"><b>Topology is part of the region definition:</b> do not remesh, decimate,
re-triangulate, or re-export <code>generic_bust_v1.stl</code> independently of its region sidecar.
Those operations can change face ordering even when the visible shape appears unchanged.</div>
<div class="warning"><b>Engineering reference, not clinical anatomy:</b> the Generic bust is a
generic external-surface model. It is not patient-specific anatomy, a diagnostic model, or a
medical anatomical atlas.</div>
<h2>Placement in the head frame</h2>
<p>The source bust was translated and rotated using NAS, LPA, and RPA as described in the
separate <b>NAS/LPA/RPA head frame</b> topic. No presentation tilt is applied.</p>
<p>New Generic bust objects are inserted at scene position <code>(0, 0, 0)</code>, rotation
<code>(0°, 0°, 0°)</code>, and scale <code>(1, 1, 1)</code>. Therefore the object's scene/world
coordinates initially match this anatomical head frame exactly. If the bust is later moved or
rotated, its local mesh frame remains anatomically defined, while world-coordinate positions
reflect the explicit scene transform.</p>
<div class="note">Do not use an import or editing operation that automatically recentres the
mesh if preservation of the anatomical origin is required.</div>
""",
        ),
    ),
    "sri24_brain": (
        "SRI24 brain",
        "Built-in SRI24 brain",
        _page(
            "Built-in SRI24 brain",
            """
<p><b>Object → SRI24 brain</b> adds a packaged atlas-derived brain surface for visualization,
design-rule exclusion, measurement, snapshots, and optimizer geometry. The object is aligned to
the same canonical NAS/LPA/RPA head frame as the Generic bust, so the two built-in anatomy
objects can be added together without a manual placement step.</p>
<h2>Atlas provenance</h2>
<p>An <span class="term">anatomical atlas</span> is a reference coordinate space and anatomy
model used to compare or combine different scans. A <span class="term">population atlas</span>
is derived from multiple people; it represents shared/average anatomy rather than one patient.</p>
<p><code>sri24_brain_gmwm_v1.stl</code> is derived from the <b>SRI24 v2.0 multichannel atlas
of normal adult human brain anatomy</b> by Torsten Rohlfing, Natalie M. Zahr, Edith V. Sullivan,
and Adolf Pfefferbaum. SRI24 is a population atlas generated from MRI of 24 normal control
subjects and is supplied at 1 mm <span class="term">isotropic</span> resolution, meaning equal
voxel spacing along all three image axes. The SRI24 project distributes the atlas
under <b>Creative Commons Attribution-ShareAlike 3.0 (CC BY-SA 3.0)</b>. The packaged derived
brain mesh remains under CC BY-SA 3.0; the Field Workbench application code remains
GPL-3.0-or-later. Full acknowledgement and licensing details are recorded in
<code>THIRD_PARTY_NOTICES.txt</code>.</p>
<p>Requested citation for work using SRI24: T. Rohlfing, N. M. Zahr, E. V. Sullivan, and
A. Pfefferbaum, “The SRI24 multichannel atlas of normal adult human brain structure,”
<i>Human Brain Mapping</i>, 31(5), 798–819, 2010. DOI 10.1002/hbm.20906.</p>
<h2>How the packaged brain surface was derived</h2>
<ol>
<li>SRI24 v2.0 <code>tissues.nii</code> was loaded in 3D Slicer. NIfTI (<code>.nii</code>) is a
medical/neuroimaging volume format; a <span class="term">voxel</span> is one 3D image cell.</li>
<li>The gray-matter (GM) and white-matter (WM) labels were unioned to form one brain segment.
Cerebrospinal fluid (CSF) was excluded so cortical sulci—the grooves between folds of the
cortex—remain visible rather than being filled into a smooth intracranial envelope.</li>
<li>A closed surface was generated in Slicer and exported explicitly in <b>RAS</b> coordinates.
The exported STL therefore retained the original SRI24 physical coordinates in millimetres.</li>
<li>The mesh was cleaned in Blender. A manifold remesh was used to eliminate fragmented /
non-manifold topology, the result was aggressively decimated while retaining the major
cortical geometry, and disconnected remnants were removed.</li>
<li>The packaged atlas-space STL contains approximately 12,855 vertices and 26,250 triangle
faces and was checked as a single connected, watertight, consistently oriented surface.</li>
</ol>
<div class="warning"><b>Atlas reference, not patient anatomy:</b> SRI24 is a population-average
normal-adult atlas. The built-in brain is not a patient-specific brain, diagnostic segmentation,
or treatment-planning model.</div>
<p>The separate <b>SRI24 registration and fiducials</b> topic documents the exact landmarks,
Slicer views, coordinates, and transform used to place this surface in the Workbench head frame.</p>
""",
        ),
    ),
    "sri24_registration": (
        "SRI24 brain",
        "SRI24 registration and fiducials",
        _page(
            "SRI24 registration and fiducials",
            """
<p><span class="term">Registration</span> maps coordinates from one reference frame into
another. Here it maps the SRI24 atlas into the Workbench NAS/LPA/RPA head frame. A
<span class="term">fiducial</span> is a precisely marked reference point used to construct or
check that mapping.</p>
<h2>RAS coordinates and rigid affine</h2>
<p><b>RAS</b> means the positive image axes point Right, Anterior, and Superior. Some medical
software instead uses <b>LPS</b> (Left, Posterior, Superior), which reverses the first two axis
directions. The SRI24 points below were recorded in Slicer's RAS coordinates.</p>
<p>An <span class="term">affine transform</span> is a 4 × 4 matrix that maps points between
coordinate systems. This registration is <b>rigid</b>: it contains rotation and translation only,
with no scaling, shearing, or non-rigid warping.</p>
<h2>Landmark placement in 3D Slicer</h2>
<p>The raw STL is deliberately stored in its original SRI24 RAS millimetre coordinates. To
register SRI24 to the Generic bust head convention, three external fiducials were marked on
SRI24 v2.0 <code>spgr_unstrip.nii</code> in 3D Slicer. NAS was placed at the nasion; LPA and
RPA were placed at the left/right helix–tragus junctions, matching the convention used for the
Generic bust.</p>
<table>
<tr><th>Fiducial</th><th>SRI24 RAS coordinate (mm)</th></tr>
<tr><td>NAS</td><td>(−120.066, +214.028, +50.000)</td></tr>
<tr><td>LPA</td><td>(−195.809, +113.054, +27.190)</td></tr>
<tr><td>RPA</td><td>(−36.738, +113.125, +27.439)</td></tr>
</table>
<div class="figure"><img src="help/sri24_fiducials_4.jpg" width="700"
alt="3D Slicer volume rendering with NAS, LPA, and RPA fiducials">
<p class="caption">Anterior 3D volume-rendering overview. Because the subject is viewed from
the front, RPA appears on the screen-left and LPA on the screen-right.</p></div>
<div class="figure"><img src="help/sri24_fiducials_1.jpg" width="700"
alt="Axial SRI24 slice showing LPA and RPA fiducials">
<p class="caption">Axial slice through the bilateral helix–tragus reference points. Axial means
a horizontal cross-section viewed along the superior/inferior direction.</p></div>
<div class="figure"><img src="help/sri24_fiducials_2.jpg" width="700"
alt="Coronal SRI24 slice showing the LPA fiducial">
<p class="caption">Coronal slice at the LPA placement. Coronal means a front-facing vertical
cross-section dividing anterior from posterior.</p></div>
<div class="figure"><img src="help/sri24_fiducials_3.jpg" width="700"
alt="Coronal SRI24 slice showing the RPA fiducial">
<p class="caption">Corresponding coronal slice at the RPA placement.</p></div>
<h2>Constructed transform</h2>
<p>The head origin is the projection of NAS onto the LPA→RPA line. +X runs from LPA toward
RPA (right), +Y runs anteriorly toward NAS and is orthogonal to +X, and +Z completes the
right-handed frame superiorly. This produces the following rigid affine, mapping SRI24 RAS
millimetres to canonical head-frame millimetres:</p>
<pre>
[ 0.999998675   0.000446341   0.001565337   119.89204464 ]
[-0.000778793   0.975651307   0.219326058  -116.41725317 ]
[-0.001429330  -0.219326990   0.975650360    -2.01201562 ]
[ 0             0             0                1          ]
</pre>
<p>The registration is <b>rigid only</b>: no scaling and no non-rigid deformation are applied.
The SRI24 brain therefore retains its atlas millimetre dimensions even though the SRI24 head
and Generic bust come from different generic anatomies and will not be a perfect individual
fit.</p>
<p>When a built-in SRI24 brain is added, Field Workbench loads the raw atlas-space STL, applies
this fixed affine, converts millimetres to SI metres, and inserts the resulting mesh at scene
position <code>(0, 0, 0)</code>, rotation <code>(0°, 0°, 0°)</code>, and scale
<code>(1, 1, 1)</code>. Its world coordinates therefore initially use the same head frame as a
new Generic bust.</p>
<div class="note"><b>Atlas targets:</b> the UPENN-GBM importer uses this same affine for each
preprocessed case segmentation, so imported tumors line up with the built-in brain and Generic
bust head frame without a manual placement transform.</div>
""",
        ),
    ),
    "upenn_gbm": (
        "UPENN-GBM cases",
        "Dataset and importer overview",
        _page(
            "UPENN-GBM dataset and importer",
            """
<p><b>UPENN-GBM</b> is a public research cohort of people diagnosed with de novo glioblastoma
(GBM). <span class="term">De novo</span> means newly arising rather than transformed from a
previous lower-grade tumor. <span class="term">De-identified</span> means direct identifiers
were removed before public release.</p>
<p>The collection includes multiparametric MRI (multiple MRI sequence types), clinical and
molecular information, standardized atlas-space images, and tumor-region annotations. Field
Workbench imports only the available atlas-space tumor <span class="term">segmentations</span>:
label volumes that classify voxels into tumor subregions.</p>
<p><b>Import UPENN-GBM case…</b> opens a searchable catalogue of subjects for which the public
per-file redistribution includes the collection's original expert/curated tumor segmentation
NIfTI. Select a case, one or more tumor regions, and a scene role, then choose
<b>Download &amp; add to scene</b>. The small mask is downloaded on demand and retained in the
application cache for later imports.</p>
<h2>Source and interpretation</h2>
<p>UPENN-GBM contains de-identified multiparametric MRI from 630 subjects with de novo
glioblastoma. The structural volumes were co-registered to the SRI24 common atlas, resampled to
1 mm isotropic resolution, and skull-stripped. <span class="term">Skull stripping</span> removes
non-brain structures from an image volume. The tumor subregions were produced computationally,
with 232 subjects evaluated and manually refined when needed. This importer bundles a snapshot
of the 147 original expert/curated segmentation files found during development and can rebuild
that list from the current per-file redistribution.</p>
<p>Collection: Bakas et al., <i>Scientific Data</i> 9, 453 (2022), DOI
10.1038/s41597-022-01560-7. TCIA collection DOI: 10.7937/TCIA.709X-DN49. Collection license:
CC BY 4.0. Full source URLs and attribution are recorded in
<code>THIRD_PARTY_NOTICES.txt</code> and on every imported object's scene metadata.</p>
<div class="warning"><b>Research geometry only:</b> atlas registration preserves each case's
tumor size and location in the shared SRI24 space, but the Generic bust and built-in brain are
generic anatomy. The result is not patient-specific scalp geometry and is not intended for
diagnosis, clinical navigation, or treatment planning.</div>
""",
        ),
    ),
    "upenn_catalogue": (
        "UPENN-GBM cases",
        "Catalogue and case measurements",
        _page(
            "UPENN-GBM catalogue and measurements",
            """
<h2>Refresh and compatibility checking</h2>
<p>The application includes a 147-case snapshot so the browser remains useful offline.
<b>Refresh catalogue</b> checks the current public release, discovers all matching segmentation
assets, and validates each mask's SRI24 dimensions, 1 mm spacing, affine/orientation, label set,
and non-empty contents. Compatible and unverified cases remain importable. Incompatible cases
stay visible but disabled; hover over their status for the reason.</p>
<p>The refreshed result and downloaded masks are cached. If the release has not changed, a
conditional request reuses the last verified catalogue. A failed or cancelled refresh leaves the
current list intact. The filter matches case ids, asset filenames, status, and validation details.</p>
<h2>Case measurements</h2>
<p>After refresh, the chooser displays approximate atlas-relative location, tumor-core volume,
whole-tumor volume, and whole-tumor X × Y × Z extent in the canonical head axes. Volumes are
calculated directly from the 1 mm segmentation voxels. Hover over a row to see the exact
whole-tumor centroid, left/right voxel percentages, measurements with additional precision,
source filename, and compatibility detail.</p>
<ul>
<li><span class="term">Centroid</span> is the arithmetic mean position of all included tumor
voxel centres; it is a geometric centre, not necessarily a point inside every irregular tumor.</li>
<li><span class="term">Extent</span> is the size of the axis-aligned bounding box along canonical
head X, Y, and Z. It is not the tumor's longest oblique diameter.</li>
<li><span class="term">Volume in mL</span> is voxel count multiplied by voxel volume. At 1 mm
isotropic resolution, 1,000 voxels equal 1 mL.</li>
</ul>
<p>The location text is intentionally coarse. Left/right comes from the distribution across the
canonical midline; anterior/posterior and inferior/superior come from thirds of the bundled
SRI24 brain bounds. Terms such as <i>Right · posterior · inferior</i> are atlas-relative summaries,
not lobe assignments or clinical localization.</p>
<h2>Status meanings</h2>
<table>
<tr><td><b>Compatible</b></td><td>The mask passed the expected grid, spacing, affine, label, and
non-empty-content checks.</td></tr>
<tr><td><b>Unverified</b></td><td>The asset was discovered but could not be fully checked, often
because its download failed temporarily. Import performs validation again.</td></tr>
<tr><td><b>Incompatible</b></td><td>The downloaded mask failed a structural/coordinate check and
is disabled, with the reason shown in its tooltip.</td></tr>
</table>
""",
        ),
    ),
    "upenn_regions": (
        "UPENN-GBM cases",
        "Tumor regions and scene import",
        _page(
            "UPENN-GBM tumor regions and scene import",
            """
<h2>Available regions</h2>
<p>A <span class="term">label map</span> stores an integer at each voxel to identify its class.
The UPENN masks use three independent labels. Field Workbench can also derive unions of them.</p>
<table>
<tr><th>Region</th><th>Definition</th></tr>
<tr><td>Necrotic/non-enhancing core</td><td>Source label 1. NCR/NET means
necrotic and non-enhancing tumor core.</td></tr>
<tr><td>Edema/infiltration</td><td>Source label 2. ED means peritumoral edematous/infiltrated
tissue, represented by abnormal T2-FLAIR signal in the source annotation convention.</td></tr>
<tr><td>Enhancing tumor</td><td>Source label 3. ET means enhancing tumor, the component that
enhances after contrast on T1-weighted MRI.</td></tr>
<tr><td>Tumor core</td><td>Derived union of labels 1 and 3. This is the default import.</td></tr>
<tr><td>Whole tumor</td><td>Derived union of labels 1, 2, and 3.</td></tr>
</table>
<p>Derived unions overlap their component objects if both are selected. This is deliberate and
lets one case supply either a composite target or separate visual subregions.</p>
<h2>Placement and mesh conversion</h2>
<p>The importer validates the expected 240 × 240 × 155, 1 mm isotropic NIfTI grid, reads its
physical RAS affine, builds a closed and consistently oriented triangle surface for every selected
region, and applies the fixed SRI24 RAS → canonical NAS/LPA/RPA head transform documented in
the SRI24 brain topic. Imported objects begin at position <code>(0, 0, 0)</code>, rotation
<code>(0°, 0°, 0°)</code>, and scale <code>(1, 1, 1)</code>. They therefore line up with newly
inserted SRI24 brain and Generic bust objects.</p>
<p>Measurement volume is the default role. Visual and Exclusion are also available. Generated
surfaces are topology-checked before being committed; all selected regions from one import are
added as one undoable scene change.</p>
<h2>Scene roles</h2>
<ul>
<li><b>Measurement volume</b> makes the closed tumor surface eligible for field sampling and
volume statistics.</li>
<li><b>Visual only</b> displays the surface without treating it as a target or keep-out object.</li>
<li><b>Exclusion zone</b> treats the surface as geometry that coils/bobbins should not enter
during design-rule checking and optimization.</li>
</ul>
""",
        ),
    ),
    "coil_construction": (
        "Coils and construction",
        "Physical coil data",
        _page(
            "Physical coil data",
            """
<p>A coil has both an electromagnetic centreline source and an optional physical winding
specification. The construction specification is used to generate detailed solver sources,
electrical estimates, the visible winding/bobbin, and design-rule geometry.</p>
<h2>Core electrical relationship</h2>
<p>The stored Magpylib source current represents effective ampere-turn current:</p>
<div class="formula">I_source = I_drive × N_turns × field_scale_factor</div>
<p>Detailed models divide this effective current among their representative turn paths so the
total ampere-turn strength is retained.</p>
<h2>Derived winding estimates</h2>
<ul>
<li>Wire length is mean turn length × turns, plus entered lead length.</li>
<li>Resistance at 20 °C uses the selected/entered ohms-per-kilometre value.</li>
<li>Operating resistance applies the copper temperature coefficient.</li>
<li>For peak-sine current, RMS current is peak current ÷ √2; DC/RMS mode uses the entered current directly.</li>
<li>Copper loss is <code>I_rms² × R_operating</code>.</li>
<li>Resistive voltage is <code>|I_drive| × R_operating</code>.</li>
<li>Current density is <code>I_rms ÷ copper cross-sectional area</code>.</li>
</ul>
<p>Racetrack turn length is calculated from two straight sections and two semicircular ends.
Closed planar Polyline coils can also use the common winding-envelope and bobbin system.</p>
""",
        ),
    ),
    "background_fields": (
        "Electromagnetic solver",
        "Background fields",
        _page(
            "Background fields",
            """
<p><b>Add Object → Background</b> opens the Background field configuration directly and adds a uniform static magnetic-flux-density
vector to the complete scene. It is a real field contributor: probes, sensors, 2D/3D maps,
measurement volumes, snapshots, comparisons, and optimizer evaluations all include every
visible Background field.</p>
<h2>Manual vector</h2>
<p>Enter <b>Bx</b>, <b>By</b>, and <b>Bz</b> directly in microtesla using Workbench scene axes.
Multiple visible background fields add vectorially. Hiding a Background field removes its
contribution without deleting it.</p>
<h2>3D vector glyph</h2>
<p>Every Background field has a visible arrow in the 3D scene showing the field direction. Its
X/Y/Z display position is editable in the object inspector. Moving the glyph changes presentation
only: a Background field remains uniform everywhere and its physical calculation does not depend
on the arrow location. The glyph uses a fixed display length so very weak or very strong fields do
not distort scene scaling; the actual B components and magnitude are shown in the object details
and hover text.</p>
<h2>Earth field — WMM2025</h2>
<p>The Earth-field tab calculates the local main geomagnetic field from WMM2025 using a WGS84
location, altitude, and date. Location can be entered as latitude/longitude or UTM, or selected
with the map picker. New Earth-field entries start at Sudbury, Ontario (46.4939° N, 80.9954° W)
rather than 0°/0°. The map itself needs internet access for map tiles; numeric location entry and
saved vectors do not.</p>
<p>WMM returns a geographic North/East/Down vector. Workbench rotates that vector into scene
coordinates using the entered heading, pitch, and roll. With all three angles at zero,
<b>+X = east, +Y = true north, +Z = up</b>. Heading is clockwise from true north.</p>
<div class="note"><b>Reproducibility:</b> the calculated WMM vector and its location/date/orientation
provenance are frozen into the scene file. Opening the scene later does not silently recalculate
the Earth's field. Use <b>Edit field… → Calculate / refresh WMM2025</b> to deliberately update it.</div>
<h2>Static-field meaning</h2>
<p>A Background field is a position-independent DC/static vector added to the Workbench
magnetostatic field solution. It does not introduce waveform phase, frequency, or time-domain
physics. For AC/RMS coil studies, interpret the result according to the instantaneous/static
field model being evaluated.</p>
""",
        ),
    ),
    "solver_models": (
        "Electromagnetic solver",
        "Coil field models",
        _page(
            "Coil field models",
            """
<p>The global preference applies to field maps, fluxlines, probes, sensors, measurement volumes,
snapshots, exports, and optimizer field evaluations. Each coil is compiled independently and may
fall back to Centreline when its required geometry is incomplete or unsupported. Auto is the default.</p>
<table>
<tr><th>Model</th><th>Representation</th><th>Best use</th></tr>
<tr><td><b>Auto</b></td><td>Uses geometry/workload heuristics, then—when practical—compares sparse Exact-reference samples against lower-cost representations.</td><td>Normal use when an evidence-based accuracy/performance choice is preferred over a forced global model.</td></tr>
<tr><td><b>Centreline</b></td><td>The original single Circle or Polyline carrying the full effective ampere-turn current.</td><td>Fast previews, sparse scene checks, and coils whose winding dimensions are unknown.</td></tr>
<tr><td><b>Exact turns</b></td><td>One current path per physical turn, distributed through the entered axial width and radial build. Each path carries the actual per-turn current including the field scale factor.</td><td>Final validation, wide/thick winding packs, and near-field work.</td></tr>
<tr><td><b>Bundled</b></td><td>Up to 32 weighted paths distributed through the winding cross-section. Each path represents one or more physical turns.</td><td>A strong accuracy/performance compromise for maps and optimization.</td></tr>
<tr><td><b>Current sheet</b></td><td>One to eight Magpylib TriangleStrip ribbon sources distributed through the radial build and spanning the axial winding width.</td><td>Densely and evenly wound layers, especially broad single-layer racetrack coils.</td></tr>
</table>
<h2>Stage one: geometric and workload screening</h2>
<ul>
<li><b>Exact turns</b> is selected for sparse windings, small primitive workloads, or manageable near-field calculations.</li>
<li><b>Current sheet</b> is currently selected conservatively for dense, broad, closed planar Polyline/racetrack windings whose observers are not close enough to resolve individual conductors.</li>
<li><b>Centreline</b> is selected when the winding pack is compact relative to the coil and the requested observers are sufficiently far away, or when a very large bundled workload can be simplified safely by the geometric rules.</li>
<li><b>Bundled</b> is the general detailed fallback when the physical winding envelope matters but Exact or Current sheet is not a suitable stage-one choice.</li>
<li>Optimizer exploration uses a conservative Auto purpose that avoids Exact turns except for genuinely sparse windings; finalist finite-pack validation remains available separately.</li>
</ul>
<p>The first-stage decision accounts for turns, finished wire diameter, winding width and radial build, turn pitch, coil characteristic size, sampled distance to the winding envelope, observer-point count, path-segment count, estimated primitive interactions, and the configured memory budget.</p>
<h2>Stage two: sparse empirical validation</h2>
<ul>
<li>For ordinary calculations with at least eight observer points, Auto selects up to 64 spatially spread sample points, including map extrema and a point near the winding.</li>
<li>When the sparse Exact-turn workload is bounded, Exact becomes the reference field for that coil and observer region.</li>
<li>Centreline, Bundled, and eligible Current-sheet models are compared against the Exact vector field.</li>
<li>The built-in balanced target is at most 1% vector RMS error and 2% 95th-percentile point error.</li>
<li>Auto chooses the lowest estimated full-map workload among the models that meet both limits.</li>
<li>If no approximation meets the limits, full Exact is used only when its requested workload remains bounded. Otherwise Auto uses the lowest-error available approximation and records that the tolerance was not met.</li>
<li>Validation results are cached by coil geometry and a quantized observer region, so a different resolution over the same region can reuse the comparison.</li>
<li>Very small observer sets, zero-current coils, unusually large Exact references, and optimizer exploration retain the deterministic first-stage choice.</li>
</ul>
<div class="note"><b>Interpretation:</b> validation is sparse and performed per coil. It is an
accuracy estimate for the requested region, not a proof of the maximum error at every unsampled
point. Strong cancellation between multiple large coil fields can also magnify the relative error of
the small combined residual. Force Exact turns for critical final checks where that distinction matters.</div>
<h2>Limits and fallback</h2>
<ul>
<li>Exact mode is limited to 20,000 turn paths per coil.</li>
<li>Current sheet requires a supported Magpylib TriangleStrip implementation and a valid closed planar path.</li>
<li>An invalid inward/outward offset, collapsed inner turn, missing winding dimensions, or unsupported path causes that coil to use Centreline.</li>
<li>When Auto's selected model cannot be compiled, it tries a conservative lower-cost alternative before retaining the Centreline source.</li>
<li>The map status bar reports the representation actually used, including mixed Auto scenes and fallbacks. Solver diagnostics include heuristic metrics, empirical errors, validation timing/cache use, and tolerance status.</li>
</ul>
<div class="warning">Exact does not imply a finite-thickness copper conductor. It means each turn
centreline is represented explicitly at its calculated position. Insulation and conductor diameter
are used to place turns, not to solve current density inside the copper.</div>
""",
        ),
    ),
    "solver_backend": (
        "Electromagnetic solver",
        "Consolidated backend",
        _page(
            "Consolidated field backend",
            """
<p>Every field tool routes through the same centralized backend so the selected coil preference
is applied consistently.</p>
<h2>How a detailed calculation is assembled</h2>
<ol>
<li>Evaluate the complete live Magpylib Studio scene. This baseline includes magnets, sensors as observers, and the original centreline coils.</li>
<li>For coils successfully compiled into the requested detailed representation, calculate and subtract their original centreline contribution.</li>
<li>Calculate the compiled Exact, Bundled, or Current-sheet sources and add that contribution.</li>
</ol>
<div class="formula">Final field = baseline scene − replaced centrelines + detailed coil sources + visible background fields</div>
<p>This replacement approach leaves all non-coil sources untouched and also handles coils copied
inside patterned collections. Coils that fall back remain in the baseline and are not subtracted.</p>
<h2>Caching</h2>
<p>Compiled coil representations are cached by physical dimensions, path, pose, and requested
model. Auto's empirical comparison is cached separately by coil geometry and a quantized observer
region, allowing a changed resolution over a similar region to reuse its error measurements. The
caches speed repeated calculations without changing saved scene data. Scene edits or
solver-preference changes invalidate relevant compiled representations.</p>
<h2>B and H</h2>
<p>The backend supports magnetic flux density <b>B</b> and magnetic field strength <b>H</b> when the
calling analysis supports them. Display units and component selection are applied after the vector
field is calculated.</p>
""",
        ),
    ),
    "solver_resources": (
        "Electromagnetic solver",
        "Performance and diagnostics",
        _page(
            "Performance, memory, and diagnostics",
            """
<p>Detailed fields can create very large temporary arrays. Workbench therefore divides calculations
by both observer points and source primitives.</p>
<ul>
<li>The memory preference is a target for one Magpylib kernel call, not a hard limit on total process memory.</li>
<li>Polyline complexity is estimated by line-segment count; current sheets are estimated by their surface primitives.</li>
<li>A call contains at most 64 modeled sources, and point batches are capped at 2,048 observers before the memory estimate makes them smaller.</li>
<li>Completed batch values are accumulated into preallocated result arrays.</li>
<li>Compiled source geometry and final result arrays still consume memory outside the temporary-call budget.</li>
</ul>
<h2>Worker concurrency</h2>
<p>An optimizer <b>worker</b> is one spawned Python process evaluating one independent candidate on
its own private copy of the scene. Independent coarse samples and global verification waves can run
concurrently; adaptive minimization and coordinate refinement remain ordered because each next
step depends on the previous result. The optimizer commits concurrent results in candidate order
and gives each candidate a deterministic random stream, so a fixed seed does not become dependent
on which process finishes first. One worker uses the original direct serial path without starting a
child process.</p>
<p><b>Edit → Preferences → Worker concurrency</b> sets the maximum worker count for every open
document. The same limit caps simultaneous crash-isolated 2D and 3D render jobs across all map windows;
one worker therefore preserves serial rendering, while larger values admit that many render processes.
Worker count is allowed to exceed the operating system's reported logical-CPU count: this
intentional <b>oversubscription</b> can occasionally improve throughput for field workloads whose native
numerical kernels do not keep every process uniformly CPU-bound. <b>Benchmark…</b> lets you choose
the highest worker count to test, from 1 through 32, without clamping that choice to the machine's
reported logical-CPU count. It then tests every count from 1 through that ceiling using repeated field
solves on a frozen packaged scene, the coil model currently selected in Preferences, and the current
per-kernel memory budget. The CPU count is informational only. The benchmark does not inspect or
change an open scene. The recommendation is the smallest
worker count within 8% of the fastest measured time; choosing the smaller near-tie reduces resource use.</p>
<div class="warning">The solver memory budget applies to each native kernel call, not to the whole
application. Every active optimizer or map-render worker owns a private scene and solver state, so
total process memory can increase roughly with worker count. The benchmark measures optimizer
throughput; demanding 2D/3D renders can use more memory than its built-in benchmark scene. Re-run the
benchmark after changing hardware, the coil field model, or the solver memory budget.</div>
<h2>Progress and cancellation</h2>
<p>Field-map, sensor-path, and measurement-volume progress count observer samples across the baseline,
subtraction, and detailed-source passes. Closing the progress dialog or pressing Escape requests
cancellation. The current native Magpylib call must finish first; cancellation takes effect at the
next safe batch boundary.</p>
<h2>Diagnostics</h2>
<p>When enabled, a temporary JSONL log records pass and batch starts/completions, source counts,
primitive counts, point counts, memory budget, and elapsed times. Entries are flushed as they are
written, making the last completed event useful when investigating an abrupt process termination.</p>
""",
        ),
    ),
    "maps_sensors": (
        "Analysis tools",
        "Maps and sensors",
        _page(
            "Maps and sensors",
            """
<h2>Sensor Plot</h2>
<p>Evaluates the selected field at the pixels of a scene Sensor. A sensor with no explicit pixel
array is treated as one pixel at local <code>[0, 0, 0]</code>. Linear axis/path sensors plot values against
path distance or sample index and can export CSV. The window is non-modal, opens before calculation,
and uses the shared determinate progress/cancellation system after <b>Calculate</b> is pressed. Its
status bar records the actual coil solver representation and total elapsed time.</p>
<h2>Magnetometer probes</h2>
<p><b>Add Object → Sensor → Magnetometer probe</b> opens the first-draft probe library. A probe is
a physical scene object with a visible housing, an explicit local origin, and one or more coloured
sensitive points. Every channel stores its own local point and unit sensitive direction. When
<b>Calculate probe reading</b> is pressed in the inspector, Workbench evaluates the complete world
field separately at every sensitive point and projects it onto that channel's transformed local
axis. The displayed combined magnitude is formed from those reported channel values; it is not a
single-point |B| measurement when the sensing elements are spatially separated.</p>
<p>The bundled SENSYS FGM3D definition uses the manufacturer's documented 26 × 26 × 149 mm envelope
and single-axis reference distances. Its scene origin is the geometric centre of that housing, so
the object position can be indexed directly from the probe body; its X/Y/Z sensitive points remain
at their documented offsets from the reference edge. A contrasting panel marks the sticker on
the local −Y face without adding floating text to the viewer. The manufacturer's sticker symbols define +X across the face, +Z toward the
reference end, and +Y inward through the sticker. A red end face marks its cable connection on
local −Z. Manufacturer and drawing links appear in both the
library and the selected-object inspector.</p>
<p>The bundled MEDA FM300 / FVM400 entry uses the manufacturer's 4.00 × 1.00 × 1.00 inch
rectangular probe drawing with its scene origin at the body centre. Local +X runs from the cable
end toward the tip, +Y follows the documented width direction, and +Z points down toward the base;
the top/label side is therefore local −Z and the red cable face is local −X. The X sensitive point
is 1.54 inches from the tip. Y and Z correctly share the other ring-core centre 0.68 inches from the
tip, with all three centres 0.49 inches below the top face. MEDA's product page, Revision A manual,
and FM300/FVM400 succession history are linked from the library and inspector. The FVM400 drawing
supplies the manufacturer geometry; the first-draft FM300 equivalence records the owner's
confirmation that both instruments use the same physical probe.</p>
<p><b>Create custom…</b> supports box-envelope definitions,
arbitrary channel points/directions, optional sticker/label and cable-connection faces, and optional source metadata. Imported
<code>.fwprobe.json</code> files use the same schema. The complete validated definition is embedded
in each saved scene, so reopening the project does not depend on an external library file.</p>
<div class="warning">A housing envelope is a visual model in this first draft and does not yet
participate in jig generation or DRC. Confirm a probe's drawing revision, coordinate convention,
and actual calibration before treating a simulated reading as metrology.</div>
<h2>2D field map</h2>
<p>Each launch opens a normal non-modal top-level workspace and rebuilds the complete model from a
private launch-time snapshot. The main Workbench remains available for opening or editing other
scenes and launching more 2D/3D workspaces. Later source-scene changes and source-tab closure cannot
alter or close an existing map window; its title identifies the captured source scene as a snapshot.
Close a workspace with the native title-bar × or the pinned <b>Exit</b> button beneath the map actions.</p>
<p>Samples a regular XY, XZ, or YZ plane. The dialog uses a tabbed viewer: the permanent
<b>Selector</b> tab shows the normal 3D scene with the selected sampling plane overlaid directly in its
own viewer, so Plane, Offset, and View width can be positioned independently of the
main-window viewer. Each successful <b>Calculate map</b> opens a separate closeable result tab and
switches to it, while the Selector tab remains available for choosing another plane at any time.
Heatmap mode displays a selected component or magnitude; fluxline mode integrates in-plane paths from
the sampled vector field. Preferences also provide independent minimum-field and conductor-distance
cutoffs for fluxline tracing; the latter uses the winding-pack envelope when physical coil dimensions
are available and otherwise falls back to the coil centreline. Combined mode performs the required
sampled phases and overlays both products. <b>Object outlines</b> is enabled by default and
adds the cross-section of visible solid scene geometry wherever the sample plane actually cuts it.
The <b>Objects…</b> selector beneath it provides an independent checked entry for every
outline-capable object, with Select all/Clear all shortcuts. Passive primitive/imported meshes,
supported solid magnets, and enabled physical coil construction are checked directly. Selected
cross-sections are clipped to the chosen View width and overlaid without changing the field
calculation. Hidden objects remain hidden even when selected, and line-only sources/sensors do not
contribute an outline.</p>
<h2>3D field map</h2>
<p>The dialog uses the same detached snapshot lifetime as the 2D workspace. Its permanent
<b>Selector</b> tab shows the captured 3D scene with
the planned slice stack overlaid. The slice nearest the volume centre is the same translucent filled
selector plane used by the 2D map, while every other projected slice is a thin wire-frame outline.
Centre X/Y/Z, Cube width, Slice direction, and Slices keep this selector synchronized live, even while
a result tab is active. Each successful <b>Calculate 3D map</b> creates a new closeable <b>Map N</b>
tab and switches to it, so multiple calculated views can be retained and compared. The Selector tab
has no close button and remains available for spatial setup at any time.</p>
<p>The calculation samples a regular volume and renders layers or integrated field paths according to
the selected view. A dense 3D grid can be substantially more expensive than a 2D plane because point
count grows with all three resolution axes.</p>
<p><b>Fullscreen viewer</b> opens a separate true-fullscreen interactive copy of the rendered
Plotly/Qt WebEngine view; the original embedded viewer is left in place. Press <b>Esc</b> or <b>F11</b>
to close the fullscreen copy and return to the workspace.</p>
<p>Manual colour limits affect visualization only. They do not clip or alter the underlying field
vectors or exported numerical results.</p>
<p>For both 2D and 3D waveform work, create an <b>Animated map</b> first and inspect the rendered
Waveform tab. <b>Export video</b> appears beside <b>Fullscreen viewer</b> only while an animated
result tab is selected. Export reuses that tab's already-calculated WebGL field payload and current
interactive camera/view, so later Selector changes do not alter the export and no magnetic-field
solve is repeated. Scripted camera timelines remain the timeline stored with the rendered animation. H.264 MP4 export
uses the bundled FFmpeg encoder on packaged Windows builds; Linux uses the selected/system FFmpeg.
A numbered PNG frame sequence is available without an encoder and is also useful for video editors.</p>
<h2>Brain View</h2>
<p><b>Brain view</b> provides SRI24/LPBA40 regional analysis in the same detached, launch-time-snapshot
workflow as the field-map windows. Static and Animated results keep their own Brain Areas table,
filters, sorting, highlighting, and cached statistics. See the dedicated <b>Brain View and statistics</b>
help topic for metric definitions, playback exposure summaries, sampling details, and export behavior.</p>
""",
        ),
    ),
    "brain_view": (
        "Analysis tools",
        "Brain View and statistics",
        _page(
            "Brain View and statistics",
            """
<p><b>Brain view</b> analyzes the magnetic field over the bundled LPBA40 parcellation registered to a
scene <b>SRI24 brain</b>. It is intended for spatial and time-varying field comparison across atlas
regions; it is not patient-specific dosimetry or a biological response model.</p>
<h2>Opening Brain View</h2>
<ul>
<li>If the scene contains one SRI24 brain, it is used automatically. If several are present, select the intended SRI24 brain before opening Brain view.</li>
<li>The atlas follows that object's position, rotation, and scale. The <b>Map volume</b> controls only the field-sampling cube; they do not move the atlas.</li>
<li>The default Brain View map volume is centred at X/Y/Z = <code>0, 0, 50 mm</code> with a <code>300 mm</code> cube width.</li>
<li><b>Static analysis</b> creates a fixed-current result. <b>Animated analysis</b> evaluates a finite waveform/sequencer playback and adds per-frame and whole-playback regional statistics.</li>
</ul>
<p>Each calculated result is a frozen launch-time scene snapshot. Its Brain Areas filters, sorting,
column visibility, widths, expansion state, highlights, and cached statistics are independent of the
Selector and other result tabs.</p>
<h2>Brain Areas hierarchy</h2>
<p>The table contains Whole brain, eight major Workbench browsing groups, anatomical structures, and
Left/Right leaves where LPBA40 provides bilateral labels. <b>Collapse</b>, <b>Expand</b>, and
<b>Expand major</b> change tree expansion only. Sorting is hierarchical: siblings are sorted within
their anatomical parent instead of flattening the anatomy into one global list.</p>
<p>Atlas sampling is volume weighted. When the analysis retains fewer sample points than source atlas
voxels, each retained point carries the voxel volume it represents. Therefore <b>sample count is not
an anatomical-volume measure</b>; the Volume column is calculated from the source LPBA40 voxels.</p>
<h2>Current-frame statistics</h2>
<table>
<tr><th>Column</th><th>Meaning</th></tr>
<tr><td><b>RMS |B|</b></td><td>Volume-weighted spatial root-mean-square field magnitude: <code>sqrt(Σw|B|² / Σw)</code>.</td></tr>
<tr><td><b>Mean |B|</b></td><td>Volume-weighted arithmetic mean field magnitude: <code>Σw|B| / Σw</code>.</td></tr>
<tr><td><b>P95 |B|</b></td><td>Volume-weighted 95th percentile of |B|. Approximately 95% of the represented regional volume is at or below this value.</td></tr>
<tr><td><b>Peak |B|</b></td><td>Largest |B| among the retained analysis samples for the region.</td></tr>
<tr><td><b>Uniformity</b></td><td>P5–P95 spread relative to the mean. Higher values indicate a narrower magnitude distribution.</td></tr>
<tr><td><b>Direction</b></td><td>Magnitude of the volume-weighted mean unit-field direction, expressed as 0–100%. Higher values mean field vectors point more consistently in one direction across the region.</td></tr>
<tr><td><b>Volume</b></td><td>LPBA40 source-voxel volume represented by the row, in cm³.</td></tr>
</table>
<div class="formula">Uniformity = 100 × [1 − (P95 − P5) / mean], clamped to 0–100%</div>
<div class="formula">Direction = 100 × |Σ(w · B̂) / Σw|</div>
<h2>Filters and highlights</h2>
<p>Filters can use the displayed numeric metrics or <b>Side = Left/Right</b>. A rule's Action changes
what matching means:</p>
<ul>
<li><b>Filter</b> is an exclusion rule: rows matching the condition are removed. With several Filter rules, a row is excluded only when it matches <b>all</b> active Filter rules. Ancestors needed to show the path to a surviving child remain visible.</li>
<li><b>Highlight</b> never removes rows. Matching rows receive the selected colour; independently evaluated later Highlight rules take colour priority when several match.</li>
</ul>
<h2>Animated results: Current frame vs Playback summary</h2>
<p><b>Current frame</b> shows the spatial statistics above at the selected playback time.
<b>Playback summary</b> compares the complete finite physical pass. The two modes retain independent
filters, sorting, visible columns, and column widths.</p>
<table>
<tr><th>Playback column</th><th>Meaning</th></tr>
<tr><td><b>Playback RMS |B|</b></td><td>Space-time RMS based on the instantaneous regional spatial RMS. Equivalent to <code>sqrt(B²-time / T)</code>.</td></tr>
<tr><td><b>Mean |B|</b></td><td>Physical-time-weighted mean of the regional spatial Mean |B|.</td></tr>
<tr><td><b>Max P95 |B|</b></td><td>Largest regional spatial P95 reached by the retained regional-frame cache.</td></tr>
<tr><td><b>Absolute peak |B|</b></td><td>Largest retained atlas-sample |B| found in the regional-frame cache. Its tooltip reports the physical playback time of that peak.</td></tr>
<tr><td><b>B-time</b></td><td>Time integral of instantaneous spatial RMS |B|. It grows with field magnitude and exposure duration.</td></tr>
<tr><td><b>B²-time</b></td><td>Time integral of instantaneous spatial RMS² |B|. It gives stronger weight to high-field intervals.</td></tr>
<tr><td><b>Volume</b></td><td>The same atlas volume used by Current frame.</td></tr>
</table>
<div class="formula">B-time = ∫ RMSspace(|B|, t) dt &nbsp;&nbsp; [µT·s]</div>
<div class="formula">B²-time = ∫ RMSspace²(|B|, t) dt &nbsp;&nbsp; [µT²·s]</div>
<div class="formula">Playback RMS = sqrt(B²-time / exposure time) &nbsp;&nbsp; [µT]</div>
<p>The summary information block reports the <b>physical exposure time</b>, generated-waveform
frequency/cycle count (or custom-trace playback pass), sequencer step/loop/activation counts,
rendered frames, numerical time samples, atlas samples, and—when the cache is bounded—retained
regional frames.</p>
<h2>Sampling and exposure interpretation</h2>
<p>Playback RMS, B-time, and B²-time use the higher-density numerical analysis timeline. Playback
Mean, Max P95, and Absolute peak reuse the Brain Areas regional-frame cache. Ordinary animations
retain every rendered regional frame; extremely long/slow-motion animations evenly sample this
auxiliary cache within its memory ceiling while the actual field/video animation keeps every
requested frame.</p>
<p>The integration interval follows physical waveform/sequencer time, including resolved slew and
voltage/RL current behaviour. Changing preview FPS, exported-video slow motion, or repeatedly looping
the preview does <b>not</b> increase the finite exposure time or its B-time/B²-time values.</p>
<div class="warning"><b>B-time and B²-time are comparative magnetic-field exposure indices, not dose.</b>
They do not model induced electric field, tissue conductivity, frequency-dependent coupling,
biological response, or clinical effect. See <b>Limitations and model boundaries</b> for the solver's
quasi-static assumptions.</div>
<h2>Traces and video export</h2>
<p>Animated-analysis <b>Traces</b> can include LPBA40 rows; each selected row contributes its
volume-weighted spatial RMS trace. <b>Select major</b> checks the eight major groups and <b>Clear</b>
removes brain-region traces.</p>
<p>After an Animated result is rendered, <b>Export video</b> appears beside <b>Fullscreen viewer</b>.
Export uses that result tab's frozen field payload and Brain Areas state rather than current Selector
settings, and it does not repeat the magnetic-field solve. The video includes whichever Brain Areas
mode is active when export is opened.</p>
<div class="note">The lobe/division grouping is a Field Workbench browsing hierarchy, not an upstream
LPBA40 taxonomy. SRI24/LPBA40 represents reference-atlas anatomy rather than an individual subject.</div>
""",
        ),
    ),
    "real_measurements": (
        "Analysis tools",
        "Real measurements and calibration",
        _page(
            "Real measurements and coil calibration",
            """
<p><b>Analyze → Measurements</b> opens the real-measurement calibration workspace. Phase 1 is
intended for bench validation of one coil or a user-selected set of coils that should behave as one
linked magnetic group.</p>
<h2>Coordinate workflow</h2>
<ol>
<li>Add a Point sensor or linear axis sensor at the physical locations where measurements are taken.</li>
<li>Open <b>Measurements</b>, choose that sensor, and choose <b>|B|</b> or one local B component.</li>
<li>Enter readings in µT directly, use <b>Add row</b> for individual measurements, paste spreadsheet cells, or import CSV/text data. X/Y/Z and optional Path coordinates can be edited directly in the table.</li>
<li>When spreadsheet cells are pasted, every numeric copied cell becomes one measurement entry. Horizontal selections are transposed into rows automatically, and additional table rows are created as needed.</li>
<li>Check the coil or coils that were energized for the measurement.</li>
<li>Press <b>Fit calibration</b>. The saved scene is not changed by fitting.</li>
<li>Review the fitted factor and residual error, then optionally save the dataset or press
<b>Apply calibration</b>.</li>
</ol>
<p>Dataset rows start from the sensor's world XYZ coordinate and local orientation. Coordinates can then be edited manually, and <b>Add row</b> duplicates the nearest row as a safe orientation/template before you enter a new position. Moving or deleting the original sensor later does not change the saved dataset coordinates.</p>
<p>The Dataset, Calibration coil group, Measurement samples, and Scale-only fit sections are separated by draggable horizontal dividers. The Measurement samples table receives most of the space by default; drag a divider whenever a larger table, coil list, or fit summary is more useful.</p>
<h2>What Phase 1 fits</h2>
<div class="formula">B_measured ≈ k × B_model</div>
<p>The least-squares fit solves one dimensionless scale factor <code>k</code>. The selected group is
evaluated with every member temporarily set to a field correction of exactly <code>1.000×</code>, so
re-fitting never multiplies an older empirical correction. Workbench isolates the selected group by
subtracting an otherwise identical model evaluation with that group switched off, so unselected coils,
permanent magnets, and saved Background fields cancel on the model side.</p>
<p>When <b>Apply calibration</b> is used, the fitted value becomes the <b>absolute</b> Field correction
factor of every checked coil. A factor of <code>1.000×</code> means no empirical amplitude correction;
<code>1.075×</code> means the physical group produced about 7.5% more field than the uncalibrated model.</p>
<div class="warning"><b>Amplitude convention:</b> entered readings must use the same RMS, peak-sine,
or DC convention as the coil drive values represented by the scene. Phase 1 also assumes any ambient
field or contribution from other sources has been removed from the measurements or is negligible.
A later calibration model can fit background offsets and independent per-coil coefficients.</div>
<h2>Interpreting residuals</h2>
<p>The dialog reports RMS error before and after scaling, the maximum absolute residual, and R² when
there are enough non-constant measurements. A scalar factor that leaves a strong position-dependent
residual may indicate geometry, alignment, polarity, current, or field-shape mismatch rather than a
simple amplitude calibration problem.</p>
""",
        ),
    ),
    "measurement_sampling": (
        "Analysis tools",
        "Measurement sampling",
        _page(
            "Measurement-volume sampling",
            """
<p>Measurement volumes can use either deterministic equal-weight voxel-centre samples or an exact
list of user-defined local sample points. Every local coordinate is transformed into world space by
the measurement object's position and rotation before field evaluation.</p>
<table>
<tr><th>Quality</th><th>Grid centres per bounding-box axis</th><th>Maximum box samples</th></tr>
<tr><td>Preview</td><td>7</td><td>343</td></tr>
<tr><td>Standard</td><td>15</td><td>3,375</td></tr>
<tr><td>Fine</td><td>21</td><td>9,261</td></tr>
</table>
<ul>
<li>Boxes retain every grid cell centre.</li>
<li>Spheres discard centres outside the sphere.</li>
<li>Cylinders discard centres outside the circular cross-section.</li>
<li>Healthy closed meshes discard centres not classified inside the surface.</li>
</ul>
<p>Every retained voxel receives equal statistical weight. For curved shapes and meshes, the
retained sample count is therefore lower than the bounding-box maximum. Very thin features may
require a finer quality before any voxel centre lands inside them.</p>
<div class="note">Sampling quality measures the discrete model of the volume, not solver accuracy by
itself. A finer grid can reveal spatial variation that a coarse grid misses, but it also increases
calculation time.</div>
<p><strong>User defined points</strong> accepts one or more <code>[X, Y, Z]</code> coordinates in local
millimetres, for example <code>[0, 0, 0], [9, 0, 0], [18, 0, 0]</code>. These points replace generated
voxels and all receive equal statistical weight. They are stored with the object, so movement and
rotation preserve the intended measurement pattern. Defined points are used exactly as entered and
are not clipped automatically to the visual object's boundary.</p>
<p>Select one or more objects already assigned the Measurement volume role, then choose
<strong>Analyze volume</strong>. With several selected volumes, Workbench evaluates them in scene
selection order and opens one independent result window for each successful analysis. Closing the
active progress window cancels the current calculation and leaves the remaining selected volumes
unanalyzed.</p>
<p>After sampling completes, each measurement result opens in its own non-modal window. Several
fresh analyses and stored snapshot results can remain visible together while the main Workbench is
used for other tasks. The <b>Cross sections</b> tab shows the central XY, XZ, and YZ sample layers
together on one shared |B| colour scale. These are the actual evaluated points nearest the measurement
volume's local centre plane on each axis; Workbench does not interpolate a new field grid for this view.</p>
""",
        ),
    ),
    "measurement_stats": (
        "Analysis tools",
        "Measurement statistics",
        _page(
            "Measurement statistics",
            """
<p>Statistics are calculated from finite values of the field magnitude <code>|B|</code> in µT at the
retained equal-weight sample points. Workbench uses the population standard deviation
(<code>numpy.std</code> with <code>ddof=0</code>).</p>
<table>
<tr><th>Statistic</th><th>Calculation and interpretation</th></tr>
<tr><td>Mean</td><td>Arithmetic average of all valid magnitudes.</td></tr>
<tr><td>Median / P5 / P25 / P75 / P95</td><td>Percentiles of the valid magnitude distribution. P5 means 5% of samples are at or below that value.</td></tr>
<tr><td>Minimum / maximum</td><td>Lowest and highest valid sampled magnitudes.</td></tr>
<tr><td>Standard deviation</td><td><code>sqrt(mean((x − mean)²))</code>, in µT.</td></tr>
<tr><td>Coefficient of variation</td><td><code>100 × standard deviation ÷ |mean|</code>. Lower values indicate less relative spread.</td></tr>
<tr><td>Uniformity spread</td><td><code>100 × (P95 − P5) ÷ |mean|</code>. This robust spread ignores the most extreme 5% at each tail; lower is more uniform.</td></tr>
<tr><td>RMS target error</td><td><code>sqrt(mean((|B| − target)²))</code>, in µT. It penalizes larger errors more strongly.</td></tr>
<tr><td>Within target band</td><td>Percentage satisfying <code>target × (1 − tolerance/100) ≤ |B| ≤ target × (1 + tolerance/100)</code>.</td></tr>
<tr><td>Below / above target</td><td>Percentages below the lower band limit or above the upper band limit.</td></tr>
</table>
<h2>Direction statistics</h2>
<ul>
<li><b>Mean vector:</b> component-wise average of the valid B vectors.</li>
<li><b>Mean direction:</b> normalized mean vector, or zero when the mean vector has no magnitude.</li>
<li><b>Directional consistency:</b> <code>100 × length(mean(unit field vectors))</code>. A value near 100% means vectors point together; a value near 0% means directions are widely dispersed or cancel.</li>
<li><b>Median and P95 angular deviation:</b> angles between each non-negligible unit vector and the resultant reference direction.</li>
</ul>
<p>Sample count reports all requested retained points; valid count excludes NaN or infinite field
results. Statistics describe the sampled volume only and should not be interpreted as a confidence
interval or proof of continuous-field bounds between samples.</p>
""",
        ),
    ),
    "snapshot_stats": (
        "Analysis tools",
        "Snapshots and comparison",
        _page(
            "Snapshots and comparison statistics",
            """
<p>A snapshot stores the measurement geometry, local sample coordinates, settings, field vectors,
statistics, scene signature, and coil-modeling report. New snapshots also store a compact replayable
copy of their field sources so selected candidate-coil currents can be fitted later, even after the
snapshot is imported into another project.</p>
<p>The Snapshot Manager accepts arbitrary multi-selection. View Results opens one independent viewer
for every selected snapshot, Compare opens one comparison containing every selected snapshot, and
Delete removes the complete selection as one undoable operation. Rename still applies to one snapshot.</p>
<h2>Performance benchmark</h2>
<p>This mode asks how well each apparatus meets one shared specification. Sample grids may differ.
The overview shows three headline measures for every snapshot:</p>
<ul>
<li><b>Target coverage:</b> percentage of finite samples inside the shared target band.</li>
<li><b>Uniformity spread:</b> <code>100 × (P95 − P5) ÷ |mean|</code>; lower is more uniform.</li>
<li><b>Directional consistency:</b> <code>100 × length(mean(unit B vectors))</code>.</li>
</ul>
<h2>Exposure matching</h2>
<p>This mode treats one selected snapshot as the reference exposure and compares every other selected
snapshot point-by-point. Local sample coordinates must match exactly. World placement may differ;
field vectors are expressed in each measurement object's local frame before comparison.</p>
<ul>
<li><b>Mean intensity:</b> the reference mean field magnitude is shown directly; each candidate shows
its mean magnitude and signed percent bias relative to the reference.</li>
<li><b>Relative vector RMS error:</b> RMS length of the complete vector difference divided by the
reference RMS field magnitude. Both strength and direction errors contribute.</li>
<li><b>Points matching:</b> percentage satisfying both the selected pointwise magnitude tolerance and
angular-direction tolerance.</li>
<li><b>Overall score:</b> a presentation score averaging inverse absolute mean-intensity bias, inverse
relative vector RMS error, and point-match coverage. It is clipped to 0-100 and does not replace the
underlying engineering statistics.</li>
</ul>
<h2>Normalize current</h2>
<p>Current normalization is optional and applies only to candidate snapshots. It does not alter the
stored snapshot or its field-scale factors. The fit minimizes the pointwise difference between the
candidate and reference <b>field magnitudes</b>, <code>|B|</code>. Vector direction is deliberately
excluded from current fitting so directional disagreement remains visible in Relative vector RMS
error, Direction difference, and the point-match statistics instead of being reduced by simply
turning the candidate field down. Each candidate appears as its own subtree. Coils that
already share the same immediate scene group begin in a current-fit group with that scene-group
name; other coils begin ungrouped. Select two or more coils and use <b>Group selected coils</b> to
create or rearrange fit groups. Every coil inside a fit group shares one <b>Fixed / Normalize
together</b> dropdown and one fitted current magnitude. Ungrouped coils have their own <b>Fixed /
Normalize independently</b> dropdown. Groups can be renamed, and either a whole group or selected
member coils can be ungrouped. Coil polarity is preserved. The fitted group and coil currents are
reported explicitly in the overview and full statistics. Fresh snapshots retain scene display names
for this selector; older local snapshots use matching live-scene names when available.</p>
<p>Plots and detailed statistics use the candidate selected in the Detailed candidate control.
Performance and matching results remain descriptive statistics of the stored sample points; they do
not prove continuous-field bounds between samples.</p>
""",
        ),
    ),
    "custom_surface_regions": (
        "Design tools",
        "Custom surface-region meshes",
        _page(
            "Custom surface-region meshes",
            """
<p>Field Workbench can import a passive triangle mesh with arbitrary named surface patches that behave like
the authored areas on the built-in Generic bust. The intended authoring workflow is Blender-first:
use ordinary Blender material assignments to mark faces, then run the bundled Field Workbench
exporter to create a matched mesh and metadata pair.</p>
<h2>Blender workflow</h2>
<ol>
<li>Prepare one mesh object in Blender. Establish the local origin and orientation you want to
preserve in Workbench. Apply transforms first if you want the current Blender object transform baked
into the mesh coordinates.</li>
<li>Create materials with whatever names are meaningful for the object: for example
<code>Scalp</code>, <code>Face</code>, <code>Mounting pad</code>, <code>Keep 12 mm away</code>, or
any other name. Workbench does not require anatomy-specific names or a fixed vocabulary.</li>
<li>In Edit Mode, select faces and use Blender's material <b>Assign</b> control. A material can cover
one contiguous patch or many disconnected patches; all faces carrying the same material name become
one Workbench region.</li>
<li>Return to Object Mode and make that mesh the active object.</li>
<li>Open Blender's <b>Scripting</b> workspace, create a text block, then use the
<b>Copy Blender region exporter</b> button at the bottom of this Help window. Paste the script into
Blender and choose <b>Run Script</b>.</li>
<li>The script opens a file picker and writes two files with the same stem: <code>name.obj</code> and
<code>name.regions.json</code>. Keep them together and do not rename either file after export; the sidecar records the exact OBJ filename.</li>
<li>In Field Workbench click <b>Import object…</b> and select the OBJ. In the
import dialog, set <b>Source coordinates</b> to the units represented by the numeric coordinates in
Blender (mm, cm, or m). Workbench automatically detects the companion <code>.regions.json</code>,
verifies the file identity/topology, and embeds the region map into the scene file.</li>
</ol>
<h2>What the exporter preserves</h2>
<p>The exporter evaluates the active Blender object, triangulates a temporary copy, and writes the
OBJ itself so polygon order and region face indices are guaranteed to correspond. The original Blender
object is not modified. Visible modifiers are included. Object transforms are not silently baked, and
raw local mesh coordinates are preserved; the Workbench import dialog is where their mm/cm/m scale is
declared. Material names are preserved exactly as the region display names and keys, including spaces
and Unicode characters.</p>
<p>Faces whose material slot is empty are left <b>unmarked</b>. That is legal: a custom object can have
only a few specially marked areas instead of partitioning its entire surface. If a material is assigned
to every face, the regions can form a complete partition like the Generic bust.</p>
<h2>After import</h2>
<p>Select the imported object and use <b>Surface regions</b> in the inspector. Every region starts
inactive and hidden so importing metadata does not silently create new design rules. Enable
<b>Active</b>, then independently enable <b>Place</b> and/or <b>DRC</b>; optionally set a regional
clearance override and display colour.</p>
<p>Regional DRC uses the complete host mesh for penetration/containment and the named patch for local
clearance. For reliable inside/outside containment the host object therefore still needs to be a healthy
closed, consistently oriented mesh. Open meshes can still provide surface-distance information but cannot
prove containment.</p>
<div class="note"><b>Matched-pair protection:</b> the exporter stores the OBJ SHA-256 in the sidecar.
Workbench rejects a sidecar copied from a different mesh even if the triangle count happens to match.
After import, Workbench stores an ordered-geometry fingerprint with the scene so later face reordering or
hand-edited scene data cannot silently move region rules onto different triangles.</div>
<div class="warning"><b>Do not independently re-export the OBJ after creating the sidecar.</b> If the
geometry changes, run the Blender exporter again and import the new matched pair.</div>
""",
        ),
    ),
    "drc": (
        "Design tools",
        "Design-rule checking",
        _page(
            "Design-rule checking",
            """
<p>The <b>Design Rule Check</b> window opens without immediately running geometry checks. Review or
change the clearance rules first, then press <b>Check design</b>. Long checks run in the background;
the progress bar reports pair-level progress and a live elapsed-time counter, while completed verdicts are added to the results table
immediately so you can watch the report populate while the remaining geometry is still being checked. The final elapsed time remains visible after the pass completes.</p>
<p><b>Check design</b> compares the shared physical coil-assembly envelope against exclusion zones
and other coil assemblies. The same winding, bobbin barrel, and flange definition is used for both
the visible construction and the DRC calculation.</p>
<ul>
<li><b>Clear:</b> the required clearance is met.</li>
<li><b>Below clearance:</b> objects do not penetrate but are closer than the configured limit.</li>
<li><b>Intersecting:</b> physical envelopes overlap.</li>
<li><b>Incomplete/invalid:</b> required construction or reliable geometry is unavailable.</li>
<li><b>Clearance-only mesh result:</b> an open or damaged imported surface can be checked for surface distance, but not reliable inside/outside containment.</li>
</ul>
<p>Primitive Box, Sphere, and Cylinder exclusions support analytic or oriented-solid checks.
Healthy closed STL/OBJ meshes can support containment. Open edges, non-manifold edges, damaged
faces, or inconsistent normals can limit what the mesh safely proves.</p>
<h2>Region-aware exclusion meshes</h2>
<p>A mesh with authored surface regions can apply different clearances to different triangle
patches. The hierarchy is scene default → object override → region override. Region visibility does
not affect DRC; the region must be Active and have its DRC capability enabled. Named patches are
used for local distance only, while the full healthy host mesh remains responsible for physical
penetration and inside/outside containment.</p>
<div class="note">Hidden exclusion zones and hidden active DRC regions remain active. A DRC report
becomes stale after a scene edit because the saved report signature no longer matches the current
scene.</div>
""",
        ),
    ),
    "optimizer": (
        "Design tools",
        "Optimizer wizard",
        _page(
            "Optimizer wizard",
            """
<p>The optimizer starts like a classic setup wizard: first choose how much help you want, then reveal only the decisions needed for that route.</p>
<h2>Start: Simple, Complex, or Skip wizard</h2>
<ul>
<li><b>Simple:</b> the first page only chooses Simple. A second setup page then asks for exactly one parameter family: current, turns, dimensions, or position. Everything unrelated is hidden later.</li>
<li><b>Complex:</b> a second setup page allows several parameter families, including orientation. The later Freedom page exposes only those selected families and their relationships.</li>
<li><b>Skip wizard / Full:</b> skips parameter-family setup and goes straight to the common workflow. The Full Freedom page shows every supported property of the coils already in the scene, but starts with no coils, link groups, parameter families, or detailed freedom rows preselected.</li>
</ul>
<h2>Target and goal</h2>
<p><b>TARGET</b> selects the Measurement object(s) where the field matters. User-defined arrays such as 96-well centres remain exact.</p>
<p><b>GOAL</b> first asks what to match: a requested target-intensity specification or a stored field snapshot. Target-intensity mode then offers plain-language objectives such as putting the most samples inside the requested band, making intensity as uniform as possible, minimizing RMS error from the requested intensity, making direction consistent, or deliberately maximizing average/weakest/peak field strength. A short explanation under the dropdown defines exactly what each choice means.</p>
<p><b>Snapshot matching</b> compares one compatible Measurement target point-for-point with a stored snapshot. The snapshot supplies its exact stored local sample grid, which Workbench maps through the current Measurement target; imported snapshots from another scene are therefore usable when the Measurement shape/dimensions are compatible even if the scenes use different arbitrary world placement/orientation. Snapshot goals can minimize full-vector RMS difference, magnitude RMS difference, mean angular difference, or maximize the fraction of points meeting magnitude+direction tolerances. The Goal page also chooses how absolute orientation is treated: <b>Keep target-frame orientation fixed</b> preserves the exact local-frame Bx/By/Bz comparison, while <b>Exposure registration</b> treats absolute orientation as a nuisance pose. During scoring, optimizer-participating exposure sources may be rigidly rotated about the Measurement target's physical centre; their spatial field pattern and B-vectors therefore rotate together rather than merely rotating arrows at fixed samples. For asymmetric/imported Measurement geometry Workbench pivots around the physical volume centroid rather than the object transform origin. The candidate's stored world pose and DRC geometry are unchanged, and fixed/background sources remain world-fixed.</p>
<p><b>Exposure-registration telemetry:</b> When Exposure registration is active, Search Trace reports the inner nuisance-pose work separately from structural candidates: structural registration solves, unique registration-pose evaluations, additional registered-pose field-basis builds, registration-cache hits, Kabsch/inherited starting seeds, Fast coarse/standard/refined effort counts and cumulative inner-solve time. Fast deliberately spends less registration polish on disposable global-skeptic candidates and retains fuller refinement for promising/adaptive/DRC-rescored candidates; this changes search effort, not the exposure-scoring definition. The Best-so-far line also shows the fitted exposure RX/RY/RZ. These angles describe the scoring registration only; they do not replace or modify the candidate's physical world orientation shown in its setting.</p>
<h2>Freedom and relationships</h2>
<p>The Freedom page lists participating coils and lets each property use independent link groups. A Current Group A can contain one set of coils while Position Group A or Geometry Group A contains a different set. Properties marked Fixed are not changed; Independent gives that coil its own search variable for that property. In Simple/Complex mode, the earlier wizard page now acts only as a preset/filter: selected families arrive checked, but every exposed family remains editable here and may be unchecked before the search. The Freedom page is the final authority on what the optimizer may actually change.</p>
<p>In guided Simple/Complex searches, the Dimension section only shows rows compatible with the participating checked coils—for example, circular-only searches do not show racetrack or square dimensions. Full mode intentionally shows the complete existing-scene inventory. Dimensions include axial winding length, circular diameter, racetrack end diameter/straight length, and square/rectangular X/Y spans. Position includes shared X/Y/Z translation offsets plus a two-coil pair-spacing helper. Orientation is explicitly split into two different physical freedoms: <b>Rigid assembly rotation</b> rotates every linked coil centre around the link-group midpoint and rotates each coil plane with it, while <b>In-place coil rotation</b> leaves the centres fixed and only tilts/spins the coil planes. Guided Orientation setup defaults to rigid assembly rotation; in-place rotation is a separate opt-in. For a linked two-coil pair, rigid rotation therefore rotates the centre-to-centre axis itself, and pair spacing subsequently slides the coils symmetrically along that rotated axis. Every unlocked freedom also has an <b>Increment</b> that sets its discrete engineering/search resolution. Numeric Freedom-page bounds and increments remember the last values used when advancing from that page, while participating/freedom/row checkboxes are deliberately not remembered so each new search makes its allowed changes explicit. The <b>Defaults</b> button beside Next restores the numeric values derived when the optimizer opened on the current source scene without changing any selections. The default increments are 0.1 mA for current, 1 turn, 0.5 mm for dimensions and position, and 1° for orientation. The lattice is anchored to the source-scene value, so for example a 0.5 mm increment means source + integer multiples of 0.5 mm while the entered minimum/maximum remain hard bounds; finalists are never reported between increment values. The optimizer searches only coils that already exist in the scene and can search any number of their bounded variables simultaneously, including multiple independent link groups.</p>
<p>The search engine treats field-shape variables and excitation differently. <b>Fast</b>, <b>Balanced</b>, and <b>Full</b> now share one cheap-first policy at every structural dimensionality: broad exploration stays in simplified physics, while the expensive user-selected coil model is reserved for the final reality check. Current remains a linear excitation variable and is solved inside each structural candidate instead of being randomly searched alongside geometry. Structural reconnaissance uses the <b>Centreline</b> model and a deterministic spatially spread subset of target samples. Pure translation searches are source-cost aware: inexpensive analytic circular sources use exact batched observer shifts (moving the observer cloud oppositely is magnetostatically identical to moving the coil), while segmented Polyline/racetrack sources prefer a reusable validated Bx/By/Bz field stamp instead of repeating expensive exact Centreline evaluations at every translated pose. More general compatible pose/scale searches also use cached vector-field stamps. Rigid assembly rotations are evaluated directly in that compact state by rotating both linked centres and coil frames about the assembly centroid, so the fast scout uses the same geometry semantics as authoritative DRC/finalist evaluation. The stamp envelope is bounded from the rigid-group pivot: XYZ rotation changes the direction of a coil's orbital radius but cannot grow that radius, so wide 180/360-degree pose freedoms do not triple-count one physical motion into an enormous interpolation cube. Those compatible maps apply geometry to compact numeric transform/parameter state and sample the cache directly, so canonical, relation, symmetry, Sobol, and adaptive scout points do not construct disposable Workbench/Studio scenes. If a field stamp is unavailable, fails its interpolation check, or a transformed pose falls outside an otherwise-useful stamp, the scout keeps the same compact numeric transforms and evaluates the Centreline source directly at the reduced target points. It does not rebuild a Workbench/Studio candidate scene merely because caching was unsuitable. Geometry changes that cannot use either scene-free representation still fall back only to an exact Centreline scout solve, not to the expensive final coil model. A one-dimensional problem maps its permitted range cheaply, including the requested increment grid where practical. For recognized linked two-coil assemblies, <b>Fast</b> now uses a target-informed engineering funnel rather than broadly sampling the raw XYZ/Euler box: it challenges the target/reference field orientation first, follows shared placement next, profiles known pair relations such as spacing/diameter one dimension at a time, and only then asks a small whole-space Sobol skeptic plus compact adaptive challenge to prove the engineering answer wrong. Snapshot matching derives a compact target fingerprint from the stored vectors (coherent mean field axis plus a first-order gradient diagnostic), and both signs of a coherent field axis are challenged explicitly so a 180-degree magnetic-direction alternative is not hidden by geometric axis equivalence. Pair spacing receives an explicit 1-D profile/bracket refinement with other freedoms frozen, allowing trends such as “closer matches better” to be discovered in a handful of cheap probes before DRC backs a blocked optimum out to the nearest legal boundary. Unrecognized Fast problems retain the generic canonical/relation/Sobol fallback. <b>Balanced</b> and <b>Full</b> deliberately retain progressively broader relation, symmetry, Sobol and adaptive coverage so deeper effort modes are the place to search for unexpected/non-engineering answers. These physics guides are seeds and trajectories, never hard constraints, so arbitrary snapshots, gradients, asymmetric targets, and non-textbook solutions remain searchable.</p>
<p><b>Optimizer 2.2 cheap-first pipeline:</b> <b>A</b> freezes the scene and target samples, but for structural searches the source assembly receives only a geometry/DRC feasibility anchor; its expensive magnetic field is deferred. Source DRC is also classified by dependency: a pre-existing blocking relationship between objects unrelated to the optimizer-participating/structurally affected coils is recorded as an <b>inherited fixed DRC baseline</b> instead of making every candidate impossible. That row remains visible in raw DRC reports; adjustable-geometry violations, new violations, and any unexpected worsening of an inherited row still fail the hard gate. <b>B</b> builds the cheapest trustworthy field representation and maps the structural freedom broadly: source-cost-aware direct observer shifts or reusable stamps for pure translations, reusable vector stamps plus scene-free numeric transform evaluation for compatible general transforms, full-range cheap 1-D reconnaissance for a single freedom, and the physics/relation/symmetry/Sobol atlas for larger problems. <b>C</b> keeps current as an inner bounded linear solve, so every cheap structural sample is compared at its best permitted excitation instead of multiplying the outer search dimensions. Once an authoritative candidate's winning current is known, Workbench applies that current to the completed candidate scene and performs one direct requested-model field solve; those direct vectors and metrics, not the linear-basis reconstruction used to choose current, become the canonical authoritative registry result. <b>D</b> overlays the real winding/bobbin geometry as a <b>geometry-only DRC feasibility map</b>. Intersecting field-good landmarks are repaired toward nearby known-clear map points to learn useful clearance boundaries; each repaired geometry is then re-scored with the same cheap scene-free Centreline evaluator so it competes on its own field rather than inheriting the score of its colliding precursor. Once several legal/boundary solutions are known, high-dimensional searches run a second cheap local Sobol refinement around that feasible terrain and spend DRC only on the strongest few local neighbours. A small number of blocked local leaders are bisected back toward their known-clear seed, allowing the search to move along a complicated DRC surface instead of only retreating from the original unconstrained optimum. If the source and every retained magnetic landmark still have an adjustable-geometry collision, Workbench performs a bounded DRC-only rescue survey across the permitted structural freedoms to establish one or more legal anchors, then bisects back toward the field-good region. Rescue budgets now reserve coupled deterministic Sobol probes before one-axis endpoint challenges, so high-dimensional Fast searches cannot spend every rescue slot on single-coordinate moves. The nearest clear rescue setting is also kept in the authoritative vet pool as insurance, so a legal design space is not discarded merely because its best magnetic landmarks lie inside an exclusion. <b>E</b> adds small local neighbours around leading low-dimensional map landmarks and probes omitted physical-only construction freedoms at relevant bounds, still without expensive magnetic exploration. <b>F</b> optionally inserts a reduced-target physical-winding promotion screen for expensive segmented sources, so cheap-map survivors can be ranked with Bundled/current-sheet-scale physics before the most expensive calculation. <b>G</b> progressively promotes only the strongest structurally diverse survivors to the complete target grid, requested coil representation, full DRC, and electrical limits, opening additional authoritative slots only when the intermediate ranking is ambiguous or an earlier finalist fails. <b>H</b> stops: after a successful cheap map there is no exploratory full-model sweep, active-set search, boundary convergence, or local full-model adaptation. Fast, Balanced, and Full therefore have intentionally different search personalities: Fast exploits recognized target/assembly structure first and uses a small skeptic challenge, Balanced broadens the cheap map around that answer, and Full reserves the widest cheap global search for unexpected solutions; the effort also controls how many survivors receive the authoritative final vet. If the cached/design-map scout itself fails, Workbench retries with a plain reduced-target Centreline scout. Automatic Optimizer 2.2 does not switch to an exploratory expensive-model search; the older authoritative search path remains only for explicit legacy/programmatic jobs.</p>
<p><b>Candidate lifecycle:</b> The optimizer keeps one compact <b>flat master solution registry</b> across reconnaissance, DRC, repair, promotion and authoritative validation. During a run, the Search History header reports this durable population as <b>solutions discovered • active • retired • blocked • authoritative</b> rather than resetting stage-local evaluated/feasible/rejected counters; active, retired and blocked partition the current master population, while authoritative independently counts solutions that have reached full-model validation. A solution identity contains only structural freedoms; an inner-solved current is stored on each scout/promotion/authoritative observation and never becomes part of the canonical solution key. This lets the same geometry legitimately solve to different currents at different fidelities while retaining one shared solution history. Structural identities are canonicalized on the requested increment lattice and periodic orientation offsets are normalized, so equivalent 0°/+360°/−360° search coordinates merge into one physical solution record. A stage may <i>retire</i> a lower-ranked solution from further compute, but it does not delete it; later feasibility or higher-fidelity information can therefore re-rank earlier solutions. DRC-boundary repairs and rescue settings are peer solutions in the same population, with provenance cross-references describing where they came from instead of parent/child ownership. Repaired/rescue solutions are immediately re-scored with the cheap scene-free Centreline evaluator when available. Reduced-target physical promotion and authoritative validation accumulate as additional evaluations on that same solution; ranking always uses the highest trustworthy fidelity available, so an optimistic scout estimate cannot outrank its own later full-model result. Strict physical duplicates merge into one record, while different geometries that are field-equivalent remain separate solutions linked as an equivalence family so diversity/decimation can choose a representative without losing history. DRC checks walk progressively down the magnetic ranking until enough naturally clear alternatives are found within the selected effort budget, but the globally best magnetic scout, the relevant source baseline, and every retained strong-basin representative are mandatory DRC verdicts before that progressive pass may stop. A blocked strong solution first receives an axis-first repair: Workbench holds every other freedom fixed, prioritizes pair spacing/translation escape directions, bisects any blocked-to-clear one-coordinate route to the nearest legal lattice point, and re-scores that repaired geometry on its own field before falling back to a generic N-dimensional clear anchor. After clear terrain is known, deterministic one-freedom coordinate polish runs beside the local Sobol wave so simple improvements such as tightening pair spacing until DRC becomes active cannot be missed merely because the random local cloud perturbed several coordinates at once. Final authoritative slots are reserved across source, naturally clear and boundary-repaired roles so illegal magnetic winners cannot crowd out a strong feasible basin. The registry also prepares stable backend views for Finalists, Best feasible, Best overall, DRC boundary, Source baseline and Strong basins for the later completed-search browser. Optimization reports include a <b>Fidelity audit</b> for the source baseline, best overall scout, best naturally DRC-clear scout, best DRC-boundary solution and every authoritatively tested solution. The audit shows scout, promotion and authoritative metrics side-by-side, including model/sample counts and scout-to-authoritative deltas, so a poor final result can be separated from an optimistic cheap-map estimate.</p>
<h2>Completed-search Results Explorer</h2>
<p>After a search completes, Results becomes a browser over the same flat master registry rather than a finalist-only table. The left-side views select <b>Finalists</b>, <b>Best feasible</b>, <b>Best overall</b> (which may include DRC-blocked magnetic ideals), <b>DRC boundary</b>, <b>Source baseline</b>, <b>Strong basins</b>, or <b>All solutions</b>. These are rankings of the same canonical solution records, so one solution may appear in several views without being duplicated. The solution list keeps the top row compact and comparison-focused: snapshot searches show match, vector/magnitude RMS, angular error, mean field, optimizer-affected DRC clearance and fidelity, while field-definition searches show coverage, RMS error, uniformity, direction, mean field, optimizer-affected DRC clearance and fidelity. Expand a solution to drill into solved current and a nested Geometry outline with optimized freedoms plus per-coil resulting position, movement from the frozen source pose, orientation and design/path changes. This avoids a very wide one-line geometry description while preserving the details when they matter. <b>Selecting a solution is deliberately data-only:</b> Results shows a lightweight Compare-style <b>Comparison</b> overview and <b>Full statistics</b> page directly from the registry's highest trustworthy retained fidelity instead of rebuilding a complete Workbench/Plotly 3-D scene for every click. Snapshot-reference searches use the same Exposure-matching card language as Field Snapshot Comparison; field-definition searches use the same shared-target benchmark card language. The physical candidate scene is materialized only when the user explicitly asks for an operation that needs it.</p>
<p>The embedded <b>Comparison</b> and <b>Full statistics</b> pages are intentionally instantaneous registry views; they do not rerun field physics merely because the selection changed. <b>Compare</b> remains available when a fresh full-field calculation, detailed difference plots, or a multi-solution comparison is wanted. Snapshot-reference optimizations use the reference snapshot's stored <i>local</i> sample grid registered through the optimizer's selected Measurement target, exactly matching the optimizer objective even if the reference snapshot came from another scene placement/orientation. Exposure-registration optimizer results are first rematerialized in the scoring-only rigid registration that produced their authoritative metrics, so the full Compare workflow starts from that already-registered exposure rather than applying a second vector-only alignment. Registry solutions are rematerialized using the coil-modeling method recorded for that evaluation rather than whatever model happens to be selected later in the live source scene. A fresh full-field comparison of an authoritative solution is checked against its stored authoritative optimizer metrics and Workbench warns if match/RMS/angle/mean-field values drift beyond numerical tolerance. Field-definition optimizations open the shared-target benchmark using the optimizer's requested intensity and tolerance. <b>Save snapshot</b> performs the candidate field analysis and stores it as an imported frozen snapshot in the optimizer source scene, including the scoring field-source scene for later replay/current normalization, so the exposure can be kept without keeping the whole candidate model. <b>Open in new tab</b> is now the explicit path to inspect a candidate as a complete unsaved 3-D Workbench scene. <b>Apply to source</b> changes the participating source coils. <b>Append to source</b> instead preserves the originals and copies only participating coils whose transform or physical/path design differs; unchanged/current-only source coils are skipped so a current adjustment alone does not create an overlapping duplicate.</p>
<h2>Concurrent candidate workers</h2>
<p>The worker count in <b>Edit → Preferences</b> controls how many independent candidate processes may be
evaluated at once. Coarse sweeps, seed grids, integer samples, endpoint checks, and global
verification waves are independent and can share this pool. Bounded minimizers and coordinate
polishing stay sequential because their later trial points depend on earlier scores. Every worker
uses a private scene adapter; completed candidates are committed in their original order and use
candidate-specific deterministic random streams, preserving the fixed-seed trace and ranking.</p>
<p>The same preference also caps concurrent crash-isolated 2D and 3D render processes globally across map
windows. Optimizer and renderer workloads maintain separate queues, so running both kinds of work at
once can use more processes than the displayed per-workload limit.</p>
<p>Use the adjacent <b>Benchmark…</b> button to test the current computer and solver preferences.
The benchmark uses a frozen built-in scene, warms one persistent process pool, and recommends the
smallest count within 8% of the fastest result. More workers are not automatically faster: process
coordination and result transfer have overhead, and private adapters increase memory demand.</p>
<h2>Limits and review</h2>
<p><b>LIMITS</b> keeps the ordinary Workbench DRC as a hard feasibility gate and can enforce current, resistive-voltage, copper-loss, and current-density limits. The optimizer pursues practical automatic convergence with dimension-aware runaway ceilings; those ceilings are guards, not quality presets. The Review page restates the complete problem—including link groups and bounds—before any computation begins.</p>
<h2>Results</h2>
<p>Results are curated into <b>distinct finalists</b> rather than showing every nearby sampled setting. Candidates must remain effectively tied on the selected primary goal, survive dominance checks on the goal's secondary metrics, and be structurally different enough to represent another useful answer. The search history still contains every evaluated setting. Current is treated as the solved excitation for a geometry, so current-only variations do not create redundant finalists.</p>
<p>Finalists report the changed setting, actual coil currents, and the metrics relevant to the selected goal. Target-intensity runs show target coverage, P95–P5 spread, directional consistency, mean field, RMS target error, and DRC clearance; snapshot runs show point-match coverage, vector/magnitude RMS differences, angular error, and DRC clearance. The optimizer window is independent and its worker uses a frozen launch-time scene, so the main Workbench remains usable while a search runs. A finalist can be previewed, opened as a separate unsaved scene tab for side-by-side inspection, or applied back to the source scene as one undoable edit; Workbench warns if that source scene changed after the optimization began. <b>Copy report</b> copies the reviewed optimization plan, run/search/convergence summary, the leading distinct finalists, and full details for the currently selected finalist so the result can be pasted into notes, email, or another analysis workflow without losing the search context.</p>
<div class="warning">Optimizer results are engineering search results, not proof of a global optimum. The optimizer changes only selected properties of coils already present in the scene.</div>
""",
        ),
    ),
    "glossary": (
        "Reference",
        "Glossary",
        _page(
            "Field Workbench glossary",
            """
<p>This glossary gives the Workbench meaning of frequently used engineering, geometry, and
statistics terms. More complete explanations remain in their dedicated topics.</p>
<table>
<tr><th>Term</th><th>Meaning in Field Workbench</th></tr>
<tr><td><b>Affine transform</b></td><td>A 4 × 4 coordinate-mapping matrix. A rigid affine uses
rotation and translation only; a general affine can also scale or shear.</td></tr>
<tr><td><b>Ampere-turn (A·turn)</b></td><td>Drive current multiplied by winding turns. It is a useful
measure of coil magnetomotive strength, though geometry still determines the resulting field.</td></tr>
<tr><td><b>Bounding box / extent</b></td><td>The smallest axis-aligned box containing the sampled
object; extent is that box's X × Y × Z size.</td></tr>
<tr><td><b>B-time / B²-time</b></td><td>Brain View playback indices formed by integrating regional
spatial RMS |B| or its square over physical playback time. They compare magnetic-field exposure
magnitude and duration; they are not biological dose.</td></tr>
<tr><td><b>Cache</b></td><td>Reusable local data retained to avoid repeating a download or expensive
calculation. Clearing a cache does not change the authoritative source data.</td></tr>
<tr><td><b>Centroid</b></td><td>The arithmetic mean of included point or voxel positions—a geometric
centre that need not lie inside a concave object.</td></tr>
<tr><td><b>Coefficient of variation (CV)</b></td><td>Standard deviation divided by the absolute
mean, usually shown as a percentage. It becomes unstable when the mean approaches zero.</td></tr>
<tr><td><b>DRC / design-rule check</b></td><td>A geometric check for overlap, containment, and
minimum clearance between coil assemblies and exclusion geometry.</td></tr>
<tr><td><b>Field component</b></td><td>One signed vector component such as Bx, By, or Bz.
<b>|B|</b> is the non-negative vector magnitude.</td></tr>
<tr><td><b>Finite sample</b></td><td>A calculated sample whose numeric components are real and
neither infinite nor missing; invalid values are excluded or reported.</td></tr>
<tr><td><b>Fluxline</b></td><td>A numerically traced curve tangent to the displayed vector field.
Its seeding and visual density do not measure magnetic flux quantity.</td></tr>
<tr><td><b>Local/world frame</b></td><td>Local coordinates belong to an object before its scene
transform; world coordinates are the final shared scene positions.</td></tr>
<tr><td><b>Manifold / watertight mesh</b></td><td>A well-connected surface with the expected face
neighbourhood at every edge; watertight additionally means no boundary holes.</td></tr>
<tr><td><b>Normal</b></td><td>A vector perpendicular to a surface face. Consistent outward normals
are needed for reliable inside/outside interpretation.</td></tr>
<tr><td><b>Observer / sample point</b></td><td>A position where the magnetic solver evaluates B or H.
A map or measurement volume represents only its discrete sample set.</td></tr>
<tr><td><b>Percentile</b></td><td>A value below which a stated percentage of samples falls. P95 − P5
describes the central 90% spread while reducing sensitivity to extreme endpoints.</td></tr>
<tr><td><b>Playback RMS</b></td><td>Brain View space-time RMS over one finite physical playback:
<code>sqrt(B²-time / exposure time)</code>.</td></tr>
<tr><td><b>RMS</b></td><td>Root mean square: square values, average them, then take the square root.
Vector RMS error uses squared vector-difference lengths.</td></tr>
<tr><td><b>Scene role</b></td><td>How passive geometry is used: Visual only, Measurement volume, or
Exclusion zone. Passive geometry does not create or alter magnetic fields.</td></tr>
<tr><td><b>Snapshot</b></td><td>A stored measurement result plus the metadata needed for later
comparison. It is not a continuously linked live calculation.</td></tr>
<tr><td><b>Solver backend/model</b></td><td>The electromagnetic representation used to evaluate a
coil, such as Centreline, Exact turns, Bundled, or Current sheet.</td></tr>
<tr><td><b>Voxel</b></td><td>One three-dimensional image cell. Isotropic voxels have equal physical
size along all three image axes.</td></tr>
<tr><td><b>Worker (optimizer)</b></td><td>One spawned local Python process evaluating an independent
optimizer candidate with its own private scene adapter. It is not a separate open scene tab or a remote computer.</td></tr>
</table>
""",
        ),
    ),
    "ai_scene_recipe": (
        "Reference",
        "AI Help and Scene Recipes",
        _page(
            "AI Help and Scene Recipes",
            """
<p><b>Help → AI Help…</b> currently uses a deliberately manual clipboard bridge between Field Workbench and an
external AI chat. Field Workbench does not connect to an AI service, transmit your scene, store an
API key, or execute pasted code. The user remains the transfer step. The Scene Recipe/validation layer
is independent of that transfer method, so a future optional direct/API connection can feed the same
validated recipe path rather than requiring a second scene format.</p>
<p>The copied guide treats the external model as a <b>magnetic-field modelling assistant</b>, not merely a
JSON generator. It now distinguishes two things explicitly: what Scene Recipe v1 can construct, and the
larger set of current Workbench tools the assistant may recommend. This lets ordinary-language requests
such as “help me make this more uniform” produce both a concrete scene proposal and sensible next actions
without inventing unsupported recipe objects.</p>
<h2>Workflow</h2>
<ol>
<li>Open <b>Help → AI Help…</b>.</li>
<li><b>1 — Copy scene + AI hints:</b> choose an AI guidance level and the scene/DRC context options, then click <b>Copy scene + AI hints</b>. The copied contract always asks for the complete desired scene so the validated result can be sent to either destination later.</li>
<li>Paste that packet into the AI chat of your choice, describe the physical/magnetic goal in ordinary language beneath the USER INSTRUCTIONS marker, and let the AI return a Scene Recipe.</li>
<li><b>2 — Paste the AI response:</b> paste the complete returned JSON into the compact recipe box and click <b>Validate AI response</b>. Most users do not need to inspect or edit the JSON itself.</li>
<li>The guide asks for one valid Scene Recipe v1 JSON object. The model can place its plain-language interpretation, assumptions, and suggested Workbench follow-ups in optional <code>assistant_message</code>, <code>assumptions</code>, and <code>suggested_next_steps</code> fields.</li>
<li><b>3 — Review the AI interpretation and import:</b> the larger review pane shows those human-facing fields first, followed by warnings and the object list. After validation, choose <b>Modify current scene</b> to apply the complete result to the current tab, or <b>Paste into new scene</b> to create a separate tab and leave the source untouched.</li>
<li>Watch the activity bar at the bottom of the window during prompt generation/copying, validation, and import/replacement. It includes a live elapsed-time counter. When current-scene/DRC context is requested, the bar reports scene-object summarization and live DRC object-pair progress rather than merely indicating that Workbench is busy.</li>
</ol>
<p>AI Help treats every returned recipe as the <b>complete desired recipe-compatible object scene</b>. The
assistant is allowed to propose preserving, moving, changing, adding, replacing, or removing representable
objects; it does so by returning the final complete object set, not by sending imperative patch/delete
commands. <b>Modify current scene</b> asks for confirmation and atomically replaces the current object scene
while preserving scene-wide DRC settings and notes as one undoable operation. <b>Paste into new scene</b>
builds the same validated recipe in a separate new tab and leaves the source untouched. The external AI has
no live Workbench authority and must not claim a scene change, solve, DRC run, optimizer result, measurement,
or export occurred merely because it proposed one.</p>
<h2>Shining Star guidance</h2>
<p>The copied prompt uses a <b>build-site preflight</b> rather than a “fill in blanks” mentality. Before
creating each object, the external model checks what the object physically represents, what important
dimensions measure, its physical orientation/relationships, how spacing is defined, electrical direction
where relevant, and whether the proposed assembly could exist without impossible overlaps or contradictory
geometry.</p>
<p><b>AI guidance</b> controls how aggressively uncertainty stops construction:</p>
<ul>
<li><b>Exploratory — make a reasonable first pass:</b> favour a useful physically plausible scene and
normally ask at most one blocking question. If exact source reconstruction is unavailable, any scaffold
or approximation must be clearly labelled and must not masquerade as the named source design.</li>
<li><b>Balanced — ask only when it changes the build:</b> the default. Do the homework first using the
request, source/figures, relevant current-scene relationships, and harmless conventions. Build from the
most defensible interpretation and move nonfatal uncertainty to <code>review_questions</code>; normally
no more than 1–3 genuine blockers survive.</li>
<li><b>Strict — reproduce only what is known:</b> unresolved evidence-backed design-changing ambiguity
still stops construction, but Strict uses the same question-eligibility rules and does not invent candidate
interpretations or demand that the user manually restate an inaccessible paper.</li>
</ul>
<p>Clarification is treated as a <b>last resort, not brainstorming</b>. Before a blocking question is allowed,
the model should have tried the request, accessible source material (including figures/captions), and
relevant current-scene relationships; the answer must materially change the real build. The model is
explicitly told not to invent possible coil counts, shapes, topologies, dimensions, polarities, or other
candidate interpretations and then ask the user to choose between its own speculation. If retrieval fails,
it should say so and ask once for the smallest useful figure/excerpt instead of asking the user to restate
the whole paper.</p>
<p>For named source designs, geometry must not be inferred from the name/acronym. Arbitrary global axis/origin
choices are not blockers unless the apparatus must register to anatomy, fixtures, gravity, or another fixed
reference. Existing scene objects can provide useful relationship evidence when their role is clear, but are
not automatically sacred geometry that must be preserved.</p>
<p>For a true blocker the model can return <code>requires_clarification: true</code>, concrete
<code>clarifying_questions</code>, and an empty <code>objects</code> array. Field Workbench validates and
displays that advisory response but disables scene import until the question is resolved. A separate
<code>review_questions</code> list is explicitly non-blocking: the review pane displays it as
<b>Worth confirming</b> while leaving the scene importable. This lets Balanced guidance say “here is a
usable scene, but these details are worth checking” without turning every uncertainty into homework.</p>
<p>The guide also warns against claiming that two magnetic exposures are equivalent merely because one
centre-point magnitude matches. Field direction, spatial gradients, uniformity, and target-volume coverage
may all differ substantially. The model is instructed not to invent calculated field values, DRC clearances,
measurements, paper parameters, or optimizer results that Workbench has not actually supplied.</p>
<h2>Current Workbench capability awareness</h2>
<p>The clipboard contract now tells the assistant about current tools that may matter to a design even when
they are not Scene Recipe object types: point and Linear axis sensor inspection; 2D contours/fluxlines;
3D Full Volume, Slices and fluxlines; waveform animation and video export; measurement volumes and explicit
sample sets (including built-in well-plate centres); target statistics, snapshots and comparison; bench-data
Field-correction fitting; DRC; and the bounded coil optimizer.</p>
<p>It also names important <b>outside-Recipe-v1</b> workflows—Manual/WMM2025 Background fields, built-in
culture plates, UPENN-GBM imports, arbitrary STL/OBJ meshes, and portable/OpenSCAD/FEMM/GetDP/COMSOL export—so the
assistant can recommend them without fabricating JSON types that the parser cannot accept. The optimizer is
described accurately as changing bounded properties of coils already present in its frozen launch scene,
not as a topology generator.</p>
<p>Detached 2D and 3D map workspaces are also identified as <b>frozen launch-time scene copies</b>. If the
source scene is changed afterward, an already-open map workspace does not become a live view of that edit;
a fresh viewer should be opened when the changed model needs to be inspected.</p>
<h2>Why a Scene Recipe instead of a native save file?</h2>
<p>The native <code>.magpy.json</code> document contains application metadata and implementation
details that an AI has no reason to manufacture. Scene Recipe v1 exposes a smaller, stable set of
engineering concepts—coils, passive primitives, magnets, sensors, groups, and the built-in anatomy
objects—and Workbench itself converts those instructions into its native scene representation.</p>
<h2>Validation and safe defaults</h2>
<p>The recipe parser rejects unknown object types and properties rather than guessing. All numeric
values must be finite and valid for the requested object. Omitted coil drive current defaults to
0 mA and omitted magnet polarization defaults to zero, with a warning. Pasted content is parsed as
JSON data only; it is never evaluated as Python or another executable language.</p>
<h2>Current-scene context</h2>
<p>When <b>Include current scene context</b> is enabled, the copied guide appends a concise description
of current objects, transforms, coil construction/electrical settings, passive-geometry roles, and mesh
world bounds. Large triangle arrays and long sensor paths are summarized instead of copied in full.
Native object IDs are explicitly context references rather than recipe keys. A <code>recipe_type_hint</code>
marks object classes that v1 can reconstruct; objects without that hint are context-only and must not be
silently approximated during a complete-scene replacement. When preservation of such an unsupported object
matters, the assistant is told to disclose the limitation and favour a new-scene comparison.</p>
<p>Coil context includes the editable winding-model inputs needed to faithfully preserve a physical
design: turns/current/AWG, current convention, lead allowance, custom 20 °C wire resistance, enamel
build, packing factor, winding temperature, added former/hardware mass, and the full winding/bobbin
construction block. It also includes a compact <code>workbench_derived</code> section with Workbench-calculated
wire length, mass, resistance, voltage, loss, current density, winding-build, and physical-assembly
estimates. Those derived values are <b>read-only context</b>: an external model may use them in its
reasoning, but the copied guide tells it not to return them as recipe fields or claim them as its own
recalculation. The context groups editable inputs beneath <code>coil</code> for readability, but canonical
Scene Recipe coil properties are direct fields of the coil object. For robustness, validation also accepts
an accidental returned <code>coil: {...}</code> wrapper and flattens known editable fields with a warning;
conflicting flat/nested values are rejected.</p>
<p>Scene Recipe v1 accepts the corresponding editable winding inputs
<code>resistance_20_ohm_per_km</code>, <code>enamel_radial_mm</code>, <code>packing_factor</code>,
<code>temperature_c</code>, and <code>additional_mass_g</code> in addition to the existing construction fields.
They are optional and backward compatible. In complete-scene AI Help recipes they should be included when an existing
coil's physical model should survive unchanged; for newly proposed coils, the guide asks the external model
to omit optional winding/construction values it does not actually know and let Workbench apply defaults.
For robustness, a returned <code>resistance_20_ohm_per_km</code> value of 0 or <code>null</code> is interpreted
as “use nominal AWG copper resistance” with a validation warning rather than rejecting the entire recipe.</p>
<p>When <b>Include DRC rules/results</b> is also enabled, copying the guide uses the most recent design check if it is still fresh for the exact current scene and DRC settings; otherwise it runs a new check. It then
adds scene clearance settings, named surface-region placement/exclusion semantics, effective regional
clearances, per-coil status, and any collision/clearance issues. This gives an external model concrete
feedback for iterative requests such as moving a proposed coil arrangement away from the head. The
context is still only copied to the clipboard; Field Workbench sends nothing itself.</p>
<h2>Coordinate convention</h2>
<p>Scene Recipes use the same Workbench head frame: <b>+X right</b>, <b>+Y anterior</b>, and
<b>+Z superior</b>. Positions and dimensions are millimetres, rotations are XYZ Euler degrees, coil
drive current is milliamperes, and magnet polarization vectors are tesla.</p>
""",
        ),
    ),
    "files_limits": (
        "Reference",
        "Files, assumptions, and limits",
        _page(
            "Files, assumptions, and limits",
            """
<h2>Scene files</h2>
<p>Workbench scenes use Magpylib Studio JSON with a <code>field_workbench</code> metadata section for
physical coil specifications, passive geometry, DRC settings, snapshots, and free-form scene notes.
Open <b>Analyze → Notes</b> to keep design decisions, measurements, and build details with the
project. The application-level
solver preference, memory budget, and worker concurrency limit are stored in local preferences rather than in the scene so a
design remains portable between installations.</p>
<h2>Open-scene tabs</h2>
<p>The tab row beneath the main toolbar keeps several independent scene documents open in one
Workbench window. Normally <b>New scene</b>, <b>New default scene</b>, <b>Open</b>, and
<b>Open Recent</b> create or activate independent tabs. The one convenience exception is a pristine
packaged default scene: while that default has never been edited, opening or creating a scene reuses
its tab instead of leaving a throwaway startup tab behind. Once the default receives a scene edit,
it behaves like any other document and is never replaced automatically. Opening a file that is
already loaded activates its existing tab so two editors cannot unknowingly save to the same
path.</p>
<ul>
<li>Save, Save As, Undo, Redo, the scene tree, inspector, analysis tools, notes, and snapshots
operate on the active tab.</li>
<li>An asterisk after a tab name marks unsaved document changes.</li>
<li>Use the tab's close button or <b>File → Close scene</b> to close one document. Workbench asks
whether to save that scene if it has unsaved changes.</li>
<li>Closing the application checks every modified tab before exiting.</li>
<li>Selection and 3D camera/zoom state are retained separately for each tab. Modeless DRC,
snapshot, sensor, and result windows stay associated with the scene that created them.</li>
</ul>
<p><b>File → Import objects from scene…</b> copies selected objects from another saved scene without
replacing the current project. Complete groups or individual children can be chosen from a checkbox
tree. Object parameters, transforms, hierarchy, coil construction metadata, mesh roles/settings,
and visibility are copied; scene notes, snapshots, and scene-wide DRC settings are deliberately not
merged. The complete import is one undoable edit.</p>
<h2>Important assumptions</h2>
<ul>
<li>Coil conductors are ideal line currents or ideal current ribbons; skin effect, proximity effect, capacitance, core hysteresis, eddy currents, and frequency-dependent impedance are not solved.</li>
<li>Electrical calculations are resistive estimates based on entered wire data and temperature.</li>
<li>Passive bobbins and imported geometry do not alter the magnetic field.</li>
<li>Measurement and map results are discrete samples, not continuous mathematical guarantees.</li>
<li>Current-sheet accuracy depends on a dense, reasonably uniform winding distribution.</li>
<li>Fluxline plots are visualization aids; line density and seed placement do not represent flux quantity.</li>
<li>Imported meshes are only as trustworthy as their scale, topology, normals, and coordinate alignment.</li>
</ul>
<p>Use Exact turns or an independent model/measurement for final validation when a simplified
representation could materially affect the decision.</p>
""",
        ),
    ),
}


CATEGORY_ORDER = (
    "Introduction",
    "Coordinates and anatomy",
    "Generic bust",
    "SRI24 brain",
    "UPENN-GBM cases",
    "Coils and construction",
    "Electromagnetic solver",
    "Analysis tools",
    "Design tools",
    "Reference",
)


class HelpTopicsDialog(QDialog):
    """Searchable, navigable built-in documentation."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        enable_standard_window_controls(self)
        self.setWindowTitle("Field Workbench Help Topics")
        self.resize(1040, 720)

        outer = QVBoxLayout(self)
        heading = QLabel("Field Workbench Help")
        heading.setObjectName("dialogHeading")
        outer.addWidget(heading)

        search = QLineEdit()
        search.setPlaceholderText("Filter help topics…")
        search.setClearButtonEnabled(True)
        search.setAccessibleName("Filter help topics")
        outer.addWidget(search)
        self.search_edit = search

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.topic_tree = QTreeWidget()
        self.topic_tree.setHeaderHidden(True)
        self.topic_tree.setMinimumWidth(230)
        self.topic_tree.setMaximumWidth(380)
        self.topic_tree.setUniformRowHeights(True)
        self.topic_tree.setAccessibleName("Help topics")
        splitter.addWidget(self.topic_tree)

        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(False)
        self.browser.setAccessibleName("Help topic contents")
        self.browser.document().setBaseUrl(
            QUrl.fromLocalFile(str(_HELP_ASSET_DIRECTORY.parent) + "/")
        )
        splitter.addWidget(self.browser)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([270, 750])
        outer.addWidget(splitter, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.copy_region_exporter_button = buttons.addButton(
            "Copy Blender region exporter", QDialogButtonBox.ButtonRole.ActionRole
        )
        self.copy_region_exporter_button.setToolTip(
            "Copy the bundled Blender material-to-surface-region exporter script"
        )
        self.copy_region_exporter_button.clicked.connect(self._copy_region_exporter)
        self.copy_region_exporter_button.setVisible(False)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        outer.addWidget(buttons)

        self._category_items: dict[str, QTreeWidgetItem] = {}
        self._topic_items: dict[str, QTreeWidgetItem] = {}
        for category in CATEGORY_ORDER:
            category_item = QTreeWidgetItem([category])
            category_item.setFlags(category_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            font = category_item.font(0)
            font.setBold(True)
            category_item.setFont(0, font)
            self.topic_tree.addTopLevelItem(category_item)
            self._category_items[category] = category_item
            for key, (topic_category, title, _html) in HELP_TOPICS.items():
                if topic_category != category:
                    continue
                item = QTreeWidgetItem([title])
                item.setData(0, _TOPIC_ROLE, key)
                category_item.addChild(item)
                self._topic_items[key] = item

        self.topic_tree.expandAll()
        self.topic_tree.currentItemChanged.connect(self._show_current_topic)
        search.textChanged.connect(self._filter_topics)
        first = self._topic_items.get("overview")
        if first is not None:
            self.topic_tree.setCurrentItem(first)

    @property
    def topic_count(self) -> int:
        return len(self._topic_items)

    def select_topic(self, key: str) -> None:
        item = self._topic_items.get(str(key))
        if item is not None:
            self.topic_tree.setCurrentItem(item)
            self.topic_tree.scrollToItem(item)

    def _show_current_topic(
        self,
        current: QTreeWidgetItem | None,
        _previous: QTreeWidgetItem | None,
    ) -> None:
        if current is None:
            return
        key = current.data(0, _TOPIC_ROLE)
        if key in HELP_TOPICS:
            key = str(key)
            self.browser.setHtml(_themed_help_html(HELP_TOPICS[key][2]))
            self.browser.moveCursor(QTextCursor.MoveOperation.Start)
            self.copy_region_exporter_button.setVisible(key == "custom_surface_regions")

    def apply_workbench_theme(self, theme: str) -> None:
        """Re-render the open help page using the selected document colours."""

        current = self.topic_tree.currentItem()
        if current is None:
            return
        key = current.data(0, _TOPIC_ROLE)
        if key not in HELP_TOPICS:
            return
        scroll = self.browser.verticalScrollBar().value()
        self.browser.setHtml(_themed_help_html(HELP_TOPICS[str(key)][2], theme))
        self.browser.verticalScrollBar().setValue(scroll)

    def _copy_region_exporter(self) -> None:
        try:
            script = _BLENDER_SURFACE_REGION_EXPORTER.read_text(encoding="utf-8")
        except OSError as error:
            self.browser.append(f"<p><b>Unable to read bundled exporter:</b> {error}</p>")
            return
        QApplication.clipboard().setText(script)
        self.copy_region_exporter_button.setText("Copied Blender exporter")

    def _filter_topics(self, text: str) -> None:
        needle = str(text).strip().casefold()
        first_visible: QTreeWidgetItem | None = None
        for key, item in self._topic_items.items():
            category, title, html = HELP_TOPICS[key]
            searchable = f"{category} {title} {html}".casefold()
            visible = not needle or needle in searchable
            item.setHidden(not visible)
            if visible and first_visible is None:
                first_visible = item
        for category, category_item in self._category_items.items():
            category_item.setHidden(
                not any(
                    not self._topic_items[key].isHidden()
                    for key, values in HELP_TOPICS.items()
                    if values[0] == category
                )
            )
        if first_visible is not None:
            current = self.topic_tree.currentItem()
            if current is None or current.isHidden():
                self.topic_tree.setCurrentItem(first_visible)


def help_topic_titles() -> Iterable[str]:
    """Return public topic titles for lightweight tests and future indexing."""
    return (values[1] for values in HELP_TOPICS.values())
