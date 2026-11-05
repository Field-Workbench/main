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
    QHBoxLayout,
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
<p>Field Workbench is organized as a project-oriented magnetic design and analysis suite built around
Magpylib and Magpylib Studio. The <b>Project Manager</b> is the launch window: each tab represents one
open project and shows its model, maps, exposures, datasets, waveform resources, and snapshots.
Several project tabs may remain open at once, and application windows from different projects may
remain open side by side.</p>
<div class="note"><b>Project architecture:</b> each Project Manager tab owns one shared project context and stable resource registry. <b>Model Editor</b>, <b>Maps / Analysis</b>, <b>Waveform Editor</b>, <b>Exposure Editor</b>, <b>Dataset Manager</b>, and <b>Saved Results</b> are project-scoped applications. Project Manager owns project lifetime and saving; no child editor needs to remain open for the others.</div>
<p>A model can contain current sources, permanent magnets, uniform background fields, sensors, passive
geometry, measurement volumes, exclusion zones, and generated physical coil construction.</p>
<h2>What the application separates</h2>
<table>
<tr><th>Layer</th><th>Purpose</th></tr>
<tr><td>Electromagnetic scene</td><td>Sources and sensors used to calculate <b>B</b> or <b>H</b>.</td></tr>
<tr><td>Physical coil specification</td><td>Turns, wire, winding dimensions, bobbin, mass, resistance/self-inductance load properties, voltage, and power estimates.</td></tr>
<tr><td>Passive geometry</td><td>Visual objects, exclusion zones, and measurement volumes. These do not create a magnetic field.</td></tr>
<tr><td>Analysis results</td><td><b>Maps / Analysis</b> provides one unified Static/Scripted and 2D/3D/Brain viewer. <b>Saved Results</b> manages cached Static maps, cached Scripted maps, persistent Exposure analyses, and quantitative Measurement-volume snapshots in one place.</td></tr>
<tr><td>Waveforms</td><td><b>Waveform Editor</b> creates/imports reusable Current or Voltage signals. Ordinary authoring uses signal-generator controls; device-specific timing and encoding appear only at import/export boundaries.</td></tr>
<tr><td>Exposure protocols</td><td><b>Exposure Editor</b> stores timed steps of source-to-load drive groups. Each group references a project waveform by stable ID, selects one or more coils, and records series/parallel wiring.</td></tr>
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
<li>Start in the <b>Project Manager</b>. New projects are intentionally blank; add the objects you need in Model Editor or open an existing <code>.magpy.json</code> project. Existing projects open in their own tabs.</li>
<li>Open <b>Model Editor</b> for the active project, then add or load coils, magnets, sensors, and geometry. Add a Background field when the apparatus operates inside a known static bias such as the Earth's field.</li>
<li>Position and rotate objects in the scene.</li>
<li>For coils, enter turns, drive current, wire information, winding width/build, and optional bobbin dimensions.</li>
<li>From the <b>Project Manager</b>, select <b>Edit → Preferences</b> to choose the Light/Dark theme, coil field model, solver temporary-memory budget, worker concurrency, and optional solver diagnostics. These settings are application-wide, and theme changes apply immediately to open windows.</li>
<li>Open <b>Waveform Editor</b> to construct or import reusable electrical signals. Use <b>Exposure Editor</b> to create named reusable drive protocols and assign waveforms to coils. Open <b>Exposure Analysis</b> from Project Manager for authoritative time-domain regional/Measurement-volume analysis; <b>Analyze exposure…</b> in Exposure Editor remains a contextual shortcut to the same workflow. Open <b>Map Creator</b>, then choose <b>Static</b> for one fixed field state or <b>Scripted</b> for waveform/exposure-driven visualization over time.</li>
<li>Render Static or Scripted maps in <b>Maps / Analysis</b>. Use <b>Save map…</b> beside Fullscreen to cache an expensive rendered map with the project for instant reopening. Measurement-volume analysis and quantitative snapshots belong to <b>Static 2D/3D</b>; Scripted 2D/3D leaves that panel out entirely. Brain view uses its separate Brain Areas panel.</li>
<li>Assign anatomy or keep-out geometry the Exclusion zone role and use <b>Check design</b>.</li>
<li>Save the project from <b>Project Manager</b> as a <code>.magpy.json</code> document.</li>
</ol>
<p>The Project Manager dashboard places <b>Project components</b> and their applications on one shared row grid, so each component is locked to the exact height of the application row or rows that manage it.
Subtle connector arrows show the relationship row by row: Model → Model Editor; Maps / Analysis → Map Creator and Exposure Analysis;
Exposures → Exposure Editor and Waveform Editor; Data / Results → Dataset Manager and Saved Results. The large component labels are also shortcuts to their primary editor: Model opens Model Editor, Maps / Analysis opens Map Creator, Exposures opens Exposure Editor, and Data / Results opens Dataset Manager. The project header reports save status, the last saved time,
and a compact contents summary. The top toolbar is reserved for project file actions; application launchers live in the dashboard and the Project menu.</p>
<p>The Model Editor is deliberately limited to model-building and model-local analysis. Its toolbar provides <b>Sensor plot | Volume analysis | Optimize | DRC | Notes</b>. Open <b>Map Creator</b>, <b>Exposure Analysis</b>, <b>Exposure Editor</b>, <b>Dataset Manager</b>, <b>Saved Results</b>, and <b>Waveform Editor</b> from Project Manager.
In the 3D model viewer, drag from either geometry or empty space to orbit; a stationary click selects an object, Ctrl/Command-click or Shift-click toggles objects into a multi-selection, and a stationary click on empty scene space clears the selection.
In <b>Maps</b>, at the top of the left pane choose <b>Static | Scripted</b>,
then choose <b>2D | 3D | Brain</b> immediately below it. Changing either selector returns to the permanent
<b>Selector</b> tab and refreshes the appropriate setup/preview while every calculated result remains in the
same tab strip. In <b>Scripted</b> mode, a compact <b>Scripted source</b> section appears immediately before
<b>Camera</b>. Choose a quick built-in sine/square/triangle waveform for
the active Scene-options coils, or select a named saved <b>Exposure</b>. Arbitrary waveform import and editing live in <b>Waveform Editor</b>; sequencing and voltage/RL drive
live in <b>Exposure Editor</b>. Viewer trace selection stays in Maps, keeping project data separate from
visualization state.</p>
<p>The unified viewer keeps Static measurement information on the right instead of hiding snapshots in a left-pane disclosure. In ordinary <b>Static 2D/3D</b> Selector and result views, the <b>Measurement volumes</b> panel lists every eligible Measurement-role object from the frozen Maps launch scene. <b>Analyze selected…</b> or <b>Analyze all</b> calculates RMS, mean, P95, peak, uniformity, direction, and volume. <b>Analyze all</b> also opens a result window for each successfully analyzed volume. <b>View / save snapshot…</b> reopens an analyzed result without recalculating it; <b>Save result…</b> in that result window stores the exact sample coordinates, B-vector grid, statistics, and launch-scene provenance in the project. Scripted 2D/3D never shows this snapshot panel, and Brain uses its own Brain Areas panel. If the live model changed after Maps opened, the saved result is correctly marked stale rather than being relabelled as a result of the newer model.</p>
<p>Rendered map tabs can also be kept as project resources. <b>Save map…</b> and
<b>Fullscreen viewer</b> are centered beneath the plot; Save map stores the already-calculated Static or Scripted map. The result-tab strip spans the full viewer width, with <b>Load saved map…</b> at its left edge before <b>Selector</b>. Use it,
or double-click a saved map under <b>Maps / Analysis → Static maps</b> or
<b>Scripted maps</b> in Project Manager, to reconstruct the interactive viewer without a new field solve.
Saved maps are cached visualizations; Snapshots remain the quantitative Static Measurement-volume result resource. <b>Exposure analyses</b> are a separate persistent result type for authoritative time-domain exposure calculations. Open <b>Saved Results</b> to manage Static maps, Scripted maps, Exposure analyses, and Snapshots. Saved maps and Exposure analyses can be reopened, renamed, or deleted there; snapshots retain their existing view, compare, rename, import, and delete tools.</p>
<p>Map windows deliberately open before calculating. This lets you choose the plane, extent,
resolution, component, and display options before committing to an expensive field evaluation. <b>Show grid</b> lives under <b>Scene options</b>; the separate <b>Heatmap options</b> section appears only when the selected field display includes a heatmap/field layer.</p>
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
<li>The exposure-drive electrical response model may use each coil's resolved resistance and self-inductance. Resistance is calculated from the winding by default; self-inductance receives an approximate air-core geometry estimate. Either can be overridden with a measured/known value. Mutual inductance between separate coils is not solved.</li>
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
whole well grid with it. These same discrete points are used by volume analysis, saved results,
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
        "Coordinates and anatomy",
        "Built-in Generic bust",
        _page(
            "Built-in Generic bust",
            """
<p><b>Object → Generic bust</b> adds a packaged human head/neck/shoulder reference mesh for
visualization, design-rule exclusion, measurement, saved results, and optimizer geometry.</p>
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
<p>The same section also contains <b>Exposure sampling depth</b>. Exposure Analysis turns each authored
surface patch into an inward near-surface volume at that depth (2 mm by default), assigns interior
samples by the nearest authored surface triangle, and uses those weighted volumes in the same regional
analysis engine as Brain Areas. The depth is a geometric field-sampling setting, not an assertion about
local skin thickness or biological dose. The user-facing range is 0.01–30 mm so the tool can examine
the superficial-to-deeper head region without accidentally requesting an effectively whole-object shell.
Changing it changes the target geometry identity and makes
older preparations for that bust stale rather than silently reusing them.</p>
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
        "Brain model and atlas",
        "Built-in SRI24 brain",
        _page(
            "Built-in SRI24 brain",
            """
<p><b>Object → SRI24 brain</b> adds a packaged atlas-derived brain surface for visualization,
design-rule exclusion, measurement, saved results, and optimizer geometry. The object is aligned to
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
        "Brain model and atlas",
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
    "lpba40_model": (
        "Brain model and atlas",
        "LPBA40 regional model",
        _page(
            "LPBA40 regional model",
            """
<p>Brain View uses the <b>LPBA40 parcellation distributed with SRI24 v2.0</b> to turn the
reference brain into named anatomical regions. This topic describes the atlas data, label indexing,
Workbench browsing hierarchy, numerical sampling, and display meshes. These are model-method details;
operating Brain View is documented separately under <b>Field views → Brain View</b>.</p>
<h2>Source atlas and label indexing</h2>
<p>The bundled <code>lpba40.nii.gz</code> and <code>LPBA40-labels.txt</code> come from the official
SRI24 v2.0 label archive. Field Workbench losslessly gzip-compresses the NIfTI volume for packaging but
does not alter its voxel values, affine, or label table. The volume uses the same 240 × 240 × 155
SRI24 RAS-millimetre grid described in the registration topic.</p>
<p>The LPBA40 volume contains <b>57 integer values: background 0 plus 56 non-background anatomical
labels</b>. The upstream text table supplies the authoritative label ID, name, and display colour for
every value present in the volume. Numerical analysis remains tied to those original integer IDs;
Workbench does not renumber atlas regions.</p>
<p>Most paired structures use names such as <code>L_superior_frontal_gyrus</code> and
<code>R_superior_frontal_gyrus</code>. Cerebellum and brainstem are unpaired in the bundled LPBA40
table. Left and Right therefore mean anatomical subject side in the canonical head frame, not screen
left/right.</p>
<h2>Workbench browsing hierarchy</h2>
<p>The upstream LPBA40 lookup table is flat. Field Workbench adds a presentation hierarchy so the
region table can be browsed without flattening all 56 labels into one list. The eight major groups are:</p>
<ul>
<li>Frontal lobe</li><li>Parietal lobe</li><li>Occipital lobe</li><li>Temporal lobe</li>
<li>Insula</li><li>Limbic structures</li><li>Deep gray matter</li><li>Cerebellum and brainstem</li>
</ul>
<p>Each named bilateral structure then contains Left and Right leaves. The hierarchy is validated as
an <b>exact partition</b>: every non-background LPBA40 label occurs exactly once and no label is
silently omitted or duplicated. The lobe/division grouping is a Workbench browsing aid, not an
upstream LPBA40 taxonomy.</p>
<h2>Numerical atlas samples</h2>
<p>Analysis begins from LPBA40 voxel centres, not from the rendered region surfaces. Atlas voxel
centres are mapped from SRI24 RAS millimetres through the documented rigid SRI24 → NAS/LPA/RPA head
transform, then through the selected SRI24 brain object's scale, rotation, and translation.</p>
<p>To bound computation, Workbench deterministically retains up to <b>1,200 voxel-centre samples per
LPBA40 label</b>. If a label contains more voxels than that cap, points are selected at evenly spaced
positions through that label's stable source-voxel ordering. Every retained point is assigned the
amount of source voxel volume it represents:</p>
<div class="formula">sample weight = source voxels in label × placed voxel volume ÷ retained samples in label</div>
<p>Consequently, a small and a large structure are not given equal influence merely because both hit
the per-label sampling cap. Parent rows combine their children with these volume weights. The reported
<b>Volume</b> comes from the complete source voxel counts, not the retained sample count.</p>
<div class="note"><b>Sample count is not anatomical volume.</b> Sampling is a numerical reduction of
the atlas. Spatial statistics use the retained points and their represented-volume weights; atlas
volume uses all source voxels belonging to the row.</div>
<h2>Display shells are separate from analysis</h2>
<p>The coloured LPBA40 surfaces in Brain View come from a packaged display cache generated from the
same unchanged label volume using a two-voxel visual stride and VTK SurfaceNets. It contains one shell
for each of the 56 non-background labels. The cache is validated against the bundled atlas and label
table checksums when loaded.</p>
<p>Those shells provide interactive anatomy, selection, and highlighting only. They do not determine
field sample locations or regional volume. Reducing visual mesh density therefore does not change the
underlying LPBA40 numerical statistics.</p>
<h2>Interpretation boundary</h2>
<p>SRI24 and LPBA40 are reference-atlas anatomy rather than an individual subject. Brain View reports
magnetic-field statistics over this atlas geometry; it does not infer patient-specific tissue
boundaries, conductivity, induced electric field, or biological response.</p>
<div class="note">SRI24 provenance, licensing, requested citation, surface derivation, and the exact
fiducial transform are documented in the two preceding Brain model and atlas topics.</div>
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
<li>The default load resistance is the operating winding resistance. <b>Electrical Load → Resistance → Override</b> may replace it with a measured or otherwise known value without changing the wire/construction description.</li>
<li>Self-inductance receives a compact air-core estimate from the mean current-path perimeter, turns, and approximate winding-bundle size. It is intended as a useful default rather than a precision coil solver; <b>Electrical Load → Inductance → Override</b> should be used for measured/manufacturer values. The magnetic <b>field correction factor</b> is not applied to this inductance estimate: it calibrates magnetic output and is not, in general, a physically valid inductance multiplier.</li>
<li>For peak-sine current, RMS current is peak current ÷ √2; DC/RMS mode uses the entered current directly.</li>
<li>Resistive loss is <code>I_rms² × R_load</code>.</li>
<li>Resistive voltage is <code>|I_drive| × R_load</code>.</li>
<li>Current density is <code>I_rms ÷ copper cross-sectional area</code>.</li>
</ul>
<div class="note"><b>Source/load boundary:</b> waveforms describe the electrical source. Coils carry the default physical load properties. Exposure drive groups may optionally override resistance and/or self-inductance per selected coil for that route without changing the model-side coil values.</div>
<div class="warning"><b>Inductance limitation:</b> the automatic value is a self-inductance estimate for an isolated air-core winding. Coupling between nearby coils can change the effective inductance of a series/parallel assembly; mutual inductance is not included in this model yet.</div>
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
measurement volumes, saved results, comparisons, and optimizer evaluations all include every
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
saved results, exports, and optimizer field evaluations. Each coil is compiled independently and may
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
<p><b>Project Manager → Edit → Preferences → Worker concurrency</b> sets the maximum worker count for every open
project. The same limit caps simultaneous crash-isolated 2D and 3D render jobs across all map windows;
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
        "Measurements and calibration",
        "Sensors and magnetometer probes",
        _page(
            "Sensors and magnetometer probes",
            """
<h2>Sensor Plot</h2>
<p>Sensor Plot evaluates the selected field at the pixels of a scene Sensor. A sensor with no explicit
pixel array is treated as one pixel at local <code>[0, 0, 0]</code>. Linear axis/path sensors plot
values against path distance or sample index and can export CSV. The window is non-modal, opens before
calculation, and uses the shared determinate progress/cancellation system after <b>Calculate</b> is
pressed. Its status bar records the actual coil solver representation and total elapsed time.</p>
<h2>Magnetometer probes</h2>
<p><b>Add Object → Sensor → Magnetometer probe</b> opens the probe library. A probe is a physical scene
object with a visible housing, an explicit local origin, and one or more coloured sensitive points.
Every channel stores its own local point and unit sensitive direction. When <b>Calculate probe
reading</b> is pressed in the inspector, Workbench evaluates the complete world field separately at
every sensitive point and projects it onto that channel's transformed local axis. The displayed
combined magnitude is formed from those reported channel values; it is not a single-point |B|
measurement when the sensing elements are spatially separated.</p>
<p>The bundled SENSYS FGM3D definition uses the manufacturer's documented 26 × 26 × 149 mm envelope
and single-axis reference distances. Its scene origin is the geometric centre of that housing, so the
object position can be indexed directly from the probe body; its X/Y/Z sensitive points remain at their
documented offsets from the reference edge. A contrasting panel marks the sticker on the local −Y
face. The manufacturer's sticker symbols define +X across the face, +Z toward the reference end, and
+Y inward through the sticker. A red end face marks the cable connection on local −Z.</p>
<p>The bundled MEDA FM300 / FVM400 entry uses the manufacturer's 4.00 × 1.00 × 1.00 inch rectangular
probe drawing with its scene origin at the body centre. Local +X runs from the cable end toward the tip,
+Y follows the documented width direction, and +Z points down toward the base. The X sensitive point is
1.54 inches from the tip. Y and Z share the other ring-core centre 0.68 inches from the tip, with all
three centres 0.49 inches below the top face.</p>
<p><b>Create custom…</b> supports box-envelope definitions, arbitrary channel points/directions,
optional sticker/label and cable-connection faces, and optional source metadata. Imported
<code>.fwprobe.json</code> files use the same schema. The complete validated definition is embedded in
each saved scene, so reopening the project does not depend on an external library file.</p>
<div class="warning">A probe housing is a visual model and does not itself alter the magnetic field.
Confirm the physical drawing revision, axis convention, sensor position, and calibration before treating
a simulated reading as metrology.</div>
""",
        ),
    ),
    "field_map_2d": (
        "Field views",
        "2D field map",
        _page(
            "2D field map",
            """
<p>Open <b>Maps</b>, choose <b>Static</b> or <b>Scripted</b>, then choose <b>2D</b>. The persistent selectors stay at the top of the shared left pane; changing either selector returns to the common Selector tab while calculated results stay listed.</p>
<p>The 2D field-map workspace samples a regular <b>XY, XZ, or YZ plane</b> and displays a field
component or magnitude as a heatmap, in-plane fluxlines, or both. It is intended for examining a
cross-section of a scene while keeping the main Workbench available.</p>
<h2>Frozen-scene lifetime and tabs</h2>
<p>Each launch opens a normal non-modal top-level workspace and rebuilds the complete model from a
private frozen launch-time scene copy. Later changes to the source scene—and even closing the source tab—cannot
alter or close the existing map window. Reopen the viewer after scene edits when a fresh calculation is
needed.</p>
<p>The permanent <b>Selector</b> tab shows the captured 3D scene with the current sampling plane
overlaid. Plane, Offset, and View width can therefore be positioned independently of the main-window
camera. Maps uses one shared tab strip across both Static/Scripted modes and all visual views. Result names include both the view and the source, for example <b>2D\\Static - 1</b>, <b>2D\\Sine - 1</b>, or later a saved exposure name. Each view/source combination numbers independently. Switching 2D/3D/Brain returns to Selector, while existing result tabs stay listed and can be revisited without recalculation.</p>
<h2>Display modes</h2>
<ul>
<li><b>Heatmap</b> displays the selected B/H component or vector magnitude over the sampled plane.</li>
<li><b>Fluxlines</b> numerically trace in-plane paths through the sampled vector field. When Fluxlines or Combined is selected, a viewer-local <b>Fluxline options</b> section appears with visual seed density plus independent minimum-field and conductor-distance cutoffs. Visual seed density is not magnetic flux quantity.</li>
<li><b>Combined</b> performs the required sampled phases and overlays heatmap and fluxlines.</li>
</ul>
<p><b>Contour lines</b> are an independent scalar-field overlay and can be enabled with Heatmap, Fluxlines, or Heatmap + fluxlines. In Fluxlines-only mode, enabling contours adds the scalar sampling pass needed to draw the contours without displaying the heatmap itself.</p>
<p>In <b>Scripted</b> mode, <b>Field view</b> is limited to Heatmap because realtime playback does not render fluxlines.</p>
<h2>Object outlines</h2>
<p><b>Object outlines</b> adds cross-sections of visible solid scene geometry wherever the sample
plane actually cuts it. <b>Objects…</b> provides independent selection of outline-capable objects.
Passive primitive/imported meshes, supported solid magnets, and enabled physical coil construction can
be outlined. Hidden objects remain hidden and line-only sources/sensors do not contribute a solid
cross-section.</p>
<h2>Result interpretation</h2>
<p>Manual colour limits affect visualization only; they do not clip the stored field vectors. The map
is a finite sampled plane, so smooth colour interpolation between samples is a display aid rather than a
continuous-field guarantee.</p>
<p>Scripted 2D results are viewer-only: they do not show the Static Measurement-volumes snapshot panel or launch cumulative exposure analysis from the result tab.</p>
""",
        ),
    ),
    "field_map_3d": (
        "Field views",
        "3D field map",
        _page(
            "3D field map",
            """
<p>Open <b>Maps</b>, choose <b>Static</b> or <b>Scripted</b>, then choose <b>3D</b>. The two persistent selectors and pane width stay fixed while each mode/view pair retains its own setup controls and renderer state while hidden.</p>
<p>The 3D field-map workspace samples a regular volume and renders the selected field representation
through a stack of slices, a full-volume display, or integrated field paths. It uses the same detached
frozen launch-time scene model as the 2D field map. <b>Field view</b> uses the same Heatmap / Fluxlines /
Heatmap + fluxlines choices as 2D. For 3D heatmaps, <b>Render mode</b> inside <b>Map options</b> chooses
between Slices and Full Volume independently of that Field view choice.</p>
<p>In <b>Scripted</b> mode, <b>Field view</b> is limited to Heatmap; 3D playback does not render fluxlines.</p>
<h2>Selector and result tabs</h2>
<p>The permanent <b>Selector</b> tab shows the captured 3D scene and planned slice stack. The slice
nearest the volume centre is a translucent filled selector plane; other projected slices are thin
wire-frame outlines. Centre X/Y/Z, Cube width, Slice direction, and Slices update this preview live,
even while a result tab is active.</p>
<p>Maps uses one shared tab strip for Static and Scripted 2D, 3D, and Brain results. Result names include both the view and source, such as <b>3D\\Static - 1</b> or <b>3D\\Square - 1</b>. Changing the View selector returns to the permanent <b>Selector</b> tab and refreshes that selector for the chosen mode, but previously calculated result tabs remain available.</p>
<h2>Resolution and performance</h2>
<p>Point count grows with all three resolution axes, so a dense 3D grid can be substantially more
expensive in time and memory than a 2D plane. The selected solver model and the shared solver-memory and
worker-concurrency preferences still apply.</p>
<h2>Viewing results</h2>
<p><b>Fullscreen viewer</b> opens a separate true-fullscreen interactive copy of the rendered
Plotly/Qt WebEngine view while leaving the embedded viewer in place. Press <b>Esc</b> or <b>F11</b> to
return. Manual colour limits change display scaling only; they do not alter or clip stored numerical
vectors.</p>
<p>Scripted 3D results are viewer-only with respect to Measurement volumes and cumulative exposure analysis. Video export remains available from the playback controls.</p>
""",
        ),
    ),
    "brain_view": (
        "Field views",
        "Brain View",
        _page(
            "Brain View",
            """
<p><b>Brain view</b> analyzes the magnetic field over the bundled LPBA40 parcellation registered to a
scene <b>SRI24 brain</b>. This topic describes how to use the viewer. Atlas provenance, label indexing,
regional sampling, and mesh derivation are documented separately under <b>Brain model and atlas</b>.</p>
<h2>Opening Brain View</h2>
<ul>
<li>Open <b>Maps</b>, choose <b>Static</b> or <b>Scripted</b>, then choose <b>Brain</b> from the persistent <b>2D | 3D | Brain</b> selector. Switching either selector swaps only the relevant setup/renderer page; it does not open another top-level workspace.</li>
<li>If the launch scene contains one SRI24 brain, it is used automatically. If several are present, select the intended SRI24 brain before opening Maps. Brain is disabled when no unambiguous SRI24 brain is available.</li>
<li>The LPBA40 atlas follows that object's position, rotation, and scale. The <b>Map options</b> controls the field-sampling cube; it does not move the atlas.</li>
<li>The default map volume is centred at X/Y/Z = <code>0, 0, 50 mm</code> with a <code>300 mm</code> cube width.</li>
<li>In <b>Static</b> mode, <b>Calculate static analysis</b> creates a fixed-current result. In <b>Scripted</b> mode, edit the shared playback controls in the left pane and choose <b>Create playback</b>; switching 2D/3D/Brain does not discard the protocol.</li>
</ul>
<p>Each calculated result is a frozen launch-time scene copy. Brain results use the shared workspace
tab strip and use source-aware names such as <b>Brain\\Static - 1</b> or <b>Brain\\Sine - 1</b>. Their Brain Areas filters, sorting,
column visibility, widths, expansion state, highlights, and cached statistics are independent of
Selector and other result tabs. Switching 2D/3D/Brain returns to Selector but does not remove any
calculated result.</p>
<h2>Brain Areas panel</h2>
<p>The table contains Whole brain, eight major Workbench browsing groups, anatomical structures, and
Left/Right leaves where LPBA40 provides bilateral labels. <b>Collapse</b>, <b>Expand</b>, and
<b>Expand major</b> change tree expansion only. Sorting remains hierarchical: siblings are sorted
within their anatomical parent rather than flattening the anatomy into one global list.</p>
<p>The Maps/Analysis workspace is responsive: on smaller windows the left controls and right analysis panel shrink before the centre viewer, and long control rows wrap inside the scrolling sidebar. The window keeps a modest minimum size so actions remain reachable without requiring a desktop-sized layout.</p>
<p>The Static Measurement volumes and Brain Areas forms scroll when the window is short. Filters and results retain readable minimum heights, and Add/Remove controls stay below the filter grid. The divider between filters and results still adjusts their relative heights.</p>
<p>Scripted playback keeps compact Play, Reset/Stop, Loop, speed, and timeline controls fixed at the bottom. Each signal and the sequencer has its own panel above this bar. Drag horizontal dividers to resize the map and traces; double-click a divider to reset the layout, or focus it and use Up/Down arrows. Trace plots redraw to fit their panel height, and each signal has an independent Invert toggle. Each Drive panel also has an <b>Audio</b> checkbox: only one source can be heard at a time, and audio always starts off when opening a viewer. Audio monitors the original authored source command, before electrical slew/load response, with DC removed and quiet normalized volume. Playback speed also changes pitch. Stop/Reset, seeking, Loop, and source switching follow the viewer; hiding or closing it stops audio. Older saved playbacks must be rebuilt to enable audio. Video exports remain silent. Hover over a signal heading for its summary statistics. <b>Export traces…</b> appears beside <b>Export video</b> below the viewer.</p>
<h2>Restore controls from a map</h2>
<p>Saving a map updates its result tab to the saved name, keeping the view prefix (2D, 3D or Brain). Duplicate names receive the same unique suffix shown in saved resources. Renaming a saved resource updates its linked open tabs. Cancelled or failed saves leave the tab title unchanged.</p>
<p>Select a rendered result tab and use <b>Restore controls from map</b> beside <b>Save map…</b> to copy that result's recorded calculation controls back into the sidebar. A new result freezes these settings when its render job is submitted, so later control edits do not rewrite older tabs. Saving a map preserves the controls record for reopening in another session.</p>
<p>This restores supported display, sampling, local coil/object selections and opacity controls, plus Scripted source inputs when recorded. It does not recalculate or replace the selected map, change the model's currents/calibration, rewind saved exposure definitions, change the solver method/worker policy, or restore camera/timeline and Brain table filters. Review those separately before recalculating if the model or source has changed.</p>
<p>Older maps may only retain explicit rendered grid and display metadata. FWB reports partial recovery and leaves missing or invalid settings unchanged. It cannot reconstruct unknown source settings or custom opacity curves; recovered old colour bounds become manual limits because the original Auto/manual choice is unknown. A map with no usable metadata leaves the controls unchanged. Unsupported record versions are rejected.</p>
<p>Before starting a background map calculation, FWB checks enabled manual colour limits. With Log scale, each enabled limit must be greater than zero; when both are enabled, the lower limit must be below the upper limit. Uncheck a limit to use Auto, or turn off Log scale to allow zero or negative limits. This check also applies after restoring map controls.</p>
<p>The viewer/panel divider remembers its preferred width while the Maps/Analysis window is open. Static 2D/3D Measurement volumes start closed so the map receives the workspace width, while Brain starts with its Brain Areas panel open. A manual choice is remembered separately for each visual view. Scripted 2D/3D results with Live measurement analysis start with their panel open. In 2D/3D use <b>Show measurements</b> / <b>Hide measurements</b>; in Brain use <b>Show brain areas</b> / <b>Hide brain areas</b>.</p>
<h2>Filters and highlights</h2>
<p>Each Brain Areas result keeps its own filters, sorting, visible columns, and column widths. In Scripted results the table follows the <b>current frame</b>; cumulative exposure statistics are handled separately by Exposure Analysis. Filters can use displayed numeric metrics or <b>Side = Left/Right</b>:</p>
<ul>
<li><b>Filter</b> is an exclusion rule. With several Filter rules, a row is excluded only when it matches <b>all</b> active Filter rules. Ancestors needed to show the path to a surviving child remain visible.</li>
<li><b>Highlight</b> never removes rows. Matching rows receive the selected colour; later matching Highlight rules take colour priority.</li>
</ul>
<p>Selecting a row highlights its retained numerical sample locations in that result viewer. The
visible highlight may be thinned for responsiveness; numerical statistics retain the full configured
analysis sample set.</p>
<h2>Traces</h2>
<p>Scripted Brain View <b>Traces</b> can include LPBA40 rows. Each selected row contributes its
volume-weighted spatial RMS trace over the visual preview. <b>Select major</b> checks the eight major groups and <b>Clear</b>
removes brain-region traces.</p>
<div class="note">Definitions for RMS, percentiles, uniformity, direction, and atlas weighting are in
<b>Statistics and exposure metrics → General spatial statistics</b>. Cumulative time-domain definitions and their research context are in <b>Exposure statistics and research basis</b>.</div>
""",
        ),
    ),
    "saved_exposures": (
        "Exposures",
        "Saved exposures",
        _page(
            "Saved exposures",
            """
<p>Open <b>Project Manager → Exposure Editor</b> to create and manage named time-varying drive protocols for the active project. Exposure Editor is a first-class suite window and can remain open without Model Editor. Exposure definitions are stored in the current <code>.magpy.json</code> project, so saving the project saves them with its other resources.</p>
<p>The exposure browser lists <b>Name / Steps / Duration / Updated</b> and shows the selected exposure’s <b>Notes</b> in a resizable panel on the right. Publication URLs are clickable. In <b>Edit exposure</b>, the multiline Notes box sits beside the name and duration controls. Use it to record derivation, references, and experimental details. Existing descriptions are preserved as notes, and notes travel with saved projects and exported exposures.</p>
<h2>Managing protocols</h2>
<p>Use <b>New…</b> to create an exposure, or double-click a row / choose <b>Edit…</b> to
change it. <b>Duplicate</b> makes an independent exposure copy while preserving its intended waveform references; <b>Rename…</b> changes only the exposure name.
Select multiple rows with Ctrl/Shift to <b>Delete</b> them; deletion can be restored with project Undo.</p>
<p>Set the physical <b>Exposure duration</b> in seconds, minutes, or hours, then add named <b>Sources</b>. Sources are edited directly in the table: choose a saved waveform, set its phase offset, and select <b>Continuous</b> or <b>Once per step</b>. A positive phase offset starts further into the selected waveform. Sources have stable IDs, so renaming a source does not break its routes.</p>
<p><b>Continuous</b> follows one waveform clock from exposure time zero regardless of routing changes. <b>Once per step</b> starts one waveform cycle from the configured phase whenever a routing step using that source begins. A short routing step truncates the shot; a longer step commands zero after the single cycle. If the source is used again in a later step, the single shot starts again. This is useful for pulses, bursts, action-potential shapes, and other triggered waveforms.</p>
<p>The <b>Routing schedule</b> contains timed steps. Edit each step duration directly in the table; <b>Add step</b> starts the new row with the duration of the previous step so repeated timing patterns are quick to build. <b>Duplicate</b> is contextual: on a step it clones the complete step, including duration and all drive groups; on a drive row it clones only that drive group inside the same step, including source, coil wiring, connection type, and drive-local R/L overrides. A duplicated drive initially references the same coils, so edit its coil assignment before saving if the original remains in that step. Each drive group connects an exposure source to selected coils in <b>Series</b> or <b>Parallel</b>. The same source may feed several drive groups in one step; those groups are treated as parallel routes at the source terminals, while each group keeps its own internal Series/Parallel wiring. Different source entries remain electrically independent even when they reference the same waveform, so separate source rows can represent isolated output channels. Empty steps are idle intervals. Source playback mode controls the waveform clock; compatible coil RL state can still carry across a routing boundary when the wiring is unchanged.</p>
<p><b>Repeat routing schedule until exposure ends</b> repeats the switching program independently of waveform periods. When unchecked, all routes turn off after the last step; any remaining exposure time is idle. The physical duration can also truncate a routing step.</p>
<p>Waveforms own their Current/Voltage signal, slew and voltage-source output resistance. Coils supply the default resistance and self-inductance. A drive group can optionally override either value per selected coil for that route. When several groups share one voltage source, its output resistance is applied once to the combined parallel network rather than once per group. Compatible coil groups retain RL current across step and routing-cycle boundaries. Removing or rewiring a group uses ideal switching: flyback paths, mutual inductance and switching transients are not modeled.</p>
<h2>Preview and analysis</h2>
<p>In <b>Maps → Scripted source → Saved exposure</b>, <b>Duration</b> limits the preview from time zero. It is capped at the physical exposure duration. Preview looping only replays those frames; it does not extend the exposure or accumulate additional dose.</p>
<p><b>Create playback is presentation-only.</b> It resolves the requested preview frames and viewer traces, but it no longer performs authoritative cumulative exposure statistics as a hidden prerequisite. This keeps a long physical exposure from turning a short visual preview into an hours-long analysis job.</p>
<p>Viewer drive/sequencer/sensor traces cover the <b>preview duration only</b> and retain at most 65,536 display samples, so a short preview is never visually compressed by a much longer exposure. Their small trace summaries describe what is plotted; they are not authoritative full-exposure results.</p>
<p><b>Exposure Analysis</b> in Project Manager opens the dedicated authoritative analysis workflow; when a project contains several saved exposures it first asks which one to analyze. <b>Analyze exposure…</b> in Exposure Editor remains a contextual shortcut that opens the same workspace for the selected saved exposure. The first page groups <b>Region sets</b> (SRI24/LPBA40 brain targets and eligible authored near-surface region targets) separately from explicit <b>Measurement volumes</b>. Selecting a target shows its details at the top and, beside the target list, a table of saved analyses for that exact exposure/target pair. For a Surface regions target, the current inward sampling depth is editable directly on this page over the supported 0.01–30 mm range; applying a different depth changes the project geometry setting and older preparations become Stale. Browsing the page never calculates fields. Choose a saved row and <b>Open selected analysis</b> to reuse it, or choose <b>Create new analysis</b> to build a fresh preparation while preserving the saved analyses already listed. Exposure Analysis always characterizes the complete saved physical exposure and is independent of Scripted playback previews.</p>
<p>Exposure Analysis reuses work explicitly rather than silently. Verified physical cycles can compress a long full exposure into transient + repeated cycle + remainder packets, and exact repeated current states are shared by regional time-domain calculations. A saved row marked <b>Current</b> still matches the live exposure/model/target dependency fingerprint and can be opened directly; <b>Stale</b> rows remain readable historical results. <b>Create new analysis</b> always performs a fresh preparation and stores it as a new saved analysis.</p>
<div class="note"><b>Exposure Analysis is cached automatically.</b> The prepared regional basis and authoritative replay source are stored as soon as preparation succeeds, and each completed analysis configuration is added to the same project cache automatically. This marks the project changed; normal <b>Project Save / Save As</b> still writes the updated cache to the <code>.magpy.json</code> file. Matching cached configurations are reused rather than rerun. Under <b>Saved Results → Exposure analyses</b>, <b>Manage cache…</b> can remove individual derived configurations while keeping the expensive preparation, while <b>Delete cache</b> removes the whole parent analysis.</div>
<h2>What an exposure stores</h2>
<p>Physical duration; named sources with waveform IDs, playback mode and phase offsets; routing-step durations, repeat setting, source IDs, coil IDs, wiring, and any optional per-coil R/L overrides attached to a drive group. Source samples are not duplicated. Existing version-3 drafts are upgraded with one continuous source per referenced waveform and a physical duration equal to one routing pass; version-4 sources normalize to Continuous playback. Review migrated durations after opening.</p>
<p>Editing a saved waveform updates every source that references it. Duplicate waveforms have independent IDs. Press <b>Save exposure</b> to commit the draft; <b>Cancel</b> discards it.</p>
<h2>Electrical diagnostics</h2>
<p>The routing table is kept as the main Exposure-editor workspace. Select a <b>drive</b> row and choose <b>Drive info…</b> when you want its electrical details; they are no longer shown permanently beneath the table. If that drive shares its source with other groups in the same step, the information dialog previews the complete shared-source parallel network so source current, terminal voltage, and output-resistance effects match playback. The Add/Edit drive-group dialog also includes same-source peer groups in its live preview.</p>
<p>Diagnostics also call out important modeling assumptions: geometry-estimated coil inductance, the uncoupled-coil approximation when several coils are combined, and ideal current steps that mathematically require voltage impulses. Current sources do not yet carry a compliance-voltage rating, so required source voltage remains diagnostic.</p>
<p>The compact preview is calculated from one saved waveform starting at zero coil current. Actual playback follows each source's Continuous/Once-per-step mode and may carry RL state across compatible steps. The diagnostic preview is not a full-exposure analysis.</p>
<h2>Portable exposure files</h2>
<p><b>Export…</b> writes one selected protocol as a versioned <code>.fwbexposure.json</code> file. Every project waveform referenced by its drive groups is bundled as a sibling resource while the protocol itself keeps only stable references. <b>Import…</b> restores those bundled waveforms under local resource IDs and rewrites the source waveform references, or can copy exposures and their referenced waveforms from another Field Workbench project.</p>
<p>After importing into another model, edit the protocol and check each drive group's coil selection. Missing coil references are reported; the stored coil label is descriptive only and does not silently assign the protocol to a replacement coil.</p>
<div class="note"><b>Electrical resolution:</b> Each drive group uses the selected coils' model-side R/L defaults, with optional exposure-local per-coil resistance/inductance overrides applied before its Series/Parallel reduction. Groups sharing one source are solved together as parallel source-terminal routes, with voltage-source output resistance applied once to the combined network; different sources remain independent. Maps playback samples the complete exposure into actual per-coil current timelines and uses those currents directly for the magnetic-field basis reconstruction. Output and sensor <b>Traces</b> remain viewer settings and are not stored in exposure definitions.</div>
""",
        ),
    ),
    "waveform_editor": (
        "Playback",
        "Waveform Editor",
        _page(
            "Waveform Editor",
            """
<p><b>Project Manager → Waveform Editor</b> opens the native waveform application for the active project. Saved waveforms are first-class project resources and the Project Manager owns the global Light/Dark theme.</p>
<p>Each catalogue entry contains exactly one signal. Select an entry to preview it. In navigation mode the left catalogue owns the resource actions: <b>Create new / Edit</b>, <b>Rename / Duplicate</b>, <b>Import… / Export…</b>, then full-width <b>Delete</b> and <b>Close</b>. The catalogue right-click menu mirrors the selected-waveform commands (Edit, Rename, Duplicate, Export, Delete), and the Delete key also removes the selected waveform. Catalogue-level import can select several signals from one source file and creates a separate waveform entry for every selected signal. A workflow strip directly below the menu bar always identifies the current state as <b>Waveform navigator</b>, <b>Waveform encoding</b>, or <b>Waveform editor</b>. Its right-side <b>Show / hide notes</b> control remains reachable even when Notes is fully collapsed, and the Notes divider is draggable while the panel is open.</p>
<h2>Signal-generator authoring</h2>
<p>Project waveforms are electrical signals: choose <b>Current</b> or <b>Voltage</b>, then use compatible engineering units (<b>mA/A</b> or <b>mV/V</b>). Normal creation does not ask for point counts, sample intervals, bit depth, or encoding. Field Workbench chooses its internal representation automatically.</p>
<p><b>Create new</b> begins with a waveform-definition step in the centre card directly beneath the persistent viewer, in the same location used by <b>Waveform information</b> while browsing. The workflow strip reads <b>Waveform encoding</b>. Enter the waveform name, choose <b>Current</b> or <b>Voltage</b> and its engineering unit, then choose <b>Create waveform</b>. Waveform tools remain hidden until this definition is confirmed. The definition then moves to a read-only <b>Waveform encoding</b> card on the left with a full-width <b>Edit</b> button; signal type and units do not change again unless that button is used. Editing an existing catalogue waveform skips the creation step and opens directly in <b>Waveform editor</b> state with its saved encoding already shown on the left.</p>
<p>Once the encoding is confirmed, the workflow strip changes to <b>Waveform editor</b>. The left column adds a <b>Components</b> list beneath the read-only encoding summary, followed by <b>Component actions</b> for <b>Edit</b>, <b>Clone</b>, <b>Move up</b>, <b>Move down</b>, and <b>Delete</b>; <b>Save waveform / Cancel</b> sit at the bottom of this left authoring panel. The selected component is mirrored in the main plot. The centre <b>Waveform tools</b> card contains every way to add waveform content, with hard minimum button sizes so labels remain readable when the window is narrowed. Choosing <b>Square wave</b>, <b>Sine wave</b>, <b>DC / pause</b>, <b>Action potential</b>, <b>Free form</b>, or <b>Import waveform…</b> replaces the tool chooser in place with that tool's controls. Every tool inherits the waveform's already-confirmed signal type and units. The persistent main viewer above shows the complete resulting waveform live and follows the available viewer-panel height; no second builder window or second preview is opened. Square/Sine use frequency, duration or cycle count, peak-to-peak magnitude, and offset; changing duration updates the cycle count and changing cycle count updates duration, while Square adds duty cycle and Sine adds phase. DC / pause adds a constant zero or non-zero hold. Action potential exposes event timing and signal levels; invalid timing order or level relationships are highlighted red while remaining freely editable. Free form is edited graphically in the persistent viewer: the complete waveform stays visible for scale, double-clicking inside the free-form component adds a control point, dragging points reshapes/repositions them, and dragging the small mirrored tangent handles adjusts curvature. Tangent handles may be dragged beyond the visible graph limits for stronger overshoot/undershoot while the anchors remain constrained to the component domain. If a true tangent handle leaves the waveform plot rectangle, a filled proxy marker stays pinned to that plot edge in the same direction so it can always be grabbed and pulled back. <b>Add point</b> inserts a new selected anchor in the largest gap. When a point is selected, <b>Vertical position</b> shows its value in the waveform's configured physical units and can be edited numerically as an alternative to dragging. <b>Reset curve</b> asks for confirmation before replacing the authored curve. <b>Reset selected tangent</b> restores the selected anchor's tangent to a neutral default, while <b>Fit handles</b> temporarily expands the plot limits to reveal all authored handles; <b>Home</b> returns to the ordinary waveform scale. Interior points can be deleted directly or with the tool control. There is no output-point or sampling control—the authored curve is converted to the internal playback representation automatically. <b>Apply</b> commits the previewed component and returns to Waveform tools; <b>Cancel</b> discards only that tool preview and returns to Waveform tools.</p>
<p>Every new generator tool previews as an append; the first component initializes an empty waveform. Each component keeps its own internal time grid. In the main viewer, use Home, pan, visible rectangle zoom, wheel zoom, and PNG save in every mode; the <b>Components</b> selection control appears only while Create/Edit authoring is active. While no tool preview is active, select a component in either the graph or the left Components list and drag it over another component in the graph to reorder it. For very short components, <b>Move up</b> and <b>Move down</b> provide one-step reordering from the left panel. <b>Clone</b> duplicates any exact authored component immediately after the selected one; <b>Delete</b> removes it and rebuilds the remaining waveform. <b>Edit</b> reopens generated components and editable runtime-encoded imports with their original controls; double-clicking either kind in the graph or Components list does the same. Generic sampled imports remain lossless, cloneable, deletable, and reorderable, but do not expose authoring controls unless their encoding profile explicitly retained runtime-editable source settings. Generator authoring is entirely embedded in this workspace; the retired standalone builder windows are no longer part of the editor.</p>
<p>While the waveform-definition form or an inline tool/import preview is active, project Save and other authoring actions are locked until that inner action is applied or cancelled. Press <b>Esc</b> to cancel the active inline preview first; while explicitly editing the encoding it cancels that encoding edit and returns to Waveform tools; otherwise it cancels Create/Edit with the usual unsaved-change confirmation. File-menu Import actions route to the same inline tools during confirmed Create/Edit. The editor also prevents a cross-window request or project close from silently throwing away an unapplied preview.</p>
<p>The left <b>Waveform encoding</b> card is read-only during normal authoring. It shows the signal's name, electrical type/units and slew setting; choose <b>Edit</b> to reopen those waveform-level controls in the centre card. <b>Voltage</b> waveforms additionally carry an <b>Output resistance</b> in ohms; 0 Ω represents an ideal voltage source, while values such as 50 Ω can describe a real generator output. Current waveforms do not use an output-resistance field. Coil resistance and inductance are load/model properties, not waveform properties. Peak-to-peak magnitude and duration are derived from the waveform itself. Builder-created waveforms also retain a versioned authoring recipe separately from the sampled playback trace: Square/Sine/Action-potential parameters and Free-form Bézier anchors/tangents are preserved in physical terms, while internal sample density is not part of the recipe. If samples are ever changed outside the builder, the prior recipe is kept only as detached provenance so it cannot be mistaken for an exact editable definition.</p>
<h2>Import and translation</h2>
<p><b>Import waveform…</b> lives in the Waveform tools card during confirmed Create/Edit, alongside the waveform generators. It is the general waveform-file importer: it can append a native Field Workbench <code>.fwbwave</code>/<code>.fwbwave.json</code> signal, an ordinary <b>text/CSV</b> table, or a source described by a reusable format profile. After choosing it, the relevant import controls temporarily replace the generator selector beneath the persistent viewer so the prospective appended component can be previewed before Apply. Text/CSV sources expose delimiter/header/encoding, a time column (or fixed interval), one signal column/name, and scale/offset in a vertically scrollable mapping form, so the controls remain usable even when the Waveform-tools area is short; signal type and units are not requested because the component inherits the waveform encoding. Compatible A↔mA or V↔mV structured inputs are converted automatically to the waveform unit; a source with the opposite electrical quantity is rejected until the waveform encoding is explicitly changed. <b>Apply</b> appends one lossless sampled component and <b>Cancel</b> discards the preview. Catalogue-level Import remains the larger multi-signal workflow and can create several independent waveform resources at once.</p>
<p>Reusable encoding profiles can declare timing and/or value mapping as <b>user-entered at import time</b> instead of hard-coding those values in the profile. Such an import uses the same embedded runtime-settings editor for every encoding: the original raw samples, the declarative encoding profile, the entered sample interval, and any entered two-point value mapping are stored with the authored component. Selecting that component later and choosing <b>Edit</b> reopens the same controls and raw samples in the persistent viewer. The bundled <b>Unsigned 8-bit DAC</b> entry is simply one built-in profile configured this way (newline-delimited text samples with a first-line point count, runtime sample interval, runtime two-point mapping); it has no DAC-only import path. Duplicating it produces an ordinary user profile that can be changed in the same Format Builder, and user-created encodings can opt into the identical runtime-timing behavior.</p>

<h2>Save and export</h2>
<ul>
<li><b>Save waveform</b> creates or updates a native waveform resource in the current <code>.magpy.json</code> project. Create/Edit uses a private working copy until that button is pressed; <b>Cancel</b> restores the saved catalogue version.</li>
<li>Closing Waveform Editor with unsaved changes asks whether to Save, Discard, or Cancel. Project Manager resolves that private working copy before deciding whether the project file itself needs saving, so a draft cannot disappear merely because the project had otherwise been clean.</li>
<li><b>Rename</b> and Edit keep the waveform's stable project ID, so existing exposure drive groups continue to reference the same source and immediately resolve its new name/data. <b>Duplicate</b> creates a new independent waveform ID. A waveform referenced by any saved exposure cannot be deleted until those references are changed or the exposure is removed.</li>
<li><b>Export…</b> always operates on the selected <em>saved</em> catalogue waveform, never an unsaved working copy. Save or Cancel edits first.</li>
<li><b>Export FWB waveform…</b> writes the portable <code>.fwbwave</code>/<code>.fwbwave.json</code> native format without quantization.</li>
<li><b>Export CSV…</b> writes a simple physical two-column table (time in seconds plus the selected waveform values) for inspection or interchange.</li>
<li><b>Export DAC…</b> maps native engineering values to the 8-bit DAC representation. Define code 0 and code 255; quantization/clipping affects only the exported copy.</li>
<li><b>Export Rigol…</b> maps the waveform onto the generator's ±5 V range at export time and uses the vendor encoding only while writing RAF/CSV output.</li>
</ul>
<div class="note"><b>Suite boundary:</b> Waveform Editor owns waveform construction, electrical source metadata, import translation, editing, and export. Exposure Editor consumes project waveform resources and owns coil assignment, sequencing, and the source-to-load connection. The waveform supplies its signal type and slew rate; voltage waveforms also supply their source output resistance. Coil resistance and inductance are kept with the model/load rather than waveform records.</div>
""",
        ),
    ),
    "playback_overview": (
        "Playback",
        "Playback overview",
        _page(
            "Playback overview",
            """
<p>Playback turns one static magnetic scene into a <b>finite physical current timeline</b>. The same
scripted source can be viewed in 2D, 3D, or Brain mode. In <b>Maps → Scripted</b>, open the compact
<b>Scripted source</b> section immediately before <b>Camera</b>. <b>Built-in waveform</b> is deliberately a
lightweight current-only source: choose Sine, Square, or Triangle plus amplitude, frequency, duration, and
slew limit, and it drives the active coils selected in Scene options. <b>Saved exposure</b> runs the complete
source/load schedule authored in <b>Exposure Editor</b>, including multiple waveform sources, timed steps,
series/parallel coil wiring, voltage-source output resistance, and the model-side coil R/L response. The viewer-local <b>Traces</b> section appears below <b>Camera</b> and applies to either source without becoming part of the saved exposure. Choose <b>Create playback</b> in the active view to calculate the selected source
with the current map/camera settings. A rendered result tab is a frozen product of the scene, scripted
source, and view settings used when it was created. Static and Scripted results share the same result-tab
strip. Tab names identify both the view and source, for example <b>2D\\Static - 1</b>, <b>3D\\Sine - 1</b>,
or <b>Brain\\SAM360 - 1</b>. Each view/source combination has its own sequence. Very slow playback and high FPS can create thousands of preview frames; Field Workbench warns before jobs of 2,000 frames or more. Create playback resolves only the requested visual preview; it does not run a separate full-exposure statistics pass. The render-job tab reports sample-grid preparation, fixed-field/coil-basis work, playback-timeline work, Brain Areas playback-frame metrics when applicable, WebGL packaging, and viewer opening.</p>
<h2>Sources of time dependence</h2>
<ul>
<li><b>Quick built-in waveforms</b> in Maps generate one Sine, Square, or Triangle current command for the currently active Scene-options coils. They do not contain voltage/load or sequencing controls.</li>
<li><b>Saved exposures</b> reference reusable project waveforms and arrange them into timed steps containing one or more source-to-coil drive groups.</li>
<li>The exposure resolver combines each waveform source with its selected coils and Series/Parallel topology, producing the actual current timeline for every driven coil before field values are evaluated.</li>
<li>Whole-sequence looping repeats the authored exposure schedule; compatible RL states carry across waveform repeats/step boundaries as described under <b>Saved exposures</b>.</li>
</ul>
<h2>Viewer traces</h2>
<p><b>Traces</b> is a viewer-local section at the bottom of Maps → Scripted, below Camera. <b>Drive trace</b> plots the source command over the preview duration only; a saved exposure can contribute several simultaneous drive traces when it contains independent drive groups. <b>Sequencer trace</b> likewise shows the active-coil/exposure-step lane only across the preview. <b>Sensor trace</b> reveals a table of eligible sensors/probes, where each checked row chooses its own sample point (where applicable) and field signal such as |B|, Bx, By, or Bz. Brain-region traces are available in Brain view. These choices apply to the current Scripted viewer and are not stored in exposure definitions.</p>
<h2>Waveform Editor and format profiles</h2>
<p>Waveform ingestion belongs in <b>Waveform Editor</b>. Catalogue <b>Import…</b> separates raw table entry from encoded files. <b>Manual input…</b> provides an editable table for typing or pasting rows directly, then maps one optional time column plus any number of electrical signal columns. Encoded files use the <b>Available encodings</b> list: select the encoding first, then choose the file. Built-in unsigned 8-bit DAC and native Field Workbench waveform JSON appear in the same list as bundled/user format profiles. With <b>Field Workbench waveform JSON</b> selected, Browse can choose either a portable <code>.fwbwave</code>/<code>.fwbwave.json</code> file or a <code>.magpy.json</code> project; choosing a project lists its saved waveform resources in <b>Waveforms to import</b> so any subset can be copied into the current project. The chosen decoder is used explicitly; Field Workbench does not guess an encoding from the filename or file contents. Every checked decoded signal is saved as its own catalogue waveform. Exposure Editor therefore selects waveform resources, never channels inside them.</p>
<ul>
<li>Text profiles can define encoding, record/field separators, comments, skipped records, and time/value field numbers.</li>
<li>Binary profiles can define signed/unsigned integer or floating-point fields, byte order, header size, record stride, and byte offsets.</li>
<li>The timeline can use a fixed interval or a stored time field. Sample count can run to EOF, be fixed, or be read from a text record/binary header.</li>
<li>A general <code>output = raw × scale + offset</code> transform converts stored codes to the declared engineering quantity/unit.</li>
<li>Profiles are portable <code>.fwbformat.json</code> files and can be imported/exported for sharing.</li>
</ul>
<div class="note">Format selection is explicit: choose an encoding before choosing its file. Field Workbench does not probe or guess an encoding, so an unusual or user-defined lab format is never silently interpreted as something else.</div>
<h2>Preview, signals, and units</h2>
<p>Catalogue import lists decoded waveform signals as soon as a file is chosen; the main encoded-file importer does not expose raw sample rows. For a source project, the same list shows the project's saved waveform names and keeps each waveform's own metadata separate. Check one or more entries and each becomes an independent project waveform in one operation. Raw rows are shown only in <b>Manual input</b>, where they are intentionally editable. During confirmed Create/Edit, import remains a one-component authoring action: select one decoded signal and preview it as an append in the persistent waveform viewer. Text/CSV mapping there remains an inline component-import tool, while native/profile inputs preserve their decoded physical timeline and samples and compatible engineering units are converted to the waveform's fixed encoding.</p>
<p>Imported signals require at least two finite samples, with one value per time and strictly
increasing finite timestamps. Duplicate or reversed timestamps are rejected. Time is normalized to
seconds; signal values retain their declared engineering unit until conversion into the drive's
current or voltage units.</p>
<p>Waveform import is restricted to the two electrical signal types used by playback: <b>Current</b> and <b>Voltage</b>. Integer-coded source files must apply the intended scale/offset and choose compatible engineering units during import.</p>
<h2>Profiles, waveform data, and exposures</h2>
<ul>
<li><b>Format profile (.fwbformat.json):</b> a reusable decoding recipe, not the waveform samples.
Profiles are saved in the application's user configuration and can be edited, duplicated, deleted,
or shared with <b>Export profile…</b> / <b>Import profile…</b>.</li>
<li><b>Project waveform:</b> a first-class single-signal resource stored in the current <code>.magpy.json</code> project and listed under <b>Waveforms</b> in Project Manager.</li>
<li><b>Field Workbench waveform JSON (.fwbwave / .fwbwave.json):</b> the portable native interchange format for one electrical signal with physical times, quantity, units, source metadata, and optional authoring recipe.</li>
<li><b>Project exposure:</b> stores only stable waveform IDs, coil IDs, step timing, and Series/Parallel drive-group topology. It never embeds another copy of the waveform samples or load R/L values.</li>
<li><b>Portable exposure (.fwbexposure.json):</b> bundles every referenced project waveform as sibling resources while the exposure protocol itself keeps stable waveform-ID references. Import creates local waveform resources and rewrites those IDs, so the exposure can travel without the source project.</li>
</ul>
<p>Format profiles may define timing as <b>User-entered sample interval at import</b> rather than storing a fixed interval or reading timestamps from each record. These runtime-timed imports retain the raw source samples, chosen interval, and profile in the authored component so timing and source points remain editable later. Profiles may independently use a <b>User-entered two-point mapping at import</b>. The bundled unsigned 8-bit DAC file is treated as raw 0..255 codes: during confirmed Create/Edit it inherits the waveform's already-selected Current/Voltage type and units, then asks only for <b>Point delay</b> plus the engineering values represented by <b>Code 0</b> and <b>Code 255</b>.</p>
<h2>Physical time is not video time</h2>
<p>The saved exposure and authored waveform clocks define physical time. Preview FPS controls only how
often the Scripted viewer samples that interval for animation; exported-video slow motion controls only
presentation speed. Neither changes the saved exposure duration.</p>
<div class="note"><b>Preview loops are not repeated exposures.</b> Letting an animation loop on screen,
pausing the UI, or exporting the same result more slowly does not add B-time, B²-time, pulse counts, or
physical exposure duration. Change the saved exposure itself when the experiment physically repeats.</div>
<h2>Rendered frames and Brain frame metrics</h2>
<p><b>Create playback</b> builds only the requested visual preview. Rendered frames drive the animation and
video. In Brain mode, Field Workbench also prepares the Brain Areas metrics needed for those rendered
frames so the current-frame table and selected region traces can follow the animation. These are viewer
data for the preview, not a cumulative full-exposure statistics pass.</p>
<p>Preview FPS therefore affects animation work and the number of Brain current-frame metric rows, but it is
not an authoritative exposure-analysis sampling setting.</p>
<h2>Authoritative exposure analysis</h2>
<p>For cumulative RMS/mean/peak, B-time/B²-time, timing, thresholds, FFT, or frequency-weighted screening,
open <b>Exposure Analysis</b> from Project Manager and choose the saved exposure to analyze. The contextual
<b>Analyze exposure…</b> shortcut in Exposure Editor opens the same workflow. It has
its own physical analysis clock, verified-cycle compression, progress reporting, and persistent cache. It
does not depend on whether a Scripted map has been created or paused on a particular frame.</p>
""",
        ),
    ),
    "playback_regions": (
        "Playback",
        "Playback region panel",
        _page(
            "Playback region panel",
            """
<p>Enable <b>Scene options → Live measurement analysis</b> before creating a Scripted 2D or 3D map. The right-hand panel follows playback and scrubbing and shows magnetic flux density statistics for each Measurement object. Well plates use their defined sample points. Other Measurement objects use their configured point or volume sampling. Select an object row to highlight its samples. Use Static 2D/3D to create quantitative Measurement snapshots.</p>
<h2>Brain Areas</h2>
<p>Brain View keeps its separate LPBA40 <b>Brain Areas</b> panel in Static and Scripted modes. New Scripted results show current-frame spatial metrics only; cumulative exposure-summary values are no longer calculated as a prerequisite for opening the animation. Use <b>Show brain areas</b> / <b>Hide brain areas</b> or the viewer divider to change its width. Atlas sample weighting, label grouping, and anatomical volume are described under <b>Brain model and atlas → LPBA40 regional model</b>.</p>
<p>Current-frame columns use the spatial definitions in <b>General spatial statistics</b>. Full-duration time-domain exposure statistics are intentionally separated from playback. Open <b>Exposure Analysis</b> from Project Manager to analyse brains, Surface regions, or Measurement volumes; <b>Analyze exposure…</b> in Exposure Editor is a shortcut for the selected exposure.</p>
""",
        ),
    ),
    "exposure_analysis": (
        "Maps / Analysis",
        "Exposure analysis",
        _page(
            "Exposure analysis",
            """
<p>Cumulative exposure analysis is intentionally detached from <b>Create playback</b>. Open <b>Exposure Analysis</b> from Project Manager and choose the saved exposure to analyze; <b>Analyze exposure…</b> in Exposure Editor is an equivalent shortcut for its selected exposure. The first page groups SRI24 brains and eligible authored Surface regions under <b>Region sets</b>, with ordinary Measurement-role volumes listed separately. An object with authored Surface regions remains a Region set regardless of its scene role; assigning the Generic bust the Measurement role does not replace Face/Scalp/etc. with a whole-object Exposure Analysis volume. The first page is <b>Exposure target selection</b>. Its reusable context header occupies roughly the upper third of the window: page headline/explanation plus the prominent target/exposure identity and target statistics stay on the left while a compact interactive 3D preview of the complete selected target fills the right third. The compact preview deliberately shows the object's ordinary unsegmented outer scene surface, with no region colouring or 3D grid; region boundaries are reserved for the dedicated Region Viewer. The preview uses display geometry only: it does not calculate fields or rebuild numerical samples. Its in-viewer Plotly toolbar exposes the native Orbit and Turntable rotation modes for visual confirmation before preparation. Every shared header keeps its navigation buttons, page title, <b>Hide analysis info</b>/<b>Show analysis info</b>, and <b>Open region viewer…</b> together on the top row. Hiding analysis info collapses the detailed information and target preview beneath that row, pauses the hidden preview renderer, and gives the current page the reclaimed vertical space. Choose <b>Open region viewer…</b> from the top row to open a modeless browser for that target: the object/region hierarchy appears in a tree on the left and a larger 3D view on the right. Selecting a tree row highlights that region while keeping the rest of the object visible. The Region Viewer uses the same <b>SCENE VIEW</b> command strip as the main 3D scene—Export image, Home, Fullscreen and Front/Right/Top—while Plotly's native in-viewer toolbar provides zoom/pan plus Orbit and Turntable rotation; Turntable is the initial rotation mode. One Region Viewer is reused for the Exposure Analysis workspace rather than opening duplicate windows. A second table beside the object list shows only saved analyses belonging to that selected exposure and target, with the same useful cache/status fields as Saved Results minus the redundant Exposure and Target columns. Browsing either table does not calculate fields.</p>
<p>Exposure Analysis always covers the complete saved physical exposure; there is no Preview-duration analysis scope. Select a saved row and choose <b>Open selected analysis</b> (or double-click it) to open that exact cached preparation, including stale historical results. Choose <b>Create new analysis</b> when you want a fresh preparation. If a current cached analysis already exists for the same exposure and target, Field Workbench asks for confirmation before creating another one. The workspace then resolves the saved source/routing clocks, computes one fixed-scene field plus one per-driven-coil field basis over the selected object’s samples, stores it as a new saved analysis, and preserves earlier saved analyses. It does not build animation frames or require a Scripted Maps result.</p>
<p>After creation, the selected object opens on <b>Exposure analysis home</b>. It reuses the same upper-third context-header template as target selection: the optional <b>← Analysis targets</b> navigation button, page headline and simple explanation appear above the same prominent target/exposure identity and compact preparation statistics, while the right third keeps one interactive whole-target 3D preview alive for the entire prepared-analysis workspace. The same preview therefore remains visible when you move from Home into Summary, Spectrum/FFT, Timing, Threshold, or frequency-weighted screening; page changes do not create another renderer or recalculate field samples. Its <b>Open region viewer…</b> button opens/reuses the same modeless Region Viewer and starts from the currently selected analysis region when one is available. The compact preview always remains an unsegmented whole-object context view; while a method page is open, changing the analysis region updates any already-open Region Viewer instead. Region Viewer tree clicks remain visual-only and do not change the analysis selection or start calculations. Both anatomy displays omit the 3D grid/axes for a cleaner object-focused view. The preview and Region Viewer use the <i>current live model geometry</i>; when a saved preparation is stale, Field Workbench labels that distinction because the live geometry may no longer exactly match the frozen numerical preparation. Instead of a bare launcher table, each method is introduced by the question it answers, a short description of the calculated quantities, and links to the literature or reference framework that motivates the method. A compact <b>Cache</b> line on each card shows the saved target-wide work for that method. Close the auxiliary window with its normal window control when finished; reopen <b>Exposure Analysis</b> from Project Manager (or use Exposure Editor's <b>Analyze exposure…</b> shortcut) when you want the object/saved-analysis chooser again.</p>
<p>Choosing a method replaces the home page inside the same window. Use <b>← Analysis home</b> at the top to return to the guided method overview; changing region on any analysis page is carried back to the target view and reused by the other pages. Returning home refreshes the cache indicators immediately, so newly completed calculations are visible without reopening the workspace. Analysis pages do not add duplicate Back or Close buttons; normal window controls own window lifetime.</p>
<h2>Characterize the exposure</h2>
<ul>
<li><b>Exposure summary</b> — a sortable all-region table of temporal mean regional RMS, exposure RMS, peak RMS/time, B-time and B²-time. Surface-region targets also show the prepared sampled volume for each row. The statistics table uses the full available width, while a hidden-by-default <b>Filters / highlights</b> panel can be revealed beneath it with <b>Show/hide filters</b>. Those rules use the same Filter/Highlight controls as the Brain Areas panel, including Left/Right rules for LPBA40 targets.</li>
<li><b>Spectrum</b> — frequency composition of the regional XYZ vector waveform and the full FFT viewer.</li>
<li><b>Timing &amp; transitions</b> — regional vector slew plus drive pulse counts, widths, active time and gaps.</li>
<li><b>Threshold &amp; duration</b> — time above a user-selected regional field threshold, fraction of exposure, excursion count and longest uninterrupted interval.</li>
</ul>
<h2>Compare against a reference framework</h2>
<p><b>Frequency-weighted screening</b> is visually separated from descriptive statistics because it
compares a complex waveform against a selected frequency-dependent ICNIRP/EU external-B curve. It is
not another cumulative-dose statistic.</p>
<h2>Object chooser and analysis workspace</h2>
<p>Exposure Analysis is a first-class Project Manager application and opens as a normal modeless Field Workbench top-level window, so it appears in the desktop taskbar/window group alongside Model Editor, Map Creator, Exposure Editor and the other suite applications. The Exposure Editor and Saved Results shortcuts open the same analysis workspace in context. Closing the project closes its Exposure Analysis windows.</p>
<p>The <b>Exposure target selection</b> page explains that a target is chosen first, then a base preparation is created and reusable statistical calculations are cached automatically. Target selection and prepared-analysis pages use one shared context-header template: roughly the upper third of the window is split into two-thirds information on the left and one-third whole-target 3D preview on the right. The remaining two-thirds of target selection is split roughly one-third available targets and two-thirds saved analyses for the selected exposure/target. These are draggable splitter proportions rather than fixed sizes. Selecting another target refreshes the preview without preparing exposure fields. Brain and Surface regions targets are grouped under <b>Region sets</b>; explicit Measurement-role objects are grouped separately. Opening a saved row or completing <b>Create new analysis</b> opens that one saved preparation in a dedicated target view. That prepared view reuses the same shared upper header, so the page headline/explanation can change while the target/exposure identity, preparation statistics and compact preview remain visually consistent on Home and the method pages. The shared top row on every one of these pages is ordered as the applicable Analysis targets/Home navigation, page title, <b>Hide analysis info</b>/<b>Show analysis info</b>, and <b>Open region viewer…</b>. Its collapsed state keeps that row intact while hiding the detailed identity/statistics area and the preview. The top-row <b>Open region viewer…</b> button opens a modeless hierarchy/3D browser. Analysis-region changes flow one way into an already-open Region Viewer for visual context, while the compact preview stays on the unsegmented whole object and tree selection inside the Region Viewer remains exploratory only and does not start or change an exposure calculation. Reopening the same unchanged target preserves the Region Viewer camera; if the live target was edited the display refreshes, and if it was deleted the saved numerical analysis remains readable while the 3D view reports that live geometry is unavailable. Use <b>← Analysis targets</b> to move directly back to the chooser and switch between a bust surface preparation and an SRI24 brain preparation without closing the Exposure Analysis workspace. The <b>Exposure analysis home</b> uses the full width of the workspace; regional navigation appears only after opening a method where it is meaningful. Brain targets use their LPBA40 hierarchy and Surface regions targets use their authored region hierarchy. Hovering a prepared near-surface region reports its sampling depth, sampled volume, source-voxel population, retained field-sample count and surface area; selecting it repeats the useful geometry summary beside the active-result path. A single-result Measurement volume does not waste space on a one-item tree. <b>Exposure summary</b> is the exception because it already shows every prepared region together: the sortable region/statistics table fills the full width of the lower workspace, while filters/highlights stay hidden by default and expand beneath the table when needed, and selecting a row synchronizes that region with the other tools. Each calculation configuration covers all prepared regions of the selected target together. Use <b>← Analysis home</b> to return from a method to the guided overview. Frequency-weighted results continue to open in a separate result window with a synchronized region tree.</p>
<p>Open a method card to view its saved result or calculate it for the first time. If the current configuration is already cached, it is displayed immediately. If it is not cached, Field Workbench submits that all-region calculation to the analysis queue. Settings that change the calculation, such as an FFT configuration or a threshold value, select a different cache key: an exact saved match is reused, otherwise that new configuration is queued once and saved. Region changes only browse the regional members of the current configuration. <b>Calculate all uncached analyses</b> on Analysis home submits one current/default configuration for Exposure Summary, Spectrum/FFT, Timing, Threshold, and one frequency-weighted reference; it does not expand every alternate setting. The Frequency-weighted page has its own <b>Calculate all uncached references</b> control when you deliberately want every reference curve for the selected sampling configuration.</p>
<p>Each Exposure Analysis window has a deliberately simple <b>single-worker queue</b>: one derived calculation runs at a time and additional uncached calculations wait in submission order. This avoids large analyses competing for CPU and memory. The stacked calculation area at the bottom shows the running job plus queued jobs. Running jobs have <b>Pause</b>/<b>Run</b> and <b>Cancel</b>; queued jobs can be cancelled before they start. Pause takes effect at the next numerical progress boundary, does not count paused time toward reported processing time, and holds the single worker slot until resumed or cancelled. When a running calculation completes, fails, or is cancelled, the next queued calculation starts automatically.</p>
<p>Each method card separates <b>cache state</b> from <b>calculation state</b>. The Cache line says <b>Not calculated</b>, <b>Cached</b>, or <b>Cached • N settings</b>; a settings count means more than one calculation configuration is stored for that target. A temporary line below it reports <b>Running…</b>, <b>Queued…</b>, pause/cancel state, a cancelled/failed calculation, or a save failure. Cancelling or failing a new configuration does not make an earlier cached result disappear. The cache belongs to the prepared object/configuration, while the setup-region tree only chooses which member of that all-region result set is displayed and is temporarily disabled while analysis work is running or queued. The Frequency-weighted auxiliary result window is read-only cached browsing, so its own region tree remains usable while another reference is being calculated. Completed cached results are kept, and cancelled calculations do not publish partial results. Closing a busy workspace requests cancellation for the running job, removes queued jobs, and closes the auxiliary window immediately; retained workers finish cancellation safely in the background, so Exposure Analysis cannot block project or application shutdown. Preparation also has its own <b>Cancel preparation</b> button. Progress describes the current stage rather than predicting the time remaining; elapsed and original processing times appear at the bottom of the analysis page. Saved results are labelled <b>Loaded saved result</b> and show their original processing time when available.</p>
<p><b>Calculation coverage:</b> Each analysis run calculates all prepared
regions of the object together. Select another region in the tree to view its saved results immediately.
Threshold values and reference profiles each have their own saved settings.
Each analysis configuration is saved as one complete set within the prepared
exposure and target. Changing FFT sampling, pulse thresholds, field thresholds or reference profiles uses a separate configuration;
a failed run leaves previously saved results available.
Summary shares the current replay across regional field calculations. Spectrum shares current FFTs.
Timing shares drive-current peaks, pulse intervals and derivatives; regional slew is calculated separately
from each region's field moments. These are regional spatial-RMS results, not pointwise maxima.</p>
<p><b>Measurement volumes:</b> Boxes, spheres, cylinders and supported closed meshes assigned the
Measurement role use the sampling settings saved on the object when they do not already provide an
authored Surface-region analysis set. Generated samples represent equal volumes. Custom XYZ points
receive equal weights and describe those points; they do not establish a volume average for an
irregular point distribution. These views are labelled <b>RMS across custom sample points</b>.
Authored Surface regions are intrinsic to their object for Exposure Analysis and therefore remain a
Region set even if that object is assigned the Measurement scene role. Overlapping objects remain
independent; their results are never added or averaged together. Brain targets continue to use the
volume-weighted LPBA40 atlas model.</p>
<p>Changing an object’s geometry, placement or active sampling settings invalidates its preparation.
Names and display colours do not. Saved Results retains a separate entry for each prepared object; opening one there shows that record, including historical results if the model has since changed.</p>
<h2>Saving and reopening</h2>
<p>Completed results update their project analysis record automatically. Save the project to retain them
on disk. Different settings calculated in separate windows are kept together; if two windows calculate
the same uncached settings concurrently, the last completed save replaces that configuration. A failed calculation does not replace
earlier results. If a calculated result cannot be written back to the project cache, its method card keeps
showing the in-memory cache and adds <b>Save failed</b> plus <b>Retry save</b>. The workspace also shows a
short warning explaining that the result currently exists only in that window. Keep the window open until
saving succeeds if you need that result.</p>
<p>Returning to an object rechecks its physical inputs and saved record. Changed or deleted preparations
are not presented as current work. Historical results remain available through Saved Results; they are
labelled as belonging to earlier project settings.</p>
<h2>Saved Exposure Analysis results</h2>
<p>Cached results appear under <b>Saved Results → Exposure analyses</b>. Each stores the exposure, target, sampling context, compressed authoritative preparation data, derived-result caches, and a dependency fingerprint. Preparation and derived work are stored separately: adding a Summary/FFT/Timing/Threshold/frequency-weighted result updates only the derived cache instead of rewriting the large frozen field basis and replay source. Numeric result arrays are stored as compressed binary rather than decimal JSON, and identical arrays shared by multiple cached results are deduplicated within the derived cache. Automatic cache writes do not consume Undo history; explicit Rename, Manage cache removals, and Delete cache actions remain undoable. <b>Current</b> means the physical inputs still match the calculation. <b>Stale</b> means a numerical dependency changed; the old result remains readable as a frozen historical result and is never silently reused for a changed physical configuration. The fingerprint tracks the resolved field-source scene, active background fields, driven-coil field/electrical properties, source/load timing, waveform samples and electrical source metadata, coil solver representation, and the target geometry, resolved placement and active sampling settings. Presentation-only edits such as names, colours, notes, display units, Sensor changes, unrelated visual meshes, and bobbin/flange decoration do not invalidate an otherwise identical analysis. <b>Manage cache…</b> lists the individual stored configurations so selected Summary, Spectrum/FFT, Timing, Threshold, or frequency-weighted results can be removed without discarding preparation data. Because configurations can share deduplicated arrays, individual rows do not claim an independent byte size; after a removal the manager reports the actual compressed derived-cache storage reclaimed using the before/after sizes. <b>Delete cache</b> discards the preparation and every derived result together.</p>
<div class="note">No single metric completely describes an ELF exposure. Field Workbench therefore
keeps amplitude, duration, timing, frequency, spatial distribution and reference-framework screening as
complementary questions rather than collapsing them into one score.</div>
""",
        ),
    ),
    "spectrum_analysis": (
        "Exposures",
        "Spectrum and FFT",
        _page(
            "Spectrum and FFT",
            """
<p>The Spectrum/FFT tool is part of <b>Exposure Analysis</b> (also reachable through the <b>Analyze exposure…</b> shortcut in Exposure Editor). It operates on the authoritative regional exposure payload and is not launched from a Scripted Maps result.</p>
<h2>Vector spectrum</h2>
<p>The spectrum combines XYZ vector-component power with the region's spatial weights. It does
<b>not</b> take an FFT of scalar <code>|B|</code>, which would lose a constant-magnitude rotating vector,
and it does not vector-average opposing fields before spectral power is formed.</p>
<p>The summary table removes the mean by default, applies a rectangular window, and reports the strongest
nonzero frequency bins as <b>spatial RMS contribution</b>. Short records use the complete physical record.
For a long exposure with a verified repeated steady state, FFT uses one verified physical cycle instead
of spreading a fixed sample count across the full exposure. The cycle duration therefore sets Δf and its
source-supported sample clock sets Nyquist. Initial settling transients and a final partial cycle are not
mixed into that steady-state-cycle spectrum; the information block states this explicitly.</p>
<p>For another long/non-periodic exposure, Field Workbench divides the complete physical duration into
equal non-overlapping FFT records and averages their vector spectral power. Each record is bounded at
65,536 samples and total spectral work is bounded. If that work budget requires coarser sampling, the
reported Nyquist decreases; Field Workbench never interpolates a sparse display trace to claim source
bandwidth that was not analysed. The information block reports the selected method, record count and
duration, bin spacing, analysis Nyquist, and source Nyquist where relevant.</p>
<h2>FFT spectrum viewer</h2>
<p><b>View FFT…</b> switches the persistent Exposure analysis workspace to the complete one-sided
spectrum as a conventional bar graph. Use <b>← Analysis home</b> to return to the guided analysis overview. The viewer keeps
only a compact result summary above the graph; detailed method explanation remains in Help rather than
occupying plot space.</p>
<ul>
<li><b>Auto</b> uses the standard Exposure Analysis spectrum settings.</li>
<li><b>Custom</b> can reduce the FFT sample/bin count, choose Rectangular, Hann, Hamming, or Blackman windowing, and retain/remove the mean field.</li>
<li><b>Plot max frequency</b> changes only the displayed X range and updates immediately.</li>
<li>DC can be displayed explicitly and the Y axis can be switched to logarithmic display.</li>
<li>Calculation settings are cache keys. Changing Auto/Custom FFT settings loads an exact saved match when available; otherwise Field Workbench calculates that all-region configuration automatically. Display-only controls such as Plot max frequency, Show DC, and Log Y never recalculate the FFT.</li>
<li><b>Fullscreen viewer</b> opens a separate true-fullscreen interactive copy of the FFT plot. The current frequency range and manual zoom are carried into fullscreen; press Esc or F11 to return.</li>
<li>Changing a display-only control also resets the affected plot view immediately, so a previous manual pan/zoom cannot hide the new range.</li>
</ul>
<p>For a whole-record or verified-cycle FFT, that record duration determines frequency-bin spacing. For
a segmented long record, each segment duration determines bin spacing and the displayed bin power is the
mean vector spectral power across all segments. Reducing FFT samples changes the record construction and
can lower represented Nyquist; it never creates source bandwidth. Non-periodic records can leak between
bins, and windowing trades leakage against spectral broadening and amplitude characteristics.</p>
""",
        ),
    ),
    "timing_analysis": (
        "Exposures",
        "Timing and transitions",
        _page(
            "Timing and transitions",
            """
<p>The Timing &amp; transitions tool is part of <b>Exposure Analysis</b> (also reachable through the <b>Analyze exposure…</b> shortcut in Exposure Editor). It characterizes how rapidly a selected regional field changes and how the drive-current pulses are arranged.</p>
<h2>Vector slew</h2>
<p>The analysis reports the maximum interval <b>spatial RMS |dB/dt|</b> over the selected region. The
derivative is formed from vector B changes before taking magnitude, so a rotating field is not mistaken
for zero slew merely because |B| is constant. This is a regional RMS descriptor, not the worst
individual sample point.</p>
<h2>Drive pulse and gap timing</h2>
<p>Pulse/gap timing is derived from threshold crossings of each drive's absolute current. The adjustable
threshold is expressed as a percentage of that drive's peak current and crossing times are linearly
interpolated between authoritative source current samples. The output includes pulse count,
active/inactive time, pulse-width range and between-pulse gaps. Verified repeated steady-state cycles
are summarized with their physical repeat count rather than expanded into millions of samples.</p>
<div class="note">The drive threshold identifies current pulses only. It is not a tissue-field
threshold and the reported widths are drive timing, not biological pulse duration.</div>
""",
        ),
    ),
    "threshold_duration": (
        "Exposures",
        "Threshold and duration",
        _page(
            "Threshold and duration",
            """
<p>The Threshold &amp; duration tool is part of <b>Exposure Analysis</b> (also reachable through the <b>Analyze exposure…</b> shortcut in Exposure Editor). It asks how long the selected region's <b>spatial-RMS |B|</b> remained above a field level chosen by the user.</p>
<p>The chosen threshold is calculated for every prepared region as one cached configuration. Other saved thresholds
remain available: changing the threshold control loads an exact match or automatically calculates the new value.
Region changes only browse the saved results. The exposure current history is replayed once per new configuration,
and each region's field and threshold crossings are evaluated in that pass. Interval previews remain
bounded even for very long repeated exposures; a complete per-point field history is not stored.</p>
<p>The analysis reports total time above threshold, fraction/percentage of physical exposure, number of
separate excursions, longest uninterrupted interval, and the regional spatial-RMS peak/time.</p>
<h2>Crossing calculation</h2>
<p>Threshold crossings are linearly interpolated between the authoritative physical analysis samples. The
result therefore respects the waveform/routing analysis timebase but cannot recover excursions faster
than that timebase resolved. Verified repeated cycles contribute their complete physical repeat count
without being expanded. If an exposure contains more than 2,000 excursions, the table shows only the
first 2,000 while totals, percentage, excursion count and longest interval still include the complete
exposure.</p>
<h2>Interpretation</h2>
<p>Threshold-duration functions and temporal-variability measures have been used as exposure
characterization approaches in the literature. Field Workbench deliberately treats the selected
threshold as a <b>descriptive experiment-comparison setting</b>, not an established biological-effect
boundary or a regulatory pass/fail criterion.</p>
<p>Research precedent includes Zhang, Nair &amp; Sahl (1997) for threshold-plus-duration/effects-function
comparisons and Kaune et al. (2001) for temporal-variability descriptors.</p>
""",
        ),
    ),
    "frequency_weighted": (
        "Exposures",
        "Frequency-weighted screening",
        _page(
            "Frequency-weighted screening",
            """
<p>The Frequency-weighted screening tool is part of <b>Exposure Analysis</b> (also reachable through the <b>Analyze exposure…</b> shortcut in Exposure Editor). It applies published frequency-dependent external magnetic-flux-density (B) reference/action-level curves to a selected region without treating ordinary playback as an authoritative exposure analysis.</p>
<h2>Choose, queue and view</h2>
<p>Choose <b>Auto</b> or <b>Custom</b> sampling. Each reference row then shows the action appropriate to that exact profile/sampling cache key: <b>Calculate</b> when it has not been run, <b>Queue</b> while another frequency-weighting reference is already active, and <b>Open</b> when the saved all-region result is ready. <b>Calculate all uncached references</b> queues every missing reference for the current sampling configuration. Duplicate clicks do not create duplicate jobs.</p>
<p>Only one derived Exposure Analysis calculation runs at a time. Frequency-weighting references therefore run serially through the same target-wide queue as Spectrum/FFT, Timing, Threshold, and Summary work. The Auto/Custom sampling controls are locked while this page has a running or pending reference so every queued item keeps the sampling configuration it was submitted with. Cancelling the frequency-weighting calculation also clears that page's still-pending reference requests.</p>
<p>Cached results remain fully browsable while a different reference is running or queued. Choose <b>Open</b> beside any Ready reference; opening it does not disturb the running job or its progress state. The separate results window contains the statistics, strongest-frequency table, graph, and its own synchronized region tree, which remains usable for cached browsing during calculation. Selecting another region displays its saved result immediately without another calculation.</p>
<p>The reported processing time is for the whole all-region batch, not one region. Changing the sampling configuration selects another cache key; exact saved matches are reused. A queued reference can also reuse the matching rectangular/detrended spectrum produced by an earlier reference, avoiding redundant FFT preparation.</p>
<h2>Shared calculations</h2>
<p>The regions share the same coil-current history. Each spectral record is transformed once, and each
frequency-weighted current block is calculated once per profile. Field Workbench then evaluates the
stored field moments for every region from that shared data. Parent-region results are calculated from
their own field moments; they are not averages of child-region peaks. Long exposures still use bounded
streaming blocks.</p>
<p>Saved frequency-weighted results use a <b>compact derived cache</b>. The authoritative scalar peak and
spectral-sum results remain at full calculation precision, while the saved result keeps one bounded
normalized plot trace plus only the <b>30 strongest spectral rows</b> needed by the result table. The
plot trace is stored at display precision; this does not change the full-resolution peak search or the
reported scalar results. Full frequency/bin arrays are not duplicated into every reference/region result.</p>
<p>A matching saved Spectrum/FFT result is reused for the spectral cross-check: the same prepared
exposure, target and sample count, with a rectangular window and detrending enabled.
A weighting run also saves this spectrum if needed, so it is available to other reference profiles and
the Spectrum/FFT view. Profile-specific filtering and weighted-peak searches are still performed for
each run. A different sampling count or FFT window never silently substitutes for the requested one.</p>
<h2>Why this is separate from cumulative statistics</h2>
<p>B-time, B²-time, time RMS, and absolute peak describe amplitude and duration. They do not by
themselves preserve the frequency dependence that matters for induction-related safety screening.
The weighted analysis therefore remains a separate layer rather than turning an exposure integral
into a presumed biological dose.</p>
<h2>Reference profiles</h2>
<p>The initial implementation includes the <b>ICNIRP 2010</b> external-B reference-level curves for
general-public and occupational exposure, plus the <b>EU Directive 2013/35/EU</b> low, high, and
localized-limb magnetic action-level curves. The exact edition/profile is shown in every result.
Field Workbench evaluates the 1 Hz–100 kHz LF portion and reports the actually resolved Nyquist band.</p>
<h2>Phase-aware weighted-peak</h2>
<p>The selected limit curve is converted into a frequency response relative to 50 Hz. In the exact
piecewise method, limit regions proportional to <code>1/f²</code>, <code>1/f</code>, constant, and
<code>f</code> use weighting phases of +180°, +90°, 0°, and −90°, respectively. The frozen drive
waveform is transformed with that gain and phase, reconstructed in time, and the maximum regional
spatial-RMS magnitude is divided by the selected 50 Hz peak reference. A result of 1.0 (100%) means
that this regional weighted descriptor equals the selected reference curve.</p>
<h2>Spectral summation</h2>
<p>The same window also reports the conservative linear sum of each nonzero FFT bin's spatial-RMS
contribution divided by the selected RMS reference at that frequency. This preserves a useful
cross-check against the phase-aware result and shows the strongest normalized frequency bins in a
table.</p>
<h2>Spatial meaning and limitations</h2>
<p>The compact frozen exposure-analysis payload retains vector second moments for every region. That allows
frequency weighting to preserve XYZ vector power and opposing fields while remaining fast, but the
reported result is a <b>regional spatial-RMS weighted index</b>. It is not the maximum field at every
point in the body or object.</p>
<div class="warning"><b>This is a screening descriptor, not compliance certification.</b> ICNIRP
reference levels assume particular exposure conditions; localized/non-uniform exposure can require
maximum-field assessment or dosimetric calculation of the induced electric field. An index below 100%
does not by itself establish compliance, and an index above 100% does not by itself prove that an
internal basic restriction is exceeded.</div>
<h2>Long exposures and sampling</h2>
<p><b>Auto</b> uses the same source-honest spectral planning as the FFT tool. A complete short record keeps
the historical whole-record method. A verified periodic long exposure analyses one already-verified
steady-state physical cycle without expanding its repeats; as with the verified-cycle FFT, initial settling
transients and a final partial cycle are intentionally not mixed into that steady-state spectral descriptor.
Other long/non-periodic exposures span the complete physical duration with bounded equal records; the phase-aware weighted peak uses the centre
record of a three-record FFT context block so artificial circular boundaries remain one record away from
the scored samples. The spectral-summation cross-check uses the full-duration power-averaged spectrum.</p>
<p>Sampling never exceeds the frozen authoritative source clock. If the bounded spectral-work budget lowers
the analysis sample rate, Field Workbench reports the lower Nyquist rather than inventing bandwidth. It
also reports the first resolvable nonzero frequency, because a high Nyquist alone does not mean the entire
1 Hz–100 kHz band is represented. Custom samples are samples per record. Non-periodic block processing is
a bounded approximation to one enormous whole-duration phase-aware transform; it is identified as such in
the result.</p>
<p>The weighted-index plot is presentation data. For very long segmented analyses it is evenly thinned to at
most the normal retained-trace budget, while the peak search still evaluates every analysed weighted
sample.</p>
<h2>Sources</h2>
<ul>
<li>ICNIRP, <i>Guidelines for limiting exposure to time-varying electric and magnetic fields (1 Hz to 100 kHz)</i>, Health Physics 99(6):818–836, 2010; Tables 3–4, non-sinusoidal section, and informative appendix.</li>
<li>European Parliament and Council, <i>Directive 2013/35/EU</i>, Annex II, especially Table B2 and Note B2-2.</li>
</ul>
""",
        ),
    ),
    "video_export": (
        "Playback",
        "Playback video export",
        _page(
            "Playback video export",
            """
<p><b>Export video</b> appears beside <b>Fullscreen viewer</b> on rendered animated 2D, 3D, and Brain
View result tabs. Export reuses that tab's already-calculated field payload, playback timeline, selected
view, and camera state; it does not repeat the magnetic-field solve.</p>
<p>Later changes in the Selector tab or source scene do not change a frozen result export. Scripted
camera paths use the timeline stored with that rendered animation. The Brain Areas panel is included when it is visible and omitted when hidden; its current-frame filters,
selection, and table state are preserved in the export.</p>
<p>H.264 MP4 export uses the bundled FFmpeg encoder on packaged Windows builds. Linux uses the selected
or system FFmpeg. A numbered PNG frame sequence is available without an encoder and is useful for
external video editors.</p>
<p>Video playback speed is presentation only. Exporting a result in slow motion does not change the
physical exposure duration or any separately cached Exposure Analysis statistic.</p>
""",
        ),
    ),
    "real_measurements": (
        "Measurements and calibration",
        "Measurement datasets and calibration",
        _page(
            "Measurement datasets and coil calibration",
            """
<p><b>Dataset Manager</b> is a project-aware suite application opened from Project Manager. Model Editor does not need to remain open.</p>
<p>The manager now separates measurement work into three layers: <b>Measurement datasets</b> are raw saved
observations; <b>Calibration / Composite fits</b> are derived recipes that combine selected portions of those
datasets; and <b>Coil correction links</b> apply a fitted calibration factor to one or more scene coils.</p>
<h2>1 · Measurement datasets</h2>
<p>The upper table lists every raw dataset in the project, including its coordinate sensor, measured quantity,
sample count, and last update time. Use <b>Add dataset…</b> to enter/import measurements, <b>Edit…</b> to reopen
them, and <b>Delete</b> to remove data that is not referenced by a calibration/composite fit. Raw-dataset entry does
not perform a fit; fitting begins only in step 2.</p>
<p>Raw datasets keep the measured values, copied sample coordinates/orientations, quantity, and provenance.
Composite calibration never merges or rewrites the source data. If a raw dataset is edited after a composite fit
was calculated, that relation is shown as <b>Needs recalculation</b>.</p>
<h2>2 · Calibration / Composite fits</h2>
<p>A calibration relation can select whole datasets or individual measured samples from several datasets. It also
selects the coil group used to model those measurements and solves one shared scale-only correction factor for all retained observations.</p>
<p>Two weighting rules are available. <b>Each measurement point equal</b> treats every selected observation as one
least-squares sample. <b>Each source selection equal</b> normalizes every dataset portion to equal total weight, so
a dense scan does not automatically dominate a smaller but equally trusted measurement set.</p>
<div class="formula">k = arg min Σ wᵢ (B measured,ᵢ − k · B model,ᵢ)²</div>
<p>The fitted relation stores its source selections, fit coils, weighting rule, source update versions, model
signature, factor, sample/source counts, RMS residuals, maximum residual, and R². Editing its source datasets or
relevant model state marks it for recalculation instead of silently replacing the saved result.</p>
<p>A multi-coil fit basis is intentionally modeled as one magnetic source group. If Coils 2 and 3 were measured only as
a pair, one calibration can fit that pair as a single source group. The resulting factor is reusable: the coil-link table
may apply it to any other scene coil when the user knows that coil has an equivalent correction relationship. Separate
coil factors still require measurement evidence that distinguishes the coils.</p>
<h2>3 · Coil correction links</h2>
<p>The lower table lets each coil use <b>Custom</b> correction or a saved <b>Calibration</b>. Several coils may link to
the same calibration and therefore share its resolved factor. Recalculating a linked calibration updates all coils
linked to it. Fit provenance and application are intentionally separate: any <b>Ready</b> relation may be linked to any
scene coil, while the relation's fit-coil group continues to record which modeled source produced the fitted factor.
Use that reuse only when the unmeasured coil is physically equivalent enough for the same factor to be justified. A
stale <b>Needs recalculation</b> relation must be recalculated before it can be newly linked. Raw datasets are never linked directly to coils.</p>
<h2>Recreate saved sensor points</h2>
<p>Select one raw dataset and click <b>Create point sensors</b>. A new group appears in the scene with one Point sensor for each saved sample row, preserving its world coordinates and local measurement orientation, including rows without a reading. Existing sensors and the raw dataset are unchanged. The entire addition can be undone in one step. Repeating the action creates another group with unique names.</p>
<p><b>(saved coordinates)</b> means the original coordinate sensor is no longer in the scene. The dataset still retains its copied positions and orientations.</p>
<h2>Create or edit a raw dataset</h2>
<ol>
<li>Add a Point sensor or linear axis sensor at the physical locations where measurements are taken.</li>
<li>Open <b>Dataset Manager → Add dataset…</b>, choose that sensor, and choose <b>|B|</b> or one local B component.</li>
<li>Enter readings in µT directly, paste spreadsheet cells, import CSV/text data, or edit X/Y/Z and optional Path coordinates.</li>
<li>Press <b>Save dataset</b>. The raw dataset contains the measured values and their copied coordinate/orientation provenance only.</li>
<li>Back in Dataset Manager, create a <b>Calibration / Composite fit</b>, choose the source dataset portions and fit-coil group, then calculate the correction and link the result to coils.</li>
</ol>
<p>Dataset rows retain their copied world XYZ coordinates and local measurement orientation even if the original
sensor later moves or is deleted.</p>
<p>Measurement comparisons use the <b>actual evaluated points nearest the measurement</b> locations rather than merely
comparing against a displayed map pixel or interpolated screenshot value.</p>
<h2>Calibration / Composite-fit model</h2>
<div class="formula">B measured ≈ k × B model</div>
<p>Workbench evaluates the selected fit coil group at exactly <code>1.000×</code> empirical correction and isolates
that group by subtracting an otherwise identical model evaluation with the group disabled. Unselected coils,
permanent magnets, and saved Background fields therefore cancel on the model side. Re-fitting does not compound
an older empirical correction.</p>
<div class="warning"><b>Amplitude convention:</b> entered readings must use the same RMS, peak-sine, or DC
convention as the coil drive values represented by the scene. A scalar fit corrects amplitude only; strong
position-dependent residuals can indicate geometry, alignment, polarity, current, or field-shape mismatch.</div>
<h2>Scrolling in the editors</h2>
<p>Mouse-wheel gestures scroll the dataset/exposure forms and tables instead of changing numeric controls or
selectors, and they do not move keyboard focus onto those controls. Use clicks, Tab, or keyboard input to select and change values.</p>
""",
        ),
    ),
    "measurement_sampling": (
        "Statistics and exposure metrics",
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
<p>For a healthy closed mesh, geometric volume and centroid come from the oriented mesh surface,
not from counting retained samples. The sampling grid approximates the field distribution inside
that geometry; changing sampling quality does not redefine the mesh's physical volume.</p>
<div class="note">Sampling quality measures the discrete model of the volume, not solver accuracy by
itself. A finer grid can reveal spatial variation that a coarse grid misses, but it also increases
calculation time.</div>
<p><strong>User defined points</strong> accepts one or more <code>[X, Y, Z]</code> coordinates in local
millimetres, for example <code>[0, 0, 0], [9, 0, 0], [18, 0, 0]</code>. These points replace generated
voxels and all receive equal statistical weight. They are stored with the object, so movement and
rotation preserve the intended measurement pattern. Defined points are used exactly as entered and
are not clipped automatically to the visual object's boundary.</p>
<p>Select one or more objects already assigned the Measurement volume role, then choose
<strong>Volume analysis</strong>. With several selected volumes, Workbench evaluates them in scene
selection order and opens one independent result window for each successful analysis. Closing the
active progress window cancels the current calculation and leaves the remaining selected volumes
unanalyzed.</p>
<p>After sampling completes, each measurement result opens in its own non-modal window. Several
fresh analyses and stored results can remain visible together while the main Workbench is
used for other tasks. The <b>Cross sections</b> tab shows the central XY, XZ, and YZ sample layers
together on one shared |B| colour scale. These are the actual evaluated points nearest the measurement
volume's local centre plane on each axis; Workbench does not interpolate a new field grid for this view.</p>
""",
        ),
    ),
    "measurement_stats": (
        "Statistics and exposure metrics",
        "General spatial statistics",
        _page(
            "General spatial statistics",
            """
<p>Field Workbench uses spatial statistics to summarize a finite set of magnetic-field samples at one
instant. Measurement-volume results, playback Current-frame rows, Brain Areas, saved results, comparison,
and optimizer targets reuse closely related definitions. Statistics describe the sampled model; they
are not confidence intervals and do not prove continuous-field bounds between samples.</p>
<h2>Weighting</h2>
<p>Ordinary generated Measurement-volume voxel centres have equal volume weight. User-defined points
also receive equal numerical weight, but should not be interpreted as volume-representative unless the
sampling pattern was designed that way. LPBA40 Brain View samples instead carry explicit represented
voxel-volume weights so per-label downsampling does not bias regional results.</p>
<p>For weights <code>wᵢ</code> and field vectors <code>Bᵢ</code>, let
<code>bᵢ = |Bᵢ|</code>. With equal-weight measurement points, the weighted formulas reduce to ordinary
arithmetic statistics.</p>
<h2>Magnitude statistics</h2>
<table>
<tr><th>Statistic</th><th>Calculation and interpretation</th></tr>
<tr><td><b>Mean |B|</b></td><td><code>Σ(wᵢ bᵢ) / Σwᵢ</code>. Arithmetic mean for equal-weight points.</td></tr>
<tr><td><b>Spatial RMS |B|</b></td><td><code>sqrt[Σ(wᵢ bᵢ²) / Σwᵢ]</code>. This is RMS across space at one instant, not temporal RMS at one point.</td></tr>
<tr><td><b>Median / percentiles</b></td><td>Quantiles of the sampled magnitude distribution. LPBA40 uses represented-volume weighted nearest-rank percentiles; ordinary measurement analysis uses its equal-weight sample distribution.</td></tr>
<tr><td><b>Minimum / maximum (Peak)</b></td><td>Lowest and highest finite sampled magnitudes.</td></tr>
<tr><td><b>Standard deviation</b></td><td>Population spread, <code>sqrt(mean((x − mean)²))</code> for equal-weight measurement samples.</td></tr>
<tr><td><b>Coefficient of variation</b></td><td><code>100 × standard deviation ÷ |mean|</code>. Lower values indicate less relative spread and the value becomes unstable as mean approaches zero.</td></tr>
<tr><td><b>P95–P5 spread</b></td><td><code>100 × (P95 − P5) ÷ |mean|</code>. A robust relative spread that ignores the most extreme 5% at each tail; lower is more uniform.</td></tr>
<tr><td><b>Brain/region Uniformity</b></td><td><code>100 × [1 − (P95 − P5) / mean]</code>, clamped to 0–100%. Higher is more uniform. This is the complement-style display used by the playback region panel and should not be confused with P95–P5 spread.</td></tr>
</table>
<h2>Direction statistics</h2>
<ul>
<li><b>Mean vector:</b> component-wise weighted or equal-weight average of valid B vectors.</li>
<li><b>Mean direction:</b> normalized mean vector when the resultant is nonzero.</li>
<li><b>Directional consistency:</b> <code>100 × |Σ(wᵢ B̂ᵢ) / Σwᵢ|</code>. Near 100% means local field vectors point together; near 0% means directions are widely dispersed or cancel.</li>
<li><b>Median and P95 angular deviation:</b> angles between non-negligible unit vectors and the resultant reference direction where that result view provides them.</li>
</ul>
<h2>Target statistics</h2>
<table>
<tr><td><b>RMS target error</b></td><td><code>sqrt(mean((|B| − target)²))</code>, in µT. Larger magnitude errors receive greater weight.</td></tr>
<tr><td><b>Within target band</b></td><td>Percentage satisfying <code>target × (1 − tolerance/100) ≤ |B| ≤ target × (1 + tolerance/100)</code>.</td></tr>
<tr><td><b>Below / above target</b></td><td>Percentages below the lower band limit or above the upper band limit.</td></tr>
</table>
<h2>Finite values and sampling</h2>
<p>Sample count reports requested/retained points; valid count excludes NaN or infinite field results.
A finer spatial grid can reveal variation missed by a coarse grid, but adding more calculated points is
not the same thing as adding independent experimental replicates.</p>
<div class="note"><b>Spatial RMS and temporal RMS are different.</b> Spatial RMS combines different
locations at one time. Temporal RMS combines different times at one location (or is built from an
explicitly defined regional time series). The <b>Exposure statistics and research basis</b> topic documents the temporal quantities used by authoritative Exposure Analysis.</div>
""",
        ),
    ),
    "playback_statistics": (
        "Statistics and exposure metrics",
        "Exposure statistics and research basis",
        _page(
            "Exposure statistics and research basis",
            """
<p>Exposure Analysis statistics summarize one finite physical ELF/LF magnetic-field exposure. They are intended
as a <b>set of complementary exposure descriptors</b>: amplitude, duration, peaks, time variation,
frequency content, direction, and spatial distribution cannot generally be preserved by one scalar.
The definitions below separate Field Workbench calculations from what the research literature actually
establishes.</p>
<div class="warning"><b>These are descriptive magnetic-field metrics, not a universal biological dose.</b>
The reviewed literature uses several cumulative, RMS, peak, threshold, variability, spectral, and
mechanism-specific quantities; it does not establish B-time, B²-time, or another single scalar as a
transferable dose for arbitrary ELF biological effects.</div>
<h2>Start with the local time-domain definitions</h2>
<p>At one fixed position, let <code>B(t)</code> be the magnetic-field vector,
<code>b(t) = |B(t)|</code>, and <code>T</code> the physical exposure duration:</p>
<table>
<tr><th>Quantity</th><th>Definition</th><th>Units</th></tr>
<tr><td>Magnitude integral D1</td><td><code>∫ b(t) dt</code></td><td>µT·s</td></tr>
<tr><td>Squared-magnitude integral D2</td><td><code>∫ b(t)² dt</code></td><td>µT²·s</td></tr>
<tr><td>Temporal mean magnitude</td><td><code>D1 / T</code></td><td>µT</td></tr>
<tr><td>Temporal RMS magnitude</td><td><code>sqrt(D2 / T)</code></td><td>µT</td></tr>
<tr><td>Peak magnitude</td><td><code>max b(t)</code></td><td>µT</td></tr>
<tr><td>Peak vector slew</td><td><code>max |dB(t)/dt|</code></td><td>µT/s</td></tr>
</table>
<p>Scaling a fixed waveform by amplitude <code>a</code> scales D1 by <code>|a|</code> and D2 by
<code>a²</code>. Repeating the same physical script doubles D1 and D2 while leaving its mean and RMS
unchanged. D2 therefore emphasizes high-amplitude intervals more strongly, but that mathematical
property is not evidence for a biological weighting.</p>
<h2>Field Workbench regional Exposure Analysis definitions</h2>
<p>For a region at time <code>t</code>, Workbench first forms the volume-weighted spatial RMS:</p>
<div class="formula">Rspace(t) = sqrt[Σ(wᵢ |Bᵢ(t)|²) / Σwᵢ]</div>
<p>The Exposure summary then defines:</p>
<table>
<tr><th>Exposure quantity</th><th>Definition and interpretation</th></tr>
<tr><td><b>Exposure RMS |B|</b></td><td><code>sqrt(B²-time / T)</code>. Space-time RMS formed from the regional spatial-RMS exposure trace.</td></tr>
<tr><td><b>Mean |B|</b></td><td>Physical-time-weighted mean regional field magnitude over the analysed exposure.</td></tr>
<tr><td><b>Peak |B|</b></td><td>Largest analysed regional spatial-RMS magnitude, reported with its physical exposure time.</td></tr>
<tr><td><b>B-time</b></td><td><code>∫ Rspace(t) dt</code>. Descriptive regional field-magnitude × duration integral.</td></tr>
<tr><td><b>B²-time</b></td><td><code>∫ Rspace(t)² dt</code>. Descriptive squared-field × duration integral.</td></tr>
</table>
<div class="formula">B-time = ∫ Rspace(t) dt &nbsp;&nbsp; [µT·s]</div>
<div class="formula">B²-time = ∫ Rspace(t)² dt &nbsp;&nbsp; [µT²·s]</div>
<div class="formula">Exposure RMS = sqrt(B²-time / exposure time) &nbsp;&nbsp; [µT]</div>
<h2>Why the regional and local integrals need different names</h2>
<p>Integrating regional spatial RMS is not generally the same calculation as integrating
<code>|B(t)|</code> independently at every point and then volume-averaging those D1 values. In other
words, <b>B-time is a regional spatial-RMS integral, not a mean local D1</b>.</p>
<p>The squared case has a useful identity because squaring removes the outer square root:</p>
<div class="formula">∫ Rspace(t)² dt = Σ(wᵢ D2ᵢ) / Σwᵢ</div>
<p>This relationship does not make B²-time absorbed energy or biological dose; it only states what the
Workbench regional calculation represents mathematically.</p>
<h2>Research precedent for complementary descriptors</h2>
<table>
<tr><th>Metric family</th><th>Literature context</th><th>Workbench interpretation</th></tr>
<tr><td><b>Cumulative field × time</b></td><td>Röösli et al. (2008) used cumulative µT-years in a Swiss railway-worker cohort. AGNIR also describes cumulative and time-weighted exposure summaries.</td><td>Supports reporting magnitude-duration descriptors such as B-time; it does not establish a universal dose-response function.</td></tr>
<tr><td><b>RMS / TWA / peaks</b></td><td>AGNIR's exposure-characterization review discusses RMS/TWA, peak exposure, harmonic content, direction, transients, and time above threshold.</td><td>Supports keeping RMS and peak alongside cumulative quantities rather than collapsing exposure to one number.</td></tr>
<tr><td><b>Threshold and variability functions</b></td><td>Zhang, Nair &amp; Sahl (1997) compared average field with threshold-plus-duration functions; Kaune et al. (2001) examined successive-difference variability metrics.</td><td>User thresholds and variability measures are descriptive or hypothesis-specific. A chosen threshold is not automatically a biological boundary.</td></tr>
<tr><td><b>Squared intensity</b></td><td>Bowman, McDevitt &amp; Breysse (2001) included full-band squared intensity among exploratory candidate exposure metrics.</td><td>There is precedent for squared-field descriptors, but not equivalence between that study's metric and arbitrary Workbench B²-time.</td></tr>
<tr><td><b>Frequency / polarization / geometry</b></td><td>AGNIR and Misakian emphasize harmonic content, direction/polarization, and sample geometry as relevant exposure descriptors.</td><td>Preserve vector waveform and geometry; cumulative magnitude statistics alone cannot describe induction-related differences.</td></tr>
</table>
<h2>Why vector slew and spectrum are retained</h2>
<p>Two sine-wave exposures with the same amplitude and duration can have the same D1 and D2 while
having very different <code>|dB/dt|</code> because frequency differs. More subtly, a rotating vector
<code>B(t) = B0[cos(ωt), sin(ωt), 0]</code> has constant magnitude, so
<code>d|B|/dt = 0</code>, while <code>|dB/dt| = ωB0</code>. Field Workbench therefore derives slew
and spectrum from the XYZ vector components before reducing them to a scalar magnitude.</p>
<h2>Physical timing and sampling</h2>
<p>Integrals use the authoritative physical exposure timestamps and dwell intervals, including programmed gaps and
resolved current transitions. UI pauses, Scripted preview loops, and exported-video timing do not add exposure.
The analysis timebase is resolved independently of rendered frames because an animation frame rate is not a
physical exposure-sampling specification.</p>
<p>Finite rise/fall times, burst envelopes, pulse widths, gaps, channel phase, and source sampling
bandwidth matter. An ideal discontinuous square edge has unlimited ideal bandwidth and no ordinary
finite derivative at the discontinuity; Workbench metrics therefore describe the modeled or supplied
finite current record.</p>
<h2>Background and applied field</h2>
<p>A static background field contributes to total-field magnitude whenever it is included in the scene.
For interpretation, distinguish applied-coil, background, and total fields when that separation matters.
FFT mean removal is explicitly a spectrum-analysis choice; subtracting a whole-script mean is not the
same operation as redefining the time-domain exposure.</p>
<h2>What B²-time is not</h2>
<p>Magnetic energy density in a simple linear medium is proportional to <code>B²/(2µ)</code>, but
integrating that density over time gives energy-density × time, not absorbed energy per mass. In a
simple ohmic tissue model, deposited energy instead depends on an expression such as
<code>∫σ|E|²/ρ dt</code>, which requires induced electric field, conductivity, density, and geometry.
Workbench B²-time is therefore <b>not joules, gray, SAR, or absorbed biological energy</b>.</p>
<h2>Safety screening is a separate problem</h2>
<p>ICNIRP 2010 low-frequency guidance uses internal electric-field basic restrictions and external
reference levels. Complex/non-sinusoidal waveforms can require frequency-sensitive weighted-peak or
spectral-summation methods; EU Directive 2013/35/EU also specifies weighted-peak evaluation for worker
exposure. Those procedures are not duration integrals, and exceeding an external reference level does
not by itself establish that an internal basic restriction is exceeded.</p>
<p>Field Workbench's B-time/B²-time, FFT, slew, pulse timing, and frequency-weighted regional screening are
analysis tools, not regulatory compliance certification. The weighted screen identifies
the exact guideline edition, population, frequency range, and implementation used.</p>
<h2>Sources underlying this help topic</h2>
<ul>
<li>Advisory Group on Non-ionising Radiation (AGNIR/HPA), <i>Power frequency electromagnetic fields, melatonin and the risk of breast cancer</i>, RCE-1, 2006, exposure definitions section.</li>
<li>Röösli M. et al., <i>Environmental Health</i> 7:35, 2008.</li>
<li>Zhang J., Nair I., Sahl J., <i>Bioelectromagnetics</i> 18(5):365–375, 1997.</li>
<li>Kaune W.T. et al., <i>Bioelectromagnetics</i>, 2001, DOI 10.1002/bem.46.</li>
<li>Bowman J.D., McDevitt J.J., Breysse P.N., X2001 conference extended abstract, 2001.</li>
<li>Misakian M., <i>Exposure parameters during studies with ELF magnetic and electric fields</i>, NIST, 1999.</li>
<li>ICNIRP, <i>Guidelines for limiting exposure to time-varying electric and magnetic fields (1 Hz to 100 kHz)</i>, Health Physics 99(6):818–836, 2010.</li>
<li>ICNIRP, <i>Guidance on determining compliance of exposure to pulsed and complex non-sinusoidal waveforms below 100 kHz</i>, Health Physics 84(3):383–387, 2003.</li>
<li>European Parliament and Council, Directive 2013/35/EU, Annex II.</li>
<li>ICNIRP, <i>Gaps in knowledge relevant to the guidelines for limiting exposure to time-varying electric and magnetic fields (1 Hz–100 kHz)</i>, Health Physics 118(5):533–542, 2020.</li>
</ul>
<div class="note">These references establish use, rationale, or limitations of metric families. A
metric appearing in a study does not by itself establish a causal biological dose-response relationship.</div>
""",
        ),
    ),
    "saved_results_stats": (
        "Statistics and exposure metrics",
        "Saved results and comparison",
        _page(
            "Saved results and comparison statistics",
            """
<p>A saved result stores the measurement geometry, local sample coordinates, settings, field vectors,
statistics, scene signature, and coil-modeling report. New entries also store a compact replayable
copy of their field sources so selected candidate-coil currents can be fitted later, even after an
entry is imported into another project.</p>
<h2>Frozen data and freshness</h2>
<p>Saved results embed compressed 64-bit local/world sample coordinates and complete B vectors in
the model, together with stored statistics and capture metadata. Viewing an older result uses those
captured values; it does not silently replace them with a new field calculation.</p>
<p>A fingerprint compares the field-source scene, coil specifications and solver choice, effective
Background field, measurement settings, and local/world sampling grid. Changes to those inputs can
mark a result <b>Stale</b>. Presentation-only source styling is excluded from this check.</p>
<ul>
<li><b>Current:</b> the source and measurement fingerprint still matches.</li>
<li><b>Stale:</b> inputs changed, or the source geometry no longer supports volume measurement;
the captured result remains available.</li>
<li><b>Imported:</b> the result was copied from another scene and is retained independently of
that scene's current contents.</li>
<li><b>Source missing:</b> the original measurement object is no longer present.</li>
<li><b>Corrupt:</b> the saved entry cannot be interpreted or checked normally.</li>
</ul>
<p>Saved results support arbitrary multi-selection. View Results opens one independent viewer
for every selected entry, Compare opens one comparison containing every selected entry, and
Delete removes the complete selection as one undoable operation. Rename still applies to one entry.</p>
<h2>Exposure analyses</h2>
<p><b>Saved Results → Exposure analyses</b> is intentionally separate from Measurement-volume snapshots.
An Exposure Analysis is an automatically maintained project cache containing the compact LPBA40 regional field basis, authoritative physical exposure source, and every completed derived-analysis configuration. Preparation and derived results are serialized independently so completing another analysis page does not rewrite the immutable preparation data. Large NumPy result arrays use compressed binary storage with within-cache deduplication instead of verbose decimal JSON, and dirty-state polling uses stored checksums rather than repeatedly embedding the large blobs in the project signature. Its freshness fingerprint covers the saved exposure protocol,
referenced waveforms, electromagnetic scene/coil properties, background fields, selected brain target, and coil
field representation. A row whose fingerprint still matches is labelled <b>Current</b> and can be opened directly; stale historical analyses remain independently readable and never participate in a new preparation. <b>Create new analysis</b> always writes a fresh preparation rather than silently choosing or replacing a saved row. Automatic cache updates mark the project modified but do not silently save the project file to disk or add a derived-data-only step to Undo history. The Saved Results catalogue shows how many derived configurations are cached and a selected-cache detail panel reports the cached tools, sample count, full physical exposure duration, original preparation time, and separate preparation/derived compressed storage without having to reopen the analysis. Rename is optional organization only. <b>Manage cache…</b> opens a selective derived-cache manager: choose individual stored configurations, remove the selection, or remove all derived results while retaining the parent preparation. These explicit removals participate in project Undo. Since multiple configurations can share one deduplicated numeric array, per-row byte estimates would overstate or understate what deleting that row actually saves; the manager instead reports the exact compressed derived-cache storage reclaimed after each removal. A stale entry remains available exactly as calculated; it is never reused as a current basis and opening it clearly identifies it as historical. <b>Delete cache</b> removes the parent preparation cache and all of its derived page configurations together.</p>
<h2>Performance benchmark</h2>
<p>This mode asks how well each apparatus meets one shared specification. Sample grids may differ.
The overview shows three headline measures for every saved result:</p>
<ul>
<li><b>Target coverage:</b> percentage of finite samples inside the shared target band.</li>
<li><b>Uniformity spread:</b> <code>100 × (P95 − P5) ÷ |mean|</code>; lower is more uniform.</li>
<li><b>Directional consistency:</b> <code>100 × length(mean(unit B vectors))</code>.</li>
</ul>
<h2>Exposure matching</h2>
<p>This mode treats one selected saved result as the reference exposure and compares every other selected
saved result point-by-point. Local sample coordinates must match exactly. World placement may differ;
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
<h2>Difference plots</h2>
<p>Signed differences use <b>candidate minus reference</b> (B minus A in two-result plots).
Positive magnitude differences mean the candidate is stronger at that sample. Vector RMS difference
uses all three local-frame components, while direction plots compare vectors only where both fields
are non-negligible. With current normalization enabled, these plots use the fitted candidate field.</p>
<h2>Normalize current</h2>
<p>Current normalization is optional and applies only to candidate saved results. It does not alter the
stored result or its field-scale factors. The fit minimizes the pointwise difference between the
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
reported explicitly in the overview and full statistics. Fresh saved results retain scene display names
for this selector; older local entries use matching live-scene names when available.</p>
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
clearance override and display colour. Closed, unambiguous region-aware meshes can also be used by
Exposure Analysis as inward near-surface volumes. <b>Exposure sampling depth</b> controls that inward
depth for the whole object; it is independent of the Active/Show/Place/DRC switches because those
switches describe design/display semantics, while exposure ownership follows the authored region map.
The supported depth range is 0.01–30 mm.</p>
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
<div class="note"><b>Full control is not the same setting as Full search effort.</b> Full control only decides which Freedom controls are exposed in the setup wizard. Fast/Balanced/Full search effort is chosen later on the Limits page and controls the fidelity/search contract described below.</div>
<h2>Target and goal</h2>
<p><b>TARGET</b> selects the Measurement object(s) where the field matters. User-defined arrays such as 96-well centres remain exact.</p>
<p><b>GOAL</b> first asks what to match: a requested target-intensity specification or a saved result. Target-intensity mode then offers plain-language objectives such as putting the most samples inside the requested band, making intensity as uniform as possible, minimizing RMS error from the requested intensity, making direction consistent, or deliberately maximizing average/weakest/peak field strength. A short explanation under the dropdown defines exactly what each choice means.</p>
<p><b>Saved-result reference matching</b> compares one compatible Measurement target point-for-point with a saved result. The entry supplies its exact stored local sample grid, which Workbench maps through the current Measurement target; imported saved results from another scene are therefore usable when the Measurement shape/dimensions are compatible even if the scenes use different arbitrary world placement/orientation. Saved-result reference goals can minimize full-vector RMS difference, magnitude RMS difference, mean angular difference, or maximize the fraction of points meeting magnitude+direction tolerances. The Goal page also chooses how absolute orientation is treated: <b>Keep target-frame orientation fixed</b> preserves the exact local-frame Bx/By/Bz comparison, while <b>Exposure registration</b> treats absolute orientation as a nuisance pose. During scoring, optimizer-participating exposure sources may be rigidly rotated about the Measurement target's physical centre; their spatial field pattern and B-vectors therefore rotate together rather than merely rotating arrows at fixed samples. For asymmetric/imported Measurement geometry Workbench pivots around the physical volume centroid rather than the object transform origin. The candidate's stored world pose and DRC geometry are unchanged, and fixed/background sources remain world-fixed.</p>
<p><b>Exposure-registration telemetry:</b> When Exposure registration is active, Search Trace reports the inner nuisance-pose work separately from structural candidates: structural registration solves, unique registration-pose evaluations, additional registered-pose field-basis builds, registration-cache hits, Kabsch/inherited starting seeds, Fast coarse/standard/refined effort counts and cumulative inner-solve time. Fast deliberately spends less registration polish on disposable global-skeptic candidates and retains fuller refinement for promising/adaptive/DRC-rescored candidates; this changes search effort, not the exposure-scoring definition. The Best-so-far line also shows the fitted exposure RX/RY/RZ. These angles describe the scoring registration only; they do not replace or modify the candidate's physical world orientation shown in its setting.</p>
<h2>Freedom and relationships</h2>
<p>The Freedom page lists participating coils and lets each property use independent link groups. A Current Group A can contain one set of coils while Position Group A or Geometry Group A contains a different set. Properties marked Fixed are not changed; Independent gives that coil its own search variable for that property. In Simple/Complex mode, the earlier wizard page now acts only as a preset/filter: selected families arrive checked, but every exposed family remains editable here and may be unchecked before the search. The Freedom page is the final authority on what the optimizer may actually change.</p>
<p>In guided Simple/Complex searches, the Dimension section only shows rows compatible with the participating checked coils—for example, circular-only searches do not show racetrack or square dimensions. Full mode intentionally shows the complete existing-scene inventory. Dimensions include axial winding length, circular diameter, racetrack end diameter/straight length, and square/rectangular X/Y spans. Position includes shared X/Y/Z translation offsets plus a two-coil pair-spacing helper. Orientation is explicitly split into two different physical freedoms: <b>Rigid assembly rotation</b> rotates every linked coil centre around the link-group midpoint and rotates each coil plane with it, while <b>In-place coil rotation</b> leaves the centres fixed and only tilts/spins the coil planes. Guided Orientation setup defaults to rigid assembly rotation; in-place rotation is a separate opt-in. For a linked two-coil pair, rigid rotation therefore rotates the centre-to-centre axis itself, and pair spacing subsequently slides the coils symmetrically along that rotated axis. Every unlocked freedom also has an <b>Increment</b> that sets its discrete engineering/search resolution. Numeric Freedom-page bounds and increments remember the last values used when advancing from that page, while participating/freedom/row checkboxes are deliberately not remembered so each new search makes its allowed changes explicit. The <b>Defaults</b> button beside Next restores the numeric values derived when the optimizer opened on the current source scene without changing any selections. The default increments are 0.1 mA for current, 1 turn, 0.5 mm for dimensions and position, and 1° for orientation. The lattice is anchored to the source-scene value, so for example a 0.5 mm increment means source + integer multiples of 0.5 mm while the entered minimum/maximum remain hard bounds; finalists are never reported between increment values. The optimizer searches only coils that already exist in the scene and can search any number of their bounded variables simultaneously, including multiple independent link groups.</p>
<h2>Search effort and fidelity</h2>
<p>The optimizer separates three jobs: <b>Centreline reconnaissance</b> finds promising basins cheaply, <b>capable-physics refinement</b> moves the freedoms that the selected effort requires at a model that can actually represent them, and <b>authoritative validation</b> verifies finalists using the coil field model selected in Preferences. Current remains a linear excitation variable and is solved inside each structural candidate rather than being randomly searched alongside geometry. Optimizer 2.4 adds a non-negotiable <b>directional correctness floor</b>: for every bounded freedom that a stage promises to optimize, Workbench tests both legal bounds, follows any improving direction, brackets an interior turnover or proves a best bound, revisits coupled coordinates when another member moves, and measures a DRC/electrical feasibility boundary when that is what stops the direction. Sobol/global exploration is a skeptic for disconnected alternatives; it is not a substitute for closing an obvious one-dimensional trend.</p>
<ul>
<li><b>Fast — correctness-first reconnaissance + verification:</b> Centreline still performs the cheap structural search, but every scout-visible bounded freedom must satisfy the directional correctness floor before Fast may report convergence. Fast saves time by shrinking discretionary global exploration and alternate-basin work, not by stopping while a tested direction is still improving. A freedom that Centreline cannot represent may remain deliberately limited; the report labels it <i>limited</i> rather than implying it was optimized.</li>
<li><b>Balanced — hybrid physical directional refinement:</b> first applies the same correctness floor to scout-visible freedoms, then every selected freedom that Centreline cannot fully represent is directionally closed with a capable physical model together with its activated coupled coordinates. Balanced cannot report convergence while any required member of that physical coupling family remains open.</li>
<li><b>Full — full-dimensional physical directional refinement:</b> uses Centreline to discover diverse starting basins only after its own correctness floor. At capable physics, <i>every selected structural freedom</i> is reopened across its permitted range and must be directionally closed in its coupled context. Full cannot report convergence until every selected freedom is physically closed, analytically solved, or explicitly proven redundant.</li>
</ul>
<p><b>Protected correctness budgets:</b> Reconnaissance and capable-physics refinement each split their evaluation ceiling into a protected correctness reserve and a separate discretionary allowance. Mandatory endpoints, directional bracketing, coupled revisits, feasibility-boundary recovery, and the first physical anchor use the protected reserve. Relation-atlas extras, Sobol/global skeptics, adaptive basin challenges, and alternate physical anchors use only the discretionary allowance. If the protected reserve is exhausted while a required direction remains open, Workbench may return the best vetted candidate found so far but reports the search as <b>incomplete</b>; discretionary samples never count as proof of closure.</p>
<p>This distinction matters for winding-pack freedoms. For example, axial winding length is invisible to a zero-thickness Centreline source even though pair spacing and racetrack path dimensions are visible. Balanced must therefore search axial length physically with relevant coupled coordinates; Full reopens axial length, spacing and all other selected structural freedoms together under capable physics instead of freezing the Centreline answer.</p>
<h2>Preferences coil model</h2>
<p>The global <b>Coil field model</b> preference remains the final authority for optimizer validation, but the physical-refinement model may use a bounded-cost capable surrogate when appropriate. <b>Current sheet</b> is used directly for supported Polyline/racetrack coils. <b>Exact turns</b> uses Bundled for multidimensional refinement and Exact turns for final authority. <b>Auto</b> keeps dynamic final authority but uses a deterministic capability floor during physical refinement so a selected freedom cannot disappear because Auto happened to choose Centreline for one candidate. <b>Centreline</b> is valid when it represents the selected freedoms; Balanced/Full reject a run up front when a required freedom needs distributed-winding physics that Centreline cannot provide. Explicit requested models are not allowed to silently fall back to Centreline on optimizer-participating coils and still be called authoritative.</p>
<p><b>Search Trace</b> mirrors this staged contract while a run is active: freeze/validate → Centreline reconnaissance with mandatory bounds/directional closure → geometry/DRC feasibility → capable-physics directional refinement when required → authoritative validation. It distinguishes protected correctness work from discretionary global exploration. The copied optimization report includes <b>Search / Fidelity Contract</b>, <b>Freedom Coverage</b>, and <b>Directional Closure</b> sections. For each structural freedom, Directional Closure reports the tested effective bounds, best scalar value, closure reason (interior turnover, lower/upper bound best, measured feasibility boundary, saturation, redundancy, or fixed range), trend, reopen count, and any measured safe/infeasible boundary pair. A result whose best value is the user bound is therefore explicitly labelled <i>bound-limited</i> rather than presented as an unconstrained optimum.</p>
<p><b>Candidate lifecycle:</b> The optimizer keeps one compact <b>flat master solution registry</b> across reconnaissance, DRC, repair, capable physical refinement/promotion, and authoritative validation. During a run, the Search History header reports this durable population as <b>solutions discovered • active • retired • blocked • authoritative</b> rather than resetting stage-local evaluated/feasible/rejected counters; active, retired and blocked partition the current master population, while authoritative independently counts solutions that have reached full-model validation. A solution identity contains only structural freedoms; an inner-solved current is stored on each scout/physical-refinement/promotion/authoritative observation and never becomes part of the canonical solution key. This lets the same geometry legitimately solve to different currents at different fidelities while retaining one shared solution history. Structural identities are canonicalized on the requested increment lattice and periodic orientation offsets are normalized, so equivalent 0°/+360°/−360° search coordinates merge into one physical solution record. A stage may <i>retire</i> a lower-ranked solution from further compute, but it does not delete it; later feasibility or higher-fidelity information can therefore re-rank earlier solutions. DRC-boundary repairs and rescue settings are peer solutions in the same population, with provenance cross-references describing where they came from instead of parent/child ownership. Repaired/rescue solutions are immediately re-scored with the cheap scene-free Centreline evaluator when available. Movable physical refinement, reduced-target fixed-coordinate promotion when needed, and authoritative validation accumulate as additional evaluations on that same solution; ranking always uses the highest trustworthy fidelity available, so an optimistic scout estimate cannot outrank its own later full-model result. Strict physical duplicates merge into one record, while different geometries that are field-equivalent remain separate solutions linked as an equivalence family so diversity/decimation can choose a representative without losing history. DRC checks walk progressively down the magnetic ranking until enough naturally clear alternatives are found within the selected effort budget, but the globally best magnetic scout, the relevant source baseline, and every retained strong-basin representative are mandatory DRC verdicts before that progressive pass may stop. A blocked strong solution first receives an axis-first repair: Workbench holds every other freedom fixed, prioritizes pair spacing/translation escape directions, bisects any blocked-to-clear one-coordinate route to the nearest legal lattice point, and re-scores that repaired geometry on its own field before falling back to a generic N-dimensional clear anchor. After clear terrain is known, deterministic one-freedom coordinate polish runs beside the local Sobol wave so simple improvements such as tightening pair spacing until DRC becomes active cannot be missed merely because the random local cloud perturbed several coordinates at once. Final authoritative slots are reserved across source, naturally clear and boundary-repaired roles so illegal magnetic winners cannot crowd out a strong feasible basin. The registry also prepares stable backend views for Finalists, Best feasible, Best overall, DRC boundary, Source baseline and Strong basins for the later completed-search browser. Optimization reports begin with a <b>Search / Fidelity Contract</b>, <b>Freedom Coverage</b>, and per-stage <b>Directional Closure</b> audit, then include a <b>Fidelity audit</b> for the source baseline, best overall scout, best naturally DRC-clear scout, best DRC-boundary solution and every authoritatively tested solution. Together these show whether each freedom was actually explored at capable physics and whether lower-fidelity optimism changed at promotion/authoritative fidelity.</p>
<h2>Completed-search Results Explorer</h2>
<p>After a search completes, Results becomes a browser over the same flat master registry rather than a finalist-only table. The left-side views select <b>Finalists</b>, <b>Best feasible</b>, <b>Best overall</b> (which may include DRC-blocked magnetic ideals), <b>DRC boundary</b>, <b>Source baseline</b>, <b>Strong basins</b>, or <b>All solutions</b>. These are rankings of the same canonical solution records, so one solution may appear in several views without being duplicated. The solution list keeps the top row compact and comparison-focused: Saved-result-reference searches show match, vector/magnitude RMS, angular error, mean field, optimizer-affected DRC clearance and fidelity, while field-definition searches show coverage, RMS error, uniformity, direction, mean field, optimizer-affected DRC clearance and fidelity. Expand a solution to drill into solved current and a nested Geometry outline with optimized freedoms plus per-coil resulting position, movement from the frozen source pose, orientation and design/path changes. This avoids a very wide one-line geometry description while preserving the details when they matter. <b>Selecting a solution is deliberately data-only:</b> Results shows a lightweight Compare-style <b>Comparison</b> overview and <b>Full statistics</b> page directly from the registry's highest trustworthy retained fidelity instead of rebuilding a complete Workbench/Plotly 3-D scene for every click. Saved-result-reference searches use the same exposure-matching card language as saved-results comparison; field-definition searches use the same shared-target benchmark card language. The physical candidate scene is materialized only when the user explicitly asks for an operation that needs it.</p>
<p>The embedded <b>Comparison</b> and <b>Full statistics</b> pages are intentionally instantaneous registry views; they do not rerun field physics merely because the selection changed. <b>Compare</b> remains available when a fresh full-field calculation, detailed difference plots, or a multi-solution comparison is wanted. Saved-result-reference optimizations use the reference entry's stored <i>local</i> sample grid registered through the optimizer's selected Measurement target, exactly matching the optimizer objective even if the reference entry came from another scene placement/orientation. Exposure-registration optimizer results are first rematerialized in the scoring-only rigid registration that produced their authoritative metrics, so the full Compare workflow starts from that already-registered exposure rather than applying a second vector-only alignment. Registry solutions are rematerialized using the coil-modeling method recorded for that evaluation rather than whatever model happens to be selected later in the live source scene. A fresh full-field comparison of an authoritative solution is checked against its stored authoritative optimizer metrics and Workbench warns if match/RMS/angle/mean-field values drift beyond numerical tolerance. Field-definition optimizations open the shared-target benchmark using the optimizer's requested intensity and tolerance. <b>Save result</b> performs the candidate field analysis and stores it as a frozen saved result in the optimizer source scene, including the scoring field-source scene for later replay/current normalization, so the exposure can be kept without keeping the whole candidate model. <b>Open as project</b> creates a complete unsaved Project Manager project for 3-D inspection. <b>Apply to source</b> changes the participating source coils. <b>Append to source</b> instead preserves the originals and copies only participating coils whose transform or physical/path design differs; unchanged/current-only source coils are skipped so a current adjustment alone does not create an overlapping duplicate.</p>
<h2>Concurrent candidate workers</h2>
<p>The worker count in <b>Project Manager → Edit → Preferences</b> controls how many independent candidate processes may be
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
<p>Finalists report the changed setting, actual coil currents, and the metrics relevant to the selected goal. Target-intensity runs show target coverage, P95–P5 spread, directional consistency, mean field, RMS target error, and DRC clearance; Saved-result-reference runs show point-match coverage, vector/magnitude RMS differences, angular error, and DRC clearance. The optimizer window is independent and its worker uses a frozen launch-time scene, so the main Workbench remains usable while a search runs. A finalist can be previewed, opened as a separate unsaved project for side-by-side inspection, or applied back to the source model as one undoable edit; Workbench warns if that source scene changed after the optimization began. <b>Copy report</b> copies the reviewed optimization plan, run/search/convergence summary, staged fidelity/budget contract, per-freedom coverage and directional-closure audit, the leading distinct finalists, and full details for the currently selected finalist so the result can be pasted into notes, email, or another analysis workflow without losing the search context.</p>
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
<tr><td><b>B-time / B²-time</b></td><td>Exposure-analysis indices formed by integrating regional
spatial RMS |B| or its square over physical exposure time. They are descriptive magnetic-field
magnitude-duration measures, not biological dose.</td></tr>
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
<tr><td><b>Exposure RMS</b></td><td>Space-time RMS for one analysed region over a finite physical exposure:
<code>sqrt(B²-time / exposure time)</code>.</td></tr>
<tr><td><b>RMS</b></td><td>Root mean square: square values, average them, then take the square root.
Vector RMS error uses squared vector-difference lengths.</td></tr>
<tr><td><b>Scene role</b></td><td>How passive geometry is used: Visual only, Measurement volume, or
Exclusion zone. Passive geometry does not create or alter magnetic fields.</td></tr>
<tr><td><b>Saved result</b></td><td>A stored field-analysis result plus the metadata needed for later
comparison. It is not a continuously linked live calculation.</td></tr>
<tr><td><b>Solver backend/model</b></td><td>The electromagnetic representation used to evaluate a
coil, such as Centreline, Exact turns, Bundled, or Current sheet.</td></tr>
<tr><td><b>Voxel</b></td><td>One three-dimensional image cell. Isotropic voxels have equal physical
size along all three image axes.</td></tr>
<tr><td><b>Worker (optimizer)</b></td><td>One spawned local Python process evaluating an independent
optimizer candidate with its own private scene adapter. It is not a separate Project Manager project or a remote computer.</td></tr>
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
<li><b>3 — Review the AI interpretation and import:</b> the larger review pane shows those human-facing fields first, followed by warnings and the object list. After validation, choose <b>Modify current scene</b> to apply the complete result to the current project model, or <b>Paste into new project</b> to create a separate Project Manager project and leave the source untouched.</li>
<li>Watch the activity bar at the bottom of the window during prompt generation/copying, validation, and import/replacement. It includes a live elapsed-time counter. When current-scene/DRC context is requested, the bar reports scene-object summarization and live DRC object-pair progress rather than merely indicating that Workbench is busy.</li>
</ol>
<p>AI Help treats every returned recipe as the <b>complete desired recipe-compatible object scene</b>. The
assistant is allowed to propose preserving, moving, changing, adding, replacing, or removing representable
objects; it does so by returning the final complete object set, not by sending imperative patch/delete
commands. <b>Modify current scene</b> asks for confirmation and atomically replaces the current object scene
while preserving scene-wide DRC settings and notes as one undoable operation. <b>Paste into new project</b>
builds the same validated recipe in a separate Project Manager project and leaves the source untouched. The external AI has
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
sample sets (including built-in well-plate centres); target statistics, saved results, and comparison; bench-data
Field-correction fitting; DRC; and the bounded coil optimizer.</p>
<p>It also names important <b>outside-Recipe-v1</b> workflows—Manual/WMM2025 Background fields, built-in
culture plates, UPENN-GBM imports, arbitrary STL/OBJ meshes, and portable/OpenSCAD/FEMM/GetDP/COMSOL export—so the
assistant can recommend them without fabricating JSON types that the parser cannot accept. The optimizer is
described accurately as changing bounded properties of coils already present in its frozen launch scene,
not as a topology generator.</p>
<p>The unified Maps workspace is also identified as a <b>frozen launch-time scene copy</b>.
Their 2D, 3D, and Brain views all derive from the same frozen launch-time scene copy. If the source scene changes afterward,
an already-open workspace does not become a live view of that edit; open a fresh workspace when the changed
model needs to be inspected.</p>
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
physical coil specifications, passive geometry, DRC settings, saved results, standalone waveform resources, and free-form scene notes.
Open <b>Analyze → Notes</b> to keep design decisions, measurements, and build details with the
project. The application-level
solver preference, memory budget, and worker concurrency limit are stored in local preferences rather than in the scene so a
design remains portable between installations.</p>
<h2>Project tabs</h2>
<p>The Project Manager tab row keeps several independent projects open in one suite session.
<b>New project</b> creates a blank project with no scene objects; <b>Open project</b> and
<b>Open Recent</b> load saved <code>.magpy.json</code> projects. There is no special packaged startup/demo scene; populated projects are ordinary project files. Opening a file that is already loaded activates its existing project tab so two editors cannot unknowingly save to the same path.</p>
<ul>
<li>Each project tab owns its own live model/data instance and summary dashboard. The dashboard groups Model, Maps / Analysis, Exposures, and Data / Results beside the applications that operate on them. Exposure Analysis is paired with Map Creator under Maps / Analysis; Saved Results is paired with Dataset Manager under Data / Results.</li>
<li>An asterisk after a project tab name marks unsaved project changes.</li>
<li>Application windows opened from different project tabs may remain open at the same time.</li>
<li>Closing a Model Editor does not close its project; closing the project tab closes that project's
associated application windows and asks about unsaved changes when necessary.</li>
<li>Model Editor edits exactly one project model. Project Manager tabs own project identity, lifetime, save state, and multi-project switching.</li>
<li><b>Waveform Editor</b> is project-aware and can remain open beside windows from other projects. Its Light/Dark appearance follows the Project Manager global preference; it has no separate WaveBuilder theme setting.</li>
<li>The project resource registry uses namespaced identities such as <code>model_object:&lt;id&gt;</code>,
<code>snapshot:&lt;id&gt;</code>, <code>saved_map:&lt;id&gt;</code>, <code>exposure_analysis:&lt;id&gt;</code>, <code>exposure:&lt;id&gt;</code>, <code>dataset:&lt;id&gt;</code>, and <code>waveform:&lt;id&gt;</code>. These are suite
references over the IDs already stored in the native scene; opening a project does not rewrite it.</li>
</ul>
<p><b>File → Import objects from scene…</b> copies selected objects from another saved scene without
replacing the current project. Complete groups or individual children can be chosen from a checkbox
tree. Object parameters, transforms, hierarchy, coil construction metadata, mesh roles/settings,
and visibility are copied; scene notes, saved results, and scene-wide DRC settings are deliberately not
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
    "Brain model and atlas",
    "Objects",
    "UPENN-GBM cases",
    "Coils and construction",
    "Electromagnetic solver",
    "Field views",
    "Playback",
    "Exposures",
    "Statistics and exposure metrics",
    "Measurements and calibration",
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

        search_row = QHBoxLayout()
        search = QLineEdit()
        search.setPlaceholderText("Filter help topics…")
        search.setClearButtonEnabled(True)
        search.setAccessibleName("Filter help topics")
        search_row.addWidget(search, 1)
        self.search_edit = search

        filter_indicator = QLabel("Filtered")
        filter_indicator.setProperty("statusTone", "muted")
        filter_indicator.setAccessibleName("Help topics filtered")
        filter_indicator.setToolTip("Only help topics matching the search are shown")
        filter_indicator.setVisible(False)
        search_row.addWidget(filter_indicator)
        self.filter_indicator = filter_indicator
        outer.addLayout(search_row)

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
        self.filter_indicator.setVisible(bool(needle))
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
