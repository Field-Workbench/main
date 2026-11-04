# Field Workbench verification

## Agent-access integration

The shared CLI/MCP command system is covered by `test_automation_commands.py`,
`test_automation_transport.py`, and `test_automation_mcp.py`. Run these with
`python -m unittest discover -s tests -p 'test_automation_*.py' -v`.
The MCP stdio round trip requires the optional `requirements-agent.txt` dependency;
without it, only that test is skipped. Local transport tests require loopback
socket access. They use temporary projects and offscreen Qt windows.
See [Agent access](docs/AGENT_ACCESS.md) for architecture, command semantics, and
extension points. Manual user testing and packaged CLI/MCP acceptance are deferred.

## Packaged release acceptance (pending)

Record the tested source commit, application version, Windows/Python versions,
ZIP checksums, and pass/fail evidence. Use the extracted onedir release from a clean
folder outside the development checkout. The GUI must run without the development
Python environment; the separate agent client deliberately has its own Python setup.

- Launch normally and confirm no new automation instance is discoverable. Launch
  with `--automation`, discover it, and read the same project through CLI and a real
  MCP client using matching source. With multiple instances, select an explicit ID.
- On a disposable project, add/edit an object, verify the GUI updates and Undo,
  reject a stale revision, and save/reopen. Verify ordinary launches remain opt-in.
- Exercise field, measurement and DRC jobs; inspect/export results, cancel work,
  and verify stale measurement/optimizer application is rejected after model edits.
  Verify shutdown cleanup and reconnect after restart with the new instance ID.
- Open `examples/4D-box.magpy.json`. Generate static and scripted 2D maps with sensor
  placements and well rims; test plate transforms, clipping and overlays independently.
- Generate Scripted 2D and 3D maps with Live measurement analysis. Check playback,
  scrubbing, whole-object aggregates, labelled well child rows, selection highlights,
  and metric-column order across updates. Generated volume grids remain aggregate rows.
  Static Measurement snapshots and Brain Areas remain separate workflows.
- Check saved-result reopening, built-in Help, 3D worker processes, video export,
  and normal project/application shutdown. Run the broader Windows checklist in
  [BUILDING_WINDOWS.md](BUILDING_WINDOWS.md).

Do not mark these checks passed until performed on the actual candidate. Automated
source coverage and the 2.5-second startup smoke test are insufficient substitutes.

## Test layers

Field Workbench uses four deliberately separate test layers. A direct parity
test and an independent physics test answer different questions and should not
be treated as substitutes for one another.

| Layer | What it verifies | Normal suite |
| --- | --- | --- |
| Workbench behaviour | Editing, persistence, UI, measurements, snapshots, rendering | Yes |
| Independent physics | Results agree with closed-form field equations and physical laws | Yes |
| Direct Magpylib parity | Workbench sends the correct geometry, transforms, current, turns, factors, and points to Magpylib | Yes |
| Upstream dependency tests | The pinned third-party releases pass their maintainers' own suites | Optional pre-release run |

## Updating an existing source tree

A Field Workbench source ZIP is a clean snapshot, but extracting it *over* an older
source directory cannot remove files that were deleted in newer releases. That can
leave obsolete test modules or retired UI modules discoverable even though they are
not part of the new ZIP. Prefer replacing the old `field_workbench/` directory. If an
overlay update is intentional, run this once before the suite:

```bash
python tools/clean_obsolete_source_files.py
```

The cleaner only removes explicitly retired release paths and Python bytecode caches;
it does not touch scenes, exports, or arbitrary user files.

## Everyday complete suite

Run from the project directory in the activated `field-workbench` environment:

```bash
python -m unittest discover -s tests -v
```

The suite includes complete `Bx`, `By`, and `Bz` comparisons for circular and
polyline coils, mixed-source aggregation, multiple vectorized observation
points, rotation and translation, current reversal, superposition,
cancellation, turns/current/correction-factor scaling, RMS/DC versus peak-sine
heating, SI/mm/µT conversion, the `B = µ₀H` free-space relationship, analytical
single-loop and Helmholtz-pair fields, and saved reference-scene fingerprints.
Scene-level notes are also checked for metadata persistence, dirty-state tracking,
and coalesced undo/redo behaviour, while fullscreen teardown is exercised after
the detached Qt host has already been scheduled for deletion.
It also verifies rendered construction solids remain electrically inert, rotated
annular DRC avoids world-box false positives, hidden exclusions stay active,
global and per-zone clearances classify correctly, coil-to-coil penetration is
detected, and saved reports become stale after scene edits. Imported-mesh
regressions separately verify watertight/open/degenerate/inconsistently oriented
health diagnostics, closed-surface inverse containment, open-surface degraded
confidence, actual triangle penetration, hidden exclusions, transformed coil
assemblies, adaptive tolerance, and the guarantee that mesh DRC cannot change B.
Optimizer regressions verify bounded existing-coil current fitting, pair-spacing
movement, directional/coverage/uniformity statistics, DRC classification, cooperative
cancellation, reusable Box/Cylinder/Sphere/closed-mesh target discovery and sampling,
broken-mesh eligibility explanations, snapshot freshness after mesh damage, and
undoable finalist application. Lower-level historical search math remains covered where
it is still reusable. The GUI layer verifies the TARGET → GOAL → FREEDOM → LIMITS →
REVIEW → RESULTS wizard structure, suggested-target selection, explicit freedom toggles,
DRC defaults, wheel scrolling without accidental editor changes, and finalist reporting
of actual currents/field metrics. Final candidate scenes are recalculated through the
ordinary Workbench field model and exact DRC rather than trusting cached sweep scores.
Waveform I/O regressions independently exercise the format-neutral reader/writer registry, explicit reader/profile selection, native FWB waveform JSON round trips, common-time normalization, strict timestamp validation, and explicit engineering-unit conversion.

New application projects intentionally start empty. Focused workflow tests use the
small scene in `tests/workflow_fixture.py`; it is a test fixture, not a packaged
default/demo project. The lone Point-sensor regression verifies both the direct
numeric value and the explicit zero-referenced display range, while CSV continues
to export exactly one measured sample.

Concurrent 3D-render supervision is tested without Qt or the field engine: an
eight-child churn test primes the supported process backend, signals every
child through a one-way cancellation pipe, and verifies clean exit/closure.
This specifically guards against reintroducing semaphore-backed per-render
Events and source children importing a second Qt/WebEngine application tree.

Surface-region regressions additionally load the bundled Generic bust sidecar, verify the
exact STL/topology/order lock and complete five-region partition, render an 8,117-triangle
Face overlay, and exercise the inspector row-selection / clear-highlight interaction. For a
release candidate, also inspect all five highlights visually once: Scalp, Face, Ear L, Ear R,
and Neck/Shoulders should reproduce the Blender-authored boundaries before any DRC or optimizer
semantics are assigned to those labels.

SRI24 atlas regressions verify the bundled archive checksum, each compressed
and decompressed asset checksum, native 240×240×155 RAS affine, label-table
coverage, all left/right pairs, alignment with the brain shell, and the nine
explicitly documented `tzo116plus` ventricular-table supplements. The shared
NIfTI tests also confirm that the UPENN-GBM wrapper retains its original grid,
affine, and label validation after using the common decoder.

Brain View regressions verify that its application-authored hierarchy partitions
all 56 non-background LPBA40 labels exactly once, deterministic sampling preserves
the full atlas volume, placed samples follow the shell transform, and display
downsampling retains every label. The checked-in mesh cache is verified against
its format version, manifest checksum, source volume/table checksums, complete
label set, offsets, triangle indices, counts, and the same scene-placement
transform. Selector tests also lock the geometry-once contract: highlighting and
grid/context changes use incremental Plotly updates. PlotView regressions verify
that live interactions send only their current delta and that traces with the same
property set are batched with per-trace values, preventing a Whole brain → major
region selection from causing one WebGL redraw per categorical region colour.
Constant fields still produce exact weighted metrics, and the compact quadratic
waveform calculation matches direct spatial RMS values.

## Regenerating the LPBA40 visual cache

The application and release environment do not need VTK. Only install the
developer asset requirements when deliberately changing the atlas, visual
stride, smoothing, or shell-generation method:

```bash
python -m pip install -r requirements-mesh-build.txt
python tools/build_lpba40_mesh_cache.py
python -m unittest tests.test_brain_mesh tests.test_sri24_atlas -v
```

The builder writes fieldworkbench/assets/sri24_labels/lpba40_meshes.npz, updates
its manifest entry, and prints the label/vertex/triangle counts and SHA-256.

The first verification test also requires the installed calculation engines to
match `requirements.txt` exactly. This prevents a passing result from being
misattributed to the pinned versions when another version is actually active.

## Release-candidate manual gate

The temporary development release-review file is intentionally not shipped. Keep the
remaining release gates here so a clean source archive has one durable checklist. Before
publishing a release candidate:

1. Run the complete unittest discovery suite in the pinned normal environment and resolve
   every failure/skip that is not an explicitly expected platform/WebEngine skip.
2. Exercise Project Manager save/open/close with an empty new project and at least one
   realistic populated project. Confirm project resources persist after a full application
   restart.
3. Check Static 2D/3D maps, a Measurement-volume snapshot, Scripted 2D/3D/Brain playback,
   Brain Areas interaction, trace export, audio start/stop/source switching, and one short
   video export. Scripted 2D/3D must not expose the Static snapshot panel.
4. Run **Project Manager → Maps / Analysis → Exposure Analysis** on a representative long/repeating exposure (the **Analyze exposure…** shortcut in Exposure Editor should open the same workflow).
   Confirm the chooser lists eligible targets plus saved analyses for the selected exposure/target,
   **Create new analysis** writes a fresh preparation, and **Calculate all uncached analyses** uses
   the single-worker queue (one Running row at a time, later jobs Queued, Pause/Run and Cancel work).
   Close/reopen the saved resource from **Saved Results → Exposure analyses** and confirm cached pages
   open without calculation. Changing a real calculation setting should create/queue only that new
   configuration, while display-only FFT controls should not. Confirm the Exposure Analysis window is
   taskbar-visible with the other Field Workbench windows and does not block project/application close.
   Finally, verify **Delete cache** removes the parent preparation plus all of its derived results.
5. Verify cache freshness deliberately: a numerical/physical dependency edit must mark the
   analysis Stale; a presentation-only edit such as a label/colour/notes change must leave the
   otherwise identical cache Current. Save/reopen the project and repeat the check.
6. Exercise waveform import from a portable waveform file and from another `.magpy.json`
   project, plus one encoded/profile import. Build and reopen an exposure that references the
   imported waveform.
7. Exercise a raw measurement dataset, a Calibration / Composite fit, and a coil correction
   link. Confirm wheel scrolling cannot accidentally change protected combo/spin editors.
8. Open Help Topics in both themes and spot-check Overview, Typical workflow, Playback, Saved
   exposures, Exposure analysis, Exposure statistics and research basis, Spectrum/FFT, Timing,
   Threshold, Frequency-weighted screening, Saved results, and Limitations.
9. On the intended Windows release machine, build the frozen executable with
   `build_windows.bat`, inspect `BUILD_REPORT.txt`, run the packaged startup smoke test, and
   repeat the key WebEngine/analysis/save-reopen paths described in `BUILDING_WINDOWS.md`.
10. Run one representative **Optimizer 2.4 racetrack directional-closure acceptance** on the 48-well
    plate scene that motivated the refactor. Keep current, axial winding length, pair spacing, racetrack
    end diameter, and racetrack straight length free. First run Fast with Current Sheet authority and
    confirm **Directional Closure — Centreline Reconnaissance** shows both legal bounds tested for end
    diameter, straight length, and spacing. If the best result is at 85 mm (or another user bound), the
    report must say upper/lower-bound limited; an interior answer is acceptable only with turnover/bracket
    evidence or a measured DRC/electrical feasibility boundary. Then run Balanced and Full and confirm the
    capable-physics Directional Closure section closes the required coupled family. Full must show every
    structural freedom physically directionally closed (or analytically solved/proven redundant), and the
    Full effort contract must be SATISFIED. Search / Fidelity Contract must also show protected correctness
    and discretionary consumption separately. A run that exhausts correctness reserve with an open required
    direction must return INCOMPLETE rather than converged.
11. If solver/export code changed, run the external Gmsh/GetDP acceptance below. If dependency
    pins changed, run the optional upstream dependency suites. Hardware-output paths should be
    checked on the actual target device when they are part of the release claim.
12. Create the matching source archive with `python tools/build_source_release.py`, inspect its
    contents, and confirm development-only review notes/build artifacts are absent.

## Focused layers

Run only the physics/parity/reference module:

```bash
python -m unittest discover -s tests -p "test_field_verification.py" -v
```

Run only the existing core workflow or GUI smoke modules:

```bash
python -m unittest discover -s tests -p "test_core_workflow.py" -v
python -m unittest discover -s tests -p "test_ui_smoke.py" -v
python -m unittest discover -s tests -p "test_optimizer.py" -v
python -m unittest discover -s tests -p "test_nifti.py" -v
python -m unittest discover -s tests -p "test_sri24_atlas.py" -v
python -m unittest discover -s tests -p "test_brain_analysis.py" -v
python -m unittest discover -s tests -p "test_brain_mesh.py" -v
```

Run the dependency-light **Optimizer 2.4 contract gate** after changes to optimizer
capabilities, effort modes, model selection, staged budgets, physical refinement,
or candidate-registry fidelity:

```bash
python -m unittest \
  tests.test_optimizer \
  tests.test_candidate_registry \
  tests.test_optimizer_23_policy \
  tests.test_optimizer_24_directional -v
```

This gate includes both optimizer failures that motivated Optimizer 2.3/2.4: axial winding length is
invisible to Centreline, and a scout-visible bounded geometry freedom may not stop at an arbitrary
interior sample while its observed trend still improves toward an untested bound. The 2.4 tests cover
monotonic lower/upper-bound closure, interior turnover, source-anchored increment lattices, measured DRC
feasibility boundaries, solver-failure rejection, racetrack shape/spacing coupling, protected correctness
budgets, and reporting/Search Trace integration.

Run the focused long-exposure regional-analysis release gate after changes to exposure clocks, Exposure Analysis payloads/caches, regional statistics, FFT, or frequency weighting:

```bash
python -m unittest \
  tests.test_continuous_exposures \
  tests.test_region_analysis \
  tests.test_region_time_domain_streaming \
  tests.test_region_spectral_streaming \
  tests.test_region_frequency_weighting_streaming \
  tests.test_frequency_weighting -v
```

This gate covers both frozen replay modes, independence from playback/display traces, compressed-cycle time-domain aggregation, source-honest long-record FFT, and bounded phase-aware frequency weighting. Exposure Analysis persistence/freshness is additionally exercised by the normal core/UI suites. It requires the normal application dependencies from `requirements.txt`; the three `test_region_*_streaming` modules and `test_frequency_weighting` remain useful dependency-light subsets.

## Runnable Gmsh/GetDP export acceptance

The ordinary suite verifies GetDP selection rules, normalized source geometry,
signed ampere-turns, arbitrary transforms, circular analytic on-axis references,
racetrack and Square/rectangular centreline references, archive contents, and the Export-dialog
path without requiring external solver binaries.
Before releasing a translator change, also exercise one generated project with
the compatibility-floor executables:

```bash
gmsh --version   # 4.8.4 or later
getdp --version  # 3.2.0 or later
unzip single_loop.getdp.zip -d single_loop_getdp
cd single_loop_getdp
sh run.sh
```

The run must create an ASCII MSH 2.2 mesh with distinct physical groups named
`Air`, `Coil_1`, and `OuterBoundary`; finish both the `Magnetostatics` resolution
and `ExportFields` post-operation; and write `results/B_uT.pos`,
`results/J_A_per_m2.pos`, and `results/samples.txt`. Match the output rows to
`samples.csv` in order. For the single-loop reference, confirm the field sign
and convergence toward the listed ideal-loop values as the local mesh is
refined; exact equality is neither expected nor desirable because GetDP uses a
finite winding pack and finite outer boundary while the reference is an ideal
centreline loop in unbounded space.

After the circular smoke test, exercise the packaged path-coil fixtures:

1. Open `tests/fixtures/single_racetrack_reference.magpy.json`, select the racetrack,
   export GetDP, run the generated project, and compare `results/samples.txt` with
   `samples.csv`.
2. Open `tests/fixtures/frilz_pair_reference.magpy.json`, select both racetracks,
   export and run again. The fixture uses 100 mm end diameter, 100 mm straight
   length, 222 turns, 27.5 mA and 89 mm centre-to-centre pair spacing.

For these fixtures `samples.csv` is the Field Workbench segmented-Polyline
centreline Biot-Savart reference, while GetDP uses a finite capsule-band winding
volume. Confirm the centre field points along +Z, the pair is symmetric about
Z=0, and the FEM result follows the listed centreline values without large
transverse leakage. The FRILZ-style pair's centreline reference is approximately
63.25 µT at the midpoint; finite-pack/mesh differences are expected.

For GetDP v5 Square/rectangular acceptance:

1. Open `tests/fixtures/single_square_reference.magpy.json`, select the Square coil,
   export/run GetDP, and compare `results/samples.txt` with `samples.csv`. The
   ideal FWB centreline target at the origin is approximately 22.6274 µT along +Z.
2. Open `tests/fixtures/square_pair_reference.magpy.json`, select both Square coils,
   export/run again, and verify symmetry about Z=0. The pair centreline target at
   the midpoint is approximately 34.1333 µT along +Z.

The Square source is a finite mitered rectangular winding band in GetDP, while
`samples.csv` remains the exact Field Workbench Polyline-centreline reference.

## Optional upstream dependency validation

Do this before a release, after changing a dependency pin, or when diagnosing a
suspected dependency installation problem. It is intentionally excluded from
normal discovery because it requires network access, clones two repositories,
creates isolated virtual environments, installs their complete test
dependencies, and may take substantially longer than the Workbench suite.

```bash
python tools/run_upstream_dependency_tests.py
```

Useful narrower runs:

```bash
python tools/run_upstream_dependency_tests.py --target magpylib
python tools/run_upstream_dependency_tests.py --target studio
```

The script reads the exact versions from `requirements.txt`, checks out their
release tags, and runs each repository's own pytest suite in a clean temporary
environment. Add `--workdir upstream-test-work` only when you want to preserve
the clones and environments for inspection.

Do not copy upstream tests into `tests/`. They verify the dependency's internal
implementation, while Field Workbench's parity tests verify the boundary
between our application and that dependency.

## Scripted audio focused checks

`python -m unittest discover -s tests -p test_waveform_audio.py -v` covers
authored-clock retention, time-weighted DC normalization, continuous and
once-per-step phase/routing, bounded audio buffers, activation failure/races,
source switching, stop/disposal, silence, and 1 kHz → 100 Hz playback at 0.10×.
The JavaScript checks use Node.js with a mock audio context and also validate the
complete embedded viewer script syntax.

For the device-level check, build a playback with Drive trace enabled, choose Audio and
press Play. Exercise Stop/Reset, seeking, Loop, speed changes, source switching,
and closing/hiding the viewer. Reopen the saved result and confirm Audio is off.
Check continuous and once-per-step sources, including an inactive routing step.
