# Field Workbench verification

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
The default-scene regression loads the packaged HM document and verifies its
grouped coils, Measurement sphere, construction metadata, snapshot, centre
field, and renderability. The lone Point-sensor regression verifies both the
direct numeric value and the explicit zero-referenced display range, while CSV
continues to export exactly one measured sample.

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
