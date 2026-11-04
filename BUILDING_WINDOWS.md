# Building the Windows release

Use a clean 64-bit Python 3.12 virtual environment. The release build is a
`onedir` application because Field Workbench uses Qt WebEngine/Chromium.

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements-build.txt
build_windows.bat
```

The batch file freezes Magpylib's Plotly display backend plus the Matplotlib
runtime that Magpylib imports at module load time. Matplotlib and Pillow are
therefore release dependencies and must not be excluded from the frozen build.
Notebook, dataframe, alternate mesh, and alternate-Qt stacks remain excluded.
The batch file also explicitly packages the internal optimizer benchmark scene, built-in anatomy,
and SRI24 anatomical-label resources; do not remove those `--add-data` entries
from the build command.

Brain View's LPBA40 display shells are already present in the packaged
sri24_labels asset directory. VTK is intentionally not installed or collected
for a normal Windows release; it is used only by the separate developer mesh
regeneration workflow documented in TESTING.md.

After PyInstaller finishes, the build automatically:

1. copies the Field Workbench GPL and project notice beside the executable;
2. builds `licenses/` from distributions found in PyInstaller's actual analysis;
3. records included Python distributions and Qt libraries in
   `THIRD_PARTY_NOTICES.txt` and `BUILD_REPORT.txt`;
4. refuses to approve a build above 800 MB by default; and
5. launches a brief packaged startup smoke test.

Qt WebEngine is genuinely required and remains the largest part of the build.
The final size varies with Python and Qt patch releases, and the audit rejects a build above the configured size limit. To adjust the guard deliberately for a future dependency release:

```bat
set FWB_MAX_BUILD_MB=850
build_windows.bat
```

When the build succeeds, exercise the release-candidate paths in the frozen executable:

- **Project lifecycle:** create an empty project, open an existing `.magpy.json`, save/save-as,
  close/reopen it, and confirm the untouched startup tab is replaced rather than duplicated.
- **Measurements:** create/save/reopen a raw dataset; create or fit a Calibration / Composite fit,
  attach it to a coil, and confirm wheel scrolling does not change protected selectors.
- **Waveforms / exposures:** create and import a waveform (including a saved format profile or
  project-to-project waveform import), build a saved exposure, and verify the driven coils and
  electrical diagnostics.
- **Maps → Static:** render at least one 2D and one 3D map and create/reopen a Measurement-volume
  snapshot.
- **Maps → Scripted:** render a saved exposure in 2D, 3D, and Brain View. Confirm Scripted 2D/3D
  can show the optional Live measurement analysis panel when enabled before generation,
  exposes no Static Measurement snapshot action, Brain shows its Brain Areas panel, audio starts off,
  and a short video exports with the bundled encoder.
- **Exposure Analysis:** open **Project Manager → Maps / Analysis → Exposure Analysis** (or use **Analyze exposure…** as a shortcut from Exposure Editor), create a fresh analysis,
  and exercise at least one uncached derived calculation (or **Calculate all uncached analyses**).
  Confirm the analysis window is taskbar-visible, the single-worker queue never runs two derived jobs
  at once, and Pause/Run/Cancel behave normally. Close/reopen the resource from Saved Results and
  confirm cached results load without recalculation. Verify a physical dependency change marks the
  cache Stale while a presentation-only edit does not, and confirm an open analysis window does not
  prevent normal project/application shutdown.
- Save and reopen the project to confirm datasets, calibration links, waveforms, exposures, saved
  maps/snapshots, and Exposure Analysis caches persist. Open the updated Help Topics from the
  frozen application.

Before approving agent access, follow the **Windows executable and separate client**
setup in [Agent access](docs/AGENT_ACCESS.md) and the packaged acceptance checklist
in [TESTING.md](TESTING.md). The startup smoke test above does not exercise the
listener, client, MCP, numerical jobs, or new measurement playback features.

Then ZIP the complete `dist\FieldWorkbench` directory. Do not remove `_internal`,
`licenses`, Qt resources, or the three top-level notice files.

Publish that Windows ZIP together with the matching source ZIP at no charge.
Create the latter with `python tools\build_source_release.py`; the script omits
virtual environments, caches, tests' temporary output, and prior build products.

## Bundled FFmpeg video encoder

Windows releases bundle the tested static encoder at
`fieldworkbench/assets/ffmpeg/windows-x86_64/ffmpeg.exe`. The PyInstaller build
places it under `_internal/fieldworkbench/assets/ffmpeg/windows-x86_64/` and
`build_windows.bat` explicitly verifies/copies that exact runtime location after
freezing. Video export resolves the bundled copy automatically on Windows and does
not fall back to an arbitrary system FFmpeg. Linux packaging resolves the system `ffmpeg`
executable.

The exact FFmpeg/x264/zlib build provenance, configure flags, SHA-256, source
locations, and license texts are in `third_party/ffmpeg/`. `build_windows.bat`
copies those files into the release `licenses/FFmpeg-n8.1.2-x264/` directory.

