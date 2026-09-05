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
The batch file also explicitly packages the default scene, built-in anatomy,
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
The final size varies with Python and Qt patch releases, but a broad accidental
700 MB collection will now fail the release audit instead of being silently
accepted. To adjust the guard deliberately for a future dependency release:

```bat
set FWB_MAX_BUILD_MB=850
build_windows.bat
```

When the build succeeds, test the main scene plus one 2D and one 3D field map,
then ZIP the complete `dist\FieldWorkbench` directory. Do not remove `_internal`,
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

