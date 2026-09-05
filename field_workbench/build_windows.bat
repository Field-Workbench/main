@echo off
setlocal EnableExtensions

REM Field Workbench reproducible Windows onedir release build.
REM Run from the source directory after activating a clean Python 3.12 venv.

if not defined VIRTUAL_ENV (
  echo ERROR: Activate the release virtual environment first.
  echo        .venv\Scripts\activate
  exit /b 1
)

python tools\verify_release_environment.py
if errorlevel 1 exit /b %errorlevel%

python -m PyInstaller ^
  --noconfirm ^
  --clean ^
  --onedir ^
  --windowed ^
  --optimize=1 ^
  --collect-data=magpylib_studio ^
  --hidden-import=magpylib._src.display.backend_plotly ^
  --hidden-import=pygeomag.wmm.wmm_2025 ^
  --collect-data=plotly ^
  --collect-data=matplotlib ^
  --collect-submodules=plotly.graph_objs ^
  --copy-metadata=magpylib-studio ^
  --copy-metadata=magpylib ^
  --copy-metadata=plotly ^
  --copy-metadata=pygeomag ^
  --exclude-module=tkinter ^
  --exclude-module=pandas ^
  --exclude-module=IPython ^
  --exclude-module=jupyter ^
  --exclude-module=notebook ^
  --exclude-module=nbformat ^
  --exclude-module=dash ^
  --exclude-module=kaleido ^
  --exclude-module=pyvista ^
  --exclude-module=trimesh ^
  --exclude-module=xarray ^
  --exclude-module=polars ^
  --exclude-module=pyarrow ^
  --exclude-module=sympy ^
  --exclude-module=numba ^
  --exclude-module=skimage ^
  --exclude-module=pytest ^
  --exclude-module=PyQt5 ^
  --exclude-module=PyQt6 ^
  --exclude-module=PySide2 ^
  --splash "fieldworkbench\assets\splash.png" ^
  --add-data "fieldworkbench\default_scene.magpy.json;fieldworkbench" ^
  --add-data "fieldworkbench\assets\generic_bust_v1.stl;fieldworkbench\assets" ^
  --add-data "fieldworkbench\assets\generic_bust_v1.regions.json;fieldworkbench\assets" ^
  --add-data "fieldworkbench\assets\sri24_brain_gmwm_v1.stl;fieldworkbench\assets" ^
  --add-data "fieldworkbench\assets\sri24_labels;fieldworkbench\assets\sri24_labels" ^
  --add-data "fieldworkbench\assets\help;fieldworkbench\assets\help" ^
  --add-binary "fieldworkbench\assets\ffmpeg\windows-x86_64\ffmpeg.exe;fieldworkbench\assets\ffmpeg\windows-x86_64" ^
  --name FieldWorkbench ^
  app.py
if errorlevel 1 exit /b %errorlevel%

REM Keep the encoder at one deterministic runtime location. PyInstaller normally
REM places --add-binary content under _internal, but copy it explicitly as well so
REM a packaging-layout change cannot silently leave Windows video export disabled.
if not exist dist\FieldWorkbench\_internal\fieldworkbench\assets\ffmpeg\windows-x86_64 mkdir dist\FieldWorkbench\_internal\fieldworkbench\assets\ffmpeg\windows-x86_64
copy /Y fieldworkbench\assets\ffmpeg\windows-x86_64\ffmpeg.exe dist\FieldWorkbench\_internal\fieldworkbench\assets\ffmpeg\windows-x86_64\ffmpeg.exe >nul
if errorlevel 1 exit /b %errorlevel%
if not exist dist\FieldWorkbench\_internal\fieldworkbench\assets\ffmpeg\windows-x86_64\ffmpeg.exe (
  echo ERROR: Bundled FFmpeg was not copied into the packaged runtime.
  exit /b 1
)

copy /Y LICENSE.txt dist\FieldWorkbench\LICENSE.txt >nul
copy /Y PROJECT_NOTICE.txt dist\FieldWorkbench\PROJECT_NOTICE.txt >nul

python tools\collect_release_licenses.py ^
  --analysis build\FieldWorkbench\Analysis-00.toc ^
  --dist dist\FieldWorkbench ^
  --notice-template THIRD_PARTY_NOTICES.txt
if errorlevel 1 exit /b %errorlevel%

if not exist dist\FieldWorkbench\licenses\FFmpeg-n8.1.2-x264 mkdir dist\FieldWorkbench\licenses\FFmpeg-n8.1.2-x264
copy /Y third_party\ffmpeg\* dist\FieldWorkbench\licenses\FFmpeg-n8.1.2-x264\ >nul
if errorlevel 1 exit /b %errorlevel%

if not defined FWB_MAX_BUILD_MB set FWB_MAX_BUILD_MB=800
python tools\audit_windows_build.py ^
  --analysis build\FieldWorkbench\Analysis-00.toc ^
  --dist dist\FieldWorkbench ^
  --max-mb %FWB_MAX_BUILD_MB%
if errorlevel 1 exit /b %errorlevel%

echo.
echo Running the packaged startup smoke test...
start "" /wait dist\FieldWorkbench\FieldWorkbench.exe --smoke-test
if errorlevel 1 (
  echo ERROR: The packaged startup smoke test failed.
  exit /b %errorlevel%
)

echo.
echo Built and checked: dist\FieldWorkbench\FieldWorkbench.exe
echo Read:              dist\FieldWorkbench\BUILD_REPORT.txt
echo Zip the complete dist\FieldWorkbench folder for the binary release.
endlocal
