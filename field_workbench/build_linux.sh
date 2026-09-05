#!/usr/bin/env bash
set -euo pipefail

python -m PyInstaller \
  --noconfirm \
  --clean \
  --onedir \
  --console \
  --collect-data magpylib_studio \
  --add-data fieldworkbench/default_scene.magpy.json:fieldworkbench \
  --add-data fieldworkbench/assets/generic_bust_v1.stl:fieldworkbench/assets \
  --add-data fieldworkbench/assets/generic_bust_v1.regions.json:fieldworkbench/assets \
  --add-data fieldworkbench/assets/sri24_brain_gmwm_v1.stl:fieldworkbench/assets \
  --add-data fieldworkbench/assets/sri24_labels:fieldworkbench/assets/sri24_labels \
  --add-data fieldworkbench/assets/help:fieldworkbench/assets/help \
  --name FieldWorkbench \
  app.py

echo "Built: dist/FieldWorkbench/FieldWorkbench"
