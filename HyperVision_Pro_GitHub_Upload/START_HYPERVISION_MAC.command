#!/bin/bash
echo "============================================================"
echo "          HYPERVISION PRO - MAC OS RUNNER"
echo "============================================================"
cd "$(dirname "$0")"

# Check for standalone Mac App in various possible locations
APP_PATH=""
if [ -d "Hypervision_Mac.app" ]; then
    APP_PATH="Hypervision_Mac.app"
elif [ -d "dist/Hypervision_Mac.app" ]; then
    APP_PATH="dist/Hypervision_Mac.app"
elif [ -d "Hypervision for mac-dist/Hypervision_Mac.app" ]; then
    APP_PATH="Hypervision for mac-dist/Hypervision_Mac.app"
fi

if [ -n "$APP_PATH" ]; then
    echo "Found $APP_PATH. Verifying permissions and launching..."
    chmod -R 755 "$APP_PATH" 2>/dev/null || true
    xattr -cr "$APP_PATH" 2>/dev/null || true
    open "$APP_PATH"
    echo "Hypervision Pro launched successfully!"
    exit 0
fi

# Fallback: run via python
echo "Running via Python..."
PY_RUN="python3"
if [ -x "/opt/anaconda3/bin/python" ]; then
    PY_RUN="/opt/anaconda3/bin/python"
fi
"$PY_RUN" spectral_gui_2026_07_16.py
