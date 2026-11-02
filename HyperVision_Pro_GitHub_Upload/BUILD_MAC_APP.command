#!/bin/bash
set -e
echo "============================================================"
echo "          HYPERVISION PRO - MAC OS APP BUILDER"
echo "============================================================"
cd "$(dirname "$0")"

# Detect best python
PY_CMD=""
if [ -x "/opt/anaconda3/bin/python" ]; then
    PY_CMD="/opt/anaconda3/bin/python"
elif command -v python3 &>/dev/null; then
    PY_CMD="python3"
elif command -v python &>/dev/null; then
    PY_CMD="python"
fi

if [ -z "$PY_CMD" ]; then
    echo "[ERROR] Python not found!"
    exit 1
fi

echo "Using Python: $PY_CMD"
"$PY_CMD" --version

echo "Step 1: Installing / verifying dependencies..."
"$PY_CMD" -m pip install -r requirements.txt 2>/dev/null || true
"$PY_CMD" -m pip install pyinstaller 2>/dev/null || true

echo "Step 2: Building Hypervision_Mac.app..."
"$PY_CMD" -m PyInstaller --noconfirm Hypervision_Mac.spec

echo "Step 3: Setting permissions, code signing, and distributing..."
if [ -d "dist/Hypervision_Mac.app" ]; then
    chmod -R 755 "dist/Hypervision_Mac.app"
    xattr -cr "dist/Hypervision_Mac.app"
    codesign --force --deep --sign - "dist/Hypervision_Mac.app"

    # Sync to root directory
    rm -rf "Hypervision_Mac.app"
    cp -pPR "dist/Hypervision_Mac.app" "Hypervision_Mac.app"

    # Sync to Hypervision for mac-dist if present
    if [ -d "Hypervision for mac-dist" ]; then
        rm -rf "Hypervision for mac-dist/Hypervision_Mac.app" "Hypervision for mac-dist/Hypervision_Mac"
        cp -pPR "dist/Hypervision_Mac.app" "Hypervision for mac-dist/Hypervision_Mac.app"
        cp -pPR "dist/Hypervision_Mac" "Hypervision for mac-dist/Hypervision_Mac"
    fi
fi

echo "============================================================"
echo "Build complete! Hypervision_Mac.app is ready in the root folder."
echo "============================================================"
