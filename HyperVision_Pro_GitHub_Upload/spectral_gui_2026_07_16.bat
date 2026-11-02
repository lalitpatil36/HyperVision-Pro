@echo off
echo =========================================
echo Launching SpectralView (2026-07-16 version) for Windows
echo =========================================

REM Change directory to the folder where this batch file is located
cd /d "%~dp0"

python spectral_gui_2026_07_16.py

pause
