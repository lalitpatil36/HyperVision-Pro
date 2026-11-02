@echo off
setlocal
cd /d "%~dp0"
set PYTHONPATH=%CD%
python -m pytest -q tests
if errorlevel 1 (
  echo Core validation failed. Do not treat this build as release-ready.
  pause
  exit /b 1
)
echo Core validation passed.
pause
endlocal
