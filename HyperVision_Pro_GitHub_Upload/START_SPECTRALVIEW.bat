@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
python START_SPECTRALVIEW.py %*
if errorlevel 1 (
  echo.
  echo SpectralView did not start successfully.
  pause
)
endlocal
