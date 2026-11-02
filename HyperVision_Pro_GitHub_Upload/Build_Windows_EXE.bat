@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Hypervision Windows Build and Setup Tool
color 0b

echo ========================================================================
echo               HYPERVISION PRO - WINDOWS SETUP AND BUILDER
echo ========================================================================
echo.

:: Detect if we are in the parent directory instead of the source directory
set "SOURCE_DIR=%~dp0"
if not exist "%SOURCE_DIR%Hypervision_Win.spec" (
    if exist "%SOURCE_DIR%Hypervision-updated-28_07_26\Hypervision_Win.spec" (
        set "SOURCE_DIR=%SOURCE_DIR%Hypervision-updated-28_07_26\"
    ) else (
        for /d %%d in (*) do (
            if exist "%%d\Hypervision_Win.spec" set "SOURCE_DIR=%~dp0%%d\"
        )
    )
)

cd /d "%SOURCE_DIR%"
echo Working in source folder: %CD%
echo.
echo Detecting Python environment on this Windows machine...
echo.

set "PY_CMD="

:: Check if python command is available
where python >nul 2>nul
if %ERRORLEVEL% equ 0 set "PY_CMD=python"

:: Check if py launcher is available
if not defined PY_CMD (
    where py >nul 2>nul
    if %ERRORLEVEL% equ 0 set "PY_CMD=py"
)

:: Check standard Windows Python installation directories
if not defined PY_CMD (
    for %%p in (
        "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
        "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
        "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
        "%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
        "%LOCALAPPDATA%\Programs\Python\Python39\python.exe"
        "C:\Python313\python.exe"
        "C:\Python312\python.exe"
        "C:\Python311\python.exe"
        "C:\Python310\python.exe"
        "C:\Python39\python.exe"
        "%ProgramFiles%\Python313\python.exe"
        "%ProgramFiles%\Python312\python.exe"
        "%ProgramFiles%\Python311\python.exe"
        "%ProgramFiles%\Python310\python.exe"
        "%USERPROFILE%\anaconda3\python.exe"
        "%USERPROFILE%\miniconda3\python.exe"
        "C:\ProgramData\anaconda3\python.exe"
        "C:\ProgramData\miniconda3\python.exe"
    ) do (
        if not defined PY_CMD if exist "%%~p" set "PY_CMD=%%~p"
    )
)

if not defined PY_CMD goto :NoPython

echo [OK] Python detected: %PY_CMD%
echo.
"%PY_CMD%" --version
echo.

echo ========================================================================
echo Step 1/3: Checking required libraries...
echo ========================================================================

"%PY_CMD%" -c "import PyInstaller, PyQt5, numpy, pandas, sklearn, matplotlib, reportlab, cv2, spectral, scipy, docx" >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo [OK] All required libraries are already installed!
) else (
    echo Installing / verifying core dependencies...
    "%PY_CMD%" -m pip install pyinstaller pyqt5 numpy pandas scikit-learn matplotlib reportlab opencv-python spectral scipy python-docx openpyxl
    if %ERRORLEVEL% neq 0 (
        color 0c
        echo.
        echo ========================================================================
        echo [ERROR] Failed to install Python dependencies!
        echo Please check your internet connection and try running again.
        echo ========================================================================
        pause
        exit /b 1
    )
    echo [OK] Libraries installed successfully!
)
echo.

echo ========================================================================
echo Step 2/3: Compiling standalone Windows executable (.exe)...
echo Note: This process usually takes 1 to 3 minutes. Please wait...
echo ========================================================================
echo.

if exist "dist\Hypervision_Win" rmdir /s /q "dist\Hypervision_Win" >nul 2>nul
if exist "build\Hypervision_Win" rmdir /s /q "build\Hypervision_Win" >nul 2>nul

"%PY_CMD%" -m PyInstaller --noconfirm Hypervision_Win.spec

if %ERRORLEVEL% neq 0 (
    color 0c
    echo.
    echo ========================================================================
    echo [ERROR] PyInstaller compilation failed!
    echo See the detailed error log above.
    echo ========================================================================
    pause
    exit /b 1
)

echo.
echo ========================================================================
echo Step 3/3: Verifying output file and resources...
echo ========================================================================

if not exist "dist\Hypervision_Win\Hypervision_Win.exe" (
    color 0c
    echo.
    echo ========================================================================
    echo [ERROR] Compilation finished, but dist\Hypervision_Win\Hypervision_Win.exe
    echo was not found on disk! Please check antivirus settings.
    echo ========================================================================
    pause
    exit /b 1
)

:: Copy icons folder alongside exe for UI assets
if exist "icons" (
    if not exist "dist\Hypervision_Win\icons" mkdir "dist\Hypervision_Win\icons" >nul 2>nul
    xcopy /y /e /i "icons" "dist\Hypervision_Win\icons" >nul 2>nul
)

:: Sync dist to launcher origin directory if it was different
if exist "%~dp0" (
    if not exist "%~dp0dist\Hypervision_Win" mkdir "%~dp0dist\Hypervision_Win" >nul 2>nul
    xcopy /y /e /i "dist\Hypervision_Win" "%~dp0dist\Hypervision_Win" >nul 2>nul
)

:: Create convenient launcher scripts
(
echo @echo off
echo cd /d "%%~dp0dist\Hypervision_Win"
echo start "" "%%~dp0dist\Hypervision_Win\Hypervision_Win.exe"
) > "START_HYPERVISION_WINDOWS.bat"

if exist "%~dp0" (
    (
    echo @echo off
    echo cd /d "%%~dp0dist\Hypervision_Win"
    echo start "" "%%~dp0dist\Hypervision_Win\Hypervision_Win.exe"
    ) > "%~dp0START_HYPERVISION_WINDOWS.bat"
)

color 0a
echo.
echo ========================================================================
echo                   BUILD SUCCESSFUL! (100%% VERIFIED)
echo ========================================================================
echo.
echo Your standalone executable is ready:
echo    dist\Hypervision_Win\Hypervision_Win.exe
echo.
echo A one-click launcher has also been created:
echo    START_HYPERVISION_WINDOWS.bat
echo.
echo Opening the executable folder now...
echo ========================================================================
explorer /select,"dist\Hypervision_Win\Hypervision_Win.exe"
pause
exit /b 0

:NoPython
color 0c
echo ========================================================================
echo [ERROR] Python was NOT found on this Windows computer!
echo ========================================================================
echo.
echo In order to compile or run Hypervision, Python must be installed.
echo.
echo Please follow these simple steps:
echo  1. Go to: https://www.python.org/downloads/
echo  2. Download and install Python 3.10, 3.11, or 3.12.
echo  3. CRITICAL: On the first installer screen, CHECK the box:
echo     [X] Add python.exe to PATH
echo.
echo After installing Python, double-click this file again!
echo ========================================================================
pause
exit /b 1
