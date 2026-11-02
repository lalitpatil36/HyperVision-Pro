@echo off
setlocal EnableDelayedExpansion
title Hypervision Pro Launcher
color 0a

echo ========================================================================
echo              HYPERVISION PRO - FAST DIRECT LAUNCHER
echo ========================================================================
echo.
cd /d "%~dp0"

:: 1. Search for Python
set "PY_CMD="

where python >nul 2>nul
if %ERRORLEVEL% equ 0 set "PY_CMD=python"

if not defined PY_CMD (
    where py >nul 2>nul
    if %ERRORLEVEL% equ 0 set "PY_CMD=py"
)

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
        if exist %%p if not defined PY_CMD set "PY_CMD=%%~p"
    )
)

if not defined PY_CMD (
    color 0c
    echo [ERROR] Python is not installed on this computer!
    echo Please install Python 3.10+ from https://www.python.org/downloads/
    echo Make sure to check "Add python.exe to PATH" during install.
    pause
    exit /b 1
)

echo Starting Hypervision Pro now...
echo.
"%PY_CMD%" spectral_gui_2026_07_16.py
if %ERRORLEVEL% neq 0 (
    echo.
    echo If it failed because of missing libraries, run Build_Windows_EXE.bat once to install them.
    pause
)
