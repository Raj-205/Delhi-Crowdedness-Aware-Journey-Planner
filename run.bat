@echo off
setlocal
title Delhi Bus Journey - Crowd Planner

cd /d "%~dp0"

echo ========================================================================
echo         DELHI CROWDEDNESS-AWARE BUS JOURNEY PLANNER
echo           Problem 2 - Transport Stack Hackathon 2026
echo ========================================================================
echo.

set "PYTHON_EXE="
if exist "%~dp0.venv\Scripts\python.exe" set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"

if "%PYTHON_EXE%"=="" (
    where python >nul 2>nul
    if not errorlevel 1 set "PYTHON_EXE=python"
)

if "%PYTHON_EXE%"=="" (
    echo [ERROR] Python not found! Please ensure Python or .venv is installed.
    pause
    exit /b 1
)

echo [OK] Using Python: %PYTHON_EXE%
echo.

echo Running pre-flight self-check...
"%PYTHON_EXE%" scripts\doctor.py
echo.

echo ========================================================================
echo  Server is starting!
echo    - Web Application : http://127.0.0.1:8000
echo    - Interactive Docs: http://127.0.0.1:8000/docs
echo    - Health Check    : http://127.0.0.1:8000/health
echo ========================================================================
echo.
echo Launching your default web browser...
start "" "http://127.0.0.1:8000"

echo Starting Uvicorn web server - Press Ctrl+C to stop
echo.
"%PYTHON_EXE%" -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

pause
