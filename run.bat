@echo off
title Dr.Assistant - Medical Imaging & Decision Support Platform
echo =======================================================================
echo   Starting Dr.Assistant - Medical Imaging & Clinical Support Platform
echo =======================================================================
echo.

cd /d "%~dp0"

set "PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

echo [1/2] Checking Python environment...
"%PYTHON%" --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH! Please install Python 3.11+.
    pause
    exit /b 1
)

echo [2/2] Launching application on http://127.0.0.1:8000 ...
"%PYTHON%" run.py

pause
