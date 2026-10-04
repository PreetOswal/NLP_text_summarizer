@echo off
title NLP Summarization Project

echo ========================================
echo   NLP Summarization Project
echo ========================================
echo.

REM Check whether Python is installed
python --version >nul 2>&1

if errorlevel 1 (
    echo ERROR: Python is not installed.
    echo Please install Python first.
    pause
    exit /b
)

REM Create virtual environment if it does not exist
if not exist "venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv venv

    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b
    )
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing/checking dependencies...
python -m pip install -r requirements.txt

if errorlevel 1 (
    echo ERROR: Failed to install dependencies.
    pause
    exit /b
)

REM Run project
echo.
echo Starting project...
echo.

python summarization_system.py --num-samples 2 --skip-long-demo

echo.
echo ========================================
echo   Project finished.
echo ========================================

pause