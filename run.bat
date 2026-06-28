@echo off
title Fake News - Run App
cd /d "%~dp0"

REM Check for virtual environment
if not exist "venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv venv 2>nul || py -m venv venv 2>nul
    if errorlevel 1 (
        echo Could not create venv. Make sure Python is installed and on PATH.
        pause
        exit /b 1
    )
)

call venv\Scripts\activate.bat

REM Install or update dependencies
echo Checking dependencies...
pip install -r requirements.txt -q
if errorlevel 1 (
    echo Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo Starting Fake News app...
echo.
python app.py

pause
