@echo off
setlocal

echo ===========================================
echo Astraa Gamified Pose Tracker - Windows Installer
echo ===========================================

rem Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in your PATH.
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

echo.
echo [1/4] Creating virtual environment...
if not exist "venv_astraa" (
    python -m venv venv_astraa
    if %errorlevel% neq 0 (
        echo Failed to create virtual environment.
        pause
        exit /b 1
    )
) else (
    echo Virtual environment already exists.
)

echo.
echo [2/4] Activating virtual environment...
call venv_astraa\Scripts\activate

echo.
echo [3/4] Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo [4/4] Starting Astraa Tracker...
python main_with_classifier.py

echo.
pause
