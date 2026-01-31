#!/bin/bash

echo "==========================================="
echo "Astraa Gamified Pose Tracker - Mac/Linux Installer"
echo "==========================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 could not be found."
    echo "Please install Python 3 (brew install python3 or apt-get install python3)."
    exit 1
fi

echo ""
echo "[1/4] Creating virtual environment..."
if [ ! -d "venv_astraa" ]; then
    python3 -m venv venv_astraa
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment."
        exit 1
    fi
else
    echo "Virtual environment already exists."
fi

echo ""
echo "[2/4] Activating virtual environment..."
source venv_astraa/bin/activate

echo ""
echo "[3/4] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Failed to install dependencies."
    exit 1
fi

echo ""
echo "[4/4] Starting Astraa Tracker..."
python main_with_classifier.py
