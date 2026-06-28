#!/bin/bash
cd "$(dirname "$0")"

# Create venv if it doesn't exist
if [ ! -f venv/bin/python ] && [ ! -f venv/bin/python3 ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv || python -m venv venv
    if [ $? -ne 0 ]; then
        echo "Could not create venv. Make sure Python 3 is installed."
        read -p "Press Enter to close..."
        exit 1
    fi
fi

# Activate venv (support both python and python3 in venv)
source venv/bin/activate

# Install/verify dependencies
echo "Checking dependencies..."
pip install -r requirements.txt -q
if [ $? -ne 0 ]; then
    echo "Failed to install dependencies."
    read -p "Press Enter to close..."
    exit 1
fi

echo ""
echo "Starting Fake News app..."
echo ""
python app.py

read -p "Press Enter to close..."
