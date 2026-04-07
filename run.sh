#!/bin/bash

# Exit on error
set -e

# Config
VENV_DIR="venv"
REQUIREMENTS_FILE="requirements.txt"
PYTHON_SCRIPT="main.py"

echo "Starting setup..."

# Check Python
if ! command -v python3 &> /dev/null
then
    echo "Python3 not found. Please install Python."
    exit 1
fi

# Create virtual environment if not exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv $VENV_DIR
fi

# Activate venv
echo "Activating virtual environment..."
source $VENV_DIR/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "Installing dependencies..."
    pip install -r $REQUIREMENTS_FILE
else
    echo "No requirements.txt found, skipping..."
fi

# Run script
if [ -f "$PYTHON_SCRIPT" ]; then
    echo "Running Python script..."
    python $PYTHON_SCRIPT
else
    echo "Python script not found!"
    exit 1
fi
