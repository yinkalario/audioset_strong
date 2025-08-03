#!/bin/bash

# AudioSet Strong Labeling Environment Setup Script
# Creates a conda environment and installs required packages

set -e  # Exit on any error

ENV_NAME="audioset_strong"
PYTHON_VERSION="3.13"

echo "=== AudioSet Strong Environment Setup ==="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: Conda not found. Please install Anaconda or Miniconda."
    exit 1
fi

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "Error: requirements.txt not found. Run from audioset_strong root directory."
    exit 1
fi

# Remove existing environment if it exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Removing existing ${ENV_NAME} environment..."
    conda remove -n ${ENV_NAME} --all -y
fi

# Create conda environment
echo "Creating conda environment: ${ENV_NAME}"
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y

# Install packages
echo "Installing required packages..."
conda run -n ${ENV_NAME} pip install --upgrade pip
conda run -n ${ENV_NAME} pip install -r requirements.txt

echo "✓ Environment setup complete!"
echo ""
echo "To activate: conda activate ${ENV_NAME}"
echo "To deactivate: conda deactivate"
