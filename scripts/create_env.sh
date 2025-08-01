#!/bin/bash

# This script demonstrates how to create a Python environment using conda.
# Conda is a package and environment management system that runs on Windows, macOS, and Linux.

# Steps to create a new Python environment using conda:

# Install conda: Download and install Anaconda or Miniconda from https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html

conda deactivate

conda remove -n audioset_strong --all -y
conda create -n audioset_strong python=3.13 -y
conda activate audioset_strong

# pip install --no-cache-dir -r requirements.txt
# pip install -r requirements.txt
