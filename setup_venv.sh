#!/bin/bash

# Setup script for pedestrian-conflict-prediction project
# Creates a Python 3.9 virtual environment and installs dependencies

echo "Setting up Python virtual environment..."

# Check if Python 3.9 is available
if ! command -v python3.9 &> /dev/null; then
    echo "Error: Python 3.9 is not found. Please install Python 3.9 first."
    echo "You can check available Python versions with: python3 --version"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment with Python 3.9..."
python3.9 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch (MPS support for macOS)
echo "Installing PyTorch with MPS support for macOS..."
pip install torch torchvision torchaudio

# Install other requirements
echo "Installing other dependencies..."
pip install -r requirements.txt

# Verify MPS availability
echo ""
echo "Verifying MPS support..."
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}'); print(f'MPS built: {torch.backends.mps.is_built()}')" || echo "MPS check failed (this is OK if not on Apple Silicon)"

echo ""
echo "Setup complete! To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To deactivate, run:"
echo "  deactivate"

