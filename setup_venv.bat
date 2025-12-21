@echo off
REM Setup script for pedestrian-conflict-prediction project (Windows)
REM Creates a Python 3.9 virtual environment and installs dependencies

echo Setting up Python virtual environment...

REM Check if Python 3.9 is available
python --version | findstr "3.9" >nul
if errorlevel 1 (
    echo Error: Python 3.9 is not found. Please install Python 3.9 first.
    exit /b 1
)

REM Create virtual environment
echo Creating virtual environment with Python 3.9...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip setuptools wheel

REM Install PyTorch (MPS support for macOS)
echo Installing PyTorch with MPS support...
pip install torch torchvision torchaudio

REM Install other requirements
echo Installing other dependencies...
pip install -r requirements.txt

REM Verify MPS availability
echo.
echo Verifying MPS support...
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}'); print(f'MPS built: {torch.backends.mps.is_built()}')"

echo.
echo Setup complete! To activate the environment, run:
echo   venv\Scripts\activate
echo.
echo To deactivate, run:
echo   deactivate

