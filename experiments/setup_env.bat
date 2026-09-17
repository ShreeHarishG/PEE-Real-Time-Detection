@echo off
echo ===================================================
echo Setting up Python Virtual Environment for EdgeVision
echo ===================================================

echo.
echo 1. Creating virtual environment (.venv)...
python -m venv .venv
if errorlevel 1 (
    echo Failed to create virtual environment! Make sure Python is installed and in your PATH.
    pause
    exit /b 1
)

echo.
echo 2. Activating virtual environment...
call .venv\Scripts\activate.bat

echo.
echo 3. Upgrading pip...
python -m pip install --upgrade pip

echo.
echo 4. Installing Windows specific requirements (PyTorch etc)...
pip install -r requirements-windows.txt

echo.
echo 5. Installing common requirements (Ultralytics, OpenCV, FastAPI etc)...
pip install -r requirements-common.txt

echo.
echo ===================================================
echo Setup Complete! 
echo ===================================================
echo To train the model, you can now run:
echo call .venv\Scripts\activate.bat
echo python train_v4_boots.py
echo ===================================================
pause
