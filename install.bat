@echo off
setlocal
echo =============================================
echo AI Image Tagger & Describer - Setup Script
echo =============================================

REM Step 1: Check Python
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python is not installed or not added to PATH.
    echo Please install Python 3.9+ and make sure to check "Add to PATH" during installation.
    pause
    exit /b
)

REM Step 2: Create virtual environment
echo.
echo Creating virtual environment...
python -m venv .venv

REM Step 3: Activate virtual environment
echo.
echo Activating virtual environment...
call .venv\Scripts\activate

REM Step 4: Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM Step 5: Install requirements
echo.
echo Installing Python dependencies...
pip install -r requirements.txt

REM Step 6: Instructions for Exempi
echo.
echo --------------------------------------------------
echo Manual Step Required: Install Exempi for XMP support
echo --------------------------------------------------
echo 1. Download a Windows binary of Exempi (e.g. from vcpkg or MSYS2)
echo 2. Add the path to 'libexempi.dll' to your system PATH
echo 3. Restart your computer (if needed)
echo.
echo If you already installed Exempi, you're good to go!
echo --------------------------------------------------

REM Step 7: Launch the app
echo.
echo Launching the application...
python image_tagger.py

endlocal
pause
