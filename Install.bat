@echo off
setlocal enabledelayedexpansion

echo =============================================
echo   AI Image Tagger & Describer - Installer
echo =============================================

REM Check Python
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.9+ and make sure to check "Add to PATH" during installation.
    pause
    exit /b
)

REM Create virtual environment
echo [STEP] Creating virtual environment...
python -m venv .venv

REM Activate virtual environment
echo [STEP] Activating virtual environment...
call .venv\Scripts\activate

REM Upgrade pip
echo [STEP] Upgrading pip...
python -m pip install --upgrade pip

REM Install Python dependencies
echo [STEP] Installing required Python packages...
pip install customtkinter Pillow transformers torch iptcinfo3 python-xmp-toolkit

REM Download AI models from Hugging Face
echo [STEP] Downloading BLIP models from Hugging Face...
python -c "from transformers import BlipProcessor, BlipForConditionalGeneration; \
BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base'); \
BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base'); \
BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-large'); \
BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-large'); \
print('BLIP models downloaded.')"

REM Reminder for Exempi
echo.
echo ==========================================================
echo !!! IMPORTANT: Exempi system library is required for XMP !!!
echo ----------------------------------------------------------
echo Manual installation is required for XMP metadata support:
echo   Windows:
echo     - Download a prebuilt 'libexempi.dll' and add it to PATH
echo     - Or install via vcpkg or MSYS2 (advanced users)
echo   Linux:
echo     sudo apt install libexempi8
echo   macOS:
echo     brew install exempi
echo ==========================================================

REM Launch the app
echo.
echo [STEP] Starting the AI Image Tagger app...
python image_tagger.py

pause
endlocal
