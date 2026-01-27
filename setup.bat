@echo off
:: ============================================================================
:: Islamic AI Assistant - Complete Auto-Setup Script
:: This script creates venv, installs everything, and sets up the project
:: ============================================================================

setlocal enabledelayedexpansion
color 0A
title Islamic AI Assistant Setup

echo.
echo ========================================================================
echo                   ISLAMIC AI ASSISTANT - AUTO SETUP
echo ========================================================================
echo.
echo This script will:
echo   1. Check system requirements
echo   2. Create virtual environment (venv)
echo   3. Install all dependencies
echo   4. Download AI models
echo   5. Configure the application
echo.
echo ========================================================================
echo.
pause

:: ============================================================================
:: STEP 1: Check Python Installation
:: ============================================================================
echo.
echo [STEP 1/7] Checking Python installation...
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH!
    echo.
    echo Please install Python 3.8 or higher from:
    echo https://www.python.org/downloads/
    echo.
    echo IMPORTANT: Check "Add Python to PATH" during installation!
    echo.
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [SUCCESS] Python %PYTHON_VERSION% found
echo.

:: ============================================================================
:: STEP 2: Check pip
:: ============================================================================
echo [STEP 2/7] Checking pip...
echo.

python -m pip --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] pip is not available!
    echo Installing pip...
    python -m ensurepip --upgrade
    if errorlevel 1 (
        echo [ERROR] Failed to install pip
        pause
        exit /b 1
    )
)

echo [SUCCESS] pip is available
echo.

:: ============================================================================
:: STEP 3: Create Virtual Environment
:: ============================================================================
echo [STEP 3/7] Creating virtual environment (venv)...
echo.

if exist "venv\" (
    echo [WARNING] Virtual environment already exists!
    choice /C YN /M "Delete existing venv and create new one?"
    if errorlevel 2 (
        echo [INFO] Using existing virtual environment
        goto :activate_venv
    )
    echo [INFO] Removing old virtual environment...
    rmdir /s /q venv
)

echo [INFO] Creating new virtual environment...
python -m venv venv
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment!
    pause
    exit /b 1
)

echo [SUCCESS] Virtual environment created
echo.

:: ============================================================================
:: STEP 4: Activate Virtual Environment
:: ============================================================================
:activate_venv
echo [STEP 4/7] Activating virtual environment...
echo.

if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment activation script not found!
    pause
    exit /b 1
)

call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment!
    pause
    exit /b 1
)

echo [SUCCESS] Virtual environment activated
echo [INFO] Python location: %VIRTUAL_ENV%
echo.

:: ============================================================================
:: STEP 5: Upgrade pip in venv
:: ============================================================================
echo [STEP 5/7] Upgrading pip in virtual environment...
echo.

python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo [WARNING] Failed to upgrade pip, continuing anyway...
) else (
    echo [SUCCESS] pip upgraded successfully
)
echo.

:: ============================================================================
:: STEP 6: Install Dependencies
:: ============================================================================
echo [STEP 6/7] Installing all dependencies...
echo.
echo This will take 2-5 minutes depending on your internet speed.
echo Please wait...
echo.

:: Check if requirements.txt exists
if not exist "requirements.txt" (
    echo [WARNING] requirements.txt not found!
    echo [INFO] Creating requirements.txt...
    (
        echo # Core FastAPI Framework
        echo fastapi==0.104.1
        echo uvicorn[standard]==0.24.0
        echo python-dotenv==1.0.0
        echo pydantic==2.5.0
        echo python-multipart==0.0.6
        echo.
        echo # Google Gemini AI
        echo google-genai==0.2.0
        echo.
        echo # Web Scraping
        echo requests==2.31.0
        echo beautifulsoup4==4.12.2
        echo lxml==4.9.3
        echo.
        echo # Vector Database
        echo sentence-transformers==2.2.2
        echo faiss-cpu==1.7.4
        echo.
        echo # Audio Processing
        echo SpeechRecognition==3.10.0
        echo pydub==0.25.1
        echo.
        echo # Data Processing
        echo numpy==1.24.3
    ) > requirements.txt
    echo [SUCCESS] requirements.txt created
    echo.
)

echo [INFO] Installing packages...
echo -----------------------------------------------------------------------
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to install some packages!
    echo.
    echo Common solutions:
    echo   1. Check your internet connection
    echo   2. Try running as Administrator
    echo   3. Update Visual C++ redistributables
    echo   4. Check pip and Python versions
    echo.
    pause
    exit /b 1
)

echo.
echo [SUCCESS] All dependencies installed successfully!
echo.

:: ============================================================================
:: STEP 7: Download Embedding Model (First Time)
:: ============================================================================
echo [STEP 7/7] Downloading AI embedding model...
echo.
echo This is a one-time download (~90MB)
echo Please wait...
echo.

python -c "from sentence_transformers import SentenceTransformer; print('Downloading...'); model = SentenceTransformer('all-MiniLM-L6-v2'); print('Model downloaded successfully!')" 2>nul
if errorlevel 1 (
    echo [WARNING] Model download may occur on first run
) else (
    echo [SUCCESS] Embedding model ready
)
echo.

:: ============================================================================
:: CONFIGURATION SETUP
:: ============================================================================
echo ========================================================================
echo                        CONFIGURATION SETUP
echo ========================================================================
echo.

:: Create .env file if it doesn't exist
if not exist ".env" (
    echo [INFO] Creating .env configuration file...
    (
        echo # Islamic AI Assistant Configuration
        echo.
        echo # REQUIRED: Gemini API Key
        echo # Get your key from: https://aistudio.google.com/apikey
        echo GEMINI_API_KEY=your_api_key_here
        echo.
        echo # Model Configuration
        echo MODEL_NAME=gemini-2.0-flash-exp
        echo VISION_MODEL=gemini-2.0-flash-exp
        echo MAX_TOKENS=2048
        echo TEMPERATURE=0.7
        echo.
        echo # Vector Database Settings
        echo EMBEDDING_MODEL=all-MiniLM-L6-v2
        echo VECTOR_DB_PATH=vector_db
        echo TOP_K_RESULTS=5
        echo CHUNK_SIZE=500
        echo CHUNK_OVERLAP=100
        echo.
        echo # Chat Configuration
        echo CHAT_MEMORY_LIMIT=20
    ) > .env
    echo [SUCCESS] .env file created
    echo.
) else (
    echo [INFO] .env file already exists
    echo.
)

:: Check if API key is configured
findstr /C:"your_api_key_here" .env >nul 2>&1
if errorlevel 0 (
    echo [WARNING] Gemini API key not configured!
    echo.
    echo -----------------------------------------------------------------------
    echo  IMPORTANT: You need to add your Gemini API key to continue!
    echo -----------------------------------------------------------------------
    echo.
    echo  1. Visit: https://aistudio.google.com/apikey
    echo  2. Sign in with your Google account
    echo  3. Click "Create API Key"
    echo  4. Copy the API key
    echo  5. Open .env file in this folder
    echo  6. Replace "your_api_key_here" with your actual key
    echo.
    echo  Example:
    echo  GEMINI_API_KEY=AIzaSyC8xYzAbc123_your_actual_key_here
    echo.
    echo -----------------------------------------------------------------------
    echo.
    
    choice /C YN /M "Do you want to open .env file now to add your API key?"
    if not errorlevel 2 (
        start notepad .env
        echo.
        echo [INFO] Please save the file after adding your API key
        echo.
        pause
    )
)

:: Create necessary directories
echo [INFO] Creating project directories...
if not exist "uploads" mkdir uploads
if not exist "vector_db" mkdir vector_db
echo [SUCCESS] Directories created
echo.

:: Check for FFmpeg (optional for audio)
echo [INFO] Checking optional dependencies...
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo [OPTIONAL] FFmpeg not found - Audio processing will be limited
    echo.
    echo To enable full audio support:
    echo   1. Download FFmpeg from: https://ffmpeg.org/download.html
    echo   2. Extract to C:\ffmpeg
    echo   3. Add C:\ffmpeg\bin to System PATH
    echo.
) else (
    echo [SUCCESS] FFmpeg is available - Full audio support enabled
    echo.
)

:: ============================================================================
:: INSTALLATION COMPLETE
:: ============================================================================
echo.
echo ========================================================================
echo                      INSTALLATION COMPLETE!
echo ========================================================================
echo.
echo [SUCCESS] Virtual environment: venv\
echo [SUCCESS] Dependencies: Installed
echo [SUCCESS] Configuration: Ready
echo.
echo ========================================================================
echo                         NEXT STEPS
echo ========================================================================
echo.
echo 1. Configure API Key (if not done):
echo    - Open .env file
echo    - Add your Gemini API key
echo    - Get key from: https://aistudio.google.com/apikey
echo.
echo 2. Run the application:
echo    - Double-click: run.bat
echo    - Or in CMD: call venv\Scripts\activate ^&^& python main.py
echo.
echo 3. Access the application:
echo    - Web Interface: http://127.0.0.1:8000
echo    - API Documentation: http://127.0.0.1:8000/docs
echo.
echo ========================================================================
echo                         QUICK COMMANDS
echo ========================================================================
echo.
echo Start server:        run.bat
echo Activate venv:       venv\Scripts\activate
echo Check health:        curl http://127.0.0.1:8000/health
echo View docs:           start http://127.0.0.1:8000/docs
echo.
echo ========================================================================
echo.

:: Create run.bat for easy server startup
echo [INFO] Creating run.bat launcher...
(
    echo @echo off
    echo :: Quick launcher for Islamic AI Assistant
    echo.
    echo echo Starting Islamic AI Assistant...
    echo echo.
    echo.
    echo :: Activate virtual environment
    echo call venv\Scripts\activate
    echo.
    echo :: Run the server
    echo python main.py
    echo.
    echo pause
) > run.bat
echo [SUCCESS] run.bat created
echo.

:: Ask if user wants to start the server now
choice /C YN /M "Would you like to start the server now?"
if errorlevel 2 (
    echo.
    echo You can start the server anytime by running: run.bat
    echo.
    pause
    exit /b 0
)

:: Check if main.py exists
if not exist "main.py" (
    echo.
    echo [ERROR] main.py not found!
    echo Please make sure main.py is in the same folder as this setup script.
    echo.
    pause
    exit /b 1
)

:: Start the server
echo.
echo ========================================================================
echo                      STARTING SERVER
echo ========================================================================
echo.
echo Server will start at: http://127.0.0.1:8000
echo API Documentation: http://127.0.0.1:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo.
echo ========================================================================
echo.

python main.py

pause