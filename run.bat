@echo off
echo ============================================
echo   Fake News Detection System Launcher
echo ============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from python.org
    pause
    exit /b 1
)

echo [INFO] Python found!
python --version
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo [INFO] Virtual environment not found. Creating...
    python -m venv venv
    echo [SUCCESS] Virtual environment created!
    echo.
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)

echo [SUCCESS] Virtual environment activated!
echo.

REM Check if requirements are installed
pip show streamlit >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing requirements...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install requirements
        pause
        exit /b 1
    )
    echo [SUCCESS] Requirements installed!
    echo.
)

REM Show menu
:menu
echo ============================================
echo   What would you like to do?
echo ============================================
echo.
echo   1. Train Models (First time only)
echo   2. Run Web Application
echo   3. Install/Update Requirements
echo   4. Exit
echo.
set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" goto train
if "%choice%"=="2" goto run
if "%choice%"=="3" goto install
if "%choice%"=="4" goto end
echo [ERROR] Invalid choice. Please try again.
echo.
goto menu

:train
echo.
echo [INFO] Training models...
echo This may take 2-5 minutes...
echo.
python train.py
if errorlevel 1 (
    echo.
    echo [ERROR] Training failed!
    pause
    goto menu
)
echo.
echo [SUCCESS] Training completed!
echo.
pause
goto menu

:run
echo.
echo [INFO] Starting web application...
echo The app will open in your browser at http://localhost:8501
echo.
echo Press Ctrl+C to stop the application
echo.
streamlit run app.py
goto menu

:install
echo.
echo [INFO] Installing/Updating requirements...
pip install --upgrade -r requirements.txt
echo [SUCCESS] Requirements updated!
echo.
pause
goto menu

:end
echo.
echo [INFO] Deactivating virtual environment...
call venv\Scripts\deactivate.bat
echo [SUCCESS] Goodbye!
echo.
pause
exit
