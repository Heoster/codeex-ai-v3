@echo off
echo 🚀 CodeEx AI Server Launcher
echo ========================================

REM Check if virtual environment exists
if not exist ".venv" (
    echo ❌ Virtual environment not found!
    echo 💡 Create one with: python -m venv .venv
    pause
    exit /b 1
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call .venv\Scripts\activate.bat

REM Check if requirements are installed
echo 📦 Checking dependencies...
python -c "import flask, requests" 2>nul
if errorlevel 1 (
    echo ❌ Dependencies missing! Installing...
    pip install -r requirements.txt
)

REM Start the server
echo 🌟 Starting CodeEx AI server...
python app.py

pause