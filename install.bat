@echo off
REM CodeEx AI - Windows Installation Script
echo 🚀 CodeEx AI - Windows Installation
echo =====================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python from https://python.org
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo ✅ Python found
python --version

REM Run the installation script
echo.
echo 📦 Running installation script...
python install.py

REM Pause to see results
echo.
echo Press any key to continue...
pause >nul