@echo off
echo 🚀 Starting CodeEx AI...

REM Check if virtual environment exists
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install/update dependencies
echo 📦 Installing dependencies...
pip install -r requirements.txt

REM Check environment variables
if not exist ".env" (
    echo ⚠️  Warning: .env file not found. Please create one with required variables.
    echo See .env.example for reference.
)

REM Initialize database
echo 🗄️ Initializing database...
python -c "from app import init_db; init_db()"

REM Start the application
echo ✅ Starting CodeEx AI application...
if "%1"=="production" (
    echo 🏭 Starting in production mode with Gunicorn...
    gunicorn --bind 0.0.0.0:5000 --workers 4 --timeout 120 app:app
) else (
    echo 🔧 Starting in development mode...
    python app.py
)

pause