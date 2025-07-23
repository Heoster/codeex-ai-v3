#!/bin/bash

# CodeEx AI Startup Script

echo "🚀 Starting CodeEx AI..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/update dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Check environment variables
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found. Please create one with required variables."
    echo "See .env.example for reference."
fi

# Initialize database
echo "🗄️ Initializing database..."
python -c "from app import init_db; init_db()"

# Start the application
echo "✅ Starting CodeEx AI application..."
if [ "$1" = "production" ]; then
    echo "🏭 Starting in production mode with Gunicorn..."
    gunicorn --bind 0.0.0.0:5000 --workers 4 --timeout 120 app:app
else
    echo "🔧 Starting in development mode..."
    python app.py
fi