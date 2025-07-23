#!/usr/bin/env python3
"""
CodeEx AI Server Launcher
Enhanced startup script with better error handling and configuration
"""

import os
import sys
import subprocess
from pathlib import Path

def check_environment():
    """Check if virtual environment is activated"""
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Virtual environment not detected!")
        print("💡 Activate your virtual environment first:")
        print("   Windows: .venv\\Scripts\\activate")
        print("   Linux/Mac: source .venv/bin/activate")
        return False
    return True

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import flask
        import requests
        print("✅ Core dependencies found")
        return True
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("💡 Install requirements: pip install -r requirements.txt")
        return False

def setup_environment():
    """Setup environment variables"""
    env_file = Path('.env')
    if not env_file.exists():
        print("⚠️  .env file not found, creating basic configuration...")
        with open('.env', 'w') as f:
            f.write("""# CodeEx AI Configuration
SECRET_KEY=your-secret-key-here
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
FLASK_DEBUG=false
PORT=5000
HOST=0.0.0.0
""")
        print("📝 Created .env file - please update with your credentials")
    
    # Set default environment variables
    os.environ.setdefault('FLASK_DEBUG', 'false')
    os.environ.setdefault('PORT', '5000')
    os.environ.setdefault('HOST', '0.0.0.0')

def main():
    """Main launcher function"""
    print("🚀 CodeEx AI Server Launcher")
    print("=" * 40)
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    if not check_dependencies():
        sys.exit(1)
    
    setup_environment()
    
    # Launch the application
    print("🌟 Starting CodeEx AI...")
    try:
        subprocess.run([sys.executable, 'app.py'], check=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Server error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()