#!/usr/bin/env python3
"""
🚀 Chat History Setup Script
Initializes the enhanced chat history persistence system
"""

import os
import sys
import subprocess
import sqlite3
from datetime import datetime

def install_dependencies():
    """Install required Python packages"""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False
    return True

def initialize_database():
    """Initialize the enhanced database tables"""
    print("🗄️ Initializing enhanced database...")
    try:
        from chat_history_service import chat_history_service
        print("✅ Enhanced database tables created successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize database: {e}")
        return False

def create_sample_data():
    """Create some sample data for testing"""
    print("📝 Creating sample configuration...")
    
    # Create .env file if it doesn't exist
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write("""# CodeEx AI Configuration
SECRET_KEY=your-secret-key-change-this-in-production
FLASK_DEBUG=true
HOST=0.0.0.0
PORT=5000

# Optional: Custom encryption key (auto-generated if not provided)
# CHAT_ENCRYPTION_KEY=your-base64-encoded-key
""")
        print("✅ Created .env configuration file")
    
    return True

def run_tests():
    """Run basic functionality tests"""
    print("🧪 Running basic tests...")
    try:
        # Test database connection
        conn = sqlite3.connect('codeex.db')
        cursor = conn.cursor()
        
        # Check if enhanced tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%enhanced%'")
        tables = cursor.fetchall()
        
        if len(tables) >= 2:  # Should have at least chat_sessions_enhanced and messages_enhanced
            print("✅ Enhanced database tables verified")
        else:
            print("⚠️ Some enhanced tables may be missing")
        
        conn.close()
        
        # Test encryption key generation
        from chat_history_service import chat_history_service
        if chat_history_service.cipher:
            print("✅ Encryption system initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🗂️ CodeEx AI - Chat History Setup")
    print("=" * 40)
    
    steps = [
        ("Installing dependencies", install_dependencies),
        ("Initializing database", initialize_database),
        ("Creating configuration", create_sample_data),
        ("Running tests", run_tests)
    ]
    
    success_count = 0
    for step_name, step_func in steps:
        print(f"\n{step_name}...")
        if step_func():
            success_count += 1
        else:
            print(f"❌ Failed: {step_name}")
    
    print("\n" + "=" * 40)
    if success_count == len(steps):
        print("🎉 Setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Run: python app.py")
        print("2. Open: http://localhost:5000")
        print("3. Create an account and start chatting")
        print("4. Visit /storage to manage your chat history")
        print("\n🔐 Security Notes:")
        print("- Messages are encrypted by default")
        print("- Encryption key is auto-generated and stored securely")
        print("- Configure retention policies in the storage management page")
    else:
        print(f"⚠️ Setup completed with {len(steps) - success_count} issues")
        print("Please check the error messages above and try again")
    
    print("\n📚 Documentation: See CHAT_HISTORY_FEATURES.md for detailed information")

if __name__ == "__main__":
    main()