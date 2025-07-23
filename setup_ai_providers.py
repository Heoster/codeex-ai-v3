"""
Setup script for Multi-Provider AI System
Helps configure API keys and test the system
"""

import os
import sys
from dotenv import load_dotenv, set_key

def setup_api_keys():
    """Interactive setup for API keys"""
    print("🔧 Multi-Provider AI Setup")
    print("=" * 40)
    print()
    
    env_file = '.env'
    
    # Load existing .env file
    load_dotenv()
    
    print("This script will help you configure API keys for:")
    print("1. OpenAI GPT models")
    print("2. Google Gemini 1.5 Flash")
    print()
    
    # OpenAI API Key
    current_openai = os.getenv('OPENAI_API_KEY', '')
    if current_openai and current_openai != 'your-openai-api-key-here':
        print(f"✅ OpenAI API Key already configured: {'*' * 20}{current_openai[-10:]}")
        update_openai = input("Update OpenAI API key? (y/N): ").lower().strip()
        if update_openai != 'y':
            current_openai = None
    else:
        current_openai = None
    
    if current_openai is None:
        print("\n📝 OpenAI API Key Setup:")
        print("1. Go to: https://platform.openai.com/api-keys")
        print("2. Create a new API key")
        print("3. Copy the key and paste it below")
        print()
        
        openai_key = input("Enter your OpenAI API key (or press Enter to skip): ").strip()
        if openai_key:
            set_key(env_file, 'OPENAI_API_KEY', openai_key)
            print("✅ OpenAI API key saved!")
        else:
            print("⏭️  Skipped OpenAI setup")
    
    # Google Gemini API Key
    current_gemini = os.getenv('GOOGLE_GEMINI_API_KEY', '')
    if current_gemini and current_gemini != 'your-google-gemini-api-key-here':
        print(f"✅ Gemini API Key already configured: {'*' * 20}{current_gemini[-10:]}")
        update_gemini = input("Update Gemini API key? (y/N): ").lower().strip()
        if update_gemini != 'y':
            current_gemini = None
    else:
        current_gemini = None
    
    if current_gemini is None:
        print("\n📝 Google Gemini API Key Setup:")
        print("1. Go to: https://makersuite.google.com/app/apikey")
        print("2. Create a new API key")
        print("3. Copy the key and paste it below")
        print()
        
        gemini_key = input("Enter your Google Gemini API key (or press Enter to skip): ").strip()
        if gemini_key:
            set_key(env_file, 'GOOGLE_GEMINI_API_KEY', gemini_key)
            print("✅ Gemini API key saved!")
        else:
            print("⏭️  Skipped Gemini setup")
    
    print("\n✨ API key setup complete!")

def install_packages():
    """Install required packages"""
    print("\n📦 Installing Required Packages")
    print("-" * 40)
    
    packages = [
        'openai==1.3.0',
        'google-generativeai==0.3.0'
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        try:
            os.system(f"{sys.executable} -m pip install {package}")
            print(f"✅ {package} installed successfully")
        except Exception as e:
            print(f"❌ Failed to install {package}: {e}")
    
    print("\n✅ Package installation complete!")

def test_setup():
    """Test the setup"""
    print("\n🧪 Testing Setup")
    print("-" * 40)
    
    try:
        # Test imports
        import openai
        print("✅ OpenAI package imported successfully")
    except ImportError:
        print("❌ OpenAI package not available")
    
    try:
        import google.generativeai as genai
        print("✅ Google Generative AI package imported successfully")
    except ImportError:
        print("❌ Google Generative AI package not available")
    
    # Test API keys
    load_dotenv()
    
    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key and openai_key != 'your-openai-api-key-here':
        print("✅ OpenAI API key configured")
    else:
        print("⚠️  OpenAI API key not configured")
    
    gemini_key = os.getenv('GOOGLE_GEMINI_API_KEY')
    if gemini_key and gemini_key != 'your-google-gemini-api-key-here':
        print("✅ Gemini API key configured")
    else:
        print("⚠️  Gemini API key not configured")

def main():
    """Main setup function"""
    print("🚀 Multi-Provider AI System Setup")
    print("=" * 50)
    print()
    
    print("This setup will:")
    print("1. Install required packages")
    print("2. Configure API keys")
    print("3. Test the configuration")
    print()
    
    proceed = input("Continue with setup? (Y/n): ").lower().strip()
    if proceed == 'n':
        print("Setup cancelled.")
        return
    
    # Install packages
    install_packages()
    
    # Setup API keys
    setup_api_keys()
    
    # Test setup
    test_setup()
    
    print("\n🎉 Setup Complete!")
    print("\nNext steps:")
    print("1. Run: python test_multi_provider_ai.py")
    print("2. Start the Flask app: python app.py")
    print("3. Visit: http://localhost:5000")

if __name__ == "__main__":
    main()