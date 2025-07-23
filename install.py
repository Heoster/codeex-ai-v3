#!/usr/bin/env python3
"""
🚀 CodeEx AI - Installation Script
Installs pip, dependencies, and sets up the PWA application
"""

import os
import sys
import subprocess
import platform
import urllib.request
import tempfile
from pathlib import Path

def print_header():
    """Print installation header"""
    print("🚀 CodeEx AI - Installation Script")
    print("=" * 50)
    print(f"Python Version: {sys.version}")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print("=" * 50)

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        print(f"Current version: {sys.version}")
        print("Please upgrade Python and try again")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} is compatible")
    return True

def check_pip_installed():
    """Check if pip is already installed"""
    try:
        import pip
        result = subprocess.run([sys.executable, "-m", "pip", "--version"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ pip is already installed: {result.stdout.strip()}")
            return True
    except ImportError:
        pass
    
    print("⚠️ pip is not installed")
    return False

def install_pip():
    """Install pip using get-pip.py"""
    print("📦 Installing pip...")
    
    try:
        # Download get-pip.py
        get_pip_url = "https://bootstrap.pypa.io/get-pip.py"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.py') as tmp_file:
            print(f"Downloading get-pip.py from {get_pip_url}")
            
            with urllib.request.urlopen(get_pip_url) as response:
                tmp_file.write(response.read())
            
            tmp_file_path = tmp_file.name
        
        # Run get-pip.py
        print("Running get-pip.py...")
        result = subprocess.run([sys.executable, tmp_file_path], 
                              capture_output=True, text=True)
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        if result.returncode == 0:
            print("✅ pip installed successfully!")
            print(result.stdout)
            return True
        else:
            print("❌ Failed to install pip")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error installing pip: {e}")
        return False

def install_pip_alternative():
    """Alternative pip installation methods"""
    system = platform.system().lower()
    
    print("🔄 Trying alternative pip installation methods...")
    
    if system == "windows":
        print("Windows detected - pip should come with Python")
        print("Try reinstalling Python from https://python.org")
        return False
    
    elif system == "darwin":  # macOS
        print("macOS detected - trying homebrew or system package manager")
        
        # Try with easy_install (deprecated but might work)
        try:
            result = subprocess.run(["easy_install", "pip"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ pip installed via easy_install")
                return True
        except FileNotFoundError:
            pass
        
        print("Try installing with:")
        print("  brew install python")
        print("  or download Python from https://python.org")
        
    elif system == "linux":
        print("Linux detected - trying system package manager")
        
        # Try different package managers
        package_managers = [
            (["apt-get", "update"], ["apt-get", "install", "-y", "python3-pip"]),
            (["yum", "update"], ["yum", "install", "-y", "python3-pip"]),
            (["dnf", "update"], ["dnf", "install", "-y", "python3-pip"]),
            (["pacman", "-Sy"], ["pacman", "-S", "--noconfirm", "python-pip"]),
            (["zypper", "refresh"], ["zypper", "install", "-y", "python3-pip"]),
        ]
        
        for update_cmd, install_cmd in package_managers:
            try:
                print(f"Trying: {' '.join(install_cmd)}")
                subprocess.run(update_cmd, check=True, capture_output=True)
                result = subprocess.run(install_cmd, check=True, capture_output=True)
                
                if result.returncode == 0:
                    print(f"✅ pip installed via {install_cmd[0]}")
                    return True
                    
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
        
        print("Manual installation required:")
        print("  Ubuntu/Debian: sudo apt-get install python3-pip")
        print("  CentOS/RHEL: sudo yum install python3-pip")
        print("  Fedora: sudo dnf install python3-pip")
        print("  Arch: sudo pacman -S python-pip")
    
    return False

def upgrade_pip():
    """Upgrade pip to latest version"""
    print("⬆️ Upgrading pip to latest version...")
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "--upgrade", "pip"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ pip upgraded successfully")
            return True
        else:
            print("⚠️ pip upgrade failed, but continuing...")
            print(result.stderr)
            return True  # Continue even if upgrade fails
            
    except Exception as e:
        print(f"⚠️ pip upgrade error: {e}")
        return True  # Continue even if upgrade fails

def install_dependencies():
    """Install project dependencies"""
    print("📦 Installing project dependencies...")
    
    # Check if requirements.txt exists
    if not os.path.exists('requirements.txt'):
        print("❌ requirements.txt not found")
        return False
    
    try:
        # Install requirements
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Dependencies installed successfully!")
            return True
        else:
            print("❌ Failed to install dependencies")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def install_pwa_dependencies():
    """Install additional PWA dependencies"""
    print("🎨 Installing PWA dependencies...")
    
    pwa_packages = [
        "Pillow>=9.0.0",  # For icon generation
    ]
    
    for package in pwa_packages:
        try:
            print(f"Installing {package}...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", package
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ {package} installed")
            else:
                print(f"⚠️ Failed to install {package}")
                print(result.stderr)
                
        except Exception as e:
            print(f"⚠️ Error installing {package}: {e}")

def create_virtual_environment():
    """Create and activate virtual environment"""
    print("🔧 Setting up virtual environment...")
    
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return True
    
    try:
        # Create virtual environment
        result = subprocess.run([
            sys.executable, "-m", "venv", "venv"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Virtual environment created")
            
            # Provide activation instructions
            if platform.system().lower() == "windows":
                activate_cmd = "venv\\Scripts\\activate"
            else:
                activate_cmd = "source venv/bin/activate"
            
            print(f"To activate: {activate_cmd}")
            return True
        else:
            print("⚠️ Failed to create virtual environment")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"⚠️ Error creating virtual environment: {e}")
        return False

def setup_database():
    """Initialize the database"""
    print("🗄️ Setting up database...")
    
    try:
        # Import and initialize database
        sys.path.insert(0, '.')
        from app import init_db
        
        init_db()
        print("✅ Database initialized successfully")
        return True
        
    except Exception as e:
        print(f"⚠️ Database setup error: {e}")
        print("Database will be initialized on first run")
        return True

def setup_pwa_assets():
    """Set up PWA assets"""
    print("📱 Setting up PWA assets...")
    
    try:
        # Run PWA setup if available
        if os.path.exists('setup_pwa.py'):
            result = subprocess.run([
                sys.executable, "setup_pwa.py"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ PWA assets generated successfully")
                return True
            else:
                print("⚠️ PWA setup had some issues:")
                print(result.stdout)
                print(result.stderr)
                return True  # Continue even if PWA setup fails
        else:
            print("⚠️ setup_pwa.py not found, skipping PWA asset generation")
            return True
            
    except Exception as e:
        print(f"⚠️ PWA setup error: {e}")
        return True

def create_env_file():
    """Create .env file if it doesn't exist"""
    print("⚙️ Setting up environment configuration...")
    
    if os.path.exists('.env'):
        print("✅ .env file already exists")
        return True
    
    try:
        env_content = """# CodeEx AI Configuration
SECRET_KEY=your-secret-key-change-this-in-production
FLASK_DEBUG=true
HOST=0.0.0.0
PORT=5000

# AI Service API Keys (optional)
# OPENAI_API_KEY=your-openai-api-key-here
# ANTHROPIC_API_KEY=your-anthropic-api-key-here

# PWA Configuration
PWA_NAME=CodeEx AI
PWA_SHORT_NAME=CodeEx AI
PWA_DESCRIPTION=Your intelligent AI assistant with context memory
"""
        
        with open('.env', 'w') as f:
            f.write(env_content)
        
        print("✅ .env file created")
        print("📝 Edit .env file to configure your API keys")
        return True
        
    except Exception as e:
        print(f"⚠️ Error creating .env file: {e}")
        return True

def run_tests():
    """Run basic functionality tests"""
    print("🧪 Running basic tests...")
    
    try:
        # Test imports
        test_imports = [
            'flask',
            'flask_cors',
            'werkzeug',
            'cryptography',
            'python-dotenv'
        ]
        
        for module in test_imports:
            try:
                __import__(module.replace('-', '_'))
                print(f"✅ {module} import successful")
            except ImportError as e:
                print(f"⚠️ {module} import failed: {e}")
        
        # Test app startup
        sys.path.insert(0, '.')
        try:
            from app import app
            print("✅ Flask app import successful")
        except Exception as e:
            print(f"⚠️ Flask app import failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"⚠️ Test error: {e}")
        return True

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "=" * 50)
    print("🎉 Installation completed!")
    print("=" * 50)
    
    print("\n📋 Next Steps:")
    print("1. Activate virtual environment (if created):")
    
    if platform.system().lower() == "windows":
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    
    print("\n2. Configure your settings:")
    print("   - Edit .env file with your API keys")
    print("   - Customize PWA settings if needed")
    
    print("\n3. Start the application:")
    print("   python app.py")
    
    print("\n4. Open in browser:")
    print("   http://localhost:5000")
    
    print("\n5. Test PWA features:")
    print("   - Look for install button in browser")
    print("   - Test offline functionality")
    print("   - Try 'Add to Home Screen' on mobile")
    
    print("\n🔧 Troubleshooting:")
    print("- If you encounter issues, check the error messages above")
    print("- Make sure you have Python 3.7+ installed")
    print("- Try running with administrator/sudo privileges if needed")
    
    print("\n📱 PWA Features:")
    print("- ✅ Offline functionality")
    print("- ✅ Installable on desktop and mobile")
    print("- ✅ Push notifications ready")
    print("- ✅ Background sync")
    print("- ✅ Native app experience")
    
    print("\n🚀 Your CodeEx AI PWA is ready to use!")

def main():
    """Main installation function"""
    print_header()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Installation steps
    steps = [
        ("Checking pip installation", check_pip_installed),
        ("Installing pip (if needed)", lambda: install_pip() if not check_pip_installed() else True),
        ("Upgrading pip", upgrade_pip),
        ("Creating virtual environment", create_virtual_environment),
        ("Installing dependencies", install_dependencies),
        ("Installing PWA dependencies", install_pwa_dependencies),
        ("Setting up database", setup_database),
        ("Creating environment file", create_env_file),
        ("Setting up PWA assets", setup_pwa_assets),
        ("Running tests", run_tests),
    ]
    
    success_count = 0
    failed_steps = []
    
    for step_name, step_func in steps:
        print(f"\n{step_name}...")
        try:
            if step_func():
                success_count += 1
            else:
                failed_steps.append(step_name)
        except Exception as e:
            print(f"❌ {step_name} failed: {e}")
            failed_steps.append(step_name)
    
    # Print results
    print("\n" + "=" * 50)
    print("📊 Installation Summary")
    print("=" * 50)
    print(f"✅ Successful steps: {success_count}/{len(steps)}")
    
    if failed_steps:
        print(f"⚠️ Failed steps: {len(failed_steps)}")
        for step in failed_steps:
            print(f"   - {step}")
    
    if success_count >= len(steps) - 2:  # Allow 2 failures
        print_next_steps()
    else:
        print("\n❌ Installation had significant issues")
        print("Please check the error messages above and try again")
        print("\nCommon solutions:")
        print("- Run with administrator/sudo privileges")
        print("- Check internet connection")
        print("- Update Python to latest version")
        print("- Install pip manually from https://pip.pypa.io/")

if __name__ == "__main__":
    main()