#!/usr/bin/env python3
"""
🔍 CodeEx AI - Setup Verification Script
Checks if pip and all dependencies are properly installed
"""

import sys
import subprocess
import importlib
import platform

def print_header():
    """Print verification header"""
    print("🔍 CodeEx AI - Setup Verification")
    print("=" * 40)
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.system()} {platform.release()}")
    print("=" * 40)

def check_python_version():
    """Check Python version"""
    print("\n🐍 Checking Python version...")
    
    if sys.version_info >= (3, 7):
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} - OK")
        return True
    else:
        print(f"❌ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} - Need 3.7+")
        return False

def check_pip():
    """Check if pip is available"""
    print("\n📦 Checking pip...")
    
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "--version"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ pip - {result.stdout.strip()}")
            return True
        else:
            print("❌ pip - Not working properly")
            return False
    except Exception as e:
        print(f"❌ pip - Error: {e}")
        return False

def check_dependencies():
    """Check required dependencies"""
    print("\n📚 Checking dependencies...")
    
    dependencies = {
        'flask': 'Flask==3.0.0',
        'flask_cors': 'Flask-CORS==4.0.0',
        'werkzeug': 'Werkzeug>=3.0.0',
        'cryptography': 'cryptography==41.0.7',
        'dotenv': 'python-dotenv==1.0.0',
        'gunicorn': 'gunicorn==21.2.0',
        'PIL': 'Pillow>=9.0.0 (for PWA icons)',
    }
    
    results = {}
    
    for module, description in dependencies.items():
        try:
            if module == 'dotenv':
                importlib.import_module('dotenv')
            else:
                importlib.import_module(module)
            print(f"✅ {description}")
            results[module] = True
        except ImportError:
            print(f"❌ {description} - Not installed")
            results[module] = False
        except Exception as e:
            print(f"⚠️ {description} - Error: {e}")
            results[module] = False
    
    return results

def check_files():
    """Check required files"""
    print("\n📁 Checking required files...")
    
    required_files = [
        'app.py',
        'requirements.txt',
        'static/manifest.json',
        'static/sw.js',
        'templates/base.html',
        'templates/chat.html',
        'templates/login.html',
    ]
    
    results = {}
    
    for file_path in required_files:
        try:
            with open(file_path, 'r') as f:
                print(f"✅ {file_path}")
                results[file_path] = True
        except FileNotFoundError:
            print(f"❌ {file_path} - Missing")
            results[file_path] = False
        except Exception as e:
            print(f"⚠️ {file_path} - Error: {e}")
            results[file_path] = False
    
    return results

def check_pwa_assets():
    """Check PWA assets"""
    print("\n📱 Checking PWA assets...")
    
    pwa_files = [
        'static/manifest.json',
        'static/sw.js',
        'static/browserconfig.xml',
        'templates/offline.html',
    ]
    
    results = {}
    
    for file_path in pwa_files:
        try:
            with open(file_path, 'r') as f:
                print(f"✅ {file_path}")
                results[file_path] = True
        except FileNotFoundError:
            print(f"⚠️ {file_path} - Missing (will be generated)")
            results[file_path] = False
        except Exception as e:
            print(f"⚠️ {file_path} - Error: {e}")
            results[file_path] = False
    
    return results

def test_app_import():
    """Test if the app can be imported"""
    print("\n🚀 Testing app import...")
    
    try:
        sys.path.insert(0, '.')
        from app import app
        print("✅ Flask app import - OK")
        return True
    except Exception as e:
        print(f"❌ Flask app import - Error: {e}")
        return False

def provide_recommendations(results):
    """Provide recommendations based on results"""
    print("\n💡 Recommendations:")
    
    # Check Python version
    if not results.get('python_version', False):
        print("🔧 Upgrade Python to 3.7 or higher")
        print("   Download from: https://python.org")
    
    # Check pip
    if not results.get('pip', False):
        print("🔧 Install pip:")
        print("   python -m ensurepip --upgrade")
        print("   or download get-pip.py from https://bootstrap.pypa.io/get-pip.py")
    
    # Check dependencies
    missing_deps = []
    for dep, status in results.get('dependencies', {}).items():
        if not status:
            missing_deps.append(dep)
    
    if missing_deps:
        print("🔧 Install missing dependencies:")
        print("   pip install -r requirements.txt")
        if 'PIL' in missing_deps:
            print("   pip install Pillow")
    
    # Check files
    missing_files = []
    for file_path, status in results.get('files', {}).items():
        if not status:
            missing_files.append(file_path)
    
    if missing_files:
        print("🔧 Missing files detected:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("   Make sure you have all project files")
    
    # Check PWA assets
    missing_pwa = []
    for file_path, status in results.get('pwa_assets', {}).items():
        if not status:
            missing_pwa.append(file_path)
    
    if missing_pwa:
        print("🔧 Generate PWA assets:")
        print("   python setup_pwa.py")

def main():
    """Main verification function"""
    print_header()
    
    results = {}
    
    # Run all checks
    results['python_version'] = check_python_version()
    results['pip'] = check_pip()
    results['dependencies'] = check_dependencies()
    results['files'] = check_files()
    results['pwa_assets'] = check_pwa_assets()
    results['app_import'] = test_app_import()
    
    # Calculate overall status
    total_checks = 0
    passed_checks = 0
    
    # Count individual results
    for category, result in results.items():
        if isinstance(result, dict):
            for item_result in result.values():
                total_checks += 1
                if item_result:
                    passed_checks += 1
        else:
            total_checks += 1
            if result:
                passed_checks += 1
    
    # Print summary
    print("\n" + "=" * 40)
    print("📊 Verification Summary")
    print("=" * 40)
    print(f"✅ Passed: {passed_checks}/{total_checks}")
    
    if passed_checks == total_checks:
        print("🎉 All checks passed! Your setup is ready.")
        print("\n🚀 To start the application:")
        print("   python app.py")
        print("   Open: http://localhost:5000")
    elif passed_checks >= total_checks * 0.8:  # 80% pass rate
        print("⚠️ Most checks passed. Minor issues detected.")
        print("Your app should work, but some features may be limited.")
    else:
        print("❌ Several issues detected. Please fix before running.")
    
    # Provide recommendations
    provide_recommendations(results)
    
    print("\n📚 For detailed installation help:")
    print("   See INSTALL.md")
    print("   Run: python install.py")

if __name__ == "__main__":
    main()