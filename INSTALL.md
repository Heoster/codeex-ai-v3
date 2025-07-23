# 🚀 CodeEx AI - Installation Guide

Welcome to CodeEx AI! This guide will help you install pip, dependencies, and set up your Progressive Web App.

## 📋 Prerequisites

- **Python 3.7+** (Required)
- **Internet connection** (For downloading dependencies)
- **Modern web browser** (Chrome, Firefox, Safari, Edge)

## 🔧 Quick Installation

### Option 1: Automated Installation (Recommended)

#### Windows
```cmd
# Double-click install.bat or run in Command Prompt:
install.bat
```

#### macOS/Linux
```bash
# Make executable and run:
chmod +x install.sh
./install.sh
```

#### Manual Python Installation
```bash
# Run the Python installation script directly:
python install.py
```

### Option 2: Manual Installation

#### Step 1: Install Python (if not installed)

**Windows:**
1. Download Python from [python.org](https://python.org)
2. ✅ **IMPORTANT**: Check "Add Python to PATH" during installation
3. Restart Command Prompt after installation

**macOS:**
```bash
# Using Homebrew (recommended):
brew install python3

# Or download from python.org
```

**Linux:**
```bash
# Ubuntu/Debian:
sudo apt-get update
sudo apt-get install python3 python3-pip

# CentOS/RHEL:
sudo yum install python3 python3-pip

# Fedora:
sudo dnf install python3 python3-pip

# Arch Linux:
sudo pacman -S python python-pip
```

#### Step 2: Verify Python Installation
```bash
python --version
# or
python3 --version
```

#### Step 3: Install pip (if not included)

**Download and install pip:**
```bash
# Download get-pip.py
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py

# Install pip
python get-pip.py

# Verify installation
pip --version
```

**Alternative pip installation:**
```bash
# Windows (if Python was installed from python.org):
python -m ensurepip --upgrade

# macOS:
python3 -m ensurepip --upgrade

# Linux (if pip is not included):
sudo apt-get install python3-pip  # Ubuntu/Debian
sudo yum install python3-pip      # CentOS/RHEL
```

#### Step 4: Install Project Dependencies
```bash
# Navigate to project directory
cd path/to/codeex-ai

# Install requirements
pip install -r requirements.txt

# Install PWA dependencies
pip install Pillow
```

#### Step 5: Set Up Environment
```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your settings (optional)
```

#### Step 6: Generate PWA Assets
```bash
# Generate icons and PWA assets
python setup_pwa.py
```

#### Step 7: Run the Application
```bash
# Start the server
python app.py

# Open in browser
# http://localhost:5000
```

## 🐍 Virtual Environment (Recommended)

Using a virtual environment keeps your project dependencies isolated:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install Pillow

# Run application
python app.py

# Deactivate when done
deactivate
```

## 🔍 Troubleshooting

### Common Issues

#### "python is not recognized"
**Solution:** Python is not in your PATH
- **Windows**: Reinstall Python and check "Add Python to PATH"
- **macOS/Linux**: Use `python3` instead of `python`

#### "pip is not recognized"
**Solution:** pip is not installed or not in PATH
```bash
# Try these commands:
python -m pip --version
python3 -m pip --version

# If still not working, install pip:
python -m ensurepip --upgrade
```

#### Permission Errors
**Solution:** Use appropriate permissions
```bash
# Windows (run as Administrator):
# Right-click Command Prompt → "Run as administrator"

# macOS/Linux:
sudo pip install -r requirements.txt
# or use --user flag:
pip install --user -r requirements.txt
```

#### "No module named 'pip'"
**Solution:** Install pip manually
```bash
# Download get-pip.py
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
# or use wget:
wget https://bootstrap.pypa.io/get-pip.py

# Install pip
python get-pip.py
```

#### SSL Certificate Errors
**Solution:** Update certificates or use trusted hosts
```bash
# Upgrade pip with trusted host
python -m pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org --upgrade pip

# Install requirements with trusted hosts
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements.txt
```

#### Pillow Installation Issues
**Solution:** Install system dependencies
```bash
# Ubuntu/Debian:
sudo apt-get install python3-dev python3-setuptools libjpeg-dev zlib1g-dev

# macOS:
brew install libjpeg zlib

# Windows:
# Usually works out of the box, try upgrading pip first
python -m pip install --upgrade pip
```

### Platform-Specific Issues

#### Windows
- Use Command Prompt or PowerShell as Administrator
- Ensure Python is added to PATH
- Use `python` instead of `python3`

#### macOS
- Use `python3` and `pip3` commands
- Install Xcode Command Line Tools: `xcode-select --install`
- Consider using Homebrew for Python installation

#### Linux
- Use `python3` and `pip3` commands
- Install development packages for your distribution
- May need `sudo` for system-wide installations

## 📱 PWA Features

After installation, your CodeEx AI will have:

- ✅ **Offline functionality** - Works without internet
- ✅ **Installable** - Add to home screen/desktop
- ✅ **Push notifications** - Ready for future features
- ✅ **Background sync** - Messages sync when online
- ✅ **Native app feel** - Full-screen, app-like experience

## 🧪 Testing Installation

### Verify Everything Works
```bash
# Test Python
python --version

# Test pip
pip --version

# Test dependencies
python -c "import flask; print('Flask OK')"
python -c "import PIL; print('Pillow OK')"

# Test application
python app.py
```

### PWA Testing
1. Open `http://localhost:5000` in Chrome/Edge
2. Look for install button in address bar
3. Test offline mode (DevTools → Network → Offline)
4. Check PWA audit in Lighthouse

## 🆘 Getting Help

If you're still having issues:

1. **Check Python version**: Must be 3.7+
2. **Update pip**: `python -m pip install --upgrade pip`
3. **Try virtual environment**: Isolates dependencies
4. **Check firewall**: May block downloads
5. **Use administrator privileges**: For system installations

### Manual Dependency Installation
If automatic installation fails, install each dependency manually:

```bash
pip install Flask==3.0.0
pip install Flask-CORS==4.0.0
pip install Werkzeug>=3.0.0
pip install cryptography==41.0.7
pip install python-dotenv==1.0.0
pip install gunicorn==21.2.0
pip install Pillow>=9.0.0
```

## 🎉 Success!

Once installed successfully, you should see:
```
🚀 CodeEx AI - Installation completed!
📋 Next Steps:
1. python app.py
2. Open: http://localhost:5000
3. Look for install button in browser
4. Test PWA features
```

Your CodeEx AI Progressive Web App is now ready to use! 🎊

---

**Need more help?** Check the error messages carefully - they usually contain the solution!