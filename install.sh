#!/bin/bash
# CodeEx AI - Unix/Linux/macOS Installation Script

echo "🚀 CodeEx AI - Installation Script"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo -e "${RED}❌ Python is not installed${NC}"
        echo "Please install Python 3.7+ from your package manager or https://python.org"
        
        # Suggest installation commands for different systems
        if [[ "$OSTYPE" == "darwin"* ]]; then
            echo "macOS: brew install python3"
        elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
            echo "Ubuntu/Debian: sudo apt-get install python3 python3-pip"
            echo "CentOS/RHEL: sudo yum install python3 python3-pip"
            echo "Fedora: sudo dnf install python3 python3-pip"
            echo "Arch: sudo pacman -S python python-pip"
        fi
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

echo -e "${GREEN}✅ Python found${NC}"
$PYTHON_CMD --version

# Make the script executable
chmod +x install.py 2>/dev/null || true

# Run the installation script
echo ""
echo -e "${BLUE}📦 Running installation script...${NC}"
$PYTHON_CMD install.py

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}🎉 Installation completed successfully!${NC}"
    echo ""
    echo -e "${YELLOW}To start the application:${NC}"
    echo "  $PYTHON_CMD app.py"
    echo ""
    echo -e "${YELLOW}Then open in browser:${NC}"
    echo "  http://localhost:5000"
else
    echo ""
    echo -e "${RED}❌ Installation encountered some issues${NC}"
    echo "Please check the error messages above"
fi