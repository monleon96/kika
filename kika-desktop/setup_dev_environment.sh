#!/bin/bash

# KIKA Desktop - Development Environment Setup
# Installs Node.js, Rust, and project dependencies

set -e  # Exit on error

echo "========================================"
echo "KIKA Desktop - Dev Environment Setup"
echo "========================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running in WSL
if grep -qi microsoft /proc/version; then
    echo -e "${GREEN}✓ Detected WSL environment${NC}"
    IS_WSL=true
else
    IS_WSL=false
fi

# ====================================
# 1. Install Node.js via nvm
# ====================================
echo ""
echo "Step 1: Installing Node.js..."
echo "--------------------------------"

if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    echo -e "${GREEN}✓ Node.js already installed: ${NODE_VERSION}${NC}"
else
    echo "Installing nvm (Node Version Manager)..."
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
    
    # Load nvm
    export NVM_DIR="$HOME/.nvm"
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
    
    echo "Installing Node.js 20 LTS..."
    nvm install 20
    nvm use 20
    nvm alias default 20
    
    NODE_VERSION=$(node --version)
    echo -e "${GREEN}✓ Node.js installed: ${NODE_VERSION}${NC}"
fi

# Verify npm
if command -v npm &> /dev/null; then
    NPM_VERSION=$(npm --version)
    echo -e "${GREEN}✓ npm installed: ${NPM_VERSION}${NC}"
else
    echo -e "${RED}✗ npm not found. Please install Node.js manually.${NC}"
    exit 1
fi

# ====================================
# 2. Install Rust
# ====================================
echo ""
echo "Step 2: Installing Rust..."
echo "--------------------------------"

if command -v rustc &> /dev/null; then
    RUST_VERSION=$(rustc --version)
    echo -e "${GREEN}✓ Rust already installed: ${RUST_VERSION}${NC}"
else
    echo "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    
    # Load cargo
    source "$HOME/.cargo/env"
    
    RUST_VERSION=$(rustc --version)
    echo -e "${GREEN}✓ Rust installed: ${RUST_VERSION}${NC}"
fi

# Verify cargo
if command -v cargo &> /dev/null; then
    CARGO_VERSION=$(cargo --version)
    echo -e "${GREEN}✓ Cargo installed: ${CARGO_VERSION}${NC}"
else
    echo -e "${RED}✗ Cargo not found. Please install Rust manually.${NC}"
    exit 1
fi

# ====================================
# 3. Install system dependencies
# ====================================
echo ""
echo "Step 3: Installing system dependencies..."
echo "--------------------------------"

if [ "$IS_WSL" = true ] || [ -f /etc/debian_version ]; then
    echo "Detected Debian/Ubuntu, installing required packages..."
    
    # Check if we have sudo
    if command -v sudo &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y \
            libwebkit2gtk-4.0-dev \
            build-essential \
            curl \
            wget \
            libssl-dev \
            libgtk-3-dev \
            libayatana-appindicator3-dev \
            librsvg2-dev
        
        echo -e "${GREEN}✓ System dependencies installed${NC}"
    else
        echo -e "${YELLOW}⚠ sudo not available. Please run as root or install packages manually:${NC}"
        echo "  apt-get install libwebkit2gtk-4.0-dev build-essential curl wget libssl-dev libgtk-3-dev libayatana-appindicator3-dev librsvg2-dev"
    fi
else
    echo -e "${YELLOW}⚠ Non-Debian system detected. Please install Tauri dependencies manually:${NC}"
    echo "  See: https://tauri.app/v1/guides/getting-started/prerequisites"
fi

# ====================================
# 4. Install project dependencies
# ====================================
echo ""
echo "Step 4: Installing project dependencies..."
echo "--------------------------------"

if [ -f "package.json" ]; then
    echo "Installing npm packages..."
    npm install
    echo -e "${GREEN}✓ npm packages installed${NC}"
else
    echo -e "${YELLOW}⚠ package.json not found. Will be created during project initialization.${NC}"
fi

# ====================================
# 5. Verify installation
# ====================================
echo ""
echo "========================================"
echo "Verification"
echo "========================================"
echo ""

# Check Node.js
if command -v node &> /dev/null; then
    echo -e "${GREEN}✓ Node.js: $(node --version)${NC}"
else
    echo -e "${RED}✗ Node.js not found${NC}"
fi

# Check npm
if command -v npm &> /dev/null; then
    echo -e "${GREEN}✓ npm: $(npm --version)${NC}"
else
    echo -e "${RED}✗ npm not found${NC}"
fi

# Check Rust
if command -v rustc &> /dev/null; then
    echo -e "${GREEN}✓ Rust: $(rustc --version)${NC}"
else
    echo -e "${RED}✗ Rust not found${NC}"
fi

# Check Cargo
if command -v cargo &> /dev/null; then
    echo -e "${GREEN}✓ Cargo: $(cargo --version)${NC}"
else
    echo -e "${RED}✗ Cargo not found${NC}"
fi

# ====================================
# 6. Next steps
# ====================================
echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Reload your shell to ensure environment variables are set:"
echo "   ${YELLOW}source ~/.bashrc${NC} or ${YELLOW}source ~/.zshrc${NC}"
echo ""
echo "2. Initialize the Tauri project:"
echo "   ${YELLOW}npm create tauri-app${NC}"
echo ""
echo "3. Start development:"
echo "   ${YELLOW}npm run tauri dev${NC}"
echo ""
echo "For more information, see README.md"
echo ""
