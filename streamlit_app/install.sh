#!/bin/bash
# Installation script for KIKA dependencies

set -e

echo "📦 Installing KIKA Dependencies"
echo "================================="
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Check if poetry is available
if command -v poetry &> /dev/null; then
    echo "✓ Poetry found"
    echo ""
    echo "Installing with Poetry..."
    poetry install --with ui
    
    echo ""
    echo "✅ Installation complete!"
    echo ""
    echo "Run the app with:"
    echo "  cd streamlit_app && ./run.sh"
    
elif command -v pip &> /dev/null; then
    echo "✓ pip found (Poetry not found, using pip)"
    echo ""
    echo "Installing KIKA in editable mode..."
    pip install -e .
    
    echo ""
    echo "Installing UI dependencies..."
    pip install -r streamlit_app/requirements.txt
    
    echo ""
    echo "✅ Installation complete!"
    echo ""
    echo "Run the app with:"
    echo "  cd streamlit_app && ./run.sh"
    
else
    echo "❌ Neither Poetry nor pip found!"
    echo ""
    echo "Please install Python package manager:"
    echo "  - Poetry: https://python-poetry.org/docs/#installation"
    echo "  - Or ensure pip is available in your Python installation"
    exit 1
fi
