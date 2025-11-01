#!/bin/bash
# Quick start script for KIKA

set -e

echo "🚀 Starting KIKA - Nuclear Data Viewer"
echo "========================================"
echo ""

# Detect if we're using poetry or pip
USE_POETRY=false
if command -v poetry &> /dev/null; then
    # Check if streamlit is in poetry venv
    if poetry run which streamlit &> /dev/null; then
        USE_POETRY=true
        echo "✓ Using Poetry environment"
    fi
fi

# If not using poetry, check for streamlit in PATH
if [ "$USE_POETRY" = false ]; then
    if ! command -v streamlit &> /dev/null; then
        echo "❌ Streamlit not found!"
        echo ""
        echo "Please install dependencies first:"
        echo "  poetry install --with ui"
        echo "  OR"
        echo "  pip install -r streamlit_app/requirements.txt"
        exit 1
    fi
    echo "✓ Using system Streamlit"
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set PYTHONPATH to include parent directory (for mcnpy import)
export PYTHONPATH="${PYTHONPATH}:${SCRIPT_DIR}/.."

echo "✓ Python path configured"
echo ""
echo "🌐 Starting Streamlit server..."
echo "   Access at: http://localhost:8501"
echo ""
echo "   💡 Tip for WSL users:"
echo "      Open http://localhost:8501 in your Windows browser"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Run streamlit
cd "$SCRIPT_DIR"

if [ "$USE_POETRY" = true ]; then
    poetry run streamlit run app.py
else
    streamlit run app.py
fi
