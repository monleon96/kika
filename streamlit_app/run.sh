#!/bin/bash
# Quick start script for KIKA

set -e

echo "🚀 Starting KIKA - Nuclear Data Viewer"
echo "========================================"
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit not found!"
    echo ""
    echo "Please install dependencies first:"
    echo "  poetry install --with ui"
    echo "  OR"
    echo "  pip install -r streamlit_app/requirements.txt"
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set PYTHONPATH to include parent directory (for mcnpy import)
export PYTHONPATH="${PYTHONPATH}:${SCRIPT_DIR}/.."

echo "✓ Dependencies found"
echo "✓ Python path configured"
echo ""
echo "🌐 Starting Streamlit server..."
echo "   Access at: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Run streamlit
cd "$SCRIPT_DIR"
streamlit run app.py
