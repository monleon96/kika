#!/bin/bash
# Start MCNPy processing server

cd "$(dirname "$0")"

echo "🚀 Starting MCNPy Processing Server on port 8001..."
echo "📊 Endpoints available at http://localhost:8001"
echo "📖 API docs at http://localhost:8001/docs"
echo ""

uvicorn app:app --reload --port 8001 --host 0.0.0.0
