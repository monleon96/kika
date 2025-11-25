#!/bin/bash
# Script para iniciar la aplicación KIKA Desktop en modo desarrollo

echo "🚀 Iniciando KIKA Desktop App..."

# Terminal 1: MCNPy Backend
echo "📦 Iniciando MCNPy server (puerto 8001)..."
cd /home/MONLEON-JUAN/MCNPy/mcnpy_server
poetry run uvicorn app:app --host 0.0.0.0 --port 8001 > /tmp/mcnpy-server.log 2>&1 &
BACKEND_PID=$!

# Esperar a que el backend inicie
sleep 3

# Terminal 2: Vite Frontend
echo "🎨 Iniciando Vite dev server (puerto 1420)..."
cd /home/MONLEON-JUAN/MCNPy/kika-desktop
npm run dev > /tmp/vite-dev.log 2>&1 &
FRONTEND_PID=$!

# Esperar a que Vite inicie
sleep 3

echo ""
echo "✅ Aplicación iniciada:"
echo "   Backend:  http://localhost:8001 (PID: $BACKEND_PID)"
echo "   Frontend: http://localhost:1420 (PID: $FRONTEND_PID)"
echo ""
echo "📝 Logs:"
echo "   Backend:  tail -f /tmp/mcnpy-server.log"
echo "   Frontend: tail -f /tmp/vite-dev.log"
echo ""
echo "🛑 Para detener: kill $BACKEND_PID $FRONTEND_PID"
