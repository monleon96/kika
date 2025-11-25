#!/bin/bash
# Script para detener la aplicación KIKA Desktop

echo "🛑 Deteniendo KIKA Desktop App..."

# Detener MCNPy server (puerto 8001)
BACKEND_PID=$(lsof -ti :8001)
if [ ! -z "$BACKEND_PID" ]; then
  echo "  Deteniendo MCNPy server (PID: $BACKEND_PID)..."
  kill $BACKEND_PID
fi

# Detener Vite server (puerto 1420)
FRONTEND_PID=$(lsof -ti :1420)
if [ ! -z "$FRONTEND_PID" ]; then
  echo "  Deteniendo Vite server (PID: $FRONTEND_PID)..."
  kill $FRONTEND_PID
fi

sleep 1
echo "✅ Servidores detenidos"
