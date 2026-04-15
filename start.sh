#!/bin/bash
# Démarre Qdrant + le serveur DocFinder
# Usage : ./start.sh (depuis /Users/julien/docfinder)

set -e

cd "$(dirname "$0")"

./qdrant &
QDRANT_PID=$!
echo "[DocFinder] Qdrant démarré (PID $QDRANT_PID)"

source .venv/bin/activate
echo "[DocFinder] Serveur disponible sur http://localhost:8000"
uvicorn server.main:app --port 8000

kill $QDRANT_PID 2>/dev/null
echo "[DocFinder] Arrêt."
