#!/usr/bin/env bash
# Installe le LaunchAgent macOS pour DocFinder uvicorn.
# Usage : bash scripts/install_launchagent.sh
# Effet : uvicorn :8000 démarre au login + redémarre automatiquement en cas de crash.
set -euo pipefail

PLIST_NAME="digital.jinkohub.docfinder.plist"
SRC="$(cd "$(dirname "$0")" && pwd)/${PLIST_NAME}"
DST="${HOME}/Library/LaunchAgents/${PLIST_NAME}"

echo "→ copie ${SRC} → ${DST}"
cp "${SRC}" "${DST}"

echo "→ si déjà chargé, unload d'abord (ignore erreur)"
launchctl unload "${DST}" 2>/dev/null || true

echo "→ load"
launchctl load "${DST}"

echo "→ état"
launchctl list | grep -i docfinder || echo "(non listé — vérifier /tmp/docfinder_uvicorn.log)"

echo ""
echo "✅ LaunchAgent installé."
echo "   Log : tail -f /tmp/docfinder_uvicorn.log"
echo "   Stop : launchctl unload ~/Library/LaunchAgents/${PLIST_NAME}"
echo "   Restart : launchctl kickstart -k gui/$(id -u)/digital.jinkohub.docfinder"
