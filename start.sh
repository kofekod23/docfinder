#!/bin/bash
# Démarre Qdrant + cloudflared (optionnel) + serveur DocFinder
# Usage : ./start.sh (depuis /Users/julien/docfinder)

set -e

cd "$(dirname "$0")"

# Charger les variables d'environnement (.env gitignored)
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# ── Qdrant ──────────────────────────────────────────────────────────────────
./qdrant &
QDRANT_PID=$!
echo "[DocFinder] Qdrant démarré (PID $QDRANT_PID)"

# ── Cloudflare Tunnel (remplace ngrok) ──────────────────────────────────────
# Le token provient de .env — il ne doit jamais apparaître en ligne de commande.
CLOUDFLARED_PID=""
if [ -n "${CLOUDFLARE_TUNNEL_TOKEN:-}" ]; then
  if command -v cloudflared >/dev/null 2>&1; then
    cloudflared tunnel --no-autoupdate run --token "$CLOUDFLARE_TUNNEL_TOKEN" \
      > cloudflared.log 2>&1 &
    CLOUDFLARED_PID=$!
    echo "[DocFinder] cloudflared démarré (PID $CLOUDFLARED_PID) — logs : cloudflared.log"
    [ -n "${DOCFINDER_PUBLIC_URL:-}" ] && echo "[DocFinder] URL publique : $DOCFINDER_PUBLIC_URL"
  else
    echo "[DocFinder] ⚠ cloudflared introuvable — installer : brew install cloudflared"
  fi
else
  echo "[DocFinder] CLOUDFLARE_TUNNEL_TOKEN absent — tunnel désactivé (mode 100% local)"
fi

# ── Nettoyage à la sortie ───────────────────────────────────────────────────
cleanup() {
  echo ""
  echo "[DocFinder] Arrêt…"
  [ -n "$CLOUDFLARED_PID" ] && kill "$CLOUDFLARED_PID" 2>/dev/null || true
  kill "$QDRANT_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ── FastAPI ─────────────────────────────────────────────────────────────────
# shellcheck disable=SC1091
source .venv/bin/activate
echo "[DocFinder] Serveur local : http://localhost:8000"
uvicorn server.main:app --port 8000
