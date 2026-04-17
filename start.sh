#!/usr/bin/env bash
# start.sh — Démarre DocFinder en une seule commande.
#
# Ce que ce script fait :
#   1. Télécharge le binaire Qdrant si absent (détection d'OS automatique)
#   2. Lance Qdrant en arrière-plan et attend qu'il soit prêt (max 15s)
#   3. Crée la collection Qdrant si elle n'existe pas (setup_qdrant.py)
#   4. Lance le serveur FastAPI sur le port 8000 (uvicorn, reload activé)
#
# Usage :
#   bash start.sh          # démarrage normal
#   bash start.sh --reset  # réinitialise la collection avant de démarrer

set -euo pipefail

cd "$(dirname "$0")"

# ── Variables d'environnement (.env gitignored) ───────────────────────────────
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

# ── Constantes ────────────────────────────────────────────────────────────────
QDRANT_VERSION="v1.13.4"
QDRANT_PORT=6333
STARTUP_TIMEOUT=15   # secondes max pour attendre que Qdrant démarre
COLLECTION="docfinder"

# ── Détection OS / architecture ───────────────────────────────────────────────
OS="$(uname -s)"
ARCH="$(uname -m)"

case "${OS}-${ARCH}" in
    Darwin-arm64)  QDRANT_TARBALL="qdrant-aarch64-apple-darwin.tar.gz" ;;
    Darwin-x86_64) QDRANT_TARBALL="qdrant-x86_64-apple-darwin.tar.gz"  ;;
    Linux-x86_64)  QDRANT_TARBALL="qdrant-x86_64-unknown-linux-musl.tar.gz" ;;
    *)
        echo "⛔  Architecture non supportée : ${OS}-${ARCH}"
        echo "   Téléchargez manuellement : https://github.com/qdrant/qdrant/releases"
        exit 1
        ;;
esac

# ── 1. Téléchargement de Qdrant si absent ─────────────────────────────────────
if [[ ! -x "./qdrant" ]]; then
    echo "→ Binaire Qdrant absent — téléchargement ${QDRANT_VERSION} (${QDRANT_TARBALL})…"
    QDRANT_URL="https://github.com/qdrant/qdrant/releases/download/${QDRANT_VERSION}/${QDRANT_TARBALL}"
    curl -fsSL "${QDRANT_URL}" | tar -xz
    chmod +x qdrant
    echo "   Qdrant téléchargé ✓"
fi

# ── 2. Démarrage de Qdrant en arrière-plan ────────────────────────────────────
echo "→ Démarrage de Qdrant (port ${QDRANT_PORT})…"

if lsof -ti "tcp:${QDRANT_PORT}" &>/dev/null; then
    echo "   Port ${QDRANT_PORT} déjà occupé — Qdrant déjà en cours d'exécution."
else
    ./qdrant --config-path qdrant_config.yaml &>/dev/null &
    echo "   PID Qdrant : $!"
fi

# Attendre que Qdrant réponde sur /healthz
echo -n "   Attente que Qdrant soit prêt"
elapsed=0
until curl -sf "http://localhost:${QDRANT_PORT}/healthz" &>/dev/null; do
    sleep 1
    elapsed=$((elapsed + 1))
    echo -n "."
    if [[ ${elapsed} -ge ${STARTUP_TIMEOUT} ]]; then
        echo ""
        echo "⛔  Qdrant n'a pas démarré dans les ${STARTUP_TIMEOUT}s."
        echo "   Lancez manuellement pour voir les logs :"
        echo "     ./qdrant --config-path qdrant_config.yaml"
        exit 1
    fi
done
echo " ✓"

# ── 3. Initialisation de la collection si absente ─────────────────────────────
RESET_FLAG="${1:-}"
if [[ "${RESET_FLAG}" == "--reset" ]]; then
    echo "→ Réinitialisation de la collection (--reset)…"
    python setup_qdrant.py --force
    echo "   Collection réinitialisée ✓"
else
    HTTP_STATUS=$(curl -sf -o /dev/null -w "%{http_code}" \
        "http://localhost:${QDRANT_PORT}/collections/${COLLECTION}" 2>/dev/null || echo "000")
    if [[ "${HTTP_STATUS}" != "200" ]]; then
        echo "→ Collection '${COLLECTION}' absente — initialisation…"
        python setup_qdrant.py
        echo "   Collection créée ✓"
    else
        echo "→ Collection '${COLLECTION}' déjà présente ✓"
    fi
fi

# ── 4. Tunnel Cloudflare (daemon indépendant) ─────────────────────────────────
if command -v cloudflared &>/dev/null && [ -n "${CLOUDFLARE_TUNNEL_TOKEN:-}" ]; then
    # Named tunnel (URL fixe) — survit aux redémarrages d'uvicorn.
    # Le token est dans .env (gitignored).
    (
        while true; do
            cloudflared tunnel --no-autoupdate run --token "$CLOUDFLARE_TUNNEL_TOKEN" \
                2>&1 | grep -E "connected|ERR|error|INF" || true
            echo "[cloudflared] tunnel coupé — redémarrage dans 5s…"
            sleep 5
        done
    ) &
    CLOUDFLARED_PID=$!
    echo "→ Cloudflare tunnel démarré (PID $CLOUDFLARED_PID)"
    echo "   URL publique : ${DOCFINDER_PUBLIC_URL:-https://docfinder.jinkohub.digital}"
elif command -v cloudflared &>/dev/null; then
    echo "→ CLOUDFLARE_TUNNEL_TOKEN absent dans .env — tunnel non démarré"
else
    echo "→ cloudflared absent — tunnel Cloudflare non démarré (optionnel)"
fi

# ── 5. Démarrage du serveur FastAPI avec watchdog ─────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  DocFinder prêt — http://localhost:8000"
echo "  Admin         — http://localhost:8000/admin"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

HEALTH_URL="http://localhost:8000/health"
FAIL_LIMIT=3   # échecs consécutifs avant redémarrage
POLL=30        # secondes entre chaque health check

# Désactiver set -e pour la boucle watchdog (kill -0 peut retourner 1)
set +e

while true; do
    # Libérer le port avant tout démarrage — un enfant --reload peut survivre à un crash
    lsof -ti tcp:8000 | xargs kill -9 2>/dev/null || true
    sleep 1

    uvicorn server.main:app --port 8000 --reload --reload-dir server --reload-dir shared &
    UVICORN_PID=$!
    echo "[watchdog] uvicorn démarré (PID $UVICORN_PID)"

    # Ne compter les échecs qu'après le premier succès
    # (évite de redémarrer pendant le chargement du modèle ~60s)
    started=0
    fails=0

    while kill -0 "$UVICORN_PID" 2>/dev/null; do
        sleep "$POLL"
        if curl -sf --max-time 10 "$HEALTH_URL" >/dev/null 2>&1; then
            started=1
            fails=0
        elif [ "$started" -eq 1 ]; then
            fails=$((fails + 1))
            echo "[watchdog] health check échoué ($fails/$FAIL_LIMIT)"
            if [ "$fails" -ge "$FAIL_LIMIT" ]; then
                echo "[watchdog] serveur bloqué — kill et redémarrage…"
                # Tuer tout le process group (parent + enfants --reload)
                kill -9 -- "-$UVICORN_PID" 2>/dev/null
                # Libérer le port au cas où un enfant survive
                lsof -ti tcp:8000 | xargs kill -9 2>/dev/null || true
                break
            fi
        fi
    done

    echo "[watchdog] uvicorn arrêté — redémarrage dans 3s…"
    sleep 3
done
