#!/bin/bash
# icloud_prewarm_daemon.sh — Force la matérialisation locale de ~/Documents
# (backed by iCloud Drive File Provider) pour que le serveur uvicorn puisse
# servir /files/raw sans EDEADLK.
#
# Stratégie :
#   1. brctl download : demande asynchrone à iCloud de télécharger le dossier.
#   2. cat > /dev/null parallèle : force la matérialisation complète + garde
#      les fichiers chauds dans le cache système (éviction iCloud rapide sinon).
#
# Lancement :
#   nohup bash scripts/icloud_prewarm_daemon.sh > /tmp/prewarm_daemon.log 2>&1 &
#   disown
#
# Stop :
#   pkill -f icloud_prewarm_daemon.sh
set -u
cd /Users/julien
N=0
while true; do
  N=$((N+1))
  # 1) Demande asynchrone à iCloud (idempotent ; retourne immédiatement)
  brctl download /Users/julien/Documents 2>/dev/null
  # 2) Lecture forcée parallèle pour finaliser la matérialisation
  find Documents -type f \
    \( -name "*.pdf" -o -name "*.docx" -o -name "*.pptx" -o -name "*.xlsx" \
       -o -name "*.doc" -o -name "*.ppt" -o -name "*.xls" \) \
    -not -name "~\$*" -print0 2>/dev/null \
    | xargs -0 -P 16 -I{} sh -c 'cat "$1" > /dev/null 2>&1' _ {}
  echo "[$(date +%H:%M:%S)] brctl+cat sweep #$N done"
  sleep 5
done
