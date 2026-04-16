# Décisions techniques — DocFinder

## D1 — Modèle d'embedding : `paraphrase-multilingual-mpnet-base-v2`
**Pourquoi :** Multilingue (fr/en/etc.), 768 dims, bon équilibre qualité/taille.  
**Alternative écartée :** `all-MiniLM-L6-v2` (anglais uniquement), `multilingual-e5-large` (trop lourd pour CPU).  
**Conséquence :** `EMBEDDING_DIM = 768` dans tout le projet.

## D2 — Singleton Embedder au démarrage FastAPI
**Pourquoi :** Charger un modèle sentence-transformers prend 3-5s et ~500 MB RAM. On le charge une seule fois via `lifespan` FastAPI.  
**Règle stricte :** Aucun appel externe au moment de la recherche — modèle local, Qdrant local.

## D3 — Sparse vectors via YAKE + hash MD5
**Pourquoi :** YAKE extrait des n-grammes statistiques sans supervision. Le hash MD5 % 2^20 mappe les tokens vers un espace sparse fixe, compatible Qdrant.  
**Alternative écartée :** BM25 complet (nécessite un index inversé séparé) ; TF-IDF (besoin du corpus entier).  
**Limite connue :** Collisions de hash possibles (~1/1M), acceptables pour ce cas d'usage.

## D4 — Reranking RRF (Reciprocal Rank Fusion)
**Pourquoi :** RRF est robuste, ne nécessite pas de calibration des scores, fonctionne bien pour fusionner dense + sparse.  
**Formule :** `score = Σ 1/(k + rank_i)` avec `k=60` (valeur standard).

## D5 — Chunking dans Colab uniquement (pas au moment de la recherche)
**Pourquoi :** Le serveur local ne dispose pas des documents. L'indexation batch (GPU Colab) est la seule source de vérité dans Qdrant.

## D6 — Connexion Colab → serveur FastAPI local via Cloudflare Tunnel
**Pourquoi :** Qdrant et FastAPI tournent sur la machine locale, inaccessibles directement depuis Colab. Cloudflare Tunnel (cloudflared) expose uniquement FastAPI (port 8000) — Colab n'appelle jamais Qdrant directement, le serveur fait relais via `/admin/upsert`.  
**Setup :** `./start.sh` lit `.env`, lance `cloudflared tunnel run --token …` en arrière-plan. L'URL publique (`DOCFINDER_PUBLIC_URL`, hostname configuré sur le dashboard Cloudflare) est **stable** entre les redémarrages.  
**Historique :** Version initiale utilisait ngrok — remplacé par Cloudflare Tunnel le 2026-04-16 (voir D14). Le fallback ngrok est conservé dans `/admin/tunnels` pour rétrocompat.

## D7 — Chunking par paragraphes avec overlap
**Pourquoi :** Préserve le contexte sémantique. Taille cible : 512 tokens (~400 mots), overlap 50 mots.  
**Raison :** `paraphrase-multilingual-mpnet-base-v2` a une fenêtre de 512 tokens max.

## D8 — Deux pipelines d'indexation : server/indexer.py (local) et Colab (GPU)
**Pourquoi :** L'indexation GPU (Colab) est 10-20x plus rapide pour les grands corpus, mais nécessite une connexion ngrok. L'indexation locale (server/indexer.py et local_indexer.py) permet d'indexer sans Internet, sur CPU, directement depuis le serveur FastAPI ou en CLI.  
**Règle :** Les deux pipelines produisent des points Qdrant identiques (même format payload, mêmes hash MD5). `local_indexer.py` est le script CLI autonome ; `server/indexer.py` est le thread daemon intégré au serveur.

## D9 — Thread daemon pour l'indexation locale (server/indexer.py)
**Pourquoi :** L'indexation peut durer plusieurs minutes. Un thread daemon permet à FastAPI de rester réactif (progression via /admin/status) sans bloquer l'event loop. Un seul job à la fois, protégé par un threading.Lock.  
**Annulation :** Via `threading.Event` (`_cancel_event`) — le worker vérifie l'événement entre chaque fichier.

## D10 — `abs_path` dans le payload Qdrant
**Pourquoi :** Le chemin relatif (`path`) dépend du dossier racine utilisé à l'indexation. Sur macOS, les chemins iCloud contiennent des espaces et des caractères Unicode. `abs_path` permet à `/doc/open` et `/doc/preview` de retrouver le fichier source sans reconstruire le chemin absolu.  
**Ajouté dans :** server/indexer.py et server/chunks.py. Absent de local_indexer.py (legacy — à unifier).  
**Rétrocompatibilité :** `SearchResult.abs_path` a une valeur par défaut `""` pour les anciens chunks sans ce champ.

## D11 — Monitoring ressources via psutil (endpoint /admin/resources)
**Pourquoi :** L'indexation locale charge le CPU et la RAM (modèle sentence-transformers ~500 MB). Le bargraph admin permet de surveiller l'impact en temps réel.  
**Implémentation :** `psutil.cpu_percent()` et `psutil.virtual_memory()` sont bloquants (~100 ms) — appelés via `run_in_executor` pour ne pas bloquer l'event loop FastAPI.

## D12 — Protocole heartbeat Colab (pause/reprise, kill)
**Pourquoi :** Colab envoie un heartbeat toutes les N secondes via POST /admin/colab/heartbeat. Si le serveur ne reçoit pas de heartbeat depuis 30 s, il considère Colab comme déconnecté. L'UI admin affiche l'état (connecté/déconnecté), le type de GPU, et les boutons pause/reprise/kill.  
**État :** Géré dans `_colab_state` (dict en mémoire dans main.py) — pas persisté entre redémarrages.

## D13 — Forcer device="cpu" sur l'Embedder local (shared/embedder.py)
**Pourquoi :** Sur Apple Silicon (M1/M2/M3), sentence-transformers tente d'utiliser le GPU Metal (MPS). MPS a des limitations avec certains opérateurs et consomme la mémoire partagée GPU, impactant les autres applis. Forcer `device="cpu"` garantit la stabilité et libère le GPU Metal pour d'autres usages.  
**Décision :** `device="cpu"` hardcodé dans le singleton Embedder local. Colab utilise CUDA sans restriction.

## D14 — Cloudflare Tunnel en remplacement de ngrok (2026-04-16)
**Pourquoi :** ngrok gratuit a trop de frictions pour un usage répété depuis Colab :
- URL random à chaque redémarrage → `NGROK_URL` à recopier dans le notebook à chaque session
- Rate limit 40 req/min pénalise les batchs d'upsert
- Page d'avertissement navigateur forçait un header `ngrok-skip-browser-warning`
- Session qui expire après quelques heures d'inactivité
- `ngrok start --all` (2 tunnels simultanés) nécessite un plan payant ou des reserved domains

**Choix :** Cloudflare Tunnel (named tunnel, token dans `.env`) — gratuit, URL stable (hostname personnalisé), pas de rate limit, pas de page d'avertissement, un seul tunnel suffit (FastAPI relaie vers Qdrant).

**Architecture de config :**
- `.env` (gitignored) : `CLOUDFLARE_TUNNEL_TOKEN` + `DOCFINDER_PUBLIC_URL`
- `start.sh` lit `.env`, lance `cloudflared tunnel run --token "$TOKEN"` en arrière-plan si le token existe
- `/admin/tunnels` retourne `{"provider": "cloudflare"|"ngrok"|None, "healthy": bool, "tunnels": {…}}` — détecte cloudflared via `http://localhost:8081/ready` (métriques), fallback ngrok sur le port 4040

**Sécurité :** Le token ne doit jamais apparaître en argv (visible dans `ps`). `start.sh` le lit via `source .env` puis passe `"$CLOUDFLARE_TUNNEL_TOKEN"` — c'est acceptable car `ps aux` sur macOS ne montre pas les env vars d'un autre user. Pour un durcissement supplémentaire, utiliser `cloudflared service install` qui stocke le token dans le keychain système.

**Rétrocompat :** ngrok continue de fonctionner si `DOCFINDER_PUBLIC_URL` n'est pas défini.
