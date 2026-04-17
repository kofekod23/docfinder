# Phase A — Configuration du 2ᵉ tunnel Cloudflare (`/encode`)

> Objectif : permettre au Mac (stateless) de déporter l'encodage BGE-M3 des
> requêtes vers Colab (T4, fp16). Sans ça, `/search` v2 tombe en `503` car le
> Mac Intel x86_64 ne peut pas charger BGE-M3 (CVE-2025-32434 impose torch ≥ 2.6,
> absent de x86_64).

Flux :

```
UI Mac  →  /search (uvicorn Mac)  →  RemoteEncoder httpx  →  tunnel CF  →  Colab :8001 /encode
                                                                              ↓
                                                                      BGE-M3 GPU fp16
```

## 1. Générer le secret partagé

Sur le Mac :

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

→ c'est la valeur de `COLAB_QUERY_TOKEN`. La **même** doit être utilisée côté
Mac et côté Colab.

## 2. Créer le 2ᵉ tunnel Cloudflare (dashboard Zero Trust)

1. Dashboard → **Networks → Tunnels → Create a tunnel** (type `cloudflared`)
2. Nom : `docfinder-query-encoder`
3. **Public hostname** :
   - Subdomain : `encode`
   - Domain : `jinkohub.digital` (ou ton domaine)
   - Service : `HTTP://localhost:8001`
4. Récupère le token du tunnel (ligne `cloudflared tunnel run --token …`).
5. **Access application** (recommandé, optionnel) : protéger `encode.*` avec un
   service token (CF-Access-Client-Id + Secret) — les variables
   `COLAB_ENCODE_CF_ACCESS_ID/SECRET` dans `.env` gèrent l'auth. Sans Access,
   seul `COLAB_QUERY_TOKEN` protège l'endpoint.

## 3. Variables d'environnement

### Côté Mac — `.env` (à la racine du repo)

```bash
USE_V2=1
COLAB_ENCODE_URL=https://encode.jinkohub.digital
COLAB_QUERY_TOKEN=<secret généré à l'étape 1>
# optionnel si Access est activé :
COLAB_ENCODE_CF_ACCESS_ID=<client-id>
COLAB_ENCODE_CF_ACCESS_SECRET=<client-secret>
```

### Côté Colab — cellule avant lancement

```python
import os
os.environ["MAC_BASE_URL"]       = "https://docfinder.jinkohub.digital"
os.environ["DOCFINDER_ROOT"]     = "/Users/julien/Documents"
os.environ["COLAB_QUERY_TOKEN"]  = "<même secret qu'étape 1>"
```

Puis lancer Colab avec les deux tunnels `cloudflared` :

```bash
# tunnel #1 (indexation Mac → Colab) déjà en place : docfinder.jinkohub.digital
# tunnel #2 (query encoder Colab → Mac) à lancer sur la VM Colab :
cloudflared tunnel run --token <token-tunnel-#2>
```

Puis exécuter `colab_helpers_cell.py` dans un seul cellule Colab.

## 4. Relancer le serveur Mac

```bash
# depuis /Users/julien/docfinder
pkill -f "uvicorn server.main" || true
bash start.sh   # ou ton script de lancement habituel
```

Vérifier dans les logs : `RemoteEncoder configured → <COLAB_ENCODE_URL>`.

## 5. Smoke test

```bash
# 1. depuis le Mac, vérifier que le tunnel Colab répond :
curl -s https://encode.jinkohub.digital/healthz
# → {"status":"ok","model":"bge-m3"}

# 2. test d'auth (doit renvoyer 401 sans token) :
curl -s -X POST https://encode.jinkohub.digital/encode \
     -H "Content-Type: application/json" \
     -d '{"queries":["bonjour"]}'
# → {"detail":"invalid token"}

# 3. test d'encodage :
curl -s -X POST https://encode.jinkohub.digital/encode \
     -H "Content-Type: application/json" \
     -H "X-Auth-Token: $COLAB_QUERY_TOKEN" \
     -d '{"queries":["bonjour"]}' | jq '.dense[0] | length'
# → 1024

# 4. search v2 end-to-end :
curl -s -X POST http://localhost:8000/search \
     -H "Content-Type: application/json" \
     -d '{"query":"architecture projet","k":5}' | jq '.results | length'
```

## 6. Validation §15 (plan v2)

Sur corpus de 20 docs (cross-lingual FR/EN, au moins un PDF image) :

- [ ] recherche FR → retourne docs EN pertinents (et inversement)
- [ ] les `keywords` top-3 retournés sont sensés (plus pertinents que YAKE v1)
- [ ] bumper `mtime` d'un fichier → le chunk est ré-indexé (cf. pending task
      « Ré-indexation des fichiers modifiés » dans `todo.md`)
- [ ] latence p50 `/search` < 1.5 s (encodage remote inclus)
- [ ] aucun fallback silencieux : Colab down ⇒ `/search` renvoie 503 clair

## 7. Commit une fois vert

```bash
git add -A
git commit -m "chore(v2): dry-run 20 docs OK — critères §15 passent"
```

## Dépendances de cette phase

- `colab/query_server.py` — FastAPI `/encode` (fait)
- `colab_helpers_cell.py` — uvicorn thread + pipeline main (fait)
- `server/encode_client.py` — `RemoteEncoder` httpx (fait)
- `server/search.py:168-191` — routage local/remote via `COLAB_ENCODE_URL` (fait)
- `.env.example` — 4 variables déjà documentées (fait)

## Phase B (après 1 semaine de validation)

- Retirer `sentence-transformers`, `torch` de `requirements.txt`
- Retirer le chemin v1 (`SearchEngine.search()` + YAKE + `Embedder` singleton)
- `DECISIONS.md` — ajouter D16 « Query encoding déporté sur Colab »
