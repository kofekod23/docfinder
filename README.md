# DocFinder

Moteur de recherche hybride de documents personnels — dense + sparse avec reranking RRF.

## Architecture

```
                   ┌─────────────────────────────────────────┐
  Indexation       │  Google Colab (GPU)                     │
  (batch)          │  colab_indexer.ipynb                    │
                   │  PDF/Word/txt/md → chunks → embeddings  │
                   └──────────────────┬──────────────────────┘
                                      │ REST via Cloudflare Tunnel
                                      ▼
                   ┌─────────────────────────────────────────┐
  Stockage         │  Qdrant local (port 6333)               │
                   │  dense vectors 768d + sparse YAKE       │
                   └──────────────────┬──────────────────────┘
                                      │ localhost
                                      ▼
  Recherche        ┌─────────────────────────────────────────┐
  (temps réel)     │  FastAPI local (port 8000)              │
  100% local       │  modèle chargé 1× au démarrage (CPU)   │
                   │  RRF fusion dense + sparse              │
                   └─────────────────────────────────────────┘
```

**Invariant clé :** la vectorisation des requêtes et la recherche sont entièrement locales. Aucun appel externe au moment de la recherche.

---

## Prérequis

- Python 3.10+
- Qdrant binaire natif (voir ci-dessous)
- cloudflared (pour l'indexation depuis Colab uniquement — voir §4)

---

## 1. Installation de Qdrant (binaire natif)

```bash
# macOS (Apple Silicon)
curl -L https://github.com/qdrant/qdrant/releases/latest/download/qdrant-aarch64-apple-darwin.tar.gz \
  | tar -xz
chmod +x qdrant

# macOS (Intel)
curl -L https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-apple-darwin.tar.gz \
  | tar -xz
chmod +x qdrant

# Linux (x86_64)
curl -L https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-unknown-linux-musl.tar.gz \
  | tar -xz
chmod +x qdrant

# Démarrage (données stockées dans ./qdrant_storage/)
./qdrant
```

Qdrant est accessible sur `http://localhost:6333`. Dashboard : `http://localhost:6333/dashboard`.

---

## 2. Installation du projet

```bash
cd docfinder
pip install -r requirements.txt
```

---

## 3. Initialisation de la collection Qdrant

```bash
# Qdrant doit être démarré (./qdrant)
python setup_qdrant.py

# Pour réinitialiser (supprime les données existantes)
python setup_qdrant.py --force
```

---

## 4. Indexation des documents (Colab via Cloudflare Tunnel)

Cloudflare Tunnel remplace ngrok : URL publique **stable** (pas à recopier à chaque session), pas de limite de débit, pas de page d'avertissement, gratuit.

### Installer cloudflared

```bash
# macOS
brew install cloudflared
# Linux
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o cloudflared
chmod +x cloudflared
```

### Configurer le tunnel (une seule fois)

1. Sur le dashboard Cloudflare → **Zero Trust** → **Networks** → **Tunnels** → *Create a tunnel* → *Cloudflared*.
2. Notez le token affiché (format `eyJ…`).
3. Ajoutez un **Public Hostname** qui pointe vers `http://localhost:8000` (service = HTTP, URL = `localhost:8000`). Ex. `docfinder.mondomaine.com`.
4. Copiez `.env.example` → `.env` et renseignez :
   ```bash
   CLOUDFLARE_TUNNEL_TOKEN=eyJ…
   DOCFINDER_PUBLIC_URL=https://docfinder.mondomaine.com
   ```
   > ⚠ `.env` est **gitignored**. Ne commitez jamais le token.

### Lancer l'indexation

1. `./start.sh` (démarre Qdrant + cloudflared + uvicorn en une commande).
2. Ouvrez `http://localhost:8000/admin` — la section *Tunnel* affiche l'URL publique + un indicateur 🟢 si cloudflared est actif.
3. Ouvrez `colab_indexer.ipynb` dans Google Colab (**Runtime → T4 GPU**).
4. Collez votre `DOCFINDER_URL` dans la cellule de configuration (une seule fois — elle ne change plus).
5. Montez votre Google Drive et exécutez toutes les cellules.

> **Fallback ngrok** : le code détecte encore `ngrok start --all` (API locale port 4040) si aucun `DOCFINDER_PUBLIC_URL` n'est défini.

---

## 5. Démarrage du serveur de recherche

```bash
uvicorn server.main:app --port 8000
```

L'interface de recherche est disponible sur **http://localhost:8000**.

Le modèle `paraphrase-multilingual-mpnet-base-v2` est chargé une seule fois au démarrage (~5s, ~500 MB RAM). Chaque requête de recherche est vectorisée localement sur CPU.

---

## 6. Utilisation

Ouvrez http://localhost:8000 dans votre navigateur et tapez une requête en langage naturel.

La recherche hybride combine :
- **Dense** : similarité sémantique (sentence-transformers)
- **Sparse** : mots-clés YAKE (BM25 maison)
- **RRF** : fusion des deux classements

---

## Formats supportés

| Format | Parseur |
|--------|---------|
| PDF    | pymupdf |
| Word (.docx) | python-docx |
| Texte (.txt) | natif Python |
| Markdown (.md) | natif Python |

---

## Structure du projet

```
docfinder/
├── colab_indexer.ipynb   # Notebook d'indexation (Colab GPU)
├── setup_qdrant.py       # Initialisation de la collection Qdrant
├── requirements.txt      # Dépendances Python versionnées
├── DECISIONS.md          # Choix techniques documentés
├── server/
│   ├── main.py           # Application FastAPI
│   ├── search.py         # Moteur de recherche hybride
│   └── templates/
│       └── index.html    # Interface de recherche (Jinja2)
└── shared/
    ├── schema.py         # Modèles Pydantic
    └── embedder.py       # Wrapper sentence-transformers (singleton)
```
