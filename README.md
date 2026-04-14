# DocFinder

Moteur de recherche hybride de documents personnels — dense + sparse avec reranking RRF.

## Architecture

```
                   ┌─────────────────────────────────────────┐
  Indexation       │  Google Colab (GPU)                     │
  (batch)          │  colab_indexer.ipynb                    │
                   │  PDF/Word/txt/md → chunks → embeddings  │
                   └──────────────────┬──────────────────────┘
                                      │ REST via ngrok
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
- ngrok (pour l'indexation depuis Colab uniquement)

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

## 4. Indexation des documents (Colab)

1. Démarrez ngrok sur votre machine locale :
   ```bash
   ./ngrok http 6333
   ```
2. Copiez l'URL HTTPS fournie par ngrok (ex. `https://abc123.ngrok-free.app`).
3. Ouvrez `colab_indexer.ipynb` dans Google Colab (**Runtime → T4 GPU**).
4. Renseignez `NGROK_URL` et `DRIVE_DOCS_PATH` dans la cellule de configuration.
5. Montez votre Google Drive et exécutez toutes les cellules.

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
