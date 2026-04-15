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
- ngrok (pour l'indexation depuis Colab uniquement)

---

## Démarrage en 3 étapes

### 1. Lancer le serveur local

```bash
pip install -r requirements.txt
bash start.sh
```

`start.sh` télécharge Qdrant si absent, initialise la collection et démarre le serveur FastAPI sur le port 8000. Pour réinitialiser la collection : `bash start.sh --reset`.

### 2. Indexer depuis Google Colab

1. Sur le Mac, démarrez le tunnel : `cloudflared tunnel --url http://localhost:8000`.
   L'URL affichée (ex. `https://xxx.trycloudflare.com`) est **stable** — pas besoin de la changer à chaque restart.
2. Ouvrez `colab_indexer.ipynb` dans Google Colab (**Runtime → T4 GPU**).
3. Renseignez l'URL tunnel dans la cellule **Config** du notebook (`DOCFINDER_URL`).
4. **Run All** — le daemon démarre automatiquement.
5. Dans l'interface admin (`http://localhost:8000/admin`), cliquez **Lancer l'indexation**.

### 3. Rechercher

Ouvrez **http://localhost:8000** et tapez une requête en langage naturel.

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
├── start.sh              # Démarrage en une commande (Qdrant + serveur)
├── colab_indexer.ipynb   # Notebook d'indexation (Colab GPU)
├── setup_qdrant.py       # Initialisation de la collection Qdrant
├── requirements.txt      # Dépendances Python versionnées
├── DECISIONS.md          # Choix techniques documentés
├── server/
│   ├── main.py           # Application FastAPI
│   ├── search.py         # Moteur de recherche hybride
│   ├── indexer.py        # Indexation locale (thread daemon)
│   ├── chunks.py         # Flux NDJSON pour Colab
│   └── templates/
│       └── index.html    # Interface de recherche (Jinja2)
└── shared/
    ├── schema.py         # Modèles Pydantic
    └── embedder.py       # Wrapper sentence-transformers (singleton)
```
