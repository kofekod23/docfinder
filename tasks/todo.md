# DocFinder — État du projet

## Complété (2026-04-14)

- [x] Arborescence du projet créée
- [x] `requirements.txt` — dépendances versionnées
- [x] `shared/schema.py` — modèles Pydantic (DocumentChunk, SearchResult, SearchQuery)
- [x] `shared/embedder.py` — singleton Embedder, chargement unique au démarrage
- [x] `setup_qdrant.py` — init collection dense + sparse, flag --force
- [x] `server/search.py` — moteur hybride dense+sparse, RRF, 100% local
- [x] `server/main.py` — FastAPI avec lifespan, GET / et POST /search
- [x] `server/templates/index.html` — UI dark mode, badges type/score/keywords
- [x] `colab_indexer.ipynb` — pipeline complet PDF/Word/txt/md → Qdrant via ngrok
- [x] `README.md` — installation Qdrant natif, démarrage serveur, usage Colab
- [x] `DECISIONS.md` — 7 décisions techniques documentées

## À faire (optionnel)

- [ ] Tests unitaires (pytest) pour search.py et embedder.py
- [ ] Filtre par type de document dans l'UI
- [ ] Pagination des résultats
- [ ] Endpoint DELETE /document/{doc_id} pour supprimer de l'index
- [ ] Support .odt et .rtf
