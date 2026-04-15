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

## Complété (2026-04-15) — plan 2026-04-15-db-interface-keyphrases-doc-preview

- [x] **T1** — `search.py:_best_excerpt()` — sélection d'extrait par score de mots-clés (remplace troncature 300 chars)
- [x] **T2** — Endpoint `/admin/db` — vue admin collection Qdrant (stats, points, scroll)
- [x] **T3** — Endpoint `/doc/preview` — prévisualisation de document via abs_path
- [x] **T4** — Endpoint `/doc/open` — ouverture du fichier source dans l'app macOS (open)
- [x] **T5** — `abs_path` dans le payload Qdrant (server/indexer.py + server/chunks.py)
- [x] Admin UI — bouton Kill (annulation indexation) + bargraph ressources (psutil)
- [x] `server/indexer.py` — bulk upsert par batch + reprise + nettoyage mémoire GPU
- [x] `server/chunks.py` — run_in_executor pour toutes les opérations bloquantes (streaming réel)
- [x] `shared/embedder.py` — device=cpu forcé pour libérer le GPU Metal (Apple Silicon)
- [x] Colab — statut connecté/GPU + pause/reprise + heartbeat protocol
- [x] Commentaires enrichis — docstrings architecture dans tous les modules clés
- [x] `DECISIONS.md` — D8 à D13 ajoutées

## À faire (optionnel)

- [ ] Tests unitaires (pytest) pour search.py et embedder.py
- [ ] Aligner `local_indexer.py` pour inclure `abs_path` dans le payload (harmoniser avec server/indexer.py)
- [ ] Filtre par type de document dans l'UI
- [ ] Pagination des résultats
- [ ] Endpoint DELETE /document/{doc_id} pour supprimer de l'index
- [ ] Support .odt et .rtf
