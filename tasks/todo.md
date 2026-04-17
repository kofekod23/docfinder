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

## En cours — Pipeline Colab v2 (Task 18 Phase A)

Code plumbing terminé côté Mac + Colab :
- [x] `colab/query_server.py` — FastAPI `/encode` + `set_wrapper()` pour éviter double-chargement GPU
- [x] `server/encode_client.py` — `RemoteEncoder` httpx (fail-fast, pas de fallback v1)
- [x] `server/search.py` — routage local/remote via `COLAB_ENCODE_URL`
- [x] `colab_helpers_cell.py` — uvicorn thread daemon + `asyncio.run(run_pipeline)` en main
- [x] `.env.example` — 4 variables ajoutées

Reste à faire (action utilisateur requise) :
- [ ] Générer `COLAB_QUERY_TOKEN` et le poser côté Mac + Colab
- [ ] Créer le 2ᵉ tunnel Cloudflare nommé → `http://localhost:8001`
- [ ] Redémarrer uvicorn Mac
- [ ] Smoke test `/healthz` + `/encode` + `/search`
- [ ] Valider §15 (cross-lingual, keywords, mtime, latence p50 < 1.5s) sur 20 docs
- [ ] Commit `chore(v2): dry-run 20 docs OK`

Procédure détaillée : `tasks/setup-phase-a-query-tunnel.md`.

## Dette sécurité (hors-scope, à traiter plus tard)

- `.env.example` a été scrubbé (placeholders vides), mais les anciennes valeurs
  (JWT tunnel CF + CF-Access-Client-Id/Secret) restent dans l'historique git.
  À rotater dès que la Phase A sera validée : régénérer tunnel token + service
  token CF Access, puis mettre à jour `.env` local et les secrets Colab.

## À faire (optionnel)

- [ ] Tests unitaires (pytest) pour search.py et embedder.py
- [ ] Aligner `local_indexer.py` pour inclure `abs_path` dans le payload (harmoniser avec server/indexer.py)
- [ ] Filtre par type de document dans l'UI
- [ ] Pagination des résultats
- [ ] Endpoint DELETE /document/{doc_id} pour supprimer de l'index
- [ ] Support .odt et .rtf

## À faire — Ré-indexation des fichiers modifiés

Problème : le `doc_id = md5(chemin)` ne tient pas compte du contenu ni de la date de modification.
Un fichier modifié depuis la dernière indexation est silencieusement skippé (ses anciens chunks restent dans Qdrant).

Plan (3 fichiers) :
- [x] `server/chunks.py` — ajouter `mtime = int(file_path.stat().st_mtime)` dans le payload de chaque chunk
- [x] `server/admin_v2.py` — `/admin/indexed-state` retourne `{doc_id: mtime}` (implémenté dans admin_v2)
- [x] `colab/pipeline.py` — compare `mtime` actuel vs Qdrant, ré-indexe si différent
- [x] `server/admin_v2.py` — endpoint `DELETE /admin/doc/{doc_id}` (supprime tous les points d'un doc)
