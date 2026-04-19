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

## Suite grid-search (2026-04-18) — leviers matériels, à prioriser

Le grid 41 × 145 plafonne à MRR 0.7513 (+0.0063 vs baseline = bruit). Les 19 requêtes ratées
ne seront pas sauvées par du tuning de poids. Pistes ROI, par ordre d'impact attendu :

- [x] **Reranker cross-encoder** — bge-reranker-v2-m3 sur top-N → réordonne head.
      Plan 1 implémenté (branch `claude/review-project-cloudflare-ggcYD`, 12 commits).
      Feature flag `RERANK_ENABLED=false` par défaut. Mesures A/B (145 queries) en attente
      d'un tunnel Colab vivant côté utilisateur.
- [x] **Qwen3-Embedding A/B** — Plan 2 (2026-04-18) implémenté, 7 tasks ✅, final holistic
      review APPROVED (reviewer python-reviewer, commits `dda01e9..16312f7`, 12 commits).
      Collection parallèle `docfinder_v2_qwen` (1024-d cosine), routage env-driven
      `DOCFINDER_COLLECTION` + `DOCFINDER_EMBEDDER={bgem3,qwen}`. Runner A/B MRR@10 :
      `tasks/qwen_ab.py` sur 145 ground-truth queries. Mesure live restante — voir
      "Plan 2 — mesure A/B live" plus bas.

## Plan 2 — mesure A/B live (à exécuter quand tunnel Colab + uvicorn dispo)

Pré-requis : pipeline Colab GPU avec `DOCFINDER_ENABLE_QWEN=1` et Qdrant joignable.

1. `python scripts/setup_qdrant_qwen.py` (crée `docfinder_v2_qwen` si absent)
2. Ré-indexer via Colab avec `DOCFINDER_ENABLE_QWEN=1` + `DOCFINDER_COLLECTION=docfinder_v2_qwen`
3. Restart uvicorn Mac avec `DOCFINDER_COLLECTION=docfinder_v2 DOCFINDER_EMBEDDER=bgem3`
   puis `python -m tasks.qwen_ab run --mode bgem3 > out_bgem3.json`
4. Restart uvicorn Mac avec `DOCFINDER_COLLECTION=docfinder_v2_qwen DOCFINDER_EMBEDDER=qwen`
   puis `python -m tasks.qwen_ab run --mode qwen > out_qwen.json`
5. `python -m tasks.qwen_ab compare out_bgem3.json out_qwen.json`
   → attend MRR@10 par mode + wins/losses/same.

## Branche `claude/review-project-cloudflare-ggcYD` — statut

Porte deux plans (tous deux shippés) :
- Plan 1 (rerank cross-encoder) — 12 commits (`880031b..dda01e9`), **T1→T6 landés, vérifié via git log + ls + grep 2026-04-18**
- Plan 2 (qwen A/B) — 12 commits (`8480d30..16312f7`), **7/7 tasks COMPLETED, holistic APPROVED**
- Cleanup 2026-04-18 : rarity_factor/rarity_threshold retirés (prouvés no-op, grid 7 variantes → MRR identique 0.7513).

## Plan 2 — mesure A/B live Qwen (2026-04-19 13:25) — BLOQUÉ

**État live (probe Mac depuis .env) :**
- `/healthz` → **530** (tunnel CF vivant mais origine Colab KO — error code 1033)
- `/rerank` → **530** (idem)
- `/encode_qwen` → **530** (idem)

Tunnel Cloudflare `encode.jinkohub.digital` ne joint plus aucun endpoint : l'instance Colab GPU a décroché (runtime idle kill / session expirée / notebook fermé). Aucun A/B ne peut tourner tant que l'utilisateur n'a pas redémarré le notebook Colab + le tunnel associé.

Fix `4a4df2e` toujours à vérifier après re-clone : `SentenceTransformer.encode()` n'accepte pas `max_length` kwarg, pilotage via `max_seq_length` attribut. Colab doit re-clone + restart (cells 2 → 3 → 6) puis re-probe `/encode_qwen` → attendu **200** avec embeddings 1024-d.

Étape par étape :
- [x] `scripts/setup_qdrant_qwen.py --variant 0.6B` → collection `docfinder_v2_qwen` créée (dim=1024, cosine).
- [x] **Leg bgem3** : `tasks/ab_out/out_bgem3.json` landé via `python -m tasks.qwen_ab run --mode bgem3`.
  - MRR@10 = **0.7159** (122/145 hits, 1 erreur transient) — cohérent avec `rerank_off.csv` (0.7179).
  - uvicorn PID 39316 avec USE_V2=1 (défauts = bgem3 + docfinder_v2).
- [ ] **Leg qwen — en attente re-clone Colab** :
  - `docfinder_v2_qwen` toujours vide (0 points) → ré-indexation Colab requise après fix.
  - `/encode_qwen` 500 → attend commit `4a4df2e` chargé sur Colab.
  - Après re-run cells 2/3/6 : re-probe, puis ré-indexer, puis `--mode qwen`.
- [ ] `python -m tasks.qwen_ab compare tasks/ab_out/out_bgem3.json tasks/ab_out/out_qwen.json`
  → attend MRR@10 par mode + wins/losses/same. Seuil décision : ΔMRR ≥ +0.01 pour adopter Qwen.

Pour débloquer le leg qwen :
1. Notebook Colab : charger le wrapper Qwen3-Embedding 0.6B et exposer `/encode_qwen` sur le tunnel `encode.jinkohub.digital` (à côté de `/encode` BGE-M3 existant).
2. Mac, health-check :
   `curl -X POST https://encode.jinkohub.digital/encode_qwen -H "X-Auth-Token: $COLAB_QUERY_TOKEN" -H "CF-Access-Client-Id: $CF_ACCESS_CLIENT_ID" -H "CF-Access-Client-Secret: $CF_ACCESS_CLIENT_SECRET" -H "User-Agent: DocFinder/1.0" -d '{"queries":["test"]}'` → doit retourner 200 avec embeddings 1024-d.
3. Ré-indexation Colab vers `docfinder_v2_qwen` (env: `DOCFINDER_ENABLE_QWEN=1 DOCFINDER_COLLECTION=docfinder_v2_qwen`), attendre `points_count` ≈ 2264 (même corpus que `docfinder_v2`).
4. Restart uvicorn Mac : `DOCFINDER_COLLECTION=docfinder_v2_qwen DOCFINDER_EMBEDDER=qwen COLAB_ENCODE_URL=https://encode.jinkohub.digital`.
5. `python -m tasks.qwen_ab run --mode qwen > tasks/ab_out/out_qwen.json` puis compare.

## Mesure A/B live Plan 1 — TERMINÉE (2026-04-19 14:13) — NO-MERGE de l'activation

**Résultat final** :
- **Baseline `rerank_off`** (port 8000, run #3 de la nuit) : MRR@10 = **0.7179**, 123/145 hits, 0 erreur → `tasks/ab_out/rerank_off.csv`.
- **Variant `rerank_on`** (port 8765, run 14:13) : MRR@10 = **0.6182**, 115/145 hits, 0 erreur → `tasks/ab_out/rerank_on.csv`.
- **Compare** (`python -m tasks.rerank_ab --compare`) : `delta: -0.0996`, `wins: 17   losses: 37   same: 91` → archivé dans `tasks/ab_out/compare_on_vs_off.txt`.

**Décision** : seuil `ΔMRR ≥ +0.01` non atteint (écart négatif de −0.10). **NO-MERGE de l'activation** : le code Plan 1 reste en place (12 commits déjà landés), `RERANK_ENABLED=false` reste le défaut — aucune régression pour l'utilisateur final, flag toujours disponible pour un retune futur du reranker (top-N plus petit, ou reranker différent, ou fusion pondérée des scores au lieu du remplacement complet).

**Pourquoi le reranker dégrade** (hypothèses à creuser si Plan 1 est réouvert un jour) :
- BGE-reranker-v2-m3 est entraîné sur paires (query, passage) courtes — nos chunks DocFinder sont longs (2000 chars) et multi-sujets.
- `rerank_results()` **remplace** l'ordre de fusion par l'ordre reranker (pas une combinaison pondérée) → perd tout le signal filename/title boost du pipeline amont.
- 91/145 requêtes inchangées (cas où top-1 de la fusion RRF est déjà une évidence lexicale) ; parmi les 54 modifiées : 37 losses vs 17 wins = le reranker se trompe 2× plus souvent qu'il ne corrige.

Runs pollués archivés : `tasks/ab_out/rerank_off.polluted{,.2}.{csv,log}` (Cloudflare 530 de la nuit) + `tasks/ab_out/rerank_on.polluted.{csv,log}` (run avec uvicorn down = 145 Connection refused — avant de passer sur port 8765).

### Dette tech (hors Plan 1)

- **Mystère des uvicorn kills résolu** (2026-04-19 16:30) : deux instances fantômes de `docfinder/start.sh` tournaient depuis vendredi matin (PID 87848/87760, ELAPSED 3 jours). Le script contient une boucle `while true` avec `lsof -ti tcp:8000 | xargs kill -9` toutes les 30s (fonction watchdog). Elles tuaient l'uvicorn du LaunchAgent et de toute autre source sur :8000 toutes les 30s → cycle 34s observé (30s + 10s throttle + ~4s startup). Après `kill 87848 87760`, uvicorn LaunchAgent a survécu 90s+ sans interruption, tunnel 200 confirmé. Cf. lessons.md L#30 pour diagnostic complet. Le `/health` résiduel vu dans les logs = heartbeat légitime de `colab_indexer.ipynb`, pas le killer. **Dette ouverte** : ajouter un `.pid` lock file à `start.sh` pour éviter instances dupliquées.

### LaunchAgent macOS — procédure d'install (action user, une seule fois)

Fichiers préparés (`scripts/digital.jinkohub.docfinder.plist` + `scripts/install_launchagent.sh`). Le plist lance `env USE_V2=1 python -m uvicorn server.main:app --host 0.0.0.0 --port 8000` avec Artefact pyenv, recharge `.env`, redémarre sur crash (KeepAlive.Crashed=true, ThrottleInterval=10s).

Commande à lancer une fois :
```bash
bash scripts/install_launchagent.sh
```

Vérification : `launchctl list | grep docfinder` → doit retourner une ligne avec PID actif.
Logs : `tail -f /tmp/docfinder_uvicorn.log`.
Stop : `launchctl unload ~/Library/LaunchAgents/digital.jinkohub.docfinder.plist`.
Restart manuel : `launchctl kickstart -k gui/$(id -u)/digital.jinkohub.docfinder`.

Après install, le tunnel `docfinder.jinkohub.digital` retourne 200 en permanence (au login, au reboot, après crash). Plus besoin de relancer uvicorn à la main.

Options suite ouvertes :
(a) merger la branche (Plan 1 code + Plan 2 code, tous deux neutres par défaut via leurs flags respectifs), (b) lancer la mesure A/B live Qwen (Plan 2) quand le notebook Colab Qwen sera réexposé, (c) grid ciblé sur les 19 misses persistantes (re-OCR, dictionnaire d'alias, etc.).
## Campagne OCR agressif (2026-04-19 16:45) — code livré, à déployer

**Cible** : 13/22 requêtes ratées appartiennent à des scans PDF (attestation-ameli, prévoyance axa, URSSAF x2, solde-de-tout-compte, fiches-paie Yakarouler, télétravail x3, courrier omp stationnement, résiliation parking x2, cle-de-secours, questionnaire lycée). Analyse : `tasks/ab_out/rerank_off.csv` ∩ `tasks/queries_semantic.py`.

**Fix livré** (`colab/extractor.py`) :
- Seuil `PAGE_OCR_MIN_CHARS` relevé de **10 → 100 chars** (une page avec juste un en-tête type "Page 1 / Mairie de Paris" était épargnée à tort avant).
- Filet de sécurité doc-level : si texte total < `DOC_OCR_FALLBACK_CHARS` (300) sur un PDF non trivial, **OCR forcée sur toutes les pages** non encore traitées.
- OCR reste lazy via `colab/ocr.py` (easyocr fr+en, GPU) — aucun coût si non déclenchée.

**À faire pour mesurer l'impact** :

1. Côté Colab (notebook) : `!cd /content/docfinder && git pull` pour récupérer le nouveau `extractor.py`. Si Colab a les changements locaux non synchronisés → `git stash && git pull && git stash pop`.
2. Vider le checkpoint pour re-extraire les scans : soit (a) `rm /content/checkpoint_v2.json` + `DOCFINDER_FORCE_REINDEX=1` sur un run, soit (b) supprimer les docs suspects via `DELETE /admin/doc/{doc_id}` ciblé. Option (a) = ~2000 docs ré-extraits (30min-2h GPU), option (b) = ~15 docs ciblés (5min).
3. Relancer cell 6 — le pipeline auto-retry redémarre avec le nouvel extracteur.
4. Une fois indexation terminée : `python -m tasks.rerank_ab --mode off --out tasks/ab_out/rerank_off_v2.csv` (Artefact pyenv).
5. `python -m tasks.rerank_ab --compare tasks/ab_out/rerank_off.csv tasks/ab_out/rerank_off_v2.csv` → attendu : ΔMRR > +0.03 si les scans étaient effectivement la cause.

**Seuil décision** : ΔMRR ≥ +0.02 → merge extractor.py dans la branche. Sinon investiguer pourquoi OCR easyocr n'a pas sorti assez de texte utile (prévoir passage à docTR).
- [ ] **Dictionnaire d'expansion perso** — table d'alias manuelle (Roy→chien, Yakarouler→employeur,
      CPAM→sécurité sociale…). Appliqué côté /search avant encoding. Seul moyen de rattraper
      les requêtes dont la sémantique nécessite un contexte biographique.
- [ ] **Supprimer `rarity_factor` / `rarity_threshold`** de `server/search.py`
      (lignes 406-407, 439-445, 481-482, 494). Prouvé no-op sur 7 variantes du grid → MRR
      identiques à 4 décimales. Ne pas faire pendant une indexation active (restart uvicorn requis).
- [ ] **Grid ciblé sur les 19 misses uniquement** après chaque nouveau levier implémenté
      pour mesurer l'impact sans bruit du reste du corpus.

## À faire — Ré-indexation des fichiers modifiés

Problème : le `doc_id = md5(chemin)` ne tient pas compte du contenu ni de la date de modification.
Un fichier modifié depuis la dernière indexation est silencieusement skippé (ses anciens chunks restent dans Qdrant).

Plan (3 fichiers) :
- [x] `server/chunks.py` — ajouter `mtime = int(file_path.stat().st_mtime)` dans le payload de chaque chunk
- [x] `server/admin_v2.py` — `/admin/indexed-state` retourne `{doc_id: mtime}` (implémenté dans admin_v2)
- [x] `colab/pipeline.py` — compare `mtime` actuel vs Qdrant, ré-indexe si différent
- [x] `server/admin_v2.py` — endpoint `DELETE /admin/doc/{doc_id}` (supprime tous les points d'un doc)
