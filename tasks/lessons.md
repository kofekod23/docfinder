# Leçons apprises

| Date | Ce qui a mal tourné | Règle pour l'éviter |
|------|---------------------|---------------------|
| 2026-04-14 | — | Projet initialisé, aucune leçon encore |
| 2026-04-15 | Opérations bloquantes (psutil, extraction PDF, YAKE) dans l'event loop FastAPI → freeze du streaming HTTP | Toujours wrapper les appels bloquants avec `await loop.run_in_executor(None, fn, *args)` dans les endpoints async |
| 2026-04-15 | sentence-transformers sur Apple Silicon tentait d'utiliser MPS (GPU Metal) → instabilité + consommation mémoire partagée | Forcer `device="cpu"` dans le singleton Embedder local ; ne pas laisser le framework choisir automatiquement sur macOS |
| 2026-04-15 | Hook `block_main_branch.py` bloque les Edit/Write sur la branche `main` | Créer une branche feature (`git checkout -b docs/...`) avant toute modification, même pour la doc |
| 2026-04-15 | `local_indexer.py` n'insère pas `abs_path` dans le payload — divergence avec server/indexer.py | Quand deux modules produisent le même format Qdrant, extraire les constantes de payload dans shared/schema.py ou documenter explicitement les différences dans DECISIONS.md |
