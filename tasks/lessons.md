# Leçons apprises

| Date | Ce qui a mal tourné | Règle pour l'éviter |
|------|---------------------|---------------------|
| 2026-04-14 | — | Projet initialisé, aucune leçon encore |
| 2026-04-15 | Opérations bloquantes (psutil, extraction PDF, YAKE) dans l'event loop FastAPI → freeze du streaming HTTP | Toujours wrapper les appels bloquants avec `await loop.run_in_executor(None, fn, *args)` dans les endpoints async |
| 2026-04-15 | sentence-transformers sur Apple Silicon tentait d'utiliser MPS (GPU Metal) → instabilité + consommation mémoire partagée | Forcer `device="cpu"` dans le singleton Embedder local ; ne pas laisser le framework choisir automatiquement sur macOS |
| 2026-04-15 | Hook `block_main_branch.py` bloque les Edit/Write sur la branche `main` | Créer une branche feature (`git checkout -b docs/...`) avant toute modification, même pour la doc |
| 2026-04-15 | `local_indexer.py` n'insère pas `abs_path` dans le payload — divergence avec server/indexer.py | Quand deux modules produisent le même format Qdrant, extraire les constantes de payload dans shared/schema.py ou documenter explicitement les différences dans DECISIONS.md |
| 2026-04-16 | Token Cloudflare Tunnel partagé en clair dans une conversation IA | Les secrets (tokens, API keys) doivent **uniquement** passer par `.env` gitignored. Ajouter `.gitignore` dès la création du repo. Si un token a été exposé : le révoquer/rotater immédiatement sur le dashboard concerné. |
| 2026-04-16 | ngrok free : URL random + rate limit + page d'avertissement + session expirée | Pour tout tunnel récurrent, préférer Cloudflare Tunnel named (URL stable, gratuit, pas de limite). ngrok reste utile pour du one-shot rapide. |
| 2026-04-16 | Reprise doc-level + flush mid-doc = silent data loss (chunks manquants dans un doc marqué "indexé") | Flush atomique à la frontière de document. Un doc est soit 100 % en Qdrant soit 0 %, jamais entre les deux. |
| 2026-04-16 | `wait=False` sur Qdrant `upsert` → ACK avant persistence → perte silencieuse si crash Qdrant | Utiliser `wait=True` pour les indexations critiques. Le coût perf (~5-10 %) est négligeable comparé à une perte de données. |
| 2026-04-16 | YAKE (pur Python) sur CPU Mac faible = goulot d'étranglement | Toujours déporter les calculs CPU-heavy vers la ressource la plus performante. Bande passante internet ≫ CPU Mac âgé. |
| 2026-04-16 | `fitz.open()` sans context manager = fuite mémoire sur exception | Toujours utiliser `with fitz.open()` pour les ressources externes (PDF, DB connections, file handles). |
| 2026-04-16 | `transformers ≥ 4.50` bloque `torch.load` dès `torch < 2.6` (CVE-2025-32434) → BGE-M3 inchargeable sur Intel Mac (x86_64 plafonné à torch 2.2.2, support droppé) | Pinner `transformers < 4.50` tant que l'on reste sur Intel Mac, ou migrer vers un environnement ARM/Linux si l'on veut bénéficier des versions récentes. À terme : faire porter l'encodage des requêtes à Colab aussi (Mac reste stateless). |
