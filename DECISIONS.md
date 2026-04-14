# Décisions techniques — DocFinder

## D1 — Modèle d'embedding : `paraphrase-multilingual-mpnet-base-v2`
**Pourquoi :** Multilingue (fr/en/etc.), 768 dims, bon équilibre qualité/taille.  
**Alternative écartée :** `all-MiniLM-L6-v2` (anglais uniquement), `multilingual-e5-large` (trop lourd pour CPU).  
**Conséquence :** `EMBEDDING_DIM = 768` dans tout le projet.

## D2 — Singleton Embedder au démarrage FastAPI
**Pourquoi :** Charger un modèle sentence-transformers prend 3-5s et ~500 MB RAM. On le charge une seule fois via `lifespan` FastAPI.  
**Règle stricte :** Aucun appel externe au moment de la recherche — modèle local, Qdrant local.

## D3 — Sparse vectors via YAKE + hash MD5
**Pourquoi :** YAKE extrait des n-grammes statistiques sans supervision. Le hash MD5 % 2^20 mappe les tokens vers un espace sparse fixe, compatible Qdrant.  
**Alternative écartée :** BM25 complet (nécessite un index inversé séparé) ; TF-IDF (besoin du corpus entier).  
**Limite connue :** Collisions de hash possibles (~1/1M), acceptables pour ce cas d'usage.

## D4 — Reranking RRF (Reciprocal Rank Fusion)
**Pourquoi :** RRF est robuste, ne nécessite pas de calibration des scores, fonctionne bien pour fusionner dense + sparse.  
**Formule :** `score = Σ 1/(k + rank_i)` avec `k=60` (valeur standard).

## D5 — Chunking dans Colab uniquement (pas au moment de la recherche)
**Pourquoi :** Le serveur local ne dispose pas des documents. L'indexation batch (GPU Colab) est la seule source de vérité dans Qdrant.

## D6 — Connexion Colab → Qdrant local via ngrok
**Pourquoi :** Qdrant tourne en binaire natif sur la machine locale, inaccessible directement depuis Colab.  
**Setup :** L'utilisateur lance `./ngrok http 6333` sur sa machine, colle l'URL HTTPS dans le notebook.

## D7 — Chunking par paragraphes avec overlap
**Pourquoi :** Préserve le contexte sémantique. Taille cible : 512 tokens (~400 mots), overlap 50 mots.  
**Raison :** `paraphrase-multilingual-mpnet-base-v2` a une fenêtre de 512 tokens max.
