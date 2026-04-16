# Colab Pipeline v2 — Design Spec

**Date** : 2026-04-16
**Auteur** : Julien (kofekod23) + Claude
**Statut** : Approuvé (en attente de writing-plans)
**Branche cible** : `claude/review-project-cloudflare-ggcYD` (ou branche dédiée `feat/colab-v2`)

---

## 1. Contexte et objectif

Le pipeline d'indexation actuel (`server/chunks.py` + `server/indexer.py` côté Mac + notebook Colab) souffre de plusieurs défauts identifiés par l'utilisateur :

- **Mauvais mots-clés** : YAKE sur des chunks trop petits/mal découpés produit du bruit (dates, stopwords, tokens techniques).
- **Mauvais extraits** : chunks coupés au milieu d'une phrase, sans respect des paragraphes/titres.
- **Cross-lingue défaillant** : un document FR ne remonte pas sur une requête EN (et inversement).
- **Recherche sémantique faible** : le modèle `intfloat/multilingual-e5-large` rate des synonymes et reformulations.
- **Doublons non détectés** : même document ré-indexé plusieurs fois sans comparaison de contenu.
- **PDF images (scannés) silencieux** : l'OCR pytesseract CPU n'est jamais sérieusement stressé.
- **Mac i7 16 Go saturé** : extraction, OCR, YAKE, embeddings tournent tous sur la même machine → latence et instabilité.

**Objectif** : basculer la totalité du calcul lourd sur Colab (T4 GPU, 16 GB VRAM, 5 h d'inactivité), laisser le Mac en pur serveur de fichiers, et produire des vecteurs + mots-clés de qualité suffisante pour rendre la recherche pertinente sur 5 000 documents administratifs (FR) et techniques (FR/EN).

---

## 2. Architecture cible

```
┌──────────────────────────────┐         Cloudflare Tunnel        ┌──────────────────────────────┐
│   MAC (i7, 16 Go) — stateless│  ◀──────── HTTP/NDJSON ────────▶ │   COLAB (T4, 16 GB VRAM)      │
│   ─ serveur de fichiers       │                                  │   ─ pipeline v2 complet       │
│   ─ admin UI + kill + state   │                                  │   ─ download → extract → OCR  │
│                              │                                  │     → chunk → embed → flush   │
│   AUCUN calcul lourd          │                                  │   ─ BGE-M3 (dense+sparse+col) │
│                              │                                  │   ─ easyocr GPU               │
└──────────────────────────────┘                                  └──────────────┬───────────────┘
                                                                                 │
                                                                                 │ upsert
                                                                                 ▼
                                                                  ┌──────────────────────────────┐
                                                                  │   QDRANT (Mac, port 6333)     │
                                                                  │   3 vecteurs nommés + payload │
                                                                  └──────────────────────────────┘
```

**Principe "Rien sur le Mac"** : le Mac ne fait que `stat()`, lecture fichier brute, et héberge Qdrant. Aucun CPU lourd (pas d'extraction PDF, pas de YAKE, pas d'embedding).

---

## 3. Modèle d'embedding — BGE-M3

### 3.1 Choix

**`BAAI/bge-m3`** remplace **`intfloat/multilingual-e5-large`** + **YAKE**.

| Critère                   | e5-large (actuel)           | BGE-M3 (cible)                      |
|---------------------------|------------------------------|--------------------------------------|
| Dimensions dense          | 1024                         | 1024                                 |
| Sparse (lexical)          | ❌                           | ✅ (intégré, remplace YAKE)          |
| Multi-vectoriel (ColBERT) | ❌                           | ✅ (rerank sans modèle séparé)       |
| Multilingue               | bon                          | meilleur (FR/EN natif)               |
| Taille modèle             | ~2.2 GB                      | ~2.3 GB                              |
| VRAM inference fp16       | ~3 GB                        | ~4 GB (3 sorties)                    |

**Gains concrets** :
- Un seul forward → 3 représentations (dense + sparse + multi-vec).
- Les tokens du sparse output **sont** les mots-clés : plus besoin de YAKE.
- MaxSim ColBERT au rerank sans télécharger un cross-encoder.

### 3.2 Paramètres

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device="cuda")
out = model.encode(
    texts,
    batch_size=32,
    max_length=512,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True,
)
# out["dense_vecs"]    : np.ndarray (N, 1024)
# out["lexical_weights"]: list[dict[token_id: weight]]
# out["colbert_vecs"]  : list[np.ndarray (tokens, 1024)]
```

---

## 4. Schéma Qdrant

### 4.1 Collection

Nom : `docfinder_v2` (nouvelle collection, ancienne `docfinder` conservée jusqu'à validation).

```python
client.create_collection(
    collection_name="docfinder_v2",
    vectors_config={
        "dense":   VectorParams(size=1024, distance=Distance.COSINE),
        "colbert": VectorParams(
            size=1024,
            distance=Distance.COSINE,
            multivector_config=MultiVectorConfig(comparator=MultiVectorComparator.MAX_SIM),
        ),
    },
    sparse_vectors_config={
        "sparse": SparseVectorParams(modifier=Modifier.IDF),
    },
)
```

### 4.2 Payload par point (= chunk)

```jsonc
{
  "doc_id":        "sha1(abs_path)",      // stable tant que le fichier n'est pas déplacé
  "path":          "relative/from/root",
  "abs_path":      "/Users/julien/Documents/...",
  "doc_type":      "pdf|docx|md|txt|…",
  "title":         "Titre extrait (ou filename)",
  "mtime":         1713254400,             // epoch seconds, pour diff
  "file_hash":     "sha1(first_1MB+size)", // détection doublons
  "content":       "texte du chunk (400-600 tokens)",
  "keywords_chunk":["token1","token2",…],  // top-10 du sparse output du chunk
  "keywords_doc":  ["token1","token2",…],  // top-15 agrégé au niveau doc, dupliqué sur chaque chunk
  "page_range":    [3, 5],                 // pages sources (PDF uniquement, null sinon)
  "chunk_idx":     7,
  "chunk_total":   42
}
```

### 4.3 Index payload

```python
client.create_payload_index("docfinder_v2", "doc_id",   PayloadSchemaType.KEYWORD)
client.create_payload_index("docfinder_v2", "doc_type", PayloadSchemaType.KEYWORD)
client.create_payload_index("docfinder_v2", "mtime",    PayloadSchemaType.INTEGER)
client.create_payload_index("docfinder_v2", "path",     PayloadSchemaType.KEYWORD)
```

---

## 5. Extraction size-aware

Décidée côté Colab **après download**, basée sur `size` et (pour PDF) `page_count`.

| Condition                                  | Action                                                    |
|--------------------------------------------|-----------------------------------------------------------|
| `size > 50 MB`                             | **Filename-only** : embed titre + path, 1 chunk.          |
| `size > 20 MB` OU (PDF ET `pages > 30`)    | **Tête uniquement** : 5-10 premières pages.               |
| Sinon                                      | **Full** : extraction complète.                           |

Le filtrage `size > 50 MB` est fait côté Mac dans `/files/list` pour ne pas télécharger inutilement.

---

## 6. Chunking par paragraphes

### 6.1 Règles

1. Split sur `\n\n+` → liste de paragraphes bruts.
2. Détection des **titres** (heading markdown `^#{1,6} `, ligne majuscules, numérotation `^\d+\.\s`) → **forced break**.
3. Regroupement glouton : on accumule des paragraphes tant que `token_count < 400`, puis on stoppe dès que :
   - le paragraphe suivant dépasserait `600` tokens, OU
   - on atteint un titre.
4. **Contraintes dures** : `min=100`, `max=800` tokens.
5. **Overlap** : 1 paragraphe entre chunks consécutifs (sauf après un titre).

### 6.2 Tokenizer

Utiliser le tokenizer de BGE-M3 (XLM-RoBERTa) pour compter → cohérent avec `max_length=512` du modèle (marge pour tokens spéciaux).

---

## 7. OCR pour PDF scannés

### 7.1 Déclencheur

Pour chaque page PDF : si `len(page.get_text().strip()) < 10` → déclencher OCR sur cette page uniquement.

### 7.2 Moteur

**`easyocr`** sur GPU T4, `lang=['fr','en']`. Charger le reader une seule fois au démarrage de Colab (~1 GB VRAM, cohabite avec BGE-M3 dans les 16 GB).

```python
import easyocr
reader = easyocr.Reader(['fr','en'], gpu=True)
text = "\n".join(reader.readtext(page_image, detail=0, paragraph=True))
```

### 7.3 Rendu de la page

`page.get_pixmap(dpi=200)` via PyMuPDF → bytes PNG → numpy array pour easyocr.

---

## 8. Mots-clés (remplacent YAKE)

### 8.1 Au niveau chunk

```python
weights = out["lexical_weights"][i]         # dict[token_id: float]
top10 = sorted(weights.items(), key=lambda x: -x[1])[:10]
keywords_chunk = [tokenizer.decode([tid]) for tid, _ in top10]
# Filtres : longueur ≥ 3, pas uniquement numérique, pas stopword FR/EN
```

### 8.2 Au niveau doc

Agréger les `lexical_weights` de tous les chunks du doc (somme pondérée par longueur du chunk), prendre le top-15.

### 8.3 Affichage UI

- Badges `keywords_doc` en en-tête de résultat (jusqu'à 5 visibles + popover).
- `keywords_chunk` mis en gras/surlignés dans l'extrait affiché.

---

## 9. Parallélisme sur Colab

### 9.1 Pipeline asyncio 3 étages

```
┌─────────────┐    queue    ┌──────────────┐    queue    ┌──────────────┐
│ 8 workers   │ ──bytes────▶│ 4 CPU workers│ ──chunks───▶│ 1 GPU worker │
│ HTTP GET    │             │ extract +    │             │ BGE-M3 batch │
│ /files/raw  │             │ OCR + chunk  │             │ 32 fp16      │
└─────────────┘             └──────────────┘             └──────────────┘
                                                                 │
                                                                 ▼
                                                          upsert Qdrant
                                                          (flush/doc, wait=True)
```

### 9.2 Paramètres

- **HTTP workers** : 8 (download concurrent depuis Mac via Cloudflare).
- **CPU workers** : 4 (extraction PyMuPDF/docx + OCR pré-processing).
- **GPU batch** : 32 chunks, fp16.
- **Flush** : atomique par document (`upsert(wait=True)` après tous les chunks d'un doc).
- **Checkpoint** : état `{doc_id: status}` écrit sur disque Colab toutes les 20 docs + copie sur Drive toutes les 100 docs (survie kernel).

### 9.3 Budget VRAM (T4 16 GB)

| Composant             | VRAM estimée |
|-----------------------|--------------|
| BGE-M3 fp16           | ~4 GB        |
| easyocr (fr+en)       | ~1 GB        |
| Batch 32 × 512 tokens | ~3 GB        |
| Activations/overhead  | ~2 GB        |
| **Total**             | **~10 GB**   |
| **Marge**             | ~6 GB        |

Si bascule sur A100/L4 plus tard : augmenter `batch_size` à 64/128.

---

## 10. Rerank à la requête

### 10.1 Flow

```
query ──▶ BGE-M3 (dense + sparse + colbert)
         │
         ├─▶ Qdrant search "dense"  (top-50)
         ├─▶ Qdrant search "sparse" (top-50)
         │
         └─▶ RRF fusion             (top-50)
             │
             └─▶ Qdrant search "colbert" with prefetch=RRF result
                 (MaxSim rerank, top-10)
                 │
                 └─▶ résultats UI
```

### 10.2 Paramètres

- `k` RRF = 60 (défaut standard).
- Top-50 après RRF → top-10 après ColBERT rerank.
- Latence ajoutée : +150-300 ms (acceptable pour qualité).

### 10.3 Implémentation Qdrant

Utiliser `client.query_points()` avec `prefetch=[...]` et `query=NearestQuery(...)` sur le vecteur `colbert` pour chaîner fusion + rerank en une seule requête serveur-side.

---

## 11. Re-indexation et détection de changement

### 11.1 Endpoint Mac

```
POST /admin/indexed-state
Body:  {"doc_ids": ["sha1_1", "sha1_2", ...]}   # optionnel, filtre
Resp:  {"sha1_1": 1713254400, "sha1_2": 1712000000, ...}
```

Renvoie `{doc_id: mtime}` depuis Qdrant (scroll avec projection `doc_id, mtime`, dédupliqué).

### 11.2 Logique Colab

```python
mac_files      = GET /files/list   # {doc_id: {path, size, mtime}}
indexed_state  = POST /admin/indexed-state
to_index       = []
to_reindex     = []
for doc_id, meta in mac_files.items():
    if doc_id not in indexed_state:
        to_index.append(doc_id)
    elif meta["mtime"] > indexed_state[doc_id]:
        to_reindex.append(doc_id)

# Pour to_reindex : DELETE points where doc_id=X, puis indexer normalement
```

### 11.3 Endpoint DELETE

```
DELETE /admin/doc/{doc_id}
```

Supprime tous les points ayant `payload.doc_id == doc_id` dans `docfinder_v2`.

---

## 12. Endpoints Mac (file server stateless)

| Endpoint                         | Méthode | Description                                            |
|----------------------------------|---------|--------------------------------------------------------|
| `/files/list?root=X&exclude=…`   | GET     | `[{path, size, mtime, doc_id}]`, filtre `size>50MB`    |
| `/files/raw?path=X`              | GET     | Bytes bruts, chunked HTTP                              |
| `/admin/indexed-state`           | POST    | `{doc_id: mtime}` depuis Qdrant                        |
| `/admin/doc/{doc_id}`            | DELETE  | Supprime tous les chunks d'un doc                      |
| `/admin/kill`                    | GET     | Inchangé — tue l'indexation en cours                   |
| `/admin/progress`                | GET     | Lit l'état pushé par Colab (voir §13)                  |

**Ce que le Mac ne fait pas** : extraction, OCR, chunking, embedding, mots-clés.

---

## 13. Observabilité

### 13.1 Push depuis Colab

Colab POST `/admin/progress` toutes les 5 s avec :

```jsonc
{
  "total":            4987,
  "done":             1243,
  "failed":           12,
  "current_doc":      "Documents/Admin/contrat_2024.pdf",
  "gpu_util_pct":     87,
  "vram_used_mb":     9420,
  "chunks_per_sec":   34.2,
  "eta_seconds":      6100,
  "stage_counts":     {"downloaded": 1250, "extracted": 1248, "embedded": 1243}
}
```

### 13.2 Admin UI

- Barre de progression `done / total`.
- Bargraph GPU util + VRAM.
- Nom du doc en cours + vitesse (chunks/s).
- ETA.
- Bouton kill (existant).

---

## 14. Non-objectifs (scope-out explicite)

- **Pas** de Late Interaction chunking type Jina/Colbert-Late (complexité non justifiée à 5 k docs).
- **Pas** de cross-encoder séparé au rerank (ColBERT suffit).
- **Pas** de fine-tuning de BGE-M3.
- **Pas** de migration de la collection `docfinder` existante — la v2 est créée à neuf (`docfinder_v2`), bascule UI après validation.
- **Pas** de UI directory/type selectors dans ce sprint (sous-projet A, plus tard).
- **Pas** de tests OCR adversariaux (sous-projet B, plus tard).

---

## 15. Critères de succès

- [ ] Indexation complète 5 000 docs en ≤ 90 min sur T4 (hors download).
- [ ] Aucun processus Python consommant > 5 % CPU sur le Mac pendant l'indexation (hors Qdrant et FastAPI).
- [ ] Recherche EN "contract renewal" retrouve un doc FR "renouvellement de contrat" dans le top-10.
- [ ] Les badges `keywords_doc` affichés sont cohérents (pas de dates brutes, pas de stopwords).
- [ ] Un PDF scanné FR de 10 pages est indexé avec texte OCR lisible (spot-check manuel).
- [ ] Après `touch` d'un fichier, re-run du pipeline → seul ce doc est ré-indexé.
- [ ] Latence de recherche (dense+sparse+rerank) < 600 ms en p95 sur 5 k docs.

---

## 16. Dépendances nouvelles

### Colab (`colab_indexer.ipynb`)

```
FlagEmbedding>=1.2.10   # BGE-M3
easyocr>=1.7.1
PyMuPDF>=1.24            # déjà utilisé
python-docx              # déjà utilisé
httpx[http2]>=0.27
qdrant-client>=1.10      # multi-vec + sparse + prefetch
```

### Mac (`server/`)

Aucune nouvelle dépendance lourde. Retirer `yake`, `sentence-transformers`, `pytesseract` du `requirements.txt` Mac (ils ne servent plus).

---

## 17. Migration et rollback

1. **Phase 1** (préparation) : créer `docfinder_v2` vide, déployer nouveaux endpoints Mac sans casser l'ancien flux.
2. **Phase 2** (indexation v2) : lancer notebook v2 sur Colab, remplit `docfinder_v2`.
3. **Phase 3** (bascule UI) : pointer `search.py` sur `docfinder_v2` derrière feature flag `USE_V2=true` dans `.env`.
4. **Phase 4** (validation 1 semaine) : comparer recherches côte à côte.
5. **Phase 5** (nettoyage) : supprimer `docfinder`, retirer l'ancien code d'extraction/YAKE Mac.

**Rollback** : `USE_V2=false` → retour immédiat sur l'ancienne collection.

---

## 18. Risques identifiés

| Risque                                                    | Mitigation                                                              |
|-----------------------------------------------------------|-------------------------------------------------------------------------|
| Colab 5 h d'inactivité → kernel kill en plein batch       | Checkpoint `{doc_id: status}` + reprise idempotente (skip déjà indexés) |
| easyocr + BGE-M3 VRAM > 16 GB sur très longs documents    | `max_length=512` strict, truncation ; batch=16 en fallback              |
| Cloudflare timeout sur `/files/raw` gros fichier          | HTTP chunked streaming côté Mac, retries exponentiels côté Colab        |
| `doc_id = sha1(abs_path)` change si le fichier est déplacé| Accepté : déplacer = supprimer+créer (cohérent avec mtime-based diff)   |
| Sparse tokens bruyants (tokens BPE tronqués)              | Post-filtre : longueur ≥ 3, pas `##` WordPiece, stopword-list FR+EN     |
| Dégradation qualité vs e5-large sur un cas particulier    | Phase 4 (comparaison côte à côte) avant nettoyage                       |

---

## 19. Questions ouvertes (à trancher au plan)

- Faut-il conserver `keywords_doc` sur chaque chunk (redondance payload) ou créer une collection `documents` séparée ? → **Défaut retenu** : dupliquer (simplicité, 5 k docs × 15 tokens = négligeable).
- ColBERT vecteurs stockés en Qdrant : storage on_disk `True` pour économiser RAM ? → **Défaut retenu** : `on_disk=True` pour ColBERT uniquement, dense reste en RAM.
- Seuil OCR `< 10 chars` : à affiner ? → **Défaut retenu** : 10 ; ajustable via constante.

---

*Fin du design. Next: writing-plans skill → plan d'implémentation détaillé.*
