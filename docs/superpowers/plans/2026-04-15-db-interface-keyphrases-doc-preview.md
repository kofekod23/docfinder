# DB Interface + Key Phrases + Doc Preview — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three user-facing features to DocFinder: a database state page, better key-phrase excerpts in search results, and one-click open + in-browser preview of documents.

**Architecture:** All three features are self-contained additions to the existing FastAPI+Jinja2 server. No new dependencies needed except `mimetypes` (stdlib). The preview endpoint streams files directly; open uses `subprocess.run(["open", path])` (macOS).

**Tech Stack:** FastAPI, Jinja2, Qdrant Python client, python-docx, pymupdf, YAKE, macOS `open` command.

---

## File map

| File | Change |
|------|--------|
| `server/main.py` | Add `/admin/db`, `/doc/open`, `/doc/preview` endpoints; add `library` nav link |
| `server/search.py` | Improve `excerpt` computation (sentence-level scoring) |
| `server/templates/admin_db.html` | New — DB state page |
| `server/templates/index.html` | Add Open + Preview buttons to each result card |

---

## Task 1 — Better excerpt via keyword-scored sentence selection

**Files:**
- Modify: `server/search.py` lines 185–189

Currently, `excerpt = content[:300]`. Replace with a function that:
1. Splits the chunk into sentences (split on `. `, `! `, `? `, `\n`)
2. Scores each sentence by the number of stored `keywords` it contains (case-insensitive)
3. Returns the top 1–2 sentences, up to 300 chars

- [ ] **Step 1: Add `_best_excerpt` helper in `search.py`**

Insert after line 82 (after `_build_sparse_vector`):

```python
def _best_excerpt(content: str, keywords: list[str], max_chars: int = 300) -> str:
    """
    Sélectionne les phrases les plus riches en mots-clés pour l'extrait.
    Si aucun mot-clé ne matche, retourne le début du contenu.
    """
    import re
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n', content) if s.strip()]
    if not sentences:
        return content[:max_chars].strip()

    kw_lower = [k.lower() for k in keywords]

    def score(s: str) -> int:
        sl = s.lower()
        return sum(1 for k in kw_lower if k in sl)

    ranked = sorted(sentences, key=score, reverse=True)
    excerpt = ""
    for s in ranked:
        candidate = (excerpt + " " + s).strip() if excerpt else s
        if len(candidate) > max_chars:
            break
        excerpt = candidate
        if len(excerpt) >= max_chars // 2:
            break

    if not excerpt:
        excerpt = sentences[0]

    if len(excerpt) > max_chars:
        excerpt = excerpt[:max_chars].rstrip() + "…"
    return excerpt
```

- [ ] **Step 2: Replace the excerpt computation in `SearchEngine.search`**

In `search.py`, replace lines 185–189:
```python
            content = payload.get("content", "")
            excerpt = content[:300].strip()
            if len(content) > 300:
                excerpt += "…"
```
With:
```python
            content = payload.get("content", "")
            kw = payload.get("keywords", [])
            excerpt = _best_excerpt(content, kw)
```

- [ ] **Step 3: Manual test**

Start the server and search for "contrat" or any term present in your indexed PDFs. Verify that excerpts now show sentences containing the search-related keywords rather than always the first 300 chars.

- [ ] **Step 4: Commit**

```bash
git add server/search.py
git commit -m "feat(search): extract excerpt from keyword-rich sentences instead of first 300 chars"
```

---

## Task 2 — Document open endpoint (`/doc/open`)

**Files:**
- Modify: `server/main.py` — add endpoint after `/admin/upsert`

This endpoint receives a relative `path` (from Qdrant payload), reconstructs the absolute path, and calls macOS `open`.

- [ ] **Step 1: Add `GET /doc/open` in `main.py`**

Add after the `admin_upsert` endpoint (around line 179):

```python
import subprocess
from urllib.parse import unquote

@app.get("/doc/open")
async def doc_open(path: str = Query(...)) -> JSONResponse:
    """
    Ouvre un document avec l'application macOS par défaut.
    `path` est le chemin relatif stocké dans Qdrant (relatif à ICLOUD_DEFAULT).
    """
    from server.indexer import ICLOUD_DEFAULT
    abs_path = Path(ICLOUD_DEFAULT) / unquote(path)
    if not abs_path.exists():
        return JSONResponse({"error": f"Fichier introuvable : {path}"}, status_code=404)
    subprocess.run(["open", str(abs_path)], check=False)
    return JSONResponse({"opened": True})
```

- [ ] **Step 2: Test manually**

```bash
curl "http://localhost:8000/doc/open?path=MonDossier/MonDoc.pdf"
```

Expected: `{"opened": true}` and the PDF opens in Preview.app. For a bad path: `{"error": "Fichier introuvable : …"}` with 404.

- [ ] **Step 3: Commit**

```bash
git add server/main.py
git commit -m "feat(doc): add /doc/open endpoint — opens file with macOS default app"
```

---

## Task 3 — Document preview endpoint (`/doc/preview`)

**Files:**
- Modify: `server/main.py` — add endpoint

Three cases:
- **PDF** → serve raw bytes with `application/pdf` (browser renders it inline)
- **txt / md** → serve as `text/plain; charset=utf-8`
- **docx / doc** → extract text with python-docx, serve as `text/html` (simple styled HTML)

- [ ] **Step 1: Add `GET /doc/preview` in `main.py`**

Add after `/doc/open`:

```python
from fastapi.responses import Response

@app.get("/doc/preview")
async def doc_preview(path: str = Query(...)) -> Response:
    """
    Renvoie le contenu du document pour aperçu dans le navigateur.
    - PDF  → servi directement (le navigateur l'affiche inline)
    - txt/md → text/plain
    - docx → HTML simple extrait via python-docx
    """
    from server.indexer import ICLOUD_DEFAULT
    abs_path = Path(ICLOUD_DEFAULT) / unquote(path)
    if not abs_path.exists():
        return Response(content=b"Fichier introuvable.", status_code=404)

    suffix = abs_path.suffix.lower()

    if suffix == ".pdf":
        data = abs_path.read_bytes()
        return Response(
            content=data,
            media_type="application/pdf",
            headers={"Content-Disposition": "inline"},
        )

    if suffix in {".txt", ".md"}:
        text = abs_path.read_text(errors="replace")
        return Response(content=text, media_type="text/plain; charset=utf-8")

    if suffix in {".docx", ".doc"}:
        try:
            from docx import Document as DocxDocument
            doc = DocxDocument(str(abs_path))
            paragraphs_html = "".join(
                f"<p>{para.text}</p>" for para in doc.paragraphs if para.text.strip()
            )
            html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  body{{font-family:-apple-system,sans-serif;max-width:720px;margin:2rem auto;
       padding:0 1rem;color:#222;line-height:1.7;font-size:15px}}
  p{{margin-bottom:.75rem}}
</style></head>
<body>{paragraphs_html}</body></html>"""
            return Response(content=html, media_type="text/html; charset=utf-8")
        except Exception as exc:
            return Response(content=f"Erreur de lecture : {exc}", status_code=500)

    return Response(content=b"Format non supporté pour l'aperçu.", status_code=415)
```

- [ ] **Step 2: Test manually**

```bash
# PDF — doit s'afficher dans le navigateur
curl -I "http://localhost:8000/doc/preview?path=MonDossier/MonDoc.pdf"
# → Content-Type: application/pdf

# TXT
curl "http://localhost:8000/doc/preview?path=MonDossier/notes.txt"
# → le texte brut

# DOCX
curl "http://localhost:8000/doc/preview?path=MonDossier/lettre.docx"
# → HTML
```

- [ ] **Step 3: Commit**

```bash
git add server/main.py
git commit -m "feat(doc): add /doc/preview — inline PDF, plain text, and docx-to-HTML"
```

---

## Task 4 — Open + Preview buttons in search results UI

**Files:**
- Modify: `server/templates/index.html`

Add two small icon-buttons per result card: "Ouvrir" (→ calls `/doc/open`) and "Aperçu" (→ opens `/doc/preview` in a modal `<dialog>`).

- [ ] **Step 1: Add CSS for action buttons and modal in `index.html`**

Inside the `<style>` block, add after `.badge-kw` rule:

```css
    /* Actions document */
    .card-actions {
      display: flex;
      gap: 0.5rem;
      margin-top: 0.75rem;
    }
    .action-btn {
      padding: 0.3rem 0.75rem;
      font-size: 0.75rem;
      font-weight: 500;
      border-radius: 4px;
      border: 1px solid var(--border);
      background: var(--surface-2);
      color: var(--muted);
      cursor: pointer;
      transition: border-color 150ms, color 150ms;
      text-decoration: none;
      display: inline-block;
    }
    .action-btn:hover { border-color: var(--accent); color: var(--accent); }

    /* Modal aperçu */
    dialog {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--r);
      padding: 0;
      width: min(90vw, 900px);
      max-height: 90vh;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }
    dialog::backdrop { background: rgba(0,0,0,0.65); }
    .dialog-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0.75rem 1rem;
      border-bottom: 1px solid var(--border);
      flex-shrink: 0;
    }
    .dialog-title {
      font-size: 0.875rem;
      font-weight: 600;
      color: var(--text);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      max-width: 80%;
    }
    .dialog-close {
      background: none;
      border: none;
      color: var(--muted);
      font-size: 1.2rem;
      cursor: pointer;
      line-height: 1;
      flex-shrink: 0;
    }
    .dialog-close:hover { color: var(--text); }
    .preview-frame {
      flex: 1;
      width: 100%;
      border: none;
      background: #fff;
    }
```

- [ ] **Step 2: Add action buttons to result cards in `index.html`**

In the `{% for r in results %}` loop, after the `<div class="card-footer">` block (after the closing `</div>`), add:

```html
          <div class="card-actions">
            <button class="action-btn" onclick="openDoc('{{ r.path | e }}')">↗ Ouvrir</button>
            <button class="action-btn" onclick="previewDoc('{{ r.path | e }}', '{{ r.title | e }}')">⊡ Aperçu</button>
          </div>
```

- [ ] **Step 3: Add modal + JS at the bottom of `index.html`**

Before the closing `</div>` (`.wrap`) tag, add:

```html
    <!-- Modal aperçu -->
    <dialog id="previewDialog">
      <div class="dialog-header">
        <span class="dialog-title" id="previewTitle"></span>
        <button class="dialog-close" onclick="document.getElementById('previewDialog').close()">✕</button>
      </div>
      <iframe class="preview-frame" id="previewFrame" src="about:blank"></iframe>
    </dialog>
```

And before `</body>`, add:

```html
  <script>
    async function openDoc(path) {
      const r = await fetch('/doc/open?path=' + encodeURIComponent(path));
      const j = await r.json();
      if (j.error) alert('Impossible d\'ouvrir : ' + j.error);
    }

    function previewDoc(path, title) {
      const dialog = document.getElementById('previewDialog');
      document.getElementById('previewTitle').textContent = title;
      document.getElementById('previewFrame').src = '/doc/preview?path=' + encodeURIComponent(path);
      dialog.showModal();
    }

    // Fermer en cliquant sur le backdrop
    document.getElementById('previewDialog').addEventListener('click', function(e) {
      if (e.target === this) this.close();
    });
  </script>
```

- [ ] **Step 4: Test in browser**

1. `uvicorn server.main:app --reload --port 8000`
2. Search for any term
3. Click "↗ Ouvrir" on a PDF result → PDF should open in Preview.app
4. Click "⊡ Aperçu" on a PDF result → modal appears with PDF rendered in the iframe
5. Click "⊡ Aperçu" on a .txt result → plain text shown
6. Click "⊡ Aperçu" on a .docx result → clean HTML shown
7. Click backdrop or ✕ → modal closes

- [ ] **Step 5: Commit**

```bash
git add server/templates/index.html
git commit -m "feat(ui): add Open and Preview actions to search result cards"
```

---

## Task 5 — Database state page (`/admin/db`)

**Files:**
- Modify: `server/main.py` — add `/admin/db` endpoint
- Create: `server/templates/admin_db.html`

The page shows:
- Total chunks in Qdrant
- Number of unique documents (by `doc_id`)
- Breakdown by doc type (PDF, DOCX, TXT, MD)
- Scrollable list of all indexed documents (title, path, type, chunk count)

The endpoint scrolls Qdrant with `scroll()` to retrieve all payloads (no vector needed), then aggregates in Python.

- [ ] **Step 1: Add `GET /admin/db` in `main.py`**

Add after `admin_ping` endpoint:

```python
@app.get("/admin/db", response_class=HTMLResponse)
async def admin_db(request: Request) -> HTMLResponse:
    """Page d'état de la base Qdrant — liste des documents indexés."""
    from qdrant_client import QdrantClient
    from server.indexer import COLLECTION, QDRANT_URL

    stats = {"total_chunks": 0, "total_docs": 0, "by_type": {}, "docs": []}
    try:
        client = QdrantClient(url=QDRANT_URL)
        docs: dict[str, dict] = {}

        offset = None
        while True:
            results, next_offset = client.scroll(
                collection_name=COLLECTION,
                with_payload=["doc_id", "title", "path", "doc_type"],
                with_vectors=False,
                limit=500,
                offset=offset,
            )
            if not results:
                break
            for pt in results:
                p = pt.payload or {}
                doc_id = p.get("doc_id", "")
                if doc_id not in docs:
                    docs[doc_id] = {
                        "title": p.get("title", "?"),
                        "path": p.get("path", ""),
                        "doc_type": p.get("doc_type", "?"),
                        "chunks": 0,
                    }
                docs[doc_id]["chunks"] += 1
            if next_offset is None:
                break
            offset = next_offset

        stats["total_chunks"] = sum(d["chunks"] for d in docs.values())
        stats["total_docs"] = len(docs)
        for d in docs.values():
            t = d["doc_type"]
            stats["by_type"][t] = stats["by_type"].get(t, 0) + 1
        stats["docs"] = sorted(docs.values(), key=lambda x: x["title"].lower())
    except Exception as exc:
        stats["error"] = str(exc)

    return templates.TemplateResponse(
        "admin_db.html",
        {"request": request, "stats": stats},
    )
```

- [ ] **Step 2: Create `server/templates/admin_db.html`**

```html
<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>DocFinder — Base de données</title>
  <style>
    :root {
      --bg:#0d1117;--surface:#161b22;--surface-2:#21262d;
      --border:#30363d;--accent:#58a6ff;--accent-2:#79c0ff;
      --text:#c9d1d9;--muted:#8b949e;--green:#3fb950;--red:#f85149;--r:8px;
    }
    *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
    body{background:var(--bg);color:var(--text);
         font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif;
         font-size:15px;line-height:1.5;min-height:100vh;padding:2.5rem 1rem 4rem}
    .wrap{max-width:860px;margin:0 auto}

    .nav{display:flex;gap:1.5rem;margin-bottom:2.5rem;
         border-bottom:1px solid var(--border);padding-bottom:1rem}
    .nav a{color:var(--muted);text-decoration:none;font-size:.9rem;font-weight:500}
    .nav a:hover{color:var(--text)}
    .nav a.active{color:var(--accent)}

    h1{font-size:1.5rem;font-weight:700;color:#fff;margin-bottom:.3rem}
    .subtitle{color:var(--muted);font-size:.875rem;margin-bottom:2rem}

    .stats-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
                gap:.75rem;margin-bottom:1.5rem}
    .stat-card{background:var(--surface);border:1px solid var(--border);
               border-radius:var(--r);padding:1rem 1.25rem}
    .stat-label{font-size:.75rem;color:var(--muted);margin-bottom:.25rem}
    .stat-value{font-size:1.75rem;font-weight:700;color:var(--accent)}
    .stat-sub{font-size:.75rem;color:var(--muted);margin-top:.1rem}

    .section-title{font-size:.9rem;font-weight:600;color:var(--text);
                   margin:1.5rem 0 .75rem}

    .type-row{display:flex;flex-wrap:wrap;gap:.5rem;margin-bottom:1.5rem}
    .type-badge{padding:.35rem .8rem;border-radius:4px;font-size:.8rem;font-weight:500;
                background:var(--surface-2);border:1px solid var(--border);color:var(--muted)}
    .type-badge strong{color:var(--text)}

    /* Search filter */
    .filter-input{width:100%;padding:.65rem 1rem;background:var(--surface);
                  border:1px solid var(--border);border-radius:var(--r);color:var(--text);
                  font-size:.875rem;outline:none;margin-bottom:.75rem;transition:border-color 150ms}
    .filter-input:focus{border-color:var(--accent)}
    .filter-input::placeholder{color:var(--muted)}

    /* Docs table */
    .doc-table{width:100%;border-collapse:collapse}
    .doc-table th{text-align:left;font-size:.75rem;font-weight:500;color:var(--muted);
                  padding:.5rem .75rem;border-bottom:1px solid var(--border)}
    .doc-table td{padding:.55rem .75rem;border-bottom:1px solid rgba(48,54,61,.5);
                  font-size:.82rem;vertical-align:top}
    .doc-table tr:hover td{background:var(--surface-2)}
    .doc-title{color:var(--accent-2);font-weight:500}
    .doc-path{font-family:ui-monospace,"SF Mono",monospace;font-size:.72rem;
              color:var(--muted);word-break:break-all}
    .doc-type{display:inline-block;padding:.15rem .45rem;border-radius:3px;
              font-size:.7rem;font-weight:500;background:var(--surface-2);
              border:1px solid var(--border);color:var(--muted)}
    .doc-chunks{color:var(--text);text-align:right;white-space:nowrap}

    .table-wrap{background:var(--surface);border:1px solid var(--border);
                border-radius:var(--r);overflow:hidden}
    .table-scroll{max-height:520px;overflow-y:auto}

    .error-box{padding:.85rem 1rem;background:rgba(248,81,73,.1);
               border:1px solid rgba(248,81,73,.35);border-radius:var(--r);
               color:var(--red);font-size:.875rem;margin-bottom:1.5rem}

    .empty{text-align:center;padding:2rem;color:var(--muted);font-size:.875rem}
  </style>
</head>
<body>
  <div class="wrap">

    <nav class="nav">
      <a href="/">Recherche</a>
      <a href="/admin">Indexation</a>
      <a href="/admin/db" class="active">Base</a>
    </nav>

    <h1>État de la base</h1>
    <p class="subtitle">Documents actuellement indexés dans Qdrant.</p>

    {% if stats.error %}
    <div class="error-box">Erreur Qdrant : {{ stats.error }}</div>
    {% else %}

    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-label">Documents</div>
        <div class="stat-value">{{ stats.total_docs }}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Chunks indexés</div>
        <div class="stat-value">{{ stats.total_chunks }}</div>
        <div class="stat-sub">~{{ (stats.total_chunks / stats.total_docs)|round|int if stats.total_docs else 0 }} par doc en moy.</div>
      </div>
      {% for t, n in stats.by_type.items() %}
      <div class="stat-card">
        <div class="stat-label">{{ t | upper }}</div>
        <div class="stat-value">{{ n }}</div>
      </div>
      {% endfor %}
    </div>

    <div class="section-title">Documents indexés ({{ stats.total_docs }})</div>

    <input class="filter-input" type="text" id="filterInput"
           placeholder="Filtrer par titre ou chemin…" oninput="filterDocs()">

    <div class="table-wrap">
      <div class="table-scroll">
        <table class="doc-table" id="docTable">
          <thead>
            <tr>
              <th>Titre</th>
              <th>Type</th>
              <th style="text-align:right">Chunks</th>
            </tr>
          </thead>
          <tbody id="docBody">
            {% for doc in stats.docs %}
            <tr data-search="{{ doc.title | lower }} {{ doc.path | lower }}">
              <td>
                <div class="doc-title">{{ doc.title }}</div>
                <div class="doc-path">{{ doc.path }}</div>
              </td>
              <td><span class="doc-type">{{ doc.doc_type | upper }}</span></td>
              <td class="doc-chunks">{{ doc.chunks }}</td>
            </tr>
            {% endfor %}
          </tbody>
        </table>
        {% if not stats.docs %}
        <div class="empty">Aucun document indexé.</div>
        {% endif %}
      </div>
    </div>

    {% endif %}
  </div>

  <script>
    function filterDocs() {
      const q = document.getElementById('filterInput').value.toLowerCase();
      document.querySelectorAll('#docBody tr').forEach(row => {
        row.style.display = row.dataset.search.includes(q) ? '' : 'none';
      });
    }
  </script>
</body>
</html>
```

- [ ] **Step 3: Update nav in `index.html` and `admin.html`**

In `server/templates/index.html`, replace:
```html
      <a href="/admin" style="color:var(--muted);font-size:.9rem;font-weight:500;text-decoration:none">Indexation</a>
```
With:
```html
      <a href="/admin" style="color:var(--muted);font-size:.9rem;font-weight:500;text-decoration:none">Indexation</a>
      <a href="/admin/db" style="color:var(--muted);font-size:.9rem;font-weight:500;text-decoration:none">Base</a>
```

In `server/templates/admin.html`, in the `.nav` block, add after the Indexation link:
```html
      <a href="/admin/db">Base</a>
```

- [ ] **Step 4: Test in browser**

1. Navigate to `http://localhost:8000/admin/db`
2. Verify stat cards show correct totals matching what was indexed
3. Type in the filter input — table should filter in real time
4. Verify chunk counts per document look sane

- [ ] **Step 5: Commit**

```bash
git add server/main.py server/templates/admin_db.html server/templates/index.html server/templates/admin.html
git commit -m "feat(admin): add /admin/db page with indexed document stats and filterable list"
```

---

## Self-review

**Spec coverage:**
- ✅ Interface pour voir l'état de la base → Task 5 (`/admin/db`)
- ✅ Améliorer l'extraction des points importants → Task 1 (keyword-scored sentences)
- ✅ Accéder en un clic (app par défaut) → Task 2 (`/doc/open`)
- ✅ Aperçu sans app externe → Task 3 + 4 (`/doc/preview` + modal)

**No placeholders:** all code is complete and runnable.

**Type consistency:** `path` is always the relative string from Qdrant payload; `abs_path` is always a `Path` object. `unquote` handles URL-encoded chars in paths (spaces → %20).
