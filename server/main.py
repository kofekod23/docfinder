"""
Application FastAPI — DocFinder.

Le modèle sentence-transformers est chargé UNE SEULE FOIS au démarrage
via le lifespan asynccontextmanager. Toutes les requêtes POST /search
réutilisent le même SearchEngine (et donc le même modèle en mémoire).

Démarrage :
    uvicorn server.main:app --reload --port 8000
"""
from __future__ import annotations

import asyncio
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List
from urllib.parse import unquote

import httpx
from qdrant_client import QdrantClient

from fastapi import FastAPI, Form, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.templating import Jinja2Templates

from server.indexer import COLLECTION, ICLOUD_DEFAULT, QDRANT_URL, cancel_indexation, current_job, start_indexation, upsert_points
from server.chunks import iter_chunks_json
from server.search import SearchEngine
from shared.schema import SearchResult

# Répertoire des templates Jinja2
TEMPLATES_DIR = Path(__file__).parent / "templates"

# Instance globale du moteur (initialisée dans lifespan)
_engine: SearchEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Cycle de vie du serveur.
    Charge le modèle embedding + initialise le moteur au démarrage.
    Libère proprement à l'arrêt.
    """
    global _engine
    print("[DocFinder] Démarrage — chargement du moteur de recherche…")
    _engine = SearchEngine()
    print("[DocFinder] Prêt sur http://localhost:8000")
    yield
    print("[DocFinder] Arrêt du serveur.")


app = FastAPI(title="DocFinder — Recherche hybride de documents", lifespan=lifespan)
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """Page d'accueil : formulaire de recherche vide."""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "results": None,
            "query": "",
            "error": None,
        },
    )


@app.post("/search", response_class=HTMLResponse)
async def search(
    request: Request,
    query: str = Form(...),
    limit: int = Form(default=10),
) -> HTMLResponse:
    """
    Exécute une recherche hybride et renvoie la page de résultats.

    La vectorisation de la requête se fait localement (CPU).
    Aucun appel externe n'est effectué.
    """
    results: List[SearchResult] = []
    error: str | None = None

    query = query.strip()
    if query:
        try:
            results = _engine.search(query=query, limit=max(1, min(limit, 50)))
        except Exception as exc:
            error = f"Erreur lors de la recherche : {exc}"

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "results": results,
            "query": query,
            "error": error,
        },
    )


@app.get("/admin", response_class=HTMLResponse)
async def admin(request: Request) -> HTMLResponse:
    """Page d'administration — lancer une indexation."""
    return templates.TemplateResponse(
        "admin.html",
        {"request": request, "default_path": ICLOUD_DEFAULT, "job": current_job()},
    )


@app.post("/admin/index")
async def admin_index(
    path: str = Form(default=ICLOUD_DEFAULT),
    reset: bool = Form(default=False),
) -> JSONResponse:
    """Lance une indexation en arrière-plan."""
    result = start_indexation(path=path.strip(), reset=reset)
    return JSONResponse(result)


@app.post("/admin/cancel")
async def admin_cancel() -> JSONResponse:
    """Annule l'indexation en cours."""
    return JSONResponse(cancel_indexation())


@app.get("/admin/status")
async def admin_status() -> JSONResponse:
    """Retourne l'état courant du job d'indexation."""
    return JSONResponse(current_job())


@app.get("/admin/ping")
async def admin_ping() -> JSONResponse:
    """Endpoint léger pour le polling Colab — pas de log."""
    job = current_job()
    return JSONResponse({
        "status": job["status"],
        "done": job["done"],
        "total": job["total"],
        "chunks": job["chunks"],
        "progress_pct": job["progress_pct"],
    })


@app.get("/admin/db", response_class=HTMLResponse)
async def admin_db(request: Request) -> HTMLResponse:
    """Page d'état de la base Qdrant — liste des documents indexés."""
    stats: dict = {"total_chunks": 0, "total_docs": 0, "by_type": {}, "docs": []}
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


@app.get("/chunks")
async def chunks(
    path: str = Query(default=ICLOUD_DEFAULT),
) -> StreamingResponse:
    """
    Endpoint NDJSON — un chunk par ligne.
    Colab consomme ce flux pour calculer les embeddings sur GPU
    puis écrit directement dans Qdrant via ngrok.
    """
    return StreamingResponse(
        iter_chunks_json(path),
        media_type="application/x-ndjson",
    )


@app.post("/admin/upsert")
async def admin_upsert(request: Request) -> JSONResponse:
    """
    Reçoit des points pré-calculés depuis Colab et les écrit dans Qdrant local.

    Body JSON : liste de dicts avec les champs :
        id             (int)
        dense          (list[float])
        sparse_indices (list[int], optionnel)
        sparse_values  (list[float], optionnel)
        payload        (dict)
    """
    try:
        points_data = await request.json()
        loop = asyncio.get_event_loop()
        n = await loop.run_in_executor(None, upsert_points, points_data)
        return JSONResponse({"inserted": n})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/doc/open")
async def doc_open(path: str = Query(...)) -> JSONResponse:
    """
    Ouvre un document avec l'application macOS par défaut.
    `path` peut être absolu (nouveau) ou relatif à ICLOUD_DEFAULT (rétrocompat).
    """
    decoded = unquote(path)
    p = Path(decoded)
    if p.is_absolute():
        abs_path = p.resolve()
        try:
            abs_path.relative_to(Path.home())
        except ValueError:
            return JSONResponse({"error": "Accès refusé."}, status_code=403)
    else:
        root = Path(ICLOUD_DEFAULT).resolve()
        abs_path = (root / decoded).resolve()
        try:
            abs_path.relative_to(root)
        except ValueError:
            return JSONResponse({"error": "Accès refusé."}, status_code=403)
    if not abs_path.exists():
        return JSONResponse({"error": f"Fichier introuvable : {path}"}, status_code=404)
    subprocess.run(["open", str(abs_path)], check=False)
    return JSONResponse({"opened": True})


@app.get("/doc/preview")
async def doc_preview(path: str = Query(...)) -> Response:
    """
    Renvoie le contenu du document pour aperçu dans le navigateur.
    - PDF  → servi directement (le navigateur l'affiche inline)
    - txt/md → text/plain
    - docx → HTML simple extrait via python-docx
    `path` peut être absolu (nouveau) ou relatif à ICLOUD_DEFAULT (rétrocompat).
    """
    decoded = unquote(path)
    p = Path(decoded)
    if p.is_absolute():
        abs_path = p.resolve()
        try:
            abs_path.relative_to(Path.home())
        except ValueError:
            return Response(content=b"Acces refuse.", status_code=403)
    else:
        root = Path(ICLOUD_DEFAULT).resolve()
        abs_path = (root / decoded).resolve()
        try:
            abs_path.relative_to(root)
        except ValueError:
            return Response(content=b"Acces refuse.", status_code=403)
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
            import html as _html
            doc = DocxDocument(str(abs_path))
            paragraphs_html = "".join(
                f"<p>{_html.escape(para.text)}</p>" for para in doc.paragraphs if para.text.strip()
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

    return Response(content=b"Format non supporte pour l'apercu.", status_code=415)


@app.get("/admin/tunnels")
async def admin_tunnels() -> JSONResponse:
    """Retourne les URLs ngrok actives (qdrant + docfinder)."""
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get("http://localhost:4040/api/tunnels", timeout=3)
            tunnels = r.json().get("tunnels", [])
        result = {}
        for t in tunnels:
            result[t["name"]] = t["public_url"]
        return JSONResponse(result)
    except Exception:
        return JSONResponse({})


@app.get("/health")
async def health() -> dict:
    """Endpoint de santé — vérifie que le moteur est initialisé."""
    return {"status": "ok", "engine_ready": _engine is not None}
