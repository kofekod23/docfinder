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
import json
import logging
import os
import subprocess
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Literal, TypedDict
from urllib.parse import unquote

import httpx
from qdrant_client import QdrantClient

from fastapi import FastAPI, Form, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.templating import Jinja2Templates

from server.encode_client import RemoteEncoder
from server.indexer import COLLECTION, ICLOUD_DEFAULT, QDRANT_URL, cancel_indexation, colab_file_skipped, current_job, resume_if_needed, start_indexation, upsert_points
from server.chunks import iter_chunks_json
from server.files_api import router as files_router
from server.admin_v2 import router as admin_v2_router, set_qdrant_client
from server.search import SearchEngine, search_v2, search_v2_tunable, retrieve_v2_channels, fuse_v2
from scripts.setup_qdrant_v2 import ensure_collection
from shared.schema import SearchResult, SearchQuery


class _SearchConfig(TypedDict):
    collection: str
    embedder: Literal["bgem3", "qwen"]

# Répertoire des templates Jinja2
TEMPLATES_DIR = Path(__file__).parent / "templates"

logger = logging.getLogger("docfinder.main")

# Instance globale du moteur (initialisée dans lifespan)
_engine: SearchEngine | None = None

# Instance globale du reranker (initialisée dans lifespan si RERANK_ENABLED)
_reranker: "RemoteReranker | None" = None

# Top-N pour reranking, parsé une fois au démarrage
_rerank_top_n: int = int(os.environ.get("RERANK_TOP_N", "20"))

# État Colab — mis à jour par heartbeat et upserts
_colab_state: dict = {
    "connected": False,
    "device": None,           # "cuda" | "cpu" | None
    "last_seen": 0.0,         # timestamp UNIX du dernier heartbeat
    "paused": False,          # flag pause demandée par l'UI
    "last_activity": 0.0,     # timestamp du dernier /admin/upsert reçu
}
COLAB_TIMEOUT = 30.0          # secondes sans heartbeat = déconnecté (marge tunnel CF)
COLAB_ACTIVE_TIMEOUT = 30.0   # secondes sans upsert = inactif (entre deux batchs)

# Sous-répertoires à exclure de l'indexation courante (mis à jour par POST /admin/index)
_current_job_excluded: list[str] = []


def _resolve_search_config() -> _SearchConfig:
    """Lit DOCFINDER_COLLECTION + DOCFINDER_EMBEDDER avec fallbacks sûrs.

    Defaults : collection=docfinder_v2, embedder=bgem3 (statu quo BGE-M3).
    """
    collection = os.environ.get("DOCFINDER_COLLECTION", "docfinder_v2").strip()
    if not collection:
        raise ValueError("DOCFINDER_COLLECTION ne peut pas être vide")
    embedder = os.environ.get("DOCFINDER_EMBEDDER", "bgem3").strip().lower()
    if embedder not in {"bgem3", "qwen"}:
        raise ValueError(f"DOCFINDER_EMBEDDER invalide : {embedder!r}")
    return {"collection": collection, "embedder": embedder}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Cycle de vie du serveur.
    Charge le modèle embedding + initialise le moteur au démarrage.
    Libère proprement à l'arrêt.
    """
    global _engine, _reranker
    print("[DocFinder] Démarrage — chargement du moteur de recherche…")
    _engine = SearchEngine()
    ensure_collection(_engine.client, name="docfinder_v2")
    set_qdrant_client(_engine.client, collection="docfinder_v2")
    if os.environ.get("RERANK_ENABLED", "false").lower() in ("true", "1", "yes", "on"):
        from server.rerank_client import RemoteReranker, RemoteRerankerError
        try:
            _reranker = RemoteReranker()
            print("[DocFinder] Reranker activé.")
        except RemoteRerankerError as exc:
            print(f"[DocFinder] Reranker désactivé ({exc}).")
            _reranker = None
    print("[DocFinder] Prêt sur http://localhost:8000")
    resume_if_needed()
    yield
    if _reranker is not None:
        _reranker.close()
    print("[DocFinder] Arrêt du serveur.")


app = FastAPI(title="DocFinder — Recherche hybride de documents", lifespan=lifespan)
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.include_router(files_router)
app.include_router(admin_v2_router)


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


def _engine_search(body: SearchQuery) -> List[SearchResult]:
    """
    V1 search path: uses the default embedder and searches docfinder collection.

    Args:
        body: SearchQuery with query and limit.

    Returns:
        List of SearchResult objects.
    """
    query = body.query.strip()
    limit = max(1, min(body.limit, 50))
    if not query:
        return []
    return _engine.search(query=query, limit=limit)


async def _parse_search_body(request: Request) -> SearchQuery:
    ct = request.headers.get("content-type", "")
    if "application/json" in ct:
        data = await request.json()
        return SearchQuery(**data)
    form = await request.form()
    return SearchQuery(
        query=str(form.get("query", "")),
        limit=int(form.get("limit", 10) or 10),
    )


@app.post("/search")
async def search(request: Request):
    """
    Exécute une recherche hybride.

    - POST JSON (`application/json`) → réponse JSON (API + tests).
    - POST form (`application/x-www-form-urlencoded`) → template rendu (UI).
    Route vers v2 si USE_V2=true, sinon v1.
    """
    is_json = "application/json" in request.headers.get("content-type", "")
    try:
        body = await _parse_search_body(request)
    except Exception as exc:
        if is_json:
            return JSONResponse({"status": "error", "message": str(exc)}, status_code=422)
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "results": None, "query": "", "error": str(exc)},
        )

    try:
        loop = asyncio.get_event_loop()
        if os.environ.get("USE_V2", "false").lower() in ("true", "1", "yes", "on"):
            cfg = _resolve_search_config()
            encoder = _engine.embedder_v2
            if cfg["embedder"] == "qwen":
                encoder = RemoteEncoder(embedder="qwen")
            results = await loop.run_in_executor(
                None,
                lambda: search_v2(
                    _engine.qdrant,
                    encoder,
                    body.query,
                    collection=cfg["collection"],
                    limit=body.limit,
                    reranker=_reranker,
                    rerank_top_n=_rerank_top_n,
                ),
            )
        else:
            results = await loop.run_in_executor(None, lambda: _engine_search(body))

        if is_json:
            return JSONResponse(
                {"status": "ok", "results": [r.model_dump() for r in results]}
            )
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "results": [r.model_dump() for r in results],
                "query": body.query,
                "error": None,
            },
        )
    except Exception as exc:
        logger.exception("/search failed for query=%r (v2=%s)", body.query, os.environ.get("USE_V2"))
        if is_json:
            return JSONResponse(
                {"status": "error", "message": str(exc)}, status_code=500
            )
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "results": None, "query": body.query, "error": str(exc)},
        )


@app.post("/search_tune")
async def search_tune(request: Request):
    """Recherche v2 paramétrable — sert au grid-search.

    Body JSON attendu (tous les poids/seuils optionnels) :
        {
            "query": "...",
            "limit": 10,
            "prefetch_limit": 300,
            "w_dense": 1.0,
            "w_sparse": 1.0,
            "w_colbert": 1.0,
            "rrf_k": 60,
            "filename_boost": 1.0
        }
    """
    try:
        body = await request.json()
    except Exception as exc:
        return JSONResponse({"status": "error", "message": f"invalid json: {exc}"}, status_code=422)

    query = str(body.get("query") or "").strip()
    if not query:
        return JSONResponse({"status": "error", "message": "query is required"}, status_code=422)

    if _engine is None or _engine.embedder_v2 is None:
        return JSONResponse({"status": "error", "message": "v2 engine not ready"}, status_code=503)

    kwargs = dict(
        collection="docfinder_v2",
        limit=int(body.get("limit", 10) or 10),
        prefetch_limit=int(body.get("prefetch_limit", 300) or 300),
        w_dense=float(body.get("w_dense", 1.0)),
        w_sparse=float(body.get("w_sparse", 1.0)),
        w_colbert=float(body.get("w_colbert", 1.0)),
        rrf_k=int(body.get("rrf_k", 60) or 60),
        filename_boost=float(body.get("filename_boost", 1.0)),
    )

    try:
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: search_v2_tunable(
                _engine.qdrant, _engine.embedder_v2, query, **kwargs
            ),
        )
        return JSONResponse(
            {"status": "ok", "results": [r.model_dump() for r in results]}
        )
    except Exception as exc:
        return JSONResponse({"status": "error", "message": str(exc)}, status_code=500)


@app.post("/search_tune_batch")
async def search_tune_batch(request: Request):
    """Batch grid-search : encode la requête une fois, applique N fusions localement.

    Body JSON :
        {
            "query": "...",
            "limit": 10,
            "prefetch_limit": 300,
            "configs": [
                {"name": "...", "w_dense": ..., "w_sparse": ..., "w_colbert": ...,
                 "rrf_k": ..., "filename_boost": ...},
                ...
            ]
        }

    Réponse :
        {"status": "ok", "results": {name: [SearchResult, ...], ...}}
    """
    try:
        body = await request.json()
    except Exception as exc:
        return JSONResponse({"status": "error", "message": f"invalid json: {exc}"}, status_code=422)

    query = str(body.get("query") or "").strip()
    if not query:
        return JSONResponse({"status": "error", "message": "query is required"}, status_code=422)

    configs = body.get("configs") or []
    if not isinstance(configs, list) or not configs:
        return JSONResponse({"status": "error", "message": "configs (list) is required"}, status_code=422)

    if _engine is None or _engine.embedder_v2 is None:
        return JSONResponse({"status": "error", "message": "v2 engine not ready"}, status_code=503)

    limit = int(body.get("limit", 10) or 10)
    prefetch_limit = int(body.get("prefetch_limit", 300) or 300)

    def _run_batch() -> dict[str, list]:
        channels = retrieve_v2_channels(
            _engine.qdrant, _engine.embedder_v2, query,
            collection="docfinder_v2", prefetch_limit=prefetch_limit,
        )
        out: dict[str, list] = {}
        for cfg in configs:
            name = str(cfg.get("name") or "unnamed")
            results = fuse_v2(
                channels, query,
                limit=limit,
                w_dense=float(cfg.get("w_dense", 1.0)),
                w_sparse=float(cfg.get("w_sparse", 1.0)),
                w_colbert=float(cfg.get("w_colbert", 1.0)),
                rrf_k=int(cfg.get("rrf_k", 60) or 60),
                filename_boost=float(cfg.get("filename_boost", 1.0)),
            )
            out[name] = [r.model_dump() for r in results]
        return out

    try:
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, _run_batch)
        return JSONResponse({"status": "ok", "results": results})
    except Exception as exc:
        return JSONResponse({"status": "error", "message": str(exc)}, status_code=500)


@app.get("/admin", response_class=HTMLResponse)
async def admin(request: Request) -> HTMLResponse:
    """Page d'administration — lancer une indexation."""
    return templates.TemplateResponse(
        "admin.html",
        {
            "request": request,
            "default_path": ICLOUD_DEFAULT,
            "job": current_job(),
            "use_v2": os.environ.get("USE_V2", "false") == "true",
        },
    )


@app.get("/admin/dirs")
async def admin_dirs(path: str = Query(default=ICLOUD_DEFAULT)) -> JSONResponse:
    """Retourne la liste des sous-répertoires immédiats non-cachés du chemin donné."""
    p = Path(path).expanduser()
    if not p.exists() or not p.is_dir():
        return JSONResponse({"dirs": []})
    dirs = sorted(
        d.name for d in p.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )
    return JSONResponse({"dirs": dirs})


@app.post("/admin/index")
async def admin_index(
    path: str = Form(default=ICLOUD_DEFAULT),
    reset: bool = Form(default=False),
    excluded_dirs: str = Form(default="[]"),
) -> JSONResponse:
    """Lance une indexation en arrière-plan."""
    global _current_job_excluded
    try:
        _current_job_excluded = json.loads(excluded_dirs)
    except Exception:
        _current_job_excluded = []
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
        "path": job["path"],
        "done": job["done"],
        "total": job["total"],
        "chunks": job["chunks"],
        "progress_pct": job["progress_pct"],
    })


@app.get("/admin/indexed-doc-ids")
async def admin_indexed_doc_ids() -> JSONResponse:
    """
    Retourne l'ensemble des doc_id déjà présents dans Qdrant.
    Colab l'utilise au démarrage pour ignorer les docs déjà indexés (reprise).
    """
    try:
        client = QdrantClient(url=QDRANT_URL)
        doc_ids: set[str] = set()
        offset = None
        while True:
            results, next_offset = client.scroll(
                collection_name=COLLECTION,
                with_payload=["doc_id"],
                with_vectors=False,
                limit=1000,
                offset=offset,
            )
            if not results:
                break
            for pt in results:
                doc_id = (pt.payload or {}).get("doc_id", "")
                if doc_id:
                    doc_ids.add(doc_id)
            if next_offset is None:
                break
            offset = next_offset
        return JSONResponse({"doc_ids": list(doc_ids)})
    except Exception as exc:
        return JSONResponse({"doc_ids": [], "error": str(exc)})


@app.get("/admin/progress-view", response_class=HTMLResponse)
async def admin_progress_view(request: Request) -> HTMLResponse:
    """Page de suivi live — poll /admin/progress toutes les 2 s."""
    return templates.TemplateResponse("admin_progress.html", {"request": request})


@app.get("/admin/db", response_class=HTMLResponse)
async def admin_db(request: Request, v: int = 2) -> HTMLResponse:
    """Page d'état de la base Qdrant — liste des documents indexés.

    ?v=1 → collection docfinder (legacy)
    ?v=2 → collection docfinder_v2 (pipeline Colab, défaut)
    """
    collection = "docfinder_v2" if v == 2 else COLLECTION
    stats: dict = {"total_chunks": 0, "total_docs": 0, "by_type": {}, "docs": [], "collection": collection, "version": v}
    try:
        client = QdrantClient(url=QDRANT_URL)
        docs: dict[str, dict] = {}

        offset = None
        while True:
            results, next_offset = client.scroll(
                collection_name=collection,
                with_payload=["doc_id", "title", "path", "abs_path", "doc_type", "is_scan"],
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
                        "abs_path": p.get("abs_path", "") or p.get("path", ""),
                        "doc_type": p.get("doc_type", "?"),
                        "is_scan": bool(p.get("is_scan", False)),
                        "chunks": 0,
                    }
                if p.get("is_scan"):
                    docs[doc_id]["is_scan"] = True
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
        iter_chunks_json(path, exclude=_current_job_excluded or None),
        media_type="application/x-ndjson",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
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
        _colab_state["last_activity"] = time.time()
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
    """
    Retourne les URLs du tunnel exposant ce serveur à Colab.

    Ordre de priorité :
      1. Cloudflare Tunnel — URL publique stable définie dans .env
         (DOCFINDER_PUBLIC_URL). Santé vérifiée via les métriques locales
         de cloudflared si disponibles (port 8081 par défaut).
      2. Fallback ngrok — API locale sur le port 4040 (rétrocompat).

    La réponse expose un dict {name: url} pour l'UI admin.
    """
    # ── 1. Cloudflare Tunnel ────────────────────────────────────────────────
    cf_url = os.getenv("DOCFINDER_PUBLIC_URL", "").strip()
    if cf_url:
        provider = "cloudflare"
        healthy = False
        try:
            # cloudflared expose /ready sur son port de métriques (défaut 8081)
            async with httpx.AsyncClient() as client:
                r = await client.get("http://localhost:8081/ready", timeout=2)
                healthy = r.status_code == 200
        except Exception:
            healthy = False  # cloudflared pas lancé ou métriques désactivées
        return JSONResponse({
            "provider": provider,
            "healthy": healthy,
            "tunnels": {"docfinder": cf_url},
        })

    # ── 2. Fallback ngrok ───────────────────────────────────────────────────
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get("http://localhost:4040/api/tunnels", timeout=3)
            ngrok_tunnels = r.json().get("tunnels", [])
        result = {t["name"]: t["public_url"] for t in ngrok_tunnels}
        return JSONResponse({
            "provider": "ngrok" if result else None,
            "healthy": bool(result),
            "tunnels": result,
        })
    except Exception:
        return JSONResponse({"provider": None, "healthy": False, "tunnels": {}})


@app.get("/admin/resources")
async def admin_resources() -> JSONResponse:
    """
    Retourne l'utilisation CPU et RAM des processus DocFinder (uvicorn + qdrant).
    Utilise cpu_percent(interval=None) pour ne pas bloquer.
    """
    import os

    import psutil

    pid = os.getpid()

    def _gather() -> dict:
        uvicorn_cpu = uvicorn_rss = qdrant_cpu = qdrant_rss = 0.0
        try:
            p = psutil.Process(pid)
            uvicorn_cpu = p.cpu_percent(interval=None)
            uvicorn_rss = p.memory_info().rss / 1_048_576
        except Exception:
            pass
        try:
            for p in psutil.process_iter(["pid", "name", "cmdline"]):
                name = (p.info.get("name") or "").lower()
                cmd = " ".join(p.info.get("cmdline") or []).lower()
                if "qdrant" in name or "qdrant" in cmd:
                    qdrant_cpu = p.cpu_percent(interval=None)
                    qdrant_rss = p.memory_info().rss / 1_048_576
                    break
        except Exception:
            pass
        return {
            "uvicorn": {"cpu": round(uvicorn_cpu, 1), "rss_mb": round(uvicorn_rss, 1)},
            "qdrant":  {"cpu": round(qdrant_cpu, 1),  "rss_mb": round(qdrant_rss, 1)},
        }

    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(None, _gather)
    return JSONResponse(data)


@app.get("/health")
async def health() -> dict:
    """Endpoint de santé — vérifie que le moteur est initialisé."""
    return {"status": "ok", "engine_ready": _engine is not None}


# ── Colab status / pause ────────────────────────────────────────────────────

@app.post("/admin/colab/heartbeat")
async def colab_heartbeat(request: Request) -> JSONResponse:
    """
    Reçoit un heartbeat de Colab.
    Body JSON : {"device": "cuda"} ou {"device": "cpu"}
    """
    body = await request.json()
    _colab_state["device"] = body.get("device")
    _colab_state["last_seen"] = time.time()
    _colab_state["connected"] = True
    return JSONResponse({"ok": True})


@app.get("/admin/colab/control")
async def colab_control() -> JSONResponse:
    """Colab interroge cet endpoint pour savoir s'il doit faire pause."""
    return JSONResponse({"paused": _colab_state["paused"]})


@app.post("/admin/colab/skip")
async def colab_skip(request: Request) -> JSONResponse:
    """Reçoit le nombre de fichiers ignorés par Colab (déjà indexés).

    Colab appelle cet endpoint par batches pour signaler les fichiers skippés,
    afin que l'UI admin reflète la vraie progression même quand tout est déjà indexé.
    """
    body = await request.json()
    count = int(body.get("count", 0))
    if count > 0:
        colab_file_skipped(count)
        _colab_state["last_activity"] = time.time()
    return JSONResponse({"ok": True})


@app.post("/admin/colab/pause")
async def colab_pause() -> JSONResponse:
    """Met Colab en pause."""
    _colab_state["paused"] = True
    return JSONResponse({"paused": True})


@app.post("/admin/colab/resume")
async def colab_resume() -> JSONResponse:
    """Reprend Colab après pause."""
    _colab_state["paused"] = False
    return JSONResponse({"paused": False})


@app.get("/admin/colab/status")
async def colab_status() -> JSONResponse:
    """Retourne l'état Colab pour l'UI (connecté, device, paused)."""
    now = time.time()
    connected = (now - _colab_state["last_seen"]) < COLAB_TIMEOUT
    if not connected and _colab_state["connected"]:
        _colab_state["connected"] = False
    active = connected and (now - _colab_state["last_activity"]) < COLAB_ACTIVE_TIMEOUT
    return JSONResponse({
        "connected": connected,
        "device": _colab_state["device"] if connected else None,
        "paused": _colab_state["paused"],
        "active": active,
    })
