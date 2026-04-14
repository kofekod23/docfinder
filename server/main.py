"""
Application FastAPI — DocFinder.

Le modèle sentence-transformers est chargé UNE SEULE FOIS au démarrage
via le lifespan asynccontextmanager. Toutes les requêtes POST /search
réutilisent le même SearchEngine (et donc le même modèle en mémoire).

Démarrage :
    uvicorn server.main:app --reload --port 8000
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

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


@app.get("/health")
async def health() -> dict:
    """Endpoint de santé — vérifie que le moteur est initialisé."""
    return {"status": "ok", "engine_ready": _engine is not None}
