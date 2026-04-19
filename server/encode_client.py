"""HTTP client adapter for Colab `/encode` — mirrors `BGEM3Wrapper.encode()`.

Le Mac n'exécute plus BGE-M3 en local (incompatible torch >= 2.6 sur Intel).
Ce client appelle le serveur d'encodage Colab via Cloudflare Tunnel et renvoie
un `EncodeResult` dont la forme est strictement identique à celle produite par
`colab.embedder_v2.BGEM3Wrapper` — `search_v2()` n'a donc rien à savoir du
transport.

Si l'URL ou le token n'est pas configuré, l'instanciation échoue explicitement :
on veut un 503 plutôt qu'un fallback silencieux.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from pathlib import Path
from typing import List, Tuple

import httpx

from colab.embedder_v2 import EncodeResult

logger = logging.getLogger("docfinder.encode_client")

DEFAULT_TIMEOUT_S = 30.0
_ENDPOINTS = {"bgem3": "/encode", "qwen": "/encode_qwen"}

# Cache JSON des query encodings : permet de faire tourner un A/B sans Colab
# vivant (les query embeddings sont déterministes pour un même texte + modèle).
# Activé via env ENCODE_CACHE_PATH=<path>. Si set, toutes les requêtes cherchent
# d'abord dans le cache ; si cache-hit total → aucune requête Colab. Cache-miss
# → appel Colab + écriture dans le cache.
_CACHE_LOCK = threading.Lock()


class RemoteEncoderError(RuntimeError):
    """Raised when the remote encoder is misconfigured or unreachable."""


class RemoteEncoder:
    """HTTP adapter with the same `encode()` signature as `BGEM3Wrapper`."""

    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_S,
        embedder: str = "bgem3",
        *,
        _client: httpx.Client | None = None,
    ) -> None:
        """Initialise le client d'encodage distant.

        Args:
            base_url: URL de base du serveur Colab (env: COLAB_ENCODE_URL si None)
            token: Token d'authentification (env: COLAB_QUERY_TOKEN si None)
            timeout: Délai d'expiration en secondes pour les requêtes
            embedder: Type d'encodeur ("bgem3" ou "qwen")
            _client: [Injection de dépendance pour tests] Client httpx injecté
        """
        url = (base_url or os.environ.get("COLAB_ENCODE_URL", "")).strip().rstrip("/")
        auth = (token or os.environ.get("COLAB_QUERY_TOKEN", "")).strip()
        if not url:
            raise RemoteEncoderError("COLAB_ENCODE_URL is not set")
        if not auth:
            raise RemoteEncoderError("COLAB_QUERY_TOKEN is not set")
        if embedder not in _ENDPOINTS:
            raise RemoteEncoderError(f"unknown embedder: {embedder!r}")

        self._url = f"{url}{_ENDPOINTS[embedder]}"
        self._headers = {"X-Auth-Token": auth, "Content-Type": "application/json"}

        cf_id = os.environ.get("CF_ACCESS_CLIENT_ID", "").strip()
        cf_secret = os.environ.get("CF_ACCESS_CLIENT_SECRET", "").strip()
        if cf_id and cf_secret:
            self._headers["CF-Access-Client-Id"] = cf_id
            self._headers["CF-Access-Client-Secret"] = cf_secret

        self._client = _client if _client is not None else httpx.Client(timeout=timeout)
        self._embedder = embedder
        self._cache_path = self._resolve_cache_path()
        self._cache: dict = self._load_cache()
        logger.info(
            "RemoteEncoder targeting %s (cache=%s entries)",
            self._url,
            len(self._cache) if self._cache_path else "off",
        )

    def _resolve_cache_path(self) -> Path | None:
        raw = os.environ.get("ENCODE_CACHE_PATH", "").strip()
        return Path(raw) if raw else None

    def _cache_key(self, text: str, max_length: int) -> str:
        h = hashlib.sha256(f"{self._embedder}|{max_length}|{text}".encode()).hexdigest()
        return h

    def _load_cache(self) -> dict:
        if not self._cache_path or not self._cache_path.exists():
            return {}
        try:
            with self._cache_path.open(encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("cache load failed (%s), starting empty", exc)
            return {}

    def _save_cache(self) -> None:
        if not self._cache_path:
            return
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._cache_path.with_suffix(self._cache_path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(self._cache, fh)
        tmp.replace(self._cache_path)

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        max_length: int = 512,
    ) -> EncodeResult:
        if not texts:
            return EncodeResult(dense=[], sparse=[], colbert=[], lexical_weights=[])

        # Cache hit vérification : collecter les textes cachés et ceux manquants.
        keys = [self._cache_key(t, max_length) for t in texts]
        cached = [self._cache.get(k) for k in keys]
        missing_idx = [i for i, v in enumerate(cached) if v is None]

        if missing_idx:
            missing_texts = [texts[i] for i in missing_idx]
            try:
                resp = self._client.post(
                    self._url,
                    headers=self._headers,
                    json={
                        "queries": missing_texts,
                        "batch_size": batch_size,
                        "max_length": max_length,
                    },
                )
            except httpx.HTTPError as exc:
                raise RemoteEncoderError(f"remote encoder unreachable: {exc}") from exc

            if resp.status_code != 200:
                raise RemoteEncoderError(
                    f"remote encoder HTTP {resp.status_code}: {resp.text[:200]}"
                )

            data = resp.json()
            # Remplir le cache pour les missing
            with _CACHE_LOCK:
                for j, i in enumerate(missing_idx):
                    entry = {
                        "dense": data["dense"][j],
                        "sparse": data["sparse"][j],
                        "colbert": data["colbert"][j],
                    }
                    self._cache[keys[i]] = entry
                    cached[i] = entry
                if self._cache_path:
                    self._save_cache()

        # Reconstituer EncodeResult depuis cache complet
        dense_list = [e["dense"] for e in cached]
        colbert_list = [e["colbert"] for e in cached]
        sparse: List[Tuple[List[int], List[float]]] = [
            ([int(i) for i in e["sparse"][0]], [float(v) for v in e["sparse"][1]])
            for e in cached
        ]
        return EncodeResult(
            dense=dense_list,
            sparse=sparse,
            colbert=colbert_list,
            lexical_weights=[],
        )

    def close(self) -> None:
        self._client.close()
