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

import logging
import os
from typing import List, Tuple

import httpx

from colab.embedder_v2 import EncodeResult

logger = logging.getLogger("docfinder.encode_client")

DEFAULT_TIMEOUT_S = 30.0
_ENDPOINTS = {"bgem3": "/encode", "qwen": "/encode_qwen"}


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
    ) -> None:
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

        self._client = httpx.Client(timeout=timeout)
        logger.info("RemoteEncoder targeting %s", self._url)

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        max_length: int = 512,
    ) -> EncodeResult:
        if not texts:
            return EncodeResult(dense=[], sparse=[], colbert=[], lexical_weights=[])
        try:
            resp = self._client.post(
                self._url,
                headers=self._headers,
                json={
                    "queries": texts,
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
        sparse: List[Tuple[List[int], List[float]]] = [
            ([int(i) for i in pair[0]], [float(v) for v in pair[1]])
            for pair in data["sparse"]
        ]
        return EncodeResult(
            dense=data["dense"],
            sparse=sparse,
            colbert=data["colbert"],
            lexical_weights=[],
        )

    def close(self) -> None:
        self._client.close()
