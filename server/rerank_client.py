"""HTTP client for Colab /rerank — mirrors RemoteEncoder shape."""
from __future__ import annotations

import logging
import os
from typing import List

import httpx

logger = logging.getLogger("docfinder.rerank_client")

DEFAULT_TIMEOUT_S = 30.0


class RemoteRerankerError(RuntimeError):
    """Raised when the remote reranker is misconfigured or unreachable."""


class RemoteReranker:
    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        url = (base_url or os.environ.get("COLAB_RERANK_URL", "")).strip().rstrip("/")
        auth = (token or os.environ.get("COLAB_QUERY_TOKEN", "")).strip()
        if not url:
            raise RemoteRerankerError("COLAB_RERANK_URL is not set")
        if not auth:
            raise RemoteRerankerError("COLAB_QUERY_TOKEN is not set")

        self._url = f"{url}/rerank"
        self._headers = {"X-Auth-Token": auth, "Content-Type": "application/json"}

        cf_id = os.environ.get("CF_ACCESS_CLIENT_ID", "").strip()
        cf_secret = os.environ.get("CF_ACCESS_CLIENT_SECRET", "").strip()
        if cf_id and cf_secret:
            self._headers["CF-Access-Client-Id"] = cf_id
            self._headers["CF-Access-Client-Secret"] = cf_secret

        self._client = httpx.Client(timeout=timeout)
        logger.info("RemoteReranker targeting %s", self._url)

    def rerank(self, query: str, documents: List[str]) -> List[float]:
        if not documents:
            return []
        try:
            resp = self._client.post(
                self._url,
                headers=self._headers,
                json={"query": query, "documents": documents},
            )
        except httpx.HTTPError as exc:
            raise RemoteRerankerError(f"remote reranker unreachable: {exc}") from exc
        if resp.status_code != 200:
            raise RemoteRerankerError(
                f"remote reranker HTTP {resp.status_code}: {resp.text[:200]}"
            )
        return [float(s) for s in resp.json()["scores"]]

    def close(self) -> None:
        self._client.close()
