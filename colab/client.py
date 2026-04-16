"""HTTP client used from Colab to talk to the Mac file server."""
from __future__ import annotations

from typing import Optional

import httpx


class MacClient:
    def __init__(self, base_url: str, transport: Optional[httpx.BaseTransport] = None,
                 timeout: float = 60.0):
        self.base = base_url.rstrip("/")
        self._client = httpx.Client(transport=transport, timeout=timeout,
                                    http2=True, follow_redirects=True)

    def list_files(self, root: str) -> list[dict]:
        r = self._client.get(f"{self.base}/files/list", params={"root": root})
        r.raise_for_status()
        return r.json()

    def download(self, abs_path: str) -> bytes:
        r = self._client.get(f"{self.base}/files/raw", params={"path": abs_path})
        r.raise_for_status()
        return r.content

    def indexed_state(self, doc_ids: list[str] | None = None) -> dict:
        r = self._client.post(f"{self.base}/admin/indexed-state",
                              json={"doc_ids": doc_ids})
        r.raise_for_status()
        return r.json()

    def delete_doc(self, doc_id: str) -> None:
        r = self._client.delete(f"{self.base}/admin/doc/{doc_id}")
        r.raise_for_status()

    def upsert_chunks_v2(self, points: list[dict]) -> dict:
        r = self._client.post(f"{self.base}/admin/upsert-v2", json={"points": points})
        r.raise_for_status()
        return r.json()

    def push_progress(self, report: dict) -> None:
        r = self._client.post(f"{self.base}/admin/progress", json=report)
        r.raise_for_status()
