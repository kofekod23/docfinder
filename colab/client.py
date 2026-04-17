"""HTTP client used from Colab to talk to the Mac file server.

Le Mac est exposé derrière Cloudflare Access. Sans les headers service-token
`CF-Access-Client-Id/Secret`, la tunnel renvoie un 302 → HTML de login qui
casse `r.json()`. On injecte donc les headers depuis l'env si présents.
"""
from __future__ import annotations

import os
from typing import Optional

import httpx


# Cloudflare tunnel rejette les bodies >~100 MB (HTTP 413). On batch côté
# client à 80 MiB pour garder une marge sous ce plafond edge.
_UPSERT_CLIENT_SOFT_LIMIT_BYTES = 24 * 1024 * 1024


def _estimate_point_bytes(p: dict) -> int:
    """Approximation JSON d'un point upsert-v2 (~20 bytes par float)."""
    dense = p.get("dense") or []
    colbert = p.get("colbert_vecs") or []
    sparse_values = p.get("sparse_values") or []
    colbert_floats = sum(len(v) for v in colbert)
    return (len(dense) + colbert_floats + len(sparse_values)) * 20


class MacClient:
    def __init__(self, base_url: str, transport: Optional[httpx.BaseTransport] = None,
                 timeout: float = 180.0):
        self.base = base_url.rstrip("/")
        headers: dict[str, str] = {}
        cf_id = os.environ.get("CF_ACCESS_CLIENT_ID", "").strip()
        cf_secret = os.environ.get("CF_ACCESS_CLIENT_SECRET", "").strip()
        if cf_id and cf_secret:
            headers["CF-Access-Client-Id"] = cf_id
            headers["CF-Access-Client-Secret"] = cf_secret
            print(f"[MacClient] CF Access headers injectés (id={cf_id[:12]}…, "
                  f"secret len={len(cf_secret)})")
        else:
            print(f"[MacClient] ⚠️ AUCUN header CF Access — id={bool(cf_id)} "
                  f"secret={bool(cf_secret)}. Le Mac renverra du HTML de login.")
        self._client = httpx.Client(transport=transport, timeout=timeout,
                                    http2=True, follow_redirects=True,
                                    headers=headers)

    def list_files(self, root: str) -> list[dict]:
        r = self._client.get(f"{self.base}/files/list", params={"root": root})
        r.raise_for_status()
        ct = r.headers.get("content-type", "")
        if "application/json" not in ct:
            snippet = r.text[:300].replace("\n", " ")
            raise RuntimeError(
                f"/files/list a renvoyé {ct!r} au lieu de JSON. "
                f"CF Access rejette probablement le service token. "
                f"Vérifier Zero Trust → Access → Applications → docfinder → "
                f"Policies → 'Service Auth' avec Include > Service Token > "
                f"<token Colab>. Corps reçu: {snippet!r}"
            )
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
        if not points:
            return {"upserted": 0}

        batches: list[list[dict]] = []
        current: list[dict] = []
        current_bytes = 0
        for p in points:
            est = _estimate_point_bytes(p)
            if current and current_bytes + est > _UPSERT_CLIENT_SOFT_LIMIT_BYTES:
                batches.append(current)
                current = []
                current_bytes = 0
            current.append(p)
            current_bytes += est
        if current:
            batches.append(current)

        upserted = 0
        for i, batch in enumerate(batches):
            r = self._client.post(
                f"{self.base}/admin/upsert-v2", json={"points": batch}
            )
            r.raise_for_status()
            data = r.json()
            upserted += int(data.get("upserted", len(batch)))
            if len(batches) > 1:
                print(
                    f"[MacClient] upsert-v2 batch {i + 1}/{len(batches)} "
                    f"({len(batch)} points, ~{sum(_estimate_point_bytes(p) for p in batch) // (1024 * 1024)} MiB)",
                    flush=True,
                )
        return {"upserted": upserted}

    def push_progress(self, report: dict) -> None:
        r = self._client.post(f"{self.base}/admin/progress", json=report)
        r.raise_for_status()
