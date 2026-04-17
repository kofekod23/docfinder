"""Test de plusieurs requêtes pour identifier les patterns de résultats faibles."""
from __future__ import annotations

import httpx

QUERIES = [
    "documents médicaux",
    "SEO",
    "PV",
    "liste de médicaments",
    "promesse de vente",
    "assurance santé",
]


def main() -> None:
    for q in QUERIES:
        resp = httpx.post(
            "http://127.0.0.1:8000/search",
            json={"query": q, "limit": 10},
            timeout=120.0,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        print(f"\n=== '{q}' — top {len(results)} ===")
        for i, r in enumerate(results):
            print(f"#{i+1:>2} score={r.get('score', 0):.4f}  {r.get('path', '?')[:80]}")


if __name__ == "__main__":
    main()
