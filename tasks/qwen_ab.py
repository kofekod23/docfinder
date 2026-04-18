"""A/B test Qwen3-Embedding vs BGE-M3 sur les 145 requêtes ground-truth.

Stratégie :
  - Deux runs /search successifs :
      * baseline : DOCFINDER_COLLECTION=docfinder_v2      + DOCFINDER_EMBEDDER=bgem3
      * variant  : DOCFINDER_COLLECTION=docfinder_v2_qwen + DOCFINDER_EMBEDDER=qwen
  - Pour chaque requête, on récupère top-10 et on calcule le rank du 1er
    résultat dont path contient `path_substring` (ground truth).
  - MRR@10 = mean(1 / rank) sur toutes les requêtes, 0 si hors top-10.
  - Comparatif : wins/losses/same + delta MRR global.

Le serveur doit être redémarré entre les deux runs (env switch via
DOCFINDER_EMBEDDER + DOCFINDER_COLLECTION). On préfère deux invocations
subprocess distinctes plutôt qu'un reload dynamique.

Usage :
    python -m tasks.qwen_ab run --mode bgem3   > out_bgem3.json
    python -m tasks.qwen_ab run --mode qwen    > out_qwen.json
    python -m tasks.qwen_ab compare out_bgem3.json out_qwen.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tasks.queries_semantic import QUERIES  # noqa: E402

SEARCH_URL = os.environ.get("DOCFINDER_SEARCH_URL", "http://localhost:8000/search")
TOP_K = 10


def rank_of_hit(results: list[dict], needle: str) -> int | None:
    """1-based rank du premier résultat dont `path` contient `needle` (lowercased)."""
    needle_norm = needle.lower()
    for i, r in enumerate(results, start=1):
        path = (r.get("path") or "").lower()
        if needle_norm in path:
            return i
    return None


def run_batch(client: httpx.Client, mode: str) -> dict:
    out: dict = {"mode": mode, "results": []}
    for query_text, path_substring in QUERIES:
        t0 = time.perf_counter()
        resp = client.post(SEARCH_URL, json={"query": query_text, "limit": TOP_K})
        dt = time.perf_counter() - t0
        if resp.status_code != 200:
            out["results"].append(
                {"query": query_text, "error": f"HTTP {resp.status_code}"}
            )
            continue
        hits = resp.json().get("results", [])
        rank = rank_of_hit(hits, path_substring)
        out["results"].append(
            {
                "query": query_text,
                "target": path_substring,
                "rank": rank,
                "latency_s": round(dt, 3),
            }
        )
    return out


def mrr_at_k(run: dict, k: int = TOP_K) -> float:
    ranks = [r.get("rank") for r in run["results"] if "rank" in r]
    rr = [(1.0 / r) if (r is not None and r <= k) else 0.0 for r in ranks]
    return sum(rr) / len(rr) if rr else 0.0


def compare(a_path: str, b_path: str) -> None:
    a = json.loads(Path(a_path).read_text())
    b = json.loads(Path(b_path).read_text())
    assert len(a["results"]) == len(b["results"])

    wins_b, wins_a, same = 0, 0, 0
    for ra, rb in zip(a["results"], b["results"]):
        na, nb = ra.get("rank"), rb.get("rank")
        if na == nb:
            same += 1
        elif nb is not None and (na is None or nb < na):
            wins_b += 1
        else:
            wins_a += 1

    mrr_a = mrr_at_k(a)
    mrr_b = mrr_at_k(b)
    print(
        f"MRR@{TOP_K}  {a['mode']}={mrr_a:.4f}  {b['mode']}={mrr_b:.4f}  "
        f"delta={mrr_b - mrr_a:+.4f}"
    )
    print(f"Wins  {b['mode']}={wins_b}  {a['mode']}={wins_a}  same={same}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run")
    p_run.add_argument("--mode", required=True, choices=["bgem3", "qwen"])

    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("baseline")
    p_cmp.add_argument("variant")

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.cmd == "run":
        with httpx.Client(timeout=30.0) as client:
            out = run_batch(client, args.mode)
        print(json.dumps(out, ensure_ascii=False, indent=2))
    elif args.cmd == "compare":
        compare(args.baseline, args.variant)


if __name__ == "__main__":
    main()
