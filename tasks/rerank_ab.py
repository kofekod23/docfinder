# tasks/rerank_ab.py
"""A/B test du reranker cross-encoder sur les 145 queries.

Usage :
    RERANK_ENABLED=false USE_V2=true uvicorn server.main:app --port 8000 &
    # mesure baseline
    python3 tasks/rerank_ab.py --mode off --out tasks/rerank_off.csv
    # restart avec reranker
    RERANK_ENABLED=true COLAB_RERANK_URL=... uvicorn server.main:app --port 8000 &
    python3 tasks/rerank_ab.py --mode on --out tasks/rerank_on.csv
    python3 tasks/rerank_ab.py --compare tasks/rerank_off.csv tasks/rerank_on.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import httpx

from tasks.queries_semantic import QUERIES


def _hit_rank(results: list[dict], path_substring: str) -> int:
    for i, r in enumerate(results):
        if path_substring.lower() in (r.get("path") or "").lower():
            return i + 1
    return 0


def run(mode: str, out_path: Path, base_url: str) -> None:
    rows = []
    with httpx.Client(timeout=60.0) as c:
        for q_tuple in QUERIES:
            query_text, path_substring = q_tuple
            resp = c.post(
                f"{base_url}/search",
                json={"query": query_text, "limit": 10},
                headers={"Content-Type": "application/json"},
            )
            results = resp.json().get("results", [])
            rank = _hit_rank(results, path_substring)
            rows.append({"query": query_text, "rank": rank, "mode": mode})
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["query", "rank", "mode"])
        w.writeheader()
        w.writerows(rows)
    mrr = sum(1 / r["rank"] for r in rows if r["rank"] > 0) / len(rows)
    print(f"[{mode}] MRR@10 = {mrr:.4f}  ({len([r for r in rows if r['rank'] > 0])}/{len(rows)} hits)")


def compare(off_csv: Path, on_csv: Path) -> None:
    off = {r["query"]: int(r["rank"]) for r in csv.DictReader(off_csv.open())}
    on = {r["query"]: int(r["rank"]) for r in csv.DictReader(on_csv.open())}
    wins, losses, same = 0, 0, 0
    for q in off:
        a, b = off[q], on.get(q, 0)
        a_s = 0.0 if a == 0 else 1 / a
        b_s = 0.0 if b == 0 else 1 / b
        if b_s > a_s: wins += 1
        elif b_s < a_s: losses += 1
        else: same += 1
    off_mrr = sum(1 / r for r in off.values() if r > 0) / len(off)
    on_mrr = sum(1 / r for r in on.values() if r > 0) / len(on)
    print(f"MRR off: {off_mrr:.4f}   MRR on: {on_mrr:.4f}   delta: {on_mrr - off_mrr:+.4f}")
    print(f"wins: {wins}   losses: {losses}   same: {same}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["on", "off"])
    p.add_argument("--out", type=Path)
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--compare", nargs=2, type=Path, metavar=("OFF_CSV", "ON_CSV"))
    a = p.parse_args()
    if a.compare:
        compare(*a.compare)
        return
    if not (a.mode and a.out):
        p.error("--mode and --out required unless --compare")
    run(a.mode, a.out, a.base_url)


if __name__ == "__main__":
    main()
