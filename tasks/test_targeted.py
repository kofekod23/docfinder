"""Test ciblé : 20 requêtes inventées d'après les documents indexés.

Chaque requête a un document cible attendu. On vérifie qu'il apparaît
dans le top-5 et sa position exacte.
"""
from __future__ import annotations

import unicodedata

import httpx


def _strip_accents(s: str) -> str:
    nfd = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")

CASES: list[tuple[str, str]] = [
    ("audit SEO Tryba", "audit seo"),
    ("promesse de vente Persoz Giraud", "promesse de vente"),
    ("CV directeur SEO Ecommerce", "cv-directeur-seo"),
    ("contrat télétravail scolaire", "teletravail"),
    ("feuille de soin Roy", "feuille de soin"),
    ("bulletins de salaire Yakarouler", "bulletins-salaire"),
    ("procuration acquérir Persoz", "procuration"),
    ("mutuelle Rakuten 2020", "mutuelle rakuten"),
    ("grille prévoyance Axa", "prevoyance axa"),
    ("store locator Tryba", "store locator"),
    ("solde de tout compte", "solde-de-tout-compte"),
    ("Louis Vuitton GEO", "geo-louis-vuitton"),
    ("argumentations Rakuten", "argumentations"),
    ("charte informatique déconnexion", "charte informatique"),
    ("facture Novotel Égypte", "novotel"),
    ("questionnaire lycée", "questionnaire_lycee"),
    ("contestation LG Rakuten", "lg contestation"),
    ("prêt étude", "pret etude"),
    ("SEPA Carrefour assurance chat", "sepa carrefour"),
    ("litiges ebay", "litiges-ebay"),
]


def main() -> None:
    total = len(CASES)
    top1 = 0
    top5 = 0
    fails: list[tuple[str, str]] = []
    print(f"{'Query':<40} {'Pos':>4}  {'Score':>7}  Target")
    print("-" * 100)
    for query, target in CASES:
        resp = httpx.post(
            "http://127.0.0.1:8000/search",
            json={"query": query, "limit": 10},
            timeout=120.0,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        pos = None
        score = 0.0
        target_norm = _strip_accents(target)
        for i, r in enumerate(results):
            if target_norm in _strip_accents(r.get("path", "")):
                pos = i + 1
                score = r.get("score", 0.0)
                break
        if pos == 1:
            top1 += 1
            mark = "✓1"
        elif pos and pos <= 5:
            top5 += 1
            mark = f"✓{pos}"
        elif pos:
            mark = f" {pos}"
        else:
            fails.append((query, target))
            mark = " ✗"
        print(f"{query:<40} {mark:>4}  {score:>7.4f}  {target}")

    print("-" * 100)
    print(f"Top-1 : {top1}/{total}  ({100*top1/total:.0f}%)")
    print(f"Top-5 : {top1+top5}/{total}  ({100*(top1+top5)/total:.0f}%)")
    if fails:
        print(f"\nÉchecs (absents du top-10) :")
        for q, t in fails:
            print(f"  - {q!r} → cherchait '{t}'")


if __name__ == "__main__":
    main()
