"""BGE-M3 wrapper: dense + sparse + ColBERT + keyword extraction (spec §3, §8)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple, Dict, Iterable

_STOPWORDS = {
    # minimal FR/EN stoplist, expand later
    "le", "la", "les", "un", "une", "des", "et", "ou", "de", "du", "au", "aux",
    "à", "en", "pour", "par", "sur", "dans", "avec", "sans",
    "the", "a", "an", "and", "or", "of", "in", "on", "for", "with", "to", "at",
    "is", "are", "was", "were",
}


@dataclass(frozen=True)
class EncodeResult:
    dense: List[List[float]]
    sparse: List[Tuple[List[int], List[float]]]
    colbert: List[List[List[float]]]
    lexical_weights: List[Dict[int, float]]


class BGEM3Wrapper:
    def __init__(self, model=None):
        self._model = model

    def _model_or_build(self):
        if self._model is None:
            self._model = _build_model()
        return self._model

    def encode(self, texts: List[str], batch_size: int = 32,
               max_length: int = 512) -> EncodeResult:
        m = self._model_or_build()
        out = m.encode(
            texts,
            batch_size=batch_size,
            max_length=max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True,
        )
        dense = [v.tolist() for v in out["dense_vecs"]]
        sparse: List[Tuple[List[int], List[float]]] = []
        for w in out["lexical_weights"]:
            items = sorted(w.items(), key=lambda kv: -kv[1])
            idx = [int(k) for k, _ in items]
            val = [float(v) for _, v in items]
            sparse.append((idx, val))
        colbert = [[row.tolist() for row in mat] for mat in out["colbert_vecs"]]
        return EncodeResult(dense=dense, sparse=sparse, colbert=colbert,
                            lexical_weights=out["lexical_weights"])


def _build_model():
    from FlagEmbedding import BGEM3FlagModel
    return BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device="cuda")


def keywords_from_weights(
    weights: Dict[int, float],
    decode: Callable[[List[int]], str],
    top_k: int = 10,
    min_len: int = 3,
) -> List[str]:
    items = sorted(weights.items(), key=lambda kv: -kv[1])
    out: list[str] = []
    seen: set[str] = set()
    for tid, _ in items:
        tok = decode([int(tid)]).strip()
        if not tok or tok.startswith("##") or tok.startswith("▁"):
            tok = tok.lstrip("#▁ ")
        if len(tok) < min_len:
            continue
        if tok.lower() in _STOPWORDS:
            continue
        if all(ch.isdigit() for ch in tok):
            continue
        key = tok.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(tok)
        if len(out) >= top_k:
            break
    return out


def aggregate_doc_keywords(
    chunk_weights_and_lengths: Iterable[Tuple[Dict[int, float], int]],
    decode: Callable[[List[int]], str],
    top_k: int = 15,
) -> List[str]:
    agg: Dict[int, float] = {}
    for weights, length in chunk_weights_and_lengths:
        weight = float(max(length, 1))
        for tid, val in weights.items():
            agg[tid] = agg.get(tid, 0.0) + val * weight
    return keywords_from_weights(agg, decode=decode, top_k=top_k)
