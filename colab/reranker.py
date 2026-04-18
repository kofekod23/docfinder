"""BGE-reranker-v2-m3 wrapper — cross-encoder scoring.

Charge le modèle une seule fois sur GPU fp16. Les pairs (query, doc) sont
scorées en lot par `compute_score`. Score normalisé entre 0 et 1.
"""
from __future__ import annotations

from typing import List, Tuple

from FlagEmbedding import FlagReranker

MODEL_NAME = "BAAI/bge-reranker-v2-m3"


class BGERerankerWrapper:
    def __init__(self, model_name: str = MODEL_NAME, use_fp16: bool = True) -> None:
        self._model_name = model_name
        self._use_fp16 = use_fp16
        self._model: FlagReranker | None = None

    def _load(self) -> FlagReranker:
        if self._model is None:
            self._model = FlagReranker(self._model_name, use_fp16=self._use_fp16)
        return self._model

    def rerank(self, pairs: List[Tuple[str, str]]) -> List[float]:
        if not pairs:
            return []
        model = self._load()
        scores = model.compute_score([list(p) for p in pairs], normalize=True)
        if isinstance(scores, float):
            return [float(scores)]
        return [float(s) for s in scores]
