from colab.reranker import BGERerankerWrapper


def test_rerank_returns_score_per_pair(monkeypatch):
    class StubFlag:
        def compute_score(self, pairs, normalize=True):
            return [0.9, 0.1, 0.5]

    monkeypatch.setattr(
        "colab.reranker.FlagReranker", lambda *a, **kw: StubFlag()
    )
    wrapper = BGERerankerWrapper()
    scores = wrapper.rerank([
        ("q", "docA"), ("q", "docB"), ("q", "docC"),
    ])
    assert scores == [0.9, 0.1, 0.5]


def test_rerank_empty_returns_empty(monkeypatch):
    monkeypatch.setattr(
        "colab.reranker.FlagReranker",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("should not load")),
    )
    wrapper = BGERerankerWrapper()
    wrapper._model = "loaded"  # short-circuit lazy load
    assert wrapper.rerank([]) == []
