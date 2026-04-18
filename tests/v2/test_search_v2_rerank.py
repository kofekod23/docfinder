from shared.schema import SearchResult
from server.search import rerank_results


class StubRe:
    def rerank(self, query, documents):
        # inverse de l'ordre reçu pour prouver le re-ordering
        return [float(i) for i in range(len(documents), 0, -1)]


def _mk(doc_id, excerpt, score=1.0):
    return SearchResult(
        chunk_id=doc_id, doc_id=doc_id, title="", path="", abs_path="",
        doc_type="", score=score, excerpt=excerpt, keywords=[],
    )


def test_rerank_results_reorders_by_cross_encoder():
    results = [_mk("A", "aaa"), _mk("B", "bbb"), _mk("C", "ccc")]
    out = rerank_results("query", results, StubRe(), top_n=3)
    assert [r.doc_id for r in out] == ["A", "B", "C"]
    # scores strictement décroissants
    assert out[0].score > out[1].score > out[2].score


def test_rerank_results_preserves_tail_beyond_top_n():
    results = [_mk(c, c) for c in "ABCDE"]
    out = rerank_results("query", results, StubRe(), top_n=2)
    assert len(out) == 5
    # les 2 premiers ont été re-scorés, les 3 derniers gardent leur ordre
    assert [r.doc_id for r in out[2:]] == ["C", "D", "E"]


def test_rerank_results_empty_passthrough():
    assert rerank_results("q", [], StubRe(), top_n=10) == []


def test_rerank_results_no_reranker_passthrough():
    results = [_mk("A", "a"), _mk("B", "b")]
    assert rerank_results("q", results, None, top_n=2) is results
