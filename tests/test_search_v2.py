from unittest.mock import MagicMock
import numpy as np


def test_search_v2_uses_query_points_with_prefetch():
    from server.search import search_v2
    qdrant = MagicMock()

    class Hit:
        def __init__(self, pid, score, payload):
            self.id, self.score, self.payload = pid, score, payload

    qdrant.query_points.return_value = MagicMock(points=[
        Hit("p1", 0.9, {"doc_id": "d", "content": "hello",
                         "title": "t", "path": "p", "abs_path": "/p",
                         "doc_type": "pdf", "keywords_doc": ["k"]}),
    ])

    embedder = MagicMock()
    embedder.encode.return_value = type("E", (), {
        "dense": [[0.0] * 1024],
        "sparse": [([1, 2], [0.5, 0.3])],
        "colbert": [[[0.0] * 1024, [0.0] * 1024]],
        "lexical_weights": [{1: 0.5, 2: 0.3}],
    })()

    results = search_v2(qdrant, embedder, "my query",
                        collection="docfinder_v2", limit=10)

    kwargs = qdrant.query_points.call_args.kwargs
    assert kwargs["collection_name"] == "docfinder_v2"
    assert kwargs["using"] == "colbert"
    prefetch = kwargs["prefetch"]
    usings = [pf.using for pf in prefetch]
    assert "dense" in usings and "sparse" in usings
    assert kwargs["limit"] == 10
    assert len(results) == 1
    assert results[0].doc_id == "d"
