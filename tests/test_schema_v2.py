import pytest
from pydantic import ValidationError

from shared.schema import DocumentChunkV2, ProgressReport, IndexedStateRequest


def test_document_chunk_v2_minimal():
    c = DocumentChunkV2(
        doc_id="abc", path="a.pdf", abs_path="/a/a.pdf", doc_type="pdf",
        title="a", mtime=1_700_000_000, file_hash="h",
        content="hello world", keywords_chunk=["hello"], keywords_doc=["hello"],
        page_range=[1, 2], chunk_idx=0, chunk_total=1,
        dense=[0.0] * 1024,
        sparse_indices=[10, 20], sparse_values=[0.5, 0.3],
        colbert_vecs=[[0.0] * 1024, [0.0] * 1024],
    )
    assert c.doc_id == "abc"
    assert len(c.dense) == 1024
    assert len(c.sparse_indices) == len(c.sparse_values)


def test_document_chunk_v2_sparse_length_mismatch_rejected():
    with pytest.raises(ValidationError):
        DocumentChunkV2(
            doc_id="abc", path="a.pdf", abs_path="/a/a.pdf", doc_type="pdf",
            title="a", mtime=1, file_hash="h", content="x",
            keywords_chunk=[], keywords_doc=[],
            page_range=None, chunk_idx=0, chunk_total=1,
            dense=[0.0] * 1024,
            sparse_indices=[1, 2, 3], sparse_values=[0.5],
            colbert_vecs=[[0.0] * 1024],
        )


def test_progress_report_roundtrip():
    p = ProgressReport(
        total=10, done=3, failed=0, current_doc="x.pdf",
        gpu_util_pct=80, vram_used_mb=9000, chunks_per_sec=30.0,
        eta_seconds=120,
        stage_counts={"downloaded": 3, "extracted": 3, "embedded": 3},
    )
    assert p.model_dump()["stage_counts"]["downloaded"] == 3


def test_indexed_state_request_accepts_empty_filter():
    r = IndexedStateRequest(doc_ids=None)
    assert r.doc_ids is None
