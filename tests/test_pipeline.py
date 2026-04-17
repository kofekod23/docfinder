import asyncio
from unittest.mock import MagicMock, AsyncMock
from pathlib import Path
import pytest


@pytest.mark.asyncio
async def test_process_one_doc_downloads_extracts_embeds_upserts(tmp_path: Path):
    from colab.pipeline import process_one_doc

    mac = MagicMock()
    mac.download = MagicMock(return_value=b"hello world")
    mac.upsert_chunks_v2 = MagicMock(return_value={"ok": True})

    extractor = MagicMock()
    extractor.return_value = type("R", (), {
        "text": "para one\n\npara two",
        "doc_type": "txt", "page_count": None, "ocr_pages": (),
    })()

    chunker = MagicMock(return_value=[type("C", (), {
        "text": "para one\n\npara two", "token_count": 3, "para_indices": (0, 1),
    })()])

    embedder = MagicMock()
    embedder.encode.return_value = type("E", (), {
        "dense": [[0.0] * 1024],
        "sparse": [([1, 2], [0.5, 0.3])],
        "colbert": [[[0.0] * 1024]],
        "lexical_weights": [{1: 0.5, 2: 0.3}],
    })()

    tokenizer_decode = lambda ids: f"tok{ids[0]}"

    meta = {"path": "a.txt", "abs_path": "/x/a.txt", "doc_id": "d1",
            "size": 10, "mtime": 1, "mode": "full_or_head"}

    await process_one_doc(
        meta, mac_client=mac, extractor=extractor, chunker=chunker,
        embedder=embedder, tokenizer_decode=tokenizer_decode, tmp_dir=tmp_path,
    )

    assert mac.upsert_chunks_v2.called
    point = mac.upsert_chunks_v2.call_args.args[0][0]
    assert point["payload"]["doc_id"] == "d1"
    assert point["payload"]["chunk_idx"] == 0
    assert len(point["dense"]) == 1024


def test_checkpoint_roundtrip(tmp_path: Path):
    from colab.pipeline import Checkpoint
    ck = Checkpoint(tmp_path / "ck.json")
    ck.mark("a", "done")
    ck.mark("b", "failed")
    ck.save()
    ck2 = Checkpoint(tmp_path / "ck.json")
    ck2.load()
    assert ck2.status("a") == "done"
    assert ck2.status("b") == "failed"
    assert ck2.status("c") is None
