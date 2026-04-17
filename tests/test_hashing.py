from pathlib import Path
import hashlib
import pytest

from shared.hashing import doc_id_for, file_hash_for


def test_doc_id_is_sha1_of_abs_path():
    abs_path = "/Users/julien/Documents/a.pdf"
    expected = hashlib.sha1(abs_path.encode("utf-8")).hexdigest()
    assert doc_id_for(abs_path) == expected


def test_doc_id_stable_for_same_path():
    assert doc_id_for("/a/b.pdf") == doc_id_for("/a/b.pdf")


def test_doc_id_differs_for_different_paths():
    assert doc_id_for("/a/b.pdf") != doc_id_for("/a/c.pdf")


def test_file_hash_reads_first_mb_plus_size(tmp_path: Path):
    p = tmp_path / "sample.bin"
    payload = b"x" * (2 * 1024 * 1024)  # 2 MB
    p.write_bytes(payload)
    expected_head = payload[: 1024 * 1024]
    expected_size = p.stat().st_size
    h = hashlib.sha1()
    h.update(expected_head)
    h.update(str(expected_size).encode("utf-8"))
    assert file_hash_for(p) == h.hexdigest()


def test_file_hash_small_file(tmp_path: Path):
    p = tmp_path / "small.txt"
    p.write_bytes(b"hello")
    h = hashlib.sha1()
    h.update(b"hello")
    h.update(b"5")
    assert file_hash_for(p) == h.hexdigest()
