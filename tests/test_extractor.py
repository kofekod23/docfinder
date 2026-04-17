from pathlib import Path
import pytest

from colab.extractor import decide_mode, extract_text, ExtractionResult


def test_decide_mode_filename_only_above_50mb():
    assert decide_mode(size=60 * 1024 * 1024, page_count=None) == "filename_only"


def test_decide_mode_head_only_for_large_file():
    assert decide_mode(size=25 * 1024 * 1024, page_count=None) == "head_only"


def test_decide_mode_head_only_for_long_pdf():
    assert decide_mode(size=5 * 1024 * 1024, page_count=40) == "head_only"


def test_decide_mode_full():
    assert decide_mode(size=1 * 1024 * 1024, page_count=5) == "full"


def test_extract_text_txt_full(tmp_path: Path):
    p = tmp_path / "a.txt"
    p.write_text("hello\n\nworld\n")
    res = extract_text(p, mode="full")
    assert "hello" in res.text and "world" in res.text
    assert res.doc_type == "txt"


def test_extract_text_filename_only(tmp_path: Path):
    p = tmp_path / "huge.pdf"
    p.write_bytes(b"%PDF-stub")
    res = extract_text(p, mode="filename_only")
    assert res.text.strip() == "huge.pdf"
    assert res.doc_type == "pdf"
