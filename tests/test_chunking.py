import pytest

from shared.chunking import split_paragraphs, is_heading, chunk_paragraphs, Chunk


def test_split_paragraphs_collapses_blank_runs():
    text = "para one\n\n\npara two\n\npara three"
    assert split_paragraphs(text) == ["para one", "para two", "para three"]


def test_split_paragraphs_strips_and_drops_empty():
    assert split_paragraphs("  \n\n  hello  \n\n\n") == ["hello"]


@pytest.mark.parametrize("line,expected", [
    ("# Title", True),
    ("### Sub", True),
    ("####### too many", False),
    ("INTRODUCTION", True),
    ("1. Section one", True),
    ("12.  Something", True),
    ("normal paragraph", False),
    ("Not a heading.", False),
])
def test_is_heading(line, expected):
    assert is_heading(line) is expected


def _tok(text: str) -> int:
    """Fake whitespace tokenizer used in tests."""
    return len(text.split())


def test_chunk_paragraphs_min_and_max_bounds():
    paras = ["x" * 5] * 50  # each paragraph = 1 token (whitespace-split)
    chunks = chunk_paragraphs(paras, tokenize_len=_tok,
                              target_min=4, target_max=6, hard_max=8, hard_min=2)
    assert all(c.token_count >= 2 for c in chunks)
    assert all(c.token_count <= 8 for c in chunks)


def test_chunk_paragraphs_forced_break_on_heading():
    paras = ["body one", "# Heading", "body two body two body two"]
    chunks = chunk_paragraphs(paras, tokenize_len=_tok,
                              target_min=1, target_max=4, hard_max=10, hard_min=1)
    assert any(c.text.startswith("# Heading") for c in chunks)
    texts_before_heading = [c.text for c in chunks if "body one" in c.text]
    assert "# Heading" not in texts_before_heading[0]


def test_chunk_paragraphs_overlap_one_paragraph():
    paras = [f"p{i}" for i in range(6)]
    chunks = chunk_paragraphs(paras, tokenize_len=_tok,
                              target_min=2, target_max=2, hard_max=3, hard_min=1)
    # consecutive chunks should share the last paragraph of the previous one
    for prev, cur in zip(chunks, chunks[1:]):
        if prev.text.split("\n\n")[-1] == "# Heading":
            continue
        assert prev.text.split("\n\n")[-1] == cur.text.split("\n\n")[0]
