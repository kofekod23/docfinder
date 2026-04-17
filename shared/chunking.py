"""Paragraph-based chunker (spec §6).

Pure function, tokenizer injected so tests can use a trivial counter and
production code can inject BGE-M3's XLM-R tokenizer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, List

_BLANK_LINE_RE = re.compile(r"\n\s*\n+")
_HEADING_MD_RE = re.compile(r"^#{1,6}\s+\S")
_HEADING_NUM_RE = re.compile(r"^\d+\.\s+\S")


@dataclass(frozen=True)
class Chunk:
    text: str
    token_count: int
    para_indices: tuple[int, ...]


def split_paragraphs(text: str) -> List[str]:
    """Split on runs of blank lines, strip, drop empties."""
    return [p.strip() for p in _BLANK_LINE_RE.split(text) if p.strip()]


def is_heading(line: str) -> bool:
    """Detect markdown heading, numeric heading, or all-caps line."""
    first = line.strip().splitlines()[0] if line.strip() else ""
    if not first:
        return False
    if _HEADING_MD_RE.match(first):
        return True
    if _HEADING_NUM_RE.match(first):
        return True
    letters = [c for c in first if c.isalpha()]
    if len(letters) >= 3 and all(c.isupper() for c in letters):
        return True
    return False


def chunk_paragraphs(
    paragraphs: List[str],
    tokenize_len: Callable[[str], int],
    target_min: int = 400,
    target_max: int = 600,
    hard_max: int = 800,
    hard_min: int = 100,
) -> List[Chunk]:
    """Greedy accumulator with forced break on headings and 1-para overlap."""
    if not paragraphs:
        return []

    chunks: List[Chunk] = []
    buf: List[str] = []
    buf_idx: List[int] = []
    buf_tokens = 0
    last_was_heading = False

    def flush() -> None:
        nonlocal buf, buf_idx, buf_tokens, last_was_heading
        if not buf:
            return
        text = "\n\n".join(buf)
        chunks.append(Chunk(text=text, token_count=buf_tokens,
                            para_indices=tuple(buf_idx)))
        # overlap: keep last paragraph unless it was a heading
        if not last_was_heading and len(buf) > 1:
            buf = [buf[-1]]
            buf_idx = [buf_idx[-1]]
            buf_tokens = tokenize_len(buf[0])
        else:
            buf, buf_idx, buf_tokens = [], [], 0
        last_was_heading = False

    for i, para in enumerate(paragraphs):
        tk = tokenize_len(para)
        heading = is_heading(para)

        if heading and buf:
            flush()

        if buf_tokens + tk > hard_max and buf:
            flush()

        buf.append(para)
        buf_idx.append(i)
        buf_tokens += tk
        last_was_heading = heading

        if buf_tokens >= target_max:
            flush()
        elif buf_tokens >= target_min:
            # look-ahead: break if next para would exceed target_max
            if i + 1 < len(paragraphs):
                nxt = tokenize_len(paragraphs[i + 1])
                if buf_tokens + nxt > target_max:
                    flush()

    if buf and buf_tokens >= hard_min:
        flush()
    elif buf and chunks:
        # append to previous chunk rather than emit a tiny one
        prev = chunks.pop()
        merged = prev.text + "\n\n" + "\n\n".join(buf)
        chunks.append(Chunk(text=merged,
                            token_count=prev.token_count + buf_tokens,
                            para_indices=prev.para_indices + tuple(buf_idx)))
    elif buf:
        # single short paragraph: emit anyway (better than losing content)
        chunks.append(Chunk(text="\n\n".join(buf),
                            token_count=buf_tokens,
                            para_indices=tuple(buf_idx)))

    return chunks
