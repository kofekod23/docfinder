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
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


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


def _explode_oversize(
    paragraphs: List[str],
    tokenize_len: Callable[[str], int],
    hard_max: int,
) -> List[str]:
    """Split paragraphs exceeding hard_max into sentence-packed pieces."""
    result: List[str] = []
    for para in paragraphs:
        if tokenize_len(para) <= hard_max:
            result.append(para)
            continue
        sentences = _SENTENCE_SPLIT_RE.split(para)
        buf: List[str] = []
        buf_tok = 0
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            st = tokenize_len(s)
            if st > hard_max:
                if buf:
                    result.append(" ".join(buf))
                    buf, buf_tok = [], 0
                # A single sentence exceeds hard_max: force-split by words so the
                # embedder receives something shorter than its context window.
                words = s.split()
                step = max(1, len(words) * hard_max // max(1, st))
                for i in range(0, len(words), step):
                    result.append(" ".join(words[i:i + step]))
                continue
            if buf_tok + st > hard_max and buf:
                result.append(" ".join(buf))
                buf, buf_tok = [s], st
            else:
                buf.append(s)
                buf_tok += st
        if buf:
            result.append(" ".join(buf))
    return result


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
    paragraphs = _explode_oversize(paragraphs, tokenize_len, hard_max)

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
