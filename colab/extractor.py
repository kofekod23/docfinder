"""Size-aware extractor (spec §5).

Decision matrix:
  size > 50 MB                        -> filename_only
  size > 20 MB OR (pdf AND pages>30)  -> head_only (first 5-10 pages)
  else                                -> full
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

TWENTY_MB = 20 * 1024 * 1024
FIFTY_MB = 50 * 1024 * 1024
LONG_PDF_PAGES = 30
HEAD_PAGES = 8


@dataclass(frozen=True)
class ExtractionResult:
    text: str
    doc_type: str
    page_count: Optional[int]
    ocr_pages: tuple[int, ...] = ()


def decide_mode(size: int, page_count: Optional[int]) -> str:
    if size > FIFTY_MB:
        return "filename_only"
    if size > TWENTY_MB or (page_count is not None and page_count > LONG_PDF_PAGES):
        return "head_only"
    return "full"


def _detect_type(path: Path) -> str:
    ext = path.suffix.lower().lstrip(".")
    return {"pdf": "pdf", "docx": "docx", "doc": "docx",
            "md": "md", "markdown": "md", "txt": "txt"}.get(ext, ext or "bin")


def extract_text(path: Path, mode: str) -> ExtractionResult:
    doc_type = _detect_type(path)
    if mode == "filename_only":
        return ExtractionResult(text=path.name, doc_type=doc_type, page_count=None)
    if doc_type == "txt" or doc_type == "md":
        return ExtractionResult(
            text=path.read_text(encoding="utf-8", errors="replace"),
            doc_type=doc_type, page_count=None,
        )
    if doc_type == "pdf":
        return _extract_pdf(path, mode)
    if doc_type == "docx":
        return _extract_docx(path)
    return ExtractionResult(text="", doc_type=doc_type, page_count=None)


def _extract_pdf(path: Path, mode: str) -> ExtractionResult:
    import fitz  # PyMuPDF
    doc = fitz.open(str(path))
    total = doc.page_count
    limit = HEAD_PAGES if mode == "head_only" else total
    parts: list[str] = []
    ocr_pages: list[int] = []
    for i in range(min(limit, total)):
        page = doc.load_page(i)
        txt = page.get_text()
        if len(txt.strip()) < 10:
            ocr_pages.append(i)  # signalled; OCR done downstream in colab/ocr.py
            continue
        parts.append(txt)
    return ExtractionResult(
        text="\n\n".join(parts),
        doc_type="pdf",
        page_count=total,
        ocr_pages=tuple(ocr_pages),
    )


def _extract_docx(path: Path) -> ExtractionResult:
    import docx
    doc = docx.Document(str(path))
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    return ExtractionResult(text="\n\n".join(parts), doc_type="docx", page_count=None)
