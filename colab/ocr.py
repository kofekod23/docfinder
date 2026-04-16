"""easyocr wrapper, lazy-loaded (spec §7)."""
from __future__ import annotations

from typing import Optional
import numpy as np

_reader = None


def _build_reader():
    import easyocr
    return easyocr.Reader(["fr", "en"], gpu=True)


def _reset_reader() -> None:
    global _reader
    _reader = None


def ocr_page(image: np.ndarray) -> str:
    """Run OCR on a rendered PDF page (RGB numpy array). Join lines with \\n."""
    global _reader
    if _reader is None:
        _reader = _build_reader()
    lines = _reader.readtext(image, detail=0, paragraph=True)
    return "\n".join(lines)
