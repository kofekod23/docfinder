from unittest.mock import MagicMock, patch
import numpy as np
import pytest


def test_ocr_page_joins_lines_with_newline():
    with patch("colab.ocr._build_reader") as build:
        reader = MagicMock()
        reader.readtext.return_value = ["bonjour", "le monde"]
        build.return_value = reader
        from colab.ocr import ocr_page, _reset_reader
        _reset_reader()
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        text = ocr_page(img)
        assert text == "bonjour\nle monde"
        reader.readtext.assert_called_once()
        # reader is cached
        ocr_page(img)
        build.assert_called_once()


def test_ocr_page_empty_when_no_detections():
    with patch("colab.ocr._build_reader") as build:
        reader = MagicMock()
        reader.readtext.return_value = []
        build.return_value = reader
        from colab.ocr import ocr_page, _reset_reader
        _reset_reader()
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        assert ocr_page(img) == ""
