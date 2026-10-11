"""Text normalisation and tokenisation helpers for Nepali (Devanagari) text."""
from __future__ import annotations

import re
import unicodedata

_URL = re.compile(r"https?://\S+|www\.\S+")
_HTML_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")

# Zero-width characters. ZWJ/ZWNJ are used inconsistently in Nepali spellings
# (e.g. गर्यो vs गर्‍यो), so dropping them merges those spelling variants.
_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍﻿"), None)
_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

# Python's \w does not match Devanagari vowel signs (matras), so scikit-learn's
# default token pattern would cut words in the middle. Split on whitespace and
# punctuation (including the danda "।") instead.
_TOKEN = re.compile(r"[^\s।॥,.!?;:\"'“”‘’()\[\]{}\-–—|/\\]+")


def clean_text(text: str) -> str:
    """Normalise a raw article/headline string.

    NFC-normalise, strip URLs and HTML tags, drop zero-width characters,
    map Devanagari digits to ASCII digits and collapse whitespace.
    """
    text = unicodedata.normalize("NFC", str(text))
    text = _URL.sub(" ", text)
    text = _HTML_TAG.sub(" ", text)
    text = text.translate(_ZERO_WIDTH).translate(_DEVANAGARI_DIGITS)
    return _WHITESPACE.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    """Whitespace/punctuation tokenizer that keeps Devanagari words intact."""
    return _TOKEN.findall(text)
