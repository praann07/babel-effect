"""Answer matching shared by template selection and the v2 filter (docs/HYPOTHESIS.md A6, A7)."""
from __future__ import annotations

import re
import unicodedata

_DROP = dict.fromkeys(map(ord, "़​‌‍﻿"))  # nukta, zero-width characters


def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s).translate(_DROP).lower().replace("-", " ").replace("’", "'")
    return re.sub(r"\s+", " ", s).strip()


def strict_match(gen: str, forms: list[str]) -> str | None:
    """Generation (leading spaces ignored; a leading newline fails) begins with one of the forms.

    Returns the matched prefix of the generation in its original script, or None.
    """
    g = gen.lstrip(" ")
    if g[:1] in ("\n", "\r", "\t"):  # a leading line break fails (A6)
        return None
    gn = norm(g)
    best, best_len = None, -1
    for f in forms:
        fn = norm(f)
        if gn.startswith(fn) and len(fn) > best_len:
            for k in range(max(1, len(f) - 4), min(len(g), len(f) + 8) + 1):  # recover the original-script prefix
                if norm(g[:k]) == fn:
                    best, best_len = g[:k], len(fn)
                    break
    return best


def lenient_match(gen: str, forms: list[str]) -> bool:
    """Object occurs within the first line and the first 60 characters of the generation."""
    head = norm(gen.lstrip(" ").split("\n")[0][:60])
    return any(norm(f) in head for f in forms)
