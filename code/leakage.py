"""A17: answer-in-subject leakage. A fact is leaky if, in any language, the (accent/case-folded) answer or an alternative
answer is a substring of the (folded) subject (e.g. Mexico City -> Mexico). Rule fixed in docs/HYPOTHESIS.md A17."""
from __future__ import annotations

import glob
import json
import unicodedata


def fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in s if not unicodedata.combining(c)).strip()


def is_leaky_lang(d: dict) -> bool:
    subj = fold(d["subject"])
    answers = [d.get("object") or d.get("target")] + list(d.get("object_alts") or []) + ([d["target"]] if d.get("target") else [])
    return any(a and len(fold(a)) >= 3 and fold(a) in subj for a in answers)


def leaky_ids(pattern: str = "data/v2/*.json") -> set[str]:
    out: set[str] = set()
    for f in glob.glob(pattern):
        try:
            j = json.load(open(f, encoding="utf8"))
        except Exception:
            continue
        for x in (j.get("facts", []) if isinstance(j, dict) else []):
            if isinstance(x, dict) and "lang" in x and any(is_leaky_lang(d) for d in x["lang"].values() if isinstance(d, dict) and "subject" in d):
                out.add(x["id"])
    return out


if __name__ == "__main__":
    L = leaky_ids()
    print(len(L), sorted(L)[:60])
