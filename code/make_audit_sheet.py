"""Build the native-speaker audit sheet: every EN/ES/HI fact used in any analysis, one row per fact, with blank columns for a checker.

Usage: python code/make_audit_sheet.py      -> data/audit/native_speaker_audit.csv (UTF-8 with BOM, opens in Excel) and data/audit/AUDIT_INSTRUCTIONS.md
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

D = Path("data/v2")
POOLS = ["facts_candidates.json", "facts_candidates_extra.json", "facts_candidates_gemma2.json", "facts_candidates_extra_gemma2.json", "facts_candidates_llama1b.json", "facts_candidates_bloom1b7.json"]
USED = {  # analysis -> kept-fact files
    "tracing_llama": ["facts_llama.json", "facts_extra_llama.json"], "tracing_gemma2": ["facts_gemma2.json", "facts_extra_gemma2.json"],
    "swap_en_es_llama": ["e2_facts_llama.json"], "swap_en_es_gemma2": ["e2_facts_gemma2.json"], "swap_en_es_llama_base": ["e2_facts_llama_base.json"],
    "swap_hi_llama": ["e2h_facts_llama.json", "e2h2_facts_llama.json"], "swap_hi_gemma2": ["e2h_facts_gemma2.json", "e2h2_facts_gemma2.json"], "swap_hi_llama_base": ["e2h2_facts_llama_base.json"],
}


def load(f: str) -> list[dict]:
    p = D / f
    return json.loads(p.read_text(encoding="utf8"))["facts"] if p.exists() else []


def main() -> None:
    cand: dict[str, dict] = {}
    for f in POOLS:
        for x in load(f):
            cand.setdefault(x["id"], x)
    used: dict[str, list[str]] = {}
    for tag, files in USED.items():
        for f in files:
            for x in load(f):
                used.setdefault(x["id"], [])
                if tag not in used[x["id"]]:
                    used[x["id"]].append(tag)
                cand.setdefault(x["id"], x)
    rows = []
    for i in sorted(used):
        x = cand[i]
        L = x["lang"]
        g = lambda lg, k: (L.get(lg) or {}).get(k, "")
        pr = lambda lg: ((L.get(lg) or {}).get("prompts") or [(L.get(lg) or {}).get("prompt", "")])[0]
        rows.append([i, x["relation"], g("en", "subject"), g("en", "object"), pr("en"), g("es", "subject"), g("es", "object"), pr("es"), g("hi", "subject"), g("hi", "object"), pr("hi"),
                     "; ".join(used[i]), "", "", "", "", ""])
    hdr = ["fact_id", "relation", "EN_subject", "EN_answer", "EN_prompt", "ES_subject", "ES_answer", "ES_prompt", "HI_subject", "HI_answer", "HI_prompt", "used_in",
           "EN_fact_true? (Y/N)", "ES_ok? (Y/N)", "HI_ok? (Y/N)", "correction (write the right word/sentence)", "comment"]
    out = Path("data/audit")
    out.mkdir(exist_ok=True)
    with open(out / "native_speaker_audit.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(hdr)
        w.writerows(rows)
    (out / "AUDIT_INSTRUCTIONS.md").write_text(f"""# Native-speaker check of the facts ({len(rows)} rows)

Each row is one fact written in English, Spanish and Hindi. Please check:
1. **EN_fact_true?** Is the English fact true? (e.g. "The capital of France is Paris".)
2. **ES_ok? / HI_ok?** Are the Spanish and Hindi subject, answer and prompt correct, natural and the same fact as the English one? Say N if a word is wrong, an alternative common spelling should also be accepted, or the sentence is unnatural or ambiguous.
3. If N, write the correction in the **correction** column (say which language) and any other note in **comment**.

The prompt is an unfinished sentence: the model must continue it with the answer. A prompt may be short or unusual; only mark N if the meaning is wrong or a native speaker would not write it.
Return the file as CSV. Rows are not changed in place; corrections are applied by the researcher and the affected analyses rerun.
Time: about {max(1, len(rows) // 60)} hours for Spanish plus Hindi, for two checkers in parallel.
""", encoding="utf8")
    print(len(rows), "facts ->", out)


if __name__ == "__main__":
    main()
