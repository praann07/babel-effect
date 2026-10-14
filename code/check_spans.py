"""P1 gate: verify subject token spans exist and are contiguous for every kept fact, language and phrasing; report length divergence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from common import LANGS, MODELS, subject_span


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama", choices=list(MODELS))
    a = ap.parse_args()
    spec = MODELS[a.model]
    tok = AutoTokenizer.from_pretrained(spec["dir"])
    facts = json.loads(Path(spec["facts_v2"]).read_text(encoding="utf8"))["facts"]
    errors, rows = [], []
    for f in facts:
        spans = {}
        for lg in LANGS:
            L = f["lang"][lg]
            for k, p in enumerate(L["prompts"]):
                try:
                    ids, pos = subject_span(tok, p, L["subject"])
                except ValueError as e:
                    errors.append((f["id"], lg, k, str(e)))
                    continue
                if pos != list(range(pos[0], pos[-1] + 1)):
                    errors.append((f["id"], lg, k, "non-contiguous span"))
                if k == 0:
                    spans[lg] = {"n_tokens": len(ids), "span_len": len(pos), "last": pos[-1], "from_end": len(ids) - 1 - pos[-1]}
        rows.append({"id": f["id"], "relation": f["relation"], **{f"{lg}_{k}": v for lg, s in spans.items() for k, v in s.items()}})
    lens = {lg: np.array([r[f"{lg}_span_len"] for r in rows if f"{lg}_span_len" in r]) for lg in LANGS}
    diverge = [r["id"] for r in rows if abs(r["hi_span_len"] - r["en_span_len"]) > 2 or abs(r["es_span_len"] - r["en_span_len"]) > 2]
    summary = {"model": spec["tl_name"], "n_facts": len(facts), "errors": errors,
               "span_len_median": {lg: float(np.median(v)) for lg, v in lens.items()},
               "span_len_max": {lg: int(v.max()) for lg, v in lens.items()},
               "n_diverging_gt2_vs_en": len(diverge), "diverging_ids": diverge}
    out = Path("data/v2") / f"span_check_{a.model}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
