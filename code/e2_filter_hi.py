"""A15 fact filter for the English-Hindi interchange: keep candidates the model answers correctly in BOTH EN and HI with fixed prefix templates.

EN templates are the ones the model used in A13 (data/v2/e2_facts_<model>.json). HI templates are selected per relation among the fixed
candidates below by accuracy on the 30 tuning facts (ties: earlier). Writes data/v2/e2h_facts_<model>.json and a write-once run dir.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import MODELS
from e2_filter import N_TUNE, POOLS
from filter_facts_v2 import generate
from matching import strict_match

HI_CANDS = {  # fixed in docs/HYPOTHESIS.md A15 before this was run
    "capital_of": ["यह सर्वविदित है कि {s} की राजधानी है", "{s} देश की राजधानी का नाम है"],
    "capital_country": ["वह देश जिसकी राजधानी {s} है, उसका नाम है", "जिस देश की राजधानी {s} है, वह देश है"],
    "landmark_city": ["वह शहर जहाँ {s} स्थित है, उसका नाम है", "{s} जिस शहर में स्थित है, वह शहर है"],
    "landmark_country": ["वह देश जहाँ {s} स्थित है, उसका नाम है", "{s} जिस देश में स्थित है, वह देश है"],
}
HI_CANDS["city_country"] = HI_CANDS["landmark_country"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["llama", "gemma2"])
    ap.add_argument("--tag", default="e2hfilter")
    a = ap.parse_args()
    spec = MODELS[a.model]
    run = Path("analysis") / f"run_{dt.date.today():%Y%m%d}_{a.tag}_{a.model}"
    run.mkdir(parents=True, exist_ok=False)
    en_t = {r: v[0] for r, v in json.loads(Path(f"data/v2/e2_facts_{a.model}.json").read_text(encoding="utf8"))["templates"].items()}
    cands = [f for p in POOLS[a.model] for f in json.loads(Path(p).read_text(encoding="utf8"))["facts"] if f["relation"] in HI_CANDS]
    cands.sort(key=lambda f: (f["relation"], f["id"]))
    tok = AutoTokenizer.from_pretrained(spec["dir"]); tok.padding_side = "left"; tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(spec["dir"], torch_dtype=torch.bfloat16, attn_implementation="eager").to("cuda").eval()
    hi_t, sel = {}, {}
    for r in HI_CANDS:
        fs = [f for f in cands if f["relation"] == r][:N_TUNE]
        accs = []
        for tp in HI_CANDS[r]:
            g = generate(model, tok, [tp.format(s=f["lang"]["hi"]["subject"]) for f in fs])
            accs.append(sum(strict_match(x, [f["lang"]["hi"]["object"], *f["lang"]["hi"]["object_alts"]]) is not None for x, f in zip(g, fs)) / len(fs))
        best = max(range(len(accs)), key=lambda i: (accs[i], -i))
        hi_t[r] = HI_CANDS[r][best]; sel[r] = dict(zip(HI_CANDS[r], accs))
        print(r, sel[r], "->", hi_t[r], flush=True)
    prompts = {(f["id"], lg): (en_t[f["relation"]] if lg == "en" else hi_t[f["relation"]]).format(s=f["lang"][lg]["subject"]) for f in cands for lg in ("en", "hi")}
    keys = list(prompts)
    gens = dict(zip(keys, generate(model, tok, [prompts[k] for k in keys])))
    (run / "generations.json").write_text(json.dumps({f"{k[0]}|{k[1]}": {"prompt": prompts[k], "gen": gens[k]} for k in keys}, ensure_ascii=False, indent=1), encoding="utf8")
    idx: dict[str, int] = {}
    kept, table = [], {}
    for f in cands:
        r = f["relation"]; i = idx.get(r, 0); idx[r] = i + 1
        split = "tune" if i < N_TUNE else "eval"
        rec, ok = {}, True
        t = table.setdefault(r, {"tune_n": 0, "tune_en": 0, "tune_hi": 0, "eval_n": 0, "eval_both": 0})
        for lg in ("en", "hi"):
            L = f["lang"][lg]
            m = strict_match(gens[(f["id"], lg)], [L["object"], *L["object_alts"]])
            rec[lg] = {"subject": L["subject"], "object": L["object"], "object_alts": L["object_alts"], "prompt": prompts[(f["id"], lg)], "target": m}
            ok &= m is not None
            if split == "tune":
                t[f"tune_{lg}"] += int(m is not None)
        t["tune_n" if split == "tune" else "eval_n"] += 1
        if split == "eval" and ok:
            t["eval_both"] += 1
            kept.append({"id": f["id"], "relation": r, "seed": f["seed"], "lang": rec})
    for r, t in table.items():
        t["template_ok"] = bool(t["tune_n"] and t["tune_en"] / t["tune_n"] >= 0.5 and t["tune_hi"] / t["tune_n"] >= 0.5)
    kept = [f for f in kept if table[f["relation"]]["template_ok"]]
    out = {"model": spec["tl_name"], "templates": {"en": en_t, "hi": hi_t}, "hi_candidate_accuracy": sel, "summary": table, "n_kept": len(kept), "facts": kept}
    Path(f"data/v2/e2h_facts_{a.model}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf8")
    (run / "accuracy.json").write_text(json.dumps(table, indent=1))
    print(json.dumps(table, indent=1)); print("kept", len(kept))


if __name__ == "__main__":
    main()
