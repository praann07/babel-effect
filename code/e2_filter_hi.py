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

from common import DTYPE, MODELS
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
HI_CANDS16 = {  # fixed in docs/HYPOTHESIS.md A16 before this was run: the two A15 candidates plus two question/answer-format candidates
    r: HI_CANDS[r] + q for r, q in {
        "capital_of": ["प्रश्न: {s} की राजधानी क्या है?\nउत्तर:", "प्रश्न: {s} देश की राजधानी का नाम क्या है?\nउत्तर:"],
        "capital_country": ["प्रश्न: वह कौन सा देश है जिसकी राजधानी {s} है?\nउत्तर:", "प्रश्न: {s} किस देश की राजधानी है?\nउत्तर:"],
        "landmark_city": ["प्रश्न: {s} किस शहर में स्थित है?\nउत्तर:", "प्रश्न: वह कौन सा शहर है जहाँ {s} स्थित है?\nउत्तर:"],
        "landmark_country": ["प्रश्न: {s} किस देश में स्थित है?\nउत्तर:", "प्रश्न: वह कौन सा देश है जहाँ {s} स्थित है?\nउत्तर:"],
        "city_country": ["प्रश्न: {s} किस देश में स्थित है?\nउत्तर:", "प्रश्न: वह कौन सा देश है जहाँ {s} स्थित है?\nउत्तर:"]}.items()}
MIN_KEPT16 = 8
HI_CANDS_A23 = {  # fixed in docs/HYPOTHESIS.md A23 before this was run: three question/answer-format candidates per relation
    "official_language": ["प्रश्न: {s} की आधिकारिक भाषा क्या है?\nउत्तर:", "प्रश्न: {s} में मुख्य रूप से कौन सी भाषा बोली जाती है?\nउत्तर:", "{s} की आधिकारिक भाषा है"],
    "currency": ["प्रश्न: {s} की मुद्रा क्या है?\nउत्तर:", "प्रश्न: {s} की आधिकारिक मुद्रा का नाम क्या है?\nउत्तर:", "{s} की मुद्रा है"],
    "profession": ["प्रश्न: {s} किस पेशे के लिए प्रसिद्ध हैं?\nउत्तर:", "प्रश्न: {s} का पेशा क्या था?\nउत्तर:", "{s} एक प्रसिद्ध"],
    "book_author": ["प्रश्न: {s} के लेखक कौन हैं?\nउत्तर:", "प्रश्न: {s} किसने लिखी?\nउत्तर:", "{s} के लेखक हैं"],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["llama", "llama_base", "gemma2"])
    ap.add_argument("--tag", default="e2hfilter")
    ap.add_argument("--set", default="base", choices=["base", "a23"], help="A23: non-geography relations (reads e2a23_facts_<model>.json, 8 tuning facts, writes e2ha23_facts_<model>.json)")
    ap.add_argument("--a16", action="store_true", help="A16: Q/A-format candidates, no 50% relation threshold, relations need >= 8 kept facts; writes e2h2_facts_<model>.json")
    a = ap.parse_args()
    spec = MODELS[a.model]
    run = Path("analysis") / f"run_{dt.date.today():%Y%m%d}_{a.tag}_{a.model}"
    run.mkdir(parents=True, exist_ok=False)
    global N_TUNE
    a23 = a.set == "a23"
    if a23:
        from build_facts_v2 import fmt
        from e2_filter import POOL_A23_FILE, N_TUNE_A23
        a.a16, N_TUNE, POOLS[a.model] = True, N_TUNE_A23, [POOL_A23_FILE]
    else:
        fmt = None
    en_t = {r: v[0] for r, v in json.loads(Path(f"data/v2/{'e2a23' if a23 else 'e2'}_facts_{a.model}.json").read_text(encoding="utf8"))["templates"].items()}
    HC = HI_CANDS_A23 if a23 else (HI_CANDS16 if a.a16 else HI_CANDS)
    cands = [f for p in POOLS[a.model] for f in json.loads(Path(p).read_text(encoding="utf8"))["facts"] if f["relation"] in HC]
    cands.sort(key=lambda f: (f["relation"], f["id"]))
    tok = AutoTokenizer.from_pretrained(spec["dir"]); tok.padding_side = "left"; tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(spec["dir"], torch_dtype=DTYPE, attn_implementation="eager").to("cuda").eval()
    hi_t, sel = {}, {}
    for r in HC:
        fs = [f for f in cands if f["relation"] == r][:N_TUNE]
        accs = []
        for tp in HC[r]:
            g = generate(model, tok, [tp.format(s=f["lang"]["hi"]["subject"], S=f["lang"]["hi"]["subject"]) for f in fs])
            accs.append(sum(strict_match(x, [f["lang"]["hi"]["object"], *f["lang"]["hi"]["object_alts"]]) is not None for x, f in zip(g, fs)) / len(fs))
        best = max(range(len(accs)), key=lambda i: (accs[i], -i))
        hi_t[r] = HC[r][best]; sel[r] = dict(zip(HC[r], accs))
        print(r, sel[r], "->", hi_t[r], flush=True)
    fill = (lambda t, lg, s_: fmt(t, lg, s_)) if fmt else (lambda t, lg, s_: t.format(s=s_))
    prompts = {(f["id"], lg): fill(en_t[f["relation"]] if lg == "en" else hi_t[f["relation"]], lg, f["lang"][lg]["subject"]) for f in cands for lg in ("en", "hi")}
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
        t["template_ok"] = bool(t["eval_both"] >= MIN_KEPT16) if a.a16 else bool(t["tune_n"] and t["tune_en"] / t["tune_n"] >= 0.5 and t["tune_hi"] / t["tune_n"] >= 0.5)
    kept = [f for f in kept if table[f["relation"]]["template_ok"]]
    out = {"model": spec["tl_name"], "templates": {"en": en_t, "hi": hi_t}, "hi_candidate_accuracy": sel, "summary": table, "n_kept": len(kept), "facts": kept}
    Path(f"data/v2/{'e2ha23' if a23 else ('e2h2' if a.a16 else 'e2h')}_facts_{a.model}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf8")
    (run / "accuracy.json").write_text(json.dumps(table, indent=1))
    print(json.dumps(table, indent=1)); print("kept", len(kept))


if __name__ == "__main__":
    main()
