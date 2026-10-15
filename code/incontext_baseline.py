"""F3 (docs/HYPOTHESIS.md A9): in-context editing baseline for the same facts and new objects as the layer sweep.

For each edit fact and language, the counterfactual is given in the prompt instead of in the weights:
  en-context : "Fact: <English canonical prompt> <new English target>.\n" + canonical prompt in the test language
  own-context: the same statement written in the test language (upper bound).
Metrics as for the weight edit: dP of the new target's full-answer probability and greedy success. Descriptive; no hypothesis.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path

import torch

from causal_trace_v2 import prompt_and_target
from common import LANGS, MODELS, load_facts, load_model
from knowledge_edit_v2 import greedy, seq_logprob
from matching import strict_match


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama", choices=list(MODELS))
    ap.add_argument("--sweep-run", required=True, help="layer-sweep run dir (gives the edit facts and their new objects)")
    ap.add_argument("--tag", default="incontext")
    a = ap.parse_args()
    spec = MODELS[a.model]
    facts = {f["id"]: f for f in load_facts(spec["facts_v2"])}
    rows = [json.loads(l) for l in (Path(a.sweep_run) / "records.jsonl").read_text(encoding="utf8").splitlines()]
    pairs = {r["fact"]: r["new_fact"] for r in rows}
    out = Path("analysis") / f"run_{dt.date.today():%Y%m%d}_{a.tag}_{a.model}"
    out.mkdir(parents=True, exist_ok=False)
    model = load_model(a.model)
    tok = model.tokenizer
    res = []
    for fid, nid in sorted(pairs.items()):
        f, new = facts[fid], facts[nid]
        row = {"fact": fid, "new_fact": nid, "relation": f["relation"]}
        for mode in ("en_context", "own_context"):
            for lg in LANGS:
                L = f["lang"][lg]
                tgt = new["lang"][lg]["target"]
                src = f["lang"]["en"] if mode == "en_context" else L
                src_tgt = new["lang"]["en"]["target"] if mode == "en_context" else tgt
                context = f"Fact: {src['prompts'][0]} {src_tgt}.\n"
                prompt = context + L["prompts"][0]
                lp_before = seq_logprob(model, L["prompts"][0], tgt)
                lp_after = seq_logprob(model, prompt, tgt)
                n_gen = len(prompt_and_target(tok, prompt, tgt)[1]) + 2
                text = greedy(model, prompt, n_gen)
                row[f"{mode}/{lg}"] = {"logp_before": lp_before, "logp_after": lp_after, "dP": math.exp(lp_after) - math.exp(lp_before),
                                       "success": strict_match(text, [tgt]) is not None}
        res.append(row)
        print(fid, {k: (round(v["dP"], 2), v["success"]) for k, v in row.items() if "/" in k}, flush=True)
    (out / "records.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf8")
    summ = {}
    import numpy as np
    for k in [f"{m}/{lg}" for m in ("en_context", "own_context") for lg in LANGS]:
        summ[k] = {"median_dP": float(np.median([r[k]["dP"] for r in res])), "success_rate": float(np.mean([r[k]["success"] for r in res]))}
    (out / "summary.json").write_text(json.dumps({"n_facts": len(res), "summary": summ}, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
