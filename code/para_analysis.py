"""A21 analysis: transfer of the English edit to (a) unseen English phrasings, (b) Spanish and Hindi (prompts[0] and the other phrasings).
transfer = dP(target) / dP(English prompts[0]); facts with English dP <= 0.1 are dropped (as in A20). Median with bootstrap-over-facts 95% CI. CPU.
Usage: python code/para_analysis.py RUN_DIR [--ref RUN_DIR ...]   (--ref: earlier A20/F7 run dirs for the reproduction sanity check)
"""
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

args = sys.argv[1:]
refs = [Path(args[i + 1]) for i, x in enumerate(args) if x == "--ref"]
run = Path(args[0])
rng = np.random.default_rng(20261030)
load = lambda p: [json.loads(l) for l in open(p / "records.jsonl", encoding="utf8")]
rows = load(run)
cond = lambda r: "traced" if r["is_traced"] else "mid"


def med(x):
    x = np.array(x, float)
    return {"median": float(np.median(x)), "ci95": [float(v) for v in np.percentile([np.median(rng.choice(x, len(x))) for _ in range(5000)], [2.5, 97.5])], "n": int(len(x))}


def para(r, lg):
    return np.mean([r["para"][k]["dP"] for k in r["para"] if k.startswith(lg)]) / r["en"]["dP"]


out = {"n_facts": len({r["fact"] for r in rows})}
for c in ("traced", "mid"):
    R = [r for r in rows if cond(r) == c and r["en"]["dP"] > 0.1 and "para" in r]
    o = {"n": len(R)}
    o["en_para"] = med([para(r, "en") for r in R])
    o["es_main"] = med([r["es"]["dP"] / r["en"]["dP"] for r in R])
    o["hi_main"] = med([r["hi"]["dP"] / r["en"]["dP"] for r in R])
    o["es_para"] = med([para(r, "es") for r in R])
    o["hi_para"] = med([para(r, "hi") for r in R])
    d = np.array([para(r, "en") - r["es"]["dP"] / r["en"]["dP"] for r in R])
    o["en_para_minus_es_main"] = med(d)
    o["en_para_minus_hi_main"] = med([para(r, "en") - r["hi"]["dP"] / r["en"]["dP"] for r in R])
    o["frac_facts_es_below_en_para"] = float(np.mean(d > 0))
    out[c] = o

if refs:  # sanity check: the re-fitted edit reproduces the stored numbers (per fact and layer)
    old = {(r["fact"], r["layer"]): r for p in refs for r in load(p)}
    pairs = [(r, old[(r["fact"], r["layer"])]) for r in rows if (r["fact"], r["layer"]) in old]
    if pairs:
        out["reproduction"] = {"n": len(pairs), "median_abs_diff_en_dP": float(np.median([abs(a["en"]["dP"] - b["en"]["dP"]) for a, b in pairs])),
                               "median_abs_diff_es_dP": float(np.median([abs(a["es"]["dP"] - b["es"]["dP"]) for a, b in pairs]))}

d_ = Path("analysis") / f"run_{dt.date.today():%Y%m%d}_para_analysis_A21"
d_.mkdir(parents=True, exist_ok=False)
(d_ / "results.json").write_text(json.dumps(out, indent=1))
for c in ("traced", "mid"):
    print(c, {k: (round(v["median"], 2) if isinstance(v, dict) else v) for k, v in out[c].items()})
print(out.get("reproduction"), "->", d_)
