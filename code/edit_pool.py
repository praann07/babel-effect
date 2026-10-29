"""A20: pooled F7 + A20 Llama edit results (traced / L//2 / layer 21): medians with bootstrap-over-facts 95% CIs. Stored records only (CPU)."""
import datetime as dt, glob, json
from pathlib import Path
import numpy as np

SEED = 20261029
load = lambda p: [json.loads(l) for l in open(p, encoding="utf8")]
recs = load("analysis/run_20261020_p4f7_llama/records.jsonl") + load(sorted(glob.glob("analysis/run_*_p4a20_llama"))[-1] + "/records.jsonl")
rng = np.random.default_rng(SEED)
cond = lambda r: "traced" if r["is_traced"] else ("mid" if r["layer"] == 14 else "L21")


def med_ci(x):
    x = np.array(x, float)
    return {"median": float(np.median(x)), "ci95": [float(v) for v in np.percentile([np.median(rng.choice(x, len(x))) for _ in range(5000)], [2.5, 97.5])], "n": int(len(x))}


out = {"n_facts": len({r["fact"] for r in recs})}
for c in ("traced", "mid", "L21"):
    R = [r for r in recs if cond(r) == c]; ok = [r for r in R if r["en"]["dP"] > 0.1]
    out[c] = {"n": len(R), "en_success": float(np.mean([r["en"]["success"] for r in R])), "n_for_transfer": len(ok),
              "es_transfer": med_ci([r["es"]["dP"] / r["en"]["dP"] for r in ok]), "hi_transfer": med_ci([r["hi"]["dP"] / r["en"]["dP"] for r in ok]),
              "specificity": med_ci([r["specificity_max_abs_dP"] for r in R]), "ppl_ratio": med_ci([r["ppl_ratio"] for r in R])}
d = Path("analysis") / f"run_{dt.date.today():%Y%m%d}_editpool_A20"; d.mkdir(parents=True, exist_ok=False)
(d / "results.json").write_text(json.dumps(out, indent=1)); print(json.dumps({c: out[c]["es_transfer"] for c in ("traced", "mid", "L21")}), "->", d)
