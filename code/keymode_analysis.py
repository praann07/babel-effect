"""A24 analysis: does writing the edit at a shared (multi-language) key raise transfer to Spanish and Hindi? CPU.

Compares each key mode of an A24 run (records with "key_mode") with the English-key baseline records of the A21 run (same facts, same layers,
same replacement answers), paired over facts. transfer = dP(target) / dP(English prompts[0]) of the same edit; facts with English dP <= 0.1 in either
condition are dropped. "held-out" = mean over the phrasings prompts[1:] of the language (not used to fit keys or delta).
Usage: python code/keymode_analysis.py A24_RUN --base A21_RUN
"""
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

args = sys.argv[1:]
run, base = Path(args[0]), Path(args[args.index("--base") + 1])
rng = np.random.default_rng(20261030)
load = lambda p: [json.loads(l) for l in open(p / "records.jsonl", encoding="utf8")]
cond = lambda r: "traced" if r["is_traced"] else "mid"


def ci(x):
    x = np.array(x, float)
    return {"median": float(np.median(x)), "ci95": [float(v) for v in np.percentile([np.median(rng.choice(x, len(x))) for _ in range(5000)], [2.5, 97.5])], "n": int(len(x))}


def held(r, lg):
    return float(np.mean([r["para"][k]["dP"] for k in r["para"] if k.startswith(lg)]))


def metrics(r):
    e = r["en"]["dP"]
    return {"en_dP": e, "en_ok": float(r["en"]["success"]), "es_main": r["es"]["dP"] / e, "hi_main": r["hi"]["dP"] / e, "es_held": held(r, "es") / e, "hi_held": held(r, "hi") / e,
            "spec": r["specificity_max_abs_dP"], "ppl": r["ppl_ratio"]}


B = {(r["fact"], r["layer"]): r for r in load(base) if "para" in r}
out = {}
for mode in sorted({r["key_mode"] for r in load(run)}):
    rows = [r for r in load(run) if r["key_mode"] == mode and "para" in r]
    out[mode] = {"key_cos_es_median": float(np.median([r["key_cos"]["es"] for r in rows])), "key_cos_hi_median": float(np.median([r["key_cos"]["hi"] for r in rows]))}
    for c in ("traced", "mid"):
        pairs = [(metrics(r), metrics(B[(r["fact"], r["layer"])])) for r in rows if cond(r) == c and (r["fact"], r["layer"]) in B]
        ok = [(m, b) for m, b in pairs if m["en_dP"] > 0.1 and b["en_dP"] > 0.1]
        o = {"n_pairs": len(pairs), "n_used": len(ok), "en_success": float(np.mean([m["en_ok"] for m, _ in pairs])) if pairs else None,
             "base_en_success": float(np.mean([b["en_ok"] for _, b in pairs])) if pairs else None}
        for k in ("es_main", "hi_main", "es_held", "hi_held"):
            if ok:
                o[k] = ci([m[k] for m, _ in ok]); o[k + "_base"] = ci([b[k] for _, b in ok]); o[k + "_diff"] = ci([m[k] - b[k] for m, b in ok])
        if pairs:
            o["spec_median"] = float(np.median([m["spec"] for m, _ in pairs])); o["spec_base_median"] = float(np.median([b["spec"] for _, b in pairs]))
            o["ppl_median"] = float(np.median([m["ppl"] for m, _ in pairs]))
        out[mode][c] = o
d_ = Path("analysis") / f"run_{dt.date.today():%Y%m%d}_keymode_analysis_A24"
d_.mkdir(parents=True, exist_ok=False)
(d_ / "results.json").write_text(json.dumps(out, indent=1))
for m in out:
    for c in ("traced", "mid"):
        o = out[m][c]
        print(m, c, {k: (round(v["median"], 2) if isinstance(v, dict) else (round(v, 3) if isinstance(v, float) else v)) for k, v in o.items() if not k.endswith("_base") or k == "base_en_success"})
print("->", d_)
