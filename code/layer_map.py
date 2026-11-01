"""A28 layer map: where does the English-key edit reach Spanish and Hindi, and does key similarity predict it within a layer? CPU.

Input: an edit run with --key-modes en,mean_all --para over several layers. Per layer and language L in {es, hi}, over facts with English dP > 0.1:
  median held-out transfer (mean dP over prompts[1:] of L / dP English prompts[0]) with a bootstrap-over-facts 95% CI,
  Spearman rho between cos(k_EN, k_L) and that transfer, and the paired gain of the mean_all key over the English key (descriptive).
Pooled statistic per language: mean over layers with >= MIN_N usable facts of the per-layer rho, bootstrap over facts (all layers resampled together).
Supported: pooled rho >= 0.20 with CI above 0 and Holm-adjusted p < 0.05 over the two languages (one-sided sign-flip-free test: share of bootstrap means <= 0).
Usage: python code/layer_map.py EDIT_RUN [EDIT_RUN ...]   (several runs, e.g. the shards of one design, are read together)
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

from key_similarity import ALPHA, BOOT, SEED, held_transfer, holm, spearman

POOLED_MIN, MIN_N = 0.20, 20
LANGS = ("es", "hi")


def usable(rows: list[dict]) -> dict[int, dict[str, dict]]:
    """layer -> fact -> {transfer per language, key cosine per language, mean_all transfer per language}, English-key rows with English dP > 0.1."""
    cos = {(r["fact"], r["layer"]): r["key_cos"] for r in rows if "key_cos" in r}
    shared = {(r["fact"], r["layer"]): r for r in rows if r.get("key_mode") == "mean_all" and "para" in r and r["en"]["dP"] > 0.1}
    out: dict[int, dict[str, dict]] = {}
    for r in rows:
        if r.get("key_mode", "en") != "en" or "para" not in r or r["en"]["dP"] <= 0.1 or (r["fact"], r["layer"]) not in cos:
            continue
        k = (r["fact"], r["layer"])
        out.setdefault(r["layer"], {})[r["fact"]] = {"y": {lg: held_transfer(r, lg) for lg in LANGS}, "x": cos[k],
                                                     "gain": {lg: held_transfer(shared[k], lg) - held_transfer(r, lg) for lg in LANGS} if k in shared else None}
    return out


def layer_rho(cells: dict[str, dict], lg: str, facts: list[str]) -> float:
    pts = [(cells[f]["x"][lg], cells[f]["y"][lg]) for f in facts if f in cells]
    return spearman([p[0] for p in pts], [p[1] for p in pts]) if len(pts) >= 5 else float("nan")


def pooled(by_layer: dict[int, dict[str, dict]], lg: str, facts: list[str]) -> float:
    v = [layer_rho(c, lg, facts) for c in by_layer.values()]
    v = [x for x in v if np.isfinite(x)]
    return float(np.mean(v)) if v else float("nan")


def main() -> None:
    rows = [json.loads(l) for run in sys.argv[1:] for l in open(Path(run) / "records.jsonl", encoding="utf8")]
    by_layer = usable(rows)
    keep = {l: c for l, c in by_layer.items() if len(c) >= MIN_N}
    rng = np.random.default_rng(SEED)
    all_facts = sorted({f for c in keep.values() for f in c})
    out = {"layers": {}, "pooled": {}, "n_facts": len(all_facts), "layers_used_in_pooled": sorted(keep)}

    def med_ci(x):
        x = np.array(x, float)
        return {"median": float(np.median(x)), "ci95": [float(v) for v in np.percentile([np.median(rng.choice(x, len(x))) for _ in range(BOOT)], [2.5, 97.5])], "n": int(len(x))}

    for layer in sorted(by_layer):
        c = by_layer[layer]
        o = {"n": len(c)}
        for lg in LANGS:
            o[lg] = {"transfer": med_ci([v["y"][lg] for v in c.values()]), "key_cos_median": float(np.median([v["x"][lg] for v in c.values()])),
                     "rho": layer_rho(c, lg, list(c))}
            g = [v["gain"][lg] for v in c.values() if v["gain"]]
            if g:
                o[lg]["mean_all_gain"] = med_ci(g)
        out["layers"][str(layer)] = o
    p_list = []
    for lg in LANGS if keep else ():  # no layer with enough facts (an unfinished run): the per-layer table is still written
        obs = pooled(keep, lg, all_facts)
        boot = [pooled(keep, lg, list(rng.choice(all_facts, len(all_facts)))) for _ in range(BOOT)]
        boot = [b for b in boot if np.isfinite(b)]
        out["pooled"][lg] = {"mean_within_layer_rho": obs, "ci95": [float(v) for v in np.percentile(boot, [2.5, 97.5])], "p_one_sided": float((np.sum(np.array(boot) <= 0) + 1) / (len(boot) + 1))}
        p_list.append(out["pooled"][lg]["p_one_sided"])
    for lg, a in zip(LANGS if keep else (), holm(p_list) if keep else ()):
        o = out["pooled"][lg]
        o["p_holm"] = a
        o["supported"] = bool(o["mean_within_layer_rho"] >= POOLED_MIN and o["ci95"][0] > 0 and a < ALPHA)
    d = Path("analysis") / f"run_{dt.date.today():%Y%m%d}_layer_map_A28"
    d.mkdir(parents=True, exist_ok=False)
    (d / "results.json").write_text(json.dumps(out, indent=1))
    for layer, o in out["layers"].items():  # the results are already saved: printing must never fail (a retry would meet the write-once folder)
        print("layer", layer, "n", o["n"], json.dumps({lg: [o[lg]["transfer"]["median"], o[lg]["rho"]] for lg in LANGS}, default=str))
    print(json.dumps(out["pooled"], default=str), "->", d)


if __name__ == "__main__":
    main()
