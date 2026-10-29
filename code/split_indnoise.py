"""A22: per-fact EN-ES increment D on the A13a independent-noise profiles, split by subject-token overlap (J = 0 vs J > 0). CPU.

Usage: python code/split_indnoise.py [--exclude-leak] [--tag split_indnoise_A22]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np

from confound_overlap import DELTA, SEED, TOK_DIR, Z975, center, cosm, per_fact_D, unit
from seeds_analysis import meta
from seeds_covariates import covs

RUNS = {"llama": "analysis/run_20261024_p3seeds_10240109_llama", "gemma2": "analysis/run_20261024_p3seeds_10240109_gemma2"}


def half(D: np.ndarray, cl: np.ndarray, mask: np.ndarray, rng, n_boot: int = 2000) -> dict:
    ok = mask & ~np.isnan(D)
    if ok.sum() < 3:
        return {"n": int(ok.sum()), "verdict": "too few"}
    d, c = D[ok], cl[ok]
    uc = np.unique(c)
    idx = {u: np.where(c == u)[0] for u in uc}
    means = [float(d[np.concatenate([idx[u] for u in rng.choice(uc, len(uc))])].mean()) for _ in range(n_boot)]
    lo, hi = np.percentile(means, [1.25, 98.75])
    return {"n": int(ok.sum()), "mean_D": float(d.mean()), "ci97.5": [float(lo), float(hi)], "n_clusters": int(len(uc)),
            "verdict": "present" if lo > 0 else ("absent" if hi < DELTA else "unresolved")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude-leak", action="store_true")
    ap.add_argument("--tag", default="split_indnoise_A22")
    a = ap.parse_args()
    if a.exclude_leak:
        import seeds_analysis
        from leakage import leaky_ids
        seeds_analysis.EXCLUDE = leaky_ids()
    from transformers import AutoTokenizer
    out = Path("analysis") / f"run_{dt.date.today():%Y%m%d}_{a.tag}"
    out.mkdir(parents=True, exist_ok=False)
    res = {}
    for m, run in RUNS.items():
        rng = np.random.default_rng(SEED)
        ids, strata, cl, cls_, _ = meta(m)
        P = {lg: np.stack([np.load(Path(run) / "grids" / f"{i}_{lg}.npz")["ind"].mean(0) for i in ids]) for lg in ("en", "es")}
        D = per_fact_D(cosm(center(unit(P["en"]), strata), center(unit(P["es"]), strata)), strata)
        cov = covs(m, ids, AutoTokenizer.from_pretrained(TOK_DIR[m]))
        J = cov["J"]
        ev = strata == "event_year"
        res[m] = {}
        for name, base in (("informative", (cls_ != "IP") & ~ev), ("all_non_IP", cls_ != "IP")):
            res[m][name] = {"all": half(D, cl, base, rng), "J0": half(D, cl, base & (J == 0), rng), "Jpos": half(D, cl, base & (J > 0), rng)}
            for k, v in res[m][name].items():
                print(m, name, k, v["n"], {x: (round(y, 3) if isinstance(y, float) else y) for x, y in v.items() if x in ("mean_D", "verdict")}, [round(z, 3) for z in v.get("ci97.5", [])])
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf8")
    (out / "config.json").write_text(json.dumps(vars(a) | {"amendment": "A22", "runs": RUNS, "z": Z975}, indent=1), encoding="utf8")
    print("->", out)


if __name__ == "__main__":
    main()
