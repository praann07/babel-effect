"""A17b: clustering sensitivity of the E2 interchange ratio I (site A, confirmatory pairs). Stored effects only."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np

from e2_analysis import COND, band_and_gate, load_run
from leakage import leaky_ids

SEED = 20261026
RUNS = {
    "A13_en2es": ("e2full_A", "en2es"), "A13_es2en": ("e2full_A", "es2en"), "A16_hi2en": ("e2h2full_A", "hi2en"),
}
SETS = {  # --set: which stored runs to analyse; output dir tag
    "a17b": (RUNS, "A17b", ("llama", "gemma2")),
    "a18b": ({"A18_en2hi": ("e2h3end_A", "en2hi"), "A18_hi2en": ("e2h3end_A", "hi2en")}, "A18b", ("llama", "gemma2")),
    "a19b": ({"A19_en2es": ("e2b_A", "en2es"), "A19_es2en": ("e2b_A", "es2en"), "A19_en2hi": ("e2hb_A", "en2hi"), "A19_hi2en": ("e2hb_A", "hi2en")}, "A19b", ("llama_base",)),
}


def latest(tag: str, m: str) -> str:
    c = sorted(Path("analysis").glob(f"run_*_{tag}_{m}"))
    return str(c[-1])


def stat(c, s, w):
    return float((w * c).sum() / w.sum() / ((w * s).sum() / w.sum())) if w.sum() > 0 else float("nan")


def components(tg, sc):
    par = {}

    def f(x):
        par.setdefault(x, x)
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for t, s in zip(tg, sc):
        par[f(t)] = f(s)
    return np.array([f(t) for t in tg])


def ci(draws):
    d = np.array(draws)
    return {"ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "ci90": [float(np.percentile(d, 5)), float(np.percentile(d, 95))]}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--n-boot", type=int, default=5000); ap.add_argument("--set", default="a17b", choices=list(SETS)); a = ap.parse_args()
    runs, ctag, models = SETS[a.set]
    out = Path("analysis") / f"run_{dt.date.today():%Y%m%d}_cluster_{ctag}"; out.mkdir(parents=True, exist_ok=False)
    leak = leaky_ids(); res = {}
    for m in models:
        for name, (tag, direction) in runs.items():
            run = latest(tag, m)
            D0, F0, pairs0, _ = load_run(run)
            for lk in (False, True):
                keep = np.array([p["direction"] == direction and p["jaccard_source"] == 0 and (not lk or (p["target"] not in leak and p["source"] not in leak)) for p in pairs0])
                if keep.sum() < 5:
                    continue
                # band and l* from ALL pairs of that direction (as e2_analysis does), leak-excluded if lk
                allsel = np.array([p["direction"] == direction and (not lk or (p["target"] not in leak and p["source"] not in leak)) for p in pairs0])
                g = band_and_gate(D0[allsel], F0[allsel], 0); band = np.array(g["band_layers"])
                pr = [p for p, k in zip(pairs0, keep) if k]; D = D0[keep]
                c = D[:, COND["cross"], 0][:, band].mean(1); s = D[:, COND["same"], 0][:, band].mean(1)
                tg = np.array([p["target"] for p in pr]); sc = np.array([p["source"] for p in pr]); n = len(pr)
                rng = np.random.default_rng(SEED); I = stat(c, s, np.ones(n))
                dr = {"S": [], "T": [], "C": [], "W": []}
                groups = {"S": sc, "T": tg, "C": components(tg, sc)}
                for key, lab in groups.items():
                    u = np.unique(lab); idx = {e: np.where(lab == e)[0] for e in u}
                    for _ in range(a.n_boot):
                        ix = np.concatenate([idx[e] for e in rng.choice(u, len(u))]); dr[key].append(stat(c[ix], s[ix], np.ones(len(ix))))
                ut, us = np.unique(tg), np.unique(sc); fs = np.unique(np.concatenate([tg, sc]))
                tidx = {e: np.where(tg == e)[0] for e in fs}; sidx = {e: np.where(sc == e)[0] for e in fs}
                for _ in range(a.n_boot):
                    wt = np.zeros(n); ws = np.zeros(n)
                    for e in rng.choice(fs, len(fs)):
                        wt[tidx[e]] += 1
                    for e in rng.choice(fs, len(fs)):
                        ws[sidx[e]] += 1
                    dr["W"].append(stat(c, s, wt * ws))
                row = {"I": I, "n_pairs": n, "n_targets": int(len(ut)), "n_sources": int(len(us)), "n_components": int(len(np.unique(groups["C"]))), "l_star": g["l_star"], "band": g["band"]}
                for key in dr:
                    r = ci(dr[key]); r["label"] = "SHARED" if r["ci95"][0] >= 0.7 else ("LANGUAGE-SPECIFIC" if r["ci90"][1] <= 0.3 else "PARTIAL"); row[key] = r
                row["labels_robust"] = len({row[k]["label"] for k in "STCW"}) == 1
                res[f"{m}|{name}|{'leakfree' if lk else 'plain'}"] = row
                print(m, name, "leakfree" if lk else "plain", f"n={n} I={I:.3f}", {k: (round(row[k]["ci95"][0], 3), round(row[k]["ci95"][1], 3), row[k]["label"][:3]) for k in "STCW"}, "ROBUST" if row["labels_robust"] else "NOT-ROBUST", flush=True)
    (out / "results.json").write_text(json.dumps(res, indent=1)); print("wrote", out)


if __name__ == "__main__":
    main()
